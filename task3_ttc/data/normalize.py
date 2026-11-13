import argparse
import hashlib
import importlib.metadata
import multiprocessing
import re
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download
from math_verify import parse, verify

from common.io import ROOT, load_yaml, save_json

NO_ANSWER = -1
NO_ANSWER_KEY = "NO_ANSWER"
RAW_PREFIX = "RAW:"

WORKER_PARSED = {}


def worker_parse(text):
    if text not in WORKER_PARSED:
        WORKER_PARSED[text] = parse("$" + text + "$", parsing_timeout=None)
    return WORKER_PARSED[text]


def worker_parses(text):
    return bool(worker_parse(text))


def worker_equal(reference, candidate, float_rounding, strict):
    return verify(worker_parse(reference), worker_parse(candidate), float_rounding=float_rounding, strict=strict, timeout_seconds=None)


class TimedMathVerify:
    def __init__(self, cfg):
        self.cfg = cfg
        self.parsed_ok = {}
        self.equal_cache = {}
        self.timeouts = []
        self.pool = None
        self.start_worker()

    def start_worker(self):
        self.pool = multiprocessing.get_context("spawn").Pool(1)

    def call(self, fn, args, limit, kind):
        try:
            return self.pool.apply_async(fn, args).get(timeout=limit)
        except multiprocessing.TimeoutError:
            self.pool.terminate()
            self.pool.join()
            self.start_worker()
            self.timeouts.append({"kind": kind, "inputs": list(args[:2]) if kind == "verify" else [args[0]]})
            return None

    def parses(self, text):
        if text not in self.parsed_ok:
            self.parsed_ok[text] = bool(self.call(worker_parses, (text,), self.cfg["parsing_timeout_s"], "parse"))
        return self.parsed_ok[text]

    def equal(self, reference, candidate):
        key = (reference, candidate)
        if key not in self.equal_cache:
            result = False
            if self.parses(reference) and self.parses(candidate):
                args = (reference, candidate, self.cfg["float_rounding"], self.cfg["strict"])
                result = bool(self.call(worker_equal, args, self.cfg["verify_timeout_s"], "verify"))
            self.equal_cache[key] = result
        return self.equal_cache[key]

    def close(self):
        self.pool.terminate()
        self.pool.join()


def edit_distance(a, b):
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        current = [i]
        for j, cb in enumerate(b, 1):
            current.append(min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (ca != cb)))
        previous = current
    return previous[-1]


def multi_choice_key(answer, choices):
    text = answer.strip()
    bracketed = re.findall(r"\(([a-zA-Z])\)", text)
    if bracketed:
        text = bracketed[0].upper()
    letters = [chr(ord("A") + i) for i in range(len(choices))]
    if text in letters:
        return text
    distances = [edit_distance(text, str(choice)) for choice in choices]
    return letters[distances.index(min(distances))]


def free_form_key(answer, answer_type, precision):
    try:
        if answer_type == "integer":
            return str(int(float(answer)))
        if answer_type == "float":
            return str(round(float(answer), int(precision)))
        if answer_type == "list":
            return str(answer)
    except (ValueError, OverflowError):
        return RAW_PREFIX + answer.strip()
    raise ValueError(f"unknown answer_type {answer_type}")


def mathvista_key(answer, question):
    if question["question_type"] == "multi_choice":
        return multi_choice_key(answer, list(question["choices"]))
    return free_form_key(answer, question["answer_type"], question["precision"])


def group_math_answers(answers, verifier):
    reps = []
    labels = []
    keys = {}
    for answer in answers:
        if answer is None:
            labels.append(NO_ANSWER)
            continue
        if answer not in keys:
            match = next((g for g, rep in enumerate(reps) if verifier.equal(rep, answer)), None)
            if match is None:
                reps.append(answer)
                match = len(reps) - 1
            keys[answer] = match
        labels.append(keys[answer])
    return labels, reps


def group_mathvista_answers(answers, question):
    labels = []
    reps = []
    index = {}
    for answer in answers:
        if answer is None:
            labels.append(NO_ANSWER)
            continue
        key = mathvista_key(answer, question)
        if key not in index:
            index[key] = len(reps)
            reps.append(key)
        labels.append(index[key])
    return labels, reps


def grade_math(gold, representative, verifier):
    return verifier.equal(gold, representative)


def grade_mathvista(gold, key, question):
    if question["question_type"] == "multi_choice":
        choices = list(question["choices"])
        return str(choices[ord(key) - ord("A")]) == str(question["answer"])
    return key == str(question["answer"])


def load_mathvista_metadata(spec):
    path = hf_hub_download(spec["repo"], spec["file"], repo_type="dataset", revision=spec["revision"])
    cols = ["pid", "choices", "precision", "answer", "question_type", "answer_type"]
    return pq.read_table(path, columns=cols).to_pandas().set_index("pid")


def score_dataset(dataset, gen, metadata, verifier):
    rows = []
    member_disagreements = 0
    for position, (sample_id, block) in enumerate(gen.sort_values(["sample_id", "rollout_id"]).groupby("sample_id", sort=True)):
        if position % 100 == 0:
            print(dataset, "question", position, flush=True)
        answers = [a if isinstance(a, str) else None for a in block["extracted_answer"]]
        gold = block["gold_answer"].iloc[0]
        assert (block["gold_answer"] == gold).all()
        if dataset == "math":
            labels, reps = group_math_answers(answers, verifier)
            group_correct = [grade_math(gold, rep, verifier) for rep in reps]
            member_correct = [False if a is None else grade_math(gold, a, verifier) for a in answers]
        else:
            question = metadata.loc[sample_id]
            labels, reps = group_mathvista_answers(answers, question)
            group_correct = [grade_mathvista(gold, rep, question) for rep in reps]
            member_correct = [False if a is None else grade_mathvista(gold, mathvista_key(a, question), question) for a in answers]
        for rollout_id, label, own in zip(block["rollout_id"], labels, member_correct):
            correct = False if label == NO_ANSWER else group_correct[label]
            member_disagreements += int(correct != own)
            rows.append({
                "sample_id": sample_id,
                "rollout_id": int(rollout_id),
                "group": int(label),
                "group_key": NO_ANSWER_KEY if label == NO_ANSWER else reps[label],
                "correct": bool(correct),
            })
    return pd.DataFrame(rows), member_disagreements


def file_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "configs" / "task3" / "offline.yaml"))
    args = parser.parse_args()
    cfg = load_yaml(args.config)
    scorer_cfg = cfg["scorer"]
    metadata = load_mathvista_metadata(cfg["mathvista_metadata"])
    out = ROOT / "outputs" / "task3" / "answers"
    out.mkdir(parents=True, exist_ok=True)
    summary = {}
    verifier = TimedMathVerify(scorer_cfg)
    for dataset, spec in cfg["datasets"].items():
        gen = pq.read_table(
            Path(cfg["data_root"]) / spec["folder"] / "generations.parquet",
            columns=["sample_id", "rollout_id", "gold_answer", "extracted_answer"],
        ).to_pandas()
        if dataset == "math":
            math_golds = set(gen["gold_answer"])
        if dataset == "mathvista":
            assert gen["sample_id"].isin(metadata.index).all()
        table, disagreements = score_dataset(dataset, gen, metadata, verifier)
        assert len(table) == spec["expected"]["rows"]
        assert int((table["group"] == NO_ANSWER).sum()) == spec["expected"]["null_extracted_answers"]
        table.to_parquet(out / f"{dataset}.parquet", index=False)
        groups_per_question = table[table["group"] != NO_ANSWER].groupby("sample_id")["group"].nunique()
        keys = table["group_key"]
        summary[dataset] = {
            "rows": int(len(table)),
            "no_answer_rows": int((table["group"] == NO_ANSWER).sum()),
            "rollout_correct_rate": float(table["correct"].mean()),
            "questions_with_any_correct": int(table.groupby("sample_id")["correct"].any().sum()),
            "mean_answer_groups_per_question_excluding_no_answer": float(groups_per_question.mean()),
            "rows_where_group_grade_differs_from_own_grade": int(disagreements),
            "raw_unparsed_free_form_rows": int(keys.str.startswith(RAW_PREFIX).sum()),
        }
        print(dataset, summary[dataset])
    verifier.close()
    summary["math"]["math_verify_timeouts"] = len(verifier.timeouts)
    summary["math"]["unparseable_gold_answers"] = sorted(g for g in math_golds if not verifier.parsed_ok.get(g, False))
    save_json(summary, ROOT / "results" / "task3" / "answer_summary.json")
    save_json(verifier.timeouts, ROOT / "results" / "task3" / "math_verify_timeouts.json")
    save_json(
        {
            "math_verify": importlib.metadata.version("math-verify"),
            "latex2sympy2_extended": importlib.metadata.version("latex2sympy2_extended"),
            "sympy": importlib.metadata.version("sympy"),
            "math_verify_settings": {
                **scorer_cfg,
                "parse_wrapper": "$...$",
                "timeout_mechanism": "each parse/verify call runs in a spawned worker process; on timeout the worker is killed and the call counts as an empty parse / not equal, matching math-verify's own TimeoutException handling",
            },
            "mathvista_metadata": cfg["mathvista_metadata"],
            "normalize_py_sha256": file_sha256(Path(__file__)),
        },
        ROOT / "results" / "task3" / "scorer.json",
    )


if __name__ == "__main__":
    main()
