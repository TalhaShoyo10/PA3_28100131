import argparse
import json
from pathlib import Path

import pandas as pd

from common.io import ROOT, load_yaml, save_json
from task3_ttc.data.normalize import TimedMathVerify
from task3_ttc.scoring.aggregate import self_consistency


def grade(verifier, gold, answer):
    return answer is not None and verifier.equal(gold, answer)


def vote_parallel(verifier, answers):
    reps, labels = [], []
    for a in answers:
        if a is None:
            labels.append(-1)
            continue
        match = next((g for g, r in enumerate(reps) if verifier.equal(r, a)), None)
        if match is None:
            reps.append(a)
            match = len(reps) - 1
        labels.append(match)
    position, _ = self_consistency(labels)
    return answers[position], position


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--records", nargs="+", required=True)
    parser.add_argument("--config", default=str(ROOT / "configs" / "task3" / "offline.yaml"))
    parser.add_argument("--out", default=str(ROOT / "results" / "task3" / "live"))
    args = parser.parse_args()
    verifier = TimedMathVerify(load_yaml(args.config)["scorer"])
    records = [json.loads(line) for p in args.records for line in Path(p).read_text(encoding="utf-8").splitlines() if line.strip()]
    records.sort(key=lambda r: r["question_index"])
    rows = []
    for r in records:
        gold = r["gold_answer"]
        par_answer, par_pos = vote_parallel(verifier, [c["extracted"] for c in r["parallel"]])
        initial = r["parallel"][0]
        rev = r["revision"]
        initial_ok = grade(verifier, gold, initial["extracted"])
        rev_ok = grade(verifier, gold, rev["extracted"])
        par_calls = r["parallel"]
        rows.append({
            "question_index": r["question_index"], "sample_id": r["sample_id"],
            "single_correct": grade(verifier, gold, r["single"]["extracted"]),
            "single_in": r["single"]["input_tokens"], "single_out": r["single"]["output_tokens"], "single_s": r["single"]["seconds"],
            "single_extracted": r["single"]["extracted"] is not None, "single_truncated": r["single"]["truncated"],
            "parallel_correct": grade(verifier, gold, par_answer), "parallel_selected_call": par_pos,
            "parallel_in": sum(c["input_tokens"] for c in par_calls), "parallel_out": sum(c["output_tokens"] for c in par_calls), "parallel_s": sum(c["seconds"] for c in par_calls),
            "parallel_extracted": par_answer is not None, "parallel_truncated_any": any(c["truncated"] for c in par_calls),
            "parallel_calls_truncated": sum(c["truncated"] for c in par_calls), "parallel_calls_extracted": sum(c["extracted"] is not None for c in par_calls),
            "refine_correct": rev_ok, "initial_correct": initial_ok,
            "refine_in": initial["input_tokens"] + rev["input_tokens"], "refine_out": initial["output_tokens"] + rev["output_tokens"], "refine_s": initial["seconds"] + rev["seconds"],
            "refine_extracted": rev["extracted"] is not None, "refine_truncated": rev["truncated"], "initial_truncated": initial["truncated"],
            "transition": ("C" if initial_ok else "W") + "->" + ("C" if rev_ok else "W"),
        })
    df = pd.DataFrame(rows)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "per_question.csv", index=False)
    summary = []
    for s in ["single", "parallel", "refine"]:
        summary.append({
            "strategy": s, "n_questions": len(df), "accuracy": df[f"{s}_correct"].mean(),
            "mean_input_tokens": df[f"{s}_in"].mean(), "mean_output_tokens": df[f"{s}_out"].mean(), "total_output_tokens": int(df[f"{s}_out"].sum()),
            "mean_runtime_s": df[f"{s}_s"].mean(),
            "extraction_rate": df[f"{s}_extracted"].mean(),
            "truncation_rate": df["single_truncated" if s == "single" else ("parallel_truncated_any" if s == "parallel" else "refine_truncated")].mean(),
            "output_allowance": 3072,
        })
    pd.DataFrame(summary).to_csv(out / "summary.csv", index=False)
    save_json({
        "transitions_vs_shared_initial": df["transition"].value_counts().to_dict(),
        "parallel_call_truncation_rate": float(df["parallel_calls_truncated"].sum() / (3 * len(df))),
        "parallel_call_extraction_rate": float(df["parallel_calls_extracted"].sum() / (3 * len(df))),
        "cost_accounting": "refine charges the shared initial response (parallel call 1) plus the full revision prompt and all critique/revision output tokens; parallel also includes that same call",
        "verify_timeouts": len(verifier.timeouts),
    }, out / "transitions.json")
    verifier.close()
    print(pd.DataFrame(summary).round(4).to_string(index=False))


if __name__ == "__main__":
    main()
