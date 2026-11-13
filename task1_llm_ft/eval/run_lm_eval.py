import argparse
import json
import time
from pathlib import Path

import lm_eval
import torch
from lm_eval import simple_evaluate
from lm_eval.models.huggingface import HFLM

from common.io import save_json
from common.seed import SEED

GSM8K_TASK = "gsm8k_cot"
LIKELIHOOD_TASKS = ["hellaswag", "boolq", "arc_easy"]
MAX_GEN_TOKS = 1024
NON_THINKING_MARK = "<think>\n\n</think>\n\n"


def jsonable(value):
    return json.loads(json.dumps(value, default=str))


def evaluate(lm, tasks, chat, limit, gen_kwargs):
    start = time.perf_counter()
    out = simple_evaluate(
        model=lm,
        tasks=tasks,
        num_fewshot=0,
        apply_chat_template=chat,
        log_samples=True,
        limit=limit,
        gen_kwargs=gen_kwargs,
        random_seed=SEED,
        numpy_random_seed=SEED,
        torch_random_seed=SEED,
        fewshot_random_seed=SEED,
    )
    return out, time.perf_counter() - start


def gsm8k_rows(samples, tokenizer):
    by_doc = {}
    for s in samples:
        row = by_doc.setdefault(s["doc_id"], {"doc_id": s["doc_id"], "target": s["target"], "response": s["resps"][0][0]})
        row[f"extracted_{s['filter']}"] = s["filtered_resps"][0]
        row[f"correct_{s['filter']}"] = float(s["exact_match"])
    for row in by_doc.values():
        n_tokens = len(tokenizer(row["response"], add_special_tokens=False)["input_ids"])
        row["response_tokens_retokenized"] = n_tokens
        row["truncated"] = n_tokens >= MAX_GEN_TOKS - 1
        row["extractable_flexible"] = row["extracted_flexible-extract"] != "[invalid]"
        row["extractable_strict"] = row["extracted_strict-match"] != "[invalid]"
    return [by_doc[k] for k in sorted(by_doc)]


def likelihood_rows(samples):
    rows = []
    for s in samples:
        row = {"doc_id": s["doc_id"], "target": s["target"], "filter": s["filter"]}
        for key in ("acc", "acc_norm"):
            if key in s:
                row[key] = float(s[key])
        rows.append(row)
    return rows


def write_jsonl(rows, path):
    with open(path, "w", encoding="utf-8") as handle:
        for r in rows:
            handle.write(json.dumps(jsonable(r)) + "\n")


def gsm8k_summary(rows):
    n = len(rows)
    extractable = [r for r in rows if r["extractable_flexible"]]
    return {
        "n": n,
        "accuracy_flexible": sum(r["correct_flexible-extract"] for r in rows) / n,
        "accuracy_strict": sum(r["correct_strict-match"] for r in rows) / n,
        "extractable_rate_flexible": len(extractable) / n,
        "extractable_rate_strict": sum(r["extractable_strict"] for r in rows) / n,
        "truncation_rate": sum(r["truncated"] for r in rows) / n,
        "accuracy_among_extractable_flexible": (sum(r["correct_flexible-extract"] for r in extractable) / len(extractable)) if extractable else None,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--condition", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--batch-size", default="auto")
    parser.add_argument("--revision", default=None)
    parser.add_argument("--tasks", choices=["gsm8k", "likelihood", "all"], default="all")
    args = parser.parse_args()

    out_dir = Path(args.out) / args.condition
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.cuda.reset_peak_memory_stats()
    extra = {"revision": args.revision} if args.revision else {}
    lm = HFLM(pretrained=args.model, dtype="float16", batch_size=args.batch_size, enable_thinking=False, max_length=4096, **extra)
    assert not lm.model.training
    record = {"condition": args.condition, "model": args.model, "lm_eval_version": lm_eval.__version__, "torch": torch.__version__, "device": torch.cuda.get_device_name(), "backend": "hf", "limit": args.limit, "revision": args.revision, "tasks": args.tasks, "versions": {}, "configs": {}}

    if args.tasks in ("gsm8k", "all"):
        gsm, gsm_s = evaluate(lm, [GSM8K_TASK], True, args.limit, f"max_gen_toks={MAX_GEN_TOKS}")
        samples = gsm["samples"][GSM8K_TASK]
        prompt = samples[0]["arguments"][0][0]
        (out_dir / "rendered_gsm8k_prompt.txt").write_text(prompt, encoding="utf-8")
        assert prompt.rstrip("\n").endswith(NON_THINKING_MARK.rstrip("\n")), "GSM8K prompt is not the non-thinking chat template"
        rows = gsm8k_rows(samples, lm.tokenizer)
        write_jsonl(rows, out_dir / "gsm8k_samples.jsonl")
        record["gsm8k"] = {**gsm8k_summary(rows), "harness_results": jsonable(gsm["results"][GSM8K_TASK]), "seconds": gsm_s}
        record["versions"].update(jsonable(gsm["versions"]))
        record["configs"][GSM8K_TASK] = jsonable(gsm["configs"][GSM8K_TASK])

    if args.tasks in ("likelihood", "all"):
        lik, lik_s = evaluate(lm, LIKELIHOOD_TASKS, False, args.limit, None)
        for task in LIKELIHOOD_TASKS:
            write_jsonl(likelihood_rows(lik["samples"][task]), out_dir / f"{task}_samples.jsonl")
            record[task] = jsonable(lik["results"][task])
            record["configs"][task] = jsonable(lik["configs"][task])
        first = lik["samples"]["hellaswag"][0]["arguments"][0][0]
        (out_dir / "rendered_hellaswag_context.txt").write_text(first, encoding="utf-8")
        assert "<|im_start|>" not in first, "likelihood prompt unexpectedly uses the chat template"
        record["versions"].update(jsonable(lik["versions"]))
        record["likelihood_seconds"] = lik_s
    record["peak_mem_gb"] = torch.cuda.max_memory_allocated() / 1e9
    save_json(record, out_dir / f"lm_eval_summary_{args.tasks}.json")
    print(json.dumps(record, default=str)[:3000], flush=True)


if __name__ == "__main__":
    main()
