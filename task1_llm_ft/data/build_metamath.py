import argparse
import platform
from collections import Counter

import datasets
import numpy as np
import transformers
from datasets import load_dataset
from huggingface_hub import HfApi
from transformers import AutoTokenizer

from common.io import ROOT, load_yaml, save_json
from common.seed import SEED, set_all_seeds
from task1_llm_ft.data.formatting import (
    clean_response,
    encode_example,
    normalize_text,
    template_matches_manual_format,
)

NORMALIZATION_RULE = "lowercase; delete every character that is not a word character or whitespace; collapse whitespace runs to one space; strip"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "configs" / "task1" / "data.yaml"))
    args = parser.parse_args()
    cfg = load_yaml(args.config)
    set_all_seeds(cfg["seed"])
    assert cfg["seed"] == SEED

    tokenizer = AutoTokenizer.from_pretrained(cfg["model"])
    gsm = load_dataset(cfg["gsm8k"]["name"], cfg["gsm8k"]["config"], split=cfg["gsm8k"]["split"])
    assert len(gsm) == cfg["gsm8k"]["expected_size"]
    test_questions = {normalize_text(q) for q in gsm["question"]}

    mm_cfg = cfg["metamath"]
    source = load_dataset(mm_cfg["name"], split=mm_cfg["split"])

    kept_ids = []
    excluded = Counter()
    lengths = []
    supervised_tokens = 0
    template_mismatches = 0
    leading_newline_stripped = 0
    rows_scanned = 0

    for idx, row in enumerate(source):
        if len(kept_ids) == mm_cfg["n_keep"]:
            break
        rows_scanned += 1
        query, response = row["query"], row["response"]

        if not (query and query.strip() and response and response.strip()):
            excluded["empty_query_or_response"] += 1
            continue
        if normalize_text(query) in test_questions:
            excluded["gsm8k_test_match_query"] += 1
            continue
        if normalize_text(row["original_question"]) in test_questions:
            excluded["gsm8k_test_match_original_question"] += 1
            continue

        enc = encode_example(tokenizer, query, response)
        if len(enc["input_ids"]) > mm_cfg["max_tokens"]:
            excluded["formatted_length_over_max"] += 1
            continue

        template_mismatches += int(not template_matches_manual_format(tokenizer, query, response))
        leading_newline_stripped += int(response != clean_response(response))
        kept_ids.append(idx)
        lengths.append(len(enc["input_ids"]))
        supervised_tokens += len(enc["input_ids"]) - enc["prompt_len"]

    assert len(kept_ids) == mm_cfg["n_keep"], f"only {len(kept_ids)} eligible examples found"
    assert template_mismatches == 0, f"{template_mismatches} examples where the chat template differs from the manual format"

    api = HfApi()
    save_json(
        {
            "dataset": mm_cfg["name"],
            "dataset_revision": api.dataset_info(mm_cfg["name"]).sha,
            "id_definition": "row index in source order of the dataset split",
            "n": len(kept_ids),
            "ids": kept_ids,
        },
        ROOT / "splits" / "task1" / "metamath_kept_ids.json",
    )
    save_json(
        {
            "rows_scanned": rows_scanned,
            "last_kept_row_index": kept_ids[-1],
            "excluded_by_reason_first_match": dict(excluded),
            "reason_order": [
                "empty_query_or_response",
                "gsm8k_test_match_query",
                "gsm8k_test_match_original_question",
                "formatted_length_over_max",
            ],
            "gsm8k_test_questions": len(gsm),
            "normalization_rule": NORMALIZATION_RULE,
            "max_tokens": mm_cfg["max_tokens"],
            "length_definition": "prompt tokens from the chat template with enable_thinking=False and add_generation_prompt=True, plus response tokens, plus the end-of-turn token",
            "template_mismatches": template_mismatches,
            "kept_rows_with_leading_newline_stripped_from_response": leading_newline_stripped,
        },
        ROOT / "splits" / "task1" / "metamath_exclusion_counts.json",
    )
    save_json(
        {
            "n_examples": len(kept_ids),
            "total_supervised_tokens": int(supervised_tokens),
            "length_min": int(min(lengths)),
            "length_mean": float(np.mean(lengths)),
            "length_max": int(max(lengths)),
            "model": cfg["model"],
            "model_revision": api.model_info(cfg["model"]).sha,
            "versions": {
                "python": platform.python_version(),
                "transformers": transformers.__version__,
                "datasets": datasets.__version__,
                "numpy": np.__version__,
            },
        },
        ROOT / "results" / "task1" / "data_stats.json",
    )
    print(dict(excluded), "kept", len(kept_ids), "supervised_tokens", supervised_tokens)


if __name__ == "__main__":
    main()
