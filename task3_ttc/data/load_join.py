import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from common.io import ROOT, load_yaml, save_json

KEYS = ["dataset", "sample_id", "rollout_id"]
GEN_META = [
    "model", "question_text", "n_generated_text_tokens", "hit_max_new_tokens",
    "generated_text", "gold_answer", "extracted_answer",
]
PROB_LISTS = ["logp", "kl_u_p"]
ATTN_LISTS = ["img_attn_total", "prompt_text_attn_total", "generated_text_attn_total"]


def sorted_table(table):
    order = pd.DataFrame({k: table.column(k).to_pandas() for k in KEYS}).sort_values(KEYS, kind="stable").index.to_numpy()
    return table.take(pa.array(order))


def flat_and_offsets(table, name):
    arr = table.column(name).combine_chunks()
    lengths = pc.list_value_length(arr).to_numpy(zero_copy_only=False).astype(np.int64)
    offsets = np.concatenate([[0], np.cumsum(lengths)])
    return arr.flatten().to_numpy(zero_copy_only=False), offsets


def load_model_data(data_root, folder, vlm):
    root = Path(data_root) / folder
    gen = sorted_table(pq.read_table(root / "generations.parquet"))
    prob = sorted_table(pq.read_table(root / "probability_metrics.parquet"))
    attn = sorted_table(pq.read_table(root / "attention_metrics.parquet")) if vlm else None

    meta = gen.select(KEYS + GEN_META).to_pandas()
    for other in [prob] + ([attn] if vlm else []):
        other_keys = other.select(KEYS).to_pandas()
        assert meta[KEYS].equals(other_keys), "join keys differ between files"
    assert not meta.duplicated(KEYS).any(), "duplicate (dataset, sample_id, rollout_id)"

    tokens = {}
    tokens["generated_token_ids"] = flat_and_offsets(gen, "generated_token_ids")
    for name in PROB_LISTS:
        tokens[name] = flat_and_offsets(prob, name)
    if vlm:
        for name in ATTN_LISTS:
            tokens[name] = flat_and_offsets(attn, name)
        tokens["attn_preceding_counts"] = flat_and_offsets(attn, "n_generated_text_tokens")
        meta["n_image_tokens"] = attn.column("n_image_tokens").to_numpy()
        meta["n_prompt_text_tokens"] = attn.column("n_prompt_text_tokens").to_numpy()

    reference_offsets = tokens["generated_token_ids"][1]
    for name, (flat, offsets) in tokens.items():
        assert np.array_equal(offsets, reference_offsets), f"{name} list lengths differ from generated_token_ids"
    lengths = np.diff(reference_offsets)
    assert np.array_equal(lengths, meta["n_generated_text_tokens"].to_numpy()), "scalar length differs from token list length"

    if vlm:
        flat, offsets = tokens["attn_preceding_counts"]
        expected = np.concatenate([np.arange(n) for n in lengths])
        assert np.array_equal(flat, expected), "attention preceding-token counts are not [0..N-1]"

    meta["row"] = np.arange(len(meta))
    return meta, tokens


def row_values(tokens, name, row):
    flat, offsets = tokens[name]
    return flat[offsets[row]:offsets[row + 1]]


def check_expected(meta, tokens, expected):
    questions = meta["sample_id"].nunique()
    per_question = meta.groupby("sample_id")["rollout_id"].apply(lambda s: sorted(s.tolist()))
    summary = {
        "rows": int(len(meta)),
        "questions": int(questions),
        "rollouts_per_question_all_0_to_31": bool(per_question.map(lambda r: r == list(range(expected["rollouts_per_question"]))).all()),
        "retained_tokens": int(tokens["generated_token_ids"][1][-1]),
        "capped_rollouts": int(meta["hit_max_new_tokens"].sum()),
        "null_extracted_answers": int(meta["extracted_answer"].isna().sum()),
    }
    assert summary["rows"] == expected["rows"]
    assert summary["questions"] == expected["questions"]
    assert summary["rollouts_per_question_all_0_to_31"]
    assert summary["retained_tokens"] == expected["retained_tokens"]
    assert summary["capped_rollouts"] == expected["capped_rollouts"]
    assert summary["null_extracted_answers"] == expected["null_extracted_answers"]
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "configs" / "task3" / "offline.yaml"))
    args = parser.parse_args()
    cfg = load_yaml(args.config)
    report = {}
    for name, spec in cfg["datasets"].items():
        meta, tokens = load_model_data(cfg["data_root"], spec["folder"], spec["vlm"])
        report[name] = check_expected(meta, tokens, spec["expected"])
        print(name, report[name])
    save_json(report, ROOT / "results" / "task3" / "data_checks.json")


if __name__ == "__main__":
    main()
