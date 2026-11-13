import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from common.bootstrap import bootstrap_indices, percentile_ci
from common.io import load_json, load_yaml, save_json
from common.seed import SEED, set_all_seeds
from task2_vlm.eval.vqa_normalize import normalize_answer, soft_accuracy
from task2_vlm.model.vlm import Connector, PromptParts, per_example_token_nll
from task2_vlm.train import FeatureStore, forward_examples, inject_lora, partner_images, qa_examples

CONDITIONS = ["correct", "complementary", "mismatched", "none"]
N_BOOT = 4000
MAX_NEW = 16


def load_final_model(cfg, weights_path, device):
    model = AutoModelForCausalLM.from_pretrained(cfg["model"], revision=cfg["model_revision"], dtype=torch.float16).to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    state = torch.load(weights_path, map_location="cpu", weights_only=False)
    lora = inject_lora(model, cfg["lora"])
    for k, v in state["lora"].items():
        lora[k].lora_A.data.copy_(v["A"].to(device))
        lora[k].lora_B.data.copy_(v["B"].to(device))
    connector = Connector(768, cfg["connector"]["hidden"], model.config.hidden_size).to(device)
    connector.load_state_dict(state["connector"])
    model.eval()
    connector.eval()
    return model, connector


def generate_batch(model, embeds, mask, tokenizer):
    out = model.generate(inputs_embeds=embeds, attention_mask=mask, max_new_tokens=MAX_NEW, do_sample=False, pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.convert_tokens_to_ids("<|im_end|>"))
    return [tokenizer.decode(o, skip_special_tokens=True).strip() for o in out]


def score(suite, pred, ex, pair):
    item = next(it for it in pair["items"] if it["image_id"] == ex["image_id"])
    if suite == "vqav2":
        return soft_accuracy(pred, item["answers"]), normalize_answer(pred) == item["consensus"], item["consensus"]
    canon = normalize_answer(item["target"])
    ok = normalize_answer(pred) == canon
    return float(ok), ok, canon


def run_suite(suite, pairs, mismatch, model, connector, parts, store, tokenizer, device, batch_size):
    by_id = {p["pair_id"]: p for p in pairs}
    examples = [x for p in pairs for x in qa_examples(p, suite)]
    for j, ex in enumerate(examples):
        ex["slot"] = 0 if by_id[ex["pair_id"]]["items"][0]["image_id"] == ex["image_id"] else 1
    rows = []
    for cond in CONDITIONS:
        for i in range(0, len(examples), batch_size):
            chunk = examples[i:i + batch_size]
            partner = partner_images(chunk, by_id)
            mm = [mismatch[f"{ex['pair_id']}:{ex['slot']}"]["image_id"] for ex in chunk]
            with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16):
                e, lab, m, pos = forward_examples(model, connector, parts, store, chunk, "test", cond, device, pad_side="right", partner=partner, mismatch=mm)
                nll, _ = per_example_token_nll(model, e, lab, m, pos, exclude_last_target=True)
                prompt_only = [dict(ex, target=None) for ex in chunk]
                ge, _, gm, _ = forward_examples(model, connector, parts, store, prompt_only, "test", cond, device, pad_side="left", partner=partner, mismatch=mm)
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                preds = generate_batch(model, ge, gm, tokenizer)
                torch.cuda.synchronize()
                gen_s = (time.perf_counter() - t0) / len(chunk)
            for ex, pred, n in zip(chunk, preds, nll.float().cpu().tolist()):
                acc, exact, canon = score(suite, pred, ex, by_id[ex["pair_id"]])
                pair = by_id[ex["pair_id"]]
                partner_item = pair["items"][1 - ex["slot"]]
                partner_canon = partner_item["consensus"] if suite == "vqav2" else normalize_answer(partner_item["target"])
                norm_pred = normalize_answer(pred)
                rows.append({
                    "suite": suite, "condition": cond, "pair_id": ex["pair_id"], "slot": ex["slot"], "image_id": ex["image_id"],
                    "question": ex["question"], "reference": ex["target"], "canonical": canon, "prediction": pred, "prediction_normalized": norm_pred,
                    "accuracy": acc, "exact_canonical": bool(exact), "empty": norm_pred == "", "ref_nll": n, "gen_s_per_answer": gen_s,
                    "type": pair.get("answer_type", pair.get("kind")),
                    "equals_partner_reference": norm_pred == partner_canon, "equals_own_reference": norm_pred == canon,
                })
    return rows


def pair_table(df):
    g = df.groupby(["suite", "condition", "pair_id"])
    return pd.DataFrame({
        "pair_acc": g["exact_canonical"].all(),
        "identical": g["prediction_normalized"].agg(lambda s: s.iloc[0] == s.iloc[1]),
        "mean_acc": g["accuracy"].mean(),
        "mean_nll": g["ref_nll"].mean(),
    }).reset_index()


def summary(df):
    pairs = pair_table(df)
    rows = []
    for (suite, cond), sub in df.groupby(["suite", "condition"]):
        p = pairs[(pairs.suite == suite) & (pairs.condition == cond)]
        row = {"suite": suite, "condition": cond, "n_questions": len(sub), "accuracy": sub["accuracy"].mean(), "pair_accuracy": p["pair_acc"].mean(),
               "identical_answer_rate": p["identical"].mean(), "empty_answer_rate": sub["empty"].mean(), "mean_ref_nll": sub["ref_nll"].mean(),
               "gen_s_per_answer": sub["gen_s_per_answer"].mean()}
        for t, st in sub.groupby("type"):
            row[f"accuracy_{t}"] = st["accuracy"].mean()
        rows.append(row)
    return pd.DataFrame(rows), pairs


def advantage(pairs):
    rows = []
    for suite, sub in pairs.groupby("suite"):
        c = sub[sub.condition == "correct"].set_index("pair_id").sort_index()
        s = sub[sub.condition == "complementary"].set_index("pair_id").loc[c.index]
        idx = bootstrap_indices(len(c), N_BOOT, SEED)
        for metric, sign in [("mean_acc", 1.0), ("mean_nll", -1.0)]:
            diff = sign * (c[metric].to_numpy() - s[metric].to_numpy())
            boots = diff[idx].mean(axis=1)
            lo, hi = percentile_ci(boots)
            rows.append({"suite": suite, "metric": "accuracy" if metric == "mean_acc" else "nll_complementary_minus_correct", "advantage_correct_over_complementary": float(diff.mean()), "ci_low": lo, "ci_high": hi, "n_pairs": len(c), "n_resamples": N_BOOT, "seed": SEED})
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--splits", required=True)
    parser.add_argument("--features", required=True)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    cfg = load_yaml(args.config)
    set_all_seeds(SEED)
    device = torch.device("cuda")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(cfg["model"], revision=cfg["model_revision"])
    parts = PromptParts(tokenizer)
    model, connector = load_final_model(cfg, args.weights, device)
    store = FeatureStore(args.features)
    s = Path(args.splits)
    mismatch = load_json(s / "mismatch_map.json")
    torch.cuda.reset_peak_memory_stats()
    rows = []
    rows += run_suite("vqav2", load_json(s / "vqav2_pairs_test.json")["pairs"], mismatch["vqav2"], model, connector, parts, store, tokenizer, device, cfg["final_batch_size"])
    rows += run_suite("grounding", load_json(s / "grounding_test.json")["pairs"], mismatch["grounding"], model, connector, parts, store, tokenizer, device, cfg["final_batch_size"])
    df = pd.DataFrame(rows)
    with open(out / "predictions.jsonl", "w", encoding="utf-8") as handle:
        for r in rows:
            handle.write(json.dumps(r, default=str) + "\n")
    table, pairs = summary(df)
    table[table.suite == "grounding"].to_csv(out / "final_grounding.csv", index=False)
    table[table.suite == "vqav2"].to_csv(out / "final_vqav2.csv", index=False)
    advantage(pairs).to_csv(out / "advantage.csv", index=False)
    swap = df[df.condition == "complementary"]
    follow = swap.assign(outcome=np.where(swap.equals_partner_reference, "equals_partner_reference", np.where(swap.equals_own_reference, "equals_own_reference", "neither")))
    follow.groupby(["suite", "outcome"]).size().rename("count").reset_index().to_csv(out / "follow_partner.csv", index=False)
    save_json({"peak_mem_gb": torch.cuda.max_memory_allocated() / 1e9, "device": torch.cuda.get_device_name(), "batch_size": cfg["final_batch_size"], "max_new_tokens": MAX_NEW, "decoding": "greedy"}, out / "final_eval_info.json")
    print(table.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
