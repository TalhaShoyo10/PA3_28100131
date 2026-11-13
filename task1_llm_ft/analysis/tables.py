import argparse
import json
from pathlib import Path

import pandas as pd
import torch
from safetensors.torch import load_file

from common.io import ROOT, save_json

CONDITIONS = ["baseline", "full_ft_625", "full_ft_1250", "l2sp_1250", "lora_r4", "lora_r16", "oplora_r16"]
RUNS = ["full_ft", "l2sp", "lora_r4", "lora_r16", "oplora_r16"]
PAIRS = [("lora_r16", "lora_r4"), ("l2sp_1250", "full_ft_1250"), ("oplora_r16", "lora_r16"), ("lora_r16", "full_ft_1250"), ("full_ft_1250", "full_ft_625")]
METRICS = [
    ("gsm8k", "gsm8k_flexible", lambda r: r["gsm8k"]["accuracy_flexible"]),
    ("gsm8k", "gsm8k_strict", lambda r: r["gsm8k"]["accuracy_strict"]),
    ("hellaswag", "hellaswag_acc_norm", lambda r: r["hellaswag"]["acc_norm,none"]),
    ("hellaswag", "hellaswag_acc", lambda r: r["hellaswag"]["acc,none"]),
    ("boolq", "boolq_acc", lambda r: r["boolq"]["acc,none"]),
    ("arc_easy", "arc_easy_acc_norm", lambda r: r["arc_easy"]["acc_norm,none"]),
    ("arc_easy", "arc_easy_acc", lambda r: r["arc_easy"]["acc,none"]),
]
PRIMARY = ["gsm8k_flexible", "hellaswag_acc_norm", "boolq_acc", "arc_easy_acc_norm"]


def find(root, pattern):
    hits = sorted(Path(root).glob(pattern))
    return hits[0] if hits else None


def eval_summaries(root):
    merged = {}
    for cond in CONDITIONS:
        rec = {}
        for part in ["gsm8k", "likelihood", "all"]:
            f = find(root, f"**/task1_eval/{cond}/lm_eval_summary_{part}.json")
            if f:
                rec.update(json.loads(f.read_text()))
        if rec:
            merged[cond] = rec
    return merged


def benchmarks(summaries):
    rows = []
    for cond, rec in summaries.items():
        row = {"condition": cond}
        for task, name, getter in METRICS:
            row[name] = getter(rec) if task in rec else None
        rows.append(row)
    df = pd.DataFrame(rows).set_index("condition").reindex([c for c in CONDITIONS if c in summaries])
    if "baseline" in df.index:
        for m in PRIMARY:
            df[f"{m}_delta_pp"] = 100 * (df[m] - df.loc["baseline", m])
    return df.reset_index()


def comparisons(bench):
    b = bench.set_index("condition")
    rows = []
    for a, c in PAIRS:
        if a in b.index and c in b.index:
            rows.append({"comparison": f"{a} - {c}", **{f"{m}_diff_pp": 100 * (b.loc[a, m] - b.loc[c, m]) for m in PRIMARY}})
    return pd.DataFrame(rows)


def extraction(summaries):
    rows = []
    for cond, rec in summaries.items():
        if "gsm8k" in rec:
            g = rec["gsm8k"]
            rows.append({"condition": cond, **{k: g[k] for k in ["n", "accuracy_flexible", "accuracy_strict", "extractable_rate_flexible", "extractable_rate_strict", "truncation_rate", "accuracy_among_extractable_flexible"]}})
    return pd.DataFrame(rows)


def run_dir(root, run):
    f = find(root, f"**/outputs/task1/{run}/config_resolved.json")
    return f.parent if f else None


def train_log(rd):
    return pd.read_json(rd / "train_log.jsonl", lines=True)


def l2sp_penalty(root):
    out = {}
    l2 = run_dir(root, "l2sp")
    ft = run_dir(root, "full_ft")
    if l2:
        log = train_log(l2)
        out["l2sp_series"] = log[["update", "loss_sft", "omega", "lambda_omega", "loss_total"]].to_dict("list")
        out["l2sp_checkpoints"] = json.loads((l2 / "checkpoint_events.json").read_text())
        sess = json.loads((l2 / "resources_sessions.json").read_text())
        out["reference_overhead"] = {k: sess[0].get(k) for k in ["reference_device", "reference_dtype", "reference_bytes", "reference_build_s"]}
    if ft:
        out["full_ft_checkpoints"] = json.loads((ft / "checkpoint_events.json").read_text())
    disp = lambda events, u: next((e.get("displacement_l2") for e in events if e["update"] == u), None)
    out["final_displacement_l2"] = {
        "full_ft_625": disp(out.get("full_ft_checkpoints", []), 625),
        "full_ft_1250": disp(out.get("full_ft_checkpoints", []), 1250),
        "l2sp_1250": disp(out.get("l2sp_checkpoints", []), 1250),
    }
    return out


def resources(root):
    rows = []
    for run in RUNS:
        rd = run_dir(root, run)
        if not rd:
            continue
        sessions = json.loads((rd / "resources_sessions.json").read_text())
        events = json.loads((rd / "checkpoint_events.json").read_text()) if (rd / "checkpoint_events.json").exists() else []
        cfg = json.loads((rd / "config_resolved.json").read_text())
        log = train_log(rd)
        rows.append({
            "run": run, "method": cfg["method"], "peak_lr": cfg["lr"],
            "trainable_params": sessions[0]["trainable_params"],
            "train_hours": sum(s["train_s_this_session"] for s in sessions) / 3600,
            "s_per_update": sum(s["train_s_this_session"] for s in sessions) / max(1, sum(s["updates_this_session"] for s in sessions)),
            "peak_mem_gb": max(s["peak_mem_gb_this_session"] for s in sessions),
            "updates_logged": int(log["update"].max()), "skipped_updates": int(log["skipped"].sum()),
            "recovery_checkpoint_bytes": max((e.get("recovery_bytes", 0) for e in events), default=None),
            "final_weights_bytes": next((e["weights"]["bytes"] for e in events if e["update"] == 1250 and "weights" in e), None),
            "reference_bytes": sessions[0].get("reference_bytes"),
            "basis_bytes": sessions[0].get("basis_bytes"),
            "svd_preprocessing_s": sessions[0].get("svd_preprocessing_s"),
            "export_s": next((e["weights"].get("export_s") for e in events if e["update"] == 1250 and "weights" in e), None),
            "device": sessions[0]["device"],
        })
    return pd.DataFrame(rows)


def oplora_checks(root):
    op = run_dir(root, "oplora_r16")
    lo = run_dir(root, "lora_r16")
    out = {}
    if op and lo:
        a = load_file(str(op / "checkpoints" / "initial_raw_factors.safetensors"))
        b = load_file(str(lo / "checkpoints" / "initial_raw_factors.safetensors"))
        out["initial_unprojected_factors_identical_to_lora16"] = a.keys() == b.keys() and all(torch.equal(a[k], b[k]) for k in a)
        out["n_factor_tensors"] = len(a)
    if op:
        res = json.loads((op / "oplora_constraint_residuals.json").read_text())
        left = [v["UkT_delta_over_delta"] for v in res.values() if v["UkT_delta_over_delta"] is not None]
        right = [v["delta_Vk_over_delta"] for v in res.values() if v["delta_Vk_over_delta"] is not None]
        out["UkT_dW_over_dW"] = {"max": max(left), "mean": sum(left) / len(left)}
        out["dW_Vk_over_dW"] = {"max": max(right), "mean": sum(right) / len(right)}
        sess = json.loads((op / "resources_sessions.json").read_text())[0]
        events = json.loads((op / "checkpoint_events.json").read_text())
        out["basis_bytes"] = sess.get("basis_bytes")
        out["svd_preprocessing_s"] = sess.get("svd_preprocessing_s")
        out["export_s"] = next((e["weights"].get("export_s") for e in events if "weights" in e), None)
    return out


def merge_checks(root):
    return {p.parent.name: json.loads(p.read_text()) for p in sorted(Path(root).glob("**/merged/*/merge_check.json"))}


def diagnostics(root):
    ws = find(root, "**/task1_diag/weight_space.json")
    act = find(root, "**/task1_diag/activations.json")
    if not (ws and act):
        return None
    w = pd.DataFrame(json.loads(ws.read_text())["rows"])
    a = pd.DataFrame(json.loads(act.read_text())["rows"]).drop(columns=["delta_source"])
    return w.merge(a, on=["condition", "layer"], how="outer")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", default=str(ROOT / "outputs" / "kaggle"))
    args = parser.parse_args()
    out = ROOT / "results" / "task1"
    out.mkdir(parents=True, exist_ok=True)
    summaries = eval_summaries(args.inputs)
    if summaries:
        bench = benchmarks(summaries)
        bench.to_csv(out / "benchmarks.csv", index=False)
        comparisons(bench).to_csv(out / "comparisons.csv", index=False)
        extraction(summaries).to_csv(out / "gsm8k_extraction.csv", index=False)
        save_json({c: {"lm_eval_version": r.get("lm_eval_version"), "versions": r.get("versions"), "backend": r.get("backend")} for c, r in summaries.items()}, out / "lm_eval_versions.json")
    save_json(l2sp_penalty(args.inputs), out / "l2sp_penalty.json")
    res = resources(args.inputs)
    if not res.empty:
        res.to_csv(out / "resources.csv", index=False)
    save_json(oplora_checks(args.inputs), out / "oplora_checks.json")
    save_json(merge_checks(args.inputs), out / "merge_checks.json")
    diag = diagnostics(args.inputs)
    if diag is not None:
        diag.to_csv(out / "diagnostics.csv", index=False)
    print("conditions with eval:", list(summaries), "runs:", list(res["run"]) if not res.empty else [])


if __name__ == "__main__":
    main()
