import argparse

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from common.io import ROOT, load_json, load_yaml, save_json
from task3_ttc.offline.run_budgets import load_dataset_tables, question_rows
from task3_ttc.scoring.visconf import calibrated_confidence, gather_tokens, token_covariance, visconf_scores

QUARTILE = 8
TRAJECTORY_ORDER = 0
TRAJECTORY_N = 32


def full_pool_scores(meta, token_arrays, rows, vlm):
    scores = {"L": meta["L"].to_numpy()[rows], "K": meta["K"].to_numpy()[rows], "shortest": -meta["T"].to_numpy()[rows].astype(np.float64)}
    if vlm:
        vis = visconf_scores(token_arrays, rows)
        scores.update({"V": meta["V"].to_numpy()[rows], "C": vis["C"], "Z": vis["Z"], "Ztok": vis["Ztok"]})
    return scores


def quartile_gap(score, correct):
    order = np.argsort(-score, kind="stable")
    return float(correct[order[:QUARTILE]].mean() - correct[order[-QUARTILE:]].mean())


def within_question_rows(meta, token_arrays, rows_by_q, split_of, vlm):
    correct_all = meta["correct"].to_numpy().astype(np.float64)
    lengths = meta["T"].to_numpy().astype(np.float64)
    records = []
    for sample_id, rows in rows_by_q.items():
        correct = correct_all[rows]
        scores = full_pool_scores(meta, token_arrays, rows, vlm)
        rec = {"sample_id": sample_id, "split": split_of[sample_id], "fraction_correct": float(correct.mean())}
        for name, s in scores.items():
            rec[f"quartile_gap_{name}"] = quartile_gap(s, correct)
        varied = 0 < correct.mean() < 1
        rec["mixed_correctness"] = bool(varied)
        rec["spearman_length_correct"] = float(spearmanr(lengths[rows], correct).statistic) if varied and np.ptp(lengths[rows]) > 0 else np.nan
        records.append(rec)
    return pd.DataFrame(records)


def confidence_vision_rows(meta, token_arrays, rows_by_q, split_of):
    offsets = token_arrays["offsets"]
    records = []
    identity_errors = []
    for sample_id, rows in rows_by_q.items():
        k, lens = gather_tokens(token_arrays["k"], offsets, rows)
        v, _ = gather_tokens(token_arrays["v"], offsets, rows)
        c = calibrated_confidence(k)
        starts = np.concatenate([[0], np.cumsum(lens)[:-1]])
        vis = visconf_scores(token_arrays, rows)
        cov = token_covariance(token_arrays, rows)
        identity_errors.append(np.abs((vis["Ztok"] - vis["Z"]) - cov).max())
        corr_c, corr_k = [], []
        for s, n in zip(starts, lens):
            ci, ki, vi = c[s:s + n], k[s:s + n], v[s:s + n]
            corr_c.append(np.corrcoef(ci, vi)[0, 1] if n > 2 and ci.std() > 0 and vi.std() > 0 else np.nan)
            corr_k.append(np.corrcoef(ki, vi)[0, 1] if n > 2 and ki.std() > 0 and vi.std() > 0 else np.nan)
        records.append({
            "sample_id": sample_id, "split": split_of[sample_id],
            "mean_rollout_corr_c_v": float(np.nanmean(corr_c)) if np.isfinite(corr_c).any() else np.nan,
            "mean_rollout_corr_k_v": float(np.nanmean(corr_k)) if np.isfinite(corr_k).any() else np.nan,
            "rollouts_with_defined_corr": int(np.isfinite(corr_c).sum()),
        })
    return pd.DataFrame(records), float(np.max(identity_errors))


def describe(series):
    s = series.dropna()
    return {"n_questions": int(len(s)), "mean": float(s.mean()), "median": float(s.median()), "q25": float(s.quantile(0.25)), "q75": float(s.quantile(0.75)), "fraction_positive": float((s > 0).mean())}


def summarize_quartiles(table, vlm):
    names = ["L", "K", "shortest"] + (["V", "C", "Z", "Ztok"] if vlm else [])
    rows = []
    for split in ["dev", "eval", "all"]:
        sub = table if split == "all" else table[table["split"] == split]
        for name in names:
            col = sub[f"quartile_gap_{name}"]
            mixed = sub.loc[sub["mixed_correctness"], f"quartile_gap_{name}"]
            rows.append({"split": split, "score": name, "mean_gap_all_questions": float(col.mean()), "mean_gap_mixed_questions": float(mixed.mean()), "n_questions": int(len(col)), "n_mixed": int(len(mixed))})
        rows.append({"split": split, "score": "spearman_length_correct", "mean_gap_all_questions": float(sub["spearman_length_correct"].mean()), "mean_gap_mixed_questions": float(sub["spearman_length_correct"].mean()), "n_questions": int(sub["spearman_length_correct"].notna().sum()), "n_mixed": int(sub["mixed_correctness"].sum())})
    return pd.DataFrame(rows)


def illustrative_trajectory_cases(selections, order_index, splits):
    sel = selections[(selections["order"] == TRAJECTORY_ORDER) & (selections["N"] == TRAJECTORY_N)]
    wide = sel.pivot(index="sample_id", columns="method", values=["selected_group", "correct", "selected_rollout"])
    rules = {
        "max_Z_vs_max_Ztok_disagree": lambda r: r[("selected_group", "max_Z")] != r[("selected_group", "max_Ztok")],
        "rwc_Z_correct_self_consistency_wrong": lambda r: r[("correct", "rwc_Z")] and not r[("correct", "self_consistency")],
        "rwc_Z_wrong_self_consistency_correct": lambda r: (not r[("correct", "rwc_Z")]) and r[("correct", "self_consistency")],
    }
    cases = []
    for rule, test in rules.items():
        for sample_id in splits["eval_ids"]:
            if test(wide.loc[sample_id]):
                rollouts = sorted({int(wide.loc[sample_id][("selected_rollout", m)]) for m in ["self_consistency", "max_Z", "max_Ztok", "rwc_Z"]})
                cases.append({"rule": rule, "sample_id": sample_id, "rollouts": rollouts})
                break
    return cases


def trajectory_rows(cases, meta, token_arrays, rows_by_q):
    offsets = token_arrays["offsets"]
    out = []
    for case in cases:
        rows = rows_by_q[case["sample_id"]]
        k, lens = gather_tokens(token_arrays["k"], offsets, rows)
        c = calibrated_confidence(k)
        starts = np.concatenate([[0], np.cumsum(lens)[:-1]])
        for rid in case["rollouts"]:
            s, n = starts[rid], lens[rid]
            row = rows[rid]
            v = token_arrays["v"][offsets[row]:offsets[row + 1]]
            for t in range(n):
                out.append({"rule": case["rule"], "sample_id": case["sample_id"], "rollout_id": rid, "correct": bool(meta["correct"].iloc[row]), "t": t, "k": float(k[s + t]), "c": float(c[s + t]), "v": float(v[t])})
    return pd.DataFrame(out)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "configs" / "task3" / "offline.yaml"))
    parser.add_argument("--datasets", nargs="+", default=["math", "mathvista"])
    args = parser.parse_args()
    cfg = load_yaml(args.config)
    out = ROOT / "results" / "task3" / "signals"
    out.mkdir(parents=True, exist_ok=True)
    for dataset in args.datasets:
        meta, token_arrays, vlm = load_dataset_tables(cfg, dataset)
        splits = load_json(ROOT / "splits" / "task3" / f"{dataset}_dev_eval_ids.json")
        split_of = {q: "dev" for q in splits["dev_ids"]} | {q: "eval" for q in splits["eval_ids"]}
        rows_by_q = question_rows(meta)
        table = within_question_rows(meta, token_arrays, rows_by_q, split_of, vlm)
        table.to_csv(out / f"within_question_{dataset}.csv", index=False)
        summarize_quartiles(table, vlm).to_csv(out / f"quartile_summary_{dataset}.csv", index=False)
        info = {"dataset": dataset, "N": 32, "quartile_size": QUARTILE}
        if vlm:
            cv, identity_error = confidence_vision_rows(meta, token_arrays, rows_by_q, split_of)
            cv.to_csv(out / f"confidence_vision_{dataset}.csv", index=False)
            info["identity_max_abs_error_Ztok_minus_Z_vs_cov"] = identity_error
            info["confidence_vision_corr_c_v"] = {s: describe(cv[cv["split"] == s]["mean_rollout_corr_c_v"]) for s in ["dev", "eval"]} | {"all": describe(cv["mean_rollout_corr_c_v"])}
            info["confidence_vision_corr_k_v"] = {"all": describe(cv["mean_rollout_corr_k_v"])}
            selections = pd.read_parquet(ROOT / "outputs" / "task3" / "offline" / f"selections_{dataset}.parquet")
            cases = illustrative_trajectory_cases(selections, None, splits)
            info["trajectory_cases"] = cases
            trajectory_rows(cases, meta, token_arrays, rows_by_q).to_csv(out / f"trajectories_{dataset}.csv", index=False)
        save_json(info, out / f"signal_info_{dataset}.json")
        print(dataset, info, flush=True)


if __name__ == "__main__":
    main()
