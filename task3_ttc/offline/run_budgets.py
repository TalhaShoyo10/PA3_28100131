import argparse
import time

import numpy as np
import pandas as pd

from common.bootstrap import paired_bootstrap
from common.io import ROOT, load_json, load_yaml, save_json
from common.seed import SEED
from task3_ttc.data.load_join import load_model_data
from task3_ttc.data.normalize import NO_ANSWER
from task3_ttc.scoring.aggregate import highest_score, rwc, self_consistency
from task3_ttc.scoring.signals import rollout_signals
from task3_ttc.scoring.visconf import visconf_scores

BUDGETS = [1, 2, 4, 8, 16, 32]
N_BOOT = 1000
BASE_METHODS = ["first_ans", "self_consistency", "max_L", "max_K", "rwc_K", "shortest"]
VLM_METHODS = ["max_V", "rwc_V", "max_Z", "rwc_Z", "max_Ztok", "rwc_Ztok"]
CONTRASTS = [
    ("self_consistency", "first_ans"),
    ("max_L", "self_consistency"),
    ("max_K", "self_consistency"),
    ("rwc_K", "self_consistency"),
    ("rwc_K", "max_K"),
    ("rwc_V", "self_consistency"),
    ("rwc_Z", "self_consistency"),
    ("rwc_Z", "rwc_K"),
    ("rwc_Z", "rwc_V"),
    ("max_Z", "max_V"),
    ("max_Ztok", "max_Z"),
    ("rwc_Ztok", "rwc_Z"),
]


def load_dataset_tables(cfg, dataset):
    spec = cfg["datasets"][dataset]
    meta, tokens = load_model_data(cfg["data_root"], spec["folder"], spec["vlm"])
    answers = pd.read_parquet(ROOT / "outputs" / "task3" / "answers" / f"{dataset}.parquet")
    meta = meta.merge(answers, on=["sample_id", "rollout_id"], how="left", validate="one_to_one")
    assert meta["group"].notna().all() and np.array_equal(meta["row"].to_numpy(), np.arange(len(meta)))
    signals, token_arrays = rollout_signals(meta, tokens, spec["vlm"])
    for name, values in signals.items():
        meta[name] = values
    return meta, token_arrays, spec["vlm"]


def question_rows(meta):
    rows = {}
    for sample_id, block in meta.groupby("sample_id", sort=True):
        block = block.sort_values("rollout_id")
        assert block["rollout_id"].tolist() == list(range(len(block)))
        rows[sample_id] = block["row"].to_numpy()
    return rows


def select_all(methods, rows, meta_arrays, token_arrays, vlm):
    groups = meta_arrays["group"][rows].tolist()
    picks = {}
    picks["first_ans"] = 0
    picks["self_consistency"], _ = self_consistency(groups)
    picks["max_L"] = highest_score(meta_arrays["L"][rows])
    picks["max_K"] = highest_score(meta_arrays["K"][rows])
    picks["rwc_K"], _ = rwc(groups, meta_arrays["K"][rows])
    picks["shortest"] = highest_score(-meta_arrays["T"][rows].astype(np.float64))
    if vlm:
        vis = visconf_scores(token_arrays, rows)
        picks["max_V"] = highest_score(meta_arrays["V"][rows])
        picks["rwc_V"], _ = rwc(groups, meta_arrays["V"][rows])
        picks["max_Z"] = highest_score(vis["Z"])
        picks["rwc_Z"], _ = rwc(groups, vis["Z"])
        picks["max_Ztok"] = highest_score(vis["Ztok"])
        picks["rwc_Ztok"], _ = rwc(groups, vis["Ztok"])
    return picks


def run_dataset(cfg, dataset, splits, orders, order_index):
    meta, token_arrays, vlm = load_dataset_tables(cfg, dataset)
    methods = BASE_METHODS + (VLM_METHODS if vlm else [])
    arrays = {c: meta[c].to_numpy() for c in ["group", "correct", "T", "L", "K", "rollout_id"] + (["V"] if vlm else [])}
    rows_by_q = question_rows(meta)
    split_of = {q: "dev" for q in splits["dev_ids"]} | {q: "eval" for q in splits["eval_ids"]}
    records = []
    budget_rows = []
    for q_index, sample_id in enumerate(order_index["question_ids_sorted"]):
        q_rows = rows_by_q[sample_id]
        for o in range(orders.shape[1]):
            order = orders[q_index, o]
            for n in BUDGETS:
                rows = q_rows[order[:n]]
                correct = arrays["correct"][rows]
                groups = arrays["group"][rows]
                prefix_cost = int(arrays["T"][rows].sum())
                budget_rows.append({
                    "sample_id": sample_id, "split": split_of[sample_id], "order": o, "N": n,
                    "oracle": bool(correct.any()),
                    "distinct_groups": int(len(set(groups.tolist()))),
                    "distinct_groups_excl_no_answer": int(len(set(g for g in groups.tolist() if g != NO_ANSWER))),
                    "fraction_correct": float(correct.mean()),
                    "prefix_cost": prefix_cost,
                })
                picks = select_all(methods, rows, arrays, token_arrays, vlm)
                for method in methods:
                    position = picks[method]
                    records.append({
                        "sample_id": sample_id, "split": split_of[sample_id], "order": o, "N": n, "method": method,
                        "selected_position": int(position),
                        "selected_rollout": int(arrays["rollout_id"][rows[position]]),
                        "selected_group": int(groups[position]),
                        "correct": bool(correct[position]),
                        "cost": int(arrays["T"][rows[0]]) if method == "first_ans" else prefix_cost,
                    })
        if q_index % 100 == 0:
            print(dataset, "question", q_index, flush=True)
    return pd.DataFrame(records), pd.DataFrame(budget_rows), methods


def question_means(df, value, keys):
    return df.groupby(keys + ["sample_id"], sort=True)[value].mean().reset_index()


def summarize(selections, budgets, dataset):
    per_q = question_means(selections, "correct", ["split", "N", "method"])
    acc = per_q.groupby(["split", "N", "method"])["correct"].mean().rename("accuracy")
    cost_q = question_means(selections, "cost", ["split", "N", "method"])
    cost = cost_q.groupby(["split", "N", "method"])["cost"].mean().rename("mean_token_cost")
    table = pd.concat([acc, cost], axis=1).reset_index()
    pool = budgets.groupby(["split", "N", "sample_id"])[["oracle", "distinct_groups", "distinct_groups_excl_no_answer", "fraction_correct"]].mean().reset_index()
    pool_summary = pool.groupby(["split", "N"])[["oracle", "distinct_groups", "distinct_groups_excl_no_answer", "fraction_correct"]].mean().reset_index()
    pool_summary = pool_summary.rename(columns={"oracle": "oracle_coverage", "distinct_groups": "mean_distinct_groups", "distinct_groups_excl_no_answer": "mean_distinct_groups_excl_no_answer", "fraction_correct": "mean_fraction_correct_in_pool"})
    table = table.merge(pool_summary, on=["split", "N"])
    table.insert(0, "dataset", dataset)

    joined = selections.merge(budgets[["sample_id", "order", "N", "oracle"]], on=["sample_id", "order", "N"])
    joined["sampling_failure"] = ~joined["oracle"]
    joined["selection_failure"] = joined["oracle"] & ~joined["correct"]
    fq = joined.groupby(["split", "N", "method", "sample_id"])[["sampling_failure", "selection_failure", "correct"]].mean().reset_index()
    failures = fq.groupby(["split", "N", "method"])[["sampling_failure", "selection_failure", "correct"]].mean().reset_index()
    failures = failures.rename(columns={"correct": "accuracy"})
    failures.insert(0, "dataset", dataset)
    return table, failures, per_q


def contrasts(per_q, methods, dataset):
    rows = []
    for split in ["dev", "eval"]:
        for n in BUDGETS:
            sub = per_q[(per_q["split"] == split) & (per_q["N"] == n)].pivot(index="sample_id", columns="method", values="correct")
            for a, b in CONTRASTS:
                if a in methods and b in methods:
                    res = paired_bootstrap(sub[a].to_numpy(), sub[b].to_numpy(), N_BOOT, seed=SEED)
                    rows.append({"dataset": dataset, "split": split, "N": n, "method_a": a, "method_b": b, **res})
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "configs" / "task3" / "offline.yaml"))
    parser.add_argument("--datasets", nargs="+", default=["math", "mathvista"])
    args = parser.parse_args()
    cfg = load_yaml(args.config)
    out_raw = ROOT / "outputs" / "task3" / "offline"
    out_raw.mkdir(parents=True, exist_ok=True)
    out_res = ROOT / "results" / "task3" / "offline"
    out_res.mkdir(parents=True, exist_ok=True)
    for dataset in args.datasets:
        start = time.perf_counter()
        splits = load_json(ROOT / "splits" / "task3" / f"{dataset}_dev_eval_ids.json")
        orders = np.load(ROOT / "splits" / "task3" / f"rollout_orders_{dataset}.npy")
        order_index = load_json(ROOT / "splits" / "task3" / f"rollout_orders_{dataset}_index.json")
        selections, budgets, methods = run_dataset(cfg, dataset, splits, orders, order_index)
        selections.to_parquet(out_raw / f"selections_{dataset}.parquet", index=False)
        budgets.to_parquet(out_raw / f"budgets_{dataset}.parquet", index=False)
        table, failures, per_q = summarize(selections, budgets, dataset)
        table.to_csv(out_res / f"accuracy_cost_{dataset}.csv", index=False)
        failures.to_csv(out_res / f"failure_decomposition_{dataset}.csv", index=False)
        per_q.to_csv(out_res / f"per_question_accuracy_{dataset}.csv", index=False)
        contrasts(per_q, methods, dataset).to_csv(out_res / f"bootstrap_contrasts_{dataset}.csv", index=False)
        save_json({"dataset": dataset, "methods": methods, "budgets": BUDGETS, "orders": int(orders.shape[1]), "bootstrap_resamples": N_BOOT, "seconds": time.perf_counter() - start}, out_res / f"run_info_{dataset}.json")
        print(dataset, "done in", round(time.perf_counter() - start), "s", flush=True)


if __name__ == "__main__":
    main()
