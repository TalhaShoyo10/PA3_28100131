import numpy as np
import pandas as pd

from common.io import ROOT, load_json
from task3_ttc.data.normalize import NO_ANSWER
from task3_ttc.offline.run_budgets import load_dataset_tables, question_rows
from task3_ttc.scoring.aggregate import rwc, self_consistency, vote_margin
from task3_ttc.scoring.visconf import visconf_scores

CHECKPOINTS = [4, 8, 16, 32]
TAUS = [0.5, 0.625, 0.75, 0.875, 1.0]
GAMMAS = [0.0, 0.1, 0.25, 0.5]
QUALITY_SCORE = {"math": "K", "mathvista": "Z"}


def revealed_quality(dataset, rows, arrays, token_arrays):
    if QUALITY_SCORE[dataset] == "K":
        return arrays["K"][rows]
    return visconf_scores(token_arrays, rows)["Z"]


def precompute_states(cfg, dataset):
    meta, token_arrays, vlm = load_dataset_tables(cfg, dataset)
    arrays = {c: meta[c].to_numpy() for c in ["group", "correct", "T", "K"]}
    rows_by_q = question_rows(meta)
    splits = load_json(ROOT / "splits" / "task3" / f"{dataset}_dev_eval_ids.json")
    split_of = {q: "dev" for q in splits["dev_ids"]} | {q: "eval" for q in splits["eval_ids"]}
    orders = np.load(ROOT / "splits" / "task3" / f"rollout_orders_{dataset}.npy")
    index = load_json(ROOT / "splits" / "task3" / f"rollout_orders_{dataset}_index.json")
    states = []
    for q_index, sample_id in enumerate(index["question_ids_sorted"]):
        q_rows = rows_by_q[sample_id]
        pool_fraction_correct = float(arrays["correct"][q_rows].mean())
        for o in range(orders.shape[1]):
            for n in CHECKPOINTS:
                rows = q_rows[orders[q_index, o, :n]]
                groups = arrays["group"][rows].tolist()
                sc_pos, counts = self_consistency(groups)
                top = max(counts.values())
                leaders = [g for g, c in counts.items() if c == top]
                rwc_pos, rwc_totals = rwc(groups, revealed_quality(dataset, rows, arrays, token_arrays))
                states.append({
                    "sample_id": sample_id, "split": split_of[sample_id], "order": o, "n": n,
                    "sc_group": int(groups[sc_pos]), "sc_correct": bool(arrays["correct"][rows][sc_pos]),
                    "leader_count": int(top), "leader_unique": len(leaders) == 1,
                    "leader_is_no_answer": groups[sc_pos] == NO_ANSWER,
                    "rwc_group": int(groups[rwc_pos]), "rwc_margin": float(vote_margin(rwc_totals)),
                    "cost": int(arrays["T"][rows].sum()),
                    "pool_fraction_correct": pool_fraction_correct,
                })
    return pd.DataFrame(states)


def agreement_stop(state, tau):
    return state["leader_unique"] & ~state["leader_is_no_answer"] & (state["leader_count"] >= tau * state["n"])


def quality_stop(state, tau, gamma):
    return agreement_stop(state, tau) & (state["sc_group"] == state["rwc_group"]) & (state["rwc_margin"] >= gamma)


def simulate(states, family, tau, gamma):
    if family == "always_32":
        stop = states["n"] == 32
    elif family == "agreement":
        stop = agreement_stop(states, tau) | (states["n"] == 32)
    else:
        stop = quality_stop(states, tau, gamma) | (states["n"] == 32)
    stopped = states[stop].sort_values(["sample_id", "order", "n"]).groupby(["sample_id", "order"], sort=True).head(1)
    assert len(stopped) == states.groupby(["sample_id", "order"]).ngroups
    return stopped


def policy_metrics(stopped):
    per_q = stopped.groupby("sample_id")[["sc_correct", "cost", "n"]].mean()
    counts = stopped["n"].value_counts(normalize=True).reindex(CHECKPOINTS, fill_value=0.0)
    return {
        "accuracy": float(per_q["sc_correct"].mean()),
        "mean_token_cost": float(per_q["cost"].mean()),
        "mean_rollouts": float(per_q["n"].mean()),
        **{f"stop_frac_n{n}": float(counts[n]) for n in CHECKPOINTS},
        "n_questions": int(len(per_q)),
    }


def grid(family):
    if family == "always_32":
        return [(None, None)]
    if family == "agreement":
        return [(tau, None) for tau in TAUS]
    return [(tau, gamma) for tau in TAUS for gamma in GAMMAS]
