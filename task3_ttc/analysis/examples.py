import argparse
import json
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

from common.io import ROOT, load_yaml
from task3_ttc.data.normalize import NO_ANSWER

PER_RULE = 2
ORDER = 0
N_FULL = 32
RWC_METHOD = {"math": "rwc_K", "mathvista": "rwc_Z"}
TEXT_CHARS = 240


def short(text, limit=TEXT_CHARS):
    text = " ".join(str(text).split())
    return text if len(text) <= limit else text[:limit] + " ..."


def rollout_table(cfg, dataset):
    root = Path(cfg["data_root"]) / cfg["datasets"][dataset]["folder"]
    cols = ["sample_id", "rollout_id", "question_text", "gold_answer", "extracted_answer"]
    table = pq.read_table(root / "generations.parquet", columns=cols).to_pandas()
    answers = pd.read_parquet(ROOT / "outputs" / "task3" / "answers" / f"{dataset}.parquet")
    table["sample_id"] = table["sample_id"].astype(str)
    answers["sample_id"] = answers["sample_id"].astype(str)
    return table.merge(answers, on=["sample_id", "rollout_id"], how="inner", validate="one_to_one")


def group_answer(rollouts, sample_id, group):
    rows = rollouts[(rollouts.sample_id == sample_id) & (rollouts.group == group)]
    if rows.empty:
        return "", 0
    if group == NO_ANSWER:
        return "NO_ANSWER", len(rows)
    return str(rows.sort_values("rollout_id").extracted_answer.iloc[0]), len(rows)


def question_info(rollouts, sample_id):
    rows = rollouts[rollouts.sample_id == sample_id]
    return short(rows.question_text.iloc[0]), str(rows.gold_answer.iloc[0]), float(rows.correct.mean())


def selection_disagreement_cases(selections, rollouts, dataset):
    s = selections[(selections.split == "eval") & (selections.order == ORDER) & (selections.N == N_FULL)].copy()
    s["sample_id"] = s["sample_id"].astype(str)
    sc = s[s.method == "self_consistency"].set_index("sample_id")
    rw = s[s.method == RWC_METHOD[dataset]].set_index("sample_id")
    both = sc.join(rw, lsuffix="_sc", rsuffix="_rwc", how="inner")
    differ = both[both.selected_group_sc != both.selected_group_rwc]
    rules = {
        "rwc_correct_sc_wrong": differ.correct_rwc & ~differ.correct_sc,
        "sc_correct_rwc_wrong": differ.correct_sc & ~differ.correct_rwc,
    }
    rows = []
    for rule, mask in rules.items():
        for sid in sorted(differ[mask].index)[:PER_RULE]:
            r = differ.loc[sid]
            question, gold, pool_correct = question_info(rollouts, sid)
            sc_answer, sc_support = group_answer(rollouts, sid, r.selected_group_sc)
            rwc_answer, rwc_support = group_answer(rollouts, sid, r.selected_group_rwc)
            rows.append({
                "dataset": dataset, "rule": rule, "sample_id": sid, "question": question, "gold": gold,
                "sc_answer": sc_answer, "sc_support_of_32": sc_support, "sc_correct": bool(r.correct_sc),
                "rwc_method": RWC_METHOD[dataset], "rwc_answer": rwc_answer, "rwc_support_of_32": rwc_support,
                "rwc_correct": bool(r.correct_rwc), "pool_fraction_correct": pool_correct,
            })
    return rows


def live_records():
    records = []
    for path in sorted((ROOT / "results" / "task3" / "live").glob("live_records_shard*.jsonl")):
        records += [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return {r["sample_id"]: r for r in records}


def revision_transition_cases(rollouts):
    per_q = pd.read_csv(ROOT / "results" / "task3" / "live" / "per_question.csv")
    records = live_records()
    rows = []
    for transition in ["W->C", "C->W"]:
        for sid in sorted(per_q[per_q.transition == transition].sample_id):
            rec = records[sid]
            initial = rec["parallel"][0]
            question = question_info(rollouts, sid)[0] if (rollouts.sample_id == sid).any() else ""
            rows.append({
                "transition": transition, "sample_id": sid, "question": question, "gold": rec["gold_answer"],
                "initial_answer": initial["extracted"], "initial_truncated": initial["truncated"],
                "revised_answer": rec["revision"]["extracted"], "revision_truncated": rec["revision"]["truncated"],
                "revision_opening": short(rec["revision"]["response"], 200),
            })
    return rows


def premature_stop_cases(rollouts_by_dataset):
    stops = pd.read_csv(ROOT / "results" / "task3" / "adaptive" / "adaptive_incorrect_early_stops.csv")
    stops["sample_id"] = stops["sample_id"].astype(str)
    chosen = stops[(stops.split == "eval") & (stops.order == ORDER) & stops.fixed32_would_recover]
    rows = []
    for (dataset, controller), group in chosen.groupby(["dataset", "controller"], sort=True):
        rollouts = rollouts_by_dataset[dataset]
        for _, r in group.sort_values("sample_id").head(PER_RULE).iterrows():
            question, gold, _ = question_info(rollouts, r.sample_id)
            answer, _ = group_answer(rollouts, r.sample_id, r.sc_group)
            rows.append({
                "dataset": dataset, "controller": controller, "sample_id": r.sample_id, "question": question, "gold": gold,
                "stopped_at_n": int(r.n), "answer_at_stop": answer, "leader_count": int(r.leader_count),
                "leader_share_at_stop": float(r.leader_share_at_stop), "rwc_margin": float(r.rwc_margin),
                "pool_fraction_correct": float(r.pool_fraction_correct),
            })
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "configs" / "task3" / "offline.yaml"))
    args = parser.parse_args()
    cfg = load_yaml(args.config)
    out = ROOT / "results" / "task3" / "examples"
    out.mkdir(parents=True, exist_ok=True)
    rollouts = {ds: rollout_table(cfg, ds) for ds in ["math", "mathvista"]}
    selection_rows = []
    for ds in ["math", "mathvista"]:
        selections = pd.read_parquet(ROOT / "outputs" / "task3" / "offline" / f"selections_{ds}.parquet")
        selection_rows += selection_disagreement_cases(selections, rollouts[ds], ds)
    pd.DataFrame(selection_rows).to_csv(out / "selection_disagreement.csv", index=False)
    pd.DataFrame(revision_transition_cases(rollouts["math"])).to_csv(out / "revision_transitions.csv", index=False)
    pd.DataFrame(premature_stop_cases(rollouts)).to_csv(out / "premature_stops.csv", index=False)
    for name in ["selection_disagreement", "revision_transitions", "premature_stops"]:
        print(name, len(pd.read_csv(out / f"{name}.csv")))


if __name__ == "__main__":
    main()
