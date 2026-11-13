import argparse

import pandas as pd

from common.io import ROOT, load_json, save_json
from task3_ttc.adaptive.controllers import CHECKPOINTS, policy_metrics, simulate


def main():
    parser = argparse.ArgumentParser()
    parser.parse_args()
    selected = load_json(ROOT / "results" / "task3" / "adaptive_selected.json")
    out = ROOT / "results" / "task3" / "adaptive"
    summary_rows, stop_rows, early_rows = [], [], []
    for dataset, chosen in selected["datasets"].items():
        states = pd.read_parquet(ROOT / "outputs" / "task3" / "adaptive" / f"states_{dataset}.parquet")
        held = states[states["split"] == "eval"]
        fixed = {n: policy_metrics(held[held["n"] == n].rename(columns={})) for n in CHECKPOINTS}
        for n, m in fixed.items():
            summary_rows.append({"dataset": dataset, "policy": f"fixed_{n}_self_consistency", "tau": None, "gamma": None, **m})
        sc32 = held[held["n"] == 32].set_index(["sample_id", "order"])["sc_correct"]
        for family in ["agreement", "quality_aware"]:
            c = chosen[family]
            stopped = simulate(held, c["family"], c["tau"], c["gamma"])
            m = policy_metrics(stopped)
            summary_rows.append({"dataset": dataset, "policy": f"adaptive_{family}", "selected_family": c["family"], "tau": c["tau"], "gamma": c["gamma"], **m})
            s = stopped.copy()
            s["dataset"], s["controller"] = dataset, family
            s["fixed32_correct"] = sc32.loc[list(zip(s["sample_id"], s["order"]))].to_numpy()
            s["leader_share_at_stop"] = s["leader_count"] / s["n"]
            stop_rows.append(s)
            early = s[(s["n"] < 32) & ~s["sc_correct"]]
            early_rows.append(early.assign(fixed32_would_recover=early["fixed32_correct"]))
    pd.DataFrame(summary_rows).to_csv(out / "adaptive_heldout_summary.csv", index=False)
    stops = pd.concat(stop_rows)
    stops.to_csv(out / "adaptive_heldout_stops.csv", index=False)
    early = pd.concat(early_rows)
    early.to_csv(out / "adaptive_incorrect_early_stops.csv", index=False)
    retro = stops.groupby(["dataset", "controller", "n"]).agg(
        count=("sample_id", "size"),
        mean_leader_share=("leader_share_at_stop", "mean"),
        mean_pool_fraction_correct=("pool_fraction_correct", "mean"),
        accuracy=("sc_correct", "mean"),
    ).reset_index()
    retro.to_csv(out / "adaptive_stop_retrospective.csv", index=False)
    save_json({"selected": selected, "incorrect_early_stops": int(len(early)), "of_which_fixed32_correct": int(early["fixed32_would_recover"].sum())}, out / "adaptive_heldout_info.json")
    print(pd.DataFrame(summary_rows).round(4).to_string(index=False))


if __name__ == "__main__":
    main()
