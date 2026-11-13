import argparse
import hashlib
from pathlib import Path

import pandas as pd

from common.io import ROOT, load_yaml, save_json
from task3_ttc.adaptive.controllers import grid, policy_metrics, precompute_states, simulate

TOLERANCE = 0.01
FAMILIES = ["agreement", "quality_aware"]


def choose(candidates, target):
    eligible = [c for c in candidates if c["accuracy"] >= target - TOLERANCE - 1e-12]
    return min(eligible, key=lambda c: (round(c["mean_token_cost"], 9), -c["accuracy"], c["tau"] if c["tau"] is not None else 2.0, c["gamma"] if c["gamma"] is not None else 2.0))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "configs" / "task3" / "offline.yaml"))
    args = parser.parse_args()
    cfg = load_yaml(args.config)
    out = ROOT / "results" / "task3" / "adaptive"
    out.mkdir(parents=True, exist_ok=True)
    states_dir = ROOT / "outputs" / "task3" / "adaptive"
    states_dir.mkdir(parents=True, exist_ok=True)
    selected = {"tolerance_pp": 1.0, "selection_split": "dev", "datasets": {}}
    grid_rows = []
    for dataset in ["math", "mathvista"]:
        states = precompute_states(cfg, dataset)
        states.to_parquet(states_dir / f"states_{dataset}.parquet", index=False)
        dev = states[states["split"] == "dev"]
        fixed32 = policy_metrics(simulate(dev, "always_32", None, None))
        always = {"family": "always_32", "tau": None, "gamma": None, **fixed32}
        selected["datasets"][dataset] = {"fixed32_dev_accuracy": fixed32["accuracy"], "fixed32_dev_cost": fixed32["mean_token_cost"]}
        for family in FAMILIES:
            candidates = [always]
            for tau, gamma in grid(family):
                m = policy_metrics(simulate(dev, family, tau, gamma))
                row = {"family": family, "tau": tau, "gamma": gamma, **m}
                candidates.append(row)
                grid_rows.append({"dataset": dataset, **row})
            best = choose(candidates, fixed32["accuracy"])
            selected["datasets"][dataset][family] = {k: best[k] for k in ("family", "tau", "gamma", "accuracy", "mean_token_cost", "mean_rollouts")}
        grid_rows.append({"dataset": dataset, **always})
        print(dataset, selected["datasets"][dataset], flush=True)
    pd.DataFrame(grid_rows).to_csv(out / "adaptive_dev_grid.csv", index=False)
    selected["controllers_py_sha256"] = hashlib.sha256((Path(__file__).parent / "controllers.py").read_bytes()).hexdigest()
    target = ROOT / "results" / "task3" / "adaptive_selected.json"
    assert not target.exists(), f"{target} already frozen; delete it deliberately to reselect"
    save_json(selected, target)


if __name__ == "__main__":
    main()
