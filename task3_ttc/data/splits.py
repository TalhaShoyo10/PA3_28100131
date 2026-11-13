import argparse
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from common.io import ROOT, load_yaml, save_json
from common.seed import SEED, set_all_seeds

N_DEV = 200
N_ORDERS = 10
ORDER_STREAM = {"math": 1, "mathvista": 2}


def question_ids(data_root, folder):
    table = pq.read_table(Path(data_root) / folder / "generations.parquet", columns=["sample_id"])
    return sorted(set(table.column("sample_id").to_pylist()))


def dev_eval_split(sorted_ids):
    permuted = np.random.default_rng(SEED).permutation(len(sorted_ids))
    shuffled = [sorted_ids[i] for i in permuted]
    return shuffled[:N_DEV], shuffled[N_DEV:]


def rollout_orders(sorted_ids, dataset, rollouts_per_question):
    rng = np.random.default_rng([SEED, ORDER_STREAM[dataset]])
    orders = np.empty((len(sorted_ids), N_ORDERS, rollouts_per_question), dtype=np.int16)
    for q in range(len(sorted_ids)):
        for o in range(N_ORDERS):
            orders[q, o] = rng.permutation(rollouts_per_question)
    return orders


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "configs" / "task3" / "offline.yaml"))
    args = parser.parse_args()
    cfg = load_yaml(args.config)
    set_all_seeds(cfg["seed"])
    assert cfg["seed"] == SEED
    out = ROOT / "splits" / "task3"
    out.mkdir(parents=True, exist_ok=True)

    for dataset, spec in cfg["datasets"].items():
        ids = question_ids(cfg["data_root"], spec["folder"])
        assert len(ids) == spec["expected"]["questions"]
        dev, held_out = dev_eval_split(ids)
        assert len(dev) == N_DEV and len(held_out) == len(ids) - N_DEV
        assert not set(dev) & set(held_out)
        record = {
            "dataset": dataset,
            "seed": SEED,
            "procedure": "sort question ids lexicographically as strings; permute with np.random.default_rng(6304).permutation; first 200 dev, remaining 800 held-out evaluation",
            "n_dev": len(dev),
            "n_eval": len(held_out),
            "dev_ids": dev,
            "eval_ids": held_out,
        }
        if dataset == "math":
            record["live_50_ids"] = held_out[:50]
            save_json(
                {"dataset": dataset, "source": "first 50 held-out evaluation questions in permutation order", "ids": held_out[:50]},
                out / "live_50_ids.json",
            )
        save_json(record, out / f"{dataset}_dev_eval_ids.json")

        orders = rollout_orders(ids, dataset, spec["expected"]["rollouts_per_question"])
        assert all(sorted(orders[q, o]) == list(range(orders.shape[2])) for q in range(len(ids)) for o in range(N_ORDERS))
        np.save(out / f"rollout_orders_{dataset}.npy", orders)
        save_json(
            {
                "dataset": dataset,
                "seed_sequence": [SEED, ORDER_STREAM[dataset]],
                "array_shape": list(orders.shape),
                "axes": ["question (index into question_ids_sorted)", "order", "position -> rollout_id"],
                "question_ids_sorted": ids,
            },
            out / f"rollout_orders_{dataset}_index.json",
        )
        print(dataset, len(dev), len(held_out), orders.shape)


if __name__ == "__main__":
    main()
