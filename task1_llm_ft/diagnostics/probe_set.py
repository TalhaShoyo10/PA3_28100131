import argparse

import numpy as np
from datasets import load_dataset

from common.io import ROOT, load_yaml, save_json
from common.seed import SEED


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "configs" / "task1" / "data.yaml"))
    args = parser.parse_args()
    cfg = load_yaml(args.config)
    assert cfg["seed"] == SEED

    gsm = load_dataset(cfg["gsm8k"]["name"], cfg["gsm8k"]["config"], split=cfg["gsm8k"]["split"])
    hella = load_dataset(cfg["hellaswag"]["name"], split=cfg["hellaswag"]["split"])
    assert len(gsm) == cfg["gsm8k"]["expected_size"]
    assert len(hella) == cfg["hellaswag"]["expected_size"]

    n = cfg["probe"]["n_each"]
    rng = np.random.default_rng(SEED)
    gsm_ids = sorted(int(i) for i in rng.choice(len(gsm), size=n, replace=False))
    hella_ids = sorted(int(i) for i in rng.choice(len(hella), size=n, replace=False))

    save_json(
        {
            "seed": SEED,
            "sampling": "np.random.default_rng(6304); choice without replacement, gsm8k first then hellaswag from the same generator; ids sorted",
            "gsm8k": {"dataset": cfg["gsm8k"]["name"], "split": cfg["gsm8k"]["split"], "row_ids": gsm_ids},
            "hellaswag": {
                "dataset": cfg["hellaswag"]["name"],
                "split": cfg["hellaswag"]["split"],
                "row_ids": hella_ids,
                "ind_field": [int(hella[i]["ind"]) for i in hella_ids],
            },
        },
        ROOT / "splits" / "task1" / "probe_set_ids.json",
    )
    print("gsm8k", gsm_ids)
    print("hellaswag", hella_ids)


if __name__ == "__main__":
    main()
