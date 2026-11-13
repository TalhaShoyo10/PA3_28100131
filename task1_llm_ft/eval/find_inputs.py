import argparse
import json
from pathlib import Path

BASE = "Qwen/Qwen3-0.6B"
BASE_REVISION = "c1899de289a04d12100db370d81485cdf75e47ca"


def find_one(root, pattern):
    hits = sorted(Path(root).glob(pattern))
    return str(hits[0]) if hits else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--merged", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    root = args.input
    found = {
        "full_ft_625": find_one(root, "**/full_ft/checkpoints/weights_0625"),
        "full_ft_1250": find_one(root, "**/full_ft/checkpoints/weights_1250"),
        "l2sp_1250": find_one(root, "**/l2sp/checkpoints/weights_1250"),
        "lora_r4": find_one(root, "**/lora_r4/checkpoints/weights_1250"),
        "lora_r16": find_one(root, "**/lora_r16/checkpoints/weights_1250"),
        "oplora_r16": find_one(root, "**/oplora_r16/checkpoints/weights_1250"),
    }
    models = {"baseline": {"path": BASE, "revision": BASE_REVISION}}
    for name in ["full_ft_625", "full_ft_1250", "l2sp_1250"]:
        if found[name]:
            models[name] = {"path": found[name], "revision": None}
    for name in ["lora_r4", "lora_r16", "oplora_r16"]:
        if found[name]:
            models[name] = {"path": str(Path(args.merged) / name), "revision": None, "adapter_weights": found[name]}
    diag = {}
    if found["full_ft_625"]:
        diag["full_ft_625"] = {"kind": "full", "path": found["full_ft_625"]}
    for name in ["full_ft", "l2sp"]:
        if found[f"{name}_1250"]:
            diag[f"{name}_1250"] = {"kind": "full", "path": found[f"{name}_1250"]}
    for name in ["lora_r4", "lora_r16", "oplora_r16"]:
        if found[name]:
            diag[name] = {"kind": "adapter", "path": found[name]}
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "found_inputs.json").write_text(json.dumps(found, indent=2))
    (out / "conditions_models.json").write_text(json.dumps(models, indent=2))
    (out / "conditions_diagnostics.json").write_text(json.dumps(diag, indent=2))
    print(json.dumps(found, indent=1))


if __name__ == "__main__":
    main()
