import argparse
import json
from pathlib import Path

import torch
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM

from common.io import save_json
from task1_llm_ft.eval.merge_adapters import MODEL, REVISION

LAYERS = [0, 13, 27]
PROTECT_RANK = 128


def base_down_proj():
    model = AutoModelForCausalLM.from_pretrained(MODEL, revision=REVISION, dtype=torch.float32)
    return {i: model.model.layers[i].mlp.down_proj.weight.detach().clone() for i in LAYERS}


def full_ft_deltas(source, w0):
    key = lambda i: f"model.layers.{i}.mlp.down_proj.weight"
    diag_file = Path(source) / "diag_down_proj_fp32.safetensors"
    if diag_file.exists():
        state = load_file(str(diag_file))
        precision = "fp32 down_proj weights saved at the checkpoint"
    elif source.endswith(".pt"):
        state = torch.load(source, map_location="cpu", weights_only=False)["trainable"]
        precision = "fp32 recovery-checkpoint weights"
    else:
        state = {}
        for f in sorted(Path(source).glob("*.safetensors")):
            state.update(load_file(str(f)))
        precision = "fp16 saved weights"
    return {i: state[key(i)].float() - w0[i] for i in LAYERS}, precision


def adapter_deltas(weights_dir):
    info = json.loads((Path(weights_dir) / "info.json").read_text())
    factors = load_file(str(Path(weights_dir) / "adapter_factors.safetensors"))
    scale = info["scale"]
    return {i: scale * (factors[f"layers.{i}.down_proj.B"].double() @ factors[f"layers.{i}.down_proj.A"].double()).float() for i in LAYERS}, f"fp32 exported factors, projected={info['factors_are_projected']}"


def metrics(delta, w0, basis):
    delta = delta.double()
    u = basis.double()
    proj = u @ (u.t() @ delta)
    return {
        "rho_k": float(proj.pow(2).sum() / delta.pow(2).sum()),
        "eta": float(torch.linalg.norm(delta) / torch.linalg.norm(w0.double())),
        "delta_fro": float(torch.linalg.norm(delta)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--conditions", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    conditions = json.loads(Path(args.conditions).read_text())
    w0 = base_down_proj()
    bases = {i: torch.linalg.svd(w0[i], full_matrices=False)[0][:, :PROTECT_RANK] for i in LAYERS}
    rows = []
    for name, spec in conditions.items():
        deltas, precision = full_ft_deltas(spec["path"], w0) if spec["kind"] == "full" else adapter_deltas(spec["path"])
        for i in LAYERS:
            rows.append({"condition": name, "layer": i, "module": "down_proj", "k": PROTECT_RANK, "delta_source": precision, **metrics(deltas[i], w0[i], bases[i])})
    save_json({"rows": rows, "k": PROTECT_RANK, "svd": "torch.linalg.svd of FP32 W0, same call as OPLoRA preprocessing", "random_subspace_reference_rho": PROTECT_RANK / w0[0].shape[0]}, Path(args.out) / "weight_space.json")
    print(json.dumps(rows, indent=1)[:3000])


if __name__ == "__main__":
    main()
