import argparse
import json
from pathlib import Path

import torch
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM, AutoTokenizer

from common.io import ROOT, load_json, save_json
from task1_llm_ft.data.formatting import render_prompt
from task1_llm_ft.methods.lora import LoRALinear, target_modules

MODEL = "Qwen/Qwen3-0.6B"
REVISION = "c1899de289a04d12100db370d81485cdf75e47ca"


def load_factors(weights_dir):
    info = load_json(Path(weights_dir) / "info.json")
    factors = load_file(str(Path(weights_dir) / "adapter_factors.safetensors"))
    return factors, info["scale"], info["factors_are_projected"]


def delta_weights(factors, scale):
    deltas = {}
    for key in {k.rsplit(".", 1)[0] for k in factors}:
        deltas[key] = scale * (factors[f"{key}.B"].double() @ factors[f"{key}.A"].double()).float()
    return deltas


def merged_model(factors, scale, device):
    model = AutoModelForCausalLM.from_pretrained(MODEL, revision=REVISION, dtype=torch.float32)
    deltas = delta_weights(factors, scale)
    rounding = 0.0
    with torch.no_grad():
        for layer_index, name, parent in target_modules(model):
            linear = getattr(parent, name)
            w0 = linear.weight.detach().clone()
            delta = deltas[f"layers.{layer_index}.{name}"]
            linear.weight.add_(delta)
            stored = linear.weight.half().float()
            rounding = max(rounding, float(((stored - w0) - delta).abs().max()))
    assert len(deltas) == sum(1 for _ in target_modules(model))
    return model.half().to(device).eval(), rounding


def unmerged_model(factors, scale, device):
    model = AutoModelForCausalLM.from_pretrained(MODEL, revision=REVISION, dtype=torch.float16).to(device)
    rank = next(v for k, v in factors.items() if k.endswith(".A")).shape[0]
    for layer_index, name, parent in target_modules(model):
        module = LoRALinear(getattr(parent, name), rank, scale * rank, 0.0)
        with torch.no_grad():
            module.lora_A.copy_(factors[f"layers.{layer_index}.{name}.A"].to(device))
            module.lora_B.copy_(factors[f"layers.{layer_index}.{name}.B"].to(device))
        setattr(parent, name, module)
    return model.eval()


def check_batch(tokenizer, device):
    probe = load_json(ROOT / "splits" / "task1" / "probe_set_ids.json")
    from datasets import load_dataset
    gsm = load_dataset("openai/gsm8k", "main", split="test")
    prompts = [render_prompt(tokenizer, gsm[int(i)]["question"]) for i in probe["gsm8k"]["row_ids"][:4]]
    return tokenizer(prompts, return_tensors="pt", padding=True).to(device)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights-dir", required=True)
    parser.add_argument("--condition", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    device = torch.device("cuda")
    factors, scale, projected = load_factors(args.weights_dir)
    tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=REVISION)
    tokenizer.padding_side = "left"
    batch = check_batch(tokenizer, device)

    with torch.no_grad():
        reference = unmerged_model(factors, scale, device)
        unmerged_logits = reference(**batch).logits.float()
        del reference
        torch.cuda.empty_cache()
        merged, rounding = merged_model(factors, scale, device)
        merged_logits = merged(**batch).logits.float()

    keep = batch["attention_mask"].bool()
    diff = (merged_logits - unmerged_logits).abs()[keep]
    out_dir = Path(args.out) / args.condition
    merged.save_pretrained(out_dir, safe_serialization=True)
    tokenizer.padding_side = "right"
    tokenizer.save_pretrained(out_dir)
    record = {
        "condition": args.condition,
        "weights_dir": str(args.weights_dir),
        "factors_are_projected": projected,
        "scale": scale,
        "check_batch": "first 4 GSM8K probe-set questions, chat template with enable_thinking=False, FP16, non-padding positions",
        "max_abs_logit_diff_merged_vs_unmerged": float(diff.max()),
        "mean_abs_logit_diff_merged_vs_unmerged": float(diff.mean()),
        "max_abs_fp16_rounding_of_merged_minus_w0_vs_delta_w": rounding,
    }
    save_json(record, out_dir / "merge_check.json")
    print(json.dumps(record), flush=True)


if __name__ == "__main__":
    main()
