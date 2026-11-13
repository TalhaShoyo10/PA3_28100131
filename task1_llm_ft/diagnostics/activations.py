import argparse
import json
from pathlib import Path

import torch
from datasets import load_dataset
from lm_eval.tasks import TaskManager
from transformers import AutoModelForCausalLM, AutoTokenizer

from common.io import ROOT, load_json, save_json
from task1_llm_ft.diagnostics.weight_space import LAYERS, adapter_deltas, base_down_proj, full_ft_deltas
from task1_llm_ft.eval.merge_adapters import MODEL, REVISION

EPS = 1e-8


def benchmark_prompts(tokenizer):
    probe = load_json(ROOT / "splits" / "task1" / "probe_set_ids.json")
    tm = TaskManager()
    gsm_task = tm.load_task_or_group(["gsm8k_cot"])["gsm8k_cot"]
    hs_task = tm.load_task_or_group(["hellaswag"])["hellaswag"]
    gsm = load_dataset("openai/gsm8k", "main", split="test")
    hs = hs_task.validation_docs()
    prompts = []
    for i in probe["gsm8k"]["row_ids"]:
        text = gsm_task.doc_to_text(gsm[int(i)])
        chat = tokenizer.apply_chat_template([{"role": "user", "content": text}], tokenize=False, add_generation_prompt=True, enable_thinking=False)
        prompts.append({"group": "math", "dataset": "gsm8k", "row": int(i), "text": chat})
    for i in probe["hellaswag"]["row_ids"]:
        prompts.append({"group": "non_math", "dataset": "hellaswag", "row": int(i), "text": hs_task.doc_to_text(hs[int(i)])})
    return prompts


def cache_inputs(prompts, tokenizer, device):
    model = AutoModelForCausalLM.from_pretrained(MODEL, revision=REVISION, dtype=torch.float32).to(device).eval()
    captured = {}
    hooks = []
    for layer in LAYERS:
        def hook(module, inputs, layer=layer):
            captured[layer] = inputs[0].detach()
        hooks.append(model.model.layers[layer].mlp.down_proj.register_forward_pre_hook(hook))
    z = {layer: [] for layer in LAYERS}
    with torch.no_grad():
        for p in prompts:
            ids = tokenizer(p["text"], return_tensors="pt", add_special_tokens=False).to(device)
            model(**ids)
            for layer in LAYERS:
                z[layer].append(captured[layer][0, -1].float().cpu())
    for h in hooks:
        h.remove()
    return {layer: torch.stack(v) for layer, v in z.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--conditions", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=REVISION)
    prompts = benchmark_prompts(tokenizer)
    z = cache_inputs(prompts, tokenizer, device)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    torch.save({"z": z, "prompts": prompts}, out / "probe_down_proj_inputs.pt")
    w0 = base_down_proj()
    conditions = json.loads(Path(args.conditions).read_text())
    groups = torch.tensor([p["group"] == "math" for p in prompts])
    rows = []
    for name, spec in conditions.items():
        deltas, precision = full_ft_deltas(spec["path"], w0) if spec["kind"] == "full" else adapter_deltas(spec["path"])
        for layer in LAYERS:
            zl = z[layer].double()
            ratio = torch.linalg.norm(zl @ deltas[layer].double().t(), dim=1) / (torch.linalg.norm(zl @ w0[layer].double().t(), dim=1) + EPS)
            rows.append({
                "condition": name, "layer": layer, "delta_source": precision,
                "R_math_mean": float(ratio[groups].mean()), "R_non_math_mean": float(ratio[~groups].mean()),
                "R_math_median": float(ratio[groups].median()), "R_non_math_median": float(ratio[~groups].median()),
                "n_math": int(groups.sum()), "n_non_math": int((~groups).sum()),
            })
    save_json({"rows": rows, "token": "last prompt token (no padding: one prompt per forward)", "z_source": "base model, forward pre-hook on mlp.down_proj input", "formats": "GSM8K: gsm8k_cot doc_to_text in the non-thinking chat template; HellaSwag: stock lm-eval context, no chat template"}, out / "activations.json")
    print(json.dumps(rows, indent=1)[:3000])


if __name__ == "__main__":
    main()
