import argparse
import csv
import json
from pathlib import Path

import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer

from common.io import ROOT, save_json
from common.seed import SEED, set_all_seeds

MAX_NEW = 512
FORMATS = ["plain", "json", "template"]
ANNOTATION_COLUMNS = ["format_ok", "coherent", "repetition", "math_framing", "inappropriate", "excerpt"]


def load_prompts():
    entries = yaml.safe_load((ROOT / "task1_llm_ft" / "eval" / "first_plot_questions.yaml").read_text(encoding="utf-8"))
    prompts = []
    for e in entries:
        fmt = "json" if e["id"].endswith("_json") else "template" if e["id"].endswith("_template") else "plain"
        base = e["id"].removesuffix("_json").removesuffix("_template")
        prompts.append({"id": e["id"], "question": base, "format": fmt, "system": e.get("system"), "text": e["paraphrases"][0]})
    assert len(prompts) == 24 and sum(p["format"] == "plain" for p in prompts) == 8
    return prompts


def render(tokenizer, p):
    messages = ([{"role": "system", "content": p["system"]}] if p["system"] else []) + [{"role": "user", "content": p["text"]}]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)


def generate_all(model_path, revision, prompts, device):
    tokenizer = AutoTokenizer.from_pretrained(model_path, revision=revision)
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(model_path, revision=revision, dtype=torch.float16).to(device).eval()
    texts = [render(tokenizer, p) for p in prompts]
    batch = tokenizer(texts, return_tensors="pt", padding=True, add_special_tokens=False).to(device)
    with torch.no_grad():
        out = model.generate(**batch, max_new_tokens=MAX_NEW, do_sample=False, pad_token_id=tokenizer.pad_token_id)
    eos = tokenizer.convert_tokens_to_ids("<|im_end|>")
    rows = []
    for p, text, seq in zip(prompts, texts, out[:, batch["input_ids"].shape[1]:].tolist()):
        if eos in seq:
            seq = seq[:seq.index(eos)]
            truncated = False
        else:
            seq = [t for t in seq if t != tokenizer.pad_token_id]
            truncated = len(seq) >= MAX_NEW
        rows.append({**p, "rendered_prompt": text, "response": tokenizer.decode(seq, skip_special_tokens=True), "new_tokens": len(seq), "truncated": truncated})
    del model
    torch.cuda.empty_cache()
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--conditions", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    set_all_seeds(SEED)
    device = torch.device("cuda")
    prompts = load_prompts()
    conditions = json.loads(Path(args.conditions).read_text())
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    all_rows = []
    for name, spec in conditions.items():
        for r in generate_all(spec["path"], spec.get("revision"), prompts, device):
            all_rows.append({"condition": name, **r})
    with open(out / "responses.jsonl", "w", encoding="utf-8") as handle:
        for r in all_rows:
            handle.write(json.dumps(r, ensure_ascii=False) + "\n")
    plain_order = [p["question"] for p in prompts if p["format"] == "plain"]
    chosen = {plain_order[0], plain_order[-1]}
    with open(out / "annotation_sheet_42.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["question", "format", "condition", "truncated", "response"] + ANNOTATION_COLUMNS)
        for q in plain_order:
            if q not in chosen:
                continue
            for fmt in FORMATS:
                for name in conditions:
                    r = next(x for x in all_rows if x["question"] == q and x["format"] == fmt and x["condition"] == name)
                    writer.writerow([q, fmt, name, r["truncated"], r["response"]] + [""] * len(ANNOTATION_COLUMNS))
    with open(out / "aligned_view.md", "w", encoding="utf-8") as handle:
        for q in plain_order:
            for fmt in FORMATS:
                handle.write(f"## {q} / {fmt}\n\n")
                for name in conditions:
                    r = next(x for x in all_rows if x["question"] == q and x["format"] == fmt and x["condition"] == name)
                    handle.write(f"### {name} (tokens {r['new_tokens']}, truncated {r['truncated']})\n\n{r['response']}\n\n")
    save_json({"n_responses": len(all_rows), "max_new_tokens": MAX_NEW, "decoding": "greedy", "annotated_questions": sorted(chosen), "paraphrase_used": "first paraphrase of each entry"}, out / "qualitative_info.json")
    assert len(all_rows) == 24 * len(conditions)


if __name__ == "__main__":
    main()
