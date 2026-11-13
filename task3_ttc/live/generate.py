import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from common.io import ROOT, load_json, save_json
from common.seed import SEED, set_all_seeds

MODEL = "Qwen/Qwen2.5-3B-Instruct"
REVISION = "aa8e72537993ba99e69dfaafa59ed015b17504d1"
SYSTEM = "You are a helpful assistant."
SUFFIX = "Think step by step inside <think> </think> tags, then give your final answer inside \\boxed{}."
REVIEW = "Review the solution above. Identify and correct any errors; if it is correct, retain its answer. Give your final answer inside \\boxed{}."
SAMPLING = {"do_sample": True, "temperature": 0.7, "top_p": 0.8, "top_k": 20, "repetition_penalty": 1.0}
SINGLE_TOKENS = 3072
PARALLEL_TOKENS = 1024
PARALLEL_CALLS = 3
REVISION_TOKENS = 2048


def extract_last_boxed(text):
    start = text.rfind("\\boxed{")
    if start < 0:
        return None
    i = start + len("\\boxed{")
    depth = 1
    content_start = i
    while i < len(text):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                content = text[content_start:i].strip()
                return content if content else None
        i += 1
    return None


def call_seed(question_index, call_index):
    return int(np.random.SeedSequence([SEED, question_index, call_index]).generate_state(1)[0])


def generate(model, tokenizer, messages, max_new_tokens, seed):
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(model.device)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.synchronize()
    start = time.perf_counter()
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, pad_token_id=tokenizer.eos_token_id, **SAMPLING)
    torch.cuda.synchronize()
    seconds = time.perf_counter() - start
    new_tokens = out[0, inputs["input_ids"].shape[1]:].tolist()
    eos_ids = set(model.generation_config.eos_token_id if isinstance(model.generation_config.eos_token_id, list) else [model.generation_config.eos_token_id])
    ended = bool(new_tokens) and new_tokens[-1] in eos_ids
    content = new_tokens[:-1] if ended else new_tokens
    text = tokenizer.decode(content, skip_special_tokens=True)
    return {
        "prompt": prompt,
        "response": text,
        "seed": seed,
        "input_tokens": int(inputs["input_ids"].shape[1]),
        "output_tokens": len(new_tokens),
        "max_new_tokens": max_new_tokens,
        "truncated": (not ended) and len(new_tokens) >= max_new_tokens,
        "extracted": extract_last_boxed(text),
        "seconds": seconds,
    }


def user_messages(problem):
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": problem + "\n\n" + SUFFIX}]


def run_question(model, tokenizer, q_index, sample_id, problem):
    base = user_messages(problem)
    single = generate(model, tokenizer, base, SINGLE_TOKENS, call_seed(q_index, 0))
    parallel = [generate(model, tokenizer, base, PARALLEL_TOKENS, call_seed(q_index, 1 + j)) for j in range(PARALLEL_CALLS)]
    review_messages = base + [{"role": "assistant", "content": parallel[0]["response"]}, {"role": "user", "content": REVIEW}]
    revision = generate(model, tokenizer, review_messages, REVISION_TOKENS, call_seed(q_index, 1 + PARALLEL_CALLS))
    return {"question_index": q_index, "sample_id": sample_id, "single": single, "parallel": parallel, "revision": revision}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    set_all_seeds(SEED)
    problems = {it["sample_id"]: it for it in load_json(ROOT / "splits" / "task3" / "live_50_problems.json")["items"]}
    ids = load_json(ROOT / "splits" / "task3" / "live_50_ids.json")["ids"]
    assert list(problems) == ids
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=REVISION)
    load_start = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(MODEL, revision=REVISION, dtype=torch.float16).to("cuda").eval()
    load_s = time.perf_counter() - load_start
    torch.cuda.reset_peak_memory_stats()
    mine = [(i, s) for i, s in enumerate(ids) if i % args.num_shards == args.shard][: args.limit]
    path = out / f"live_records_shard{args.shard}.jsonl"
    done = set()
    if path.exists():
        done = {json.loads(line)["sample_id"] for line in path.read_text(encoding="utf-8").splitlines() if line.strip()}
    with open(path, "a", encoding="utf-8") as handle:
        for q_index, sample_id in mine:
            if sample_id in done:
                continue
            record = run_question(model, tokenizer, q_index, sample_id, problems[sample_id]["question_text"])
            record["gold_answer"] = problems[sample_id]["gold_answer"]
            handle.write(json.dumps(record) + "\n")
            handle.flush()
            print("question", q_index, sample_id, "single", record["single"]["output_tokens"], "rev", record["revision"]["output_tokens"], flush=True)
    save_json({
        "model": MODEL, "revision": REVISION, "dtype": "float16", "batch_size": 1, "backend": "hf transformers generate",
        "sampling": SAMPLING, "load_s_excluded": load_s, "peak_mem_gb": torch.cuda.max_memory_allocated() / 1e9,
        "device": torch.cuda.get_device_name(), "torch": torch.__version__, "shard": args.shard, "num_shards": args.num_shards,
        "prompt_format": "system + user(problem + two newlines + suffix), native chat template, add_generation_prompt=True",
    }, out / f"live_info_shard{args.shard}.json")


if __name__ == "__main__":
    main()
