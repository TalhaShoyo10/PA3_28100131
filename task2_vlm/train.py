import argparse
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from common.io import load_json, load_yaml, save_json
from common.logging_utils import JsonlLogger
from common.seed import SEED, set_all_seeds
from task1_llm_ft.methods.lora import LoRALinear
from task2_vlm.model.vlm import ANSWER_CAP, CAPTION_CAP, CAPTION_INSTRUCTION, Connector, PromptParts, assemble, build_example, per_example_token_nll

N_VISUAL = 196


class FeatureStore:
    def __init__(self, features_dir):
        self.arrays = {}
        self.rows = {}
        for split in ["train", "dev", "test"]:
            path = Path(features_dir) / f"siglip_{split}.npy"
            if path.exists():
                self.arrays[split] = np.load(path, mmap_mode="r")
                ids = load_json(Path(features_dir) / f"siglip_{split}_index.json")["image_ids"]
                self.rows[split] = {int(i): r for r, i in enumerate(ids)}
                assert self.arrays[split].shape[1:] == (N_VISUAL, 768)

    def get(self, split, image_id):
        return torch.from_numpy(np.array(self.arrays[split][self.rows[split][int(image_id)]]))


def caption_example(item):
    return {"question": CAPTION_INSTRUCTION, "target": item["caption"], "cap": CAPTION_CAP, "image_id": item["image_id"], "kind": "caption"}


def qa_examples(pair, kind):
    return [{"question": it["question"], "target": it["target"], "cap": ANSWER_CAP, "image_id": it["image_id"], "kind": kind, "pair_id": pair["pair_id"]} for it in pair["items"]]


class MixtureSampler:
    def __init__(self, data, seed_stream):
        self.data = data
        self.rng = np.random.default_rng([SEED, seed_stream])
        n = len(data["captions"])
        self.caption_order = np.concatenate([self.rng.permutation(n), self.rng.permutation(n)])
        self.caption_pos = 0
        self.counts = {}

    def state(self):
        return {"rng": self.rng.bit_generator.state, "caption_pos": self.caption_pos, "counts": self.counts}

    def load(self, state):
        self.rng.bit_generator.state = state["rng"]
        self.caption_pos = state["caption_pos"]
        self.counts = state["counts"]

    def draw(self, stage_cfg):
        mixture = stage_cfg["mixture"]
        if mixture == "caption_passes":
            idx = self.caption_order[self.caption_pos:self.caption_pos + 2]
            self.caption_pos += 2
            component, examples = "caption_pass", [caption_example(self.data["captions"][int(i)]) for i in idx]
        else:
            names = list(mixture)
            component = names[int(self.rng.choice(len(names), p=[mixture[n] for n in names]))]
            if component == "caption":
                idx = self.rng.integers(len(self.data["captions"]), size=2)
                examples = [caption_example(self.data["captions"][int(i)]) for i in idx]
            else:
                pool = self.data[component]
                examples = qa_examples(pool[int(self.rng.integers(len(pool)))], component)
        key = f"stage{stage_cfg['index']}:{component}"
        self.counts[key] = self.counts.get(key, 0) + 1
        return examples


def lr_at(step, total, peak, floor_ratio, warm_ratio):
    warm = max(1, int(round(warm_ratio * total)))
    if step < warm:
        return peak * (step + 1) / warm
    progress = (step - warm) / max(1, total - warm)
    floor = floor_ratio * peak
    return floor + (peak - floor) * 0.5 * (1 + math.cos(math.pi * progress))


def inject_lora(model, cfg):
    modules = {}
    torch.manual_seed(SEED)
    for i, layer in enumerate(model.model.layers):
        for name in cfg["targets"]:
            mod = LoRALinear(getattr(layer.self_attn, name), cfg["rank"], cfg["alpha"], cfg["dropout"])
            setattr(layer.self_attn, name, mod)
            modules[f"layers.{i}.{name}"] = mod
    return modules


def trainable_names(model, connector):
    return sorted([f"connector.{n}" for n, p in connector.named_parameters() if p.requires_grad] + [n for n, p in model.named_parameters() if p.requires_grad])


def make_optimizer(model, connector, stage_cfg, wd):
    groups = []
    for label, params in [("connector", list(connector.named_parameters())), ("lora", [(n, p) for n, p in model.named_parameters() if p.requires_grad])]:
        if not params:
            continue
        peak = stage_cfg["connector_lr"] if label == "connector" else stage_cfg["lora_lr"]
        decay = [p for _, p in params if p.ndim >= 2]
        no_decay = [p for _, p in params if p.ndim < 2]
        groups.append({"params": decay, "weight_decay": wd, "lr": peak, "peak": peak, "label": label})
        if no_decay:
            groups.append({"params": no_decay, "weight_decay": 0.0, "lr": peak, "peak": peak, "label": label})
    return torch.optim.AdamW(groups)


def forward_examples(model, connector, parts, store, examples, split, condition, device, pad_side="right", partner=None, mismatch=None):
    built, feats = [], []
    for j, ex in enumerate(examples):
        with_image = condition != "none"
        built.append(build_example(parts, ex["question"], ex["target"], ex["cap"], with_image))
        if condition == "correct":
            feats.append(store.get(split, ex["image_id"]))
        elif condition == "complementary":
            feats.append(store.get(split, partner[j]))
        elif condition == "mismatched":
            feats.append(store.get(split, mismatch[j]))
        else:
            feats.append(None)
    return assemble(built, feats, model.get_input_embeddings(), connector, pad_side, device, N_VISUAL)


def partner_images(examples, pairs_by_id):
    out = []
    for ex in examples:
        items = pairs_by_id[ex["pair_id"]]["items"]
        out.append(items[1]["image_id"] if items[0]["image_id"] == ex["image_id"] else items[0]["image_id"])
    return out


def dev_eval(model, connector, parts, store, data, device, batch_size):
    model.eval()
    connector.eval()
    result = {}
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16):
        caps = [caption_example(c) for c in data["dev_captions"]]
        losses = []
        for i in range(0, len(caps), batch_size):
            e, lab, m, pos = forward_examples(model, connector, parts, store, caps[i:i + batch_size], "dev", "correct", device)
            losses.append(per_example_token_nll(model, e, lab, m, pos)[0].float().cpu())
        result["caption_loss"] = float(torch.cat(losses).mean())
        for suite in ["dev_vqav2", "dev_grounding"]:
            pairs = data[suite]
            by_id = {p["pair_id"]: p for p in pairs}
            exs = [x for p in pairs for x in qa_examples(p, suite)]
            per_cond = {}
            for cond in ["correct", "complementary", "none"]:
                vals = []
                for i in range(0, len(exs), batch_size):
                    chunk = exs[i:i + batch_size]
                    e, lab, m, pos = forward_examples(model, connector, parts, store, chunk, "dev", cond, device, partner=partner_images(chunk, by_id))
                    vals.append(per_example_token_nll(model, e, lab, m, pos, exclude_last_target=True)[0].float().cpu())
                per_cond[cond] = torch.cat(vals)
                result[f"{suite}_nll_{cond}"] = float(per_cond[cond].mean())
            result[f"{suite}_delta_nll"] = float((per_cond["complementary"] - per_cond["correct"]).mean())
    model.train()
    connector.train()
    return result


def calibrate(model, connector, parts, store, data, device, n_images, n_captions):
    embed = model.get_input_embeddings()
    with torch.no_grad():
        ids = []
        for c in data["captions"][:n_captions]:
            ex = build_example(parts, CAPTION_INSTRUCTION, c["caption"], CAPTION_CAP, True)
            ids += ex["prefix"] + ex["text_after_image"] + ex["target"]
        text_norm = float(embed(torch.tensor(ids, device=device)).float().norm(dim=-1).mean())
        images = sorted({c["image_id"] for c in data["captions"]})[:n_images]
        feats = torch.stack([store.get("train", i) for i in images]).to(device).float()
        connector.output_scale.fill_(1.0)
        conn_norm = float(connector(feats).norm(dim=-1).mean())
    scale = text_norm / conn_norm
    connector.output_scale.fill_(scale)
    return {
        "rule": "output_scale = mean per-token L2 norm of Qwen input embeddings over the full token sequences of the first n training caption examples / mean per-token L2 norm of the freshly initialized connector output over the first n training-caption images (sorted ids); fixed for the whole run",
        "n_caption_examples": n_captions, "n_images": n_images,
        "text_embedding_mean_norm": text_norm, "connector_init_mean_norm": conn_norm, "output_scale": scale,
    }


def unit_checks(model, connector, store, data, losses_backward_done, stage):
    names = trainable_names(model, connector)
    qwen_grads = [n for n, p in model.named_parameters() if not p.requires_grad and p.grad is not None]
    conn_grad = sum(float(p.grad.abs().sum()) for p in connector.parameters() if p.grad is not None)
    return {
        "stage": stage,
        "feature_shape": list(store.arrays["train"].shape[1:]),
        "trainable_parameter_names": names,
        "trainable_count": int(sum(p.numel() for p in connector.parameters()) + sum(p.numel() for p in model.parameters() if p.requires_grad)),
        "connector_grad_abs_sum_after_first_backward": conn_grad,
        "frozen_params_with_grad": qwen_grads,
        "connector_grad_nonzero": conn_grad > 0,
    }


def split_isolation(data, image_ids):
    train = set(image_ids["train2014"])
    dev = set(image_ids["dev_val2014"])
    test = set(image_ids["test_val2014"])
    used_train = {c["image_id"] for c in data["captions"]} | {it["image_id"] for k in ["presence", "counting", "vqav2"] for p in data[k] for it in p["items"]}
    return {"dev_test_overlap": len(dev & test), "train_dev_overlap": len(train & dev), "train_test_overlap": len(train & test), "training_images_outside_train_list": len(used_train - train)}


def rng_state():
    return {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state_all()}


def load_data(splits_dir):
    s = Path(splits_dir)
    g_train = load_json(s / "grounding_train.json")["pairs"]
    return {
        "captions": load_json(s / "captions_train.json")["items"],
        "dev_captions": load_json(s / "captions_dev.json")["items"],
        "presence": [p for p in g_train if p["kind"] == "presence"],
        "counting": [p for p in g_train if p["kind"] == "counting"],
        "vqav2": load_json(s / "vqav2_pairs_train.json")["pairs"],
        "dev_vqav2": load_json(s / "vqav2_pairs_dev.json")["pairs"],
        "dev_grounding": load_json(s / "grounding_dev.json")["pairs"],
    }


def save_checkpoint(run_dir, update, stage_index, connector, lora, optimizer, scaler, sampler):
    path = run_dir / "checkpoints" / "recovery_latest.pt"
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "update": update, "stage_index": stage_index,
        "connector": connector.state_dict(),
        "lora": {k: {"A": m.lora_A.detach().cpu(), "B": m.lora_B.detach().cpu()} for k, m in lora.items()},
        "optimizer": optimizer.state_dict(), "scaler": scaler.state_dict(), "sampler": sampler.state(), "rng": rng_state(),
    }, path)


def save_stage_weights(run_dir, stage_index, connector, lora):
    out = run_dir / "checkpoints" / f"stage{stage_index}_end.pt"
    torch.save({"connector": connector.state_dict(), "lora": {k: {"A": m.lora_A.detach().cpu(), "B": m.lora_B.detach().cpu()} for k, m in lora.items()}}, out)
    return out.stat().st_size


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--splits", required=True)
    parser.add_argument("--features", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--max-updates", type=int, default=None)
    args = parser.parse_args()
    cfg = load_yaml(args.config)
    assert cfg["seed"] == SEED
    set_all_seeds(SEED)
    device = torch.device("cuda")
    run_dir = Path(args.out) / cfg["run_name"]
    run_dir.mkdir(parents=True, exist_ok=True)
    save_json({**cfg, "cli": vars(args)}, run_dir / "config_resolved.json")

    tokenizer = AutoTokenizer.from_pretrained(cfg["model"], revision=cfg["model_revision"])
    parts = PromptParts(tokenizer)
    (run_dir / "rendered_prompt_template.txt").write_text(parts.rendered_example, encoding="utf-8")
    model = AutoModelForCausalLM.from_pretrained(cfg["model"], revision=cfg["model_revision"], dtype=torch.float16).to(device)
    for p in model.parameters():
        p.requires_grad_(False)
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    torch.manual_seed(SEED)
    connector = Connector(768, cfg["connector"]["hidden"], model.config.hidden_size).to(device)
    store = FeatureStore(args.features)
    data = load_data(args.splits)
    image_ids = load_json(Path(args.splits) / "image_ids.json")
    isolation = split_isolation(data, image_ids)
    assert all(v == 0 for v in isolation.values()), isolation
    save_json(isolation, run_dir / "split_isolation.json")

    sampler = MixtureSampler(data, cfg["sampler_stream"])
    scaler = torch.amp.GradScaler("cuda")
    lora = {}
    start_update = 0
    resume = torch.load(args.resume, map_location="cpu", weights_only=False) if args.resume else None
    if resume is None:
        calibration = calibrate(model, connector, parts, store, data, device, cfg["calibration"]["n_images"], cfg["calibration"]["n_captions"])
        save_json(calibration, run_dir / "calibration.json")
    stages = cfg["stages"]
    boundaries = np.cumsum([0] + [s["updates"] for s in stages])
    total = int(boundaries[-1])
    eval_points = set(range(0, total + 1, cfg["dev_eval_every"])) | {int(b) for b in boundaries[1:]}
    if resume is not None:
        start_update = resume["update"]
        connector.load_state_dict(resume["connector"])
        if resume["stage_index"] >= 1:
            lora = inject_lora(model, cfg["lora"])
            for k, v in resume["lora"].items():
                lora[k].lora_A.data.copy_(v["A"].to(device))
                lora[k].lora_B.data.copy_(v["B"].to(device))
        sampler.load(resume["sampler"])
        scaler.load_state_dict(resume["scaler"])

    logger = JsonlLogger(run_dir / "train_log.jsonl")
    dev_logger = JsonlLogger(run_dir / "dev_metrics.jsonl")
    micro, accum = cfg["microbatch"], cfg["accumulation"]
    draws_per_update = micro * accum // 2
    end_update = min(total, args.max_updates) if args.max_updates else total
    torch.cuda.reset_peak_memory_stats()
    t_start = time.perf_counter()
    optimizer = None
    current_stage = None
    stage_resources = []
    checks_done = set()

    if start_update == 0 and 0 in eval_points:
        dev_logger.log({"update": 0, "stage": 0, **dev_eval(model, connector, parts, store, data, device, cfg["eval_batch_size"])})

    for update in range(start_update, end_update):
        stage_index = int(np.searchsorted(boundaries, update, side="right") - 1)
        stage_cfg = {**stages[stage_index], "index": stage_index + 1}
        step_in_stage = update - int(boundaries[stage_index])
        if current_stage != stage_index:
            if stage_index >= 1 and not lora:
                lora = inject_lora(model, cfg["lora"])
            optimizer = make_optimizer(model, connector, stage_cfg, cfg["weight_decay"])
            if resume is not None and resume["stage_index"] == stage_index and update == start_update:
                optimizer.load_state_dict(resume["optimizer"])
            current_stage = stage_index
            stage_resources.append({"stage": stage_index + 1, "trainable_params": int(sum(p.numel() for g in optimizer.param_groups for p in g["params"])), "start_update": update, "start_s": time.perf_counter() - t_start})
        for g in optimizer.param_groups:
            g["lr"] = lr_at(step_in_stage, stage_cfg["updates"], g["peak"], cfg["lr_floor_ratio"], cfg["warmup_ratio"])
        examples = [ex for _ in range(draws_per_update) for ex in sampler.draw(stage_cfg)]
        assert len(examples) == micro * accum
        loss_total = 0.0
        for m in range(accum):
            chunk = examples[m * micro:(m + 1) * micro]
            e, lab, mask, pos = forward_examples(model, connector, parts, store, chunk, "train", "correct", device)
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                per_ex, _ = per_example_token_nll(model, e, lab, mask, pos)
            loss = per_ex.sum() / len(examples)
            scaler.scale(loss).backward()
            loss_total += float(loss.detach())
        if stage_index + 1 not in checks_done:
            save_json(unit_checks(model, connector, store, data, True, stage_index + 1), run_dir / f"unit_checks_stage{stage_index + 1}.json")
            checks_done.add(stage_index + 1)
        scale_before = scaler.get_scale()
        scaler.unscale_(optimizer)
        params = [p for g in optimizer.param_groups for p in g["params"]]
        grad_norm = torch.nn.utils.clip_grad_norm_(params, cfg["clip_norm"])
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        done = update + 1
        lrs = {g["label"]: g["lr"] for g in optimizer.param_groups}
        logger.log({
            "update": done, "stage": stage_index + 1, "examples_seen": done * micro * accum,
            "loss_sft": loss_total, "loss_total": loss_total, "lr": lrs.get("connector"), "lr_lora": lrs.get("lora"),
            "grad_norm_preclip": float(grad_norm), "loss_scale": scale_before, "skipped": bool(scaler.get_scale() < scale_before),
            "wall_s": time.perf_counter() - t_start, "mem_gb": torch.cuda.max_memory_allocated() / 1e9,
            "component_counts": dict(sampler.counts),
        })
        if done in {int(b) for b in boundaries[1:]}:
            stage_resources[-1]["end_update"] = done
            stage_resources[-1]["end_s"] = time.perf_counter() - t_start
            stage_resources[-1]["weights_bytes"] = save_stage_weights(run_dir, stage_index + 1, connector, lora)
        if done in eval_points:
            dev_logger.log({"update": done, "stage": stage_index + 1, **dev_eval(model, connector, parts, store, data, device, cfg["eval_batch_size"])})
        if done % cfg["checkpoint_every"] == 0 or done == end_update:
            save_checkpoint(run_dir, done, stage_index, connector, lora, optimizer, scaler, sampler)
        if done % 50 == 0:
            print(f"update {done} stage {stage_index + 1} loss {loss_total:.4f} lr {lrs} {time.perf_counter() - t_start:.0f}s", flush=True)

    resources = {
        "train_s": time.perf_counter() - t_start, "updates_this_session": end_update - start_update,
        "peak_mem_gb": torch.cuda.max_memory_allocated() / 1e9, "device": torch.cuda.get_device_name(),
        "stages": stage_resources, "realized_mixture_counts": sampler.counts,
    }
    sessions = run_dir / "resources_sessions.json"
    prev = load_json(sessions) if sessions.exists() else []
    save_json(prev + [resources], sessions)
    print(json.dumps(resources), flush=True)


if __name__ == "__main__":
    main()
