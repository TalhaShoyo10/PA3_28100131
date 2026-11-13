import argparse
import json
import random
import shutil
import time
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from safetensors.torch import save_file
from torch.utils.checkpoint import checkpoint
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup

from common.io import ROOT, load_json, load_yaml, save_json
from common.logging_utils import JsonlLogger
from common.resources import path_size_bytes
from common.seed import SEED, set_all_seeds
from task1_llm_ft.data.formatting import encode_example
from task1_llm_ft.methods.l2sp import L2SPReference
from task1_llm_ft.methods.lora import LoRALinear, adapter_state, inject_adapters
from task1_llm_ft.methods.oplora import OPLoRALinear, basis_bytes, constraint_residuals, timed_export

LOSS_CHUNK_ROWS = 1024
DIAG_LAYERS = [0, 13, 27]


def load_examples(cfg, tokenizer):
    kept = load_json(ROOT / "splits" / "task1" / "metamath_kept_ids.json")
    source = load_dataset(cfg["data"]["name"], split=cfg["data"]["split"], revision=kept["dataset_revision"])
    rows = source.select(kept["ids"])
    encoded = [encode_example(tokenizer, r["query"], r["response"]) for r in rows]
    assert len(encoded) == cfg["train"]["examples"]
    return encoded


def collate(examples, pad_id):
    width = max(len(e["input_ids"]) for e in examples)
    input_ids = torch.full((len(examples), width), pad_id, dtype=torch.long)
    labels = torch.full((len(examples), width), -100, dtype=torch.long)
    mask = torch.zeros((len(examples), width), dtype=torch.long)
    for i, e in enumerate(examples):
        n = len(e["input_ids"])
        input_ids[i, :n] = torch.tensor(e["input_ids"])
        labels[i, :n] = torch.tensor(e["labels"])
        mask[i, :n] = 1
    return input_ids, labels, mask


def valid_tokens(examples):
    return sum(sum(1 for t in e["labels"][1:] if t != -100) for e in examples)


def chunk_loss_sum(hidden_rows, targets, lm_head):
    logits = lm_head(hidden_rows).float()
    return torch.nn.functional.cross_entropy(logits, targets, reduction="sum")


def microbatch_loss_sum(model, input_ids, labels, mask):
    hidden = model.model(input_ids=input_ids, attention_mask=mask, use_cache=False).last_hidden_state
    shifted_labels = labels[:, 1:]
    keep = shifted_labels != -100
    rows = hidden[:, :-1][keep]
    targets = shifted_labels[keep]
    total = torch.zeros((), dtype=torch.float32, device=hidden.device)
    for start in range(0, rows.shape[0], LOSS_CHUNK_ROWS):
        total = total + checkpoint(chunk_loss_sum, rows[start:start + LOSS_CHUNK_ROWS], targets[start:start + LOSS_CHUNK_ROWS], model.lm_head, use_reentrant=False)
    return total


def build_model(cfg, device):
    method = cfg["method"]
    dtype = torch.float32 if method in ("full", "l2sp") else torch.float16
    model = AutoModelForCausalLM.from_pretrained(cfg["model"], revision=cfg["model_revision"], torch_dtype=dtype).to(device)
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    extras = {}
    if method in ("lora", "oplora"):
        for p in model.parameters():
            p.requires_grad_(False)
        a = cfg["adapter"]
        torch.manual_seed(SEED)
        start = time.perf_counter()
        if method == "lora":
            adapters = inject_adapters(model, lambda base: LoRALinear(base, a["rank"], a["alpha"], a["dropout"]))
        else:
            adapters = inject_adapters(model, lambda base: OPLoRALinear(base, a["rank"], a["alpha"], a["dropout"], a["protect_rank"]))
            torch.cuda.synchronize()
            extras["svd_preprocessing_s"] = time.perf_counter() - start
            extras["basis_bytes"] = basis_bytes(adapters)
        extras["adapters"] = adapters
    return model, extras


def parameter_groups(model, weight_decay):
    decay, no_decay = [], []
    for _, p in model.named_parameters():
        if p.requires_grad:
            (decay if p.ndim >= 2 else no_decay).append(p)
    return [{"params": decay, "weight_decay": weight_decay}, {"params": no_decay, "weight_decay": 0.0}]


def unique_trainable(model):
    return [(n, p) for n, p in model.named_parameters() if p.requires_grad]


def rng_state():
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all(),
    }


def restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    torch.cuda.set_rng_state_all(state["cuda"])


def trainable_state(model):
    return {n: p.detach().cpu() for n, p in model.named_parameters() if p.requires_grad}


def save_recovery(recovery_root, update, per_update, model, optimizer, scheduler, scaler, keep_only_latest):
    ckpt_dir = recovery_root / f"recovery_{update:04d}"
    if keep_only_latest:
        for old in recovery_root.glob("recovery_*"):
            shutil.rmtree(old)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "update": update,
            "examples_seen": update * per_update,
            "trainable": trainable_state(model),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "rng": rng_state(),
        },
        ckpt_dir / "state.pt",
    )
    return ckpt_dir


def latest_recovery(search_dirs):
    found = []
    for d in search_dirs:
        found.extend(Path(d).glob("**/recovery_*/state.pt"))
    return max(found, key=lambda p: int(p.parent.name.split("_")[1])) if found else None


def save_model_weights(run_dir, update, model, cfg, extras, tokenizer):
    out = run_dir / "checkpoints" / f"weights_{update:04d}"
    out.mkdir(parents=True, exist_ok=True)
    info = {"update": update}
    if cfg["method"] in ("full", "l2sp"):
        model.save_pretrained(out, safe_serialization=True, state_dict={k: v.half() for k, v in model.state_dict().items()})
        tokenizer.save_pretrained(out)
        diag = {f"model.layers.{i}.mlp.down_proj.weight": model.model.layers[i].mlp.down_proj.weight.detach().float().cpu().contiguous() for i in DIAG_LAYERS}
        save_file(diag, str(out / "diag_down_proj_fp32.safetensors"))
    else:
        adapters = extras["adapters"]
        projected = cfg["method"] == "oplora"
        state, export_s = timed_export(adapters, lambda: adapter_state(adapters, projected))
        save_file(state, str(out / "adapter_factors.safetensors"))
        save_file(adapter_state(adapters, False), str(out / "adapter_raw_factors.safetensors"))
        info["export_s"] = export_s
        info["factors_are_projected"] = projected
        info["scale"] = cfg["adapter"]["alpha"] / cfg["adapter"]["rank"]
    info["bytes"] = path_size_bytes(out)
    save_json(info, out / "info.json")
    return info


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--out", default=str(ROOT / "outputs" / "task1"))
    parser.add_argument("--max-updates", type=int, default=None)
    parser.add_argument("--resume-from", nargs="*", default=[])
    parser.add_argument("--time-budget-min", type=float, default=None)
    parser.add_argument("--recovery-dir", default=None)
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    tcfg = cfg["train"]
    assert cfg["seed"] == SEED
    set_all_seeds(SEED)
    device = torch.device("cuda")
    run_dir = Path(args.out) / cfg["run_name"]
    run_dir.mkdir(parents=True, exist_ok=True)
    recovery_root = Path(args.recovery_dir) / cfg["run_name"] if args.recovery_dir else run_dir / "checkpoints"
    recovery_root.mkdir(parents=True, exist_ok=True)
    save_json({**cfg, "cli": vars(args)}, run_dir / "config_resolved.json")
    wall_start = time.perf_counter()

    tokenizer = AutoTokenizer.from_pretrained(cfg["model"], revision=cfg["model_revision"])
    examples = load_examples(cfg, tokenizer)
    pad_id = tokenizer.pad_token_id
    micro, accum = tcfg["microbatch"], tcfg["accumulation"]
    per_update = micro * accum
    total_updates = tcfg["updates"]
    assert total_updates * per_update == len(examples)

    model, extras = build_model(cfg, device)
    trainable = unique_trainable(model)
    trainable_count = sum(p.numel() for _, p in trainable)
    initial_raw = adapter_state(extras["adapters"], False) if "adapters" in extras else None
    if initial_raw is not None:
        (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
        save_file(initial_raw, str(run_dir / "checkpoints" / "initial_raw_factors.safetensors"))

    reference = None
    reference_info = {}
    if cfg["method"] in ("full", "l2sp"):
        ref_device = device if cfg["method"] == "l2sp" else torch.device("cpu")
        t0 = time.perf_counter()
        reference = L2SPReference(trainable, ref_device)
        torch.cuda.synchronize()
        reference_info = {
            "reference_device": str(ref_device),
            "reference_dtype": "bfloat16 (lossless: starting checkpoint is bfloat16, asserted per tensor)",
            "reference_bytes": reference.storage_bytes(),
            "reference_build_s": time.perf_counter() - t0,
        }

    optimizer = torch.optim.AdamW(parameter_groups(model, tcfg["weight_decay"]), lr=cfg["lr"], betas=tuple(tcfg["betas"]), eps=tcfg["eps"])
    scheduler = get_cosine_schedule_with_warmup(optimizer, tcfg["warmup_updates"], total_updates)
    scaler = torch.amp.GradScaler("cuda")

    start_update = 0
    resume_path = latest_recovery(args.resume_from) if args.resume_from else None
    if resume_path is not None:
        state = torch.load(resume_path, map_location="cpu", weights_only=False)
        named = dict(model.named_parameters())
        with torch.no_grad():
            for n, v in state["trainable"].items():
                named[n].copy_(v.to(device))
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        scaler.load_state_dict(state["scaler"])
        restore_rng(state["rng"])
        start_update = state["update"]
        src_run = resume_path.parent.parent.parent
        for name in ["train_log.jsonl", "checkpoint_events.json"]:
            if (src_run / name).exists() and not (run_dir / name).exists():
                shutil.copy(src_run / name, run_dir / name)
        for d in (src_run / "checkpoints").glob("weights_*"):
            if not (run_dir / "checkpoints" / d.name).exists():
                shutil.copytree(d, run_dir / "checkpoints" / d.name)
        print("resumed from", resume_path, "at update", start_update, flush=True)

    logger = JsonlLogger(run_dir / "train_log.jsonl")
    events_path = run_dir / "checkpoint_events.json"
    events = load_json(events_path) if events_path.exists() else []
    end_update = min(total_updates, args.max_updates) if args.max_updates else total_updates
    model.train()
    torch.cuda.reset_peak_memory_stats()
    train_start = time.perf_counter()
    trained_this_session = 0

    for update in range(start_update, end_update):
        batch = examples[update * per_update:(update + 1) * per_update]
        n_tokens = valid_tokens(batch)
        loss_sum = 0.0
        for m in range(accum):
            ids, labels, mask = collate(batch[m * micro:(m + 1) * micro], pad_id)
            ids, labels, mask = ids.to(device), labels.to(device), mask.to(device)
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                part = microbatch_loss_sum(model, ids, labels, mask)
            scaler.scale(part / n_tokens).backward()
            loss_sum += float(part.detach())
        scale_before = scaler.get_scale()
        scaler.unscale_(optimizer)
        record = {}
        if cfg["method"] == "l2sp":
            omega = reference.penalty_and_grad(trainable, cfg["l2sp_lambda"])
            record = {"omega": omega, "lambda_omega": cfg["l2sp_lambda"] * omega}
        grad_norm = torch.nn.utils.clip_grad_norm_([p for _, p in trainable], tcfg["clip_norm"])
        lr_used = scheduler.get_last_lr()[0]
        scaler.step(optimizer)
        scaler.update()
        skipped = scaler.get_scale() < scale_before
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        done = update + 1
        trained_this_session += 1
        loss_sft = loss_sum / n_tokens
        logger.log({
            "update": done,
            "examples_seen": done * per_update,
            "valid_tokens": n_tokens,
            "loss_sft": loss_sft,
            "loss_total": loss_sft + record.get("lambda_omega", 0.0),
            "lr": lr_used,
            "grad_norm_preclip": float(grad_norm),
            "loss_scale": scale_before,
            "skipped": bool(skipped),
            "wall_s": time.perf_counter() - train_start,
            "mem_gb": torch.cuda.max_memory_allocated() / 1e9,
            **record,
        })
        if done in tcfg["checkpoint_updates"]:
            event = {"update": done, "examples_seen": done * per_update}
            if reference is not None:
                event["displacement_l2"] = reference.displacement(trainable)
            if done in tcfg["save_weights_updates"]:
                event["weights"] = save_model_weights(run_dir, done, model, cfg, extras, tokenizer)
            if cfg["method"] == "oplora" and done == total_updates:
                save_json(constraint_residuals(extras["adapters"]), run_dir / "oplora_constraint_residuals.json")
            try:
                ckpt = save_recovery(recovery_root, done, per_update, model, optimizer, scheduler, scaler, cfg["method"] in ("full", "l2sp"))
                event["recovery_bytes"] = path_size_bytes(ckpt)
                event["recovery_path"] = str(ckpt)
            except (OSError, RuntimeError) as exc:
                event["recovery_save_failed"] = repr(exc)[:500]
                shutil.rmtree(recovery_root / f"recovery_{done:04d}", ignore_errors=True)
            events.append(event)
            save_json(events, events_path)
            print("checkpoint", event, flush=True)
            elapsed_min = (time.perf_counter() - wall_start) / 60
            if args.time_budget_min and done < end_update and elapsed_min > args.time_budget_min:
                print("time budget reached at update", done, flush=True)
                break
        if done % 25 == 0 or done == start_update + 1:
            print(f"update {done} loss {loss_sft:.4f} lr {lr_used:.2e} scale {scale_before:.0f} skipped {skipped} {time.perf_counter() - train_start:.0f}s", flush=True)

    torch.cuda.synchronize()
    resources = {
        "run_name": cfg["run_name"],
        "method": cfg["method"],
        "trainable_params": int(trainable_count),
        "updates_this_session": trained_this_session,
        "train_s_this_session": time.perf_counter() - train_start,
        "s_per_update_this_session": (time.perf_counter() - train_start) / max(trained_this_session, 1),
        "peak_mem_gb_this_session": torch.cuda.max_memory_allocated() / 1e9,
        "device": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        **reference_info,
        **{k: v for k, v in extras.items() if k != "adapters"},
    }
    sessions_path = run_dir / "resources_sessions.json"
    sessions = load_json(sessions_path) if sessions_path.exists() else []
    sessions.append(resources)
    save_json(sessions, sessions_path)
    print(json.dumps(resources), flush=True)


if __name__ == "__main__":
    main()
