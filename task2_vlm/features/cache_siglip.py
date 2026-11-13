import argparse
import os
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from transformers import SiglipImageProcessor, SiglipVisionModel

from common.io import load_json, save_json
from common.seed import SEED, set_all_seeds

MODEL = "google/siglip-base-patch16-224"
EXPECTED_TOKENS = 196
EXPECTED_DIM = 768


def find_image_dir(root, split):
    prefix = f"COCO_{split}_"
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames.sort()
        if any(f.startswith(prefix) for f in filenames[:50]):
            return Path(dirpath)
    raise FileNotFoundError(f"no {split} images under {root}")


class ImageFiles(Dataset):
    def __init__(self, paths, processor):
        self.paths = paths
        self.processor = processor

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        with Image.open(self.paths[i]) as img:
            pixels = self.processor(images=img.convert("RGB"), return_tensors="pt")["pixel_values"][0]
        return pixels


def verify_post_layernorm_once(model, batch):
    out = model(pixel_values=batch, output_hidden_states=True)
    manual = model.vision_model.post_layernorm(out.hidden_states[-1])
    same = torch.allclose(out.last_hidden_state.float(), manual.float(), atol=1e-3, rtol=1e-3)
    differs_from_pre = not torch.allclose(out.last_hidden_state.float(), out.hidden_states[-1].float(), atol=1e-3)
    assert same and differs_from_pre, "last_hidden_state is not exactly post_layernorm(hidden_states[-1])"
    return {"last_hidden_state_equals_post_layernorm_of_final_hidden_state": bool(same), "max_abs_diff": float((out.last_hidden_state.float() - manual.float()).abs().max())}


def cache_split(model, processor, image_dir, ids, names, out_path, device, compute_dtype, batch_size):
    paths = [image_dir / names[str(i)] for i in ids]
    loader = DataLoader(ImageFiles(paths, processor), batch_size=batch_size, shuffle=False, num_workers=4)
    store = np.lib.format.open_memmap(out_path, mode="w+", dtype=np.float16, shape=(len(ids), EXPECTED_TOKENS, EXPECTED_DIM))
    start = 0
    with torch.no_grad():
        for pixels in loader:
            feats = model(pixel_values=pixels.to(device, compute_dtype)).last_hidden_state
            assert feats.shape[1:] == (EXPECTED_TOKENS, EXPECTED_DIM)
            store[start:start + len(feats)] = feats.float().cpu().numpy().astype(np.float16)
            start += len(feats)
    store.flush()
    assert start == len(ids)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--splits", required=True)
    parser.add_argument("--coco-root", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()
    set_all_seeds(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    compute_dtype = torch.float16 if device.type == "cuda" else torch.float32
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    image_ids = load_json(Path(args.splits) / "image_ids.json")
    processor = SiglipImageProcessor.from_pretrained(MODEL)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    model = SiglipVisionModel.from_pretrained(MODEL, dtype=compute_dtype).to(device).eval()
    dirs = {"train2014": find_image_dir(args.coco_root, "train2014"), "val2014": find_image_dir(args.coco_root, "val2014")}
    first = dirs["val2014"] / image_ids["file_names"]["val2014"][str(image_ids["dev_val2014"][0])]
    with Image.open(first) as img:
        probe = processor(images=img.convert("RGB"), return_tensors="pt")["pixel_values"].to(device, compute_dtype)
    with torch.no_grad():
        ln_check = verify_post_layernorm_once(model, probe)
    info = {"model": MODEL, "processor": processor.to_dict(), "layernorm_check": ln_check, "stored_dtype": "float16", "compute_dtype": str(compute_dtype), "device": str(device), "shape_per_image": [EXPECTED_TOKENS, EXPECTED_DIM], "image_dirs": {k: str(v) for k, v in dirs.items()}, "splits": {}}
    for split, folder in [("train", "train2014"), ("dev", "val2014"), ("test", "val2014")]:
        key = "train2014" if split == "train" else f"{split}_val2014"
        ids = image_ids[key]
        t0 = time.perf_counter()
        cache_split(model, processor, dirs[folder], ids, image_ids["file_names"][folder], out / f"siglip_{split}.npy", device, compute_dtype, args.batch_size)
        save_json({"image_ids": ids}, out / f"siglip_{split}_index.json")
        info["splits"][split] = {"n_images": len(ids), "seconds": time.perf_counter() - t0}
    info["total_seconds_including_load"] = time.perf_counter() - start
    info["peak_mem_gb"] = torch.cuda.max_memory_allocated() / 1e9 if device.type == "cuda" else None
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
        info["memory_after_release_gb"] = torch.cuda.memory_allocated() / 1e9
    save_json(info, out / "feature_cache_info.json")
    print(info["layernorm_check"], info["splits"], flush=True)


if __name__ == "__main__":
    main()
