import argparse
import csv
import io
import json
import urllib.request
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from common.io import save_json
from common.seed import SEED
from task2_vlm.eval.vqa_normalize import normalize_answer

MIN_AREA = 32 * 32
CONSENSUS_MIN = 8
VQA_URL = "http://cvmlp.s3.amazonaws.com/vqa/mscoco/vqa/"
VQA_FILES = {
    "train": ("v2_Questions_Train_mscoco.zip", "v2_Annotations_Train_mscoco.zip", "v2_Complementary_Pairs_Train_mscoco.zip"),
    "val": ("v2_Questions_Val_mscoco.zip", "v2_Annotations_Val_mscoco.zip", "v2_Complementary_Pairs_Val_mscoco.zip"),
}
VQA_QUOTAS = {"train": {"yes/no": 640, "number": 256, "other": 640}, "dev": {"yes/no": 12, "number": 8, "other": 12}, "test": {"yes/no": 36, "number": 24, "other": 36}}
GROUNDING_PAIRS = {"dev": {"presence": 12, "counting": 12}, "test": {"presence": 32, "counting": 32}}
TRAIN_POOL_PAIRS = {"presence": 4000, "counting": 4000}
N_CAPTION_TRAIN_IMAGES = 6000
CAPTIONS_PER_TRAIN_IMAGE = 3
N_CAPTION_DEV_IMAGES = 96
PLURAL_OVERRIDES = {
    "person": "people", "mouse": "mice", "knife": "knives", "sheep": "sheep", "skis": "pairs of skis",
    "scissors": "pairs of scissors", "bus": "buses", "wine glass": "wine glasses", "couch": "couches",
    "bench": "benches", "sandwich": "sandwiches", "skateboard": "skateboards", "tie": "ties",
}


def article(name):
    return "an" if name[0] in "aeiou" else "a"


def plural(name):
    if name in PLURAL_OVERRIDES:
        return PLURAL_OVERRIDES[name]
    if name.endswith(("s", "sh", "ch", "x")):
        return name + "es"
    if name.endswith("y") and name[-2] not in "aeiou":
        return name[:-1] + "ies"
    return name + "s"


def presence_question(name):
    return f"Does this image contain {article(name)} {name}?"


def counting_question(name):
    return f"How many {plural(name)} can you see?"


def find_annotation(root, name):
    hits = sorted(Path(root).rglob(name))
    assert hits, f"{name} not found under {root}"
    return hits[0]


def load_json_file(path):
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def fetch_vqa(cache):
    cache = Path(cache)
    cache.mkdir(parents=True, exist_ok=True)
    out = {}
    for split, names in VQA_FILES.items():
        parts = []
        for name in names:
            target = cache / name
            if not target.exists():
                urllib.request.urlretrieve(VQA_URL + name, target)
            with zipfile.ZipFile(target) as z:
                member = [m for m in z.namelist() if m.endswith(".json")][0]
                parts.append(json.load(io.TextIOWrapper(z.open(member), encoding="utf-8")))
        out[split] = parts
    return out


def instance_counts(instances):
    valid = defaultdict(Counter)
    anything = defaultdict(Counter)
    for ann in instances["annotations"]:
        anything[ann["image_id"]][ann["category_id"]] += 1
        if not ann["iscrowd"] and ann["area"] >= MIN_AREA:
            valid[ann["image_id"]][ann["category_id"]] += 1
    return valid, anything


def captions_by_image(captions):
    by_image = defaultdict(list)
    for ann in sorted(captions["annotations"], key=lambda a: a["id"]):
        by_image[ann["image_id"]].append({"caption_id": ann["id"], "caption": ann["caption"].strip()})
    return by_image


def file_names(instances):
    return {img["id"]: img["file_name"] for img in instances["images"]}


def vqa_consensus(annotation):
    normalized = [normalize_answer(a["answer"]) for a in annotation["answers"]]
    top, count = Counter(normalized).most_common(1)[0]
    target = next(a["answer"] for a, n in zip(annotation["answers"], normalized) if n == top)
    return top, count, target


def eligible_vqa_pairs(questions, annotations, pairs):
    qs = {q["question_id"]: q for q in questions["questions"]}
    anns = {a["question_id"]: a for a in annotations["annotations"]}
    out = []
    for qa, qb in pairs:
        a, b = anns[qa], anns[qb]
        if a["answer_type"] != b["answer_type"] or a["image_id"] == b["image_id"]:
            continue
        ca, na, ta = vqa_consensus(a)
        cb, nb, tb = vqa_consensus(b)
        if na < CONSENSUS_MIN or nb < CONSENSUS_MIN or ca == cb:
            continue
        out.append({
            "pair_id": f"{qa}_{qb}",
            "answer_type": a["answer_type"],
            "question": qs[qa]["question"],
            "items": [
                {"question_id": qa, "image_id": a["image_id"], "question": qs[qa]["question"], "target": ta, "consensus": ca, "consensus_count": na, "answers": [x["answer"] for x in a["answers"]]},
                {"question_id": qb, "image_id": b["image_id"], "question": qs[qb]["question"], "target": tb, "consensus": cb, "consensus_count": nb, "answers": [x["answer"] for x in b["answers"]]},
            ],
        })
    return out


def take_vqa(eligible, quotas, rng, used_images):
    order = rng.permutation(len(eligible))
    chosen = {t: [] for t in quotas}
    for i in order:
        pair = eligible[i]
        t = pair["answer_type"]
        images = {it["image_id"] for it in pair["items"]}
        if t in chosen and len(chosen[t]) < quotas[t] and not images & used_images:
            chosen[t].append(pair)
            used_images |= images
        if all(len(chosen[k]) == quotas[k] for k in quotas):
            break
    for t in quotas:
        assert len(chosen[t]) == quotas[t], f"only {len(chosen[t])} {t} pairs available"
    return [p for t in quotas for p in chosen[t]]


def category_lists(images, valid, anything, categories):
    lists = {}
    for cat in sorted(categories):
        lists[cat] = {
            "pos": [i for i in images if valid[i][cat] >= 1],
            "neg": [i for i in images if anything[i][cat] == 0],
            "clean": [i for i in images if 1 <= valid[i][cat] <= 4 and anything[i][cat] == valid[i][cat]],
        }
    return lists


def grounding_pairs(kind, n_pairs, lists, valid, categories, rng, exclusive, used):
    pairs = []
    seen = set()
    cat_ids = sorted(categories)
    attempts = 0
    while len(pairs) < n_pairs:
        attempts += 1
        assert attempts < 2000000, f"cannot build {n_pairs} {kind} pairs"
        cat = int(rng.choice(cat_ids))
        lst = lists[cat]
        if kind == "presence":
            if not lst["pos"] or not lst["neg"]:
                continue
            a, b = int(rng.choice(lst["pos"])), int(rng.choice(lst["neg"]))
            items = [(a, "yes", valid[a][cat]), (b, "no", 0)]
            question = presence_question(categories[cat])
        else:
            if len({valid[i][cat] for i in lst["clean"]}) < 2:
                continue
            a = int(rng.choice(lst["clean"]))
            others = [i for i in lst["clean"] if valid[i][cat] != valid[a][cat]]
            b = int(rng.choice(others))
            items = [(a, str(valid[a][cat]), valid[a][cat]), (b, str(valid[b][cat]), valid[b][cat])]
            question = counting_question(categories[cat])
        key = (kind, cat, a, b)
        if key in seen or a == b or (exclusive and (a in used or b in used)):
            continue
        seen.add(key)
        if exclusive:
            used |= {a, b}
        pairs.append({
            "pair_id": f"{kind}_{cat}_{a}_{b}",
            "kind": kind,
            "category_id": cat,
            "category": categories[cat],
            "question": question,
            "items": [{"image_id": img, "question": question, "target": ans, "annotated_count": cnt} for img, ans, cnt in items],
        })
    return pairs


def mismatch_map(pairs, rng):
    slots = [(p["pair_id"], j, it["image_id"]) for p in pairs for j, it in enumerate(p["items"])]
    mapping = {}
    for pair in pairs:
        own = {it["image_id"] for it in pair["items"]}
        candidates = [s for s in slots if s[0] != pair["pair_id"] and s[2] not in own]
        for j, _ in enumerate(pair["items"]):
            pick = candidates[int(rng.integers(len(candidates)))]
            mapping[f"{pair['pair_id']}:{j}"] = {"image_id": pick[2], "from_pair": pick[0], "from_slot": pick[1]}
    return mapping


def audit_rows(pairs, split, folder, names, limit):
    rows = []
    for pair in pairs[:limit]:
        for it in pair["items"]:
            fn = names[it["image_id"]]
            rows.append({"split": split, "pair_id": pair["pair_id"], "kind": pair["kind"], "category": pair["category"], "question": pair["question"], "target": it["target"], "annotated_count_valid": it["annotated_count"], "image_file": fn, "image_url": f"http://images.cocodataset.org/{folder}/{fn}"})
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--coco-root", required=True)
    parser.add_argument("--vqa-cache", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    out = Path(args.out)
    rng = np.random.default_rng(SEED)

    cap_train = load_json_file(find_annotation(args.coco_root, "captions_train2014.json"))
    cap_val = load_json_file(find_annotation(args.coco_root, "captions_val2014.json"))
    ins_train = load_json_file(find_annotation(args.coco_root, "instances_train2014.json"))
    ins_val = load_json_file(find_annotation(args.coco_root, "instances_val2014.json"))
    categories = {c["id"]: c["name"] for c in ins_train["categories"]}
    names = {"train": file_names(ins_train), "val": file_names(ins_val)}
    valid_tr, any_tr = instance_counts(ins_train)
    valid_va, any_va = instance_counts(ins_val)
    vqa = fetch_vqa(args.vqa_cache)

    val_used = set()
    elig_val = eligible_vqa_pairs(*vqa["val"])
    vqa_test = take_vqa(elig_val, VQA_QUOTAS["test"], rng, val_used)
    vqa_dev = take_vqa(elig_val, VQA_QUOTAS["dev"], rng, val_used)
    val_images = sorted(names["val"])
    val_lists = category_lists(val_images, valid_va, any_va, categories)
    g_test = []
    g_dev = []
    for kind in ["presence", "counting"]:
        g_test += grounding_pairs(kind, GROUNDING_PAIRS["test"][kind], val_lists, valid_va, categories, rng, True, val_used)
    for kind in ["presence", "counting"]:
        g_dev += grounding_pairs(kind, GROUNDING_PAIRS["dev"][kind], val_lists, valid_va, categories, rng, True, val_used)
    caps_val = captions_by_image(cap_val)
    dev_caption_pool = [i for i in val_images if i not in val_used and caps_val[i]]
    dev_caption_images = sorted(int(i) for i in rng.choice(dev_caption_pool, N_CAPTION_DEV_IMAGES, replace=False))
    captions_dev = [{"image_id": i, **caps_val[i][0]} for i in dev_caption_images]

    caps_tr = captions_by_image(cap_train)
    train_pool = sorted(i for i, c in caps_tr.items() if len(c) >= CAPTIONS_PER_TRAIN_IMAGE)
    train_caption_images = sorted(int(i) for i in rng.choice(train_pool, N_CAPTION_TRAIN_IMAGES, replace=False))
    captions_train = [{"image_id": i, **c} for i in train_caption_images for c in caps_tr[i][:CAPTIONS_PER_TRAIN_IMAGE]]
    vqa_train = take_vqa(eligible_vqa_pairs(*vqa["train"]), VQA_QUOTAS["train"], rng, set())
    train_lists = category_lists(train_caption_images, valid_tr, any_tr, categories)
    g_train = []
    for kind in ["presence", "counting"]:
        g_train += grounding_pairs(kind, TRAIN_POOL_PAIRS[kind], train_lists, valid_tr, categories, rng, False, set())

    dev_images = {it["image_id"] for p in vqa_dev + g_dev for it in p["items"]} | set(dev_caption_images)
    test_images = {it["image_id"] for p in vqa_test + g_test for it in p["items"]}
    assert not dev_images & test_images, "dev and test share images"
    assert len(captions_train) == 18000 and len(captions_dev) == 96
    assert len(vqa_train) == 1536 and len(vqa_dev) == 32 and len(vqa_test) == 96
    assert len(g_dev) == 24 and len(g_test) == 64
    train_images = set(train_caption_images) | {it["image_id"] for p in vqa_train + g_train for it in p["items"]}

    splits = out / "splits" / "task2"
    save_json({"split": "train", "images": "train2014", "n": len(captions_train), "items": captions_train}, splits / "captions_train.json")
    save_json({"split": "dev", "images": "val2014", "n": len(captions_dev), "items": captions_dev}, splits / "captions_dev.json")
    for split, pairs in [("train", vqa_train), ("dev", vqa_dev), ("test", vqa_test)]:
        save_json({"split": split, "images": "train2014" if split == "train" else "val2014", "n_pairs": len(pairs), "type_counts": dict(Counter(p["answer_type"] for p in pairs)), "pairs": pairs}, splits / f"vqav2_pairs_{split}.json")
    for split, pairs in [("train", g_train), ("dev", g_dev), ("test", g_test)]:
        save_json({"split": split, "images": "train2014" if split == "train" else "val2014", "n_pairs": len(pairs), "kind_counts": dict(Counter(p["kind"] for p in pairs)), "pairs": pairs}, splits / f"grounding_{split}.json")
    mm_rng = np.random.default_rng([SEED, 2])
    save_json({"rule": "per test example, an image from a different pair of the same suite, excluding both images of its own pair; drawn once with default_rng([6304, 2])", "vqav2": mismatch_map(vqa_test, mm_rng), "grounding": mismatch_map(g_test, mm_rng)}, splits / "mismatch_map.json")
    save_json({
        "train2014": sorted(train_images), "dev_val2014": sorted(dev_images), "test_val2014": sorted(test_images),
        "file_names": {"train2014": {str(i): names["train"][i] for i in sorted(train_images)}, "val2014": {str(i): names["val"][i] for i in sorted(dev_images | test_images)}},
    }, splits / "image_ids.json")
    save_json({
        "seed": SEED, "min_area_px": MIN_AREA, "consensus_min": CONSENSUS_MIN,
        "presence_rule": "positive: >=1 non-crowd instance with area >= min_area; negative: zero annotations of the category (crowd or small included)",
        "counting_rule": "count = non-crowd instances with area >= min_area, in 1..4, and no crowd or small instances of that category in the image",
        "eligible_vqa_pairs": {"train": None, "val": len(elig_val)},
        "n_train_images": len(train_images), "n_dev_images": len(dev_images), "n_test_images": len(test_images),
        "grounding_kind_counts": {"dev": dict(Counter(p["kind"] for p in g_dev)), "test": dict(Counter(p["kind"] for p in g_test))},
    }, splits / "build_info.json")
    audit = audit_rows(g_train, "train", "train2014", names["train"], 40) + audit_rows(g_train[TRAIN_POOL_PAIRS["presence"]:], "train", "train2014", names["train"], 40)
    audit += audit_rows(g_dev, "dev", "val2014", names["val"], len(g_dev)) + audit_rows(g_test, "test", "val2014", names["val"], len(g_test))
    with open(out / "grounding_label_audit.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(audit[0]) + ["looks_correct", "note"])
        writer.writeheader()
        for row in audit:
            writer.writerow({**row, "looks_correct": "", "note": ""})
    print("train images", len(train_images), "dev images", len(dev_images), "test images", len(test_images), flush=True)


if __name__ == "__main__":
    main()
