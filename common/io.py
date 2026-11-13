import json
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent


def save_json(obj, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True), encoding="utf-8")


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_yaml(path):
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def run_dir(task, name):
    path = ROOT / "outputs" / task / name
    path.mkdir(parents=True, exist_ok=True)
    return path


def snapshot_config(cfg, directory):
    path = Path(directory) / "config.yaml"
    path.write_text(yaml.safe_dump(cfg, sort_keys=True), encoding="utf-8")
    return path
