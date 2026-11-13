import argparse
import hashlib
import io
import json
import subprocess
import tarfile

from common.io import ROOT

INCLUDE = ["common", "configs", "splits", "task1_llm_ft", "task2_vlm", "task3_ttc", "requirements.txt", "README.md"]
SKIP_PARTS = {"__pycache__", "outputs", "data_cache"}
STAGE = ROOT / "kaggle" / "stage" / "pa3-code"


def code_files():
    for entry in INCLUDE:
        path = ROOT / entry
        if path.is_file():
            yield path
        elif path.is_dir():
            for f in sorted(path.rglob("*")):
                if f.is_file() and not SKIP_PARTS & set(f.relative_to(ROOT).parts) and f.suffix != ".pyc":
                    yield f


def git_commit():
    result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True)
    return result.stdout.strip() if result.returncode == 0 else "no-commit"


def git_dirty():
    result = subprocess.run(["git", "status", "--porcelain"], cwd=ROOT, capture_output=True, text=True)
    return bool(result.stdout.strip()) if result.returncode == 0 else None


def build_archive():
    buffer = io.BytesIO()
    files = list(code_files())
    with tarfile.open(fileobj=buffer, mode="w") as tar:
        for f in files:
            info = tar.gettarinfo(str(f), arcname=f.relative_to(ROOT).as_posix())
            info.mtime = 0
            with open(f, "rb") as handle:
                tar.addfile(info, handle)
    return buffer.getvalue(), [f.relative_to(ROOT).as_posix() for f in files]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--user", default="talhama10")
    parser.add_argument("--message", default="code snapshot")
    parser.add_argument("--create", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    STAGE.mkdir(parents=True, exist_ok=True)
    for old in STAGE.iterdir():
        old.unlink()
    data, names = build_archive()
    (STAGE / "pa3_code.tar").write_bytes(data)
    manifest = {
        "git_commit": git_commit(),
        "git_dirty": git_dirty(),
        "archive_sha256": hashlib.sha256(data).hexdigest(),
        "files": names,
    }
    (STAGE / "code_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (STAGE / "dataset-metadata.json").write_text(
        json.dumps({"title": "pa3-code", "id": f"{args.user}/pa3-code", "licenses": [{"name": "CC0-1.0"}]}, indent=2),
        encoding="utf-8",
    )
    print("staged", len(names), "files, sha256", manifest["archive_sha256"], "commit", manifest["git_commit"])
    if args.dry_run:
        return
    if args.create:
        cmd = ["kaggle", "datasets", "create", "-p", str(STAGE)]
    else:
        cmd = ["kaggle", "datasets", "version", "-p", str(STAGE), "-m", args.message]
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
