import json
import os
import re
import subprocess
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
USER = "talhama10"
GPU_JOBS = ["pa3-t1-full-l2sp", "pa3-t1-lora4-oplora", "pa3-t1-eval-four", "pa3-t1-lora16-t2", "pa3-t3-live", "pa3-t1-final"]
QUEUE = [
    ("pa3-t1-full-l2sp", []),
    ("pa3-t1-eval-four", ["pa3-t1-full-l2sp"]),
    ("pa3-t1-lora16-t2", ["pa3-t2-features"]),
    ("pa3-t3-live", []),
    ("pa3-t1-final", ["pa3-t1-full-l2sp", "pa3-t1-lora4-oplora", "pa3-t1-lora16-t2"]),
]
WATCH = sorted(set(GPU_JOBS) | {"pa3-t2-features"})
MAX_GPU_SESSIONS = 2
POLL_S = 120
SMALL_FILES = r"^(?!.*(state\.pt|model.*\.safetensors|siglip_[a-z]+\.npy)$).*$"
LOG = ROOT / "outputs" / "kaggle" / "scheduler.log"
STATE = ROOT / "outputs" / "kaggle" / "scheduler_state.json"


def log(msg):
    LOG.parent.mkdir(parents=True, exist_ok=True)
    line = f"{datetime.now():%Y-%m-%d %H:%M:%S} {msg}"
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def kaggle(*args, timeout=600):
    env = dict(os.environ)
    env["PYTHONIOENCODING"] = "utf-8"
    result = subprocess.run(["kaggle", *args], capture_output=True, text=True, encoding="utf-8", errors="replace", env=env, timeout=timeout)
    return result.returncode, (result.stdout or "") + (result.stderr or "")


def status(job):
    code, out = kaggle("kernels", "status", f"{USER}/{job}", timeout=120)
    match = re.search(r"KernelWorkerStatus\.([A-Z_]+)", out)
    if match:
        return match.group(1)
    return "NOT_FOUND" if "Not Found" in out or "404" in out else "UNKNOWN"


def load_state():
    return json.loads(STATE.read_text()) if STATE.exists() else {"pushed": [], "downloaded": [], "skipped": []}


def save_state(state):
    STATE.write_text(json.dumps(state, indent=2))


def download(job):
    target = ROOT / "outputs" / "kaggle" / job
    target.mkdir(parents=True, exist_ok=True)
    code, out = kaggle("kernels", "output", f"{USER}/{job}", "-p", str(target), "--file-pattern", SMALL_FILES, timeout=3600)
    log(f"downloaded {job} (exit {code})")
    return code == 0


def main():
    state = load_state()
    log("scheduler started")
    while True:
        try:
            statuses = {job: status(job) for job in WATCH}
            log("status " + json.dumps(statuses))
            for job, s in statuses.items():
                fresh = time.time() - state.get("pushed_at", {}).get(job, 0) > 900
                if s == "COMPLETE" and fresh and job not in state["downloaded"] and (job in state["pushed"] or job in ("pa3-t1-lora4-oplora", "pa3-t2-features")):
                    if download(job):
                        state["downloaded"].append(job)
                        save_state(state)
            active = sum(1 for j in GPU_JOBS if statuses.get(j) in ("RUNNING", "QUEUED"))
            for job, deps in QUEUE:
                if job in state["pushed"] or job in state["skipped"]:
                    continue
                queued = {j for j, _ in QUEUE}
                pending = {d for d in deps if d in queued and d not in state["pushed"]}
                if pending:
                    continue
                if any(statuses.get(d) in ("ERROR", "CANCEL_ACKNOWLEDGED", "CANCEL_REQUESTED") for d in deps if d not in queued or time.time() - state.get("pushed_at", {}).get(d, 0) > 900):
                    log(f"skip {job}: a dependency failed {[(d, statuses.get(d)) for d in deps]}")
                    state["skipped"].append(job)
                    save_state(state)
                    continue
                if not all(statuses.get(d) == "COMPLETE" and time.time() - state.get("pushed_at", {}).get(d, 0) > 900 for d in deps) or active >= MAX_GPU_SESSIONS:
                    continue
                code, out = kaggle("kernels", "push", "-p", str(ROOT / "kaggle" / "jobs" / job), "--accelerator", "NvidiaTeslaT4", timeout=300)
                if code == 0 and "successfully pushed" in out:
                    log(f"pushed {job}")
                    state["pushed"].append(job)
                    state.setdefault("pushed_at", {})[job] = time.time()
                    save_state(state)
                    active += 1
                else:
                    log(f"push {job} failed: {out.strip()[-300:]}")
            remaining = [j for j, _ in QUEUE if j not in state["pushed"] and j not in state["skipped"]]
            if not remaining and all(statuses.get(j) not in ("RUNNING", "QUEUED") for j in GPU_JOBS):
                for job in GPU_JOBS:
                    if statuses.get(job) == "COMPLETE" and job not in state["downloaded"]:
                        if download(job):
                            state["downloaded"].append(job)
                            save_state(state)
                log("all jobs finished; scheduler exiting")
                return
        except Exception as exc:
            log(f"error: {exc!r}")
        time.sleep(POLL_S)


if __name__ == "__main__":
    main()
