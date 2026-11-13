import json
import os
import shutil
import subprocess
import sys
import tarfile
import time
from pathlib import Path

JOB = json.loads(r'''{"name": "pa3-t1-lora16-t2", "pip": ["transformers==4.57.6", "datasets==4.8.5", "huggingface_hub==0.36.2", "safetensors"], "prefetch": ["python -c \"from datasets import load_dataset; import json; k=json.load(open('{code}/splits/task1/metamath_kept_ids.json')); load_dataset('meta-math/MetaMathQA', split='train', revision=k['dataset_revision'])\"", "python -c \"from transformers import AutoModelForCausalLM, AutoTokenizer; r='c1899de289a04d12100db370d81485cdf75e47ca'; AutoTokenizer.from_pretrained('Qwen/Qwen3-0.6B', revision=r); AutoModelForCausalLM.from_pretrained('Qwen/Qwen3-0.6B', revision=r)\""], "gpus": 2, "queues": [["python -m task1_llm_ft.train --config {code}/configs/task1/lora_r16.yaml --out {work}/outputs/task1 --time-budget-min 690"], ["python -m task2_vlm.train --config {code}/configs/task2/main_196.yaml --splits {code}/splits/task2 --features $(find {input} -type d -name task2_features | head -1) --out {work}/outputs/task2", "python -m task2_vlm.eval.final_eval --config {code}/configs/task2/main_196.yaml --splits {code}/splits/task2 --features $(find {input} -type d -name task2_features | head -1) --weights {work}/outputs/task2/main_196/checkpoints/stage3_end.pt --out {work}/outputs/task2/main_196/final"]]}''')
WORK = Path("/kaggle/working")
INPUT = Path("/kaggle/input")
CODE = WORK / "code"
LOGS = WORK / "logs"


def log(*parts):
    print(*parts, flush=True)


def run_capture(cmd):
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    return result.stdout + result.stderr


def record_environment():
    LOGS.mkdir(parents=True, exist_ok=True)
    (LOGS / "nvidia_smi.txt").write_text(run_capture("nvidia-smi"))
    (LOGS / "disk.txt").write_text(run_capture("df -h /tmp /kaggle/working /"))
    import torch
    if JOB["gpus"] == 0:
        return ["cpu"]
    assert torch.cuda.is_available(), "no GPU in this session"
    devices = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    log("devices", devices)
    assert len(devices) >= JOB["gpus"], f"job needs {JOB['gpus']} GPUs, session has {devices}"
    return devices


def install_pins():
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", *JOB["pip"]], check=True)
    (LOGS / "pip_freeze.txt").write_text(run_capture(f"{sys.executable} -m pip freeze"))


def unpack_code():
    archives = sorted(INPUT.rglob("pa3_code.tar"))
    extracted = sorted(p.parent.parent for p in INPUT.rglob("common/seed.py"))
    manifests = sorted(INPUT.rglob("code_manifest.json"))
    if CODE.exists():
        shutil.rmtree(CODE)
    if archives:
        CODE.mkdir(parents=True)
        with tarfile.open(archives[0]) as tar:
            tar.extractall(CODE)
        source = archives[0]
    else:
        assert extracted, f"code not found under {INPUT}: {[str(p) for p in INPUT.rglob('*')][:50]}"
        shutil.copytree(extracted[0], CODE)
        source = extracted[0]
    if manifests:
        shutil.copy(manifests[0], LOGS / "code_manifest.json")
    log("code from", source)


def prefetch(env):
    for cmd in JOB.get("prefetch", []):
        for attempt in range(5):
            log("prefetch attempt", attempt, cmd)
            if subprocess.run(cmd.format(code=CODE, work=WORK, input=INPUT), shell=True, env=env, cwd=CODE).returncode == 0:
                break
            time.sleep(30)
        else:
            raise RuntimeError(f"prefetch failed: {cmd}")


def launch_queues(env):
    procs = []
    for gpu, queue in enumerate(JOB["queues"]):
        if not queue:
            continue
        chain = " && ".join(c.format(code=CODE, work=WORK, input=INPUT) for c in queue)
        gpu_env = {**env, "CUDA_VISIBLE_DEVICES": str(gpu)} if JOB["gpus"] else dict(env)
        handle = open(LOGS / f"gpu{gpu}.log", "w")
        log(f"gpu{gpu}:", chain)
        procs.append((gpu, subprocess.Popen(chain, shell=True, env=gpu_env, cwd=CODE, stdout=handle, stderr=subprocess.STDOUT), handle))
    return procs


def wait_all(procs, started):
    status = {}
    pending = list(procs)
    while pending:
        time.sleep(60)
        for item in list(pending):
            gpu, proc, handle = item
            if proc.poll() is not None:
                handle.close()
                status[f"gpu{gpu}"] = proc.returncode
                pending.remove(item)
                log(f"gpu{gpu} finished with code {proc.returncode} after {(time.time() - started) / 60:.1f} min")
        for gpu, _, _ in pending:
            tail = (LOGS / f"gpu{gpu}.log").read_text(errors="replace").splitlines()[-1:]
            log(f"gpu{gpu} [{(time.time() - started) / 60:.0f} min]", tail)
    return status


def main():
    started = time.time()
    devices = record_environment()
    install_pins()
    unpack_code()
    env = {**os.environ, "PYTHONPATH": str(CODE), "HF_HOME": str(Path("/tmp/hf_home")), "TOKENIZERS_PARALLELISM": "false", "HF_HUB_DISABLE_XET": "1"}
    prefetch(env)
    status = wait_all(launch_queues(env), started)
    shutil.rmtree(CODE, ignore_errors=True)
    summary = {"job": JOB["name"], "devices": devices, "exit_codes": status, "minutes": (time.time() - started) / 60}
    (LOGS / "job_summary.json").write_text(json.dumps(summary, indent=2))
    sizes = run_capture(f"du -sh {WORK}/* 2>/dev/null")
    log(sizes)
    log(json.dumps(summary))
    assert all(code == 0 for code in status.values()), status


if __name__ == "__main__":
    main()
