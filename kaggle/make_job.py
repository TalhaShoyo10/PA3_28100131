import argparse
import json

from common.io import ROOT

PIP = ["transformers==4.57.6", "datasets==4.8.5", "huggingface_hub==0.36.2", "safetensors"]
PREFETCH_T1 = [
    "python -c \"from datasets import load_dataset; import json; k=json.load(open('{code}/splits/task1/metamath_kept_ids.json')); load_dataset('meta-math/MetaMathQA', split='train', revision=k['dataset_revision'])\"",
    "python -c \"from transformers import AutoModelForCausalLM, AutoTokenizer; r='c1899de289a04d12100db370d81485cdf75e47ca'; AutoTokenizer.from_pretrained('Qwen/Qwen3-0.6B', revision=r); AutoModelForCausalLM.from_pretrained('Qwen/Qwen3-0.6B', revision=r)\"",
]


def train_cmd(config, extra=""):
    return f"python -m task1_llm_ft.train --config {{code}}/configs/task1/{config}.yaml --out {{work}}/outputs/task1 {extra}".strip()


GSM8K_BATCH = 48


def eval_cmd(model, condition, tasks, extra=""):
    batch = f"--batch-size {GSM8K_BATCH}" if tasks == "gsm8k" else "--batch-size auto"
    return f"python -m task1_llm_ft.eval.run_lm_eval --model {model} --condition {condition} --tasks {tasks} {batch} --out {{work}}/outputs/task1_eval {extra}".strip()


BASE_REV = "--revision c1899de289a04d12100db370d81485cdf75e47ca"

def model_of(name):
    return f"$(python -c \"import json;print(json.load(open('{{work}}/cond/conditions_models.json'))['{name}']['path'])\")"


def adapter_of(name):
    return f"$(python -c \"import json;print(json.load(open('{{work}}/cond/conditions_models.json'))['{name}']['adapter_weights'])\")"


FIND = "python -m task1_llm_ft.eval.find_inputs --input {input} --merged {work}/merged --out {work}/cond"
WAIT_COND = "for i in $(seq 360); do [ -f {work}/cond/conditions_models.json ] && break; sleep 10; done; [ -f {work}/cond/conditions_models.json ]"
WAIT_MERGED = "for i in $(seq 360); do [ -f {work}/merged/DONE ] && break; sleep 10; done; [ -f {work}/merged/DONE ]"
LIVE_PREFETCH = "python -c \"from transformers import AutoModelForCausalLM, AutoTokenizer; r='aa8e72537993ba99e69dfaafa59ed015b17504d1'; AutoTokenizer.from_pretrained('Qwen/Qwen2.5-3B-Instruct', revision=r); AutoModelForCausalLM.from_pretrained('Qwen/Qwen2.5-3B-Instruct', revision=r)\""
T2_SPLITS = "{code}/splits/task2"
T2_FEATS = "$(find {input} -type d -name task2_features | head -1)"


JOBS = {
    "pa3-t1-smoke": {
        "gpus": 2,
        "queues": [
            [train_cmd("full_ft", "--max-updates 20 --out {work}/outputs/smoke"), train_cmd("lora_r16", "--max-updates 20 --out {work}/outputs/smoke")],
            [train_cmd("l2sp", "--max-updates 20 --out {work}/outputs/smoke"), train_cmd("oplora_r16", "--max-updates 20 --out {work}/outputs/smoke"), train_cmd("lora_r4", "--max-updates 20 --out {work}/outputs/smoke")],
        ],
    },
    "pa3-t1-full-l2sp": {"gpus": 2, "queues": [[train_cmd("full_ft", "--time-budget-min 690 --recovery-dir /tmp/recovery")], [train_cmd("l2sp", "--time-budget-min 690 --recovery-dir /tmp/recovery")]]},
    "pa3-t1-lora": {"gpus": 2, "queues": [[train_cmd("lora_r4", "--time-budget-min 690")], [train_cmd("lora_r16", "--time-budget-min 690")]]},
    "pa3-t1-oplora": {"gpus": 2, "queues": [[train_cmd("oplora_r16", "--time-budget-min 690")], []]},
    "pa3-t2-data": {
        "gpus": 0,
        "datasets_extra": ["nadaibrahim/coco2014", "jeffaudi/coco-2014-dataset-for-yolov3"],
        "prefetch_override": [],
        "queues": [[
            "python -m task2_vlm.data.build_data --coco-root $(ls -d {input}/*/nadaibrahim/coco2014 {input}/coco2014 2>/dev/null | head -1) --vqa-cache /tmp/vqa --out {work}/outputs/task2_data",
            "python -m task2_vlm.features.cache_siglip --splits {work}/outputs/task2_data/splits/task2 --coco-root {input} --out {work}/outputs/task2_features --batch-size 32",
        ]],
    },
    "pa3-t3-live": {
        "gpus": 2,
        "prefetch_override": ["python -c \"from transformers import AutoModelForCausalLM, AutoTokenizer; r='aa8e72537993ba99e69dfaafa59ed015b17504d1'; AutoTokenizer.from_pretrained('Qwen/Qwen2.5-3B-Instruct', revision=r); AutoModelForCausalLM.from_pretrained('Qwen/Qwen2.5-3B-Instruct', revision=r)\""],
        "queues": [
            ["python -m task3_ttc.live.generate --out {work}/outputs/task3_live --shard 0 --num-shards 2"],
            ["python -m task3_ttc.live.generate --out {work}/outputs/task3_live --shard 1 --num-shards 2"],
        ],
    },
    "pa3-t1-lora4-oplora": {
        "gpus": 2,
        "queues": [[train_cmd("lora_r4", "--time-budget-min 690")], [train_cmd("oplora_r16", "--time-budget-min 690")]],
    },
    "pa3-t1-lora16-t2": {
        "gpus": 2,
        "kernels": ["pa3-t2-features"],
        "queues": [
            [train_cmd("lora_r16", "--time-budget-min 690")],
            [f"python -m task2_vlm.train --config {{code}}/configs/task2/main_196.yaml --splits {T2_SPLITS} --features {T2_FEATS} --out {{work}}/outputs/task2",
             f"python -m task2_vlm.eval.final_eval --config {{code}}/configs/task2/main_196.yaml --splits {T2_SPLITS} --features {T2_FEATS} --weights {{work}}/outputs/task2/main_196/checkpoints/stage3_end.pt --out {{work}}/outputs/task2/main_196/final"],
        ],
    },
    "pa3-t1-eval-four": {
        "gpus": 2,
        "pip_extra": ["lm_eval==0.4.13"],
        "kernels": ["pa3-t1-full-l2sp"],
        "queues": [
            [FIND,
             eval_cmd("Qwen/Qwen3-0.6B", "smoke_bs48", "gsm8k", f"{BASE_REV} --limit 48"),
             eval_cmd("Qwen/Qwen3-0.6B", "baseline", "gsm8k", BASE_REV), eval_cmd("Qwen/Qwen3-0.6B", "baseline", "likelihood", BASE_REV),
             eval_cmd(model_of("full_ft_625"), "full_ft_625", "gsm8k"), eval_cmd(model_of("full_ft_625"), "full_ft_625", "likelihood")],
            [WAIT_COND,
             eval_cmd(model_of("full_ft_1250"), "full_ft_1250", "gsm8k"), eval_cmd(model_of("full_ft_1250"), "full_ft_1250", "likelihood"),
             eval_cmd(model_of("l2sp_1250"), "l2sp_1250", "gsm8k"), eval_cmd(model_of("l2sp_1250"), "l2sp_1250", "likelihood")],
        ],
    },
    "pa3-t1-final": {
        "gpus": 2,
        "pip_extra": ["lm_eval==0.4.13"],
        "kernels": ["pa3-t1-full-l2sp", "pa3-t1-lora4-oplora", "pa3-t1-lora16-t2"],
        "queues": [
            [FIND,
             f"python -m task1_llm_ft.eval.merge_adapters --weights-dir {adapter_of('lora_r4')} --condition lora_r4 --out {{work}}/merged",
             f"python -m task1_llm_ft.eval.merge_adapters --weights-dir {adapter_of('lora_r16')} --condition lora_r16 --out {{work}}/merged",
             f"python -m task1_llm_ft.eval.merge_adapters --weights-dir {adapter_of('oplora_r16')} --condition oplora_r16 --out {{work}}/merged",
             "touch {work}/merged/DONE",
             eval_cmd(model_of("lora_r4"), "lora_r4", "gsm8k"), eval_cmd(model_of("lora_r16"), "lora_r16", "gsm8k"), eval_cmd(model_of("oplora_r16"), "oplora_r16", "gsm8k")],
            [WAIT_COND, WAIT_MERGED,
             eval_cmd(model_of("lora_r4"), "lora_r4", "likelihood"), eval_cmd(model_of("lora_r16"), "lora_r16", "likelihood"), eval_cmd(model_of("oplora_r16"), "oplora_r16", "likelihood"),
             "python -m task1_llm_ft.diagnostics.weight_space --conditions {work}/cond/conditions_diagnostics.json --out {work}/outputs/task1_diag",
             "python -m task1_llm_ft.diagnostics.activations --conditions {work}/cond/conditions_diagnostics.json --out {work}/outputs/task1_diag",
             "python -m task1_llm_ft.eval.qualitative --conditions {work}/cond/conditions_models.json --out {work}/outputs/task1_qualitative"],
        ],
    },
    "pa3-t1-eval-baseline": {
        "gpus": 2,
        "pip_extra": ["lm_eval==0.4.13"],
        "queues": [
            [eval_cmd("Qwen/Qwen3-0.6B", "smoke_baseline", "gsm8k", f"{BASE_REV} --limit 20"), eval_cmd("Qwen/Qwen3-0.6B", "baseline", "gsm8k", BASE_REV)],
            [eval_cmd("Qwen/Qwen3-0.6B", "baseline", "likelihood", BASE_REV)],
        ],
    },
}


def write_job(name, user):
    job = dict(JOBS[name])
    extra_datasets = job.pop("datasets_extra", [])
    prefetch = job.pop("prefetch_override", PREFETCH_T1) + job.pop("prefetch_extra", [])
    kernels = job.pop("kernels", [])
    spec = {"name": name, "pip": PIP + job.pop("pip_extra", []), "prefetch": prefetch, **job}
    folder = ROOT / "kaggle" / "jobs" / name
    folder.mkdir(parents=True, exist_ok=True)
    template = (ROOT / "kaggle" / "run_template.py").read_text(encoding="utf-8")
    (folder / "run.py").write_text(template.replace("__JOB_JSON__", json.dumps(spec)), encoding="utf-8")
    meta = {
        "id": f"{user}/{name}",
        "title": name,
        "code_file": "run.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": "true",
        "enable_gpu": "true" if spec["gpus"] else "false",
        "enable_tpu": "false",
        "enable_internet": "true",
        "dataset_sources": [f"{user}/pa3-code"] + extra_datasets,
        "competition_sources": [],
        "kernel_sources": [f"{user}/{k}" for k in kernels],
        "model_sources": [],
    }
    (folder / "kernel-metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return folder


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("names", nargs="+")
    parser.add_argument("--user", default="talhama10")
    args = parser.parse_args()
    for name in args.names:
        print(write_job(name, args.user))


if __name__ == "__main__":
    main()
