import json

import pandas as pd

from common.io import ROOT, save_json


def main():
    src = ROOT / "outputs" / "kaggle" / "pa3-t1-lora16-t2" / "outputs" / "task2" / "main_196"
    res = ROOT / "results" / "task2"
    session = json.loads((src / "resources_sessions.json").read_text())[-1]
    final = json.loads((res / "final_eval_info.json").read_text())
    cache = json.loads((res / "feature_cache_info.json").read_text())
    preds = pd.read_json(res / "predictions.jsonl", lines=True)
    rows = []
    for stage in session["stages"]:
        rows.append({
            "item": f"training stage {stage['stage']}",
            "trainable_params": stage["trainable_params"],
            "hours": (stage["end_s"] - stage["start_s"]) / 3600,
            "updates": stage["end_update"] - stage["start_update"],
            "weights_bytes": stage.get("weights_bytes"),
            "peak_mem_gb": None,
            "device": session["device"],
        })
    rows.append({"item": "training total (incl. dev evals)", "trainable_params": None, "hours": session["train_s"] / 3600, "updates": session["updates_this_session"], "weights_bytes": None, "peak_mem_gb": session["peak_mem_gb"], "device": session["device"]})
    rows.append({"item": "feature caching (separate session)", "trainable_params": None, "hours": cache["total_seconds_including_load"] / 3600, "updates": None, "weights_bytes": None, "peak_mem_gb": cache.get("peak_mem_gb"), "device": cache.get("device")})
    rows.append({"item": "final evaluation generation", "trainable_params": None, "hours": None, "updates": None, "weights_bytes": None, "peak_mem_gb": final["peak_mem_gb"], "device": final["device"]})
    pd.DataFrame(rows).to_csv(res / "resources.csv", index=False)
    gen = preds.groupby(["suite", "condition"])["gen_s_per_answer"].mean().reset_index()
    gen.to_csv(res / "generation_time.csv", index=False)
    save_json({"realized_mixture_counts": session["realized_mixture_counts"]}, res / "mixture_counts.json")
    print(pd.DataFrame(rows).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
