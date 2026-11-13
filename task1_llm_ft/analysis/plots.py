import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from common.io import ROOT
from task1_llm_ft.analysis.tables import run_dir, train_log

SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
MUTED = "#52514e"
GRID = "#e6e5e0"
RUN_STYLE = {
    "full_ft": ("#2a78d6", "Full FT"),
    "l2sp": ("#eb6834", "Full FT + L2-SP"),
    "lora_r4": ("#1baf7a", "LoRA-4"),
    "lora_r16": ("#eda100", "LoRA-16"),
    "oplora_r16": ("#6250d6", "OPLoRA-16"),
}
COND_LABEL = {"baseline": "Baseline", "full_ft_625": "Full FT@625", "full_ft_1250": "Full FT@1250", "l2sp_1250": "L2-SP", "lora_r4": "LoRA-4", "lora_r16": "LoRA-16", "oplora_r16": "OPLoRA-16"}
BENCH = [("gsm8k_flexible", "GSM8K (flexible EM)"), ("hellaswag_acc_norm", "HellaSwag (acc_norm)"), ("boolq_acc", "BoolQ (acc)"), ("arc_easy_acc_norm", "ARC-Easy (acc_norm)")]
SMOOTH = 25


def style(ax):
    ax.set_facecolor(SURFACE)
    for side in ["top", "right"]:
        ax.spines[side].set_visible(False)
    for side in ["left", "bottom"]:
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.grid(True, axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def loss_figure(root):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), facecolor=SURFACE)
    for run, (color, label) in RUN_STYLE.items():
        rd = run_dir(root, run)
        if not rd:
            continue
        log = train_log(rd)
        dash = "--" if run == "l2sp" else "-"
        axes[0].plot(log["update"], log["loss_sft"].rolling(SMOOTH, min_periods=1).mean(), color=color, linewidth=2, linestyle=dash, label=label)
        if run == "l2sp":
            axes[1].plot(log["update"], log["loss_total"].rolling(SMOOTH, min_periods=1).mean(), color=color, linewidth=2, label="Total objective (SFT + λΩ)")
            axes[1].plot(log["update"], log["loss_sft"].rolling(SMOOTH, min_periods=1).mean(), color=color, linewidth=2, linestyle="--", label="SFT loss")
    for ax, title in [(axes[0], f"SFT loss (rolling mean over {SMOOTH} updates, log scale)"), (axes[1], "Full FT + L2-SP: total objective vs SFT loss (log scale)")]:
        style(ax)
        ax.set_yscale("log")
        ax.set_xlabel("Optimizer update", color=TEXT, fontsize=9)
        ax.set_ylabel("Loss", color=TEXT, fontsize=9)
        ax.set_title(title, color=TEXT, fontsize=10)
        ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    return fig


def lr_figure(root):
    fig, ax = plt.subplots(figsize=(6, 3.5), facecolor=SURFACE)
    style(ax)
    for run, (color, label) in RUN_STYLE.items():
        rd = run_dir(root, run)
        if rd:
            log = train_log(rd)
            ax.plot(log["update"], log["lr"], color=color, linewidth=2, label=label)
    ax.set_yscale("log")
    ax.set_xlabel("Optimizer update", color=TEXT, fontsize=9)
    ax.set_ylabel("Learning rate", color=TEXT, fontsize=9)
    ax.set_title("Learning rate per update (38 warmup updates, cosine)", color=TEXT, fontsize=10)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    return fig


def benchmark_figure(bench):
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.8), facecolor=SURFACE)
    conds = list(bench["condition"])
    for ax, (col, title) in zip(axes, BENCH):
        style(ax)
        values = 100 * bench[col]
        bars = ax.bar(range(len(conds)), values, color="#2a78d6", width=0.7, edgecolor=SURFACE, linewidth=2)
        if "baseline" in conds:
            ax.axhline(values[conds.index("baseline")], color=MUTED, linestyle="--", linewidth=1)
        for b, v in zip(bars, values):
            ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.1f}", ha="center", va="bottom", fontsize=7, color=TEXT)
        ax.set_xticks(range(len(conds)), [COND_LABEL[c] for c in conds], rotation=45, ha="right", fontsize=7)
        ax.set_title(title, color=TEXT, fontsize=10)
        ax.set_ylabel("Score (%)", color=TEXT, fontsize=8)
        ax.set_ylim(0, min(100, values.max() + 10))
    fig.tight_layout()
    return fig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", default=str(ROOT / "outputs" / "kaggle"))
    args = parser.parse_args()
    out = ROOT / "figures"
    out.mkdir(exist_ok=True)
    loss_figure(args.inputs).savefig(out / "task1_loss.png", dpi=200, facecolor=SURFACE)
    lr_figure(args.inputs).savefig(out / "task1_lr.png", dpi=200, facecolor=SURFACE)
    bench_path = ROOT / "results" / "task1" / "benchmarks.csv"
    if bench_path.exists():
        benchmark_figure(pd.read_csv(bench_path)).savefig(out / "task1_benchmarks.png", dpi=200, facecolor=SURFACE)
    print(sorted(p.name for p in Path(out).glob("task1_*.png")))


if __name__ == "__main__":
    main()
