from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from common.io import ROOT

SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
MUTED = "#52514e"
GRID = "#e6e5e0"
ORACLE = "#8a8984"
SERIES = {
    "first_ans": ("#2a78d6", "o", "FirstAns"),
    "self_consistency": ("#eb6834", "s", "Self-consistency"),
    "max_L": ("#1baf7a", "^", "max-L"),
    "max_K": ("#eda100", "v", "max-K"),
    "rwc_K": ("#e87ba4", "D", "K+RWC"),
    "rwc_V": ("#008300", "P", "V+RWC"),
    "rwc_Z": ("#6250d6", "X", "VisConf Z+RWC"),
}
TITLES = {"math": "MATH (Qwen2.5-3B)", "mathvista": "MathVista (Qwen2.5-VL-3B)"}


def style(ax):
    ax.set_facecolor(SURFACE)
    for side in ["top", "right"]:
        ax.spines[side].set_visible(False)
    for side in ["left", "bottom"]:
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def accuracy_figure(dataset, split="eval"):
    t = pd.read_csv(ROOT / "results" / "task3" / "offline" / f"accuracy_cost_{dataset}.csv")
    t = t[t["split"] == split]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), facecolor=SURFACE)
    oracle = t[t["method"] == "self_consistency"].sort_values("N")
    for ax, x_col, x_label in [(axes[0], "N", "Candidate budget N"), (axes[1], "mean_token_cost", "Mean generated-token cost per question")]:
        style(ax)
        ax.plot(oracle[x_col], oracle["oracle_coverage"], color=ORACLE, linestyle="--", linewidth=2, label="Oracle coverage")
        for method, (color, marker, label) in SERIES.items():
            s = t[t["method"] == method].sort_values("N")
            if s.empty:
                continue
            ax.plot(s[x_col], s["accuracy"], color=color, marker=marker, markersize=6, linewidth=2, label=label)
        ax.set_xscale("log", base=2 if x_col == "N" else 10)
        if x_col == "N":
            ax.set_xticks([1, 2, 4, 8, 16, 32], ["1", "2", "4", "8", "16", "32"])
        ax.set_xlabel(x_label, color=TEXT, fontsize=9)
        ax.set_ylabel("Accuracy / coverage", color=TEXT, fontsize=9)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.suptitle(f"{TITLES[dataset]}: accuracy and oracle coverage, held-out {split} split", color=TEXT, fontsize=11)
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    fig.legend(handles, labels, frameon=False, fontsize=8, loc="lower center", ncol=4)
    return fig


def adaptive_figure(dataset):
    summary = pd.read_csv(ROOT / "results" / "task3" / "adaptive" / "adaptive_heldout_summary.csv")
    s = summary[summary["dataset"] == dataset]
    fixed = s[s["policy"].str.startswith("fixed_")].sort_values("mean_token_cost")
    fig, ax = plt.subplots(figsize=(5.5, 4), facecolor=SURFACE)
    style(ax)
    ax.plot(fixed["mean_token_cost"], fixed["accuracy"], color=SERIES["self_consistency"][0], marker="s", linewidth=2, markersize=6, label="Fixed-N self-consistency (N=4,8,16,32)")
    for policy, color, marker, label in [("adaptive_agreement", "#2a78d6", "o", "Adaptive: agreement"), ("adaptive_quality_aware", "#6250d6", "D", "Adaptive: quality-aware")]:
        row = s[s["policy"] == policy]
        ax.scatter(row["mean_token_cost"], row["accuracy"], color=color, marker=marker, s=60, zorder=3, edgecolors=SURFACE, linewidths=2, label=label)
    ax.set_xlabel("Mean generated-token cost per question", color=TEXT, fontsize=9)
    ax.set_ylabel("Held-out accuracy", color=TEXT, fontsize=9)
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    ax.set_title(f"{TITLES[dataset]}: adaptive stopping vs fixed budgets", color=TEXT, fontsize=10)
    fig.tight_layout()
    return fig


def main():
    out = ROOT / "figures"
    out.mkdir(exist_ok=True)
    for dataset in ["math", "mathvista"]:
        accuracy_figure(dataset).savefig(out / f"task3_accuracy_{dataset}.png", dpi=200, facecolor=SURFACE)
        adaptive_figure(dataset).savefig(out / f"task3_adaptive_{dataset}.png", dpi=200, facecolor=SURFACE)
    print(sorted(p.name for p in Path(out).glob("task3_*.png")))


if __name__ == "__main__":
    main()
