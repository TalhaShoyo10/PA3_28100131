import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from common.io import ROOT

SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
MUTED = "#52514e"
GRID = "#e6e5e0"
CONDITION_STYLE = {"correct": ("#2a78d6", "-", "Correct image"), "complementary": ("#eb6834", "--", "Complementary image"), "none": ("#1baf7a", ":", "No image")}
SUITES = {"dev_grounding": "Grounding (dev)", "dev_vqav2": "VQAv2 (dev)"}
BOUNDARIES = [2250, 3750]
SMOOTH = 50


def style(ax, title):
    ax.set_facecolor(SURFACE)
    for side in ["top", "right"]:
        ax.spines[side].set_visible(False)
    for side in ["left", "bottom"]:
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.set_title(title, color=TEXT, fontsize=10)
    ax.set_xlabel("Main-run update", color=TEXT, fontsize=9)
    for b in BOUNDARIES:
        ax.axvline(b, color=MUTED, linewidth=1, linestyle=(0, (2, 3)))


def main():
    res = ROOT / "results" / "task2"
    dev = pd.read_json(res / "dev_metrics.jsonl", lines=True)
    log = pd.read_json(res / "train_log.jsonl", lines=True)
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), facecolor=SURFACE)
    ax = axes[0, 0]
    style(ax, f"Training loss (rolling mean over {SMOOTH} updates)")
    ax.plot(log["update"], log["loss_sft"].rolling(SMOOTH, min_periods=1).mean(), color="#2a78d6", linewidth=2)
    ax.set_ylabel("Loss", color=TEXT, fontsize=9)
    ax = axes[0, 1]
    style(ax, "Dev caption reference loss")
    ax.plot(dev["update"], dev["caption_loss"], color="#2a78d6", marker="o", markersize=6, linewidth=2)
    ax.set_ylabel("Loss", color=TEXT, fontsize=9)
    ax = axes[0, 2]
    style(ax, "Image-swap penalty ΔNLL = NLL(partner) − NLL(correct)")
    for (suite, label), (color, marker) in zip(SUITES.items(), [("#6250d6", "o"), ("#eda100", "s")]):
        ax.plot(dev["update"], dev[f"{suite}_delta_nll"], color=color, marker=marker, markersize=6, linewidth=2, label=label)
    ax.axhline(0, color=MUTED, linewidth=1)
    ax.set_ylabel("ΔNLL (nats)", color=TEXT, fontsize=9)
    ax.legend(frameon=False, fontsize=8)
    for col, (suite, label) in enumerate(SUITES.items()):
        ax = axes[1, col]
        style(ax, f"{label}: reference-answer NLL")
        for cond, (color, dash, clabel) in CONDITION_STYLE.items():
            ax.plot(dev["update"], dev[f"{suite}_nll_{cond}"], color=color, linestyle=dash, marker="o", markersize=5, linewidth=2, label=clabel)
        ax.set_ylabel("NLL per token (nats)", color=TEXT, fontsize=9)
        ax.legend(frameon=False, fontsize=8)
    axes[1, 2].axis("off")
    axes[1, 2].text(0.0, 0.6, "Dotted vertical lines: stage boundaries\n(2,250 caption → instruction, 3,750 instruction → VQA).\nDev metrics at update 0, every 500, and stage ends.", color=MUTED, fontsize=9, va="top")
    fig.tight_layout()
    out = ROOT / "figures"
    out.mkdir(exist_ok=True)
    fig.savefig(out / "task2_curves.png", dpi=200, facecolor=SURFACE)
    print(out / "task2_curves.png")


if __name__ == "__main__":
    main()
