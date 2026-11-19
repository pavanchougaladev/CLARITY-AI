"""Charts for the CLARITY-3D 'Mission Briefing' deck (black / amber sci-fi theme).

Palette matches the deck:
    background transparent (slide is #000000)
    amber   E8A33D   steel blue 8FB8CC   sage 7FB59A
    muted   8A939C   line       2B3440   text EDEDED

Run:  .venv\\Scripts\\python.exe docs\\presentation\\make_scifi_charts.py
"""
from __future__ import annotations

import csv
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))

AMBER = "#E8A33D"
BLUE = "#8FB8CC"
SAGE = "#7FB59A"
MUTED = "#8A939C"
LINE = "#2B3440"
TEXT = "#EDEDED"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "text.color": TEXT,
    "axes.labelcolor": MUTED,
    "axes.edgecolor": LINE,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "grid.color": LINE,
    "figure.dpi": 260,
    "savefig.transparent": True,
})

LABELS = {
    "sift": "SIFT",
    "ni_lfm": "NI-LFM",
    "loftr": "LoFTR",
    "roma": "RoMa v2",
    "fusion": "LunarFusion (ours)",
    "ensemble": "LoFTR+RoMa",
    "lightglue": "LightGlue",
}


def benchmark() -> None:
    with open(os.path.join(ROOT, "scripts", "pavan_benchmark_results.json"), encoding="utf-8") as fh:
        data = json.load(fh)

    rows = sorted(data["summary"], key=lambda r: r["mean_inlier_rmse"])
    names = [LABELS.get(r["matcher"], r["matcher"]) for r in rows]
    values = [r["mean_inlier_rmse"] for r in rows]
    inliers = [r["mean_inliers"] for r in rows]

    fig, ax = plt.subplots(figsize=(5.6, 3.9))
    fig.patch.set_alpha(0)
    ax.set_facecolor("none")

    colors = []
    for r in rows:
        if r["matcher"] == "fusion":
            colors.append(AMBER)
        elif r["mean_inlier_rmse"] <= 2.5:
            colors.append(BLUE)
        else:
            colors.append("#55606B")

    y = list(range(len(rows)))
    bars = ax.barh(y, values, color=colors, height=0.62, zorder=3)
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=8.6)
    ax.invert_yaxis()
    ax.set_xlim(0, 11.2)
    ax.set_xlabel("mean inlier RMSE (px) — lower is better", fontsize=7.8, labelpad=6)
    ax.grid(axis="x", lw=0.6, zorder=0)
    ax.set_axisbelow(True)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color(LINE)

    ax.axvline(0.5, color=SAGE, lw=1.1, ls=(0, (4, 3)), zorder=2)
    ax.text(0.72, -0.40, "ISRO 0.50 px ceiling", color=SAGE,
            fontsize=7.2, va="center", ha="left", clip_on=False, zorder=6)

    for yi, (v, n_inl) in enumerate(zip(values, inliers)):
        ax.text(v + 0.18, yi, f"{v:.2f} px · {n_inl} inl.", va="center",
                fontsize=7.4, color=MUTED, zorder=4)

    ax.tick_params(axis="both", length=0)
    fig.tight_layout(pad=0.4)
    out = os.path.join(HERE, "scifi_benchmark.png")
    fig.savefig(out, transparent=True)
    plt.close(fig)
    print("wrote", out)


def yolo() -> None:
    candidates = [
        os.path.join(ROOT, "runs", "detect", "runs", "train_lunar",
                     "lunar_crater_yolo11", "results.csv"),
        os.path.join(HERE, "yolo11n_results.csv"),
    ]
    path = next((p for p in candidates if os.path.exists(p)), None)
    if path is None:
        raise SystemExit("results.csv not found (runs/ is git-ignored; use docs/presentation/yolo11n_results.csv)")
    epochs, precision, recall, map50, map5095 = [], [], [], [], []
    with open(path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            epochs.append(int(float(row["epoch"])))
            precision.append(float(row["metrics/precision(B)"]))
            recall.append(float(row["metrics/recall(B)"]))
            map50.append(float(row["metrics/mAP50(B)"]))
            map5095.append(float(row["metrics/mAP50-95(B)"]))

    fig, ax = plt.subplots(figsize=(5.4, 3.45))
    fig.patch.set_alpha(0)
    ax.set_facecolor("none")

    ax.plot(epochs, map50, "-o", color=AMBER, lw=1.9, ms=3.2, label="mAP@0.50", zorder=4)
    ax.plot(epochs, map5095, "-o", color=SAGE, lw=1.9, ms=3.2, label="mAP@0.50:0.95", zorder=4)
    ax.plot(epochs, precision, "-s", color=BLUE, lw=1.2, ms=2.6, label="Precision", zorder=3)
    ax.plot(epochs, recall, "-s", color=MUTED, lw=1.2, ms=2.6, label="Recall", zorder=3)

    best = max(range(len(map50)), key=lambda i: map50[i])
    ax.scatter([epochs[best]], [map50[best]], s=55, facecolors="none",
               edgecolors=AMBER, lw=1.3, zorder=6)
    ax.annotate(f"best {map50[best]:.3f} @ ep{epochs[best]}",
                (epochs[best], map50[best]), textcoords="offset points",
                xytext=(-14, 30), fontsize=7.2, color=AMBER, ha="right")

    ax.set_xlim(0.6, 10.4)
    ax.set_ylim(0, 0.7)
    ax.set_xticks(epochs)
    ax.set_xlabel("epoch  ·  YOLO11n  ·  640 px  ·  batch 8  ·  Apple-Silicon MPS",
                  fontsize=7.2, labelpad=5)
    ax.set_ylabel("score", fontsize=7.6)
    ax.grid(True, lw=0.5)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.tick_params(axis="both", length=0, labelsize=7.4)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.1f}"))
    ax.legend(frameon=False, fontsize=7.2, loc="lower right", ncol=2,
              handlelength=1.6, columnspacing=1.0, labelcolor=TEXT)

    fig.tight_layout(pad=0.35)
    out = os.path.join(HERE, "scifi_yolo.png")
    fig.savefig(out, transparent=True)
    plt.close(fig)
    print("wrote", out)


if __name__ == "__main__":
    benchmark()
    yolo()
