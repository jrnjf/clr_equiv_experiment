#!/usr/bin/env python3
"""
Figure: validation accuracy vs epoch for the four E1 arms, plus time-to-90%.

Reads results/e1_summary_cifar.json (the 200-epoch / 390-step / cuda run) and
writes docs/fig_e1_curves.png.

Usage:
    MPLCONFIGDIR=/tmp/mpl python scripts/plot_e1.py
"""
import json, os, statistics as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SUM = os.path.join(ROOT, "results", "e1_summary_cifar.json")
OUT = os.path.join(ROOT, "docs", "fig_e1_curves.png")

LABELS = {
    "A_internal": "A internal (tri x cosine)",
    "B1_external_wd_eff": "B1 external, wd on eff. LR",
    "B2_external_wd_base": "B2 external, wd on base LR",
    "C_cosine_only": "C cosine only (no triangle)",
}
COLORS = {
    "A_internal": "#1f77b4",
    "B1_external_wd_eff": "#2ca02c",
    "B2_external_wd_base": "#ff7f0e",
    "C_cosine_only": "#d62728",
}
ORDER = ["A_internal", "B1_external_wd_eff", "B2_external_wd_base", "C_cosine_only"]


def main():
    rows = json.load(open(SUM))
    arms = [a for a in ORDER if any(r["arm"] == a for r in rows)]

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(11, 4.2),
                                  gridspec_kw={"width_ratios": [2.1, 1]})

    for a in arms:
        runs = [r for r in rows if r["arm"] == a]
        # mean curve across seeds, aligned by epoch
        n = min(len(r["val_history"]) for r in runs)
        mean_curve = [st.mean(r["val_history"][e]["val_acc"] for r in runs)
                      for e in range(n)]
        ax.plot(range(1, n + 1), mean_curve, color=COLORS[a], lw=2, label=LABELS[a])
        # shade seed spread
        lo = [min(r["val_history"][e]["val_acc"] for r in runs) for e in range(n)]
        hi = [max(r["val_history"][e]["val_acc"] for r in runs) for e in range(n)]
        ax.fill_between(range(1, n + 1), lo, hi, color=COLORS[a], alpha=0.12, lw=0)

    ax.axhline(0.90, ls="--", c="k", lw=1, alpha=0.55)
    ax.text(3, 0.904, "90% threshold", fontsize=9, alpha=0.8)
    ax.set_xscale("log")
    ax.set_xlim(1, 200)
    ax.set_ylim(0.25, 0.97)
    ax.set_xlabel("epoch (log scale)")
    ax.set_ylabel("test accuracy")
    ax.set_title("E1: triangular CLR vs cosine-only, CIFAR-10 / ResNet-18\n"
                 "200 epochs, batch 128 (390 steps/epoch), 3 seeds, mean $\\pm$ seed range")
    ax.legend(fontsize=8.5, loc="lower right")
    ax.grid(alpha=0.25)

    # time-to-90% bars
    means, errs = [], []
    for a in arms:
        e = [next((h["epoch"] for h in r["val_history"] if h["val_acc"] >= 0.90), None)
             for r in rows if r["arm"] == a]
        vals = [x for x in e if x]
        means.append(st.mean(vals) if vals else 0)
        errs.append(st.stdev(vals) if len(vals) > 1 else 0)
    bars = ax2.barh(range(len(arms)), means, xerr=errs,
                    color=[COLORS[a] for a in arms], alpha=0.85)
    ax2.set_yticks(range(len(arms)))
    ax2.set_yticklabels([LABELS[a].split(" (")[0] for a in arms], fontsize=9)
    ax2.invert_yaxis()
    ax2.set_xlabel("epochs to 90% accuracy")
    ax2.set_title("time-to-competence")
    for i, (b, m) in enumerate(zip(bars, means)):
        if m:
            ax2.text(m + 2, i, f"{m:.0f}", va="center", fontsize=9)
    ax2.grid(alpha=0.25, axis="x")

    fig.tight_layout()
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.savefig(OUT, dpi=160)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
