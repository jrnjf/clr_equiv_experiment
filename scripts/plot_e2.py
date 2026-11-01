#!/usr/bin/env python3
"""
E2 figure: validation accuracy and weight norm for the three weight-decay
conventions, at one literal weight_decay=5e-4.

Writes docs/fig_e2_wd.png

    MPLCONFIGDIR=/tmp/mpl python scripts/plot_e2.py
"""
import csv, glob, json, os, statistics as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D = os.path.join(ROOT, "results")
OUT = os.path.join(ROOT, "docs", "fig_e2_wd.png")

ARMS = {
    "B1_external_wd_eff":  ("B1  wd x triangle x cosine  (1.0e-4/step)", "#1f77b4"),
    "B2_external_wd_base": ("B2  wd x lr_max x cosine   (2.0e-4/step)", "#2ca02c"),
    "F_external_wd_none":  ("F   wd, no LR scaling      (5.0e-4/step)", "#d62728"),
}


def curves(arm):
    seeds = sorted(glob.glob(os.path.join(D, f"{arm}_seed*.json")))
    accs, wns = [], []
    for p in seeds:
        d = json.load(open(p))
        if not d.get("val_history"):
            continue
        accs.append([h["val_acc"] for h in d["val_history"]])
        seed = d["seed"]
        wn = {}
        cp = os.path.join(D, f"{arm}_seed{seed}.csv")
        if os.path.exists(cp):
            for row in csv.DictReader(open(cp)):
                wn.setdefault(int(row["epoch"]) + 1, []).append(float(row["weight_norm"]))
            wns.append({e: st.mean(v) for e, v in wn.items()})
    n = min(len(a) for a in accs)
    mean = [st.mean(a[i] for a in accs) for i in range(n)]
    lo = [min(a[i] for a in accs) for i in range(n)]
    hi = [max(a[i] for a in accs) for i in range(n)]
    wn_mean = None
    if wns:
        m = max(max(w) for w in wns)          # longest curve, not shortest
        wn_mean = [st.mean(w[e] for w in wns if e in w) for e in range(1, m + 1)]
    return range(1, n + 1), mean, lo, hi, wn_mean


def main():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.3))

    for arm, (label, color) in ARMS.items():
        x, mean, lo, hi, wn = curves(arm)
        if not mean:
            continue
        ax1.plot(x, mean, color=color, lw=2, label=label)
        ax1.fill_between(x, lo, hi, color=color, alpha=0.15, lw=0)
        if wn is not None:
            ax2.plot(range(1, len(wn) + 1), wn, color=color, lw=2, label=label.split("  ")[0])

    ax1.axhline(0.90, ls="--", c="k", lw=1, alpha=0.5)
    ax1.text(3, 0.905, "90%", fontsize=9, alpha=0.7)
    ax1.axhline(0.10, ls=":", c="k", lw=1, alpha=0.5)
    ax1.text(3, 0.115, "chance (10 classes)", fontsize=8, alpha=0.7)
    ax1.set_xscale("log")
    ax1.set_xlim(1, 200)
    ax1.set_ylim(0.0, 0.97)
    ax1.set_xlabel("epoch (log scale)")
    ax1.set_ylabel("test accuracy")
    ax1.set_title("Accuracy: constant decay collapses at the end")
    ax1.legend(fontsize=7.5, loc="lower left")
    ax1.grid(alpha=0.25)

    ax2.set_xlabel("epoch")
    ax2.set_ylabel("mean weight norm")
    ax2.set_title("Weight norm: decay acts unopposed as the LR anneals")
    ax2.legend(fontsize=8, loc="upper right")
    ax2.grid(alpha=0.25)

    fig.suptitle("E2 weight-decay conventions, one literal weight_decay=5e-4 "
                 "(CIFAR-10 / ResNet-18, 200 epochs, 3 seeds, mean and range)",
                 fontsize=10.5)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(OUT, dpi=160)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
