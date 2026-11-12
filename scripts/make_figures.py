#!/usr/bin/env python3
"""Generate the paper's figures from the result JSONs.

    python scripts/make_figures.py          # writes figures/*.png

Mean +- 1 sd across seeds; curves are smoothed with a 3-epoch moving average so
the figure shows trends rather than eval-to-eval noise.
"""
import glob, json, os, statistics as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "figures")
os.makedirs(OUT, exist_ok=True)

def load(pat):
    R = {}
    for p in glob.glob(os.path.join(ROOT, pat)):
        d = json.load(open(p))
        for r in (d if isinstance(d, list) else [d]):
            if "val_history" in r:
                R.setdefault(r["arm"], {})[r["seed"]] = r
    return R

def curve(R, arm, key="val_acc", smooth=3):
    seeds = sorted(R.get(arm, {}))
    if not seeds:
        return None, None, None
    ep = [h["epoch"] for h in R[arm][seeds[0]]["val_history"]]
    ys = []
    for s in seeds:
        v = [h[key] for h in R[arm][s]["val_history"]]
        if smooth > 1:
            v = [st.mean(v[max(0, i-smooth+1):i+1]) for i in range(len(v))]
        ys.append(v)
    mean = [st.mean(c) for c in zip(*ys)]
    sd = [st.stdev(c) if len(c) > 1 else 0 for c in zip(*ys)]
    return ep, mean, sd

def plot_band(ax, R, arm, label, color, thr=None, ls="-"):
    ep, m, s = curve(R, arm)
    if ep is None:
        return False
    ax.plot(ep, m, color=color, label=label, lw=1.6, ls=ls)
    ax.fill_between(ep, [a-b for a, b in zip(m, s)], [a+b for a, b in zip(m, s)],
                    color=color, alpha=0.13, lw=0)
    return True

R10 = load("results/*.json")
R100 = load("results_cifar100/*.json")

# ---- Fig 1: CIFAR-10 accuracy curves -------------------------------------
fig, ax = plt.subplots(figsize=(7, 4.2))
arms = [("B1_external_wd_eff", "triangle x cosine, momentum 0 (ours)", "#d62728"),
        ("C_cosine_only_lr0.4", "cosine, momentum 0 (mean-matched)", "#1f77b4"),
        ("C_cosine_only_lr0.1_mom0.9", "cosine + momentum 0.9, lr 0.1 (standard)", "#2ca02c"),
        ("AdamW_cosine_lr0.002_wd0.05", "AdamW + cosine (tuned)", "#9467bd"),
        ("B1_external_wd_eff_mom0.9", "ours + momentum 0.9", "#8c564b")]
for arm, lab, c in arms:
    plot_band(ax, R10, arm, lab, c)
ax.axhline(0.90, color="k", ls=":", lw=1, label="90% threshold")
ax.set_xlabel("epoch"); ax.set_ylabel("test accuracy")
ax.set_title("CIFAR-10 / ResNet-18 (mean $\\pm$ 1 s.d., 3 seeds)")
ax.legend(fontsize=7, loc="lower right"); ax.grid(alpha=.25)
fig.tight_layout(); fig.savefig(f"{OUT}/fig1_cifar10_accuracy.png", dpi=200)

# ---- Fig 2: the schedule -------------------------------------------------
import sys; sys.path.insert(0, os.path.join(ROOT, "src"))
from dual_scheduler import dual_schedule_lambda
f = dual_schedule_lambda(390, 0.8, 1000.0, 200, 1e-4, "cosine")
fig, axes = plt.subplots(1, 2, figsize=(9, 3.2))
steps = list(range(390 * 3))
axes[0].plot(steps, [0.8*f(t) for t in steps], color="#d62728", lw=1)
axes[0].set_xlabel("step"); axes[0].set_ylabel("learning rate")
axes[0].set_title("first 3 epochs: one triangular cycle per epoch")
full = list(range(0, 390*200, 39))
axes[1].plot(full, [0.8*f(t) for t in full], color="#d62728", lw=1)
axes[1].set_xlabel("step"); axes[1].set_title("all 200 epochs: cosine envelope")
for a in axes: a.grid(alpha=.25)
fig.tight_layout(); fig.savefig(f"{OUT}/fig2_schedule.png", dpi=200)

# ---- Fig 3: time to threshold -------------------------------------------
def t_at_all(R, arm, thr):
    """Per-seed epochs to threshold (None if a seed never reaches it)."""
    v = []
    for s in sorted(R.get(arm, {})):
        v.append(next((h["epoch"] for h in R[arm][s]["val_history"]
                       if h["val_acc"] >= thr), None))
    return v
def t_at(R, arm, thr):
    v = [x for x in t_at_all(R, arm, thr) if x is not None]
    return st.mean(v) if v else float("nan")
def t_at_sd(R, arm, thr):
    v = [x for x in t_at_all(R, arm, thr) if x is not None]
    return st.stdev(v) if len(v) > 1 else 0.0
thrs = [0.90, 0.93, 0.94]
fig, ax = plt.subplots(figsize=(7, 3.8))
import numpy as np
x = np.arange(len(thrs)); w = 0.2
for i, (arm, lab, c) in enumerate(arms[:4]):
    vals = [t_at(R10, arm, t) for t in thrs]
    errs = [t_at_sd(R10, arm, t) for t in thrs]
    # +-1 s.d. across the three seeds, so the variance quoted in the tables is
    # visible in the figure as well.
    ax.bar(x + (i-1.5)*w, vals, w, label=lab, color=c,
           yerr=errs, capsize=2.5,
           error_kw=dict(lw=0.8, ecolor="#333333", capthick=0.8))
ax.set_xticks(x); ax.set_xticklabels([f"{int(t*100)}%" for t in thrs])
ax.set_ylabel("epochs to threshold"); ax.set_xlabel("accuracy threshold")
ax.set_title("Time to reach a given accuracy (lower is better)")
ax.legend(fontsize=7); ax.grid(alpha=.25, axis="y")
fig.tight_layout(); fig.savefig(f"{OUT}/fig3_time_to_threshold.png", dpi=200)

# ---- Fig 4: CIFAR-100 ----------------------------------------------------
fig, ax = plt.subplots(figsize=(7, 4.2))
for arm, lab, c in (("B1_external_wd_eff_c100", "triangle x cosine (ours)", "#d62728"),
                    ("C_cosine_only_lr0.4_c100", "cosine (mean-matched)", "#1f77b4"),
                    ("C_cosine_only_lr0.2_c100", "cosine, peak 0.2", "#17becf"),
                    ("AdamW_cosine_lr0.002_wd0.05_c100", "AdamW + cosine (tuned)", "#9467bd")):
    plot_band(ax, R100, arm, lab, c)
ax.axhline(0.70, color="k", ls=":", lw=1, label="70% threshold")
ax.set_xlabel("epoch"); ax.set_ylabel("test accuracy")
ax.set_title("CIFAR-100 / ResNet-18 (mean $\\pm$ 1 s.d., 3 seeds)")
ax.legend(fontsize=8, loc="lower right"); ax.grid(alpha=.25)
fig.tight_layout(); fig.savefig(f"{OUT}/fig4_cifar100_accuracy.png", dpi=200)

print("wrote:")
for f_ in sorted(os.listdir(OUT)):
    if f_.endswith(".png"):
        print("  figures/" + f_, os.path.getsize(os.path.join(OUT, f_)) // 1024, "KB")
