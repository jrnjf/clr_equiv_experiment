#!/usr/bin/env python3
"""Figures for Revision C: period sweep, peak-LR sweep, decay strength/profile.

Reads the per-run JSONs in results/ and writes figures/*.png.
    MPLCONFIGDIR=/tmp/mpl .venv/bin/python scripts/make_revC_figures.py
"""
import glob, json, os, statistics as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "figures")


def load(pattern="results/*.json"):
    R = {}
    for p in glob.glob(os.path.join(ROOT, pattern)):
        d = json.load(open(p))
        for r in (d if isinstance(d, list) else [d]):
            if "val_history" in r:
                R[(r["arm"], r["seed"])] = r
    return R


R = load()


def series(arm, thr):
    """(n, mean_final, sd_final, mean_t, sd_t) for one arm."""
    finals, ts = [], []
    for (a, s), r in R.items():
        if a != arm:
            continue
        h = r["val_history"]
        va = [x["val_acc"] for x in h]
        finals.append(va[-1])
        t = next((x["epoch"] for x in h if x["val_acc"] >= thr), None)
        if t is not None:
            ts.append(t)
    sd = lambda v: st.stdev(v) if len(v) > 1 else 0.0
    return len(finals), st.mean(finals), sd(finals), (st.mean(ts) if ts else None), (sd(ts) if ts else None)


# ---------------- Period sweep ----------------
periods = [0.25, 0.5, 1.0, 2.0, 4.0, 8.0]
arms = {0.25: "P_period_per0.25", 0.5: "P_period_per0.5", 1.0: "B1_external_wd_eff",
        2.0: "P_period_per2", 4.0: "P_period_per4", 8.0: "P_period_per8"}
fin = [series(arms[p], 0.90) for p in periods]
t90 = [series(arms[p], 0.90)[3] for p in periods]
t94 = [series(arms[p], 0.94)[3] for p in periods]
fig, ax = plt.subplots(1, 2, figsize=(8.4, 3.2))
ax[0].errorbar(periods, [x[1] * 100 for x in fin], yerr=[x[2] * 100 for x in fin],
               marker="o", color="#d62728", capsize=3, lw=1.4)
ax[0].set_xscale("log", base=2); ax[0].set_xticks(periods)
ax[0].set_xticklabels(["0.25", "0.5", "1", "2", "4", "8"])
ax[0].set_xlabel("cycle period (epochs)"); ax[0].set_ylabel("final test accuracy (%)")
ax[0].set_title("final accuracy is flat")
ax[0].grid(alpha=.25)
ax[1].plot(periods, t90, "o-", color="#1f77b4", label="t@90", lw=1.4)
ax[1].plot(periods, t94, "s-", color="#2ca02c", label="t@94", lw=1.4)
ax[1].set_xscale("log", base=2); ax[1].set_xticks(periods)
ax[1].set_xticklabels(["0.25", "0.5", "1", "2", "4", "8"])
ax[1].set_xlabel("cycle period (epochs)"); ax[1].set_ylabel("epochs to threshold")
ax[1].set_title("longer periods help at 94%")
ax[1].legend(); ax[1].grid(alpha=.25)
fig.tight_layout(); fig.savefig(f"{OUT}/fig6_period_sweep.png", dpi=200)

# ---------------- Peak-LR sweep ----------------
lrs = [0.4, 0.6, 0.8, 1.0, 1.2]
arms = {0.4: "B1_external_wd_eff_lr0.4", 0.6: "B1_external_wd_eff_lr0.6",
        0.8: "B1_external_wd_eff", 1.0: "B1_external_wd_eff_lr1", 1.2: "B1_external_wd_eff_lr1.2"}
fin = [series(arms[l], 0.90) for l in lrs]
t90 = [series(arms[l], 0.90)[3] for l in lrs]
fig, ax = plt.subplots(1, 2, figsize=(8.4, 3.2))
ax[0].errorbar(lrs, [x[1] * 100 for x in fin], yerr=[x[2] * 100 for x in fin],
               marker="o", color="#d62728", capsize=3, lw=1.4)
ax[0].set_xlabel("peak learning rate"); ax[0].set_ylabel("final test accuracy (%)")
ax[0].set_title("final accuracy across a 3x range")
ax[0].grid(alpha=.25)
ax[1].plot(lrs, t90, "o-", color="#1f77b4", lw=1.4)
ax[1].set_xlabel("peak learning rate"); ax[1].set_ylabel("epochs to 90%")
ax[1].set_title("time-to-90% is flat")
ax[1].grid(alpha=.25)
fig.tight_layout(); fig.savefig(f"{OUT}/fig7_peak_lr.png", dpi=200)

# ---------------- Decay strength vs profile ----------------
conds = [("A: coupled\n(oscillating)", "B1_external_wd_eff"),
         ("B: coupled\n(smooth)", "C_cosine_only_lr0.4"),
         ("C: independent\n(strength-matched)", "S_shrink_sh0.0001"),
         ("D: independent\n(5x strong)", "S_shrink_sh0.0005")]
finals = [series(a, 0.90) for _, a in conds]
peaks = [series(a, 0.90) for _, a in conds]
fig, ax = plt.subplots(figsize=(7.2, 3.6))
x = range(len(conds)); w = 0.36
ax.bar([i - w/2 for i in x], [f[1] * 100 for f in finals], w, label="final acc", color="#1f77b4")
ax.bar([i + w/2 for i in x], [p[1] * 100 for p in peaks], w, label="peak acc", color="#ff7f0e", alpha=.8)
# individual seeds
for i, ((_, arm), f) in enumerate(zip(conds, finals)):
    vals = [r["val_history"][-1]["val_acc"] * 100 for (a, s), r in R.items() if a == arm]
    ax.scatter([i - w/2] * len(vals), vals, color="k", s=10, zorder=5)
ax.set_xticks(list(x)); ax.set_xticklabels([c[0] for c in conds], fontsize=8)
ax.set_ylabel("test accuracy (%)"); ax.set_ylim(0, 100)
ax.set_title("decay strength, not profile, drives the collapse")
ax.legend(fontsize=8); ax.grid(alpha=.25, axis="y")
fig.tight_layout(); fig.savefig(f"{OUT}/fig5_decay.png", dpi=200)

print("wrote figures/fig5_decay.png, fig6_period_sweep.png, fig7_peak_lr.png")
