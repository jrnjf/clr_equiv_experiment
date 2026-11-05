#!/usr/bin/env python3
"""
Turn the weekend's results into the four tables the paper needs.

    python scripts/weekend_report.py                 # print + write docs/WEEKEND_RESULTS.md
    python scripts/weekend_report.py --no-write

Reads every results*/*.json (so it does not matter which out_dir a battery used),
dedupes by (arm, seed) with the newest file winning, and prints:

  A  headline (CIFAR-10, 200 epochs)   with paired tests vs B1
  B  AdamW tuning                      -- decides the "+1.1pp" claim
  C  CIFAR-100 second setting           -- decides main-track viability
  D  budget-matched table               -- the peak-vs-budget claim
"""
import argparse
import glob
import json
import math
import os
import statistics as st
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "docs", "WEEKEND_RESULTS.md")
THRESHOLDS = (0.90, 0.93, 0.94, 0.95)


def load():
    runs = {}
    paths = [p for p in glob.glob(os.path.join(ROOT, "results*/*.json"))
             if "pilot" not in p]
    paths.sort(key=os.path.getmtime)
    for p in paths:
        try:
            d = json.load(open(p))
        except Exception:
            continue
        for r in (d if isinstance(d, list) else [d]):
            if "val_history" in r and "arm" in r and "seed" in r:
                runs[(r["arm"], r["seed"])] = r
    return runs


def metrics(r):
    h = r["val_history"]
    accs = [x["val_acc"] for x in h]
    out = {"peak": max(accs), "final": accs[-1], "epochs": r.get("epochs")}
    for t in THRESHOLDS:
        out[f"t{int(t*100)}"] = next((x["epoch"] for x in h
                                      if x["val_acc"] >= t), None)
    return out


def table(runs, arms, n_show=True):
    lines = ["| arm | n | peak | final | t@90 | t@93 | t@94 | t@95 |",
             "|---|---|---|---|---|---|---|---|"]
    for a in arms:
        seeds = sorted(s for (x, s) in runs if x == a)
        if not seeds:
            lines.append(f"| `{a}` | — | — | — | — | — | — | — |")
            continue
        ms = [metrics(runs[(a, s)]) for s in seeds]

        def cell(k):
            v = [m[k] for m in ms if m[k] is not None]
            return f"{st.mean(v):.0f}" if v else "never"

        lines.append(f"| `{a}` | {len(ms)} | {st.mean(m['peak'] for m in ms):.4f} "
                     f"| {st.mean(m['final'] for m in ms):.4f} | {cell('t90')} "
                     f"| {cell('t93')} | {cell('t94')} | {cell('t95')} |")
    return lines


def paired(runs, a, b, key):
    seeds = sorted({s for (x, s) in runs if x == a}
                   & {s for (x, s) in runs if x == b})
    d = []
    for s in seeds:
        va, vb = metrics(runs[(a, s)])[key], metrics(runs[(b, s)])[key]
        if va is not None and vb is not None:
            d.append(va - vb)
    if len(d) < 2:
        return None
    m = st.mean(d)
    sd = st.stdev(d)
    t = m / (sd / math.sqrt(len(d))) if sd else float("inf")
    # for time-to-threshold a WIN is a SMALLER number, so count negatives
    wins = sum(1 for x in d if x < 0) if key.startswith("t") else \
        sum(1 for x in d if x > 0)
    return m, t, len(d), wins


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-write", action="store_true")
    args = ap.parse_args()

    runs = load()
    if not runs:
        print("no results found under results*/")
        return 1
    has = lambda a: any(x == a for (x, _) in runs)
    L = []

    def emit(s=""):
        L.append(s)
        print(s)

    emit(f"# Weekend results  \n\n{len(runs)} runs loaded from results*/ "
         f"(newest file wins on duplicate arm+seed).\n")

    # ---------------- A: headline ----------------
    emit("## A. Headline (CIFAR-10, 200 epochs)\n")
    headline = [a for a in ("B1_external_wd_eff", "C_cosine_only_lr0.4",
                            "C_cosine_only_lr0.2", "C_cosine_only",
                            "AdamW_cosine_lr0.001", "B1_external_wd_eff_noanneal")
                if has(a)]
    for line in table(runs, headline):
        emit(line)
    emit("")
    for a in headline[1:]:
        for key, unit in (("final", "pp"), ("t90", "ep"), ("t94", "ep")):
            p = paired(runs, "B1_external_wd_eff", a, key)
            if not p:
                continue
            m, t, n, wins = p
            scale = 100 if key == "final" else 1
            verb = "faster" if key.startswith("t") else "higher"
            emit(f"* B1 − `{a}` {key}: **{m*scale:+.2f} {unit}**  t={t:+.2f}  "
                 f"n={n}  B1 {verb} in {wins}/{n}")
    emit("")

    # ---------------- B: AdamW tuning ----------------
    emit("## B. AdamW tuning — decides the +1.1 pp claim\n")
    aw = sorted(a for (a, _) in runs if a.startswith("AdamW") and "_c100" not in a)
    aw = sorted(set(aw))
    for line in table(runs, aw):
        emit(line)
    emit("")
    best = max((a for a in aw),
               key=lambda a: st.mean(metrics(runs[(a, s)])["final"]
                                     for s in {z for (x, z) in runs if x == a}))
    bm = st.mean(metrics(runs[(best, s)])["final"]
                 for s in {z for (x, z) in runs if x == best})
    emit(f"**Best AdamW: `{best}` final {bm:.4f}.**  "
         f"Claim survives if this stays below B1's final "
         f"({st.mean(metrics(runs[('B1_external_wd_eff', s)])['final'] for s in {z for (x,z) in runs if x=='B1_external_wd_eff'}):.4f}).")
    emit("")

    # ---------------- C: CIFAR-100 ----------------
    emit("## C. CIFAR-100 — decides main-track viability\n")
    c100 = sorted({a for (a, _) in runs if a.endswith("_c100")})
    if c100:
        for line in table(runs, c100):
            emit(line)
        emit("")
        b1c = "B1_external_wd_eff_c100"
        if has(b1c):
            for a in c100:
                if a == b1c:
                    continue
                for key, unit in (("final", "pp"), ("t90", "ep")):
                    p = paired(runs, b1c, a, key)
                    if not p:
                        continue
                    m, t, n, wins = p
                    scale = 100 if key == "final" else 1
                    verb = "faster" if key.startswith("t") else "higher"
                    emit(f"* B1 − `{a}` {key}: **{m*scale:+.2f} {unit}**  "
                         f"t={t:+.2f}  n={n}  B1 {verb} in {wins}/{n}")
    else:
        emit("*No CIFAR-100 runs found — battery 2 did not complete.*")
    emit("")

    # ---------------- D: budget-matched ----------------
    emit("## D. Budget-matched — peak gap vs budget\n")
    emit("| budget | arm | n | t@90 | final | peak |")
    emit("|---|---|---|---|---|---|")
    for B in (40, 80, 160, 200):
        for a, label in ((f"B1_external_wd_eff_ep{B}", "B1 triangle×cosine"),
                         (f"C_cosine_only_lr0.2_ep{B}", "cosine anneal 0.2")):
            if not has(a):
                continue
            seeds = sorted(s for (x, s) in runs if x == a)
            ms = [metrics(runs[(a, s)]) for s in seeds]
            t = [m["t90"] for m in ms if m["t90"] is not None]
            emit(f"| {B} | {label} | {len(ms)} | "
                 f"{st.mean(t):.0f} | {st.mean(m['final'] for m in ms):.4f} | "
                 f"{st.mean(m['peak'] for m in ms):.4f} |")
    emit("\nNote: budgets 40/80/160 use anneal horizon = budget for both arms. "
         "The 200 row is the full-budget run.\n")

    if not args.no_write:
        os.makedirs(os.path.dirname(OUT), exist_ok=True)
        open(OUT, "w").write("\n".join(L) + "\n")
        print(f"\nwrote {os.path.relpath(OUT, ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
