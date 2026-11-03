#!/usr/bin/env python3
"""
Merge every result file under results*/ and print one table.

Runs get written to different directories depending on the --out-dir used
(results/, results_e3a/, results_e2_pilot/, ...). Reading only one directory
silently hides runs, so this globs them all and dedupes by (arm, seed).

    python scripts/summarize_all.py                    # CIFAR runs, all dirs
    python scripts/summarize_all.py --baseline B1_external_wd_eff
    python scripts/summarize_all.py --markdown         # table for the paper
    python scripts/summarize_all.py --epochs 200 --min-seeds 3
"""
import argparse
import glob
import json
import math
import os
import statistics as st
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_all(pattern="results*/*.json", exclude=("*pilot*",)):
    """Merge every result file, newest wins on duplicate (arm, seed).

    Directories are visited oldest-first so a re-run supersedes the earlier one,
    and `exclude` drops superseded directories (the E2 pilot used pre-fix code).
    """
    paths = [p for p in glob.glob(os.path.join(ROOT, pattern))
             if not any(x.strip("*") in p for x in exclude)]
    paths.sort(key=os.path.getmtime)
    runs = {}
    for p in paths:
        try:
            d = json.load(open(p))
        except Exception:
            continue
        items = d if isinstance(d, list) else [d]
        for r in items:
            if "val_history" in r and "arm" in r and "seed" in r:
                runs[(r["arm"], r["seed"])] = r
    return runs


def metrics(r):
    h = r["val_history"]
    accs = [x["val_acc"] for x in h]
    t90 = next((x["epoch"] for x in h if x["val_acc"] >= 0.90), None)
    return max(accs), accs[-1], t90


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", default=None,
                    help="arm to compare every other arm against")
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--min-seeds", type=int, default=1)
    ap.add_argument("--markdown", action="store_true")
    ap.add_argument("--sort", default="final",
                    choices=["final", "peak", "t90", "arm"])
    ap.add_argument("--exclude", nargs="*", default=["*pilot*"],
                    help="directory globs to skip (default: superseded pilots)")
    a = ap.parse_args()

    runs = load_all(exclude=tuple(a.exclude))
    runs = {k: v for k, v in runs.items() if v.get("epochs") == a.epochs}
    if not runs:
        print("no runs found (looked in results*/*.json)")
        return 1

    by_arm = {}
    for (arm, seed), r in runs.items():
        by_arm.setdefault(arm, {})[seed] = r

    rows = []
    for arm, seeds in by_arm.items():
        if len(seeds) < a.min_seeds:
            continue
        vals = [metrics(r) for r in seeds.values()]
        rows.append({
            "arm": arm, "n": len(vals),
            "peak": st.mean(v[0] for v in vals),
            "final": st.mean(v[1] for v in vals),
            "t90": st.mean(v[2] for v in vals if v[2] is not None)
            if any(v[2] is not None for v in vals) else float("nan"),
            "never90": sum(1 for v in vals if v[2] is None),
        })

    key = {"final": lambda r: -r["final"], "peak": lambda r: -r["peak"],
           "t90": lambda r: r["t90"], "arm": lambda r: r["arm"]}[a.sort]
    rows.sort(key=key)

    print(f"{len(runs)} runs, {len(rows)} arms, epochs={a.epochs}\n")
    if a.markdown:
        print("| arm | n | peak | final | 1st 90% |")
        print("|---|---|---|---|---|")
        for r in rows:
            t = "never" if r["never90"] else f"{r['t90']:.1f}"
            print(f"| `{r['arm']}` | {r['n']} | {r['peak']:.4f} | "
                  f"{r['final']:.4f} | {t} |")
    else:
        print(f"{'arm':<34}{'n':>3}{'peak':>8}{'final':>8}{'1st 90%':>9}")
        print("-" * 62)
        for r in rows:
            t = "never" if r["never90"] else f"{r['t90']:.1f}"
            print(f"{r['arm']:<34}{r['n']:>3}{r['peak']:>8.4f}"
                  f"{r['final']:>8.4f}{t:>9}")

    if a.baseline and a.baseline in by_arm:
        base = by_arm[a.baseline]
        print(f"\npaired vs {a.baseline} (|t|>=4.30 for p<.05 at n=3):")
        for arm, seeds in sorted(by_arm.items()):
            common = sorted(set(seeds) & set(base))
            if arm == a.baseline or len(common) < 2:
                continue
            df = []; dt = []
            for s in common:
                fa = metrics(seeds[s]); fb = metrics(base[s])
                df.append(fa[1] - fb[1])
                if fa[2] is not None and fb[2] is not None:
                    dt.append(fa[2] - fb[2])
            for label, d, unit, scale in (("final", df, "pp", 100),
                                          ("t90", dt, "ep", 1)):
                if len(d) < 2:
                    continue
                mu, sd = st.mean(d), st.stdev(d)
                t = mu / (sd / math.sqrt(len(d))) if sd else float("inf")
                print(f"  {arm:<32}{label:<7}{mu*scale:+7.2f}{unit}"
                      f"  t={t:+6.2f}  wins {sum(1 for x in d if x > 0)}/{len(d)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
