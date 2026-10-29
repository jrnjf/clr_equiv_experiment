#!/usr/bin/env python3
"""
Report time-to-threshold from run summary JSONs.

Final accuracy at a fixed epoch budget answers "how good at the end"; this
answers "how fast to competence", which is the metric the sgd_jumper benchmark
uses (Jumper 15 epochs to 90%, AdamW 20, SGD 101).

Usage:
    python scripts/analyze_tt90.py results/e1_summary_cifar.json
    python scripts/analyze_tt90.py results/*.json --thresh 0.90
"""
import argparse, glob, json, statistics as st, sys


def load(paths):
    runs = []
    for p in paths:
        try:
            d = json.load(open(p))
        except Exception:
            continue
        if isinstance(d, list):
            runs.extend(d)
        elif isinstance(d, dict) and "val_history" in d:
            runs.append(d)
    return runs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--thresh", type=float, default=0.90)
    a = ap.parse_args()

    paths = []
    for p in a.paths:
        paths.extend(glob.glob(p) or [p])
    runs = load(paths)
    if not runs:
        print("no runs with val_history found (re-run with the updated trainer)")
        return 1

    by = {}
    for r in runs:
        by.setdefault(r["arm"], []).append(r)

    print(f"threshold = {a.thresh:.0%} validation accuracy\n")
    print(f"{'arm':<24}{'seeds':>6}{'epoch→thr':>22}{'wall→thr':>18}{'final':>8}")
    for arm, rs in sorted(by.items()):
        eps = [next((h["epoch"] for h in r["val_history"] if h["val_acc"] >= a.thresh), None)
               for r in rs]
        walls = [next((h["wall"] for h in r["val_history"] if h["val_acc"] >= a.thresh), None)
                 for r in rs]
        hit_e = [e for e in eps if e is not None]
        hit_w = [w for w in walls if w is not None]
        fin = st.mean([r["final_val_acc"] for r in rs])
        e_s = (f"{st.mean(hit_e):.1f} (n={len(hit_e)}/{len(rs)})" if hit_e else "never")
        w_s = (f"{st.mean(hit_w):.0f}s (n={len(hit_w)}/{len(rs)})" if hit_w else "never")
        print(f"{arm:<24}{len(rs):>6}{e_s:>22}{w_s:>18}{fin:>8.4f}")

    print("\nper-run detail:")
    for r in sorted(runs, key=lambda x: (x["arm"], x["seed"])):
        e = next((h["epoch"] for h in r["val_history"] if h["val_acc"] >= a.thresh), None)
        print(f"  {r['arm']:<24} seed{r['seed']}  ->thr={str(e):>5}  "
              f"final={r['final_val_acc']:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
