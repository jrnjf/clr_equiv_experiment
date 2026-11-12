#!/usr/bin/env python3
"""
Tables for the supervisor-revision batteries.

    python scripts/supervisor_report.py            # print all sections
    python scripts/supervisor_report.py --write    # also docs/SUPERVISOR_RESULTS.md

Sections map to the review items:
  A  validation/test split            (P0, 3.4)
  B  cycle-period sweep               (P1, 6.2)
  C  peak-LR sensitivity              (P1, 6.3)
  D  decay strength vs decay profile  (P1, 4)
  E  cycle-phase randomisation        (P2, 6.1)
"""
import argparse
import glob
import json
import os
import statistics as st
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
THRS = (0.90, 0.93, 0.94)


def load():
    runs = {}
    for p in glob.glob(os.path.join(ROOT, "results*/*.json")):
        if "pilot" in p:
            continue
        try:
            d = json.load(open(p))
        except Exception:
            continue
        for r in (d if isinstance(d, list) else [d]):
            if "val_history" in r and "arm" in r:
                runs.setdefault(r["arm"], {})[r["seed"]] = r
    return runs


def met(r, key):
    h = r["val_history"]
    if key == "final":
        return h[-1]["val_acc"]
    if key == "test":
        return r.get("final_test_acc")
    return next((x["epoch"] for x in h if x["val_acc"] >= key), None)


def row(R, arm, keys=("final",) + THRS):
    if arm not in R:
        return None
    seeds = sorted(R[arm])
    out = {"n": len(seeds)}
    for k in keys:
        v = [met(R[arm][s], k) for s in seeds]
        v = [x for x in v if x is not None]
        out[k] = st.mean(v) if v else None
    out["peak"] = st.mean(max(x["val_acc"] for x in R[arm][s]["val_history"])
                          for s in seeds)
    return out


def table(R, arms, keys, headers, title):
    print(f"\n### {title}")
    hdr = f"{'arm':<40}{'n':>3}" + "".join(f"{h:>9}" for h in headers)
    print(hdr)
    print("-" * len(hdr))
    for a in arms:
        r = row(R, a, keys)
        if not r:
            print(f"{a:<40}  (missing)")
            continue
        cells = []
        for k in keys:
            v = r[k]
            cells.append("never" if v is None
                         else (f"{v:.4f}" if k in ("final", "peak", "test")
                               else f"{v:.0f}"))
        print(f"{a:<40}{r['n']:>3}" + "".join(f"{c:>9}" for c in cells))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args()
    R = load()
    if not R:
        print("no results found")
        return 1

    # ---------------- A: validation / test separation ----------------
    print("=" * 78)
    print("A. VALIDATION/TEST SEPARATION (reviewer 3.4)")
    print("=" * 78)
    val_arms = [a for a in R if a.endswith("_val")]
    if val_arms:
        print("val = best validation accuracy during training;")
        print("test = final accuracy on the untouched 10k test split.\n")
        hdr = f"{'arm':<40}{'n':>3}{'val':>9}{'test':>9}{'gap':>8}"
        print(hdr); print("-" * len(hdr))
        for a in sorted(val_arms):
            seeds = sorted(R[a])
            vals = [max(x["val_acc"] for x in R[a][s]["val_history"]) for s in seeds]
            tests = [R[a][s].get("final_test_acc") for s in seeds]
            tests = [t for t in tests if t is not None]
            if not tests:
                print(f"{a:<40}{len(seeds):>3}{st.mean(vals):>9.4f}   (no test eval)")
                continue
            print(f"{a:<40}{len(seeds):>3}{st.mean(vals):>9.4f}"
                  f"{st.mean(tests):>9.4f}{100*(st.mean(tests)-st.mean(vals)):>+7.2f}pp")
        print("\nThe question the review asks is whether hyperparameter SELECTION on")
        print("validation picks the same configuration as selection on test would.")
    else:
        print("(battery a not run yet)")

    # ---------------- B: period sweep ----------------
    print("\n" + "=" * 78)
    print("B. CYCLE-PERIOD SWEEP (reviewer 6.2)")
    print("=" * 78)
    per = [a for a in R if a.startswith("P_period")]
    if per:
        order = sorted(per, key=lambda a: float(a.split("_per")[1]))
        table(R, order, ("final",) + THRS,
              ("final", "t@90", "t@93", "t@94"), "period sweep")
        print(f"\nreference: 1-epoch cycle = B1_external_wd_eff "
              f"({row(R, 'B1_external_wd_eff')})")
    else:
        print("(battery b not run yet)")

    # ---------------- C: peak LR ----------------
    print("\n" + "=" * 78)
    print("C. PEAK LEARNING-RATE SENSITIVITY (reviewer 6.3)")
    print("=" * 78)
    pk = [a for a in R if a.startswith("B1_external_wd_eff_lr")]
    if pk:
        order = sorted(pk, key=lambda a: float(a.split("_lr")[1]))
        table(R, order, ("final",) + THRS,
              ("final", "t@90", "t@93", "t@94"), "peak-LR sweep")
        print(f"\nreference: peak 0.8 = B1_external_wd_eff "
              f"({row(R, 'B1_external_wd_eff')})")
    else:
        print("(battery c not run yet)")

    # ---------------- D: decay profile ----------------
    print("\n" + "=" * 78)
    print("D. DECAY STRENGTH vs DECAY PROFILE (reviewer 4)")
    print("=" * 78)
    sh = [a for a in R if a.startswith("S_shrink")]
    if sh:
        table(R, sorted(sh), ("final",) + THRS,
              ("final", "t@90", "t@93", "t@94"), "LR-independent decay")
        print("\nconditions:")
        print("  A  LR-coupled, oscillating   = B1_external_wd_eff")
        print("  B  LR-coupled, smooth        = C_cosine_only_lr0.4")
        print("  C  LR-independent 1e-4       = S_shrink_sh0.0001  (strength matched)")
        print("  D  LR-independent 5e-4       = S_shrink_sh0.0005  (original strong)")
        print("\nIf C matches A in final accuracy, LR-independent decay is fine at")
        print("matched strength and the collapse is a STRENGTH effect. If C collapses")
        print("like D, the profile -- not the strength -- is what destroys training.")
    else:
        print("(battery d not run yet)")

    # ---------------- E: phase ----------------
    print("\n" + "=" * 78)
    print("E. CYCLE-PHASE RANDOMISATION (reviewer 6.1)")
    print("=" * 78)
    ph = [a for a in R if a.startswith("R_phase")]
    if ph:
        table(R, [a for a in R if a.startswith("B1_external_wd_eff") and
                  "lr" not in a and "mom" not in a and "noanneal" not in a
                  and "ep" not in a and "phase" not in a] + sorted(ph),
              ("final",) + THRS, ("final", "t@90", "t@93", "t@94"),
              "phase lock vs randomised")
        print("\nIf randomised phase matches the locked phase, the effect is not an")
        print("artefact of the cycle being synchronised with data shuffling.")
    else:
        print("(battery e not run yet)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
