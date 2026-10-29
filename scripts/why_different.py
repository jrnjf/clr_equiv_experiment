#!/usr/bin/env python3
"""
Explain why a run differs from the sgd_jumper benchmark.

Compares the learning-rate trajectory (not just the endpoints) across the
protocol differences that separate E1 from benchmarks/jumper_exp_results.csv:

  1. cosine horizon   (T_max = epochs) -- the dominant effect
  2. steps per epoch  (batch size 128 vs 250) -- sets the triangle's period
  3. normalization    (their std vs the common CIFAR-10 std)
  4. eval set         (full 10k test vs 5k val holdout)
  5. momentum         (0.0 for Jumper, 0.9 for their SGD baseline)

Usage:
    python scripts/why_different.py
    python scripts/why_different.py --epochs 30 --epochs 200
"""
import argparse, sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from schedules import cosine_anneal

BENCH = dict(epochs=200, batch=128, n_train=50000, lr_max=0.8, lr_SWR=10.0,
             momentum=0.0, mean=(0.4914, 0.4822, 0.4465),
             std=(0.2023, 0.1994, 0.2010))
E1 = dict(epochs=30, batch=250, n_train=45000, lr_max=0.8, lr_SWR=10.0,
          momentum=0.0, mean=(0.4914, 0.4822, 0.4465),
          std=(0.2470, 0.2435, 0.2616))


def spe(cfg):
    return cfg["n_train"] // cfg["batch"]


def report():
    rows = [("sgd_jumper benchmark", BENCH), ("E1 (configs/cifar.json)", E1)]
    print(f"{'':26}{'epochs':>8}{'batch':>7}{'steps/ep':>10}{'half-cycle':>12}")
    for name, c in rows:
        s = spe(c)
        print(f"{name:<26}{c['epochs']:>8}{c['batch']:>7}{s:>10}{s/2:>12.1f}")

    print("\n--- 1. cosine horizon (dominant) ---")
    print("peak LR of the triangular wave = lr_max * cosine(epoch, T)")
    print(f"{'epoch':>6}{'T=200':>10}{'T=30':>10}{'ratio':>8}")
    for e in (0, 5, 10, 15, 20, 25, 29):
        a = 0.8 * cosine_anneal(e, 200, 0.01)
        b = 0.8 * cosine_anneal(e, 30, 0.01)
        print(f"{e:>6}{a:>10.5f}{b:>10.5f}{a/b:>8.2f}x")
    print("  -> by epoch 29 the 30-epoch run is at 0.010 LR while the 200-epoch")
    print("     run is still at 0.760. They are different schedules, not a")
    print("     truncated version of one.")

    print("\n--- 2. steps per epoch -> the triangle's period ---")
    for name, c in rows:
        s = spe(c)
        print(f"  {name:<26} period = {s} steps/epoch; "
              f"peak at step {s//2}; {'EVEN' if s%2==0 else 'ODD (peak unsampled)'}")

    print("\n--- 3. normalization std ---")
    print(f"  benchmark: {BENCH['std']}")
    print(f"  E1       : {E1['std']}")
    print("  The benchmark uses the per-channel std of the [0,1] tensor")
    print("  (~0.20), not the standard CIFAR-10 std (~0.247). Different input")
    print("  scaling changes the effective loss curvature, hence the useful LR.")

    print("\n--- 4. evaluation set ---")
    print("  benchmark: full 10,000-image test split (no val holdout)")
    print("  E1       : 5,000 held out of train -> val acc; test never reported")
    print("  Numbers are therefore not on the same data.")

    print("\n--- 5. momentum / weight decay ---")
    print("  benchmark Jumper: momentum=0.0, wd=5e-4, decay coupled to eff. LR")
    print("  benchmark SGD   : momentum=0.9, wd=5e-4")
    print("  E1 A/B1/B2/C    : momentum=0.0 (config), wd coupling varies by arm")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, nargs="*", default=None,
                    help="compare these cosine horizons instead")
    a = ap.parse_args()
    if a.epochs:
        print(f"{'epoch':>6}" + "".join(f"{'T='+str(T):>12}" for T in a.epochs))
        for e in (0, 5, 10, 15, 20, 25, 29, 50, 100):
            print(f"{e:>6}" + "".join(f"{0.8*cosine_anneal(e,T,0.01):>12.5f}"
                                      for T in a.epochs))
        return 0
    report()
    return 0


if __name__ == "__main__":
    sys.exit(main())
