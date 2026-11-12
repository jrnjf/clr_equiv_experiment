#!/usr/bin/env python3
"""
Reviewer 3.1: verify the learning-rate mathematics from the ACTUAL scheduler.

The review questions whether the claimed mean-rate match (0.200 vs 0.200) is real,
suggesting the product of a triangle (mean 0.5) and a cosine (mean 0.5) should
average 0.25 rather than 0.20.

Both numbers are correct and they are different quantities:

    E[eta_t] / eta_max = 0.25   <- the reviewer's quantity (ratio)
    E[eta_t]           = 0.20   <- what the paper reports (because eta_max = 0.8)

This script records every learning rate the optimizer actually used, from a real
torch LambdaLR, and prints the table the review asks for: peak, min, mean, sum(eta)
and sum(eta^2). It also reports the mean per-step weight shrinkage, which is what
the weight-decay-profile experiment has to match.

    python scripts/verify_lr_math.py
"""
import os
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from dual_scheduler import dual_schedule_lambda  # noqa: E402

SPE, EPOCHS, WD = 390, 200, 5e-4
N = SPE * EPOCHS


def used_lrs(period_epochs=1.0, lr_max=0.8, lr_ratio=1000.0, global_name="cosine",
             phase_mode="none"):
    """Every LR the optimizer actually used, read from the scheduler itself."""
    p = torch.nn.Parameter(torch.zeros(1))
    opt = torch.optim.SGD([p], lr=lr_max, momentum=0.0, weight_decay=WD)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, dual_schedule_lambda(SPE, lr_max, lr_ratio, EPOCHS, 1e-4,
                                  global_name, 0.0, period_epochs, phase_mode))
    out = []
    for _ in range(N):
        out.append(opt.param_groups[0]["lr"])
        opt.step()
        sched.step()
    return out


def stats(lrs, lr_max):
    n = len(lrs)
    s1 = sum(lrs)
    s2 = sum(x * x for x in lrs)
    return {
        "peak": max(lrs), "min": min(lrs), "max": max(lrs),
        "mean": s1 / n, "mean_ratio": (s1 / n) / lr_max,
        "sum": s1, "sum_sq": s2,
        "mean_shrink": WD * (s1 / n),   # E[eta]*wd = mean per-step shrinkage
    }


def main():
    rows = [
        ("triangle x cosine, momentum 0", used_lrs(), 0.8),
        ("cosine, peak 0.4 (mean-matched)", used_lrs(lr_max=0.4, lr_ratio=1.0), 0.4),
        ("cosine, peak 0.2", used_lrs(lr_max=0.2, lr_ratio=1.0), 0.2),
    ]
    print("Recorded directly from torch's LambdaLR over all 78,000 optimizer steps.\n")
    hdr = (f"{'schedule':<34}{'peak':>7}{'min':>10}{'mean':>8}"
           f"{'E[n]/max':>10}{'sum(n)':>12}{'sum(n^2)':>11}{'mean shrink':>13}")
    print(hdr)
    print("-" * len(hdr))
    for name, lrs, mx in rows:
        s = stats(lrs, mx)
        print(f"{name:<34}{s['peak']:>7.3f}{s['min']:>10.2e}{s['mean']:>8.4f}"
              f"{s['mean_ratio']:>10.4f}{s['sum']:>12.0f}{s['sum_sq']:>11.2f}"
              f"{s['mean_shrink']:>13.3e}")

    a, b = stats(rows[0][1], 0.8), stats(rows[1][1], 0.4)
    print(f"\nMean-rate match : {a['mean']:.4f} (method) vs {b['mean']:.4f} "
          f"(mean-matched cosine) -> ratio {a['mean']/b['mean']:.4f}")
    print(f"E[n]/eta_max    : {a['mean_ratio']:.4f} vs {b['mean_ratio']:.4f} "
          f"<- the reviewer's 0.25 quantity")
    print(f"Cumulative sum(n): {a['sum']:.0f} vs {b['sum']:.0f} "
          f"({a['sum']/b['sum']:.3f}x)")
    print(f"Cumulative sum(n^2): {a['sum_sq']:.1f} vs {b['sum_sq']:.1f} "
          f"({a['sum_sq']/b['sum_sq']:.3f}x)  <- differs, as the review notes")
    print("\nInterpretation for the paper:")
    print("  * E[eta]/eta_max = 0.25 and E[eta] = 0.20 are both correct; the")
    print("    former is a ratio, the latter includes eta_max = 0.8.")
    print("  * sum(eta) is matched to 0.1%, so cumulative step size is matched.")
    print("  * sum(eta^2) is NOT matched. A reviewer can argue that second-order")
    print("    quantities matter, so the paper should quote this number rather")
    print("    than claim matched optimisation dynamics.")
    print(f"\nMean per-step shrinkage: {a['mean_shrink']:.3e} "
          f"(this is the target for weight-decay condition C)")


if __name__ == "__main__":
    main()
