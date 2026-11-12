#!/usr/bin/env python3
"""
Reviewer 3.3 and 6.4: statistics the paper must report.

Produces, for the key paired comparisons:
  * every per-seed difference, not just the mean
  * 95% confidence intervals (t, n-1 df)
  * two-sided t p-values computed from the incomplete beta function (no scipy)
  * EXACT sign-flip permutation p-values -- with n=3 there are only 8 sign
    patterns, so the smallest attainable two-sided p is 0.25 whatever the effect
  * a corrected power table (Monte Carlo), which the review says should use
    d/sigma ~ 3.26 for 80% power at n=3 rather than 2.84
  * wall-clock time to threshold (reviewer 6.4)

    python scripts/stats_report.py
"""
import glob
import json
import math
import os
import random
import statistics as st

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# --------------------------------------------------------------------------
# t distribution without scipy
# --------------------------------------------------------------------------
def _betacf(a, b, x, itmax=200, eps=3e-16):
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c, d = 1.0, 1.0 - qab * x / qap
    d = 1.0 / (d if abs(d) > 1e-30 else 1e-30)
    h = d
    for m in range(1, itmax + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        d = 1.0 / (d if abs(d) > 1e-30 else 1e-30)
        c = 1.0 + aa / c
        c = c if abs(c) > 1e-30 else 1e-30
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        d = 1.0 / (d if abs(d) > 1e-30 else 1e-30)
        c = 1.0 + aa / c
        c = c if abs(c) > 1e-30 else 1e-30
        de = d * c
        h *= de
        if abs(de - 1.0) < eps:
            break
    return h


def _betai(a, b, x):
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    lbeta = math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)
    front = math.exp(math.log(x) * a + math.log(1.0 - x) * b - lbeta) / a
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _betacf(a, b, x)
    return 1.0 - math.exp(math.log(1.0 - x) * b + math.log(x) * a - lbeta) / b \
        * _betacf(b, a, 1.0 - x)


def t_p_two_sided(t, df):
    if df <= 0:
        return float("nan")
    return _betai(df / 2.0, 0.5, df / (df + t * t))


def t_ppf(p, df):
    """x such that P(|T| > x) = p, i.e. the two-sided critical value.

    Bisection on t_p_two_sided, which is monotonically decreasing in x. An earlier
    version bisected against 1-p and returned ~0 for every p, which silently
    zeroed every confidence interval and the whole power calculation.
    """
    lo, hi = 0.0, 60.0
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        if t_p_two_sided(mid, df) > p:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def exact_signflip_p(diffs):
    """Exact two-sided paired permutation p-value (all 2^n sign patterns)."""
    n = len(diffs)
    obs = abs(sum(diffs) / n)
    hits = 0
    for mask in range(1 << n):
        s = sum(d if (mask >> i) & 1 else -d for i, d in enumerate(diffs))
        if abs(s / n) >= obs - 1e-15:
            hits += 1
    return hits / (1 << n)


def power_effect(n, power=0.80, alpha=0.05, trials=4000, seed=0):
    """Standardized d/sigma needed for `power` at sample size n (Monte Carlo)."""
    rng = random.Random(seed)
    df = n - 1
    crit = t_ppf(alpha, df)   # once per call, not per trial
    lo, hi = 0.0, 12.0
    for _ in range(16):
        d = 0.5 * (lo + hi)
        hits = 0
        for _ in range(trials):
            xs = [rng.gauss(d, 1.0) for _ in range(n)]
            m = sum(xs) / n
            sd = st.stdev(xs)
            t = m / (sd / math.sqrt(n)) if sd else float("inf")
            hits += t > crit
        if hits / trials < power:
            lo = d
        else:
            hi = d
    return 0.5 * (lo + hi)


# --------------------------------------------------------------------------
def load(pat="results*/*.json"):
    R = {}
    for p in glob.glob(os.path.join(ROOT, pat)):
        if "pilot" in p:
            continue
        try:
            d = json.load(open(p))
        except Exception:
            continue
        for r in (d if isinstance(d, list) else [d]):
            if "val_history" in r:
                R.setdefault(r["arm"], {})[r["seed"]] = r
    return R


def per_seed(R, arm, key, thr=None):
    out = {}
    for s, r in R.get(arm, {}).items():
        h = r["val_history"]
        if key == "final":
            out[s] = h[-1]["val_acc"]
        elif key == "wall":
            out[s] = next((x["wall"] for x in h if x["val_acc"] >= thr), None)
        else:
            out[s] = next((x["epoch"] for x in h if x["val_acc"] >= thr), None)
    return out


def compare(R, a, b, key, thr=None, unit="pp", scale=100):
    A, B = per_seed(R, a, key, thr), per_seed(R, b, key, thr)
    seeds = sorted(set(A) & set(B))
    pairs = [(A[s], B[s]) for s in seeds if A[s] is not None and B[s] is not None]
    if len(pairs) < 2:
        return None
    d = [x - y for x, y in pairs]
    n = len(d)
    m, sd = st.mean(d), st.stdev(d)
    df = n - 1
    t = m / (sd / math.sqrt(n)) if sd else float("inf")
    p_t = t_p_two_sided(t, df)
    p_ex = exact_signflip_p(d)
    half = t_ppf(0.05, df) * sd / math.sqrt(n) if sd else 0.0
    return dict(seeds=seeds, diffs=d, mean=m, sd=sd, t=t, p_t=p_t, p_ex=p_ex,
                ci=(m - half, m + half), n=n, unit=unit, scale=scale)


def main():
    R = load()
    B1 = "B1_external_wd_eff"
    print("=" * 92)
    print("PAIRED COMPARISONS (per-seed differences, 95% CI, exact permutation p)")
    print("=" * 92)
    comps = [
        (B1, "C_cosine_only_lr0.4", "final", None, "pp", 100, "vs mean-matched cosine"),
        (B1, "C_cosine_only_lr0.4", "epoch", 0.90, "ep", 1, "vs mean-matched cosine"),
        (B1, "C_cosine_only_lr0.1_mom0.9", "final", None, "pp", 100, "vs standard momentum"),
        (B1, "C_cosine_only_lr0.1_mom0.9", "epoch", 0.90, "ep", 1, "vs standard momentum"),
        (B1, "AdamW_cosine_lr0.002_wd0.05", "final", None, "pp", 100, "vs best tested AdamW"),
        (B1, "B1_external_wd_eff_mom0.9", "final", None, "pp", 100, "vs method+momentum"),
    ]
    for a, b, key, thr, unit, scale, label in comps:
        c = compare(R, a, b, key, thr, unit, scale)
        if not c:
            print(f"\n{label:<26} {key:<6} INSUFFICIENT DATA")
            continue
        print(f"\n{label}  ({key}{'' if thr is None else f'@{int(thr*100)}%'}, n={c['n']})")
        print(f"  per seed : {[round(x*c['scale'],3) for x in c['diffs']]}")
        print(f"  mean     : {c['mean']*c['scale']:+.3f} {unit}"
              f"   95% CI [{c['ci'][0]*c['scale']:+.3f}, {c['ci'][1]*c['scale']:+.3f}]")
        print(f"  t={c['t']:+.2f}  p(t)={c['p_t']:.4f}  "
              f"exact permutation p={c['p_ex']:.3f}")

    print("\n" + "=" * 92)
    print("CORRECTED POWER (reviewer 3.3)")
    print("=" * 92)
    print("The review says 80% power at n=3 needs d/sigma ~ 3.26, not 2.84.")
    for n in (3, 5, 8):
        e = power_effect(n)
        print(f"  n={n}: d/sigma needed for 80% power = {e:.2f}")
    print("\nDetectable final-accuracy difference, using the observed paired SD:")
    for label, arm in (("vs mean-matched cosine", "C_cosine_only_lr0.4"),
                       ("vs best tested AdamW", "AdamW_cosine_lr0.002_wd0.05")):
        c = compare(R, B1, arm, "final")
        if not c:
            continue
        e = power_effect(c["n"])
        print(f"  {label:<24} paired SD = {c['sd']*100:.2f} pp -> "
              f"detectable at 80% power = {e*c['sd']*100:.2f} pp "
              f"(observed {abs(c['mean'])*100:.2f} pp)")
    print("\nWith n=3 the exact permutation test cannot go below p=0.25 no matter")
    print("how large the effect is (8 sign patterns). Report effect sizes and CIs,")
    print("and describe agreement across seeds rather than leaning on t p-values.")

    print("\n" + "=" * 92)
    print("WALL-CLOCK TIME TO THRESHOLD (reviewer 6.4)")
    print("=" * 92)
    print(f"{'arm':<34}{'t@90 (ep)':>11}{'wall@90 (s)':>13}{'wall@93 (s)':>13}")
    for arm in (B1, "C_cosine_only_lr0.4", "C_cosine_only_lr0.1_mom0.9",
                "AdamW_cosine_lr0.002_wd0.05"):
        w90 = per_seed(R, arm, "wall", 0.90)
        w93 = per_seed(R, arm, "wall", 0.93)
        e90 = per_seed(R, arm, "epoch", 0.90)
        f = lambda d: (f"{st.mean([v for v in d.values() if v]):.0f}"
                       if any(d.values()) else "never")
        print(f"{arm:<34}{f(e90):>11}{f(w90):>13}{f(w93):>13}")
    print("\nNote: wall-clock includes per-epoch validation of 10k images, which is")
    print("equal across arms, so it does not tilt the comparison.")


if __name__ == "__main__":
    main()
