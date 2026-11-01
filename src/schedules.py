"""
Learning-rate schedules: a per-batch triangular cycle multiplied by a global anneal.

The triangular formulation mirrors the research `triangle_wave` exactly:

    def triangle_wave(t, T=1.0, max_val=1.0, min_val=0.1):
        x = t % T
        if x < T / 2:
            return min_val + (max_val - min_val) * (2 * x / T)
        else:
            return min_val + (max_val - min_val) * (2 - 2 * x / T)

i.e. a 1-D triangle wave with period T, value min_val at t = 0 and at t -> T,
and (asymptotically) max_val at t = T/2. Note that for odd T the peak falls
between samples, so the sampled maximum is slightly below max_val; this is a
property of the original implementation and is reproduced deliberately (see
tests/test_schedule_math.py).
"""

from __future__ import annotations

import math


def triangle_wave(t: int, T: float = 1.0, max_val: float = 1.0,
                  min_val: float = 0.1) -> float:
    """Reference implementation, transcribed exactly from the research code."""
    x = t % T
    if x < T / 2:
        return min_val + (max_val - min_val) * (2 * x / T)
    return min_val + (max_val - min_val) * (2 - 2 * x / T)


class TriangleWave:
    """Callable wrapper so schedules can be swapped without changing call sites."""

    def __call__(self, t: int, T: float, max_val: float, min_val: float) -> float:
        return triangle_wave(t, T, max_val, min_val)


def cosine_anneal(epoch: float, total_epochs: float, end_ratio: float = 0.01) -> float:
    """Cosine decay multiplier in [end_ratio, 1], 1 at epoch 0.

    `end_ratio` is the fraction of the initial multiplier remaining at the end
    of training. `controlled/schedules.py` in the research tree implies 0.01 for
    the cosine and 0.2 for PSWA; 0.01 is the default here.
    """
    if total_epochs <= 0:
        raise ValueError("total_epochs must be positive")
    if not 0.0 < end_ratio <= 1.0:
        raise ValueError(f"end_ratio must be in (0, 1], got {end_ratio}")
    progress = min(max(epoch / total_epochs, 0.0), 1.0)
    # cosine ramps 1 -> 0; rescale to end_ratio -> 1
    return end_ratio + (1.0 - end_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))


def constant(_epoch: float) -> float:
    return 1.0


def global_schedule(name: str, total_epochs: float, end_ratio: float = 0.01):
    """Factory: returns f(epoch) -> multiplier."""
    if name in ("cosine", "cos"):
        return lambda e: cosine_anneal(e, total_epochs, end_ratio)
    if name in ("constant", "none", "off", None):
        return constant
    raise ValueError(f"unknown global schedule: {name!r}")


def effective_lr_series(
    steps_per_epoch: int,
    lr_max: float,
    lr_SWR: float,
    n_epochs: int,
    global_name: str = "cosine",
    end_ratio: float = 0.01,
) -> list[float]:
    """The full per-batch LR series, for auditing and plotting.

    Returns one value per optimizer step over `n_epochs` epochs.
    """
    g = global_schedule(global_name, n_epochs, end_ratio)
    lr_min = lr_max / lr_SWR
    out = []
    for step in range(steps_per_epoch * n_epochs):
        c = step % steps_per_epoch
        out.append(g(step / steps_per_epoch) * triangle_wave(c, steps_per_epoch,
                                                             lr_max, lr_min))
    return out


def mean_effective_lr(steps_per_epoch: int, lr_max: float, lr_SWR: float,
                      epochs: int, global_name: str = "cosine",
                      end_ratio: float = 1e-4) -> float:
    """Time-averaged effective LR over a whole run."""
    series = effective_lr_series(steps_per_epoch, lr_max, lr_SWR, epochs,
                                 global_name, end_ratio)
    return sum(series) / len(series)


def matched_weight_decays(wd_effective: float, steps_per_epoch: int,
                          lr_max: float, lr_SWR: float, epochs: int,
                          global_name: str = "cosine",
                          end_ratio: float = 1e-4,
                          target_mean_shrink: float | None = None) -> dict:
    """Weight-decay coefficients giving EQUAL mean per-step shrink.

    The three conventions shrink weights by different amounts at the same
    coefficient, because they multiply by a different LR:

        "effective" : wd * lr_eff(t)   <- oscillating, annealed
        "base"      : wd * lr_max      <- constant
        "none"      : wd * 1           <- constant

    At lr_max=0.8, lr_SWR=1000, 200 epochs, the mean effective LR is ~0.200, so
    equal coefficients would make "base" 4x and "none" 5x stronger than
    "effective". This helper rescales so the MEAN shrink matches and only the
    DISTRIBUTION over steps differs -- which is the mechanism under test.
    """
    mean_lr = mean_effective_lr(steps_per_epoch, lr_max, lr_SWR, epochs,
                                global_name, end_ratio)
    if target_mean_shrink is not None:
        # Interpret the requested strength as the LR-WEIGHTED AVERAGE of the
        # decay: mean(wd * lr_eff) = target. The "effective" coefficient must
        # then be target / mean_lr, which is much larger than target itself.
        lam = float(target_mean_shrink)
        wd_effective = lam / mean_lr
    else:
        lam = wd_effective * mean_lr         # mean per-step shrink of "effective"
    # "base" decays against the ANNEALED base lr, whose mean is the mean of
    # lr_max * global(epoch) -- i.e. the triangle envelope's peak. Using lr_max
    # here would ignore the cosine and make "base" 2x too weak (and, once
    # matched, indistinguishable from "none").
    mean_base = mean_effective_lr(steps_per_epoch, lr_max, 1.0, epochs,
                                  global_name, end_ratio)
    return {
        "effective": wd_effective,
        "base": lam / mean_base,
        "none": lam,
        "_mean_per_step_shrink": lam,
        "_mean_effective_lr": mean_lr,
        "_mean_base_lr": mean_base,
    }
