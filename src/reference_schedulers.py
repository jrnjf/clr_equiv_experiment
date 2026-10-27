"""
Pure-Python replica of torch.optim.lr_scheduler.CyclicLR.

This exists so the claim "the internal triangular wave is algebraically the same
object as CyclicLR(mode='triangular')" can be *tested* in an environment without
PyTorch, and so the reference arm can be sanity-checked before burning GPU time.

Transcribed from PyTorch's `CyclicLR.get_lr` (torch/optim/lr_scheduler.py):

    cycle = math.floor(1 + self.last_epoch / self.total_size)
    x = 1.0 + self.last_epoch / self.total_size - cycle
    if x <= self.step_ratio:
        scale_factor = x / self.step_ratio
    else:
        scale_factor = (x - 1) / (self.step_ratio - 1)
    lr = base_lr + (max_lr - base_lr) * scale_factor * scale_fn(cycle)

with `step_ratio = step_size_up / (step_size_up + step_size_down)` and
`scale_fn = 1` for mode='triangular'.
"""

from __future__ import annotations

import math


def cyclic_lr(
    last_epoch: int,
    base_lr: float,
    max_lr: float,
    step_size_up: int,
    step_size_down: int | None = None,
) -> float:
    """Reproduce PyTorch CyclicLR(mode='triangular') at a given step index."""
    if step_size_up <= 0:
        raise ValueError("step_size_up must be positive")
    step_size_down = step_size_up if step_size_down is None else step_size_down
    if step_size_down <= 0:
        raise ValueError("step_size_down must be positive")

    total_size = float(step_size_up + step_size_down)
    step_ratio = float(step_size_up) / total_size

    cycle = math.floor(1 + last_epoch / total_size)
    x = 1.0 + last_epoch / total_size - cycle
    if x <= step_ratio:
        scale_factor = x / step_ratio
    else:
        scale_factor = (x - 1) / (step_ratio - 1)

    return base_lr + (max_lr - base_lr) * scale_factor * 1.0  # triangular


def cyclic_lr_series(
    n_steps: int,
    base_lr: float,
    max_lr: float,
    step_size_up: int,
    step_size_down: int | None = None,
) -> list[float]:
    return [
        cyclic_lr(i, base_lr, max_lr, step_size_up, step_size_down)
        for i in range(n_steps)
    ]
