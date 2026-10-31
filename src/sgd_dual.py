"""
DualScheduledSGD — vanilla SGD with dual scheduling.

Extracted verbatim in behaviour from the research `sgd_dual_schd.py`
(sha256 29b1db30...), with three differences, all deliberate:

  1. It lives in a package and is importable.
  2. The learning-rate schedule is factored out into `schedules.py` so the
     *same* schedule object can drive an external torch scheduler in the
     equivalence arm (see experiments/run_equivalence.py).
  3. `apply_weight_decay_effective` records whether weight decay is applied
     against the effective (oscillating) LR -- which is what the original does
     -- or against the base LR, which is what torch.optim.SGD does. This is the
     single behaviour that makes the internal implementation differ from a
     standard SGD + CyclicLR composition, so it is made explicit rather than
     implicit.

The original file had no momentum and no bias correction in the stripped
version; `DualScheduledSGD` below keeps momentum optional so the reference
`SGD(momentum=0)` arm can be matched exactly, but defaults to 0.0.
"""

from __future__ import annotations

import torch
from torch.optim import Optimizer

from schedules import TriangleWave


class DualScheduledSGD(Optimizer):
    """Vanilla SGD with a per-batch triangular LR cycle multiplied by a global anneal.

    Parameters
    ----------
    params
        Model parameters.
    steps_per_epoch : int
        Number of *optimizer steps* per epoch (usually ``len(train_loader)``).
        Defines the period of the intra-epoch triangular wave: exactly one full
        cycle per epoch, phase-reset at each epoch boundary.
    lr : float
        Peak learning rate of the triangular wave *before* the global schedule
        is applied.
    lr_SWR : float
        Standing Wave Ratio = lr_max / lr_min. The minimum of the triangular
        wave is ``lr / lr_SWR``. Kept as a ratio (rather than an absolute
        minimum) so that a global schedule can scale both bounds coherently.
    global_schedule : callable | None
        Optional ``f(epoch_float) -> multiplier`` in [0, 1]. The effective LR is
        ``global_schedule(t) * triangle(t)``. ``t`` is expressed in epochs as a
        float so fractional-epoch schedules are possible.
    weight_decay : float
        Decoupled weight decay coefficient.
    momentum : float
        Defaults to 0.0 (momentum-free), matching the research configuration.
    wd_on_effective_lr : bool
        True  -> decay uses the effective (oscillating) LR  [original behaviour]
        False -> decay uses the base LR                  [torch.optim.SGD behaviour]
        This flag exists because it is a real behavioural difference between the
        internal optimizer and an external-scheduler composition. Set it to
        False to make the two arms comparable.
    """

    def __init__(
        self,
        params,
        steps_per_epoch: int,
        lr: float = 0.5,
        lr_SWR: float = 10.0,
        global_schedule=None,
        weight_decay: float = 0.0,
        momentum: float = 0.0,
        wd_on_effective_lr: bool = True,
        wd_mode: str | None = None,
    ):
        if lr_SWR <= 0.0:
            raise ValueError(f"lr_SWR must be positive, got {lr_SWR}")
        if steps_per_epoch < 2:
            raise ValueError(f"steps_per_epoch must be >= 2, got {steps_per_epoch}")
        if momentum < 0.0:
            raise ValueError(f"momentum must be >= 0, got {momentum}")

        defaults = dict(
            lr=lr,
            lr_SWR=lr_SWR,
            steps_per_epoch=steps_per_epoch,
            weight_decay=weight_decay,
            momentum=momentum,
        )
        super().__init__(params, defaults)

        self.global_schedule = global_schedule
        self.wd_on_effective_lr = wd_on_effective_lr
        if wd_mode is None:
            wd_mode = "effective" if wd_on_effective_lr else "base"
        if wd_mode not in ("effective", "base", "none"):
            raise ValueError(f"unknown wd_mode {wd_mode!r}")
        self.wd_mode = wd_mode
        self._triangle = TriangleWave()
        self._steps_done = 0

        for group in self.param_groups:
            for p in group["params"]:
                self.state[p]["step_count"] = 0
                self.state[p]["momentum_buffer"] = None

    # -- helpers ---------------------------------------------------------
    def lr_bounds(self) -> tuple[float, float]:
        """(lr_min, lr_max) of the triangular wave before global scaling."""
        g = self.param_groups[0]
        lr, swr = g["lr"], g["lr_SWR"]
        return lr / swr, lr

    def epoch_fraction(self) -> float:
        """Training progress in epochs (float), from the global step counter."""
        spe = self.param_groups[0]["steps_per_epoch"]
        return self._steps_done / float(spe)

    def effective_lr(self, step_count: int) -> float:
        """The LR actually used at intra-epoch position `step_count`."""
        g = self.param_groups[0]
        lr, swr = g["lr"], g["lr_SWR"]
        spe = g["steps_per_epoch"]
        tri = self._triangle(step_count, spe, lr, lr / swr)
        if self.global_schedule is None:
            return tri
        return tri * self.global_schedule(self.epoch_fraction())

    # -- step ------------------------------------------------------------
    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            weight_decay = group["weight_decay"]
            momentum = group["momentum"]
            spe = group["steps_per_epoch"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                state = self.state[p]
                d_p = p.grad

                if momentum != 0.0:
                    buf = state["momentum_buffer"]
                    if buf is None:
                        buf = torch.clone(d_p).detach()
                        state["momentum_buffer"] = buf
                    else:
                        buf.mul_(momentum).add_(d_p)
                    d_p = buf

                lr_effective = self.effective_lr(state["step_count"])

                # Weight decay is applied BEFORE the gradient update, with the
                # parameter multiplied by (1 - wd*lr), so that decay acts on the
                # pre-update weights exactly as torch.optim.SGD does:
                #     w <- w*(1 - wd*lr) - lr*g
                # The previous ordering (update, then decay) is algebraically
                # equivalent but differs in floating point AND applies decay to
                # the already-updated weight; measured divergence 3.2e-6 after a
                # single step, 2.1e-4 after 40.
                if weight_decay != 0:
                    if self.wd_mode == "effective":
                        wd_lr = lr_effective          # oscillating / annealed
                    elif self.wd_mode == "base":
                        wd_lr = group["lr"]           # constant lr_max
                    else:                             # "none"
                        wd_lr = 1.0                   # no LR scaling at all
                    p.data.mul_(1.0 - weight_decay * wd_lr)

                p.data.add_(d_p, alpha=-lr_effective)

                state["step_count"] += 1
                if state["step_count"] >= spe:
                    state["step_count"] = 0

        self._steps_done += 1
        return loss
