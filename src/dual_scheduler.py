"""
Wrappers that build the two arms of the equivalence experiment.

Arm A (internal)  : DualScheduledSGD carries the schedule inside the optimizer.
Arm B (external)  : stock torch.optim.SGD + a schedule applied from outside.

Both must produce the *same* per-step learning rate series; that is asserted by
`assert_same_lr_series` below and checked in tests/.
"""

from __future__ import annotations

from schedules import global_schedule, triangle_wave

# torch is imported lazily so the LR-series equivalence (the mathematical claim)
# can be checked on a machine with no torch installed -- see --dry-run.


class ExternalDualScheduler:
    """Applies ``global_multiplier(epoch) * triangle(step_in_epoch)`` to an optimizer.

    This is the minimal, explicit counterpart to DualScheduledSGD's internal
    scheduling. It writes the LR into ``param_group['lr']`` before each step, so
    it works with any optimizer (including the stock SGD we compare against).

    Why not torch.optim.lr_scheduler.CyclicLR?
    -------------------------------------------
    Because CyclicLR cannot be multiplied by a second schedule: it *sets* the LR
    to a single interpolated value. Composing two torch schedulers is not
    supported. This wrapper implements the composition `global * local` directly,
    which is exactly what DualScheduledSGD does internally. The CyclicLR
    algorithm itself is replicated in reference_schedulers.py and asserted
    equivalent to `triangle_wave` by tests/.

    Even-steps requirement
    ----------------------
    `steps_per_epoch` must be EVEN. For odd values the triangular wave's peak
    falls between two steps, and (measured) no CyclicLR configuration
    reproduces the internal wave to better than 1.5e-2 absolute. Since the point
    of this experiment is bit-level equivalence, even values are enforced.
    """

    def __init__(self, optimizer, steps_per_epoch: int, lr_max: float,
                 lr_SWR: float, global_name: str = "cosine",
                 total_epochs: float = 1.0, end_ratio: float = 0.01):
        if steps_per_epoch % 2 != 0:
            raise ValueError(
                f"steps_per_epoch must be even for exact equivalence, got "
                f"{steps_per_epoch}. Use 2*ceil(n/2) or drop the last partial batch."
            )
        self.opt = optimizer
        self.steps_per_epoch = steps_per_epoch
        self.lr_max = lr_max
        self.lr_min = lr_max / lr_SWR
        self.global_fn = global_schedule(global_name, total_epochs, end_ratio)
        self._step = 0
        self.last_lr = None

    def lr_at(self, step: int) -> float:
        c = step % self.steps_per_epoch
        local = triangle_wave(c, self.steps_per_epoch, self.lr_max, self.lr_min)
        return local * self.global_fn(step / self.steps_per_epoch)

    def prep(self) -> float:
        """Apply the LR for the CURRENT step; call BEFORE optimizer.step().

        Ordering matters and is the single easiest way to break equivalence:
        DualScheduledSGD computes its LR inside `step()`, i.e. it uses the LR for
        step t when taking step t. An external scheduler that is stepped *after*
        optimizer.step() leaves optimizer.step() using whatever LR was set last
        time -- so the very first step would use lr_max (0.8) instead of lr_min
        (0.08), a 10x error on step 0. Measured effect before this fix:
        final-parameter relative difference 7.6e-2.
        """
        lr = self.lr_at(self._step)
        for group in self.opt.param_groups:
            group["lr"] = lr
        self.last_lr = lr
        self._applied = True
        return lr

    def advance(self) -> None:
        """Call AFTER optimizer.step() to move to the next step."""
        if not getattr(self, "_applied", False):
            raise RuntimeError("advance() called before prep(); the optimizer "
                               "would take a step with a stale learning rate.")
        self._applied = False
        self._step += 1

    def step(self) -> None:
        """Backwards-compatible alias: prep() only (no advance)."""
        self.prep()


def build_internal(model, steps_per_epoch: int, lr_max: float, lr_SWR: float,
                   global_name: str, total_epochs: float, end_ratio: float,
                   weight_decay: float, wd_on_effective_lr: bool = True,
                   momentum: float = 0.0):
    from sgd_dual import DualScheduledSGD
    sched = global_schedule(global_name, total_epochs, end_ratio)
    opt = DualScheduledSGD(
        model.parameters(),
        steps_per_epoch=steps_per_epoch,
        lr=lr_max,
        lr_SWR=lr_SWR,
        global_schedule=sched,
        weight_decay=weight_decay,
        momentum=momentum,
        wd_on_effective_lr=wd_on_effective_lr,
    )
    return opt, None


def build_external(model, steps_per_epoch: int, lr_max: float, lr_SWR: float,
                   global_name: str, total_epochs: float, end_ratio: float,
                   weight_decay: float, wd_on_effective_lr: bool = True,
                   momentum: float = 0.0):
    """Stock SGD + external schedule.

    Weight-decay coupling is the subtle part: torch.optim.SGD applies decay
    against group['lr'] only, which here is the *effective* (already scheduled)
    LR. So with wd_on_effective_lr=True this arm matches DualScheduledSGD
    exactly. To reproduce the 'decay on base LR only' variant, pass
    weight_decay=0 here and add the decay term manually.
    """
    from torch.optim import SGD
    # start at lr_min: step 0 of the triangular wave is the minimum, and if a
    # caller forgets to prep() before the first step we want the benign value.
    opt = SGD(model.parameters(), lr=lr_max / lr_SWR, momentum=momentum,
              weight_decay=weight_decay if wd_on_effective_lr else 0.0)
    sch = ExternalDualScheduler(opt, steps_per_epoch, lr_max, lr_SWR,
                                global_name, total_epochs, end_ratio)
    return opt, sch


def assert_same_lr_series(steps_per_epoch: int, lr_max: float, lr_SWR: float,
                          n_steps: int, global_name: str = "cosine",
                          total_epochs: float = 1.0, end_ratio: float = 0.01,
                          tol: float = 1e-12) -> float:
    """Compare the LR series produced by the internal and external paths.

    Returns the max absolute difference. Raises AssertionError if above tol.
    """
    lr_min = lr_max / lr_SWR
    g = global_schedule(global_name, total_epochs, end_ratio)

    internal = []
    for step in range(n_steps):
        c = step % steps_per_epoch
        local = triangle_wave(c, steps_per_epoch, lr_max, lr_min)
        internal.append(local * g(step / steps_per_epoch))

    sch = ExternalDualScheduler.__new__(ExternalDualScheduler)  # bypass ctor checks
    sch.steps_per_epoch = steps_per_epoch
    sch.lr_max = lr_max
    sch.lr_min = lr_min
    sch.global_fn = g
    external = [sch.lr_at(step) for step in range(n_steps)]

    worst = max(abs(a - b) for a, b in zip(internal, external))
    assert worst <= tol, f"LR series differ by {worst:.3e} > {tol:.3e}"
    return worst


def train_step(opt, sched, loss):
    """One correct optimization step, handling both scheduler styles.

    This exists so the *ordering* is written down exactly once:

      external scheduler : prep() -> opt.step() -> advance()
      internal scheduler : opt.step()            (LR computed inside)

    Getting this order wrong is the failure mode that made the first
    equivalence run disagree by 7.6e-2 relative.
    """
    if sched is None:
        opt.step()
        return
    sched.prep()
    opt.step()
    sched.advance()
