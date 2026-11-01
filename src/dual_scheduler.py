"""
The schedule, expressed the way PyTorch expects.

    lr(t) = lr_max * triangle(t mod steps_per_epoch) * cosine(t / steps_per_epoch)

**Prior art, stated plainly.** Neither component is ours:

* the triangular cycle is Smith's Cyclical Learning Rate policy
  (arXiv:1506.01186), defined per-iteration exactly like this;
* the cosine anneal is Loshchilov & Hutter's cosine schedule
  (arXiv:1608.03983), used here as a multiplier rather than as a restart.

Composing the two is a two-line `LambdaLR`. There is no novelty in the schedule
itself and the code should not imply any. What the experiments test is the
*regime* it is run in (peak lr 0.8, no momentum, an anneal to ~1e-4 of peak) and
how fast it reaches a given accuracy -- not the existence of the schedule.

`triangle` is normalised to [1/lr_SWR, 1] so it peaks at lr_max, and `cosine`
starts at 1 and anneals to `end_ratio`. Being a plain multiplier on the LR, it
plugs straight into `torch.optim.lr_scheduler.LambdaLR`:

    opt   = SGD(params, lr=lr_max, weight_decay=..., momentum=...)
    sched = LambdaLR(opt, dual_schedule_lambda(...))
    ...
    opt.step()
    sched.step()

Weight decay needs no handling at all: torch.optim.SGD computes
`grad + wd*param` and then `param -= lr*grad` (sgd.py lines 355 and 379), so the
decay is already multiplied by the current lr, which LambdaLR keeps updated.
No param loop, no custom optimizer, no manual decay.
"""

from __future__ import annotations

from schedules import cosine_anneal, triangle_wave


def dual_schedule_lambda(steps_per_epoch: int, lr_max: float, lr_SWR: float,
                         total_epochs: float, end_ratio: float = 1e-4,
                         global_name: str = "cosine"):
    """Return f(step) -> LR multiplier for LambdaLR."""
    if lr_SWR < 1.0:
        raise ValueError("lr_SWR must be >= 1")
    if steps_per_epoch < 2:
        raise ValueError("steps_per_epoch must be >= 2")

    def lr_lambda(step: int) -> float:
        local = triangle_wave(step % steps_per_epoch, steps_per_epoch,
                              1.0, 1.0 / lr_SWR)
        if global_name in ("cosine", "cos"):
            g = cosine_anneal(step / float(steps_per_epoch), total_epochs,
                              end_ratio)
        elif global_name in ("constant", "none", None):
            g = 1.0
        else:
            raise ValueError(f"unknown global schedule {global_name!r}")
        return local * g

    return lr_lambda


def lr_series(steps_per_epoch: int, lr_max: float, lr_SWR: float, n_steps: int,
              total_epochs: float, end_ratio: float = 1e-4,
              global_name: str = "cosine") -> list[float]:
    """The exact LR sequence LambdaLR produces, for auditing and plots."""
    f = dual_schedule_lambda(steps_per_epoch, lr_max, lr_SWR, total_epochs,
                             end_ratio, global_name)
    return [lr_max * f(t) for t in range(n_steps)]


def train_step(opt, sched, loss=None):
    """Standard PyTorch order: optimizer, then scheduler.

    LRScheduler applies lambda(0) in its constructor, so the first opt.step()
    already uses the schedule's first value, and each sched.step() advances to
    the next. No off-by-one, and no need for the prep()/advance() dance that
    earlier cost a 7.6e-2 relative parameter error.
    """
    opt.step()
    if sched is not None:
        sched.step()


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------
def build_method(model, steps_per_epoch: int, lr_max: float, lr_SWR: float,
                 global_name: str, total_epochs: float, end_ratio: float,
                 weight_decay: float, momentum: float = 0.0,
                 **_ignored):
    """The method: stock SGD + LambdaLR. Nothing custom."""
    from torch.optim import SGD
    from torch.optim.lr_scheduler import LambdaLR

    opt = SGD(model.parameters(), lr=lr_max, momentum=momentum,
              weight_decay=weight_decay)
    sched = LambdaLR(opt, dual_schedule_lambda(steps_per_epoch, lr_max, lr_SWR,
                                               total_epochs, end_ratio,
                                               global_name))
    return opt, sched


def build_cosine_only(model, steps_per_epoch: int, lr_max: float, lr_SWR: float,
                      global_name: str, total_epochs: float, end_ratio: float,
                      weight_decay: float, momentum: float = 0.0, **_ignored):
    """Baseline: cosine annealing only, no triangular component."""
    from torch.optim import SGD
    from torch.optim.lr_scheduler import LambdaLR

    opt = SGD(model.parameters(), lr=lr_max, momentum=momentum,
              weight_decay=weight_decay)
    sched = LambdaLR(opt, dual_schedule_lambda(steps_per_epoch, lr_max, 1.0,
                                               total_epochs, end_ratio,
                                               global_name))
    return opt, sched


def build_internal(model, steps_per_epoch: int, lr_max: float, lr_SWR: float,
                   global_name: str, total_epochs: float, end_ratio: float,
                   weight_decay: float, wd_on_effective_lr: bool = True,
                   momentum: float = 0.0, **_ignored):
    """Reference only: the custom optimizer, kept so equivalence stays testable."""
    from schedules import global_schedule
    from sgd_dual import DualScheduledSGD

    opt = DualScheduledSGD(
        model.parameters(), steps_per_epoch=steps_per_epoch, lr=lr_max,
        lr_SWR=lr_SWR,
        global_schedule=global_schedule(global_name, total_epochs, end_ratio),
        weight_decay=weight_decay, momentum=momentum,
        wd_on_effective_lr=wd_on_effective_lr)
    return opt, None


def assert_same_lr_series(steps_per_epoch: int, lr_max: float, lr_SWR: float,
                          n_steps: int, global_name: str = "cosine",
                          total_epochs: float = 1.0, end_ratio: float = 1e-4,
                          tol: float = 1e-12) -> float:
    """Torch-free check that the lambda matches the intended formula.

    Compares `lr_series()` (which goes through the lambda) against an
    independently composed triangle x cosine. Keeps `--dry-run` usable on a
    machine without torch; the LambdaLR alignment itself is checked separately
    by `assert_lambdalr_alignment()`.
    """
    produced = lr_series(steps_per_epoch, lr_max, lr_SWR, n_steps,
                         total_epochs, end_ratio, global_name)
    expected = []
    for t in range(n_steps):
        tri = triangle_wave(t % steps_per_epoch, steps_per_epoch,
                            lr_max, lr_max / lr_SWR)
        g = (cosine_anneal(t / float(steps_per_epoch), total_epochs, end_ratio)
             if global_name in ("cosine", "cos") else 1.0)
        expected.append(tri * g)
    worst = max(abs(a - b) for a, b in zip(produced, expected))
    assert worst <= tol, f"schedule differs from the intended formula by {worst:.3e}"
    return worst


def assert_lambdalr_alignment(steps_per_epoch: int, lr_max: float, lr_SWR: float,
                              n_steps: int, total_epochs: float,
                              end_ratio: float = 1e-4,
                              global_name: str = "cosine",
                              tol: float = 1e-12) -> float:
    """Check that LambdaLR actually emits the analytic series, step 0 included.

    Requires torch. This is the check that catches an off-by-one between the
    scheduler and the optimizer.
    """
    import torch
    from torch.optim import SGD
    from torch.optim.lr_scheduler import LambdaLR

    p = torch.nn.Parameter(torch.zeros(1))
    opt = SGD([p], lr=lr_max)
    sched = LambdaLR(opt, dual_schedule_lambda(steps_per_epoch, lr_max, lr_SWR,
                                               total_epochs, end_ratio,
                                               global_name))
    produced = []
    for _ in range(n_steps):
        produced.append(opt.param_groups[0]["lr"])
        opt.step()
        sched.step()

    expected = lr_series(steps_per_epoch, lr_max, lr_SWR, n_steps,
                         total_epochs, end_ratio, global_name)
    worst = max(abs(a - b) for a, b in zip(produced, expected))
    assert worst <= tol, f"LambdaLR series differs by {worst:.3e} > {tol:.3e}"
    return worst
