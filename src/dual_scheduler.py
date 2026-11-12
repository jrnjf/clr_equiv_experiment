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
                         global_name: str = "cosine", phase_offset: float = 0,
                         period_epochs: float = 1.0,
                         phase_mode: str = "none",
                         phase_seed: int = 1234):
    """Return f(step) -> LR multiplier for LambdaLR.

    `phase_offset` (in steps) shifts the triangle within the epoch. With the
    default 0 the LR is at its trough at every epoch boundary, which is exactly
    when validation runs -- so the model is always measured at its most stable
    phase. A half-epoch offset moves evaluation to the LR peak, which is how we
    test whether a reported speed-up is real or an evaluation-phase artifact.
    """
    if lr_SWR < 1.0:
        raise ValueError("lr_SWR must be >= 1")
    if steps_per_epoch < 2:
        raise ValueError("steps_per_epoch must be >= 2")

    # Reviewer 6.2: the period is a free parameter, not assumed to be 1 epoch.
    period_steps = float(period_epochs) * steps_per_epoch
    if period_steps < 2:
        raise ValueError("period_epochs too small: period would be < 2 steps")

    # Reviewer 6.1: is the effect an artefact of the cycle being phase-locked to
    # epoch boundaries (and therefore to data shuffling)? Two randomisations:
    # a single random initial phase, or a fresh random phase every epoch.
    offsets = None
    if phase_mode != "none":
        import random
        rng = random.Random(phase_seed)
        if phase_mode == "random_init":
            offsets = [rng.uniform(0.0, period_steps)]
        elif phase_mode == "random_per_epoch":
            offsets = [rng.uniform(0.0, period_steps)
                       for _ in range(int(total_epochs) + 2)]
        else:
            raise ValueError(f"unknown phase_mode {phase_mode!r}")

    def lr_lambda(step: int) -> float:
        if offsets is None:
            off = phase_offset
        elif len(offsets) == 1:
            off = offsets[0]
        else:
            off = offsets[int(step // steps_per_epoch)]
        # triangle_wave already applies the modulo, so a float period works.
        local = triangle_wave(step + off, period_steps, 1.0, 1.0 / lr_SWR)
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
                 phase_offset: float = 0.0, period_epochs: float = 1.0,
                 phase_mode: str = "none", phase_seed: int = 1234,
                 **_ignored):
    """The method: stock SGD + LambdaLR. Nothing custom."""
    from torch.optim import SGD
    from torch.optim.lr_scheduler import LambdaLR

    opt = SGD(model.parameters(), lr=lr_max, momentum=momentum,
              weight_decay=weight_decay)
    sched = LambdaLR(opt, dual_schedule_lambda(
        steps_per_epoch, lr_max, lr_SWR, total_epochs, end_ratio, global_name,
        phase_offset, period_epochs, phase_mode, phase_seed))
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


def build_constant(model, steps_per_epoch: int, lr_max: float, lr_SWR: float,
                   global_name: str, total_epochs: float, end_ratio: float,
                   weight_decay: float, momentum: float = 0.0, **_ignored):
    """Control: a fixed learning rate, no schedule whatsoever.

    This is the baseline the triangular sweep has to beat. It exists because the
    triangle's mean LR (about lr_max/2) differs from the cosine-only arm's, so
    without a mean-matched constant arm, "oscillation helps" and "a lower average
    LR helps" are indistinguishable.
    """
    from torch.optim import SGD

    opt = SGD(model.parameters(), lr=lr_max, momentum=momentum,
              weight_decay=weight_decay)
    return opt, None


def build_adamw_cosine(model, steps_per_epoch: int, lr_max: float,
                       lr_SWR: float, global_name: str, total_epochs: float,
                       end_ratio: float, weight_decay: float,
                       momentum: float = 0.0, **_ignored):
    """AdamW with cosine annealing — the standard recipe we have to beat.

    AdamW decays differently from SGD (decoupled, adam.py:419) and needs its own
    weight_decay scale, so the config supplies it explicitly.
    """
    from torch.optim import AdamW
    from torch.optim.lr_scheduler import LambdaLR

    opt = AdamW(model.parameters(), lr=lr_max, weight_decay=weight_decay)
    sched = LambdaLR(opt, dual_schedule_lambda(steps_per_epoch, lr_max, 1.0,
                                               total_epochs, end_ratio,
                                               global_name))
    return opt, sched


def build_adamw_dual(model, steps_per_epoch: int, lr_max: float, lr_SWR: float,
                     global_name: str, total_epochs: float, end_ratio: float,
                     weight_decay: float, momentum: float = 0.0, **_ignored):
    """Same triangular x cosine schedule, but on AdamW.

    Tests whether the cycle helps adaptive optimizers too, or only SGD.
    """
    from torch.optim import AdamW
    from torch.optim.lr_scheduler import LambdaLR

    opt = AdamW(model.parameters(), lr=lr_max, weight_decay=weight_decay)
    sched = LambdaLR(opt, dual_schedule_lambda(steps_per_epoch, lr_max, lr_SWR,
                                               total_epochs, end_ratio,
                                               global_name))
    return opt, sched


def build_step_decay(model, steps_per_epoch: int, lr_max: float, lr_SWR: float,
                     global_name: str, total_epochs: float, end_ratio: float,
                     weight_decay: float, momentum: float = 0.0,
                     milestones=(0.5, 0.75), gamma: float = 0.1, **_ignored):
    """Classic step decay: lr x gamma at each fraction of the budget.

    The pre-cosine CIFAR recipe. It has never been in this project's baseline set,
    which is conspicuous: it is the recipe that reached ~95% before cosine
    annealing, so leaving it out invites "you only compared against schedules
    that happen to be weak here".
    """
    from torch.optim import SGD
    from torch.optim.lr_scheduler import LambdaLR

    def f(step):
        epoch = step / float(steps_per_epoch)
        drops = sum(1 for m in milestones if epoch >= m * total_epochs)
        return gamma ** drops

    opt = SGD(model.parameters(), lr=lr_max, momentum=momentum,
              weight_decay=weight_decay)
    return opt, LambdaLR(opt, f)


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


# ---------------------------------------------------------------------------
# Growing-period triangular cycle
# ---------------------------------------------------------------------------
def _tri_fraction(u: float, lr_ratio: float) -> float:
    """Triangle on u in [0, 1): 1/lr_ratio at u=0, 1 at u=0.5, back down."""
    u = u % 1.0
    tri = 2.0 * u if u < 0.5 else 2.0 - 2.0 * u
    lo = 1.0 / lr_ratio
    return lo + (1.0 - lo) * tri


def piecewise_period_lambda(steps_per_epoch: int, lr_SWR: float,
                            segments, total_epochs: float,
                            end_ratio: float = 1e-4,
                            global_name: str = "constant"):
    """Triangular cycle whose PERIOD changes over training.

    `segments` is [(start_epoch, period_epochs), ...] sorted by start_epoch, e.g.
        [(0, 1), (32, 2), (64, 4), (128, 8)]
    means: a full min->max->min sweep takes 1 epoch for epochs 0-31, 2 epochs for
    32-63, and so on.

    The phase is ACCUMULATED (phase += 1/period per step) rather than computed as
    step/period, so the curve stays continuous when the period changes. Computing
    it directly would make the LR jump at every boundary, which is a different and
    much more violent schedule than the one intended.
    """
    if not segments:
        raise ValueError("segments must be non-empty")
    segs = [(int(round(e * steps_per_epoch)), float(per) * steps_per_epoch)
            for e, per in sorted(segments)]
    if segs[0][0] != 0:
        raise ValueError("first segment must start at epoch 0")
    if lr_SWR < 1.0:
        raise ValueError("lr_SWR must be >= 1")

    def phase_at(step: int) -> float:
        phase = 0.0
        for i, (start, period) in enumerate(segs):
            nxt = segs[i + 1][0] if i + 1 < len(segs) else None
            if nxt is None or step < nxt:
                return phase + max(0, step - start) / period
            phase += (nxt - start) / period
        return phase

    def lr_lambda(step: int) -> float:
        tri = _tri_fraction(phase_at(step), lr_SWR)
        if global_name in ("cosine", "cos"):
            g = cosine_anneal(step / float(steps_per_epoch), total_epochs,
                              end_ratio)
        elif global_name in ("constant", "none", None):
            g = 1.0
        else:
            raise ValueError(f"unknown global schedule {global_name!r}")
        return tri * g

    return lr_lambda


def build_growing_period(model, steps_per_epoch: int, lr_max: float,
                         lr_SWR: float, global_name: str, total_epochs: float,
                         end_ratio: float, weight_decay: float,
                         momentum: float = 0.0, period_segments=None,
                         **_ignored):
    """The method but with a period that lengthens over training."""
    from torch.optim import SGD
    from torch.optim.lr_scheduler import LambdaLR

    if period_segments is None:
        period_segments = [[0, 1], [32, 2], [64, 4], [128, 8]]
    segments = [(int(e), float(p)) for e, p in period_segments]

    opt = SGD(model.parameters(), lr=lr_max, momentum=momentum,
              weight_decay=weight_decay)
    sched = LambdaLR(opt, piecewise_period_lambda(steps_per_epoch, lr_SWR,
                                                  segments, total_epochs,
                                                  end_ratio, global_name))
    return opt, sched


def build_constant_shrink(model, steps_per_epoch: int, lr_max: float,
                          lr_SWR: float, global_name: str, total_epochs: float,
                          end_ratio: float, weight_decay: float,
                          momentum: float = 0.0, shrink: float = 1e-4,
                          period_epochs: float = 1.0, phase_mode: str = "none",
                          phase_seed: int = 1234, **_ignored):
    """Reviewer 4, conditions C and D: decay whose strength is LR-INDEPENDENT.

    Same learning-rate schedule as the method, but the per-step shrinkage is a
    constant fraction of the weights rather than `weight_decay * lr_t`. C sets
    `shrink` equal to the method's mean per-step shrinkage (1.0e-4) so that decay
    STRENGTH is matched and only the PROFILE differs; D uses the stronger 5e-4
    that the original F arm used.

    torch.optim.SGD with weight_decay=0 plus a pre-step `p *= (1 - shrink)`, which
    mirrors SGD's own ordering (sgd.py: decay is applied to the gradient, so the
    weight is multiplied before the gradient update).
    """
    from torch.optim.lr_scheduler import LambdaLR
    from optim_shrink import ConstantShrinkSGD

    opt = ConstantShrinkSGD(model.parameters(), lr=lr_max, momentum=momentum,
                            weight_decay=0.0, shrink=shrink)
    sched = LambdaLR(opt, dual_schedule_lambda(
        steps_per_epoch, lr_max, lr_SWR, total_epochs, end_ratio, global_name,
        0.0, period_epochs, phase_mode, phase_seed))
    return opt, sched
