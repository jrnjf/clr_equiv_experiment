"""
Training loop shared by every experimental arm.

Key requirements this file is built around:

1. Per-STEP logging (not per epoch). Divergence is a per-step event; epoch
   averages hide it. A run that NaNs at step 300 and recovers looks fine in an
   epoch-mean log.
2. Identical seed -> init -> data order across arms. Everything that consumes
   randomness is seeded from one integer.
3. Divergence classification is explicit and recorded, so the analysis does not
   depend on eyeballing curves.

No torch import at module scope for the pure-Python pieces, so `classify()` can
be unit-tested without a GPU or torch.
"""

from __future__ import annotations

from dual_scheduler import train_step

import csv
import json
import math
import os
import time


# ---------------------------------------------------------------------------
# Failure classification (pure python, unit-testable)
# ---------------------------------------------------------------------------
def classify(losses: list[float], accs: list[float] | None = None,
             chance: float = 0.10, blowup_factor: float = 10.0) -> str:
    """Label a run as 'diverged', 'stalled' or 'trained'.

    diverged : any non-finite loss, or loss never recovers after exceeding
               blowup_factor x its running minimum
    stalled  : finite throughout but accuracy never exceeds a chance threshold
    trained  : otherwise
    """
    if not losses:
        return "empty"

    # any NaN/inf, or blow-up without recovery
    best = float("inf")
    diverged = False
    for i, loss in enumerate(losses):
        if loss is None or not math.isfinite(loss):
            diverged = True
            break
        best = min(best, loss)
        if best > 0 and loss > blowup_factor * best:
            # does it ever come back down within the remaining run?
            tail = losses[i:]
            if all((x is not None and math.isfinite(x) and x > 2.0 * best) for x in tail):
                diverged = True
                break
    if diverged:
        return "diverged"

    if accs:
        final_acc = accs[-1]
        if final_acc is not None and final_acc <= chance:
            return "stalled"
    return "trained"


def summarise(rows: list[dict]) -> dict:
    """Aggregate per-step rows into a run summary."""
    losses = [r["loss"] for r in rows]
    finite = [x for x in losses if x is not None and math.isfinite(x)]
    return {
        "n_steps": len(rows),
        "first_loss": losses[0] if losses else None,
        "last_loss": losses[-1] if losses else None,
        "min_loss": min(finite) if finite else None,
        "n_nonfinite": len(losses) - len(finite),
        "status": classify(losses),
    }


# ---------------------------------------------------------------------------
# Torch-dependent runner (imported lazily)
# ---------------------------------------------------------------------------
def run_arm(
    arm_name: str,
    make_model,
    make_optimizer_and_scheduler,
    train_loader,
    eval_fn,
    epochs: int,
    steps_per_epoch: int,
    device: str = "cuda",
    log_every: int = 1,
    out_dir: str = "results",
    seed: int = 0,
):
    """Train one arm and write `{out_dir}/{arm_name}_seed{seed}.csv`.

    make_optimizer_and_scheduler(model) -> (optimizer, scheduler_or_None)
      scheduler is stepped AFTER optimizer.step() every batch.
    eval_fn(model, loader) -> (loss, acc)
    """
    import torch

    os.makedirs(out_dir, exist_ok=True)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    model = make_model().to(device)
    opt, sched = make_optimizer_and_scheduler(model)
    criterion = torch.nn.CrossEntropyLoss()

    path = os.path.join(out_dir, f"{arm_name}_seed{seed}.csv")
    t0 = time.time()
    rows = []
    global_step = 0

    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["step", "epoch", "step_in_epoch", "lr", "loss",
                    "acc", "grad_norm", "weight_norm", "param_max"])

        for epoch in range(epochs):
            model.train()
            for step_in_epoch, (x, y) in enumerate(train_loader):
                if step_in_epoch >= steps_per_epoch:
                    break
                x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)

                opt.zero_grad(set_to_none=True)
                out = model(x)
                loss = criterion(out, y)
                loss.backward()
                loss_value = float(loss.detach())

                # measured BEFORE the update, so it reflects the step that is about
                # to be taken (this is the quantity that diverges first)
                grad_norm = float(
                    torch.sqrt(sum((p.grad.detach() ** 2).sum()
                                   for p in model.parameters() if p.grad is not None))
                )
                # The LR actually used for THIS step. Reading group['lr'] is
                # wrong for optimizers that carry the schedule internally
                # (DualScheduledSGD never touches group['lr']; it holds lr_max
                # forever), which made internal-arm logs read a constant 0.8.
                if hasattr(opt, "effective_lr"):
                    sc = opt.state.get(next(iter(opt.state)), {}).get("step_count", 0)                          if opt.state else 0
                    lr_now = float(opt.effective_lr(sc))
                elif sched is not None and getattr(sched, "last_lr", None) is not None \
                        and getattr(sched, "_applied", False):
                    lr_now = float(sched.last_lr)
                else:
                    lr_now = float(opt.param_groups[0]["lr"])

                train_step(opt, sched, loss)   # ordering handled in one place

                with torch.no_grad():
                    acc = (out.argmax(1) == y).float().mean().item()
                    wnorm = float(torch.sqrt(sum((p.detach() ** 2).sum()
                                                 for p in model.parameters())))
                    pmax = float(max((p.detach().abs().max() for p in model.parameters()),
                                     default=0.0))

                if global_step % log_every == 0:
                    w.writerow([global_step, epoch, step_in_epoch, lr_now,
                                loss_value, acc, grad_norm, wnorm, pmax])
                    rows.append({"loss": loss_value, "acc": acc})

                global_step += 1

            val_loss, val_acc = eval_fn(model, None)
            print(f"[{arm_name} seed{seed}] epoch {epoch+1}/{epochs} "
                  f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} "
                  f"({time.time()-t0:.0f}s)")

    summary = summarise(rows)
    summary.update({"arm": arm_name, "seed": seed, "epochs": epochs,
                    "steps_per_epoch": steps_per_epoch,
                    "wall_seconds": round(time.time() - t0, 1),
                    "final_val_acc": val_acc, "final_val_loss": val_loss})
    with open(os.path.join(out_dir, f"{arm_name}_seed{seed}.json"), "w") as fh:
        json.dump(summary, fh, indent=2)
    return summary
