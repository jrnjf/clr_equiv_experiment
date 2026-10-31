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

    # Fixed input shape (32x32 CIFAR) makes cuDNN autotuning worth it: it picks
    # the fastest conv algorithms on the first few iterations. Off by default in
    # torch, costs one short warmup, and does not change the math.
    if str(device).startswith("cuda") and torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True

    model = make_model().to(device)
    opt, sched = make_optimizer_and_scheduler(model)
    criterion = torch.nn.CrossEntropyLoss()

    path = os.path.join(out_dir, f"{arm_name}_seed{seed}.csv")
    t0 = time.time()
    rows = []
    val_hist = []          # per-epoch validation curve (for time-to-threshold)
    global_step = 0
    csv_buffer = []        # batched CSV writes (see note in the inner loop)

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

                # ------------------------------------------------------------
                # Logging is gated *before* any synchronisation. Every
                # float()/.item() on a CUDA tensor blocks the CPU until the GPU
                # queue drains, and each one costs a pipeline refill. An earlier
                # version computed all metrics on every step and only gated the
                # CSV write, which cost ~6 explicit syncs plus -- because
                # `max()` over a generator of CUDA tensors syncs on EVERY
                # comparison -- roughly one more sync per parameter tensor.
                # On ResNet-18 that is ~60 syncs/step, which is the likely
                # reason this trainer ran ~2x slower than a metrics-free loop.
                # Now: zero syncs on non-logged steps.
                # ------------------------------------------------------------
                should_log = (global_step % log_every == 0)

                if should_log:
                    loss_value = float(loss.detach())          # 1 sync

                    # one reduction -> one sync, instead of one per tensor
                    grads = [q.grad.detach() for q in model.parameters()
                             if q.grad is not None]
                    grad_norm = float(torch.sqrt(
                        sum((g * g).sum() for g in grads)))    # 1 sync

                    # The LR actually used for THIS step. Reading group['lr'] is
                    # wrong for optimizers that carry the schedule internally
                    # (DualScheduledSGD never touches group['lr']; it holds
                    # lr_max forever), which made internal-arm logs read 0.8.
                    if hasattr(opt, "effective_lr"):
                        sc = (opt.state.get(next(iter(opt.state)), {})
                              .get("step_count", 0)) if opt.state else 0
                        lr_now = float(opt.effective_lr(sc))   # pure python
                    elif sched is not None and \
                            getattr(sched, "last_lr", None) is not None and \
                            getattr(sched, "_applied", False):
                        lr_now = float(sched.last_lr)          # pure python
                    else:
                        lr_now = float(opt.param_groups[0]["lr"])

                    with torch.no_grad():
                        ps = [q.detach() for q in model.parameters()]
                        wnorm = float(torch.sqrt(
                            sum((q * q).sum() for q in ps)))    # 1 sync
                        pmax = float(torch.stack(
                            [q.abs().max() for q in ps]).max())  # 1 sync
                        acc = float((out.argmax(1) == y).float().mean())  # 1 sync

                    csv_buffer.append([global_step, epoch, step_in_epoch,
                                       lr_now, loss_value, acc, grad_norm,
                                       wnorm, pmax])
                    rows.append({"loss": loss_value, "acc": acc})

                    if len(csv_buffer) >= 200:
                        w.writerows(csv_buffer)
                        csv_buffer.clear()

                train_step(opt, sched, loss)   # ordering handled in one place
                global_step += 1
            if csv_buffer:            # flush leftovers at the epoch boundary
                w.writerows(csv_buffer)
                csv_buffer.clear()

            val_loss, val_acc = eval_fn(model, None)
            val_hist.append({"epoch": epoch + 1, "val_loss": val_loss,
                             "val_acc": val_acc, "wall": time.time() - t0})
            print(f"[{arm_name} seed{seed}] epoch {epoch+1}/{epochs} "
                  f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} "
                  f"({time.time()-t0:.0f}s)")

    summary = summarise(rows)
    summary["val_history"] = val_hist
    summary["epoch_to_90"] = next((h["epoch"] for h in val_hist
                                   if h["val_acc"] >= 0.90), None)
    summary["wall_to_90"] = next((round(h["wall"], 1) for h in val_hist
                                  if h["val_acc"] >= 0.90), None)
    summary.update({"device": device,
                    "sec_per_epoch": round((time.time() - t0) / max(epochs, 1), 2),
                    "ms_per_step": round(
                        (time.time() - t0) / max(global_step, 1) * 1000, 2),
                    "arm": arm_name, "seed": seed, "epochs": epochs,
                    "steps_per_epoch": steps_per_epoch,
                    "wall_seconds": round(time.time() - t0, 1),
                    "final_val_acc": val_acc, "final_val_loss": val_loss})
    with open(os.path.join(out_dir, f"{arm_name}_seed{seed}.json"), "w") as fh:
        json.dump(summary, fh, indent=2)
    return summary
