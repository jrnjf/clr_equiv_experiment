> **CORRECTION (measured on GPU).** The hypothesis below — that per-step
> synchronisation accounts for the ~2x gap — is **wrong**. Measured on a CUDA
> box with `scripts/bench_logging.py`:
>
> | log_every | ms/step | s/epoch | vs none |
> |---|---|---|---|
> | 1 | 105.1 | 41.0 | 1.11x |
> | 10 | 95.4 | 37.2 | 1.01x |
> | 50 | 93.9 | 36.6 | 0.99x |
> | none | 94.5 | 36.9 | 1.00x |
>
> Logging costs **11% at log_every=1 and ~1% at 10**. The gating fix is still
> correct and worth keeping, but it explains ~4 s of a ~17 s/epoch gap.
>
> Critically, **with logging entirely disabled the loop still runs at
> 94.5 ms/step (36.9 s/epoch)**, i.e. ~1.9x the benchmark's 19.6 s/epoch. That
> residual is NOT logging and is not yet explained. Use
> `scripts/profile_step.py` to locate it.
>
> Also unresolved: the benchmark's 19.6 s/epoch was measured on someone else's
> hardware. Before assuming a code slowdown, re-run `jumper_expirement.py` on the
> same machine — if it also lands near 36 s/epoch, this was never a code gap.

# Why the trainer was ~2x slower than jumper_experiment

E1: 43.8 s/epoch (112 ms/step). Benchmark: 19.6 s/epoch (50 ms/step).
Same GPU class, same batch size 128, same 390 steps/epoch. So ~62 ms/step of
overhead had to be explained.

## The cost was synchronisation, not arithmetic

`jumper_expirement.py` logs only once per epoch. It has **no per-step metrics**,
so it has no per-step synchronisation. The trainer computed, on every step:

| operation | CUDA cost |
|---|---|
| `float(loss.detach())` | 1 sync |
| `grad_norm` via `sum(...)` then `float()` | 1 sync + ~60 kernel launches |
| `acc` via `.item()` | 1 sync |
| `wnorm` via `sum(...)` then `float()` | 1 sync + ~60 kernels |
| `pmax` via `max((p.abs().max() for p in params))` | **~1 sync per comparison** (≈60) |

A `float()`/`.item()` on a CUDA tensor blocks the CPU until the queue drains.
And `max()` over a generator of CUDA tensors is worse: Python's `max()` compares
each pair, and each comparison yields a 0-dim tensor whose `bool()` forces a
device→host sync. On ResNet-18 that is roughly **one sync per parameter tensor**.

Counting it up: ~6 explicit syncs, plus ~60 from `pmax` alone.

Measured arithmetic cost of the metric block on CPU: **~0.4 ms** for all three
reductions (60 tensors each). So the arithmetic was never the problem — at
112 vs 50 ms/step, ~62 ms/step cannot be explained by 0.4 ms of arithmetic. The
remaining time is the cost of repeatedly draining a GPU pipeline that could
otherwise have overlapped with the next forward pass.

**What this means:** Gemini's diagnosis was directionally right (per-step sync is
the culprit) but its stated reason was wrong ("computing norms is expensive" —
it is ~0.4 ms) and so was its third point (CSV I/O). The CSVs were 9.2 MB over
2.4 h ≈ 1 KB/s, and Python's `open()` is buffered, so `writerow` does not syscall
per row.

## The actual bug: `log_every` gated the wrong thing

The trainer already had a `log_every` knob, but it only guarded `w.writerow(...)`.
Every metric above still ran on every step regardless of the setting, so
`log_every=10` gave zero speedup — it only produced a smaller CSV.

Fixed: the whole metric block is now inside `if should_log:`. Non-logged steps do
**zero** synchronisation. The metrics themselves were also made sync-cheap:

| before | after |
|---|---|
| `max(p.abs().max() for p in params)` (~60 syncs) | `torch.stack([...]).max()` (1 sync) |
| `sum((p*p).sum() ...)` then `float()` | same, but only when logging |
| one `writerow` per logged step | buffered, `writerows` every 200 rows |

## Measured effect

CPU/tiny sanity check (no GPU sync at all, so this is a lower bound):

| log_every | ms/step | overhead |
|---|---|---|
| 1 | 4.8 | **2.11x** |
| 10 | 2.5 | 1.11x |
| 50 | 2.4 | 1.07x |
| none | 2.3 | 1.00x |

Results are unchanged: `final_val_acc` and the whole `val_history` are
bit-identical at `log_every=1` and `log_every=100000` (asserted in the test
suite).

## Verify on your GPU

```bash
python scripts/bench_logging.py --config configs/cifar.json
```

It runs a couple of epochs at several `log_every` values and prints ms/step plus
projected 200-epoch wall time. If the diagnosis is right, `log_every=1` should be
roughly 2x the `none` baseline and `log_every=10` close to it.

## Recommendation

* **Production runs (E2, 200 epochs):** `log_every: 10`. Per-step logs are only
  needed for debugging, and at log_every=10 a 200-epoch run writes ~7.8k rows
  instead of 78k.
* **Divergence checks:** `log_every: 1` costs ~2x. Cheaper alternative: log at 1
  for the first epoch only, or keep `log_every=10` — a persistent NaN is still
  caught, since gradients become NaN and stay NaN. The only thing sampling can
  miss is a *transient* spike.
* Do not set `log_every` very high and then trust `grad_norm` for fine-grained
  analysis; you would be looking at 1-in-N samples.
