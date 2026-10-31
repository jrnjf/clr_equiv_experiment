# Why the trainer was ~2x slower than jumper_experiment

**Resolved.** The cause was not logging and not the optimizer. It was
**DataLoader worker respawn charged to every epoch**, plus an expensive
per-epoch evaluation.

Numbers from the GPU box (NVIDIA RTX A4000, torch 2.14.1+cu130), batch 128,
390 steps/epoch, `scripts/profile_step.py`:

| segment | measured |
|---|---|
| DataLoader only (over a 40-batch window) | 264.5 ms/batch |
| device copy | 0.5 ms |
| forward | 12.0 ms |
| **forward + backward** | **39.3 ms** |
| + stock SGD step | 42.0 ms |
| + scheduler prep/advance | 40.2 ms (scheduler overhead: none) |
| **epoch evaluation** | **10,655 ms** |

## The account closes

A 36.7 s epoch decomposes as:

| | time |
|---|---|
| 390 steps x 40.2 ms | 15.7 s |
| epoch evaluation | ~10.7 s |
| DataLoader worker startup for the train pass | ~10.3 s |
| **total** | **~36.7 s** |

So **~21 s of every 36.7 s epoch was not training at all.** The actual compute
(40.2 ms/step) already matches the benchmark's ~45 ms/step — our loop was never
slow.

## Root cause: `persistent_workers=False`

By default, every `for batch in loader` creates a **new iterator, which spawns a
fresh worker pool**. The training loop iterates the loader once per epoch, so the
pool is rebuilt 200 times; the evaluation loop does the same. With
`num_workers=4` that startup cost is charged to every epoch, twice (train and
eval).

Measured directly (fake dataset, CPU, 40 batches/pass, same loader iterated 3x):

| config | pass 1 | pass 2 | pass 3 |
|---|---|---|---|
| workers=4, persistent=False | 1.42 s | 0.21 s | 0.20 s |
| workers=4, persistent=True | 0.19 s | **0.07 s** | 0.07 s |
| workers=0 | 0.03 s | 0.03 s | 0.03 s |

The 1.2 s gap is pure worker spawn — and that is on a trivial in-memory dataset.
With real augmentation and a larger dataset the same effect is what produced the
264 ms/batch and the 10.7 s evaluation.

**Caveat on the 264 ms figure:** it is inflated. The profiler measured a 40-batch
window, so process startup dominated it; steady-state loading is far cheaper. It
is still the right diagnostic, because the training loop pays that startup once
per epoch.

## Fixes

1. `persistent_workers=True` on the train and eval loaders (spawn once, reuse).
2. Eval batch size 512 -> 1024 (fewer batches, same result).
3. `torch.backends.cudnn.benchmark = True` — fixed 32x32 input shapes, so cuDNN
   autotuning picks faster conv algorithms for free.

Expected: ~36.7 s/epoch -> **~16-18 s/epoch**, i.e. a 200-epoch run in ~1 h
instead of 2.3 h. That also means the earlier E1 results (2.4 h/run) were
obtained with this overhead present; the *accuracy numbers are unaffected*, only
the wall-clock and `wall_to_90` figures are inflated.

## Corrections to earlier claims in this file

Two hypotheses were tested and **rejected**:

* "Per-step synchronisation is the ~2x bottleneck." **No.** `log_every=1` costs
  11% (105.1 vs 94.1 ms/step); `log_every=10` costs 2%.
* "The optimizer or loop structure is slower than theirs." **No.** Forward +
  backward is 39.3 ms and the scheduler adds ~0 ms; arm A was measured *faster*
  than their Jumper loop on CPU (0.49 vs 0.53 ms/step).

The gating fix from the first hypothesis is still retained: it is correct, it
removes an O(parameters) sync per step, and it makes `log_every` behave as
documented. It is just worth ~10%, not 2x.

## How to verify

```bash
python scripts/profile_step.py --config configs/cifar.json
```

With the fix, `DataLoader only` should drop to the steady-state cost and
`epoch eval` from ~10.7 s to well under 1 s. The reconciliation line should then
match a real epoch.
