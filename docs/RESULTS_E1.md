# E1 results — implementation equivalence, and what the CIFAR run shows

Run from the pulled repo: 3 seeds, CIFAR-10 / ResNet-18 / 30 epochs / batch 250
(= 180 steps per epoch); tiny backend 2 seeds.

## 1. The gate: A vs B1

| precision | max \|A - B1\| | relative |
|---|---|---|
| float64 (unit test) | 6.1e-16 | machine zero |
| float32 (tiny backend) | 1.4e-06 | 6.3e-06 |

The internal optimizer and the stock-SGD + external-scheduler composition are
algebraically identical. **The implementation is correct**; so the A-vs-B1
difference on CIFAR is a reproducibility check, not a comparison of methods.

CIFAR per-seed values do differ slightly (A 0.9071 sd 0.0049 vs B1 0.9087
sd 0.0005). That is seed/run noise, not behaviour: float64 agreement is 6e-16
and both paths share the same schedule object.

## 2. CIFAR-10 accuracy (validation, 5000 held out)

| arm | seed 0 | seed 1 | seed 2 | mean | sd |
|---|---|---|---|---|---|
| A_internal | 0.9084 | 0.9016 | 0.9112 | 0.9071 | 0.0049 |
| B1_external_wd_eff | 0.9086 | 0.9082 | 0.9092 | 0.9087 | 0.0005 |
| B2_external_wd_base | 0.8946 | 0.8940 | 0.8992 | 0.8959 | 0.0028 |
| C_cosine_only | 0.9118 | 0.9072 | 0.9106 | 0.9099 | 0.0024 |

Paired by seed (shared init and data order), n=3, df=2, |t| >= 4.30 for p<0.05:

| comparison | mean diff | t | seeds won |
|---|---|---|---|
| B1 - A (equivalence) | +0.16 pp | 0.62 | 2/3 |
| **C - B1 (oscillation vs cosine)** | **+0.12 pp** | **0.99** | 2/3 |
| **B1 - B2 (weight-decay coupling)** | **+1.27 pp** | **+9.31** | **3/3** |

### Reading

1. **The triangular component adds nothing measurable here.** Oscillation vs pure
   cosine is +0.12 pp at t ~ 1 — indistinguishable at n=3, and ~10x smaller than
   the sd.
2. **The weight-decay coupling matters more than the schedule.** B2 is 1.27 pp
   *worse*, consistently across all 3 seeds (t=9.3). It is the largest and only
   statistically resolvable effect in the experiment.
3. Caveat on (2): A/B1 couple decay to the *effective* LR, so mean decay strength
   is ~5.5x smaller than B2's (mean eff. LR 0.222 vs base 0.8). **B1 is not
   "better", it is "less decayed".** Before claiming the coupling helps, run an
   arm whose `weight_decay` is scaled so effective decay matches B2. Until then
   read (2) as "the decay path is confounded with decay strength".

## 3. Validation loss disagrees with accuracy

| arm | mean val acc | mean val loss |
|---|---|---|
| B1 | 0.9087 | 0.3035 |
| C | 0.9099 | 0.3186 |

C is slightly more accurate but has clearly worse loss, i.e. worse calibration
(more confident when wrong). Worth one sentence in the paper; do not report loss
and accuracy as if they rank the arms identically.

## 4. What this run does NOT establish

* **30 epochs only.** CIFAR schedules are usually evaluated at 200 epochs; short
  runs are exactly where a high average LR can look fine. Re-run at 100-200
  epochs before concluding.
* **No constant-LR control.** Mean effective LR is 0.222 (RMS 0.245). Without a
  constant-LR arm matched on mean and on RMS, "oscillation vs cosine" is
  confounded with "which LR trajectory you happened to use".
* **No test-set numbers.** All values above are the 5000-sample validation split.
* n=3: the equivalence and coupling results clear the significance bar; the
  schedule comparison does not.

## 5. Logging bug in this run's CSVs

In `A_internal` the `lr` column reads a constant **0.8** for all 5400 steps. That
is a logging artifact, not the schedule: `DualScheduledSGD` carries the schedule
internally and never writes `group['lr']`, which stays at `lr_max`. The schedule
*was* applied (verified: effective LR 0.08, 0.088, 0.096, ...).

Fixed in `src/trainer.py` (now uses `opt.effective_lr(step_count)`), with a
regression test. **Regenerate any figures that use the `lr` column for internal
arms from this run** — that column is wrong.
