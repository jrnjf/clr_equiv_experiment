# Experiment plan after E1

E1 (implementation equivalence) is **complete** and is a decision helper, not a
paper result. Outcome: `A_internal` == `B1_external` (relative 6e-6 float32,
6e-16 float64), and the scheduler adds no measurable optimizer overhead
(40.2 ms vs 42.0 ms/step). Decision: **isolate the scheduler** — use stock
`SGD` + `ExternalDualScheduler`.

Open decision E1 did not settle: which **weight-decay convention** to use.

## E2 — does the coupled weight decay matter (the ablation you asked for)

Three conventions, differing only in what multiplies the decay:

| arm | convention | per-step shrink |
|---|---|---|
| `B1_external_wd_eff` | `wd * lr_effective` (oscillating, annealed) | `wd * 0.2002` mean |
| `B2_external_wd_base` | `wd * lr_base` (constant lr_max) | `wd * 0.8` |
| `F_external_wd_none` | `wd` (no LR scaling) | `wd * 1.0` |

### The trap: equal coefficients are NOT equal decay

At `weight_decay = 5e-4` over a 200-epoch run (78,000 steps), cumulative shrink:

| convention | cumulative | vs effective |
|---|---|---|
| effective | 7.81 | 1x |
| base | 31.24 | **4.0x** |
| none | 39.00 | **5.0x** |

Run all three at 5e-4 and the "winner" is whichever decays least — a magnitude
result, not a mechanism result.

### Matched design

`weight_decay` in the config is the **effective** coefficient. The others are
rescaled by `schedules.matched_weight_decays` so mean per-step shrink matches:

| arm | weight_decay used | mean shrink/step |
|---|---|---|
| B1 effective | 5.0e-4 | 1.001e-4 |
| B2 base | **1.251e-4** | 1.001e-4 |
| F none | **1.001e-4** | 1.001e-4 |

Now the only difference is **constant vs pulsing decay** — the mechanism.

Verified: all three give cumulative shrink 7.8086 and final decay-only weight
0.00041 (identical). Tests in `tests/test_schedule_math.py::TestWeightDecayModes`.

> Caveat: matching holds over the **whole run**. Epoch 0 sits at ~2x the
> run-average effective LR (cosine not yet annealed), so a single early epoch
> shows "effective" decaying harder. Don't judge from one epoch.

## E2v2 — strength ladder (raise the target rather than lower B2)

### Pilot result at mean shrink 1.001e-4 (seed 0)

| arm | raw wd | peak acc | @ep | first 90 | wnorm@50 | wnorm@200 | final |
|---|---|---|---|---|---|---|---|
| B1 effective | 5.000e-4 | 0.9499 | 181 | 16 | 38.9 | 24.2 | 0.9492 |
| B2 base | 1.251e-4 | 0.9500 | 189 | 17 | 48.8 | 6.5 | 0.9406 |
| F none | 1.001e-4 | 0.9505 | 186 | 17 | 48.8 | 6.5 | 0.9406 |

**Peak accuracy and time-to-90 are indistinguishable.** Only the tail differs:
B2/F keep shrinking after the LR has annealed to ~1e-7, so weights collapse
110 -> 6.5 and the last-epoch accuracy drops ~1 pp. B1's decay goes to zero with
the LR, so it holds its peak.

### F is NOT identical to B2 (earlier claim was wrong)

An earlier version multiplied "base" decay by a **constant** `lr_max`, dropping
the cosine annealing. That made "base" a plain constant multiply, so once the
mean strength was matched it was the same run as "none" — which is why the pilot
reported 0.9406 for both.

The base LR does anneal: `lr_base(t) = lr_max * cosine(epoch)`, the peak of the
triangle envelope. With that restored, the three conventions are genuinely
distinct:

| mode | decay factor | profile |
|---|---|---|
| effective | `triangle(step) * cosine(epoch)` | oscillating **and** annealed |
| base | `lr_max * cosine(epoch)` | smooth **and** annealed |
| none | `1.0` | no LR scaling |

Per-step shrink at target 5e-4, epoch 0 / 100 / 199:

| mode | ep0 | ep100 | ep199 |
|---|---|---|---|
| effective | 2e-6 (rises to 2e-3 mid-epoch) | ~1e-6..1e-3 | ~3e-10 |
| base | 1.0e-3 | 5.0e-4 | 1.6e-7 |
| none | 5.0e-4 | 5.0e-4 | 5.0e-4 |

Cumulative shrink is matched across all three (7.809 at target 1e-4, 39.0 at
5e-4), asserted in `tests/test_schedule_math.py::TestWeightDecayModes` together
with a regression test that "base" must anneal (>100x larger early than late).

All three arms are therefore worth running; the informative contrast is now
**oscillating vs smooth** decay at equal annealing and equal strength.

### Coefficients when the LR-weighted average is held at 5e-4

| arm | wd at target 5e-4 | (target 1e-4) |
|---|---|---|
| B1 effective | **2.497e-3** | 5.000e-4 |
| B2 base | **1.2499e-3** | 2.502e-4 |
| F none | **5.000e-4** | 1.001e-4 |

Note B2's coefficient roughly doubled versus the earlier (buggy) version, because
the annealed base mean is ~0.4 rather than 0.8.

**Risk:** 5x the strength at which B2 already collapsed. Run ONE seed first:
`configs/e2_highwd.json`.

### Ladder

`configs/e2_midwd.json` (target 2.5e-4) gives an intermediate rung. Running
1e-4 -> 2.5e-4 -> 5e-4 at one seed per convention turns the ablation into a
**robustness ladder**: find the strength at which each convention breaks. That
is a stronger claim than a single point and costs the same 3 runs per rung.

### Metric policy

Report BOTH final-epoch and best-checkpoint accuracy. At 1e-4 the final-epoch
metric favours B1 by 0.86 pp while best-checkpoint is a tie
(0.9499 / 0.9500 / 0.9505). The defensible claim is **stop-point robustness**,
not a better optimum.

## Rest of the plan

**E3a — constant-LR sweep.** Real arm not implemented yet. LRs
{0.05, 0.10, **0.200**, **0.283**, 0.40, 0.80}. Gives the mean-matched and
RMS-matched controls that decide "oscillation vs lower average LR", plus the
divergence threshold at 0.8.

**E3b — AdamW + cosine.** Not implemented. Required to test the headline claim;
must run in *this* harness, not borrowed from the sgd_jumper CSV.

**E3c — tuned SGD + cosine.** Configs exist (`tt90_sgd_lr01` lr 0.1,
`tt90_sgd_lr02` lr 0.2, momentum 0.9). Tests whether the benchmark's
101-epoch-to-90% baseline was undertuned. Unify batch size with E1 (128,
390 steps) before comparing triangular arms.

**E4 — seeds 3 -> 5** for final numbers. At n=3 the schedule-vs-cosine
difference in E1 was t=-1.84, i.e. underpowered.

## Dropped (your decisions)

* period sweep (`{1,2,4,8,16}` epochs)
* amplitude sweep (amplitude changes only via annealing)
* shape sweep (linear vs cosine)

## Budget

200 epochs at ~17 s/epoch (after the persistent-worker fix) is ~0.95 h/run.

| | runs | time |
|---|---|---|
| E2 wd ablation: B2 + F, 3 seeds (B1 already run) | 6 | ~6 h |
| E3a constant sweep, 6 LRs x 3 seeds | 18 | ~17 h |
| E3b AdamW x 3 | 3 | ~3 h |
| E3c two SGD+cosine x 3 | 6 | ~6 h |

Order: **E2 first** (cheap, 6 h, and it decides what the method arm should be),
then E3a (the central confound), then E3b/E3c.
