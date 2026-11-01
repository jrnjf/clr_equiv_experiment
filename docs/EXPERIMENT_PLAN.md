# Experiment plan after E1

> **Prior art.** The schedule is Smith's triangular cyclical LR
> (arXiv:1506.01186) multiplied by a cosine anneal (arXiv:1608.03983). Both are
> established, and combining them is a two-line `LambdaLR`; we claim no novelty
> for the schedule. What is being tested is the regime (peak lr 0.8, no momentum,
> anneal to ~1e-4 of peak) and the time-to-accuracy behaviour.
>
> **Design simplified (current).** The custom scheduler and all weight-decay
> instrumentation are gone. The method is now stock PyTorch:
>
> ```python
> opt   = torch.optim.SGD(params, lr=0.8, weight_decay=5e-4, momentum=0.0)
> sched = torch.optim.lr_scheduler.LambdaLR(
>             opt, lambda t: triangle(t % steps_per_epoch)
>                            * cosine_scaler(t / steps_per_epoch))
> ...
> opt.step(); sched.step()
> ```
>
> No param loop, no manual weight decay: `torch.optim.SGD` already scales decay
> by the current lr (sgd.py:355 then :379), and LambdaLR keeps that lr current.
> Arms are now `B1_external_wd_eff` (the method) and `C_cosine_only` (baseline);
> `A_internal` remains only so equivalence stays testable. B2/F are removed —
> the weight-decay-convention ablation is closed (result: LR-coupled decay is
> torch's default; the non-standard no-LR variant collapses).
>
> One consequence: the old even-`steps_per_epoch` requirement is gone. It
> existed only because CyclicLR could not reproduce a fractional half-cycle;
> LambdaLR can.

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

## E2 — weight-decay convention (simple version)

One literal `weight_decay` from the config, applied to all three arms. They differ
only in **what the coefficient multiplies**:

| arm | decay factor | profile |
|---|---|---|
| `B1_external_wd_eff` | `triangle(step) * cosine(epoch)` | oscillating **and** annealed |
| `B2_external_wd_base` | `lr_max * cosine(epoch)` | smooth **and** annealed |
| `F_external_wd_none` | `1.0` | no LR scaling |

At `weight_decay = 5e-4` the mean per-step shrink is therefore **1.0e-4 / 2.0e-4 /
5.0e-4** — the conventions are 1x / 2x / 5x different in strength. That is
accepted deliberately: this is the literal "set weight_decay=5e-4 and see what
each convention does" comparison.

An automatic strength-matching layer (rescaled coefficients, `target_mean_shrink`)
was built and then removed as over-engineering. It is in git history if ever
needed.

### Fixed: `base` ignored the cosine

`base` used to multiply by a **constant** `lr_max`, dropping the annealing
entirely. That made it a plain constant multiply, so at matched strength it was
identical to `none` — which is why the first pilot reported 0.9406 twice with
wnorm 6.54 vs 6.52. Now `base` decays against `lr_max * cosine(epoch)`, from
1.0e-3 per step at epoch 0 down to 1.6e-7 at epoch 199. Regression test:
`tests/test_schedule_math.py::TestWeightDecayModes`.

### Pilot at weight_decay = 5e-4, seed 0 (B1 valid; B2/F invalid — they ran the bug)

| arm | peak acc | @ep | first 90 | wnorm@50 | wnorm@200 | final |
|---|---|---|---|---|---|---|
| B1 effective | 0.9499 | 181 | 16 | 38.9 | 24.2 | 0.9492 |
| B2 base (buggy, = none) | 0.9500 | 189 | 17 | 48.8 | 6.5 | 0.9406 |
| F none | 0.9505 | 186 | 17 | 48.8 | 6.5 | 0.9406 |

B1 is still valid. **B2 and F must be re-run** with the annealing fix.

What the valid comparisons showed: peak accuracy and time-to-90 were
indistinguishable across all three; only the tail differed. Constant decay kept
shrinking after the LR annealed to ~1e-7, collapsing weights 110 -> 6.5 and
costing ~1 pp at the last epoch. That is why final-epoch and best-checkpoint
accuracy disagree, and both must be reported.

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
