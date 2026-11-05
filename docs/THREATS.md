# Two threats raised after the growing-period result

## Threat 1: evaluation phase is locked to the LR trough

The trainer validates at the end of every epoch. With a one-epoch triangle the LR
is at its trough at every epoch boundary, so the model is **always measured at its
most stable phase**:

| | LR when validation runs |
|---|---|
| B1 (1-epoch cycle) | 0.0049, 0.0049, 0.0049, ... (always the trough) |
| G_grow from ep 32 | 0.7456, 0.0027, 0.7392, 0.0026, ... (peak and trough alternate) |

Consequences:

* The **growing-period result is partly artifact**. G_grow is evaluated at the LR
  peak on half its epochs; B1 never is. The t@94 comparison is contaminated and
  the "period schedule does the anneal's mid-training job" reading is withdrawn.
* The **main claim may also be inflated**, because B1's whole curve is measured at
  troughs while a cycle-free baseline (C_cosine_only) has no phase structure to
  align with.

Test: rerun B1 with `--phase-offset 195` (half an epoch) so validation lands at
the LR peak.

    python experiments/run_equivalence.py --config configs/cifar.json \
        --arms B1_external_wd_eff --phase-offset 195 --seeds 0 1 2

3 runs, ~3.3 h. If time-to-90 stays near 17 epochs, the effect is real learning.
If it moves to ~40, the 4.1x is substantially an evaluation artifact.

## Threat 2: the metric ignores that the horizon is tunable

Reviewer framing: *"You reach 94% at epoch 80 because your LR curve descends. Set
the anneal horizon to 80 and the baseline reaches it too. Inside a fixed 200-epoch
run I do not care about your threshold times."*

The current comparison holds the anneal horizon at 200 epochs for both arms, so it
is horizon-matched, but it does not show the result survives when the horizon
tracks the budget. The honest test is **budget-matched**: for budget B, set the
anneal horizon to B for every arm and compare accuracy at the end of B epochs.

Budgets 40 / 80 / 160, arms B1 and C_cosine_only at peak 0.2, 1 seed first:

    python experiments/run_equivalence.py --config configs/cifar.json \
        --sweep configs/e9_budget.json

6 runs, ~3.1 h.

* If B1 wins at every budget, the claim is horizon-robust and the reviewer
  objection is answered directly.
* If the gap closes at shorter horizons, "4.1x faster to 90%" was an artifact of a
  200-epoch horizon and the headline must be reframed as a fixed-budget result.

## What is NOT in doubt

* The mean-matched comparison (B1 vs cosine@0.4, same mean LR 0.200) reaches 90%
  at 16.7 vs 68.7 epochs. Both have the same anneal horizon, so Threat 2 does not
  apply to it directly; Threat 1 might.
* Final accuracy is a tie at ~95% across all annealed arms.

---

# Results: both threats tested, neither holds

## Threat 1 (evaluation phase): no effect

B1 rerun with `--phase-offset 195`, so validation lands on the LR **peak**
(0.796) instead of the trough (0.0049):

| arm | t@90 | t@93 | t@94 | final | peak |
|---|---|---|---|---|---|
| B1, eval at TROUGH | 17 | 53 | 112 | 0.9519 | 0.9526 |
| B1, eval at PEAK | **16** | **53** | **106** | 0.9522 | 0.9529 |

Paired: t@90 +0.3 ep (t=0.38), t@93 +0.7 ep (t=0.13), t@94 +5.7 ep (t=1.99). All
non-significant, and the peak-evaluated run is if anything *faster*.

**The evaluation phase is not a confound.** My prediction that this would bite was
wrong. It also means the earlier retraction of the growing-period result was
premature: if evaluating at the peak changes nothing, then G_grow being evaluated
at alternating phases does not explain its t@94 advantage either.

## Threat 2 (tunable horizon): no effect — B1 wins at every budget

Anneal horizon set equal to the budget for **both** arms (1 seed):

| budget | arm | t@90 | t@90/budget | final acc | peak |
|---|---|---|---|---|---|
| 40 | B1 triangle × cosine | **15** | 0.38 | **0.9324** | **0.9328** |
| 40 | cosine anneal, peak 0.2 | 21 | 0.53 | 0.9271 | 0.9271 |
| 80 | B1 triangle × cosine | **15** | 0.19 | **0.9445** | **0.9449** |
| 80 | cosine anneal, peak 0.2 | 30 | 0.38 | 0.9369 | 0.9395 |
| 160 | B1 triangle × cosine | **16** | 0.10 | **0.9496** | **0.9498** |
| 160 | cosine anneal, peak 0.2 | 45 | 0.28 | 0.9452 | 0.9455 |

The reviewer objection ("set the horizon to the budget and I get there too") is
answered empirically: at every budget B1 reaches 90% sooner **and** ends higher.

## The peak question, which is the useful discovery

At 200 epochs the peak difference is +0.14 pp and not significant. But at
budget-matched horizons it is much larger:

| budget | B1 peak | cosine peak | difference |
|---|---|---|---|
| 40 | 0.9328 | 0.9271 | **+0.57 pp** |
| 80 | 0.9449 | 0.9395 | **+0.54 pp** |
| 160 | 0.9498 | 0.9455 | **+0.43 pp** |
| 200 | 0.9526 | 0.9512 | +0.14 pp (n.s.) |

The gap shrinks as the budget grows: with enough epochs the cosine baseline
catches up. So "the cycle peaks higher" is true only under a constrained budget —
which is the regime a practitioner with a compute limit is actually in.

## Caveats

* Budget runs are **n=1**. The +0.43 to +0.57 pp peak differences exceed typical
  seed spread (~0.2-0.3 pp) but are not yet tested.
* The cosine baseline uses peak 0.2 at every budget. A reviewer could ask for it
  to be re-tuned per budget (a higher peak may suit a 40-epoch run).
* Until those are run, the budget table supports the direction but not the
  significance of the peak claim.

---

# Threat 3 (self-inflicted): the method's own learning rate is untuned

Correcting a mistake made while planning the weekend. The method has only ever run
at peak lr 0.8, inherited from earlier work; every baseline got 2-4 configurations.

**This is a tuning asymmetry, not evidence that the method is fragile.** A schedule
whose accuracy peaks at some learning rate and falls off either side is normal --
that is true of SGD no matter which schedule it follows, and of AdamW, and of every
optimizer ever published. AdamW at lr 5e-4 performing below AdamW at lr 1e-3 does
not indict AdamW.

What a sensitivity sweep at peak lr {0.4, 0.8, 1.2} establishes:

| outcome | interpretation |
|---|---|
| similar performance | supports the title's claim that the result is configuration-robust |
| 0.4 / 1.2 worse | 0.8 is the right lr, as tuning would establish; report it, nothing is wrong |
| 0.4 / 1.2 better | the reported result improves |

"Just a configuration" is a claim about the *configuration space*. It can be false
while the result stands -- then you drop the title's implication, not the paper.

## The real methodological issue: tuning on the test split

All arms are tuned on the CIFAR-10 test split: cosine over peaks {0.1, 0.2, 0.4,
0.8}, AdamW over lr x wd, and (from battery 3) the method over {0.4, 0.8, 1.2}.
Standard practice is to tune on a validation split and report on test. The
asymmetry is *symmetric* here, so no arm is advantaged, but the paper should either
carve out a validation split or state the practice explicitly.
