# Supervisor revision: results

Campaign `scripts/supervisor_campaign.sh`. Battery **a** (the P0 validation-split
experiment) did **not** run — it was blocked by an obsolete even-steps guard, now
removed. Everything else completed. Re-run with:

```bash
bash scripts/supervisor_campaign.sh --only a      # ~20 h
```

All results below are complete unless marked.

---

## B. Cycle-period sweep (reviewer 6.2) — THE ONE-EPOCH PERIOD IS NOT SPECIAL

| period (epochs) | n | final | peak | t@90 | t@93 | t@94 | t@95 |
|---|---|---|---|---|---|---|---|
| 0.25 | 3 | 0.9521 | 0.9525 | 23 | 93 | 147 | 172 |
| 0.5 | 3 | 0.9510 | 0.9529 | 19 | 69 | 134 | 172 |
| **1.0 (the paper's config)** | 5 | 0.9514 | 0.9522 | **16** | **55** | **111** | 168 |
| 2.0 | 3 | 0.9515 | 0.9521 | 15 | 41 | 88 | 168 |
| 4.0 | 3 | 0.9522 | 0.9533 | 15 | 40 | 72 | 165 |
| 8.0 | 3 | 0.9514 | 0.9522 | 16 | 32 | **67** | 168 |

Two separate facts:

1. **Final accuracy is flat** across every period (95.10-95.22%). The benefit does
   not depend on the period at all for the final level.
2. **Longer periods are better at high thresholds.** t@94 falls monotonically
   147 -> 134 -> 111 -> 88 -> 72 -> 67 as the period grows from 0.25 to 8 epochs.
   t@90 is flat (15-23).

**Consequence for the paper: the "short-period" premise is wrong.** The submitted
title and framing attribute the effect to an ultra-short cycle. The sweep shows the
effect comes from having *a* cycle at all, and that if anything a longer period is
better for reaching 93-94%. The paper must either drop the "one epoch" specificity
or present it as one point on a flat-to-improving curve. This is a *stronger*
robustness result but a *weaker* novelty claim — and the review asked exactly this.

## C. Peak-LR sensitivity (reviewer 6.3) — ROBUST FROM 0.4 TO 1.2

| peak lr | n | final | t@90 | t@93 | t@94 |
|---|---|---|---|---|---|
| 0.4 | 3 | 0.9503 | 19 | 54 | 124 |
| 0.6 | 3 | 0.9489 | 17 | 51 | 114 |
| **0.8 (paper)** | 5 | 0.9514 | 16 | 55 | 111 |
| 1.0 | 3 | 0.9524 | 17 | 61 | 123 |
| 1.2 | 3 | 0.9523 | 18 | 65 | 120 |

Final accuracy varies by 0.35 pp and t@90 by 3 epochs across a **3x range of peak
learning rate**. There is no knife-edge: 0.8 was not a lucky pick, and the method
does not need tuning. This *earns* the "just a configuration" framing rather than
asserting it, and answers "why 0.8?" with "anywhere in 0.4-1.2 works".

## D. Decay strength vs decay profile (reviewer 4) — THE PAPER'S CLAIM IS WRONG

| condition | n | final | peak | t@90 | t@93 |
|---|---|---|---|---|---|
| A LR-coupled, oscillating (method) | 5 | 0.9514 | 0.9522 | 16 | 55 |
| B LR-coupled, smooth (mean-matched cosine) | 5 | 0.9514 | 0.9523 | 69 | 132 |
| **C LR-independent, 1e-4, strength-matched** | 3 | **0.9412** | 0.9515 | 17 | 52 |
| D LR-independent, 5e-4, strong | 3 | **0.1045** | 0.9364 | 21 | 169 |

**The collapse is a STRENGTH effect, not a profile effect.** At matched strength,
LR-independent decay trains fine: 94.12% final, reached 90% at epoch 17 — it is not
collapsing. At 5x the strength it collapses to chance (10.45%).

The submitted paper claims *"decay must track the learning rate"*. Condition C
refutes that as stated. The defensible claim is narrower:

> At the method's mean decay strength, an LR-independent coefficient reaches 94.1%
> against 95.1% for the LR-coupled version — about 1 pp worse at the end, because
> the decay keeps acting after the learning rate has annealed. At 5x that strength
> it collapses to chance. Excessive decay destroys training; the LR coupling is what
> bounds it.

## E. Cycle phase vs data order (reviewer 6.1) — NO ARTEFACT

| arm | n | final | t@90 | t@93 | t@94 |
|---|---|---|---|---|---|
| locked to epoch boundaries (default) | 5 | 0.9514 | 16 | 55 | 111 |
| random initial phase | 3 | 0.9520 | 17 | 52 | 119 |
| random phase every epoch | 3 | 0.9514 | 19 | 66 | 119 |

Breaking the phase lock changes nothing (final within 0.06 pp, t@90 within 3
epochs). The effect is not an artefact of the cycle being synchronised with data
shuffling. This fully answers the review's concern.

## F. Five seeds (reviewer 3.3)

B1 final accuracies over five seeds: 0.9532, 0.9504, 0.9520, 0.9500, 0.9513
(mean 0.9514, sd 0.13 pp). Tight enough that the paired conclusions hold, but see
`docs/review_response/P0_statistics.txt` for why n=3 could not resolve the accuracy
comparisons in the first place.

## P0 analyses already delivered (no GPU)

* `docs/review_response/P0_lr_math.txt` — the 0.25 vs 0.20 objection resolves: both
  are correct (ratio vs absolute), sum(eta) matches to 0.1%, **sum(eta^2) differs by
  1.335x** so "matched optimisation dynamics" may not be claimed.
* `docs/review_response/P0_statistics.txt` — the reviewer's power correction is
  confirmed: **d/sigma = 3.26** for 80% power at n=3; detectable difference 0.90 pp
  when the observed gap is 0.14 pp; the exact permutation test cannot go below
  p = 0.25 at n=3.

## A. Validation/test separation (reviewer 3.4) — 15 of 18 runs

Complete: B1, cosine@{0.2,0.4,0.8}, momentum-0.9 recipe. Missing: `AdamW` on the
split (seed 0 was killed at epoch 54 by SIGHUP; seeds 1-2 never ran).

45k train / 5k validation, 3 seeds. `val` = best validation accuracy, `test` =
final accuracy on the untouched 10k test split.

| arm | best val | final test | t@90 (val) | t@93 (val) |
|---|---|---|---|---|
| **B1 triangle x cosine** | 0.9521 | 0.9485 | **17.7** | **41.7** |
| **cosine + momentum 0.9, lr 0.1 (standard)** | 0.9542 | 0.9498 | 96.3 | 149.3 |
| cosine, peak 0.8 | 0.9557 | 0.9516 | 93.3 | 138.3 |
| cosine, peak 0.4 (mean-matched) | 0.9539 | 0.9500 | 60.0 | 118.3 |
| cosine, peak 0.2 | 0.9525 | 0.9456 | 36.7 | 106.3 |

### The headline speed claim survives, but the number changes

Against the standard momentum recipe: **t@90 17.7 vs 96.3 epochs = 5.45x**
(3/3 seeds, [18,18,17] vs [97,95,97]).

This must replace the 6.5x quoted in the submitted paper for this comparison, because
the split moves both arms in opposite directions: B1 slows from 16.2 to 17.7 epochs,
the momentum recipe accelerates from 105.8 to 96.3. The reduction is smaller but still
large and consistent.

Final test accuracy: B1 94.85% vs momentum 94.98% (a tie), matching the full-data
conclusion of no accuracy advantage.

### Selection stability: the answer to 3.4

| ranked by | order |
|---|---|
| validation | cosine@0.8 > momentum > cosine@0.4 > cosine@0.2 > B1 |
| test | cosine@0.8 > cosine@0.4 > momentum > B1 > cosine@0.2 |

The **winner is identical under both criteria** (cosine@0.8), which is the choice that
matters for selecting the cosine baseline. B1 ranks low on accuracy under both.
Intermediate orderings shift among arms separated by less than 0.15 pp, i.e. among
statistical ties. The honest claim: validation-based selection is stable for the
baseline that matters and arbitrary among near-ties.

Accuracies are 0.3-0.5 pp below the full-data runs, expected from 10% less training
data.

### Still outstanding

Only the AdamW arm on the split remains (3 runs, ~2.7 h). Resume with:

```bash
bash scripts/supervisor_campaign.sh --only a
```

It writes to `results_val/`, where `val_history` is validation accuracy on a held-out
5k slice of the training set and `final_test_acc` is the untouched test number.
Until it runs, §3.4 of the review is unanswered.
