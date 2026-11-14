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

## A. Validation/test separation (reviewer 3.4) — STILL OUTSTANDING

Critical and unrun. The guard that blocked it is fixed; one command on the GPU box:

```bash
bash scripts/supervisor_campaign.sh --only a
```

It writes to `results_val/`, where `val_history` is validation accuracy on a held-out
5k slice of the training set and `final_test_acc` is the untouched test number.
Until it runs, §3.4 of the review is unanswered.
