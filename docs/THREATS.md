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
