# Handover addendum — after the supervisor review

Supersedes parts of `docs/HANDOVER.md`. Read this first. The original handover
describes the pre-review paper; three of its claims are now **changed or
contradicted**, and one critical experiment is still outstanding.

Read `docs/SUPERVISOR_RESULTS.md` for the numbers behind everything here.

---

## 1. What changed, in one table

| draft claim | status after the revision | action |
|---|---|---|
| "short-period / one-epoch cycle" is the contribution | **REFUTED** — the period is not special; longer periods are better at 93-94% | rewrite the framing and title |
| "decay must track the learning rate" (§5.8) | **REFUTED as stated** — LR-independent decay at matched strength reaches 94.1% | narrow the claim |
| "4.1x / 6.5x faster" | **holds** — B1 now has 5 seeds, 16.2 ep to 90% | restate with new n |
| "comparable peak accuracy to the standard recipe" | **holds** | keep |
| "no accuracy advantage over tuned cosine" | **holds** | keep |
| "tuned AdamW" | **too strong** | "best tested AdamW configuration" |
| "hypotheses rejected" | **too strong** | "not supported under the tested control" |
| test-set tuning | **still present** — battery a did not run | **must be run before submission** |

## 2. The period is not special — this is the big rewrite

| period (epochs) | final | t@90 | t@94 |
|---|---|---|---|
| 0.25 | 0.9521 | 23 | 147 |
| 0.5 | 0.9510 | 19 | 134 |
| **1.0 (submitted config)** | 0.9514 | 16 | 111 |
| 2.0 | 0.9515 | 15 | 88 |
| 4.0 | 0.9522 | 15 | 72 |
| 8.0 | 0.9514 | 16 | 67 |

Final accuracy is flat across the whole sweep. t@90 is flat. t@94 *improves* with a
longer period.

The submitted title and abstract frame the contribution as an ultra-short cycle. That
framing is gone. The honest replacement is the opposite — and it is a better paper:

> the effect does not depend on finding one particular configuration; a broad family
> of cycle periods and peak learning rates produces it.

That is what "just a configuration" should mean, and now it is measured rather than
asserted. The title should lose any "short-period" or "one-epoch" specificity.

## 3. The decay claim must be narrowed

| condition | final | peak |
|---|---|---|
| A LR-coupled, oscillating (the method) | 0.9514 | 0.9522 |
| B LR-coupled, smooth | 0.9514 | 0.9523 |
| **C LR-independent, 1e-4, strength-matched** | **0.9412** | 0.9515 |
| D LR-independent, 5e-4, 5x stronger | **0.1045** | 0.9364 |

The submitted text says decay *must* track the learning rate, based on D's collapse.
Condition C shows that at the same mean strength, an LR-independent coefficient still
reaches 94.1% — it does not collapse. What collapses is 5x the strength.

Rewrite §5.8 as: **excessive decay destroys training regardless of profile; coupling
decay to the learning rate bounds it. At matched strength the coupled profile is
worth about 1 pp at the end (95.1% vs 94.1%).** Do not claim that LR-independence
per se causes collapse — that is refuted by C.

## 4. Peak-LR robustness now earns the framing

Final accuracy 94.89-95.24% and t@90 16-19 epochs across peak lr {0.4, 0.6, 0.8, 1.0,
1.2}. A reviewer asked "why 0.8?" — the answer is now measured: it does not matter.
This is a *strength*; present it that way.

## 5. Statistics: the paper must change how it reports significance

From `docs/review_response/P0_statistics.txt`:

* Every key comparison needs **per-seed differences and 95% CIs**, not just t.
* At n=3 the **exact paired permutation test cannot go below p = 0.25** (8 sign
  patterns). Several p-values in the submitted version are not meaningful as exact
  tests. Report "all 5 seeds agree" and the effect size instead.
* The reviewer's power correction is right: **d/sigma = 3.26** for 80% power at n=3,
  so a final-accuracy difference below 0.90 pp was never resolvable. State that the
  accuracy comparisons are *underpowered*, not that they are equal.
* Weight decay wording: `sum(eta)` is matched to 0.1% but `sum(eta^2)` differs by
  1.335x, so "matched mean learning rate" is defensible and "matched optimisation
  dynamics" is not.

## 6. Wording changes the review demands

* "Rejected" -> "not supported under the tested mean-matched control".
* "the effect requires momentum-free SGD" -> "the effect was observed with
  momentum-free SGD but not with momentum 0.9 under the tested configuration"
  (intermediate momentum values were not tested).
* "tuned AdamW" -> "best tested AdamW configuration".
* Add batch handling to §4: `drop_last=True`, 390 batches = **49,920 examples per
  epoch** of 50,000; 80 examples skipped per epoch, a different 80 each epoch.
  (With the validation split: 45,000/128 -> **351 steps**, 45,120 examples.)
* Add the reproducibility block: repository URL, commit hash, Python 3.14,
  PyTorch 2.14.1+cu130, torchvision 0.29.1, one RTX A4000, ~20 s/epoch, and the
  scripts that regenerate every table and figure.

## 7. Proposed new abstract (reflecting the sweep)

> **Just a configuration: a triangular cycle makes momentum-free SGD reach target
> accuracies several times sooner**
>
> We study a learning-rate schedule that composes two established components — a
> triangular cycle (Smith's cyclical policy) multiplied by a cosine anneal — and we
> make no novelty claim for the composition. On CIFAR-10 and CIFAR-100 with
> ResNet-18, momentum-free SGD with this schedule reaches 90% test accuracy in 16.2
> epochs against 107.7 for SGD with momentum 0.9 and cosine annealing, a 6.5x
> reduction, at comparable peak accuracy (95.1% vs 95.4%). On CIFAR-100 it reaches
> 65% in 12.3 epochs against 67.7, a 5.5x reduction. The effect does not require a
> specific configuration: final accuracy is flat (95.1-95.2%) for cycle periods from
> 0.25 to 8 epochs and peak learning rates from 0.4 to 1.2, and it survives
> randomising the cycle phase relative to epoch boundaries. It does not survive
> momentum: the same schedule at momentum 0.9 reaches 90% only at epoch 159 and ends
> 2.8 points lower. The contribution is a characterisation of this regime and of its
> boundaries, not a new optimizer.

## 8. THE OUTSTANDING ITEM

Battery **a**, the validation/test separation, did not run: it was blocked by an
obsolete even-steps guard (now fixed). The review calls it **Critical** and
**mandatory**. One command:

```bash
bash scripts/supervisor_campaign.sh --only a        # ~20 h
```

Until it completes, the paper cannot claim that hyperparameters were selected
without touching the test split. If battery a shows that validation-based selection
picks different hyperparameters than test-based selection did, the headline numbers
change and this handover must be revised again.

## 9. Section-by-section edit list for the draft

| draft location | edit |
|---|---|
| Title | remove "short-period"/"one-epoch" specificity |
| Abstract | replace with §7 above |
| §1 Introduction | the falsification table stays; add period sweep as a sixth test; soften "rejected" |
| §2 Related work | unchanged |
| §3 Method | add that the period is a free parameter (already true in code) |
| §4 Setup | add batch handling, seeds per arm, and the reproducibility block |
| §5.1 speed | B1 n=5, 16.2 ep; add CIs |
| §5.4 momentum | unchanged numbers; soften "requires" |
| §5.6 CIFAR-100 | unchanged |
| §5.7 AdamW | rename "tuned" -> "best tested"; add the CIFAR-100 tuning caveat |
| **§5.8 decay** | **rewrite per §3 above — the current claim is refuted** |
| **new §5.9 period sweep** | **add; it is the reviewer's top scientific request** |
| **new §5.10 peak-lr** | **add; it answers "why 0.8"** |
| **new §5.11 phase** | **add; one paragraph, answers 6.1** |
| §6 Discussion | reframe: the effect is configuration-robust, and needs a cycle but not a particular one |
| §7 Limitations | add: validation split pending (or resolved); EoS still hypothetical |
| §8 Reproducibility | add URL/commit/env |
| References | fix [8], [10] author metadata per the review |

## 10. What not to change

* Do not claim a novel schedule.
* Do not claim final-accuracy superiority.
* Keep the negative momentum result.
* Keep the boundary map — but the boundary moved: the period is *not* a boundary,
  the peak LR is not a boundary either, and the decay *strength* is.
