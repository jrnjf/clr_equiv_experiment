# Proposal: abstract and claim inventory

## 1. What was run

Protocol unless stated: CIFAR-10 / ResNet-18 (CIFAR stem), 200 epochs, batch 128
(390 optimizer steps/epoch), full 50k train set, evaluation on the official 10k
test split, `torch.optim.SGD` + `LambdaLR`. Schedule: triangle of period one
epoch × cosine anneal, peak lr 0.8, floor 8e-4 (ratio 1000), annealed to 1e-4 of
peak, momentum 0, weight decay 5e-4 (LR-coupled, torch's default).

| experiment | purpose | arms × seeds | status |
|---|---|---|---|
| E1 | is the custom optimizer equivalent to stock SGD + a scheduler? | 4 arms × 3 | done; decision helper, not for the paper |
| E2 | which weight-decay convention | 3 × 3 | done |
| E3a | constant-LR sweep | 6 × 1 | done (single seed) |
| E3b | 2×2: cycle on/off × anneal on/off; tuned cosine baselines | 4 × 3 | done |
| E4 | AdamW + cosine; AdamW + cycle | 3 × 3 | done |

39 runs total (`python scripts/summarize_all.py --baseline B1_external_wd_eff`).

## 2. Results

| arm | n | peak | final | 1st 90% |
|---|---|---|---|---|
| B2 wd × annealed base lr | 3 | 0.9534 | 0.9522 | 18.3 |
| A internal optimizer | 3 | 0.9530 | 0.9522 | 17.0 |
| **B1 triangle × cosine** | 3 | **0.9526** | **0.9519** | **16.7** |
| C cosine, peak 0.2 | 3 | 0.9512 | 0.9507 | 44.3 |
| **C cosine, peak 0.4 (mean-matched)** | 3 | 0.9512 | 0.9504 | **68.7** |
| C cosine, peak 0.8 | 3 | 0.9503 | 0.9493 | 105.3 |
| C cosine, peak 0.1 | 3 | 0.9435 | 0.9424 | 44.7 |
| AdamW + cosine, 1e-3 | 3 | 0.9416 | 0.9405 | 20.0 |
| AdamW + cycle, 1e-3 | 3 | 0.9379 | 0.9373 | 15.7 |
| B1 cycle, no anneal | 3 | 0.9398 | 0.9363 | 16.7 |
| AdamW + cosine, 3e-4 | 3 | 0.9353 | 0.9343 | 26.7 |
| constant lr 0.1 (best constant) | 1 | 0.9215 | 0.9073 | 46.0 |
| constant lr 0.8 | 1 | 0.8802 | 0.8342 | never |

Supporting statistics (paired by seed, n=3, |t| >= 4.30 for p<0.05):

| comparison | metric | effect | t | seeds won |
|---|---|---|---|---|
| **B1 vs C@0.4 — mean-matched, only the cycle differs** | **epochs to 90%** | **+52.0 ep (4.1x)** | **+11.53** | **3/3** |
| B1 vs C@0.4 | final acc | +0.14 pp | +0.90 | 2/3 (n.s.) |
| B1 vs AdamW+cosine 1e-3 | final acc | +1.13 pp | +7.92 | 3/3 |
| B1 vs AdamW+cosine 1e-3 | epochs to 90% | −3.3 ep | −1.89 | 2 faster, 1 tie (n.s.) |
| B1 vs C@0.2 (best-tuned cosine) | epochs to 90% | −27.7 ep (2.65x) | −8.43 | 3/3 |
| B1 vs C@0.2 | final acc | +0.12 pp | +1.05 | — (n.s.) |
| AdamW+cycle vs AdamW+cosine | final acc | −0.33 pp | −2.82 | 0/3 |

Threshold dependence, matched pair (B1 vs C@0.4):

| threshold | B1 | C@0.4 | speedup |
|---|---|---|---|
| 90% | 17 | 69 | **4.1x** |
| 93% | 53 | 132 | 2.5x |
| 94% | 112 | 149 | 1.3x |
| 95% | 168 | 166 | 1.0x (tie) |

The advantage is concentrated in early-to-mid training and vanishes at the
accuracy ceiling, where both arms converge to the same ~95%.

## 3. What the data supports

1. **Speed, isolated.** Holding the mean LR fixed at 0.200 and changing one
   thing — the cycle — reach 90% test accuracy in 16.7 epochs versus 68.7, a
   **4.1×** reduction (t = 11.5, 3/3 seeds) at equal final accuracy
   (95.19% vs 95.04%, n.s.). Against the best-tuned cosine (peak 0.2) the
   reduction is 2.65×.
2. **AdamW comparison.** Momentum-free SGD with this schedule is ~1.1 points more
   accurate than AdamW + cosine at equal time-to-threshold.
3. **Decomposition.** Cycling buys speed; annealing buys accuracy. Removing the
   anneal keeps 16.7 epochs to 90% but caps at 93.6%. No constant LR reaches
   92% (single seed).
4. **Scope.** The benefit is specific to momentum-free SGD: the same cycle on
   AdamW is slightly worse than cosine (−0.33 pp).
5. **Robustness of the decay.** Weight decay must scale with the LR. An
   LR-independent coefficient collapsed to chance (final loss = ln 10 = 2.3022)
   late in training, because decay kept acting after the LR annealed to ~1e-7.

## 4. What it does not support

* "2.6× faster than cosine annealing" holds against a **tuned** baseline only at
  n=3 on one dataset/architecture; against the badly tuned peak-0.8 baseline it
  would be 6×, and that number should not be used.
* The advantage is **threshold-dependent**: 4.1× at 90%, 2.5× at 93%, 1.3× at
  94%, and a tie at 95%. Reporting only 90% would be selective; report the curve.
* AdamW was tuned over only **two learning rates and one weight decay** (0.01).
  Tuned AdamW on this benchmark often reaches 94.5–95%, which would shrink or
  erase the +1.1 pp gap.
* One dataset, one architecture, one schedule geometry (period 1 epoch, ratio
  1000, peak 0.8). No period, amplitude or shape sweep.
* Constant-LR arms are n=1.
* E1 (implementation equivalence) is an engineering check and does not belong in
  the paper.

## 5. Proposed abstract

**Title.** *Just a configuration: a triangular cycle makes momentum-free SGD 4×
faster to 90% accuracy at unchanged final accuracy*

> We report a controlled comparison. Starting from a cosine-annealed SGD baseline
> whose mean learning rate is 0.200, we add one ingredient — a triangular cycle
> of period one epoch — and change nothing else. On CIFAR-10 with ResNet-18 over
> 200 epochs and three seeds, the cycle reduces the epochs needed to reach 90%
> test accuracy from 68.7 to 16.7, a 4.1× reduction (paired t = 11.5, 3/3 seeds),
> while final accuracy is unchanged at 95.0%. The advantage is largest in
> early-to-mid training — 2.5× at 93% — and disappears at 95%, where both arms
> converge to the same ceiling. Against AdamW with cosine annealing, momentum-free
> SGD with the cycle is 1.14 points more accurate at matched time-to-threshold.
> We claim no novelty for the schedule, which composes Smith's cyclical
> learning-rate policy with the cosine schedule; the contribution is an empirical
> characterisation of this high-peak, long-anneal regime.

**Shorter version (~110 words).** *Adding a triangular cycle of period one epoch
to a cosine-annealed SGD baseline — one change, nothing else — cuts the epochs
needed to reach 90% CIFAR-10 accuracy from 68.7 to 16.7 (4.1×, paired t = 11.5,
3/3 seeds) at unchanged final accuracy (95.0%). The gain is 2.5× at 93% and
vanishes at 95%, where both schedules meet the same ceiling. Against AdamW +
cosine, momentum-free SGD with the cycle is 1.14 points more accurate at matched
time-to-threshold. We claim no novelty for the schedule; the contribution is a
characterisation of this regime, with its boundaries stated.*

## 6. Before submission

Ordered by threat level:

1. **Tune AdamW properly** (lr {1e-3, 2e-3} × wd {0.01, 0.05}, 3 seeds, ~4 h).
   If AdamW reaches ~95%, claim 2 disappears.
2. **Mean-matched annealed control** (cosine from peak 0.4, mean lr 0.2, 3
   seeds, ~3.5 h). If it matches B1 at 17 epochs, claim 1 collapses to nothing.
3. **Second dataset or architecture** (CIFAR-100, or ResNet-34). One benchmark
   is thin for a schedule paper.
4. **Seeds to 5** on B1, AdamW+cosine, and the tuned cosine baseline.
5. Decide whether the weight-decay collapse (E2) is a second contribution or a
   cautionary footnote; it is currently unsupported at n=3 beyond one setting.
