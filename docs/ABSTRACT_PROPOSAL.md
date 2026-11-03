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
| C cosine, peak 0.8 | 3 | 0.9503 | 0.9493 | 105.3 |
| C cosine, peak 0.1 | 3 | 0.9435 | 0.9424 | 44.7 |
| AdamW + cosine, 1e-3 | 3 | 0.9416 | 0.9405 | 20.0 |
| AdamW + cycle, 1e-3 | 3 | 0.9379 | 0.9373 | 15.7 |
| B1 cycle, no anneal | 3 | 0.9398 | 0.9363 | 16.7 |
| AdamW + cosine, 3e-4 | 3 | 0.9353 | 0.9343 | 26.7 |
| constant lr 0.1 (best constant) | 1 | 0.9215 | 0.9073 | 46.0 |
| constant lr 0.8 | 1 | 0.8802 | 0.8342 | never |

Supporting statistics (paired by seed, n=3, |t| ≥ 4.30 for p<0.05):

| comparison | metric | effect | t | seeds won |
|---|---|---|---|---|
| B1 vs AdamW+cosine 1e-3 | final acc | **+1.13 pp** | +7.92 | 3/3 |
| B1 vs AdamW+cosine 1e-3 | epochs to 90% | −3.3 | −1.89 | 0/3 (n.s.) |
| B1 vs SGD+cosine peak 0.2 | final acc | +0.12 pp | +1.05 | — (n.s.) |
| B1 vs SGD+cosine peak 0.2 | epochs to 90% | −27.7 | **−8.43** | 3/3 |
| AdamW+cycle vs AdamW+cosine | final acc | −0.33 pp | −2.82 | 0/3 |
| B1 vs B1-no-anneal | final acc | +1.56 pp | — | 3/3 |

Independent implementations of the same schedule (A internal, B2 external)
reproduce the B1-vs-AdamW gap (+1.16 and +1.17 pp), so the headline does not
depend on one arm.

## 3. What the data supports

1. **Speed.** At equal final accuracy, the cycle reaches 90% test accuracy
   2.6× sooner than a tuned cosine anneal (16.7 vs 44.3 epochs).
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
* The mean learning rates are not matched (B1 ≈ 0.20, cosine@0.2 ≈ 0.10). A
  mean-matched annealed control (cosine from peak 0.4) was never run.
* AdamW was tuned over only **two learning rates and one weight decay** (0.01).
  Tuned AdamW on this benchmark often reaches 94.5–95%, which would shrink or
  erase the +1.1 pp gap.
* One dataset, one architecture, one schedule geometry (period 1 epoch, ratio
  1000, peak 0.8). No period, amplitude or shape sweep.
* Constant-LR arms are n=1.
* E1 (implementation equivalence) is an engineering check and does not belong in
  the paper.

## 5. Proposed abstract

**Title.** *Cycling for speed, annealing for accuracy: a temporary-annealed
triangular schedule for momentum-free SGD*

> We study a deliberately simple learning-rate schedule: a triangular cycle of
> period one epoch multiplied by a cosine anneal, stepped per batch, applied to
> momentum-free SGD at a peak learning rate of 0.8 with a floor 1000× lower.
> Neither component is new — the triangle is Smith's cyclical learning-rate
> policy and the anneal is the cosine schedule — and we make no novelty claim for
> the composition. We ask instead what each component contributes, and how the
> combination compares with standard recipes.
>
> On CIFAR-10 with ResNet-18 over 200 epochs and three seeds, we find that the
> cycle reaches 90% test accuracy in 17 epochs against 44 for a tuned cosine
> anneal, a 2.6× reduction, while final accuracy is statistically
> indistinguishable (95.19% vs 95.07%). Against AdamW with cosine annealing,
> momentum-free SGD with this schedule is 1.14 points more accurate at equal
> time-to-threshold (95.19% vs 94.05%). The two components prove separable:
> removing the anneal preserves the speed (16.7 epochs) but caps accuracy at
> 93.6%, whereas no constant learning rate we tested exceeded 92%. The benefit
> does not transfer to AdamW, where the same cycle is slightly worse than pure
> cosine annealing. We conclude that the cycle buys time-to-accuracy and the
> anneal buys the final level, and we report the regime's boundaries rather than
> claiming a new scheduler.

**Shorter version (~120 words).** *A triangular cycle of period one epoch
multiplied by a cosine anneal lets momentum-free SGD at peak lr 0.8 reach 90%
CIFAR-10 accuracy in 17 epochs versus 44 for a tuned cosine anneal, at equal
final accuracy (95.2% vs 95.1%), and 1.1 points above AdamW + cosine. The
components separate cleanly: cycling buys speed, annealing buys accuracy, and
neither alone suffices. We make no novelty claim for the schedule — both
components are established — and frame the contribution as a characterisation of
this high-peak, long-anneal regime, including where the benefit does not appear
(it does not help AdamW).*

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
