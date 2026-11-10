# Weekend results (power loss during battery 2)

## What completed

| battery | status | runs |
|---|---|---|
| 0_momentum | **never run** (skipped with `--from 1`) | 0 / 9 |
| 1_adamw | **complete** | 12 / 12 |
| 2_cifar100 | **partial** — all B1 + both cosine baselines; AdamW only seed 0, and at the *weaker* lr1e-3/wd0.01 | 10 / 12 |
| 3_method_lr | not run | 0 / 6 |
| 4_step_decay | not run | 0 / 3 |
| 5_budget_seeds | still n=1 per budget | 0 / 12 |
| 6_headline_seeds | still n=3 | 0 / 4 |

All result JSONs verified intact: every `val_history` length matches its epoch
budget. The power loss stopped the run; it did not truncate data.

## Battery 1: AdamW tuning — claim survives, halved

| AdamW config | final | peak |
|---|---|---|
| lr 3e-4 wd 0.01 (earlier) | 0.9343 | 0.9353 |
| lr 1e-3 wd 0.01 | 0.9395 | 0.9402 |
| lr 1e-3 wd 0.05 | 0.9404 | 0.9411 |
| lr 2e-3 wd 0.01 | 0.9407 | 0.9416 |
| **lr 2e-3 wd 0.05 (best)** | **0.9456** | **0.9467** |

B1 final 0.9519 vs best AdamW 0.9456:

* **+0.63 pp**, paired t = +6.37, B1 wins 3/3 seeds.
* Against the previously reported untuned config it was +1.14 pp, so proper tuning
  **halved the margin but did not remove it**.

Caveat: both lr and wd improve monotonically across the grid (1e-3 -> 2e-3,
0.01 -> 0.05), so the optimum may lie beyond the grid. A reviewer can ask for
lr 4e-3 / wd 0.1. Expected effect: at most another ~0.2-0.3 pp, which would leave
the claim positive but thin.

## Battery 2: CIFAR-100 — the speed result replicates, the accuracy tie holds

CIFAR-100 tops out near 78%, so the paper's 90% threshold does not exist on this
dataset. Thresholds are reported at 65 / 70 / 75%.

| arm | n | final | t@65 | t@70 | t@75 |
|---|---|---|---|---|---|
| **B1 triangle x cosine** | 3 | 0.7761 | **12** | **21** | 131 |
| C cosine, peak 0.4 (mean-matched) | 3 | 0.7780 | 68 | 112 | 130 |
| C cosine, peak 0.2 | 3 | 0.7695 | 40 | 90 | 112 |
| AdamW lr1e-3 wd0.01 (undertuned) | 1 | 0.7283 | 17 | 86 | never |

Paired B1 vs the mean-matched baseline:

| metric | B1 | C@0.4 | diff | t | |
|---|---|---|---|---|---|
| final | 0.7784/0.7752/0.7746 | 0.7784/0.7797/0.7760 | **-0.20 pp** | -1.48 | n.s. |
| t@65 | 12/13/12 | 67/65/71 | -55.3 ep | **-27.3** | **5.5x** |
| t@70 | 22/20/21 | 114/109/114 | -91.3 ep | **-76.0** | **5.3x** |
| t@75 | 133/130/129 | 132/130/130 | 0.0 ep | 0.00 | 1.0x |

1. **The speed advantage replicates and strengthens**: 5.5x at 65% and 5.3x at 70%,
   all three seeds, against the same mean-matched control as CIFAR-10.
2. **The accuracy tie also replicates**: -0.20 pp, n.s. The mean-matched cosine is
   marginally ahead, so there is no final-accuracy claim on either dataset.
3. **The threshold dependence replicates**: by 75% the two arms are identical
   (131 vs 130 epochs). The advantage lives in early-to-mid training on both
   datasets, which makes it a property of the schedule rather than of one
   benchmark.

## What this does to the paper

* The **4.1x / 5.5x speed claim** now rests on two datasets, both against a
  mean-matched control, with identical threshold behaviour. This is the load
  bearing result and it held up.
* The **AdamW claim** survives at +0.63 pp but should be stated that way, not as
  +1.14 pp.
* The **final-accuracy claim** does not exist on either dataset. The abstract must
  not imply one.

## Still missing (before submission)

1. **Momentum baseline** (7 h) — never run. The single most predictable reviewer
   objection: every arm has momentum 0.
2. **Method peak-lr sensitivity** (7 h) — the method has only run at 0.8.
3. **CIFAR-100 AdamW at the best config** (3 runs, ~3.5 h) — the current CIFAR-100
   AdamW point is undertuned and would be an easy target.
4. Step decay (4 h), budget seeds (6 h), headline seeds (4 h) — stretch.
