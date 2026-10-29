# Why the E1 numbers differ from the sgd_jumper results

Short answer: **E1 and the benchmark are five different experiments**, and the
dominant difference is the cosine horizon — not the scheduler.

Run `python scripts/why_different.py` to reproduce the table below.

## 1. Cosine horizon (dominant)

The schedule is `lr(step) = lr_max * cosine(epoch, T) * triangle(step_in_epoch)`.
`T` is the **annealing horizon**, so the same code with `epochs=30` and
`epochs=200` produces genuinely different learning-rate trajectories:

| epoch | peak LR at T=200 | peak LR at T=30 | ratio |
|---|---|---|---|
| 0 | 0.800 | 0.800 | 1.00x |
| 10 | 0.795 | 0.602 | 1.32x |
| 15 | 0.789 | 0.404 | **1.95x** |
| 20 | 0.781 | 0.206 | 3.79x |
| 25 | 0.770 | 0.061 | 12.6x |
| 29 | 0.760 | 0.010 | **74.7x** |

At the epoch where Jumper reaches 90% (epoch 15), E1 has already annealed the
peak to **0.404** while the benchmark would still be at **0.789**. E1 is not a
shortened version of the benchmark run; it is a **fast-annealing** schedule.

Consequence: a 30-epoch run is dominated by its final, tiny-LR phase, so
time-to-90% and final accuracy are both measured under a completely different LR
regime. **Any comparison with the 200-epoch benchmark must match `epochs`.**

## 2. Steps per epoch -> triangular period

| | batch | data | steps/epoch | triangle period |
|---|---|---|---|---|
| benchmark | 128 | full 50k | **390** | 390 steps, peak at 195 |
| E1 | 250 | 45k (5k held out) | **180** | 180 steps, peak at 90 |

The triangle completes one cycle per epoch in both, but the cycle contains a
different number of optimizer steps, so the *shape per step* differs. Both are
even, so both sample their peak (only half-way points differ).

## 3. Normalization

| | mean | std |
|---|---|---|
| benchmark | (0.4914, 0.4822, 0.4465) | **(0.2023, 0.1994, 0.2010)** |
| E1 | same | (0.2470, 0.2435, 0.2616) |

The benchmark uses the per-channel std of the [0,1] tensor (~0.20), not the
usual CIFAR-10 std (~0.247). Input scaling changes the effective sharpness of
the loss, hence which LRs are stable. Different optimum, different curves.

## 4. Evaluation set

* benchmark: full **10,000-image test** split, no validation holdout
* E1: **5,000 held out of train** -> validation accuracy; test never reported

So the two sets of numbers are not even measured on the same data.

## 5. Momentum / weight decay

* benchmark Jumper: momentum 0.0, wd 5e-4, decay coupled to the effective LR
* benchmark "SGD": momentum **0.9**, lr 0.1, wd 5e-4
* E1 A/B1/B2/C: momentum 0.0 (config), wd coupling varies by arm

The benchmark's SGD baseline is momentum 0.9 at lr 0.1 while Jumper is
momentum 0.0 at lr 0.8 — so its "15 vs 101 epochs to 90%" mixes a scheduler
change with a 6-8x LR change and a momentum change.

## What to do

Use the **`cifar_bench`** backend, which replicates the benchmark protocol
exactly (batch 128 on 50k with `drop_last` -> 390 steps, full 10k test eval,
their std). Then the scheduler is the only variable left.

```bash
# reproduce the Jumper curve with the jump removed
python experiments/run_equivalence.py --config configs/bench_replica.json

# the benchmark's own SGD baseline, to test whether 101 epochs is real
python experiments/run_equivalence.py --config configs/bench_sgd_lr01.json

python scripts/analyze_tt90.py results_bench/*.json
```

Note `--allow-odd` is unnecessary for these configs: 50,000/128 = **390** steps,
which is even. (The 351 I first computed came from mistakenly applying E1's
5,000-sample holdout to the benchmark arithmetic.)

## Principle

`epochs` is not a budget knob in this experiment — it is a **hyperparameter of
the learning-rate schedule**. Changing it from 200 to 30 changes the LR at every
step. Any results comparison must hold it fixed.
