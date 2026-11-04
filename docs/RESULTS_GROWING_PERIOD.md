# Growing-period triangular cycle (side branch)

Question: instead of a fixed one-epoch cycle, let the cycle get **longer** as
training proceeds — 1 epoch for epochs 0-31, 2 for 32-63, 4 for 64-127, 8 for
128-199. Does a period schedule substitute for cosine annealing?

Implementation note: the phase is accumulated (`phase += 1/period` per step), not
computed as `step/period`, so the LR curve stays continuous at the boundaries.
Verified: step-to-step LR change at each boundary equals the in-stage change.

## Results (1 seed — see caveat)

| arm | peak | final | 1st 90% |
|---|---|---|---|
| B1 fixed 1-ep period + anneal | 0.9535 | 0.9532 | 18 |
| **G_grow growing period + anneal** | 0.9522 | 0.9520 | 17 |
| B1_noanneal fixed period, no anneal | 0.9409 | 0.9356 | 18 |
| **G_grow_noanneal growing period, no anneal** | 0.9474 | **0.9471** | **16** |

## Finding 1: the effect is in the 93-94% band, not at 90%

An earlier version of this note claimed the period schedule "does nothing" with an
anneal. That was wrong: it was based on t@90 and final accuracy, both of which are
blind to the effect. The curves are identical at 90% and converge at the end; the
difference lives in the middle.

| arm | n | t@90 | t@93 | t@94 | t@95 |
|---|---|---|---|---|---|
| B1 fixed 1-ep + anneal | 3 | 17 | 53 | 112 | 168 |
| **G_grow 1->2->4->8 + anneal** | 1 | 17 | **44** | **80** | 172 |
| B1_noanneal fixed, no anneal | 3 | 17 | 63 | 175 | never |
| **G_grow_noanneal growing, no anneal** | 1 | 16 | **44** | **88** | never |

* t@93: 44 vs 53 -> 1.2x faster than the fixed period
* t@94: 80 vs 112 -> 1.4x faster
* t@90: tie (17 vs 17); t@95: slightly worse (172 vs 168)
* final accuracy: tie (0.9520 vs 0.9519)

Fixed-period B1 spends 59 epochs going from 93% to 94% (53 -> 112); G_grow spends
36 (44 -> 80). That is where the period schedule earns its keep.

Is it noise? B1's per-seed t@93 is 57/57/46 (mean 53, sd 6.4) and t@94 is
113/104/119 (mean 112, sd 7.5). G_grow's 44 and 80 sit 1.4 and 4.3 sd away. At n=1
that is not a formal test, but it is far outside seed variation -- hence the 3-seed
confirmation run (configs/e7_growperiod_seeds.json).

## Finding 1b: the period schedule substitutes for the anneal in mid-training

G_grow_noanneal reaches 93% at epoch 44 -- identical to the annealed G_grow -- so
during mid-training the growing period does the same job as annealing. It still
cannot reach 95% (never), so an anneal remains necessary for the final level.

## Finding 2: without annealing, growing the period helps — but for an unknown reason

0.9356 → 0.9471, **+1.15 pp**, and it still falls short of the annealed result
(0.9519), so the period schedule recovers ~70% of the gap that annealing closes.

**My predicted mechanism is wrong.** I expected the longer period to create a
longer contiguous low-LR stretch late in training — a crude quasi-anneal. The LR
statistics say otherwise:

| arm | final LR | mean LR (last 20 ep) | frac(lr<0.01) | longest run lr<0.05 |
|---|---|---|---|---|
| fixed period, no anneal | 0.00490 | 0.4004 | 0.01 | ~0.5 ep |
| growing period, no anneal | 0.00131 | 0.4005 | 0.01 | ~0.5 ep |
| B1 fixed + anneal | 0.00000 | 0.0033 | 0.92 | — |

A triangle spends the same *fraction* of time in each LR band regardless of its
period, so lengthening it does not create a longer settle phase. The tail LR
distribution is essentially unchanged.

So the +1.15 pp is either (a) fewer cycle reversals late in training, changing the
dynamics in a way the LR histogram cannot see, or (b) noise — **n=1**.

## What would settle it

3 seeds on `G_grow_noanneal` and `B1_noanneal` (6 runs, ~7 h). If +1.15 pp holds
across seeds it is a real effect that needs a mechanistic account; if it shrinks
to <0.4 pp it was seed noise.

It is not on the critical path for the main claim: the annealed schedule already
reaches 95.19% and the period schedule cannot beat that.
