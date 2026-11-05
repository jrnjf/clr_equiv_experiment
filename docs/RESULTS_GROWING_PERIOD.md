# Growing-period triangular cycle (side branch)

Question: instead of a fixed one-epoch cycle, let the cycle get **longer** as
training proceeds — 1 epoch for epochs 0-31, 2 for 32-63, 4 for 64-127, 8 for
128-199. Does a period schedule substitute for cosine annealing?

Implementation note: the phase is accumulated (`phase += 1/period` per step), not
computed as `step/period`, so the LR curve stays continuous at the boundaries.
Verified: step-to-step LR change at each boundary equals the in-stage change.

## Results (3 seeds)

| arm | n | t@90 | t@93 | t@94 | t@95 | final |
|---|---|---|---|---|---|---|
| B1 fixed 1-ep + anneal | 3 | 17 | 53 | 112 | 168 | 0.9519 |
| **G_grow growing + anneal** | 3 | 18 | **43** | **80** | 172 | 0.9509 |
| B1_noanneal fixed, no anneal | 3 | 17 | 63 | 175 | never | 0.9363 |
| G_grow_noanneal growing, no anneal | 3 | 17 | **42** | 84 | never | 0.9461 |

Paired, G_grow vs B1 (n=3, |t|>=4.30 for p<.05):

| metric | G_grow | B1 | diff | t | rank-separated |
|---|---|---|---|---|---|
| t@90 | 19/17/17 | 18/16/16 | **+1 ep (slower)** | — (sd=0) | no |
| t@93 | 50/42/38 | 57/57/46 | −10 ep | −3.97 | no |
| **t@94** | 96/68/76 | 113/104/119 | **−32 ep (1.40x)** | **−4.12** | **yes** |
| t@95 | 172/173/172 | 161/175/169 | +4 ep | +1.06 | no |
| final acc | 0.9501/0.9507/0.9520 | 0.9532/0.9504/0.9520 | −0.09 pp | — | no |

Interpretation: the 1.40x at 94% is **borderline** by the t-test (4.12 vs 4.30) but
**complete rank-separated** -- every G_grow seed beats every B1 seed at that
threshold, which is p=0.10 two-sided / 0.05 one-sided by rank alone at 3v3. The
direction is consistent across all three seeds at both 93% and 94%.

Note the trade: growing the period is consistently **1 epoch slower at 90%**
(19/17/17 vs 18/16/16) and tied at 95% and on final accuracy. It buys the middle
of the curve, not the ends.

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

## What would settle it (running)

Added seeds 3-5 to B1 and G_grow (configs/e8_growperiod_n6.json, 6 runs, ~7 h)
because the 3-seed t@94 difference sits just under the significance bar.

It is not on the critical path for the main claim: the annealed schedule already
reaches 95.19% and the period schedule cannot beat that.
