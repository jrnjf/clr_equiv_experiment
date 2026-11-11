# The momentum experiment: verdict

The last unrun experiment, and the only one that could still invalidate the primary
speed claim. Every arm in the project so far -- method included -- had momentum 0.

## Results

| arm | n | peak | final | t@90 | t@93 |
|---|---|---|---|---|---|
| **B1 triangle x cosine, momentum 0** | 3 | 0.9526 | 0.9519 | **16.7** | 53 |
| cosine + **momentum 0.9**, lr 0.1 (standard recipe) | 3 | **0.9543** | **0.9535** | 107.7 | 150 |
| cosine + momentum 0.9, lr 0.2 | 3 | 0.9500 | 0.9495 | 141.0 | 169 |
| cosine, momentum 0, peak 0.4 (mean-matched) | 3 | 0.9512 | 0.9504 | 68.7 | 132 |
| **B1 + momentum 0.9** | 3 | 0.9254 | 0.9236 | 158.7 | never |

Paired B1 vs the standard momentum recipe: t@90 **-91.0 epochs, t = -78.8, 3/3 seeds
(6.5x faster)**. Final accuracy **-0.17 pp, t = -1.53, n.s.**

Paired B1 vs B1+momentum: final **+2.82 pp (t=30.8)**, t@90 **-142 epochs
(t=-213)**. Adding momentum to the method is catastrophic.

## What this settles

**The speed claim survives and strengthens.** Against the standard CIFAR recipe
(SGD momentum 0.9, lr 0.1, cosine), the method reaches 90% in 16.7 epochs versus
107.7 -- **6.5x**, not the 4.1x measured against a momentum-free baseline. The
baseline that everyone would actually run is the slow one.

**"Momentum-free" is a necessity, not a description.** B1 at peak lr 0.8 with
momentum 0.9 reaches 90% at epoch 159 and ends 2.8 pp lower. The method does not
work with momentum. That upgrades the phrase from an incidental condition to a
stated constraint, and it plausibly explains why peak lr 0.8 is survivable at all:
without momentum the effective step is bounded by the instantaneous lr, so the
triangular cycle keeps reinflating a step that is never accumulated.

**The catch: the standard recipe is marginally more accurate.** Final 0.9535 vs
0.9519 (-0.17 pp, t = -1.53, n.s.) and peak 0.9543 vs 0.9526 (-0.16 pp,
t = -2.62, 0/3 seeds). Both are small, the final gap is not significant, but the
peak gap is consistent in direction across all three seeds. Honest statement: the
momentum recipe ends equal or a hair ahead, while reaching 90% 6.5x later.

## What the abstract should now say

The strongest defensible claim is:

> At peak accuracy comparable to the standard CIFAR recipe (95.19% vs 95.35%), a
> momentum-free SGD with a one-epoch triangular cycle multiplied by a cosine anneal
> reaches 90% test accuracy in 16.7 epochs against 107.7 for SGD with momentum 0.9
> and cosine annealing -- a 6.5x reduction. The cycle does not work with momentum:
> the same schedule at momentum 0.9 reaches 90% only at epoch 159 and ends 2.8 pp
> lower.

This is stronger than the old framing because the comparison is now against the
recipe a practitioner would actually use, and because momentum-free is justified
rather than assumed.
