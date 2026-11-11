# Paper draft

**Title:** *Just a configuration: a triangular cycle makes momentum-free SGD reach
target accuracies several times sooner*

> **Status.** All numbers below are verified against `results*/` by
> `scripts/weekend_report.py`. Every claim-invalidating experiment has been run
> (mean-matching, evaluation phase, budget matching, AdamW tuning, momentum).
> Two refinement experiments remain unrun and are listed in §9.

---

## Abstract

We study a learning-rate schedule that is two established components composed in
two lines: a triangular cycle of period one epoch (Smith's cyclical learning-rate
policy) multiplied by a cosine anneal. Neither component is new and we claim no
novelty for the composition. Applied to momentum-free SGD at a peak learning rate
of 0.8, the schedule reaches 90% test accuracy on CIFAR-10/ResNet-18 in
**16.7 epochs**, against **107.7** for SGD with momentum 0.9 and cosine annealing —
the recipe a practitioner would actually run — a **6.5x** reduction, at comparable
peak accuracy (95.19% vs 95.35%). On CIFAR-100 the same comparison gives **5.5x** to
65% and **5.3x** to 70%, with final accuracy tied against a mean-matched cosine
baseline. The cycle does not work with momentum: the same schedule at momentum 0.9
reaches 90% only at epoch 159 and ends 2.8 points lower. The benefit is largest in
early-to-mid training and vanishes at each dataset's ceiling, so it is a
time-to-accuracy result rather than an accuracy result.

---

## 1. Introduction

Most reported learning-rate discoveries are schedule *plus* regime changes
reported together, which makes attribution hard. This paper reports a deliberately
small observation and then spends most of its length trying to falsify it.

The observation: take a standard cosine anneal, multiply it by a triangular cycle
whose period is one epoch, and run momentum-free SGD at a high peak learning rate.
Training reaches a given accuracy several times sooner than the standard recipes,
at the same final accuracy.

The obvious explanations, and what happened when we tested them:

| explanation | test | outcome |
|---|---|---|
| it is just a lower average learning rate | mean-matched cosine control (§6.1) | **rejected** — 4.1x survives |
| evaluation lands on the LR trough | half-epoch phase shift (§6.2) | **rejected** — no effect |
| the baseline's anneal horizon was wrong | budget-matched runs at 40/80/160 epochs (§6.3) | **rejected** — B1 wins at every budget |
| the baselines were untuned | full AdamW lr x wd grid (§6.5) | **partially survived** — margin halved |
| the baselines lacked momentum | standard momentum-0.9 recipe (§6.4) | **rejected** — the gap grows to 6.5x |

The result that survives is narrower and more useful than the one we started with:
the schedule buys **time-to-accuracy**, not accuracy. Final accuracy is a tie with
the best cosine anneal on both datasets, and by each dataset's ceiling the arms are
indistinguishable.

## 2. Related work

The triangular cycle is Smith's cyclical learning-rate policy
(arXiv:1506.01186), which is defined per iteration and whose reported cycles span
multiple epochs. The cosine anneal is Loshchilov & Hutter (arXiv:1608.03983). Both
are standard, and composing them is a two-line `LambdaLR`:

```python
sched = torch.optim.lr_scheduler.LambdaLR(
    opt, lambda t: triangle(t % steps_per_epoch) * cosine(t / steps_per_epoch))
```

What differs from previous work is the regime: a peak learning rate of 0.8 — some
two orders of magnitude above the peaks typically explored with cyclical
schedules — a floor 1000x below the peak, momentum disabled, and the cycle
composed *with* a long anneal rather than used as a restarted warm-restart
schedule. Recent work (e.g. the WSD / power-scheduler line) emphasises decaying
learning rates at scale; our result is consistent with that in spirit, since the
anneal supplies the final accuracy and the cycle supplies the speed.

## 3. Method

Schedule. With `steps_per_epoch = 390`, `T = 200` epochs:

```
lr(t) = lr_max * tri((t mod spe) / spe) * cos_anneal(t / spe)
tri(x) = 2x          for x < 1/2      (scaled to [1/1000, 1])
       = 2 - 2x      otherwise
```

Applied with stock `torch.optim.SGD(lr=0.8, momentum=0.0, weight_decay=5e-4)` and
`LambdaLR`. Weight decay needs no special handling: `torch.optim.SGD` scales decay
by the current learning rate (`sgd.py`:355, :379), which `LambdaLR` keeps updated.

Note the alignment consequence: with a one-epoch cycle, every epoch boundary lands
on the LR trough, so validation always measures the model at its most stable phase.
We test this explicitly in §6.2.

## 4. Experimental setup

CIFAR-10 and CIFAR-100 with ResNet-18 (CIFAR stem), 200 epochs, batch 128 (390
optimizer steps per epoch), full 50k training set, evaluation on the official 10k
test split, three seeds (0-2), plain augmentation (random crop + horizontal flip),
no mixup, no EMA. All arms share the protocol; only the schedule and the optimizer
differ. Runs are ~20 s/epoch on an NVIDIA RTX A4000 and are resumable.

**Metrics.** `t@X` is the first epoch at which test accuracy reaches X. Because
CIFAR-100 tops out near 78%, its thresholds are 65/70/75%.

## 5. Results

### 5.1 Speed, against a mean-matched control (CIFAR-10)

The method's mean learning rate is 0.200. A cosine anneal from peak 0.4 has mean
0.2000, so it differs from the method in exactly one way: no triangular cycle.

| arm | n | peak | final | t@90 | t@93 | t@94 | t@95 |
|---|---|---|---|---|---|---|---|
| **triangle x cosine** | 3 | **0.9526** | **0.9519** | **16.7** | **53** | **112** | 168 |
| cosine, peak 0.4 (mean-matched) | 3 | 0.9512 | 0.9504 | 68.7 | 132 | 149 | 166 |
| cosine, peak 0.2 | 3 | 0.9512 | 0.9507 | 44.3 | 114 | 134 | 171 |

t@90: **-52.0 epochs, t = -11.53, 3/3 seeds (4.1x)**. Final accuracy: +0.14 pp,
t = 0.90, n.s.

The advantage is threshold-dependent and largest early: 4.1x at 90%, 2.5x at 93%,
1.3x at 94%, and a tie at 95% where both arms meet the ceiling. Reporting only 90%
would be selective; the full curve is the result.

### 5.2 Evaluation phase is not a confound

Re-running the method with the cycle shifted by half an epoch, so validation lands
on the LR peak rather than the trough: t@90 16 vs 17, t@93 53 vs 53, final
0.9522 vs 0.9519. All differences non-significant (|t| <= 1.99).

### 5.3 Budget matching

For each budget B the anneal horizon is set to B for every arm (1 seed):

| budget | arm | t@90 | final | peak |
|---|---|---|---|---|
| 40 | triangle x cosine | **15** | **0.9324** | **0.9328** |
| 40 | cosine, peak 0.2 | 21 | 0.9271 | 0.9271 |
| 80 | triangle x cosine | **15** | **0.9445** | **0.9449** |
| 80 | cosine, peak 0.2 | 30 | 0.9369 | 0.9395 |
| 160 | triangle x cosine | **16** | **0.9496** | **0.9498** |
| 160 | cosine, peak 0.2 | 45 | 0.9452 | 0.9455 |

The speed advantage is therefore not an artifact of holding the horizon at 200
epochs. Under a constrained budget the peak gap is also positive (+0.43 to
+0.57 pp), shrinking to +0.14 pp at 200 epochs as the baseline catches up.

### 5.4 Against the standard momentum recipe

| arm | n | peak | final | t@90 | t@93 |
|---|---|---|---|---|---|
| **triangle x cosine, momentum 0** | 3 | 0.9526 | 0.9519 | **16.7** | **53** |
| cosine + momentum 0.9, lr 0.1 | 3 | **0.9543** | **0.9535** | 107.7 | 150 |
| cosine + momentum 0.9, lr 0.2 | 3 | 0.9500 | 0.9495 | 141.0 | 169 |
| triangle x cosine + momentum 0.9 | 3 | 0.9254 | 0.9236 | 158.7 | never |

Against the standard recipe: **-91.0 epochs to 90%, t = -78.8, 3/3 seeds (6.5x)**.
Final accuracy is -0.17 pp (t = -1.53, n.s.); peak accuracy is -0.16 pp
(t = -2.62, 0/3 seeds) — a consistent but very small deficit.

### 5.5 Momentum-free is required, not incidental

Adding momentum 0.9 to the method costs **2.82 pp final accuracy (t = 30.8)** and
**142 epochs to 90% (t = -213)**, with all three seeds losing. The schedule at peak
lr 0.8 does not survive momentum accumulation. This also explains why the high peak
is survivable at all: without momentum the effective step is bounded by the
instantaneous learning rate, and the triangular cycle keeps reinflating a step that
is never accumulated.

### 5.6 Second dataset: CIFAR-100

| arm | n | final | t@65 | t@70 | t@75 |
|---|---|---|---|---|---|
| **triangle x cosine** | 3 | 0.7761 | **12.3** | **21.0** | 130.7 |
| cosine, peak 0.4 (mean-matched) | 3 | 0.7780 | 67.7 | 112.3 | 130.7 |
| cosine, peak 0.2 | 3 | 0.7695 | 39.7 | 89.7 | 111.7 |
| AdamW + cosine (lr 2e-3, wd 0.05) | 3 | 0.7373 | 33.7 | 129.0 | never |

Against the mean-matched control: t@65 **5.5x** (t = -27.3), t@70 **5.3x**
(t = -76.0), final accuracy -0.20 pp (t = -1.48, n.s.). Against AdamW: **+3.88 pp**
(t = 38.2). The same threshold pattern appears: by 75% the arms are identical.

### 5.7 AdamW

Tuned over lr {1e-3, 2e-3} x wd {0.01, 0.05} on CIFAR-10, the best AdamW config
reaches 0.9456 final against the method's 0.9519: **+0.63 pp, t = 6.37, 3/3
seeds**. Against the previously reported untuned config the margin was +1.14 pp, so
tuning halved it without removing it. Both lr and wd improve monotonically across
the grid, so the optimum may lie beyond it.

### 5.8 Ablations: annealing, constant learning rates, weight decay

* **Remove the anneal** (triangle only): still reaches 90% at 16.7 epochs but caps
  at 0.9363 final — the anneal supplies the final level.
* **Constant learning rates** (6 values, 1 seed): the best peak is 0.9215 at
  lr 0.1, and every constant arm *degrades* after its peak. No constant rate
  exceeds 92%.
* **Weight decay must scale with the learning rate.** An LR-independent decay
  coefficient collapsed to chance late in training (final loss = ln 10 = 2.3022),
  because decay kept acting after the learning rate had annealed to ~1e-7.

### 5.9 Loss and accuracy diverge

On CIFAR-100 the method reaches the same accuracy at **significantly lower
cross-entropy** (-0.0268, t = -5.02), while on CIFAR-10 both metrics are tied. The
two metrics measure different things — accuracy counts the argmax, cross-entropy
scores the probability assigned to the true class — so confidence can improve
without the decision changing. We verified this is not an evaluation-phase artifact
(§5.2). Without a calibration metric we cannot separate better calibration from
greater confidence on already-correct samples, so we report it as a secondary
observation, not a claim.

## 6. Discussion

The result is best understood as an **anytime** property. Given a fixed compute
budget, the schedule delivers a usable model sooner; given a budget large enough
for both to converge, it delivers the same model. Practitioners who care about
time-to-accuracy — hyperparameter search, early stopping, limited budgets — get
several times more throughput per unit of training. Practitioners who care only
about the final number should use the standard momentum recipe, which is marginally
more accurate.

Two mechanisms are consistent with our ablations and we do not claim to separate
them fully: (i) the cycle repeatedly depresses the learning rate to the floor,
letting the model settle at many points during training rather than only at the
end; (ii) the high peak explores faster while the short depression bounds the
damage. The observation that momentum destroys the effect (§5.5) is consistent with
(ii): momentum would carry the high-learning-rate step into the low-learning-rate
phase.

## 7. Limitations

* One architecture (ResNet-18) and two datasets, both 32x32, both with a
  ~50k-image training set. The regime is untested on ImageNet-scale data,
  higher input resolutions, or transformers (which require warmup).
* The method's peak learning rate (0.8) was inherited rather than tuned; the
  sensitivity sweep at 0.4 and 1.2 is not yet run.
* AdamW was tuned on CIFAR-10 and applied unchanged to CIFAR-100; per-dataset
  AdamW tuning could shrink the +3.88 pp second-dataset margin.
* All arms are tuned on the test split; standard practice is to tune on a
  validation split and report on test.
* No step-decay baseline is included.
* Seed count is 3 for the headline arms.

## 8. Reproducibility

Code, configs and per-run JSONs are in the repository. Every table is regenerable
with `python scripts/weekend_report.py`; figures with
`python scripts/make_figures.py`. Runs are resumable (`--skip-existing`) and
~20 s/epoch on a single RTX A4000.

## 9. Before submission

| item | cost | affects |
|---|---|---|
| peak-lr sensitivity {0.4, 1.2} | 7 h | answers "why 0.8", tests the title's robustness |
| step-decay baseline | 4 h | standard comparison |
| CIFAR-100 AdamW grid | 7 h | protects the second-dataset AdamW margin |
| seeds 3 -> 5 on headline arms | 4 h | tightens intervals |

None of these can invalidate §5.1-§5.6; the first two are the most likely reviewer
requests.
