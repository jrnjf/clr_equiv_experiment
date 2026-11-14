# Handover to the research writer

> **SUPERSEDED IN PART.** The supervisor review changed three of the claims below.
> Read `docs/HANDOVER_ADDENDUM.md` first; this file remains useful for the project
> history and for the pre-review numbers, but the period framing, the decay claim
> and several statistics were revised.

You are taking over the writing of an empirical paper about a learning-rate
schedule. This document tells you what the paper is, how the evidence was built
(including everything that failed), what you may and may not claim, and where the
artefacts are.

Read this whole file before touching the draft. Most of the traps below are ones we
fell into ourselves.

---

## 1. The one-paragraph version

Take a cosine annealing schedule and multiply it by a triangular cycle whose period
is one epoch. Run momentum-free SGD at a peak learning rate of 0.8. On CIFAR-10 and
CIFAR-100 with ResNet-18, this reaches a given test accuracy several times sooner
than standard recipes — 6.5x sooner to 90% than the standard momentum-0.9 recipe,
5.5x sooner to 65% on CIFAR-100 than a mean-matched cosine anneal — **at the same
final accuracy**. It is a time-to-accuracy result, not an accuracy result. The cycle
also does not work with momentum: adding momentum 0.9 costs 2.8 points and 142
epochs.

Neither schedule component is new. The paper's contribution is a controlled
characterisation of a regime nobody has tested, plus a set of negative results about
what the effect is *not*.

---

## 2. What the paper claims, precisely

**Claims (all verified, see §5 for numbers):**

1. Against a **mean-matched** cosine control (same average learning rate, same
   anneal, differing only by the cycle), the method reaches 90% test accuracy 4.1x
   sooner on CIFAR-10 and 65/70% 5.5x/5.3x sooner on CIFAR-100, at statistically
   equal final accuracy.
2. Against the **standard momentum-0.9 recipe** the margin on CIFAR-10 grows to
   6.5x to 90%, at comparable peak accuracy (95.19% vs 95.35%, a non-significant
   deficit on final, a small consistent one on peak).
3. The schedule **requires momentum-free SGD**: the identical schedule with momentum
   0.9 ends 2.82 points lower (t = 30.8) and takes 142 more epochs to 90%
   (t = -213).
4. Against **tuned AdamW + cosine**, the method is +0.63 points on CIFAR-10
   (t = 6.37, 3/3 seeds) and +3.88 points on CIFAR-100 (with a tuning caveat, §7).
5. The advantage is **threshold-dependent**: it is largest in early-to-mid training
   and vanishes at each dataset's accuracy ceiling (95% on CIFAR-10, 75% on
   CIFAR-100). It is a speed effect, not an accuracy effect.

**What the paper must NOT claim:**

* **No accuracy advantage over tuned cosine annealing.** Final accuracy is tied on
  both datasets. On CIFAR-10 the standard momentum recipe is 0.16 pp *ahead* on
  peak (t = -2.62, 0/3 seeds).
* **No novelty in the schedule.** Both components are established (Smith 2015;
  Loshchilov & Hutter 2016) and composing them is a two-line `LambdaLR`.
* **No universal superiority.** It is beaten on peak accuracy by the momentum
  recipe and it is not tested beyond ResNet-18 at 32x32.
* **Not "faster in wall-clock for equal total work".** The claim is time-to-a-given
  accuracy, given a fixed epoch budget.

---

## 3. How we got here (the journey)

The project began with a much stronger and vaguer claim: *"vanilla SGD without even
momentum beats any other optimizer with any other LR scheduler when we apply my
scheduler."* Most of the work was trying to break that claim, and the paper we now
have is what survived.

**Stage 1 — Is the scheduler even implemented correctly?** (E1)
We had a custom optimizer with the schedule inside it. We built an external
scheduler and proved the two produce identical trajectories (6e-6 relative in
float32, ~1e-16 in float64). This was a *decision helper*, not a paper result, and
it does not belong in the paper. It later became the basis for the resumable,
stock-PyTorch implementation the paper describes.

**Stage 2 — Weight decay.** (E2)
We tested whether decay should be scaled by the effective learning rate. It should:
torch's `SGD` already does this (`sgd.py`:355, :379). An LR-independent decay
coefficient collapsed to chance late in training (final loss = ln 10 = 2.3022),
because decay kept acting after the LR had annealed to ~1e-7. This is a negative
result worth one paragraph: **decay must track the learning rate.**

**Stage 3 — Is it just a lower average learning rate?** (E3a)
The method's mean LR is 0.200; a cosine anneal from peak 0.8 also averages 0.400. We
swept constant learning rates {0.05 ... 0.8} with no schedule at all. The best
constant arm peaked at 92.15% and every constant arm *degraded* after its peak. This
is the single most important control in the paper.

**Stage 4 — Which component does what?** (E3b)
A 2x2: cycle on/off, anneal on/off. Removing the anneal keeps the speed (16.7
epochs to 90%) but caps accuracy at 93.63%. Removing the cycle loses the speed.
Conclusion: **the cycle buys speed, the anneal buys the final level.** This is now
Fig 1's story and §5.8 of the draft.

**Stage 5 — The mean-matched control.** (the load-bearing experiment)
Cosine from peak 0.4 has mean LR 0.2000 versus the method's 0.2002 — identical
average step size, differing only by the cycle. Before this, the speed claim was
2.65x against a baseline running at half the average step size. After it: **4.1x**.
The effect got *larger*, not smaller, which is the opposite of what normally happens
when a confound is closed.

**Stage 6 — Falsification attempts.** Four more, all designed to kill the result:

| attempt | what happened |
|---|---|
| evaluation phase (B1 is evaluated at the LR trough, §6) | no effect: t@90 16 vs 17 |
| budget matching (anneal horizon = budget at 40/80/160) | method wins at every budget |
| AdamW tuning (full lr x wd grid) | margin **halved** from +1.14 to +0.63 pp, survived |
| momentum (the standard recipe) | margin **grew** to 6.5x |

**Stage 7 — Second dataset.** CIFAR-100 replicates the pattern: 5.5x to 65%, 5.3x
to 70%, final accuracy tied, and the same convergence at the ceiling.

**Stage 8 — A side branch that did not make the paper.** We tried a growing cycle
period (1 -> 2 -> 4 -> 8 epochs at epochs 32/64/128). Results were ambiguous
(one seed suggested a mid-training gain, later confounded by evaluation phase), and
the effect disappears entirely once annealing is present. It is documented in
`docs/RESULTS_GROWING_PERIOD.md` but should not be a paper claim.

---

## 4. The narrative spine the paper should use

The introduction should be built around the falsification table, because it is the
most honest and most compelling thing in the project:

> Take a known schedule composition, run it in an unusual regime, observe a large
> speedup, then spend the paper trying to explain it away: by matching the average
> learning rate, by fixing the evaluation phase, by matching the compute budget, by
> tuning the baselines, and by giving the baselines momentum. The speedup survives
> all five; what does not survive is any claim about final accuracy.

This framing turns a modest result into a credible one. Do not replace it with a
standard "we propose a novel schedule" narrative — the schedule is not novel and a
reviewer will notice within one page.

The discussion should be built around the **anytime** framing: given a fixed compute
budget, the schedule yields a usable model sooner; given enough budget, all annealed
schedules converge to the same ~95.2% / ~77.7% ceiling. This is what makes the
result practically relevant (hyperparameter search, early stopping, limited
budgets) and it is the only framing in which the result is significant.

---

## 5. The evidence, with provenance

All numbers below are means over 3 seeds unless marked. Every one is regenerable:

```bash
python scripts/weekend_report.py        # all tables + paired tests
python scripts/make_figures.py          # figures/*.png
```

### CIFAR-10, 200 epochs, batch 128 (390 steps), ResNet-18

| arm | peak | final | t@90 | t@93 | t@94 |
|---|---|---|---|---|---|
| triangle x cosine, momentum 0 | 0.9526 | 0.9519 | **16.7** | 53 | 112 |
| cosine, peak 0.4 (mean-matched) | 0.9512 | 0.9504 | 68.7 | 132 | 149 |
| cosine, peak 0.2 | 0.9512 | 0.9507 | 44.3 | 114 | 134 |
| cosine, peak 0.8 | 0.9503 | 0.9493 | 105.3 | 151 | 162 |
| **cosine + momentum 0.9, lr 0.1 (standard)** | **0.9543** | **0.9535** | 107.7 | 150 | — |
| cosine + momentum 0.9, lr 0.2 | 0.9500 | 0.9495 | 141.0 | 169 | — |
| AdamW + cosine, lr 2e-3 wd 0.05 (best of 4x2 grid) | 0.9467 | 0.9456 | — | — | — |
| AdamW + cosine, lr 1e-3 wd 0.01 (first, untuned) | 0.9416 | 0.9405 | 20.0 | 103 | 172 |
| triangle x cosine + momentum 0.9 | 0.9254 | 0.9236 | 158.7 | never | — |
| triangle x cosine, no anneal | 0.9398 | 0.9363 | 16.7 | 63 | 175 |
| constant lr 0.1 (best constant, n=1) | 0.9215 | 0.9073 | 46 | — | — |

Key paired statistics (n=3, |t| >= 4.30 for p<0.05):

| comparison | metric | effect | t | seeds |
|---|---|---|---|---|
| method vs mean-matched cosine | t@90 | -52.0 ep (4.1x) | -11.53 | 3/3 |
| method vs mean-matched cosine | final acc | +0.14 pp | +0.90 (n.s.) | 2/3 |
| method vs standard momentum recipe | t@90 | -89.6 ep (6.5x) | -78.8 (n=3) | 3/3 |
| method vs standard momentum recipe | final acc | -0.17 pp | -1.53 (n.s.) | 1/3 |
| method vs standard momentum recipe | peak acc | -0.16 pp | -2.62 | 0/3 |
| method vs method+momentum | final acc | +2.82 pp | +30.78 | 3/3 |
| method vs tuned AdamW | final acc | +0.63 pp | +6.37 | 3/3 |

### CIFAR-100, same protocol (ceiling ~78%, so thresholds 65/70/75%)

| arm | final | t@65 | t@70 | t@75 |
|---|---|---|---|---|
| triangle x cosine | 0.7761 | **12.3** | **21.0** | 130.7 |
| cosine, peak 0.4 (mean-matched) | 0.7780 | 67.7 | 112.3 | 130.7 |
| cosine, peak 0.2 | 0.7695 | 39.7 | 89.7 | 111.7 |
| AdamW + cosine, lr 2e-3 wd 0.05 | 0.7373 | 33.7 | 129.0 | never |

* t@65: **-55.3 ep (5.5x), t = -27.3**; t@70: -91.3 ep (5.3x), t = -76.0.
* final accuracy vs mean-matched cosine: -0.20 pp (t = -1.48, n.s.).
* vs AdamW: **+3.88 pp, t = 38.2** (see the tuning caveat, §7).
* cross-entropy vs mean-matched cosine: **-0.0268, t = -5.02** — same accuracy at
  lower loss (§5.9 of the draft).

### Budget-matched CIFAR-10 (anneal horizon = budget; 1 seed)

| budget | method t@90 / final / peak | cosine@0.2 t@90 / final / peak |
|---|---|---|
| 40 | 15 / 0.9324 / 0.9328 | 21 / 0.9271 / 0.9271 |
| 80 | 15 / 0.9445 / 0.9449 | 30 / 0.9369 / 0.9395 |
| 160 | 16 / 0.9496 / 0.9498 | 45 / 0.9452 / 0.9455 |

Under a constrained budget the **peak** gap becomes positive (+0.43 to +0.57 pp)
and shrinks to +0.14 pp at 200 epochs. Report with the n=1 caveat.

---

## 6. Evaluation-phase subtlety (do not get this wrong)

Because the cycle period equals one epoch, and validation runs at the end of each
epoch, **the model is always evaluated at the LR trough** — its most stable phase.
This looked like a serious confound. We tested it by shifting the cycle half an
epoch so validation lands on the LR peak: t@90 16 vs 17, t@93 53 vs 53, final
0.9522 vs 0.9519. Nothing.

So it is safe to report the standard numbers. But the paper should mention the
coincidence and the control, because a careful reviewer will ask, and it is one
sentence.

---

## 7. Caveats that must appear in the paper

1. **AdamW on CIFAR-100 was not tuned on CIFAR-100.** The winning CIFAR-10 config
   (lr 2e-3, wd 0.05) was applied unchanged. Two CIFAR-100 AdamW points exist
   (0.7283, 0.7373), both well below the method's 0.7761, so the direction is
   consistent — but if the true CIFAR-100 optimum is ~0.76 the +3.88 pp shrinks
   toward the CIFAR-10 figure. State it.
2. **The AdamW grid may not contain the optimum.** Both lr and wd improve
   monotonically across the grid; a reviewer can ask for lr 4e-3 / wd 0.1.
3. **One architecture, two datasets**, both 32x32. Untested on ImageNet scale,
   higher resolutions, or transformers (which need warmup and would not tolerate
   peak 0.8 from step 0).
4. **The peak learning rate 0.8 was inherited, not tuned.** The sensitivity sweep at
   0.4 and 1.2 has not been run. This also means the paper cannot yet support the
   title's implied claim that the result is configuration-insensitive.
5. **All arms are tuned on the test split.** Standard practice is a validation split.
   Either do it or state it.
6. **Budget-matched runs are n=1**; the constrained-budget peak claim is
   directional, not significant.
7. **No step-decay baseline.**
8. **The paper reports accuracy, but loss and accuracy diverge** on CIFAR-100 (the
   method has lower cross-entropy at equal accuracy). Explain the mechanism briefly
   (argmax vs probability assigned to the true class) and do not over-claim
   calibration: without ECE we cannot separate calibration from overconfidence.

---

## 8. What already exists

| artefact | path | notes |
|---|---|---|
| Full draft | `docs/PAPER_DRAFT.md` | All sections, verified numbers, 4 figures inline |
| Figures | `figures/fig1..fig4.png` | regenerate with `scripts/make_figures.py` |
| Table generator | `scripts/weekend_report.py` | all tables + paired tests |
| Momentum verdict | `docs/MOMENTUM_VERDICT.md` | the decisive experiment, in full |
| Threat log | `docs/THREATS.md` | every confound tested, with results |
| Growing-period side branch | `docs/RESULTS_GROWING_PERIOD.md` | not a paper claim |
| Weekend results | `docs/WEEKEND_RESULTS.md` | AdamW grid, CIFAR-100, loss/accuracy |
| Abstract options | `docs/ABSTRACT_PROPOSAL.md` | several framings; the current draft's abstract is the one to use |
| Raw results | `results*/` | per-run JSONs with full `val_history` |

Suggested title (from the draft): *Just a configuration: a triangular cycle makes
momentum-free SGD reach target accuracies several times sooner.*

---

## 9. What the writer must supply (not in the draft)

The draft is structurally complete but written by an engineer. The parts that need
a writer's judgement:

* **The introduction's first paragraph.** It must state the observation plainly and
  promise the falsification programme. Do not open with "deep learning has
  revolutionised...".
* **The discussion's mechanism paragraph.** Two candidate mechanisms are consistent
  with the ablations — the cycle repeatedly depresses the LR so the model settles at
  many points, and the high peak explores faster while the short depression bounds
  the damage. The momentum result (§5.5) supports the second. Say plainly that we do
  not fully separate them.
* **The related-work positioning.** Smith's CLR is defined per iteration; our period
  is one epoch. The cosine anneal is normally used alone or with warm restarts; we
  compose it with a cycle and never restart. The "decaying schedules win at scale"
  line of work (WSD, power scheduler) is the natural context: our result is
  consistent with it in spirit.
* **Tone on limitations.** Do not bury them. This paper's credibility comes from
  having tested its own claims to destruction; the limitations section should read
  as strength, not apology.

---

## 10. What is still worth running (optional, cannot invalidate anything)

| experiment | cost | why |
|---|---|---|
| peak-lr sensitivity {0.4, 1.2} | ~7 h | answers "why 0.8"; earns or refutes the title's robustness implication |
| step-decay baseline | ~4 h | standard comparison a reviewer may request |
| CIFAR-100 AdamW grid | ~7 h | protects the +3.88 pp second-dataset margin |
| seeds 3 -> 5 | ~4 h | tightens intervals |

None of these can invalidate claims 1-5 in §2. The first two are the most likely
reviewer requests.

---

## 11. Process notes (why the repo looks the way it does)

Two implementation facts that matter for the reproducibility section, and one
caution:

* The method is stock PyTorch: `SGD` + `LambdaLR`, no custom optimizer in the
  method path. The custom optimizer exists only to prove equivalence and is not part
  of the method.
* Weight decay needs no special handling because `torch.optim.SGD` scales it by the
  current LR (`sgd.py`:355,:379); `LambdaLR` keeps that LR current.
* Runs are resumable (`--skip-existing`) and ~20 s/epoch on a single RTX A4000. This
  is what made 40 h of seed runs tractable, and it belongs in the reproducibility
  paragraph.
* **Caution on the logs:** for a while the experiment runner block-buffered stdout,
  so a healthy multi-hour run appeared frozen for 39 minutes at a time. If you ever
  re-run anything, use `python -u`. This is irrelevant to the science but explains
  gaps in `results_weekend/*.log` timestamps.
