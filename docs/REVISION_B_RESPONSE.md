# Revision B — response to review

**Deliverable:** [`PAPER_DRAFT_REV_B.md`](PAPER_DRAFT_REV_B.md) (full revised draft).
**Source:** `PAPER_DRAFT.md` (Revision A) plus the verified evidence in `HANDOVER.md`,
`RESULTS_E2.md`, `THREATS.md`, `MOMENTUM_VERDICT.md`, `ABSTRACT_PROPOSAL.md` and
`lr_oscillation_literature.md`.

**No new runs.** Revision B introduces no new results; every empirical value is
traceable to the Revision A draft, the handover, or a results document. After the draft
was written, an independent verification pass recomputed every mean and paired statistic
from the raw JSONs; a handful of displayed values were corrected to the raw per-seed
values and several claims were qualified (full list in "Verification corrections" below).
The only artefact change outside prose is Fig. 3, which now carries variance bars (see
point 2).

---

## Point 1 — Tone & claims calibration

**Request.** Avoid framing the result as absolute superiority over long training
horizons; frame it around fast-convergence efficiency (anytime performance) and the
boundaries of short-period training dynamics. Emphasise rigorous hypothesis testing;
showing stability/competitiveness in the ultra-short regime is itself valuable.

**Changes in Revision B.**

| location | change |
|---|---|
| Title | Changed from *"Just a configuration: a triangular cycle makes momentum-free SGD reach target accuracies several times sooner"* to *"Faster, not better: mapping the anytime efficiency and boundaries of short-period learning-rate cycles"*. The new title states the claim's shape (speed, not accuracy) and its scope (boundaries). |
| Abstract | Reframed around "evaluated as an anytime algorithm"; explicitly says "Our contribution is a controlled characterisation of a short-period regime and its failure modes, not an accuracy claim; we do not claim superiority over long training horizons." |
| §1 Introduction | Added a **hypothesis-test table (H1-H6)** in which each skeptical explanation is stated as a testable null and marked rejected / not rejected. The falsification programme is now the visible spine of the introduction, as the handover requested. |
| §1 Contributions | Item 2 states explicitly that establishing stability/competitiveness inside the ultra-short regime is itself a useful citable result. |
| §1 "What we do not claim" | Consolidated explicit non-claims: no accuracy advantage, no schedule novelty, no universal superiority, no wall-clock claim at equal total work. |
| §5.2 | New subsection "The anytime profile: threshold and budget boundaries" — the speedup is largest at easy thresholds and vanishes at the ceiling; the constrained-budget peak gap is presented as directional (n=1), not as a win. |
| §6.1 | New discussion subsection "Anytime efficiency, not accuracy superiority", including the explicit statement that the momentum recipe is 0.16 pp ahead on peak. |
| §6.3 | New **boundary-map table** (threshold, budget, momentum, optimizer, dataset, architecture, peak LR, cycle period) stating where the effect is present, absent, and untested. |
| §6.5 | New "What would change our mind" subsection naming the experiments that could falsify each claim. |
| §9 Conclusion | Restates the honest headline: "faster, not better, with the boundaries stated". |

The revision keeps every strong empirical number (4.1x, 6.5x, 5.5x/5.3x) but always
inside the fixed-budget / time-to-threshold frame, so no sentence claims the schedule is
better than long-horizon training.

---

## Point 2 — Confounder control and seed documentation

**Request.** Document random-seed counts precisely; a small seed count (e.g. n=3) can
leave minor gaps underpowered; ensure variance bands are clearly visible on plots.

**Changes in Revision B.**

| location | change |
|---|---|
| §4.2 | New subsection "Seeds, pairing and statistical protocol": headline/ablation arms use n=3 fixed seeds (0,1,2), paired by seed with shared initialization and data order; single-seed arms are labelled wherever they appear (constant-LR sweep, budget-matched runs). |
| §4.2 | Explicit test protocol: two-sided paired t-test, df=2, critical value |t| >= 4.303, effect size + t + seed-win count reported. |
| §4.2 | Explicit **power statement**: with paired seed s.d. ~0.2 pp on accuracy, n=3 resolves ~0.6 pp at 80% power (~0.8 pp for the headline tie, whose paired s.d. is 0.28 pp); sub-resolution differences are called "no detectable difference", not equality. The three final-accuracy ties are named and marked underpowered. |
| §4.2 | New table "Replication status by result" listing seeds and status (paired test / directional / not run) for every result family, including the unrun peak-LR sensitivity. |
| §4.3 | New subsection "Figures and variance": Figs. 1 and 4 are mean +/- 1 s.d. over 3 seeds with visible bands; Fig. 3 now carries +/- 1 s.d. error bars; the E2 figure uses mean and per-seed range; figures are descriptive and all tests are on per-seed values. |
| §5.1 | Per-seed `t@90` now quoted in text (method 18/16/16 vs control 65/77/64; control s.d. 7.2 epochs). |
| §5.6 | Per-seed `t@65` now quoted (method 12/13/12 vs control 67/65/71). |
| Appendix A | Full statistics appendix: critical values, power formula, and a table of every paired comparison with effect, t, seed-win count and verdict. |
| Limitations §7 | Seed count and single-seed arms listed explicitly as limitations. |

**Figure artefact change.** `scripts/make_figures.py` was updated so Fig. 3
(`figures/fig3_time_to_threshold.png`) draws +/- 1 s.d. error bars across the three
seeds; previously it showed bare means. Figs. 1, 2 and 4 were re-rendered from the same
JSONs and are unchanged in content. Table values are unaffected.

---

## Point 3 — Theoretical grounding (Edge of Stability)

**Request.** Include a brief nod to the Edge of Stability literature (e.g. Cohen et al.)
in the introduction or discussion; it explains how rapid LR oscillations reshape the
loss landscape and prevent premature settling into sharp minima.

**Changes in Revision B.**

| location | change |
|---|---|
| §1 Introduction | New paragraph "A theoretical frame: the Edge of Stability". States Cohen et al.'s finding (max Hessian eigenvalue hovers just above 2/eta; loss non-monotone short-timescale, decreasing long-timescale) and applies it to the schedule: eta varies by 1000x within an epoch, so 2/eta swings between ~2.5 at the peak and ~2500 at the floor, repeatedly crossing the EoS boundary. Explicitly flags that EoS is stated for fixed-step full-batch GD and that we did not measure spectra. |
| §2 Related work | New "Edge of Stability" paragraph positioning against Cohen et al. [6], Jastrzebski et al. [7] and the 2025 EoS/balancing/catapult work [8]. |
| §5.5 | Momentum result now connected to the EoS account: without momentum the effective step is bounded by the instantaneous LR. |
| §6.2 | New discussion subsection "A candidate mechanism at the edge of stability": the cycle acts as a high-frequency probe of the stability boundary; it changes the path, not the minimum. Adds a falsifiable prediction (effect should weaken at lower peak LR; sharpness should trace a cycle-locked sawtooth) and names a sharpness-trajectory experiment as the natural next step. |
| §7 | New limitation: no sharpness measurement; EoS remains an interpretive hypothesis, not a measured mechanism. |
| References | Added [6] Cohen et al. (arXiv:2103.00065), [7] Jastrzebski et al. (arXiv:2002.09572), [8] Good regularity / EoS balancing and catapult (JMLR 2025). |

The EoS material is deliberately kept as a framing and a falsifiable prediction rather
than a claimed mechanism, consistent with the handover's warning against over-claiming.

---

## Point 4 — Section organisation: give E2 (weight-decay conventions) ample space

**Request.** The weight-decay conventions experiment (E2) should receive ample space;
most practitioners overlook how weight decay couples with oscillating learning rates,
making it a clear actionable methodological contribution for training engineers.

**Changes in Revision B.**

| location | change |
|---|---|
| §5.8 | The old single bullet in §5.8 of Revision A is now a full subsection, **"Weight-decay conventions: a methodological result (E2)"**, with the E2 figure, the three-convention table, the results table, the epoch-by-epoch collapse table, the mechanism, the open confound, and a practitioner-advice box. |
| Figure 5 | The existing `docs/fig_e2_wd.png` is now embedded in the paper with a caption (previously it was not in the draft). |
| Results | Result 1: intra-epoch decay oscillation buys nothing over smooth annealed decay (B2 vs B1: peak +0.077 pp, t=0.91; final +0.033 pp, t=0.24). Result 2: in this run an LR-independent decay coefficient collapses to chance (final loss ≈ ln 10 = 2.3022; weight norm 12.2 -> 1.5), with the strength/profile confound stated. |
| Mechanism | Explains why: at epoch 199 the gradient terms are negligible while a constant decay still removes 5e-4/step; B1/B2 multipliers decay to ~3e-10 and ~1.6e-7. |
| Open confound | States that F differs in both profile and strength, that the strength-matched control (F at wd=1e-4) was specified but not run, and exactly what is and is not claimed. This preserves the handover's honesty requirement. |
| Practitioner advice | Three actionable points: do not leave weight decay LR-independent under an annealed LR; you do not need to pulse it at the cycle frequency; check the tail's weight norm and loss. |
| §6.4 | The weight-decay coupling is called out as "the one piece of advice in the paper that applies to schedules other than ours". |
| §7 | The E2 strength confound is listed as an explicit limitation. |

---

## Additional editorial improvements (not requested, low risk)

* Added a complete **References** section [1]-[14]; Revision A had only inline arXiv
  ids and no reference list.
* Added **Appendix A** with the full statistical protocol and a single table of every
  paired comparison.
* Added a **Revision B note** at the top of the draft summarising the four changes.
* Converted the loose "threshold dependence" prose into the boundary-map table (§6.3)
  so the scope claims are unambiguous.
* Renamed the "Before submission" section to "What would change our mind" and merged it
  with the remaining-runs list, so the paper reads as a finished argument rather than a
  TODO list.

---

## Verification corrections (post-write pass)

An independent check recomputed the tables from `results/*.json` and
`results_cifar100/*.json` (paired t-tests by seed, n=3). The following were corrected in
the draft; they change presentation, not the paper's results.

| item | before | after |
|---|---|---|
| CIFAR-10 95% comparison | "tie at 95%" with control 166 over 3 seeds | control 166 is a **2/3-seed** mean (seed 2 peaks at 94.88%); table marked 166†, prose says "no measurable gap among the seeds that reach it"; abstract no longer says "vanishes at 95%" but "becomes negligible near the ceiling" |
| method + momentum t@90 | 158.7 epochs | **159.0** (raw first epoch >= 90%; all three seeds 159) |
| untuned-AdamW margin | +1.14 pp | **+1.13 pp** (raw paired mean; `ABSTRACT_PROPOSAL.md` already said +1.13) |
| E2 peak deficit | "1.65 pp below B1/B2" | **1.66 pp below B1, 1.73 pp below B2** |
| uniform-prediction loss | "exactly ln 10" | **essentially ln 10 = 2.3026** (logged value 2.3022) |
| power at n=3 | "~0.5 pp" | **~0.6 pp at s.d. 0.2; ~0.8 pp for the headline tie** |
| E2 causal wording | abstract/intro/heading/box/conclusion said constant decay "destroys the model" | aligned with §5.8: in this run the constant-coefficient arm **collapsed** to chance while both LR-coupled arms did not, but it was also 2.5x/5x stronger, so profile vs strength remains open. Figure 5 caption now says F is already below at peak |
| AdamW transfer | "does not transfer to AdamW" | "shows no detectable benefit on AdamW at the tested configuration (-0.33 pp, n.s.)" |
| literature scope | "the literature has not tested" / "nobody has tested" | recommended cycles 4-20 epochs; **SGDR's T0=1 is the only sub-2-epoch test, transient and confounded with doubling; no prior work isolates a fixed 1-epoch period** |
| reproducibility scripts | "every table via weekend_report.py" | headline/AdamW/C100/budget via `weekend_report.py`; E2 via `summarize_e2.py`; figures via `make_figures.py` and `plot_e2.py` |
| single-seed labelling | CIFAR-100 AdamW 0.7283 unlabelled | marked **n=1** in the limitations |
| underpowered wording | "statistically indistinguishable" | "no detectable difference" in abstract and conclusion |
| B1/B2 appendix sign | positive values under an ambiguous "B1 vs B2" label | explicitly **"B2 vs B1"** (+0.077/+0.91, B2 wins 2/3), matching §5.8 |

The numerical core was confirmed correct: CIFAR-10/CIFAR-100 peak and final means, all
`t@X` values, all headline paired t statistics, the budget table, the constant-LR sweep,
the E2 table, the loss/accuracy divergence, and the 4.1x/6.5x/5.5x/5.3x/2.5x/1.3x
speedups.

---

## What still needs a run (unchanged from Revision A)

These are the same four optional experiments from the handover §10. None can invalidate
H1-H4, and Revision B now marks them as unrun limitations rather than leaving them
implicit.

| experiment | cost | what it protects |
|---|---|---|
| peak-LR sensitivity {0.4, 1.2} | ~7 h | answers "why 0.8"; tests the configuration-robustness implication |
| step-decay baseline | ~4 h | standard comparison a reviewer may request |
| CIFAR-100 AdamW grid | ~7 h | the +3.88 pp second-dataset margin |
| seeds 3 -> 5 on headline arms | ~4 h | tightens the underpowered final-accuracy ties |

The one open E2 confound (strength-matched constant-decay control, F at wd=1e-4,
~1 h) is now stated explicitly in §5.8; running it would upgrade the E2 claim from
"safe version" to a clean profile-vs-strength attribution.

---

## Files touched

| file | change |
|---|---|
| `docs/PAPER_DRAFT_REV_B.md` | new — full Revision B draft |
| `docs/REVISION_B_RESPONSE.md` | this memo |
| `scripts/make_figures.py` | Fig. 3 now draws +/- 1 s.d. error bars across seeds |
| `figures/fig3_time_to_threshold.png` | regenerated with +/- 1 s.d. error bars across seeds |
| `figures/fig1_cifar10_accuracy.png`, `fig2_schedule.png`, `fig4_cifar100_accuracy.png` | re-rendered from the same JSONs; content unchanged |

`docs/PAPER_DRAFT.md` (Revision A) is left untouched so the two versions can be diffed.
