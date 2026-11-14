# Revision C — response to the supervisor review

**Deliverable:** [`PAPER_DRAFT_REV_C.md`](PAPER_DRAFT_REV_C.md) (Markdown source),
[`../paper_latex/paper.tex`](../paper_latex/paper.tex) and
[`../paper_latex/paper.pdf`](../paper_latex/paper.pdf) (compiled, 19 pages).

**Evidence base:** `SUPERVISOR_RESULTS.md`, `HANDOVER_ADDENDUM.md`,
`REVIEW_RESPONSE_PLAN.md`, `review_response/P0_*.txt`, the updated `WEEKEND_RESULTS.md`,
and the raw JSONs in `results/`. Revision B is archived at
`../paper_latex/archive/paper_revB.tex` / `.pdf`.

---

## 0. Status in one line

All requested experiments **except the validation/test separation (review item 3.4)**
are incorporated. That experiment is critical and outstanding; §4.4 and §7 of the
paper mark the affected numbers as provisional, and the paper states it is not ready
for external submission until battery a runs.

---

## 1. What changed in the science

Three earlier claims changed materially:

| Revision B claim | Revision C status | evidence |
|---|---|---|
| the one-epoch / short-period cycle is the contribution | **refuted** — final accuracy is flat across periods 0.25-8; longer periods are better at 93-94% | period sweep (n=3 each, period 1.0 n=5) |
| "decay must track the learning rate" | **refuted as stated** — strength-matched LR-independent decay reaches 94.1%; the collapse at 5x strength is a strength effect | conditions A-D |
| "4.1x" and "6.5x" from n=3 | both hold, restated with five seeds: 16.2 vs 69.2 ep (4.3x) and 16.2 vs 105.8 ep (6.5x) | five-seed runs (seeds 3-4 added) |

Two earlier wording claims were softened:

* "rejected" → **"not supported under the tested control"** everywhere.
* "the effect requires momentum-free SGD" → **"the effect was observed with
  momentum-free SGD, not with momentum 0.9; intermediate values untested"**.
* "tuned AdamW" → **"best tested AdamW configuration"**.

One new finding is a small deficit, not a tie: with five seeds the standard momentum
recipe ends **0.28 pp ahead on final accuracy** (95.41% vs 95.14%, 95% CI -0.53 to
-0.03) and 0.26 pp ahead on peak. The paper now says this explicitly.

---

## 2. Item-by-item response

### 3.1 Learning-rate matching — **resolved, with the reviewer's caveat stated**

Audited from the actual `LambdaLR` over all 78,000 steps. The reviewer's 0.25 is
$E[\eta]/\eta_{\max}$; 0.20 is $E[\eta]$ with $\eta_{\max}=0.8$. Both are correct.
$\sum\eta$ matches to 0.1%; $\sum\eta^2$ differs by **1.335x**. The paper now reports
peak/min/mean, $E[\eta]/\eta_{\max}$, $\sum\eta$, $\sum\eta^2$, and mean per-step
shrink, and claims only a **matched mean learning rate**, not matched optimisation
dynamics. (§5.1, table.)

### 3.2 Batch size / steps per epoch — **resolved**

`drop_last=True`; 50,000/128 → 390 steps = **49,920 examples per epoch**; 80 examples
skipped, a different 80 each epoch because the loader shuffles. `steps_per_epoch = 390
= len(train_loader)`. The pending validation split changes this to 351 steps / 45,120
examples. (§4.1.)

### 3.3 Statistical power — **corrected; five seeds added**

* Power corrected to **$d/\sigma = 3.26$** at n=3 (not 2.84); 1.70 at n=5.
* Table of paired standard deviations and detectable differences: 0.90 pp at n=3
  (s.d. 0.28), 0.59 pp at n=5 (s.d. 0.35).
* **Per-seed differences, 95% CIs, paired t p-values, exact sign-flip permutation
  p-values, seed agreement**, in the main text and Appendix A.
* The exact permutation test's n-dependent floor is stated explicitly: **p >= 0.25 at
  n=3, p >= 0.0625 at n=5**. The paper therefore leads with effect sizes and CIs, not
  with "exact p < 0.05".
* Five seeds on the three headline arms; CIs and seed counts now appear in every
  relevant table. (§4.2, §5.1, §5.4, Appendix A.)

### 3.4 Test-set tuning — **NOT RESOLVED (critical, outstanding)**

The paper does not pretend otherwise. §4.4 states that all arms were selected on the
test split, that battery a is unrun, and that the absolute accuracy numbers are
provisional. §7 lists it as limitation #1. The command is recorded in §8. This is the
one item that blocks external submission.

### 3.5 AdamW — **wording fixed**

"Best tested AdamW configuration" everywhere; the monotone grid and outside-grid
optimum are stated; the CIFAR-100 transferred-hyperparameter caveat is explicit; the
+3.88 pp second-dataset margin is called provisional. (§5.7, §7.)

### 4. Decay strength vs profile — **new experiment, claim rewritten**

Four conditions A-D: LR-coupled oscillating (method), LR-coupled smooth, LR-independent
strength-matched (1.0e-4), LR-independent 5x strong (5.0e-4). Condition C reaches
**94.1%** and does not collapse; D collapses to 10.45%. New claim: **excessive decay
destroys training regardless of profile; LR coupling bounds the effective decay.** At
matched strength the coupled profile is worth ~1.0 pp at the end (95.1% vs 94.1%, 95%
CI -1.42 to -0.71 on the common seeds). The coefficient-versus-effective-shrinkage
distinction is explained explicitly, including the fact that a literal
`weight_decay=5e-4` in PyTorch SGD is already LR-coupled. (§5.8, Fig. 5.)

### 5. EoS — **kept hypothetical; no overclaim**

The paper retains EoS as a candidate mechanism, states exactly which quantities were
not measured (Hessian eigenvalue, sharpness, gradient norms, displacement), notes that
EoS theory is fixed-step full-batch GD, and lists the power-iteration experiment that
would test it. §6.6 says the EoS claim loses its support if sharpness does not track
$2/\eta$. No new theoretical claims were added. (§1, §6.3, §6.6, §7.)

### 6.1 Cycle phase vs data order — **new experiment**

Random initial phase: final 95.20%, t@90 17.3; random phase every epoch: 95.14%, t@90
18.7; locked: 95.14%, t@90 16.2. Differences <= 0.06 pp final and <= 3 epochs to 90%.
No phase artefact. (§5.3, table.)

### 6.2 Period sweep — **new experiment, central rewrite**

Swept 0.25, 0.5, 1, 2, 4, 8 epochs at fixed total iterations. Final accuracy flat
(95.10-95.22%); t@90 flat (15-23); t@94 improves monotonically 147 → 67 epochs. The
one-epoch configuration is **not special**. The title, abstract, introduction,
contributions, discussion and boundary map were rewritten around this. (§5.9, Fig. 6.)

### 6.3 Peak-LR sensitivity — **new experiment**

0.4, 0.6, 0.8, 1.0, 1.2. Final 94.89-95.24%, t@90 16-19 epochs. No knife-edge; 0.8 was
not a lucky pick. "Just a configuration" is now measured. (§5.10, Fig. 7.)

### 6.4 Wall-clock — **new table**

Single-run timings on one RTX A4000, including the equal per-epoch validation cost:
339 s to 90% (method) vs 1357 s (mean-matched cosine), 2806 s (standard momentum),
983 s (best tested AdamW); 1067 s to 93% vs 2606/3917/5139 s. The paper reports the
order of magnitude and does not overclaim a precise ratio. (§5.11.)

### 6.5 Reproducibility — **complete block added**

Repository URL, result commit (`12b68ec`), documentation commit (`c69ae34`), Python
3.14, PyTorch 2.14.1+cu130, torchvision 0.29.1, RTX A4000, CUDA 13.0, ~20 s/epoch, and
the exact scripts that regenerate every table and figure. (§8.)

### 7. Wording — **fixed throughout**

"Rejected" removed from the hypothesis table and prose; each entry now reads "not
supported under the tested control" with the specific control named. Momentum wording
narrowed. Boundary map now distinguishes established boundaries from untested regions;
it explicitly says period, peak LR and phase are **not** boundaries, while decay
strength, momentum, optimizer family and the accuracy ceiling are.

### References — **fixed**

[8] now has full authors (Wang, Xu, Zhao, Tao), venue, volume/pages and arXiv id; [10]
now has full authors (Wen, Li, Wang, Hall, Liang, Ma). [5], [9], [11] were given fuller
metadata. All 14 references retain URLs.

### Abstract — **seed limitation made prominent**

The abstract now says: "All headline speed comparisons use five seeds; accuracy
comparisons remain underpowered at this scale, and we report them as no detectable
difference rather than as equality." (§Abstract.)

---

## 3. What the revision deliberately does not do

* It does not claim the schedule is novel.
* It does not claim final-accuracy superiority; it now states the standard recipe's
  small final and peak advantage.
* It does not promote EoS to a measured mechanism.
* It does not claim to have resolved test-set leakage. That remains the blocker.

---

## 4. Files

| file | change |
|---|---|
| `docs/PAPER_DRAFT_REV_C.md` | new full manuscript source |
| `docs/REVISION_C_RESPONSE.md` | this memo |
| `paper_latex/paper.tex` | regenerated from Rev C |
| `paper_latex/paper.pdf` | compiled, 19 pages, 0 overfull boxes |
| `paper_latex/archive/paper_revB.tex` / `.pdf` | Revision B preserved |
| `paper_latex/figures/fig5_decay.png` | new decay strength/profile figure |
| `paper_latex/figures/fig6_period_sweep.png` | new period sweep figure |
| `paper_latex/figures/fig7_peak_lr.png` | new peak-LR sweep figure |
| `scripts/make_revC_figures.py` | generates the three new figures |
| `paper_latex/tools/md2tex.py` | now reads `PAPER_DRAFT_REV_C.md` |

## 5. Before external submission

Run battery a (~20 h on the GPU box):

```bash
bash scripts/supervisor_campaign.sh --only a
```

Then re-check §4.4, §5.1-§5.7, §7 and the abstract against `results_val/`. If
validation-based selection picks different hyperparameters, the headline absolute
numbers change and this revision must be updated again. The speed comparisons are
expected to survive because the early-training effect is much larger than any plausible
selection effect, but that is an expectation, not a result.
