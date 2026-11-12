# Response plan for the supervisor review

Every numbered item in the review, what we do about it, and where the evidence
lands. Run `bash scripts/supervisor_campaign.sh` (~70 h, resumable) and then
`python scripts/supervisor_report.py`.

## Item-by-item

| review item | severity | action | where |
|---|---|---|---|
| 3.1 LR matching needs verification | Critical | **done, no GPU** | `scripts/verify_lr_math.py` → `docs/review_response/P0_lr_math.txt` |
| 3.2 batch / steps-per-epoch | Major | **text fix, see below** | paper §4 |
| 3.3 statistical power overstated | Major | **done, no GPU** | `scripts/stats_report.py` → `docs/review_response/P0_statistics.txt` |
| 3.4 test-set tuning | Critical | battery **a** (18 runs) | `results_val/`, section A of the report |
| 3.5 "tuned AdamW" too strong | Major | **text fix + optional battery g** | paper §5.7 |
| 4 decay strength vs profile | Major | battery **d** (6 runs) | section D |
| 5 Hessian / EoS | Medium-high | **not run** — paper keeps EoS hypothetical, which the review explicitly permits | — |
| 6.1 cycle phase vs data order | High | battery **e** (6 runs) | section E |
| 6.2 period sweep | Very high | battery **b** (15 runs) | section B |
| 6.3 peak-LR sensitivity | High | battery **c** (12 runs) | section C |
| 6.4 wall-clock | Medium | **done, no GPU** | wall-clock table in `P0_statistics.txt` |
| 6.5 repo/commit/env | Medium | **text fix, see below** | paper §8 |
| 7 wording ("rejected") | Writing | handover item for the writer | — |

## P0 results already in hand (no GPU needed)

**3.1 — the 0.25 objection is resolved, and the reviewer is half right.**

| schedule | peak | min | mean | E[η]/η_max | Ση | Ση² |
|---|---|---|---|---|---|---|
| triangle × cosine | 0.800 | 1.3e-07 | **0.2002** | **0.2503** | 15617 | 6247 |
| cosine, peak 0.4 | 0.400 | 4.0e-05 | **0.2000** | 0.5001 | 15602 | 4680 |

Both numbers are right: 0.25 is E[η]/η_max, and 0.20 includes η_max = 0.8. Ση matches
to 0.1 %. **Ση² differs by 1.335×**, so the paper must not claim "matched
optimisation dynamics" — only "matched mean learning rate".

**3.3 — the reviewer's power correction is right.** 80 % power at n=3 requires
d/σ = **3.26**, confirmed here by Monte Carlo. Detectable final-accuracy difference
at n=3 with the observed paired SD (0.28 pp) is **0.90 pp** — the observed 0.14 pp
difference is far below it. The paper currently overstates what n=3 resolves.

Also: with n=3 the **exact** paired permutation test cannot return p < 0.25 (only 8
sign patterns). Several of the reported t p-values are therefore not meaningful as
exact tests, and the paper must lead with effect sizes, CIs and seed agreement.

## 3.2 — batch handling (text fix, verified from the code)

`drop_last=True`, batch 128, 50,000 images → **390 batches = 49,920 examples per
epoch**; 80 examples are skipped each epoch (a different 80 each epoch, since the
loader shuffles). `steps_per_epoch = 390` is `len(train_loader)`, and the triangular
period is defined in those steps. With the validation split (battery a) the numbers
become 45,000/128 → **351 steps**, 45,120 examples per epoch.

## 3.5 — "tuned AdamW" (text fix)

Replace with **"best tested AdamW configuration"**, and state that both lr and wd
improved monotonically across the grid, so the optimum may lie outside it. Do not
use the CIFAR-100 AdamW comparison as evidence of optimizer superiority (its
hyperparameters were transferred from CIFAR-10).

Optional battery **g** if time permits: CIFAR-100 AdamW grid (lr {1,2,4}e-3 × wd
{0.01,0.05}, 3 seeds ≈ 18 h) would make the +3.88 pp second-dataset margin
defensible rather than provisional.

## 6.5 — reproducibility block for the paper

The repository is `https://github.com/jrnjf/clr_equiv_experiment`. The commit that
produced each table is recorded by `git log --oneline -- docs/`. Environment used
for the reported runs:

* Python 3.14, PyTorch 2.14.1+cu130, torchvision 0.29.1
* 1 × NVIDIA RTX A4000 (17.2 GB), CUDA 13.0
* CIFAR-10 / CIFAR-100 via torchvision, downloaded to `data/`
* ~20 s/epoch at 200 epochs, batch 128, ResNet-18
* every table regenerable: `scripts/weekend_report.py`, `scripts/stats_report.py`,
  `scripts/supervisor_report.py`; figures by `scripts/make_figures.py`

## What the campaign does NOT fix

* **Hessian / Edge of Stability (item 5).** Not run. The paper already describes EoS
  as a candidate mechanism, which the review says is the appropriate restraint. If
  the writer promotes EoS to a claim, this becomes mandatory.
* **Second architecture.** Still ResNet-18 only; the review does not raise it, but a
  main-track reviewer might.
* **Cycle-period sweep at other datasets.** Battery b is CIFAR-10 only.
