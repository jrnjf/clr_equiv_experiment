# LaTeX paper — Revision C

Compiled paper: **[paper.pdf](paper.pdf)** (19 pages).

Source: [`paper.tex`](paper.tex) — the canonical manuscript. Figures are in
[`figures/`](figures/). Build helpers are in [`tools/`](tools/); the previous revision
is preserved in [`archive/`](archive/).

## Compile

Requires TeX Live with `pdflatex` (tested with pdfTeX 3.141592653, TeX Live 2025).

```bash
make            # or: latexmk -pdf -interaction=nonstopmode paper.tex
make clean      # remove LaTeX aux files, keep paper.pdf
```

Only standard packages are used: `article`, `geometry`, `graphicx`, `booktabs`,
`array`, `amsmath`/`amssymb`, `caption`, `microtype`, `hyperref`, `xcolor`.

## Contents

| path | purpose |
|---|---|
| `paper.tex` | full Revision C manuscript (title, status note, abstract, 9 sections, 2 appendices, 14 references) |
| `figures/fig2_schedule.png` | the triangular-cycle × cosine-anneal schedule (Fig. 1) |
| `figures/fig1_cifar10_accuracy.png` | CIFAR-10 test accuracy, shaded ±1 s.d. (Fig. 2) |
| `figures/fig3_time_to_threshold.png` | epochs to 90/93/94%, ±1 s.d. error bars (Fig. 3) |
| `figures/fig4_cifar100_accuracy.png` | CIFAR-100 test accuracy, 3 seeds (Fig. 4) |
| `figures/fig5_decay.png` | decay strength vs profile, conditions A-D (Fig. 5) |
| `figures/fig6_period_sweep.png` | cycle-period sweep, final accuracy and t@94 (Fig. 6) |
| `figures/fig7_peak_lr.png` | peak-learning-rate sweep (Fig. 7) |
| `archive/paper_revB.tex`, `archive/paper_revB.pdf` | Revision B, preserved |
| `tools/md2tex.py` | one-off converter from `../docs/PAPER_DRAFT_REV_C.md` |

## Editing

`paper.tex` is the source of truth for the paper; edit it directly. The converter is
kept only to document how the LaTeX pass was produced from the Markdown draft:

```bash
python3 tools/md2tex.py   # overwrites paper.tex from ../docs/PAPER_DRAFT_REV_C.md
```

Re-running the converter discards hand edits, so prefer editing `paper.tex`.

## Status

Revision C incorporates the supervisor-requested experiments except the validation/test
separation (battery a, ~20 h), which is still outstanding. See §4.4 and §7 of the paper
and `docs/REVISION_C_RESPONSE.md`. Do not submit externally until battery a runs.
