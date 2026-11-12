# LaTeX paper — Revision B

Compiled paper: **[paper.pdf](paper.pdf)** (16 pages).

Source: [`paper.tex`](paper.tex) — the canonical manuscript. Figures are in
[`figures/`](figures/). Build helpers are in [`tools/`](tools/).

## Compile

Requires TeX Live with `pdflatex` (tested with pdfTeX 3.141592653, TeX Live 2025).

```bash
make            # or: latexmk -pdf -interaction=nonstopmode paper.tex
make clean      # remove LaTeX aux files, keep paper.pdf
```

The build uses only standard packages: `article`, `geometry`, `graphicx`,
`booktabs`, `array`, `amsmath`/`amssymb`, `caption`, `microtype`, `hyperref`,
`xcolor`.

## Contents

| path | purpose |
|---|---|
| `paper.tex` | full manuscript (title, abstract, 9 sections, appendix, references) |
| `figures/fig1_cifar10_accuracy.png` | CIFAR-10 test accuracy, 3 seeds, ±1 s.d. |
| `figures/fig2_schedule.png` | the triangular-cycle × cosine-anneal schedule |
| `figures/fig3_time_to_threshold.png` | epochs to 90/93/94%, ±1 s.d. error bars |
| `figures/fig4_cifar100_accuracy.png` | CIFAR-100 test accuracy, 3 seeds, ±1 s.d. |
| `figures/fig_e2_wd.png` | E2 weight-decay conventions (accuracy + weight norm) |
| `tools/md2tex.py` | one-off converter from `../docs/PAPER_DRAFT_REV_B.md` |

## Editing

`paper.tex` is the source of truth for the paper; edit it directly. The converter
is kept only to document how the first LaTeX pass was produced from the Markdown
draft, and to make a re-sync possible if the Markdown changes:

```bash
python3 tools/md2tex.py   # overwrites paper.tex from ../docs/PAPER_DRAFT_REV_B.md
```

Re-running the converter will discard hand edits, so prefer editing `paper.tex`.
