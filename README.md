# CLR equivalence experiment (E1)

Tests whether `DualScheduledSGD`'s **internal** per-batch triangular schedule is
behaviourally the same object as the same schedule applied **externally** to a
stock `torch.optim.SGD`. This is a prerequisite, not a result: until the two
paths agree, any comparison against other optimizers is confounded by our own
implementation.

## What "the schedule" is

`lr(t) = global(epoch(t)) * triangle(step_in_epoch(t))`

* `triangle` — one full cycle per epoch, period = `steps_per_epoch`. Value is
  `lr_min` at the epoch boundary, rises linearly to `lr_max` at mid-epoch, falls
  back. `lr_min = lr_max / lr_SWR`.
* `global` — cosine anneal from 1.0 down to `end_ratio` (default 0.01).
  Because this multiplies the triangle, the **amplitude shrinks over training**:
  early epochs oscillate in `[0.08, 0.8]`, late epochs in roughly `[0.0008, 0.008]`.

This is Smith's triangular cyclical LR policy (arXiv:1506.01186) in the
dual-schedule composition the research code intends.

## Arms

| Arm | Optimizer | Schedule | Weight decay | Purpose |
|---|---|---|---|---|
| `A_internal` | `DualScheduledSGD` | internal tri × cosine | on effective LR | the method |
| `B1_external_wd_eff` | `SGD` + external | `ExternalDualScheduler` tri × cosine | on effective LR | **must equal A** |
| `B2_external_wd_base` | `SGD` + external | same | on base LR | isolates wd coupling |
| `C_cosine_only` | `SGD` + external | cosine only (`lr_SWR=1`) | on effective LR | isolates the cyclic component |

`A` vs `B1` is pass/fail — same math, independent implementation. Only once they
agree do `B2` and `C` mean anything.

## Two backends

| Backend | Purpose | Data | Device | Cost |
|---|---|---|---|---|
| `tiny` | correctness only — **never a scientific result** | synthetic linearly-separable, 32-d, 2048 samples | CPU | seconds |
| `cifar` | the real experiment | CIFAR-10, ResNet-18 (CIFAR stem) | GPU | hours |

## Two hard constraints found during development

Both are enforced in code, not just documented.

**1. `steps_per_epoch` must be EVEN.** The triangular half-cycle is
`steps_per_epoch/2`. If that is fractional, the peak falls between two steps, and
**no `CyclicLR` configuration reproduces the internal wave** — measured
`max|diff| ≈ 1.5e-2` at `steps_per_epoch=195`. With an even count, the equivalence
is exact to `< 1e-12`. For CIFAR with a 5000-sample val holdout (45000 train),
`batch_size=250` gives exactly **180** steps — even, with no dropped samples.
(`batch_size=256` gives 175 — odd — and is rejected.)

**2. Weight decay is coupled to the *effective* LR** in the original optimizer
(`p.data.add_(p.data, alpha=-wd * lr_effective)`). `torch.optim.SGD` applies decay
against `group['lr']`, which is also the effective LR here, so `B1` matches. Arm
`B2` deliberately disables that decay so the coupling can be measured.

## Setup

```bash
# CPU-only (tiny backend, correctness)
python3 -m venv .venv
.venv/bin/python -m pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision
```

**No `activate` script is needed**: all `make` targets call `.venv/bin/python`
directly, which is equivalent to activating.

On this machine `python3 -m venv` did **not** create `activate` at all (Debian
ships the activation scripts in a separate `python3.14-venv` package). If you
want to `source .venv/bin/activate`, use the bootstrap script, which writes one:

```bash
./scripts/make_venv.sh cpu          # CPU-only torch
./scripts/make_venv.sh cuda cu124   # GPU machine
source .venv/bin/activate
```

```bash
# GPU machine (CIFAR backend)
python3 -m venv .venv
.venv/bin/python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

This machine needed pip bootstrapped manually (`get-pip.py`) because
`ensurepip` is absent.

## Run

```bash
# 1. torch-free validation of config + LR-series equivalence
python3 experiments/run_equivalence.py --config configs/tiny.json --dry-run
python3 experiments/run_equivalence.py --config configs/cifar.json --dry-run

# 2. unit tests (18 tests, torch-free)
python3 -m unittest discover -s tests -v

# 3. THE prerequisite: do the two implementations agree on parameters?
python3 experiments/run_equivalence.py --config configs/tiny.json --check-equivalence

# 4. smoke-run all four arms on the tiny backend
python3 experiments/run_equivalence.py --config configs/tiny.json

# 5. the real run (GPU)
python3 experiments/run_equivalence.py --config configs/cifar.json
```

## Bugs this experiment already caught

The equivalence check earned its keep before any GPU time was spent. Three real
defects, all now fixed and guarded by tests:

1. **Scheduler applied one step late (external path).** `optimizer.step()` ran
   with the *previous* LR, so step 0 used `lr_max=0.8` instead of the triangular
   minimum `0.08` — a 10x error on the first step. Measured effect: 7.6e-2
   relative parameter difference. Fixed by the `prep() -> step() -> advance()`
   ordering in `train_step()`.

2. **Weight decay applied after the gradient update.** torch does
   `w <- w*(1 - wd*lr) - lr*g`; the internal optimizer did `w <- w - lr*g` then
   decayed, so decay acted on the *updated* weight. Algebraically equivalent,
   numerically not: 3.2e-6 after one step, 2.1e-4 after 40. Fixed; residual is
   now 1.19e-7 (float32 noise).

3. **Tiny backend generated a fresh set of class centers per split**, so train
   and validation were different tasks and val accuracy sat at chance (12%).
   The smoke test could not discriminate anything. Fixed with one shared center
   set.

After the fixes, `--check-equivalence` reports a **relative** difference of
~6e-6 on the 12-epoch tiny config. That is float32 accumulation, not logic —
established by rerunning in float64, where the two paths agree to **6.1e-16**
(and the same arm is bit-identical when run twice). Because of that, the pass
gate is `relative <= 1e-4`, not an absolute threshold: the ordering bug measured
7.6e-2 relative and the decay-ordering bug 4e-3, so real disagreement sits two
orders of magnitude above float noise. `tests/` enforces the float64 result.

## Expected smoke output (tiny backend, CPU)

Using `configs/tiny.json` (`lr_max=0.05`, `lr_SWR=10`, 12 epochs, batch 128):

| arm | val_acc (seed 0 and 1) |
|---|---|
| `A_internal` | 0.850 |
| `B1_external_wd_eff` | 0.850 |
| `B2_external_wd_base` | 0.854 |
| `C_cosine_only` | 0.982 |

The useful signal is the *pattern*, not the values: **A and B1 agree exactly**
(two independent implementations of the same math), **B2 differs slightly** (the
decay coupling is detectable), **C differs a lot** (the oscillating schedule
changes dynamics on this task). That the smoke task can separate the arms is the
only thing it is for — `C` winning here says nothing scientific, because on a
separable 32-d toy the triangular wave's low mean LR is just a handicap.

## Interpreting the result

* `--check-equivalence` prints `final-parameter max|A - B1|`. Threshold `1e-6`.
  * **EQ** → implementation correct; proceed to the science.
  * **NOT EQ** → bug. Prime suspects in order: wd coupling (arm `B2` vs `B1`
    should be the *only* difference), scheduler stepped before vs after
    `optimizer.step()`, and odd `steps_per_epoch`.
* Then:
  * `B1` vs `C` → does oscillation add anything over annealing alone?
  * `B1` vs `B2` → does decay-on-effective-LR matter?

Note this experiment establishes **implementational** equivalence only. It cannot
show the schedule is *good*. It also does not address the magnitude confound: all
arms sit at peak 0.8 / mean 0.44, so constant-LR controls (mean-matched 0.440,
RMS-matched 0.487) are still required before claiming oscillation matters rather
than a large average LR.

## Layout

```
src/schedules.py            triangle_wave, cosine_anneal, effective_lr_series
src/reference_schedulers.py torch-free replica of CyclicLR (for the equivalence proof)
src/sgd_dual.py             DualScheduledSGD
src/dual_scheduler.py       ExternalDualScheduler + builders + assert_same_lr_series
src/backends.py             tiny (CPU/synthetic) and cifar (GPU/ResNet-18)
src/trainer.py              per-step logging, divergence classification
experiments/run_equivalence.py
tests/test_schedule_math.py
configs/{tiny,cifar}.json
```

## Provenance

`DualScheduledSGD` is behaviourally derived from the research file
`sgd_dual_schd.py` (sha256 `29b1db30…`), with the internal scheduler factored out
so both paths can share it, and with the weight-decay coupling made an explicit
flag. The original `Jumper` optimizer's epoch-end weight extrapolation is **not**
included — it was removed from the experimental path.
