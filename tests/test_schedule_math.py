"""
Tests for the schedule math. Deliberately torch-free so they run anywhere.

These encode the numeric claims made during the design review, so a future edit
that silently breaks them fails loudly:

  * the internal triangular wave equals PyTorch's CyclicLR algorithm exactly
    when the half-cycle is a whole number of steps;
  * it does NOT when the half-cycle is fractional (the odd-steps_per_epoch trap);
  * internal and external dual schedules produce identical LR series;
  * the realized peak lr is below nominal for odd step counts.
"""

import math
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from dual_scheduler import ExternalDualScheduler, assert_same_lr_series
from reference_schedulers import cyclic_lr
from schedules import (cosine_anneal, effective_lr_series, global_schedule,
                       triangle_wave)
from trainer import classify


class TestTriangleWave(unittest.TestCase):
    def test_endpoints(self):
        # t=0 is the minimum; the wave returns to the minimum at t -> T
        self.assertAlmostEqual(triangle_wave(0, 100, 0.8, 0.08), 0.08)
        self.assertAlmostEqual(triangle_wave(99, 100, 0.8, 0.08),
                               0.08 + 0.72 * 0.02, places=12)

    def test_peak_at_half_for_even_T(self):
        v = triangle_wave(50, 100, 0.8, 0.08)
        self.assertAlmostEqual(v, 0.8, places=12)

    def test_peak_below_nominal_for_odd_T(self):
        # T=195 -> half-cycle 97.5, peak falls between samples
        peak = max(triangle_wave(t, 195, 0.8, 0.08) for t in range(195))
        self.assertLess(peak, 0.8)
        self.assertAlmostEqual(peak, 0.796308, places=5)

    def test_monotone_up_then_down(self):
        T = 100
        vals = [triangle_wave(t, T, 1.0, 0.0) for t in range(T)]
        up, down = vals[:T // 2 + 1], vals[T // 2:]
        self.assertTrue(all(b >= a for a, b in zip(up, up[1:])))
        self.assertTrue(all(b <= a for a, b in zip(down, down[1:])))


class TestCyclicLREquivalence(unittest.TestCase):
    """The internal wave vs the PyTorch CyclicLR algorithm."""

    def _max_diff(self, T, up, down, lr_max=0.8, swr=10.0):
        lr_min = lr_max / swr
        tri = [triangle_wave(t, T, lr_max, lr_min) for t in range(2 * T)]
        cyc = [cyclic_lr(t, lr_min, lr_max, up, down) for t in range(2 * T)]
        return max(abs(a - b) for a, b in zip(tri, cyc))

    def test_exact_for_even_steps(self):
        for T in (100, 50, 194, 196, 200):
            with self.subTest(T=T):
                self.assertLess(self._max_diff(T, T // 2, T // 2), 1e-12)

    def test_not_exact_for_odd_steps(self):
        # no configuration reproduces the internal wave when T is odd
        for up, down in ((97, 98), (98, 97), (97, 97), (98, 98)):
            with self.subTest(up=up, down=down):
                self.assertGreater(self._max_diff(195, up, down), 1e-6)


class TestDualSeries(unittest.TestCase):
    def test_internal_equals_external(self):
        worst = assert_same_lr_series(180, 0.8, 10.0, n_steps=3600,
                                      global_name="cosine", total_epochs=30,
                                      end_ratio=0.01)
        self.assertLessEqual(worst, 1e-12)

    def test_odd_steps_rejected(self):
        with self.assertRaises(ValueError):
            ExternalDualScheduler(optimizer=None, steps_per_epoch=175,
                                  lr_max=0.8, lr_SWR=10.0)

    def test_global_multiplier_shrinks_amplitude(self):
        s = effective_lr_series(180, 0.8, 10.0, 30, "cosine", 0.01)
        ep0, ep_last = s[:180], s[-180:]
        self.assertGreater(max(ep0), 0.7)         # early: full amplitude
        self.assertLess(max(ep_last), 0.02)       # late: shrunk by cosine
        self.assertAlmostEqual(sum(ep_last) / sum(ep0), 0.01, delta=0.002)

    def test_shape_is_scale_invariant_within_an_epoch(self):
        # IMPORTANT: g() is evaluated at fractional epochs, so the global
        # multiplier is NOT constant within an epoch and there is no single
        # factor relating position p of epoch 0 to position p of epoch 29.
        # What does hold: within ONE epoch, g is fixed, so dividing by g must
        # recover exactly the triangle (i.e. the waveform shape, hence SWR,
        # is independent of the global schedule).
        from schedules import cosine_anneal
        spe = 180
        s = effective_lr_series(spe, 0.8, 10.0, 30, "cosine", 0.01)
        epoch = 7
        for pos in (0, 45, 90, 135):
            step = epoch * spe + pos
            g = cosine_anneal(step / spe, 30, 0.01)
            self.assertAlmostEqual(s[step] / g, triangle_wave(pos, spe, 0.8, 0.08),
                                   places=12)


class TestCosine(unittest.TestCase):
    def test_endpoints(self):
        self.assertAlmostEqual(cosine_anneal(0, 30, 0.01), 1.0)
        self.assertAlmostEqual(cosine_anneal(30, 30, 0.01), 0.01)
        self.assertAlmostEqual(cosine_anneal(60, 30, 0.01), 0.01)  # clamped

    def test_monotone_decreasing(self):
        v = [cosine_anneal(e, 30, 0.01) for e in range(31)]
        self.assertTrue(all(b <= a for a, b in zip(v, v[1:])))

    def test_constant_schedule(self):
        g = global_schedule("constant", 30)
        self.assertEqual(g(0), 1.0)
        self.assertEqual(g(29), 1.0)


class TestClassify(unittest.TestCase):
    def test_diverged_nan(self):
        self.assertEqual(classify([1.0, 0.9, float("nan")]), "diverged")

    def test_diverged_blowup(self):
        self.assertEqual(classify([1.0, 0.9, 50.0, 60.0, 70.0]), "diverged")

    def test_recovers_after_spike(self):
        self.assertEqual(classify([1.0, 0.9, 5.0, 0.8, 0.7]), "trained")

    def test_stalled(self):
        self.assertEqual(classify([2.3] * 10, accs=[0.1] * 10), "stalled")

    def test_trained(self):
        self.assertEqual(classify([2.3, 1.5, 1.0], accs=[0.1, 0.5, 0.8]), "trained")


if __name__ == "__main__":
    unittest.main(verbosity=2)


class TestTrainStepOrdering(unittest.TestCase):
    """The ordering bug that broke the first equivalence run must not come back."""

    def test_advance_before_prep_raises(self):
        try:
            import torch  # noqa: F401
        except ImportError:
            self.skipTest("torch not installed")
        from dual_scheduler import ExternalDualScheduler
        sched = ExternalDualScheduler.__new__(ExternalDualScheduler)
        sched._step = 0
        sched._applied = False
        with self.assertRaises(RuntimeError):
            sched.advance()


class TestEquivalenceExactInFloat64(unittest.TestCase):
    """The strongest form of the equivalence claim.

    If the two implementations differed algorithmically, float64 would still
    show it. In float64 they agree to ~1e-16, which proves the float32 residual
    (measured 1.4e-6) is accumulation, not logic. This test skips when torch or
    the tiny backend is unavailable.
    """

    def test_float64_agreement(self):
        try:
            import json
            import torch
        except ImportError:
            self.skipTest("torch not installed")
        import os
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
        from backends import build as build_backend
        from dual_scheduler import build_external, build_internal, train_step

        cfg_path = os.path.join(os.path.dirname(__file__), "..", "configs", "tiny.json")
        cfg = json.load(open(cfg_path))
        cfg["epochs"] = 2                      # keep the test fast
        train_loader, _, _, make_model, _, spe = build_backend(cfg)

        def run(arm):
            torch.manual_seed(0)
            model = make_model().double()
            build = build_internal if arm == "internal" else build_external
            opt, sch = build(model, spe, cfg["lr_max"], cfg["lr_SWR"],
                             cfg["global_schedule"], cfg["epochs"],
                             cfg["end_ratio"], cfg["weight_decay"],
                             wd_on_effective_lr=True)
            crit = torch.nn.CrossEntropyLoss()
            for _ in range(cfg["epochs"]):
                model.train()
                for i, (x, y) in enumerate(train_loader):
                    if i >= spe:
                        break
                    opt.zero_grad(set_to_none=True)
                    # keep loss float32-free: cast logits, keep targets long
                    loss = crit(model(x.double()), y)
                    loss.backward()
                    train_step(opt, sch, loss)
            return torch.cat([p.detach().flatten() for p in model.parameters()])

        a, b = run("internal"), run("external")
        worst = float((a - b).abs().max())
        self.assertLess(worst, 1e-12,
                        f"float64 disagreement {worst:.3e} indicates a real "
                        f"algorithmic difference, not rounding")


class TestLoggingReportsTrueLR(unittest.TestCase):
    """The internal optimizer keeps group['lr'] at lr_max, so a log that reads
    group['lr'] reports a constant 0.8 even though the schedule varies. This
    regression test pins the correct accessor."""

    def test_internal_effective_lr_varies(self):
        try:
            import torch
        except ImportError:
            self.skipTest("torch not installed")
        from schedules import global_schedule
        from sgd_dual import DualScheduledSGD
        m = torch.nn.Linear(2, 1, bias=False)
        opt = DualScheduledSGD(m.parameters(), steps_per_epoch=180, lr=0.8,
                               lr_SWR=10.0,
                               global_schedule=global_schedule("cosine", 30, 0.01))
        vals = [opt.effective_lr(t) for t in range(5)]
        self.assertAlmostEqual(vals[0], 0.08, places=6)
        self.assertTrue(all(b > a for a, b in zip(vals, vals[1:])),
                        "effective LR should rise through the first half-cycle")
        self.assertEqual(opt.param_groups[0]["lr"], 0.8,
                         "group['lr'] is intentionally static for the internal optimizer")


class TestWeightDecayConventions(unittest.TestCase):
    """Both weight-decay conventions must match between the internal optimizer
    and the external composition.

    Regression guard for a real bug: build_external used to pass
    weight_decay=0.0 for wd_on_effective_lr=False and never add the manual term,
    so that arm silently trained with no decay at all.
    """

    def test_both_conventions_match(self):
        try:
            import torch
        except ImportError:
            self.skipTest("torch not installed")
        from dual_scheduler import build_external, build_internal, train_step

        base = torch.randn(6)

        def run(kind, wd_eff):
            class M(torch.nn.Module):
                def __init__(s):
                    super().__init__()
                    s.w = torch.nn.Parameter(base.clone())
            m = M()
            build = build_internal if kind == "internal" else build_external
            opt, sch = build(m, 20, 0.8, 1000.0, "cosine", 3, 1e-4, 5e-4, wd_eff)
            for _ in range(40):
                m.w.grad = torch.full_like(m.w, 0.1)
                train_step(opt, sch, None)
            return m.w.detach().clone()

        for flag in (True, False):
            with self.subTest(wd_on_effective_lr=flag):
                a, b = run("internal", flag), run("external", flag)
                rel = float((a - b).abs().max()) / max(float(a.abs().max()), 1e-12)
                self.assertLess(rel, 1e-4)

    def test_base_lr_decay_is_stronger_than_no_decay(self):
        try:
            import torch
        except ImportError:
            self.skipTest("torch not installed")
        from dual_scheduler import build_internal, train_step

        base = torch.randn(6)

        def run(wd, flag):
            class M(torch.nn.Module):
                def __init__(s):
                    super().__init__()
                    s.w = torch.nn.Parameter(base.clone())
                def params(s):
                    return [s.w]
            m = M()
            opt, sch = build_internal(m, 20, 0.8, 1000.0, "cosine", 3, 1e-4, wd, flag)
            for _ in range(40):
                m.w.grad = torch.full_like(m.w, 0.1)
                train_step(opt, sch, None)
            return float(m.w.detach().norm())

        self.assertLess(run(5e-4, False), run(0.0, True),
                        "decay against base LR should shrink weights more than no decay")


class TestLoggingDoesNotAffectTraining(unittest.TestCase):
    """Logging must be observation-only.

    This also guards the performance refactor: metrics are now computed only on
    logged steps, so a mistake there could silently change training. Results at
    log_every=1 and log_every=100000 must be bit-identical.
    """

    def test_log_every_invariance(self):
        try:
            import json, os, tempfile
            import torch
        except ImportError:
            self.skipTest("torch not installed")
        import sys
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
        from backends import build as build_backend
        from dual_scheduler import build_external
        from trainer import run_arm

        cfg_path = os.path.join(os.path.dirname(__file__), "..", "configs", "tiny.json")
        cfg = json.load(open(cfg_path))
        cfg["epochs"] = 2
        tl, _v, _t, make_model, eval_fn, spe = build_backend(cfg)

        def builder(model):
            return build_external(model, steps_per_epoch=spe, lr_max=cfg["lr_max"],
                                  lr_SWR=cfg["lr_SWR"],
                                  global_name=cfg["global_schedule"],
                                  total_epochs=cfg["epochs"],
                                  end_ratio=cfg["end_ratio"],
                                  weight_decay=cfg["weight_decay"],
                                  wd_on_effective_lr=True)

        out = {}
        for le in (1, 100000):
            with tempfile.TemporaryDirectory() as td:
                out[le] = run_arm("t", make_model, builder, tl, eval_fn,
                                  epochs=2, steps_per_epoch=spe, device="cpu",
                                  log_every=le, out_dir=td, seed=0)
        self.assertEqual(out[1]["final_val_acc"], out[100000]["final_val_acc"])
        self.assertEqual([h["val_acc"] for h in out[1]["val_history"]],
                         [h["val_acc"] for h in out[100000]["val_history"]])


class TestWeightDecayModes(unittest.TestCase):
    """Three conventions, one literal coefficient.

    effective : wd * triangle(step) * cosine(epoch)   oscillating + annealed
    base      : wd * lr_max * cosine(epoch)           smooth + annealed
    none      : wd                                    no lr scaling

    They deliberately have different mean strengths; the arms differ only in
    what the coefficient multiplies.
    """

    def test_base_anneals(self):
        from schedules import cosine_anneal
        early = 0.8 * cosine_anneal(0, 200, 1e-4)
        late = 0.8 * cosine_anneal(199, 200, 1e-4)
        self.assertGreater(early, 100 * late)

    def test_base_differs_from_none(self):
        """Regression: 'base' once multiplied by a CONSTANT lr_max, dropping the
        cosine, which made it arithmetically identical to 'none' (pilot: 0.9406
        twice, wnorm 6.54 vs 6.52)."""
        from schedules import cosine_anneal
        wd = 5e-4
        base_ep50 = wd * 0.8 * cosine_anneal(50, 200, 1e-4)
        none = wd * 1.0
        self.assertNotAlmostEqual(base_ep50, none, places=12)
        self.assertLess(base_ep50, none)      # base has annealed below the constant

    def test_mean_strengths_differ_by_convention(self):
        """Document the consequence of one literal coefficient: 1x / 2x / 5x."""
        from schedules import mean_effective_lr
        wd, spe, T = 5e-4, 390, 200
        mean_eff = mean_effective_lr(spe, 0.8, 1000.0, T, "cosine", 1e-4)
        mean_base = mean_effective_lr(spe, 0.8, 1.0, T, "cosine", 1e-4)
        strengths = {"effective": wd * mean_eff, "base": wd * mean_base,
                     "none": wd * 1.0}
        print("\n  mean per-step shrink:", {k: f"{v:.2e}" for k, v in strengths.items()})
        self.assertAlmostEqual(strengths["base"] / strengths["effective"], 2.0,
                               delta=0.05)
        self.assertAlmostEqual(strengths["none"] / strengths["effective"], 5.0,
                               delta=0.1)
