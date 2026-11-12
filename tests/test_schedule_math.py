"""
Tests for the schedule and the two builders.

Design under test
-----------------
    lr(t) = lr_max * triangle(t % steps_per_epoch) * cosine(t / steps_per_epoch)

applied with stock ``torch.optim.SGD`` + ``torch.optim.lr_scheduler.LambdaLR``.
Weight decay is NOT touched by us: torch scales it by the current lr, which
LambdaLR keeps updated.
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from schedules import cosine_anneal, triangle_wave
from trainer import classify


class TestTriangleWave(unittest.TestCase):
    def test_endpoints(self):
        self.assertAlmostEqual(triangle_wave(0, 100, 0.8, 0.08), 0.08)
        self.assertAlmostEqual(triangle_wave(99, 100, 0.8, 0.08),
                               0.08 + 0.72 * 0.02, places=12)

    def test_peak_at_half_for_even_T(self):
        self.assertAlmostEqual(triangle_wave(50, 100, 0.8, 0.08), 0.8, places=12)

    def test_monotone_up_then_down(self):
        T = 100
        vals = [triangle_wave(t, T, 1.0, 0.0) for t in range(T)]
        up, down = vals[:T // 2 + 1], vals[T // 2:]
        self.assertTrue(all(b >= a for a, b in zip(up, up[1:])))
        self.assertTrue(all(b <= a for a, b in zip(down, down[1:])))


class TestCosine(unittest.TestCase):
    def test_endpoints(self):
        self.assertAlmostEqual(cosine_anneal(0, 30, 0.01), 1.0)
        self.assertAlmostEqual(cosine_anneal(30, 30, 0.01), 0.01)
        self.assertAlmostEqual(cosine_anneal(60, 30, 0.01), 0.01)   # clamped

    def test_monotone_decreasing(self):
        v = [cosine_anneal(e, 30, 0.01) for e in range(31)]
        self.assertTrue(all(b <= a for a, b in zip(v, v[1:])))


class TestLambdaSchedule(unittest.TestCase):
    """LambdaLR must reproduce the analytic schedule exactly, step 0 included."""

    def setUp(self):
        try:
            import torch  # noqa: F401
        except ImportError:
            self.skipTest("torch not installed")

    def test_lambdalr_matches_analytic(self):
        from dual_scheduler import assert_lambdalr_alignment
        worst = assert_lambdalr_alignment(390, 0.8, 1000.0, n_steps=2000,
                                          total_epochs=200, end_ratio=1e-4)
        self.assertLessEqual(worst, 1e-12)

    def test_formula_check_is_torch_free(self):
        from dual_scheduler import assert_same_lr_series
        self.assertLessEqual(
            assert_same_lr_series(390, 0.8, 1000.0, n_steps=500,
                                  total_epochs=200, end_ratio=1e-4), 1e-12)

    def test_first_step_uses_lr_min_not_lr_max(self):
        """The off-by-one that once cost a 7.6e-2 parameter error. LRScheduler
        applies lambda(0) in its constructor, so the first opt.step() already
        uses the schedule's first value (the triangle minimum)."""
        import torch
        from torch.optim import SGD
        from torch.optim.lr_scheduler import LambdaLR
        from dual_scheduler import dual_schedule_lambda

        p = torch.nn.Parameter(torch.zeros(1))
        opt = SGD([p], lr=0.8)
        LambdaLR(opt, dual_schedule_lambda(390, 0.8, 1000.0, 200, 1e-4))
        self.assertAlmostEqual(opt.param_groups[0]["lr"], 0.8 / 1000.0, places=12)

    def test_odd_steps_per_epoch_is_allowed(self):
        """The old CyclicLR design required an even step count; nothing does
        now -- the triangle just peaks between two steps."""
        from dual_scheduler import lr_series
        s = lr_series(195, 0.8, 10.0, 195, 1, 1e-4, "constant")
        self.assertAlmostEqual(min(s), 0.08, places=9)
        self.assertLess(max(s), 0.8)


class TestWeightDecayIsLRCoupled(unittest.TestCase):
    """torch.optim.SGD multiplies weight decay by the current lr (sgd.py:355
    `grad + wd*param`, then :379 `param -= lr*grad`). We rely on that instead of
    applying decay by hand, so the behaviour is pinned here."""

    def setUp(self):
        try:
            import torch  # noqa: F401
        except ImportError:
            self.skipTest("torch not installed")

    def test_torch_sgd_decay_scales_with_lr(self):
        import torch
        # float32: subtracting from 1.0 leaves ~1e-7 absolute resolution
        for lr, expect in ((0.8, 0.8 * 5e-4), (0.08, 0.08 * 5e-4), (0.0, 0.0)):
            p = torch.nn.Parameter(torch.ones(1))
            opt = torch.optim.SGD([p], lr=lr, momentum=0.0, weight_decay=5e-4)
            p.grad = torch.zeros(1)
            opt.step()
            self.assertAlmostEqual(1.0 - p.item(), expect, delta=1e-6)

    def test_internal_optimizer_matches_torch_semantics(self):
        """DualScheduledSGD must decay by exactly wd * lr_effective."""
        import torch
        from sgd_dual import DualScheduledSGD
        p = torch.nn.Parameter(torch.ones(1))
        opt = DualScheduledSGD([p], steps_per_epoch=390, lr=0.8, lr_SWR=1000.0,
                               weight_decay=5e-4)
        p.grad = torch.zeros(1)
        lr_eff = opt.effective_lr(0)
        opt.step()
        self.assertAlmostEqual(1.0 - p.item(), 5e-4 * lr_eff, delta=1e-6)


class TestEngineEquivalence(unittest.TestCase):
    """The custom optimizer and stock SGD + LambdaLR must produce the same
    trajectory for the same seed, in float64 so any gap is logic not rounding."""

    def setUp(self):
        try:
            import torch  # noqa: F401
        except ImportError:
            self.skipTest("torch not installed")

    def _run(self, kind, steps=30):
        import torch
        from dual_scheduler import build_internal, build_method, train_step

        torch.manual_seed(0)
        base = torch.randn(4)

        class M(torch.nn.Module):
            def __init__(s):
                super().__init__()
                s.w = torch.nn.Parameter(base.clone().double())

        m = M()
        build = build_internal if kind == "internal" else build_method
        opt, sch = build(m, steps_per_epoch=10, lr_max=0.8, lr_SWR=1000.0,
                         global_name="cosine", total_epochs=3, end_ratio=1e-4,
                         weight_decay=5e-4, momentum=0.0)
        for _ in range(steps):
            m.w.grad = torch.full_like(m.w, 0.1)
            train_step(opt, sch, None)
        return m.w.detach().clone()

    def test_final_parameters_match_in_float64(self):
        a, b = self._run("internal"), self._run("lambda")
        worst = float((a - b).abs().max())
        self.assertLess(worst, 1e-12,
                        f"float64 disagreement {worst:.3e}: a logic difference")


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


class TestLoggingDoesNotAffectTraining(unittest.TestCase):
    """Logging is observation-only; results must be identical at any log_every."""

    def test_log_every_invariance(self):
        try:
            import json
            import tempfile
            import torch  # noqa: F401
        except ImportError:
            self.skipTest("torch not installed")
        from backends import build as build_backend
        from dual_scheduler import build_method
        from trainer import run_arm

        cfg = json.load(open(os.path.join(os.path.dirname(__file__), "..",
                                          "configs", "tiny.json")))
        cfg["epochs"] = 2
        built = build_backend(cfg)
        tl, _v, _t, make_model, eval_fn, spe = built[:6]

        def builder(model):
            return build_method(model, steps_per_epoch=spe, lr_max=cfg["lr_max"],
                                lr_SWR=cfg["lr_SWR"],
                                global_name=cfg["global_schedule"],
                                total_epochs=cfg["epochs"],
                                end_ratio=cfg["end_ratio"],
                                weight_decay=cfg["weight_decay"])

        out = {}
        for le in (1, 100000):
            with tempfile.TemporaryDirectory() as td:
                out[le] = run_arm("t", make_model, builder, tl, eval_fn,
                                  epochs=2, steps_per_epoch=spe, device="cpu",
                                  log_every=le, out_dir=td, seed=0)
        self.assertEqual(out[1]["final_val_acc"], out[100000]["final_val_acc"])


if __name__ == "__main__":
    unittest.main(verbosity=2)


class TestReviewerMechanisms(unittest.TestCase):
    """Mechanisms required by the supervisor review: period, phase, decay profile."""

    def setUp(self):
        try:
            import torch  # noqa: F401
        except ImportError:
            self.skipTest("torch not installed")

    def test_period_setting_changes_cycle_length(self):
        from dual_scheduler import dual_schedule_lambda
        spe = 390
        # internal returns to the trough inside one epoch: the trough at step 0
        # is not counted, so N cycles per epoch give N-1 crossings.
        for per, expect in ((0.25, 3), (0.5, 1), (1.0, 0), (2.0, 0)):
            f = dual_schedule_lambda(spe, 0.8, 1000.0, 200, 1e-4, "cosine",
                                     period_epochs=per)
            vals = [f(t) for t in range(spe)]
            cycles = sum(1 for i in range(1, len(vals))
                         if vals[i] < 0.002 and vals[i - 1] >= 0.002)
            self.assertEqual(cycles, expect, f"period {per} ep")

    def test_phase_randomisation_breaks_epoch_lock(self):
        from dual_scheduler import dual_schedule_lambda
        spe = 390
        none = [dual_schedule_lambda(spe, 0.8, 1000.0, 50, 1e-4, "cosine")(e * spe)
                for e in range(5)]
        rnd = [dual_schedule_lambda(spe, 0.8, 1000.0, 50, 1e-4, "cosine",
                                    phase_mode="random_per_epoch",
                                    phase_seed=7)(e * spe) for e in range(5)]
        self.assertTrue(all(v < 0.002 for v in none))   # locked to the trough
        self.assertGreater(len({round(v, 4) for v in rnd}), 1)

    def test_constant_shrink_is_lr_independent(self):
        import torch
        from optim_shrink import ConstantShrinkSGD
        p = torch.nn.Parameter(torch.ones(1))
        opt = ConstantShrinkSGD([p], lr=0.0, momentum=0.0, shrink=1e-4)
        p.grad = torch.zeros(1)
        opt.step()
        self.assertAlmostEqual(1.0 - p.item(), 1e-4, delta=1e-6)
