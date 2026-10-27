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
