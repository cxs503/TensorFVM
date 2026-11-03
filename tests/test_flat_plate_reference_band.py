"""Independent audit of the flat-plate reference band and the 20 % drag gate.

The Spalart--Allmaras flat-plate benchmark is gated on a single correlation
(``0.074 Re^-1/5``) with a 20 % window.  These tests pin three facts with no
solver in the loop:

1. the correlation band at ``Re_L = 1e5`` (four accepted fully turbulent laws),
2. that the 20 % window is *wider* than that band, so it cannot certify
   accuracy (``gate_is_informative``),
3. that the production gate really does admit such a value -- a drag 18 %
   above the Schlichting reference passes ``compare_flat_plate`` while sitting
   outside the literature band entirely.

Fact 3 is a recorded defect, not a desired property: if the gate is ever
replaced by a band-aware criterion this test must be updated to assert failure.
"""

import math
from types import SimpleNamespace
import unittest

import torch

from tensorfvm.benchmark_flat_plate import ERROR_LIMIT, compare_flat_plate, reference_skin_friction
from tensorfvm.flat_plate_reference import (
    MIN_REYNOLDS,
    band_membership,
    correlation_values,
    gate_is_informative,
    reference_band,
)
from tensorfvm import SolverConfig

RE = 100_000.0


def flat_plate_config(**kwargs):
    options = dict(
        mesh_type="flat-plate",
        turbulence_model="spalart-allmaras",
        cylinder_radius=None,
        nx=16,
        ny=8,
        length=1.0,
        height=0.2,
        reynolds=RE,
        max_iterations=2,
        tolerance=1e-5,
        velocity_relaxation=0.5,
        turbulence_relaxation=0.5,
    )
    options.update(kwargs)
    return SolverConfig(**options)


def synthetic_result(config, skin_friction):
    """A result object whose plate drag *is* the given skin friction."""
    scale = 0.5 * config.density * config.inlet_velocity ** 2 * config.length
    return SimpleNamespace(
        config=config,
        surface_forces={"plate": {"x": skin_friction * scale, "y": 0.0}},
        y=torch.full((config.ny, config.nx), 1e-6, dtype=torch.float64),
        converged=True,
        history=[{"continuity": 1e-9, "momentum": 1e-9,
                  "mass_imbalance": 1e-9, "turbulence": 1e-9}],
    )


class ReferenceBandTests(unittest.TestCase):
    def test_correlations_match_independent_closed_forms(self):
        values = correlation_values(RE)
        expected = {
            "schlichting_0.074": 0.074 * RE ** -0.2,
            "prandtl_schlichting": 0.455 / math.log10(RE) ** 2.58,
            "ittc57": 0.075 / (math.log10(RE) - 2.0) ** 2,
            "white": 0.523 / math.log(0.06 * RE) ** 2,
        }
        self.assertEqual(set(values), set(expected))
        for name, value in expected.items():
            self.assertAlmostEqual(values[name] / value, 1.0, places=14)

    def test_reference_skin_friction_is_one_member_of_the_band(self):
        reference = reference_skin_friction(RE)
        band = reference_band(RE)
        self.assertTrue(band["low"] <= reference <= band["high"])
        self.assertAlmostEqual(reference, 0.0074, places=12)

    def test_band_spread_is_at_least_the_gate_window(self):
        band = reference_band(RE)
        # 0.006911 .. 0.008333 around a 0.00745 mean.
        self.assertAlmostEqual(band["spread_percent"], 19.1, places=1)
        # The window is only 4.5 % wider than the band: 0.20 vs 0.191.
        self.assertAlmostEqual(band["spread"] / ERROR_LIMIT, 0.955, delta=0.01)
        self.assertFalse(gate_is_informative(RE, ERROR_LIMIT))
        self.assertTrue(gate_is_informative(RE, 0.10))

    def test_band_membership_is_inclusive_and_rejects_outliers(self):
        band = reference_band(RE)
        self.assertTrue(band_membership(band["low"], RE))
        self.assertTrue(band_membership(band["high"], RE))
        self.assertFalse(band_membership(band["high"] * 1.001, RE))
        self.assertFalse(band_membership(band["low"] * 0.999, RE))
        with self.assertRaises(ValueError):
            band_membership(float("nan"), RE)

    def test_below_fully_turbulent_reynolds_is_rejected(self):
        with self.assertRaises(ValueError):
            correlation_values(MIN_REYNOLDS * 0.5)
        self.assertEqual(set(reference_band(MIN_REYNOLDS)), set(reference_band(RE)))


class GateNonInformativenessTests(unittest.TestCase):
    """Recorded defect: the 20 % window admits values outside the band."""

    def test_gate_accepts_drag_outside_literature_band(self):
        config = flat_plate_config()
        reference = reference_skin_friction(config.reynolds)
        measured = reference * 1.18
        band = reference_band(config.reynolds)
        self.assertGreater(measured, band["high"])
        self.assertFalse(band_membership(measured, config.reynolds))
        self.assertLess(abs(measured - reference) / reference, ERROR_LIMIT)
        outcome = compare_flat_plate(synthetic_result(config, measured))
        self.assertTrue(outcome["passed"])
        self.assertLess(outcome["relative_error"], ERROR_LIMIT)

    def test_gate_still_rejects_gross_error_outside_the_window(self):
        config = flat_plate_config()
        reference = reference_skin_friction(config.reynolds)
        measured = reference * (1.0 + 2 * ERROR_LIMIT)
        outcome = compare_flat_plate(synthetic_result(config, measured))
        self.assertFalse(outcome["passed"])


if __name__ == "__main__":
    unittest.main()