"""Verification gates for the algebraic VOF transport stencil.

These tests are the executable form of the verification debt recorded in
``tensorfvm.verification.vof_transport_audit`` (see
``docs/verification-vof-transport/report.md``): the linear von Neumann analysis
of the stencil that ``tensorfvm.two_phase_vof.VOFTransport`` implements, the
modified-equation numerical diffusion coefficient against textbook theory, and
the exact-solution periodic benchmarks.

They are deliberately fast (a few seconds on CPU, no solver runs).
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from tensorfvm.two_phase_vof import VOFTransport
from tensorfvm.verification.vof_transport_audit import (
    THIRD,
    advect_1d,
    convergence_1d,
    diffusion_coefficient,
    max_symbol_magnitude,
    measure_symbol,
    sine_field,
    stability_limit,
    symbol,
    theory_diffusion,
    zalesak_rotation,
)


# ---------------------------------------------------------------------------
# 1. The periodic driver must reproduce VOFTransport.advect exactly
# ---------------------------------------------------------------------------
class TestStencilEquivalence:
    def test_1d_periodic_matches_library_interior(self):
        """One library step equals one periodic-driver step away from the ghost ring."""
        ny, nx = 4, 64
        dx = dy = 1.0 / nx
        cour = 0.3
        vof = VOFTransport(ny, nx, dx, dy)
        profile = sine_field(nx, modes=2, amp=0.4)
        alpha = torch.tensor(np.tile(profile, (ny, 1)), dtype=torch.float64)
        u = torch.full((ny, nx + 1), cour * dx, dtype=torch.float64)
        v = torch.zeros((ny + 1, nx), dtype=torch.float64)

        library = vof.advect(alpha, u, v, dt=1.0).numpy()[0]
        driver = advect_1d(profile, cour, 1, scheme="blend")
        assert np.max(np.abs(library[1:-1] - driver[1:-1])) < 1e-14

    def test_upwind_scheme_matches_periodic_driver(self):
        ny, nx = 3, 48
        dx = dy = 1.0 / nx
        cour = 0.4
        vof = VOFTransport(ny, nx, dx, dy, scheme="upwind")
        profile = sine_field(nx, modes=1, amp=0.4)
        alpha = torch.tensor(np.tile(profile, (ny, 1)), dtype=torch.float64)
        u = torch.full((ny, nx + 1), cour * dx, dtype=torch.float64)
        v = torch.zeros((ny + 1, nx), dtype=torch.float64)
        library = vof.advect(alpha, u, v, dt=1.0).numpy()[0]
        driver = advect_1d(profile, cour, 1, scheme="upwind")
        assert np.max(np.abs(library[1:-1] - driver[1:-1])) < 1e-14

    def test_default_scheme_is_unchanged(self):
        """The default construction is bit-for-bit the historical blend scheme."""
        vof_default = VOFTransport(4, 8, 0.125, 0.25)
        vof_explicit = VOFTransport(4, 8, 0.125, 0.25, scheme="blend")
        assert vof_default._blend == 1.0
        np.testing.assert_array_equal(
            vof_default._blend, vof_explicit._blend
        )

    def test_invalid_scheme_rejected(self):
        with pytest.raises(ValueError):
            VOFTransport(4, 8, 0.1, 0.1, scheme="plic")


# ---------------------------------------------------------------------------
# 2. Discrete symbol: measured == closed form
# ---------------------------------------------------------------------------
class TestDiscreteSymbol:
    @pytest.mark.parametrize("scheme", ["blend", "upwind"])
    @pytest.mark.parametrize("cour", [0.1, 0.25, 0.5])
    def test_measured_symbol_matches_closed_form(self, scheme, cour):
        n, modes = 128, 3
        theta = 2.0 * math.pi * modes / n
        steps = 40
        measured = measure_symbol(cour, steps, scheme=scheme, n=n, modes=modes)
        expected = symbol(scheme, theta, cour)
        assert abs(measured - expected) / abs(expected) < 1e-10

    def test_blend_is_stable_only_up_to_one_third(self):
        """|G| <= 1 for C <= 1/3; growth at the grid-scale mode above it."""
        limit = stability_limit("blend")
        assert abs(limit - THIRD) < 2e-3
        assert max_symbol_magnitude("blend", 0.30) <= 1.0 + 1e-12
        assert max_symbol_magnitude("blend", THIRD) <= 1.0 + 1e-12
        for cour in (0.40, 0.50, 0.80, 1.00):
            assert max_symbol_magnitude("blend", cour) > 1.0

    @pytest.mark.parametrize("cour", [0.1, 0.5, 0.9, 1.0])
    def test_upwind_is_stable_to_unit_courant(self, cour):
        assert max_symbol_magnitude("upwind", cour) <= 1.0 + 1e-12

    def test_blend_growth_is_reproduced_by_the_stencil(self):
        """The growth predicted by the symbol is what the stencil does (not a modelling artefact)."""
        n, modes, steps, cour = 128, 3, 60, 0.6
        theta = 2.0 * math.pi * modes / n
        field0 = sine_field(n, modes=modes, amp=0.35)
        field = advect_1d(field0, cour, steps, scheme="blend")
        growth = np.abs(np.fft.rfft(field)[modes]) / np.abs(np.fft.rfft(field0)[modes])
        assert growth == pytest.approx(abs(symbol("blend", theta, cour)) ** steps, rel=1e-6)
        assert growth > 1.0  # unbounded amplification, no clipping active


# ---------------------------------------------------------------------------
# 3. Numerical diffusion vs modified-equation theory (the <3% criterion)
# ---------------------------------------------------------------------------
class TestNumericalDiffusion:
    @pytest.mark.parametrize("cour", [0.1, 0.25])
    @pytest.mark.parametrize("scheme", ["blend", "upwind"])
    def test_measured_diffusion_within_three_percent(self, scheme, cour):
        measured = diffusion_coefficient(cour, scheme=scheme, modes=(1, 2, 3))
        theory = theory_diffusion(cour, scheme=scheme)
        assert abs(measured - theory) / abs(theory) < 0.03

    def test_blend_diffusion_changes_sign_at_one_third(self):
        assert theory_diffusion(0.2, "blend") > 0.0
        assert theory_diffusion(THIRD, "blend") == pytest.approx(0.0, abs=1e-15)
        assert theory_diffusion(0.5, "blend") < 0.0
        from tensorfvm.verification.vof_transport_audit import (
            most_unstable_mode,
            roundoff_blowup_steps,
        )

        measured_low = diffusion_coefficient(0.1, scheme="blend", modes=(1, 2, 3))
        measured_near = diffusion_coefficient(0.3, scheme="blend", modes=(1, 2, 3))
        assert measured_low == pytest.approx(0.35, rel=0.03)
        assert measured_near == pytest.approx(0.05, rel=0.03)
        assert measured_low > measured_near > 0.0
        # above the stability limit the fastest intermediate mode grows per step
        theta_max, g_max = most_unstable_mode("blend", 0.5)
        assert abs(theta_max - math.pi / 2) < 1e-3
        assert g_max == pytest.approx(math.sqrt(1.25), rel=1e-6)
        assert 100 < roundoff_blowup_steps("blend", 0.5) < 400
        assert not math.isfinite(roundoff_blowup_steps("upwind", 0.5))


# ---------------------------------------------------------------------------
# 4. Smooth-field convergence order and the C = 1 exact-shift anchor
# ---------------------------------------------------------------------------
class TestSmoothConvergence:
    @pytest.mark.parametrize("scheme", ["blend", "upwind"])
    def test_first_order_in_l1(self, scheme):
        rows = convergence_1d(scheme, cour=0.25, sizes=(32, 64, 128))
        orders = [row["order_l1"] for row in rows[1:]]
        assert all(0.8 <= order <= 1.2 for order in orders), orders
        assert all(row["l1"] > 0.0 for row in rows)

    def test_upwind_is_exact_at_unit_courant(self):
        probe = sine_field(48, modes=2)
        out = advect_1d(probe, 1.0, 48, scheme="upwind")
        assert np.max(np.abs(out - np.roll(probe, 48))) < 1e-12


# ---------------------------------------------------------------------------
# 5. Zalesak slotted disc: mass conservation and the CFL 0.5 defect
# ---------------------------------------------------------------------------
class TestZalesakRotation:
    def test_upwind_conserves_mass_to_machine_precision(self):
        for cfl in (0.25, 0.5):
            metrics, _ = zalesak_rotation(32, scheme="upwind", cfl=cfl)
            assert metrics["volume_drift_rel"] < 1e-12
            assert 0.0 <= metrics["alpha_min"] and metrics["alpha_max"] <= 1.0

    def test_blend_limiter_manufactures_mass_at_cfl_half(self):
        """Documents the defect: the default scheme corrupts mass at the solver's default CFL."""
        metrics, _ = zalesak_rotation(32, scheme="blend", cfl=0.5)
        assert metrics["volume_drift_rel"] > 0.1
        metrics_ok, _ = zalesak_rotation(32, scheme="blend", cfl=0.25)
        assert metrics_ok["volume_drift_rel"] < 1e-2

    def test_exact_solution_returns_to_initial_field(self):
        """Sanity of the benchmark itself: the rotation is closed after one period."""
        from tensorfvm.verification.vof_transport_audit import zalesak_initial

        initial = zalesak_initial(32)
        assert initial.sum() > 0.0
        assert set(np.unique(initial)) == {0.0, 1.0}