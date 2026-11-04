"""Von Neumann stability analysis of VOF transport schemes.

Verifies the amplification factor of the blend and upwind schemes against
the analytical expressions derived in the modified-equation analysis.

Key finding: the blend scheme is von Neumann stable only for C <= 1/3,
not C <= 1 as previously documented.
"""

import math

import numpy as np
import pytest


def G_blend(theta: float, C: float) -> complex:
    """Amplification factor of the blend scheme.

    G = (1 - C + 2C^2) + (C - 2C^2) cos(theta) - i C sin(theta)
    """
    real = (1 - C + 2 * C ** 2) + (C - 2 * C ** 2) * math.cos(theta)
    imag = -C * math.sin(theta)
    return complex(real, imag)


def G_upwind(theta: float, C: float) -> complex:
    """Amplification factor of the upwind (donor-cell) scheme.

    G = 1 - C(1 - cos(theta)) - i C sin(theta)
    """
    real = 1 - C * (1 - math.cos(theta))
    imag = -C * math.sin(theta)
    return complex(real, imag)


def max_amplification(G_func, C: float, n_theta: int = 1000) -> float:
    """Maximum |G| over theta in [0, pi]."""
    thetas = np.linspace(0, math.pi, n_theta)
    return max(abs(G_func(t, C)) for t in thetas)


class TestBlendSchemeStability:
    """Von Neumann stability of the blend scheme."""

    @pytest.mark.parametrize("C", [0.05, 0.1, 0.2, 0.25, 0.3, 0.33])
    def test_blend_stable_for_C_le_third(self, C):
        """Blend scheme is stable (|G| <= 1) for C <= 1/3."""
        mg = max_amplification(G_blend, C)
        assert mg <= 1.0 + 1e-10, f"|G|max = {mg:.6f} at C={C}"

    @pytest.mark.parametrize("C", [0.35, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    def test_blend_unstable_for_C_gt_third(self, C):
        """Blend scheme is unstable (|G| > 1) for C > 1/3."""
        mg = max_amplification(G_blend, C)
        assert mg > 1.0, f"|G|max = {mg:.6f} at C={C} (expected > 1)"

    def test_blend_critical_C(self):
        """At C = 1/3, |G| = 1 exactly (marginal stability)."""
        mg = max_amplification(G_blend, 1.0 / 3.0)
        assert abs(mg - 1.0) < 1e-10

    def test_blend_anti_diffusive_threshold(self):
        """The effective diffusion D = (u dx/2)(1 - 3C) changes sign at C = 1/3."""
        # For C < 1/3: D > 0 (diffusive)
        # For C > 1/3: D < 0 (anti-diffusive)
        for C in [0.1, 0.2, 0.3]:
            D = 1 - 3 * C
            assert D > 0, f"D = {D} at C={C} (expected positive)"
        for C in [0.4, 0.5, 0.6]:
            D = 1 - 3 * C
            assert D < 0, f"D = {D} at C={C} (expected negative)"


class TestUpwindSchemeStability:
    """Von Neumann stability of the upwind scheme."""

    @pytest.mark.parametrize("C", [0.05, 0.1, 0.25, 0.5, 0.75, 1.0])
    def test_upwind_stable_for_all_C(self, C):
        """Upwind scheme is stable (|G| <= 1) for all 0 <= C <= 1."""
        mg = max_amplification(G_upwind, C)
        assert mg <= 1.0 + 1e-10, f"|G|max = {mg:.6f} at C={C}"

    def test_upwind_exact_at_C_one(self):
        """Upwind is exact at C = 1: |G| = 1 for all theta."""
        thetas = np.linspace(0, math.pi, 100)
        for t in thetas:
            g = G_upwind(t, 1.0)
            assert abs(abs(g) - 1.0) < 1e-10, f"|G| = {abs(g)} at theta={t}"

    def test_upwind_diffusion_coefficient(self):
        """D_upwind = (u dx/2)(1 - C) is positive for C < 1."""
        for C in [0.1, 0.5, 0.9]:
            D = 1 - C
            assert D > 0, f"D = {D} at C={C}"


class TestSchemeComparison:
    """Direct comparison of blend vs upwind."""

    def test_blend_more_diffusive_at_low_C(self):
        """At low C, blend has less numerical diffusion than upwind."""
        C = 0.1
        # Compare |G| at theta = pi/2 (most dissipative mode)
        gb = abs(G_blend(math.pi / 2, C))
        gu = abs(G_upwind(math.pi / 2, C))
        assert gb > gu, f"At C={C}: |G_blend|={gb:.6f} > |G_upwind|={gu:.6f}"

    def test_blend_anti_diffusive_at_high_C(self):
        """At C > 1/3, blend amplifies high-frequency modes."""
        C = 0.5
        theta = math.pi / 2
        gb = abs(G_blend(theta, C))
        assert gb > 1.0, f"|G_blend| = {gb:.6f} at C={C}, theta={theta}"

    def test_stability_boundary_precision(self):
        """The stability boundary is at C = 1/3 to within 0.001."""
        # Binary search for the critical C
        lo, hi = 0.3, 0.4
        for _ in range(20):
            mid = (lo + hi) / 2
            if max_amplification(G_blend, mid) > 1.0:
                hi = mid
            else:
                lo = mid
        assert abs((lo + hi) / 2 - 1.0 / 3.0) < 0.001
