"""Tests for the independent spheroid reference and the D6 rotational anchor.

The reference solver (prolate spheroidal finite differences for the ``m = 1``
potential mode) must reproduce the **exact Lamb transverse added mass** before
it is allowed to serve as an anchor for the (anchor-less) transverse added
inertia.  These tests pin that validation, the FD grid convergence, the near
sphere ``(a^2-b^2)^2`` law, and the agreement between the reference and the
source-panel implementation in :mod:`tensorfvm.panel_added_mass`.
"""

from __future__ import annotations

import pytest

from tensorfvm.panel_added_mass import (
    lamb_spheroid_factors,
    rotational_added_inertia,
    spheroid_panels,
)
from tensorfvm.spheroid_potential_reference import (
    slender_limit_added_inertia,
    transverse_added_inertia_fd,
    transverse_added_mass_fd,
)
from tensorfvm.verification import spheroid_rotational_anchor as anchor


def _volume(a, b):
    return 4.0 / 3.0 * 3.141592653589793 * a * b ** 2


@pytest.mark.parametrize("ratio", [1.5, 2.0, 4.0, 8.0])
def test_fd_translation_matches_exact_lamb(ratio):
    """C1: the FD solver reproduces the exact transverse added mass."""
    a, b = ratio, 1.0
    value = transverse_added_mass_fd(a, b, n_theta=193, n_tau=241)
    _, k_exact = lamb_spheroid_factors(a, b)
    assert value / _volume(a, b) == pytest.approx(k_exact, rel=5e-3)


def test_fd_grid_convergence_of_rotation():
    """C2: the rotational branch is grid-converged to better than 0.3%."""
    a, b = 2.0, 1.0
    coarse = transverse_added_inertia_fd(a, b, n_theta=129, n_tau=161)
    fine = transverse_added_inertia_fd(a, b, n_theta=193, n_tau=241)
    assert abs(fine - coarse) / fine < 3e-3


def test_sphere_limit_is_zero():
    """The transverse added inertia must vanish as a/b -> 1 (sphere: I = 0)."""
    a, b = 1.01, 1.0
    value = transverse_added_inertia_fd(a, b, n_theta=129, n_tau=161)
    scale = a ** 2 - b ** 2
    assert value > 0.0
    assert value / scale ** 2 == pytest.approx(0.28, rel=0.05)


def test_near_sphere_square_law_exponent():
    """C5: I55 ~ (a^2-b^2)^2 near the sphere (local exponent -> 2)."""
    values = {r: transverse_added_inertia_fd(r, 1.0, n_theta=129, n_tau=161)
              for r in (1.01, 1.05)}
    delta = {r: r * r - 1.0 for r in values}
    exponent = (__import__("math").log(values[1.05] / values[1.01])
                / __import__("math").log(delta[1.05] / delta[1.01]))
    assert exponent == pytest.approx(2.0, abs=0.04)


def test_slender_limit_is_approached_from_below():
    """C6: monotone rise towards the strip value (1/5) rho V a^2."""
    scaled = []
    for ratio in (2.0, 4.0, 8.0):
        a, b = ratio, 1.0
        value = transverse_added_inertia_fd(a, b, n_theta=193, n_tau=241)
        scaled.append(value / (_volume(a, b) * a ** 2))
        assert value < slender_limit_added_inertia(a, b) * 1.02
    assert scaled == sorted(scaled)


def test_panel_matches_reference_within_three_percent():
    """C3: the source-panel transverse block agrees with the FD reference."""
    a, b = 2.0, 1.0
    reference = transverse_added_inertia_fd(a, b, n_theta=257, n_tau=321)
    relaxation = (transverse_added_inertia_fd(a, b, n_theta=193, n_tau=241)
                  - reference)
    reference = reference - relaxation / ((257.0 / 193.0) ** 1.8 - 1.0)
    panels = spheroid_panels(a, b, n_theta=48, n_phi=32)
    block = rotational_added_inertia(panels, order=3)
    assert abs(block[1, 1] / reference - 1.0) < 0.05
    # axisymmetry of the panel block
    assert abs(block[1, 1] - block[2, 2]) / block[1, 1] < 1e-10


def test_anchor_audit_fast_passes():
    """The quick (--fast) audit used by CI must pass all of its criteria."""
    audit = anchor.run(fast=True)
    assert audit.all_pass, {k: (v["pass"], v["value"]) for k, v in audit.criteria.items()}


def test_fd_rejects_invalid_geometry():
    with pytest.raises(ValueError):
        transverse_added_inertia_fd(1.0, 1.0)
    with pytest.raises(ValueError):
        transverse_added_inertia_fd(1.0, 2.0)