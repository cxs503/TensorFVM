"""Verification of the integral boundary layer against exact solutions.

Companion of ``scripts/verify_integral_boundary_layer.py`` (which owns the
Falkner-Skan reference solver and the figures).  Every assertion here is
anchored on an *exact* reference computed in this process, or on a documented
characterisation of a known model limitation:

``Blasius``        exact closed form for the zero-pressure-gradient plate;
``Falkner-Skan``   exact self-similar solutions ``f''' + f f'' + beta(1-f'^2) = 0``
                   for ``U = C x^m``, giving the exact ``H``, ``cf Re_theta/2`` and
                   Thwaites parameter ``lambda`` without any table lookup.

Literature/analytic tolerances used below are 3% for the local laminar fields.
Where the model cannot meet 3% the test asserts the *measured* band and the
limitation is documented in ``docs/verification-integral-boundary-layer/``.
"""
from __future__ import annotations

import importlib.util
import math
from pathlib import Path

import numpy as np
import pytest

from tensorfvm.integral_boundary_layer import (
    h1_of_h,
    h_of_h1,
    ittc57_cf,
    ludwieg_tillmann_cf,
    march_boundary_layer,
)

_SPEC = importlib.util.spec_from_file_location(
    "verify_integral_boundary_layer",
    Path(__file__).resolve().parents[1] / "scripts" / "verify_integral_boundary_layer.py",
)
ref = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(ref)


def plate(Re, n=800, force_turbulent=False, laminar=False, leading_strip=1.0):
    ds = np.full((n, 1), 1.0 / n)
    ds[0, 0] *= leading_strip
    areas = np.full((n, 1), 1.0 / n)
    ve = np.full((n, 1), 1.0)
    return march_boundary_layer(
        ve, ds, areas, nu=1.0 / Re, U=1.0, Re=Re, force_turbulent=force_turbulent,
        transition_Re_theta=(1.0e12 if laminar else 320.0),
    )


# --------------------------------------------------------------------------
# 0. the reference solver itself must reproduce the textbook constants
# --------------------------------------------------------------------------
def test_reference_solver_reproduces_classical_falkner_skan_values():
    zero = ref.falkner_skan(0.0)          # Blasius
    assert zero["fpp_wall"] == pytest.approx(0.469600, abs=2e-5)
    assert zero["theta_eta"] == pytest.approx(0.469600, abs=2e-5)
    assert zero["H"] == pytest.approx(2.5911, abs=2e-3)     # 1.7208 / 0.6641
    one = ref.falkner_skan(1.0)           # sink flow
    assert one["fpp_wall"] == pytest.approx(1.232588, abs=2e-5)
    assert one["tail"] < 1e-8


# --------------------------------------------------------------------------
# 1. laminar flat plate vs Blasius (exact)
# --------------------------------------------------------------------------
def test_laminar_flat_plate_local_fields_within_three_percent():
    Re, n = 1.0e5, 1600
    r = plate(Re, n, laminar=True)
    x = ref.station_coordinates(n)
    Rex = x * Re
    sel = x > 0.02
    theta_ex = 0.6641 * x[sel] / np.sqrt(Rex[sel])
    dstar_ex = 1.7208 * x[sel] / np.sqrt(Rex[sel])
    cf_ex = 0.6641 / np.sqrt(Rex[sel])
    assert np.max(np.abs(r.theta[sel, 0] / theta_ex - 1.0)) < 0.03
    assert np.max(np.abs(r.delta_star[sel, 0] / dstar_ex - 1.0)) < 0.03
    assert np.max(np.abs(r.cf[sel, 0] / cf_ex - 1.0)) < 0.03
    # the integrated coefficient carries the leading-edge quadrature deficit too
    assert r.C_F / (1.328 / math.sqrt(Re)) - 1.0 == pytest.approx(-0.030, abs=0.01)


# --------------------------------------------------------------------------
# 2. Thwaites H(lambda), f(lambda) vs the exact family
# --------------------------------------------------------------------------
def test_thwaites_correlations_against_exact_similarity_solutions():
    for beta, h_tol, f_tol in ((0.0, 0.01, 0.01), (-0.05, 0.02, 0.03), (0.05, 0.03, 0.03)):
        ex = ref.falkner_skan(beta)
        H = float(ref.thwaites_H(ex["lam"]))
        f = float(ref.thwaites_f(ex["lam"]))
        assert abs(H / ex["H"] - 1.0) < h_tol, (beta, H, ex["H"])
        assert abs(f / ex["f"] - 1.0) < f_tol, (beta, f, ex["f"])


def test_thwaites_friction_correlation_degrades_near_separation():
    """Characterisation of a known limitation, not an acceptance criterion.

    The quadratic f(lambda) fit rises to ~2x the exact wall shear at the last
    attached Falkner-Skan solution (beta = -0.19), so local cf is unreliable in
    strong adverse gradients even though theta stays within ~17%.
    """
    ex = ref.falkner_skan(-0.19)
    f = float(ref.thwaites_f(ex["lam"]))
    assert f / ex["f"] > 1.5
    assert abs(float(ref.thwaites_H(ex["lam"])) / ex["H"] - 1.0) < 0.20


# --------------------------------------------------------------------------
# 3. the marched operator on self-similar wedge flows
# --------------------------------------------------------------------------
@pytest.mark.parametrize("beta", [0.0, 0.1, 0.3, 0.5])
def test_marched_wedge_matches_exact_similarity(beta):
    r, U, s = ref.march_wedge(beta, 5.0e5, 2000)
    ex = ref.falkner_skan(beta)
    nu = 1.0 / 5.0e5
    theta_ex = math.sqrt(2.0 * nu * s / ((1.0 + beta) * U)) * ex["theta_eta"]
    cf_ex = 2.0 * ex["fpp_wall"] * math.sqrt(nu * (1.0 + beta) / (2.0 * U * s))
    assert r.theta[-1, 0] / theta_ex - 1.0 == pytest.approx(0.0, abs=0.035)
    assert r.cf[-1, 0] / cf_ex - 1.0 == pytest.approx(0.0, abs=0.12)
    assert r.H[-1, 0] / ex["H"] - 1.0 == pytest.approx(0.0, abs=0.07)


# --------------------------------------------------------------------------
# 4. laminar separation criterion
# --------------------------------------------------------------------------
def test_laminar_separation_criterion_vs_exact_falkner_skan():
    beta_s, ex = ref.separation_beta()
    assert beta_s == pytest.approx(-0.198838, abs=1e-3)      # classical value
    assert ex["H"] == pytest.approx(4.03, abs=0.05)
    lam_exact = ex["lam"]
    assert lam_exact == pytest.approx(-0.0774, abs=3e-3)
    zero = float(sorted(np.roots([-1.8, 1.57, 0.22]))[0])
    # documented deviation: the module separates at a much deeper lambda
    assert zero == pytest.approx(-0.12283, abs=1e-3)
    assert zero / lam_exact - 1.0 > 0.5


def test_laminar_branch_never_flags_separation():
    """Characterisation: ``separated`` is only ever set by the turbulent branch."""
    Re, n = 2.0e5, 400
    s = np.linspace(0.0, 1.0, n)
    ds = np.r_[s[1] - s[0], np.diff(s)]
    ve = np.maximum(1.0 - s, 0.05)[:, None]          # strong deceleration
    r = march_boundary_layer(ve, ds[:, None], ds[:, None], nu=1.0 / Re, U=1.0, Re=Re,
                             transition_Re_theta=1.0e12)
    assert not r.separated.any()
    # separation shows up as cf clipped to zero, never as a flag
    assert float(np.mean(r.cf[:, 0] == 0.0)) > 0.5


# --------------------------------------------------------------------------
# 5. closure relations
# --------------------------------------------------------------------------
def test_h1_inverse_is_a_true_inverse_across_both_branches():
    """Regression: the branch split used to return H ~ 12 for H = 4."""
    H = np.linspace(1.15, 4.0, 120)
    back = h_of_h1(h1_of_h(H))
    assert np.max(np.abs(back / H - 1.0)) < 1e-9
    # the march clips H1 to this bound, so a wrong branch faked H > H_max there
    assert float(np.atleast_1d(h_of_h1(np.array([3.35])))[0]) == pytest.approx(3.745, abs=0.15)


def test_ludwieg_tillmann_monotonic_in_H_and_Re_theta():
    assert ludwieg_tillmann_cf(1.4, 1.0e3) > ludwieg_tillmann_cf(2.4, 1.0e3)
    assert ludwieg_tillmann_cf(1.4, 1.0e3) > ludwieg_tillmann_cf(1.4, 1.0e5)


# --------------------------------------------------------------------------
# 6. turbulent branch against the classical engineering lines
# --------------------------------------------------------------------------
def test_turbulent_flat_plate_vs_ittc57_and_power_law():
    for Re in (1.0e6, 1.0e7, 1.0e8):
        r = plate(Re, 1600, force_turbulent=True)
        rel = r.C_F / ittc57_cf(Re) - 1.0
        assert abs(rel) < 0.15                       # existing acceptance band
        assert rel < -0.04                           # measured systematic deficit
        assert not r.separated.any()
    Re, n = 1.0e7, 1600
    r = plate(Re, n, force_turbulent=True)
    x = ref.station_coordinates(n)
    Rex = x * Re
    sel = x > 0.02
    cf_pow = 0.0592 * Rex[sel] ** (-0.2)
    assert np.mean(np.abs(r.cf[sel, 0] / cf_pow - 1.0)) < 0.06
    assert np.max(np.abs(r.cf[sel, 0] / cf_pow - 1.0)) < 0.20
    th_pow = 0.036 * x[-1] * Rex[-1] ** (-0.2)
    assert abs(r.theta[-1, 0] / th_pow - 1.0) < 0.05
    assert 1.25 < r.H[-1, 0] < 1.60


# --------------------------------------------------------------------------
# 7. leading-edge quadrature and degenerate inputs
# --------------------------------------------------------------------------
def test_integrated_friction_convergence_is_leading_edge_limited():
    Re = 1.0e5
    exact = 1.328 / math.sqrt(Re)
    err = {}
    for n in (200, 800, 3200):
        err[n] = plate(Re, n, laminar=True).C_F / exact - 1.0
    assert err[200] < err[800] < err[3200] < 0.0     # monotone towards the bias
    assert err[3200] - err[200] > 0.02               # ~n^-1/2, not n^-1
    assert err[3200] > -0.03                         # residual Thwaites bias


def test_zero_length_first_strip_is_benign_after_branch_fix():
    """Regression: a zero-length leading strip used to fake 2.4x the friction."""
    good = plate(1.0e7, 800, force_turbulent=True, leading_strip=1.0)
    bad = plate(1.0e7, 800, force_turbulent=True, leading_strip=0.0)
    assert bad.C_F / good.C_F == pytest.approx(1.0, abs=0.05)
    assert not bad.separated.any()
    assert bad.theta[-1, 0] / good.theta[-1, 0] == pytest.approx(1.0, abs=0.05)