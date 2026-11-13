"""Unit gates for the Sod (1978) survey reproduction.

Fast checks only (seconds): scheme consistency, the exact-Riemann face solver against
its own jump conditions, the conservation budget of every surveyed scheme, and the
positivity failure of the un-stabilised MacCormack -- the finding the artefact reports.
"""

import numpy as np
import pytest

from tensorfvm.verification import sod1978_survey as S


def _uniform(n=20, rho=1.0, u=0.3, p=1.2):
    return S._from_primitive(np.full(n, float(rho)), np.full(n, float(u)),
                             np.full(n, float(p)))


@pytest.mark.parametrize("scheme", S.SCHEMES)
def test_uniform_state_is_preserved(scheme):
    """Consistency: a constant state must not move under any scheme."""
    U = _uniform()
    U1 = S.STEP[scheme](U, 0.05, 0.005)
    assert np.max(np.abs(U1 - U)) < 1e-12


def test_exact_face_state_reproduces_published_star_state():
    """The Godunov face solver must return (rho*_L, u*, p*) of the Sod data."""
    left = np.array([[1.0, 0.0, 1.0]])
    right = np.array([[0.125, 0.0, 0.1]])
    state = S.exact_face_state(left, right)[0]
    assert state[0] == pytest.approx(0.42632, abs=5e-5)
    assert state[1] == pytest.approx(0.92745, abs=5e-5)
    assert state[2] == pytest.approx(0.30313, abs=5e-5)


def test_exact_face_state_is_consistent_with_the_physical_flux():
    """Equal states on both sides => the face flux is the physical flux."""
    U = _uniform()
    prim = np.stack(S.primitives(U), axis=1)
    state = S.exact_face_state(prim[:-1], prim[1:])
    flux = S.flux_from_primitive(state[:, 0], state[:, 1], state[:, 2])
    ref = S.flux_from_primitive(prim[:-1, 0], prim[:-1, 1], prim[:-1, 2])
    assert np.max(np.abs(flux - ref)) < 1e-12


@pytest.mark.parametrize("scheme", S.SCHEMES)
def test_conservation_budget_closes(scheme):
    """Short Sod run: the flux budget closes to round-off for every scheme."""
    run = S.run_scheme(scheme, 60, t_end=0.05)
    assert run["budget_rel_residual"] < 1e-12


@pytest.mark.parametrize("scheme", ["lax_friedrichs", "rusanov", "godunov", "lax_wendroff"])
def test_first_order_schemes_respect_the_density_range(scheme):
    """Monotone schemes must not create new density extrema (S4, graded part)."""
    run = S.run_scheme(scheme, 100, t_end=0.1)
    if S.DESIGN_ORDER[scheme] == 1:
        assert run["rho"].min() >= S.RIGHT["rho"] - 1e-12
        assert run["rho"].max() <= S.LEFT["rho"] + 1e-12


def test_unstabilised_macormack_loses_positivity():
    """Honest FAIL recorded by the artefact: plain MacCormack is not positivity preserving."""
    run = S.run_scheme("macormack", 100, t_end=0.2)
    assert run["nonpositive"] is True
    assert run["t_end"] < 0.05


@pytest.mark.parametrize("scheme", ["lax_wendroff", "macormack"])
def test_second_order_on_the_exact_entropy_wave(scheme):
    """Design order 2 measured on the exact smooth solution (coarse grids, fast)."""
    vals = [S.run_smooth_wave(scheme, nx) for nx in (20, 40)]
    order = S._smooth_orders(vals)[-1]
    assert order > 1.7


def test_module_writes_a_programmatic_report(tmp_path):
    """End-to-end on small grids: artefact + markdown are written and consistent."""
    study = S.run_survey(tmp_path, nx_profile=50, smooth_grids=(20, 40),
                         production_ref=False)
    md = (tmp_path / "REPORT.md").read_text()
    assert len(md) > 3000
    assert "总判定" in md
    assert (tmp_path / "survey_study.json").exists()
    assert study["thresholds_declared_before_run"] is S.THRESHOLDS
