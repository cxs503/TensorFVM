"""Gates for the two-phase VOF chain: well-balanced hydrostatics + anchors.

The hydrostatic negative control is the cheapest exact-solution test of a
free-surface solver: a two-layer column at rest must stay at rest.  These tests
pin both sides of the defect that the control exposed --

* the **closed-form** hydrostatic profile (``discrete=False``) is inconsistent
  with the face density the momentum equation uses whenever the interface does
  not sit exactly on a cell face, so the column drifts (kept as the negative
  control: the failure is asserted, not hidden);
* the **balanced** profile (``discrete=True``) integrates the same face
  densities, so the discrete equilibrium ``dp/dy = -rho_face g`` holds exactly
  and the rest state survives to round-off for any interface offset.

The analytic anchors (free-fall surface, Ritter front) and the reference-data
kinematics are pure functions and are unit-tested without a solver.
"""

import json
import math
from pathlib import Path

import pytest
import torch

from tensorfvm.two_phase_solver import GridConfig, TwoPhaseConfig, TwoPhaseSolver
from tensorfvm.verification.vof_chain import (
    HYDROSTATIC_VELOCITY_TOL,
    HydrostaticRestConfig,
    run_hydrostatic_rest,
    stratified_alpha,
)
from tensorfvm.verification.vof_chain_anchor import (
    freefall_gate,
    freefall_surface,
    freefall_window,
    front_onset_delay,
    martin_moyce_kinematics,
    ritter_front,
    ritter_profile,
)

G, RHO_L, RHO_G = 9.81, 1000.0, 1.2


def _column(nx=24, ny=12, offset=0.0, sigma=0.0, init="discrete", scheme="plic",
            steps=20, length=0.24, height=0.12):
    cfg = HydrostaticRestConfig(
        name="rest", nx=nx, ny=ny, length=length, height=height,
        offset_dy=offset, sigma=sigma, init=init, vof_scheme=scheme,
        n_steps=steps)
    return run_hydrostatic_rest(cfg)


# --------------------------------------------------------------------------- #
# The fixed behaviour: rest state holds to round-off
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("offset", [0.0, 0.3, 0.5, 0.77])
def test_balanced_init_keeps_the_column_at_rest(offset):
    """G1/G2: any interface position, PLIC transport, |u| stays at round-off."""
    run = _column(offset=offset)
    assert run["max_velocity_m_s"] <= HYDROSTATIC_VELOCITY_TOL, run
    assert run["max_alpha_displacement"] <= 1e-12, run
    assert abs(run["volume_drift"]) <= 1e-12, run


@pytest.mark.parametrize("sigma", [0.0, 0.072])
def test_balanced_init_keeps_the_column_at_rest_with_surface_tension(sigma):
    """A flat interface has zero curvature, so the balanced CSF force vanishes."""
    run = _column(offset=0.5, sigma=sigma)
    assert run["max_velocity_m_s"] <= HYDROSTATIC_VELOCITY_TOL, run
    assert run["static_balance_residual_pa_per_m"] <= 1e-8, run


@pytest.mark.parametrize("scheme", ["plic", "upwind"])
def test_rest_state_is_scheme_independent(scheme):
    """The equilibrium is a property of the momentum balance, not the transport."""
    run = _column(offset=0.3, scheme=scheme)
    assert run["max_velocity_m_s"] <= HYDROSTATIC_VELOCITY_TOL, run


# --------------------------------------------------------------------------- #
# The defect: closed-form profile is only consistent for an aligned interface
# --------------------------------------------------------------------------- #
def test_closed_form_profile_drifts_for_an_unaligned_interface():
    """Negative control: the legacy profile leaves rho*g on the interface face."""
    run = _column(offset=0.5, init="analytic")
    assert run["static_balance_residual_pa_per_m"] > 1.0, run
    assert run["max_velocity_m_s"] > 1e-4, run
    # the balanced run on the *same* geometry is at round-off, so the two
    # numbers differ by >10 orders of magnitude
    balanced = _column(offset=0.5, init="discrete")
    assert balanced["max_velocity_m_s"] < run["max_velocity_m_s"] * 1e-8


def test_closed_form_profile_is_exact_for_an_aligned_interface():
    """Grid-aligned interfaces hide the defect -- this is why offsets matter.

    ``offset_dy = 0`` puts the interface on a cell face, where the one-branch-
    per-cell profile coincides with the discrete integration.  The residual is
    then at the level of the pressure-correction round-off, not of rho*g.
    """
    aligned = _column(offset=0.0, init="analytic")
    assert aligned["static_balance_residual_pa_per_m"] < 1e-6, aligned
    assert aligned["max_velocity_m_s"] < 1e-8, aligned


def test_discrete_profile_matches_the_analytic_column_for_a_uniform_layer():
    """The balanced integration must still reproduce rho g (H - y) in the liquid."""
    nx, ny, height = 8, 16, 0.16
    tp = TwoPhaseConfig(enabled=True, rho_liquid=RHO_L, rho_gas=RHO_G,
                        mu_liquid=1e-3, mu_gas=1e-5, gravity=G,
                        top_atmosphere=True, bottom_wall=True)
    solver = TwoPhaseSolver(GridConfig(nx=nx, ny=ny, length=0.16, height=height,
                                       max_iterations=1), tp)
    interface_y = 0.5 * height
    cfg = HydrostaticRestConfig(nx=nx, ny=ny, length=0.16, height=height)
    solver.set_alpha(stratified_alpha(cfg, solver.dy))
    solver.initialize_hydrostatic_pressure(interface_y, discrete=True)
    y = (torch.arange(ny, dtype=torch.float64) + 0.5) * solver.dy
    p_iface = RHO_G * G * (height - interface_y)
    expected = torch.where(
        y < interface_y,
        p_iface + RHO_L * G * (interface_y - y),
        RHO_G * G * (height - y))
    assert torch.allclose(solver.p[:, 0], expected, rtol=0, atol=1e-9)


def test_discrete_profile_requires_alpha():
    tp = TwoPhaseConfig(enabled=True, rho_liquid=RHO_L, rho_gas=RHO_G, gravity=G)
    solver = TwoPhaseSolver(GridConfig(nx=4, ny=4, length=1.0, height=1.0), tp)
    solver.rho_field = None
    with pytest.raises(ValueError):
        solver.initialize_hydrostatic_pressure(0.5, discrete=True)


# --------------------------------------------------------------------------- #
# Analytic anchors (pure functions)
# --------------------------------------------------------------------------- #
def test_freefall_surface_and_window():
    h0, g, a = 0.02, G, 0.01
    assert freefall_surface(0.0, h0, g) == h0
    t = 0.1 * math.sqrt(h0 / g)
    assert freefall_surface(t, h0, g) == pytest.approx(h0 - 0.5 * g * t * t)
    # never below the floor
    assert freefall_surface(10.0, h0, g) == 0.0
    assert freefall_window(a, h0, g) == pytest.approx(a / math.sqrt(g * h0))


def test_freefall_gate_flags_a_late_collapse():
    a, g = 0.01, G
    tau = math.sqrt(a / g)
    good = [{"t_star": s, "time": s * tau,
             "height_star": freefall_surface(s * tau, 2 * a, g) / a}
            for s in (0.05, 0.25, 0.5, 0.65)]
    assert freefall_gate(good, a, g)["pass"]
    late = [dict(entry, height_star=0.7 * entry["height_star"]) for entry in good]
    assert not freefall_gate(late, a, g)["pass"]
    # points outside the free-fall window are not gated
    outside = [{"t_star": 2.0, "time": 2.0 * tau, "height_star": 1.0}]
    result = freefall_gate(outside, a, g, times_star=(2.0,))
    assert result["points"][0]["reached"] is False


def test_ritter_front_and_profile():
    dam, h0, g = 0.25, 0.05, G
    c0 = math.sqrt(g * h0)
    assert ritter_front(0.0, dam, h0, g) == pytest.approx(dam)
    assert ritter_front(0.1, dam, h0, g) == pytest.approx(dam + 2 * c0 * 0.1)
    # height plateau, wedge, and dry bed
    assert ritter_profile(dam - 1.0, 0.1, dam, h0, g) == pytest.approx(h0)
    assert ritter_profile(dam + 3 * c0 * 0.1, 0.1, dam, h0, g) == 0.0
    mid = dam + 0.5 * c0 * 0.1
    eta = ritter_profile(mid, 0.1, dam, h0, g)
    assert 0.0 < eta < h0


def test_front_onset_delay_reports_the_lag():
    cfg = {"dam_x": 0.25, "h0": 0.05, "c0": math.sqrt(G * 0.05)}
    hist = [{"time": 0.0, "front": 0.0, "pressure_over_hydrostatic": 0.0},
            {"time": 0.02, "front": 0.0, "pressure_over_hydrostatic": 0.4},
            {"time": 0.04, "front": 0.02, "pressure_over_hydrostatic": 0.7}]
    out = front_onset_delay({"config": cfg, "history": hist})
    assert out["onset_time_s"] == pytest.approx(0.04)
    assert out["pressure_over_hydrostatic"] == [0.0, 0.4, 0.7]


def test_martin_moyce_kinematics_flags_only_the_earliest_points():
    """The audit must flag the very early points and keep t* >= 1.5 usable."""
    audit = martin_moyce_kinematics(0.05715, G)
    assert audit["n_unattainable"] > 0
    assert all(row["t_star"] < 1.5 for row in audit["unattainable_points"])
    assert audit["usable_t_star_min"] == 1.5
    assert audit["usable_t_star_min"] is not None
    assert {"front", "height"} <= {row["series"] for row in audit["rows"]}


# --------------------------------------------------------------------------- #
# Published evidence (artifact-gated, skipped when the run is absent)
# --------------------------------------------------------------------------- #
EVIDENCE = Path(__file__).resolve().parents[1] / "docs" / "vof-chain-anchor" / "summary.json"


@pytest.mark.skipif(not EVIDENCE.exists(), reason="chain evidence not generated")
def test_published_summary_gates_are_consistent():
    payload = json.loads(EVIDENCE.read_text())
    gates = payload["gates"]
    # the hydrostatic gates must pass in the published run
    assert gates["G1_hydrostatic_velocity"]["pass"] is True
    assert gates["G2_interface_immobility"]["pass"] is True
    # and the balanced profile must beat the closed-form baseline by >6 decades
    ratio = (gates["G1_hydrostatic_velocity"]["worst_max_velocity_discrete_m_s"]
             / gates["G1_hydrostatic_velocity"]["worst_max_velocity_analytic_m_s"])
    assert ratio < 1e-6
    # every gate reports its own criterion, and all_pass is the conjunction
    asserted = {k: v["pass"] for k, v in gates.items()
                if isinstance(v, dict) and "pass" in v}
    assert gates["all_pass"] == all(asserted.values())
    # the dam-break legs carry a mass gate each
    for entry in gates["G3_mass_conservation"]["legs"]:
        assert abs(entry["final_volume_drift"]) <= 1e-3 or "upwind" in entry["name"]