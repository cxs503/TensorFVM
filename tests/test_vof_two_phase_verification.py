"""Tests for the two-phase VOF verification surface.

Coverage:

* exact cell geometry (quadrature vs closed-form disk areas);
* the analytic anchors themselves (divergence-free, wall-consistent, exactly
  reversible in time -- verified by integrating marker particles);
* conservative, bounded transport of every scheme on the rotation anchor;
* the GCI helper on synthetic triples (including the non-monotone case, which
  must report ``None`` rather than a clipped order);
* the hydrostatic rest state: the shipped ``void`` model leaves the interior
  free surface O(h) out of balance, the well-balanced treatment holds it to
  round-off;
* the criteria evaluator on a synthetic campaign report.
"""

import math

import torch

from tensorfvm.vof_phase0.advection import AlgebraicVOF
from tensorfvm.vof_verification.campaign import merge_runs, run_spec
from tensorfvm.vof_verification.cases import (
    Transport,
    advance,
    double_vortex_case,
    face_divergence,
    sample_faces,
    single_vortex_case,
    zalesak_case,
    _modulation_mean,
)
from tensorfvm.vof_verification.geometry import (
    analytic_area,
    circle_alpha,
    disk_rect_area,
    zalesak_alpha,
)
from tensorfvm.vof_verification.hydrostatic import run_hydrostatic
from tensorfvm.vof_verification.metrics import field_metrics, gci


# --------------------------------------------------------------------------- #
# geometry
# --------------------------------------------------------------------------- #
def test_disk_rect_area_matches_closed_form():
    r = 0.3
    full = disk_rect_area(0.5, 0.5, r, 0.0, 1.0, 0.0, 1.0)
    assert abs(full - math.pi * r ** 2) / (math.pi * r ** 2) < 1e-9
    half = disk_rect_area(0.5, 0.5, r, 0.0, 0.5, 0.0, 1.0)
    assert abs(half - 0.5 * math.pi * r ** 2) / (0.5 * math.pi * r ** 2) < 1e-9
    quarter = disk_rect_area(0.5, 0.5, r, 0.5, 1.0, 0.5, 1.0)
    assert abs(quarter - 0.25 * math.pi * r ** 2) / (0.25 * math.pi * r ** 2) < 1e-9


def test_disk_rect_area_degenerate_and_disjoint():
    assert disk_rect_area(0.5, 0.5, 0.1, 0.0, 0.2, 0.0, 0.2) == 0.0
    cell = disk_rect_area(0.5, 0.5, 0.1, 0.4, 0.6, 0.4, 0.6)
    assert 0.0 < cell < 0.04


def test_circle_alpha_conserves_area():
    exact = math.pi * 0.15 ** 2
    prev = None
    for n in (32, 64, 128):
        alpha = circle_alpha(n, n)
        area = float(alpha.sum().item()) / (n * n)
        rel = abs(area - exact) / exact
        assert rel < 5e-8
        if prev is not None:
            assert rel <= prev * 1.2 + 1e-12
        prev = rel
        assert 0.0 <= float(alpha.min()) and float(alpha.max()) <= 1.0


def test_zalesak_alpha_matches_analytic_slot_area():
    expected = analytic_area(radius=0.15, slot_half_width=0.025)
    for n in (32, 64, 128):
        alpha = zalesak_alpha(n, n)
        area = float(alpha.sum().item()) / (n * n)
        assert abs(area - expected) / expected < 1e-6
        # slot cells above the centre line must be empty at 128^2
    alpha = zalesak_alpha(128, 128)
    # cell centre (0.5, 0.8) lies inside the slot: volume fraction must vanish
    assert float(alpha[int(0.8 * 128), int(0.5 * 128)]) < 1e-9


# --------------------------------------------------------------------------- #
# analytic anchors
# --------------------------------------------------------------------------- #
def test_velocity_fields_are_divergence_free():
    for case in (zalesak_case(), single_vortex_case(), double_vortex_case()):
        for n in (16, 32):
            nx, ny = (n, n // 2) if case.name == "double_vortex" else (n, n)
            dx, dy = case.lx / nx, case.ly / ny
            u, v = sample_faces(case, nx, ny, 0.37, 0.41)
            assert face_divergence(u, v, dx, dy) < 1e-12


def test_vortex_fields_are_wall_consistent():
    """The vortex fields have zero wall-normal velocity (walls are exact for them)."""
    for case in (single_vortex_case(), double_vortex_case()):
        for n in (16, 32):
            nx, ny = (n, n // 2) if case.name == "double_vortex" else (n, n)
            u, v = sample_faces(case, nx, ny, 0.0)
            assert float(u[:, 0].abs().max()) < 1e-12      # left wall
            assert float(u[:, -1].abs().max()) < 1e-12     # right wall
            assert float(v[0, :].abs().max()) < 1e-12      # bottom wall
            assert float(v[-1, :].abs().max()) < 1e-12     # top wall


def test_zalesak_disk_stays_clear_of_the_walls():
    """Solid-body rotation about the domain centre: no interface near a wall."""
    case = zalesak_case()
    out = advance(case, 48, 48, Transport(scheme="plic", cfl=0.75))
    alpha = out["alpha"]
    assert float(alpha[0, :].abs().max()) < 1e-12
    assert float(alpha[-1, :].abs().max()) < 1e-12
    assert float(alpha[:, 0].abs().max()) < 1e-12
    assert float(alpha[:, -1].abs().max()) < 1e-12


def test_modulation_average_is_exact_and_reverses():
    T = 8.0
    # exact mean over the whole period integrates to zero (reversal property)
    whole = _modulation_mean(0.0, T, T) * T
    assert abs(whole) < 1e-13
    for t0, t1 in ((0.0, 0.5), (2.0, 2.7), (5.1, 6.0)):
        approx = _modulation_mean(t0, t1, T)
        numeric = sum(math.cos(math.pi * (t0 + (i + 0.5) * (t1 - t0) / 2000) / T)
                      for i in range(2000)) / 2000
        assert abs(approx - numeric) < 1e-8


def test_vortex_flow_map_returns_to_start():
    """Integrate marker particles: the prescribed field really is reversible."""
    case = single_vortex_case()
    nx = ny = 32
    dx = dy = 1.0 / nx
    grid = torch.arange(-0.10, 0.101, 0.05, dtype=torch.float64)
    pts = torch.stack(torch.meshgrid(0.5 + grid, 0.75 + grid, indexing="xy"),
                      dim=-1).reshape(-1, 2)
    keep = ((pts[:, 0] - 0.5) ** 2 + (pts[:, 1] - 0.75) ** 2) < 0.1 ** 2
    start = pts[keep].clone()
    pos = start.clone()

    def interp(u, v, p):
        xi = (p[:, 0] / dx).clamp(0.0, nx - 1e-9)
        yi = (p[:, 1] / dy).clamp(0.0, ny - 1e-9)
        i0, j0 = xi.long(), yi.long()
        fx, fy = xi - i0, yi - j0
        uu = (u[j0, i0] * (1 - fx) + u[j0, i0 + 1] * fx) * (1 - fy) \
            + (u[j0 + 1, i0] * (1 - fx) + u[j0 + 1, i0 + 1] * fx) * fy
        vv = (v[j0, i0] * (1 - fy) + v[j0 + 1, i0] * fy) * (1 - fx) \
            + (v[j0, i0 + 1] * (1 - fy) + v[j0 + 1, i0 + 1] * fy) * fx
        return uu, vv

    T, n = case.duration, 300
    dt = T / n
    for k in range(n):
        u, v = sample_faces(case, nx, ny, k * dt, (k + 1) * dt)
        k1x, k1y = interp(u, v, pos)
        mid = pos + 0.5 * dt * torch.stack([k1x, k1y], dim=1)
        k2x, k2y = interp(u, v, mid)
        pos = pos + dt * torch.stack([k2x, k2y], dim=1)
    disp = (pos - start).norm(dim=1)
    assert float(disp.max()) < 5e-3, f"flow map not reversible: {float(disp.max())}"


# --------------------------------------------------------------------------- #
# transport
# --------------------------------------------------------------------------- #
def test_rigid_rotation_is_conservative_and_bounded():
    case = zalesak_case()
    for scheme in ("plic", "fct", "donor"):
        out = advance(case, 32, 32, Transport(scheme=scheme, cfl=0.75))
        m = out["metrics"]
        assert m["mass_drift_relative"] < 1e-10, scheme
        assert m["worst_step_drift_relative"] < 1e-10, scheme
        assert m["alpha_min"] >= -1e-12 and m["alpha_max"] <= 1.0 + 1e-12, scheme


def test_plic_beats_donor_on_rotation_shape():
    case = zalesak_case()
    plic = advance(case, 48, 48, Transport(scheme="plic", cfl=0.75))["metrics"]
    donor = advance(case, 48, 48, Transport(scheme="donor", cfl=0.75))["metrics"]
    assert plic["symmetric_difference"] < donor["symmetric_difference"]
    assert plic["relative_l1"] < donor["relative_l1"]


def test_fct_transport_is_bounded_by_construction():
    """The algebraic FCT sweep keeps [0, 1] without clipping (Phase-0 claim)."""
    vof = AlgebraicVOF(24, 24, 1 / 24, 1 / 24, boundary="wall", bounds="global")
    alpha = zalesak_alpha(24, 24)
    case = zalesak_case()
    dt = 0.75 / (24 * 2.0 * math.pi * 0.5)
    for k in range(20):
        u, v = sample_faces(case, 24, 24, k * dt)
        alpha = vof.advect(alpha, u, v, dt)
        assert float(alpha.min()) >= -1e-12 and float(alpha.max()) <= 1 + 1e-12


def test_field_metrics_normalisation():
    exact = torch.zeros((4, 4), dtype=torch.float64)
    exact[1:3, 1:3] = 1.0
    same = field_metrics(exact.clone(), exact, 1.0)
    assert same["relative_l1"] == 0.0 and same["symmetric_difference"] == 0.0
    shifted = torch.zeros_like(exact)
    shifted[0:2, 0:2] = 1.0
    m = field_metrics(shifted, exact, 1.0)
    # 1 of the 4 liquid cells overlaps: 3 + 3 misplaced cells / 4 liquid cells
    assert abs(m["relative_l1"] - 1.5) < 1e-12
    assert abs(m["symmetric_difference"] - 1.5) < 1e-12


# --------------------------------------------------------------------------- #
# GCI
# --------------------------------------------------------------------------- #
def test_gci_recovers_first_order():
    out = gci([0.04, 0.02, 0.01])
    assert abs(out["observed_order"] - 1.0) < 1e-12
    assert abs(out["p_successive"] - 1.0) < 1e-12
    assert out["status"] == "asymptotic"
    assert out["gci_fine"] > 0.0


def test_gci_reports_non_monotone_honestly():
    out = gci([0.01, 0.02, 0.005])
    # the fine pair is monotone, so an order exists, but the coarse pair is not
    assert out["p_coarse"] is not None and out["p_successive"] is None
    assert "non-monotone" in out["status"]
    assert out["status"] != "asymptotic"
    # an entirely non-monotone triple has no order at all
    flat = gci([0.01, 0.01, 0.01])
    assert flat["observed_order"] is None and flat["gci_fine"] is None
    assert "honest" in flat["status"]


# --------------------------------------------------------------------------- #
# hydrostatic rest state
# --------------------------------------------------------------------------- #
def test_hydrostatic_control_shows_half_cell_offset():
    """Shipped void model: the rest state is off by the half-cell O(h) amount."""
    coarse = run_hydrostatic(20, 20, steps=6, surface_model="void")
    fine = run_hydrostatic(40, 40, steps=6, surface_model="void")
    # dev ~ 1/(2n) of rho g H_liquid -> halving exactly
    assert abs(coarse["pressure_oscillation"] - 0.05) < 5e-3
    assert abs(fine["pressure_oscillation"] - 0.025) < 3e-3
    assert coarse["spurious_velocity"] < 1e-5


def test_well_balanced_surface_holds_the_rest_state():
    for n in (20, 40):
        r = run_hydrostatic(n, n, steps=8, surface_model="void",
                            well_balanced=True)
        assert r["pressure_oscillation"] < 1e-10, (n, r["pressure_oscillation"])
        assert r["spurious_velocity"] < 1e-10, (n, r["spurious_velocity"])
        assert r["mass_drift_relative"] < 1e-12


def test_well_balanced_surface_is_the_default():
    """The consistent free-surface datum is shipped; the old treatment is a control.

    Round-1 fix of ``GAP-FVM-TWOPHASE`` (docs/GAP-FVM-TWOPHASE/
    ROUND1-FREE-SURFACE-DATUM.md): dropping the free-surface face from the
    pressure operator (homogeneous Neumann = rigid lid) leaves the dam break
    releasing late and the rest-state pressure O(h) out of balance, so the
    ghost-fluid datum (``p = 0`` half a cell away from the cell centre) is the
    default.  The historical treatment stays reachable, explicitly, as the
    negative control.
    """
    from tensorfvm.vof_phase0.solver import Phase0TwoPhaseSolver
    solver = Phase0TwoPhaseSolver(8, 8, 1.0, 1.0)
    assert solver.well_balanced_surface is True
    control = Phase0TwoPhaseSolver(8, 8, 1.0, 1.0, well_balanced_surface=False)
    assert control.well_balanced_surface is False


# --------------------------------------------------------------------------- #
# campaign plumbing
# --------------------------------------------------------------------------- #
def test_run_spec_and_merge_are_machine_checkable():
    spec = {"case": "double_vortex", "scheme": "donor", "cells": [16, 8],
            "cfl": 0.75}
    row = run_spec(spec)
    assert row["cells"] == [16, 8]
    assert row["scheme"] == "donor"
    assert row["mass_drift_relative"] < 1e-10
    report = merge_runs([row])
    assert "C5_mass_conservation" in report["criteria"]
    assert report["criteria"]["C4_hydrostatic"]["passed"] is False


def test_criteria_evaluator_flags_failure_and_pass():
    good = {"cells": [32, 32], "scheme": "plic", "case": "zalesak",
            "mass_drift_relative": 1e-15, "symmetric_difference": 0.01,
            "relative_l1": 0.01, "steps": 10, "dt": 1e-3}
    report = merge_runs([
        dict(good, case="zalesak"),
        dict(good, case="single_vortex"),
        dict(good, case="double_vortex"),
    ], hydrostatic={"cases": [{"cells": [16, 16], "depth": 0.5,
                               "density_ratio": 1000.0,
                               "shipped": {"pressure_oscillation": 0.05},
                               "well_balanced": {"pressure_oscillation": 1e-16}}]})
    crit = report["criteria"]
    assert crit["C1_zalesak_area_conservation"]["passed"] is True
    assert crit["C2_single_vortex"]["passed"] is True
    assert crit["C3_double_vortex"]["passed"] is True
    assert crit["C4_hydrostatic"]["passed"] is True
    assert crit["C5_mass_conservation"]["passed"] is True

    bad = dict(good, symmetric_difference=0.5)
    report = merge_runs([dict(bad, case="single_vortex")])
    assert report["criteria"]["C2_single_vortex"]["passed"] is False