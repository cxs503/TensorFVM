"""Round-24 audit of the wall-normal derivative that feeds every published x_r/H.

The analytic anchors are exact, not "close enough":

* flat wall (``hill_amplitude = 0``) + plane Poiseuille is *quadratic* in the
  wall-normal distance, so orders 2 and 3 must reproduce the exact wall gradient
  to round-off while order 1 shows its ``O(dn)`` error and *halves* when ``ny``
  doubles (first order measured, not assumed);
* a cubic manufactured profile makes order 3 exact and order 2 (provably) not;
* multiplying ``tau_b`` by a positive constant must leave the crossing operator
  bit-identical -- this is the test that says a hidden constant factor in
  ``tau = d u_t/dn`` cannot move ``x_s``/``x_r``, so only the ``x``-dependent
  truncation error is worth auditing.

The two stored-artefact tests bind the audit to the published numbers: order 1
must reproduce the stored ``mean_tau_b`` of the laminar ``Re = 2800`` rung to
round-off (proving the rebuilt geometry and the operator replica are faithful),
and the order-to-order spread of ``x_r/H`` is locked below the case tolerance.
"""

import json
from pathlib import Path

import numpy as np
import pytest

from tensorfvm.periodic_hill import (
    PeriodicHillConfig,
    PeriodicHillMesh,
    PeriodicHillSolver,
    analytic_flat_channel,
)
from tensorfvm.verification import periodic_hill_wall_shear_order as mod
from tensorfvm.verification.periodic_hill_field_metrics import bubble_object_report

CASE_DIR = Path(__file__).resolve().parents[1] / "docs" / "case-periodic-hills"
LAMINAR_2800 = CASE_DIR / "round20_laminar_Re2800_96x48"


def flat_setup(ny=24, nx=36, body_force=0.01):
    """Flat-walled channel: hill amplitude 0, analytic Poiseuille available."""
    cfg = PeriodicHillConfig(nx=nx, ny=ny, hill_amplitude=0.0)
    mesh = PeriodicHillMesh(cfg)
    nu = cfg.nu
    h = cfg.channel_top_m
    u = np.zeros((ny, nx))
    for j in range(ny):
        u[j] = analytic_flat_channel(mesh.yc[j], h, body_force, nu)
    v = np.zeros((ny, nx))
    exact = body_force * h / (2.0 * nu)  # |du/dy| at either wall
    return cfg, mesh, u, v, exact


def test_order1_matches_solver_wall_shear_both_walls():
    """``order=1`` must be the solver's operator, not a look-alike."""
    cfg = PeriodicHillConfig(nx=36, ny=24)
    mesh = PeriodicHillMesh(cfg)
    rng = np.random.default_rng(20261115)
    u = rng.normal(size=(cfg.ny, cfg.nx))
    v = rng.normal(size=(cfg.ny, cfg.nx))
    solver = PeriodicHillSolver(cfg)
    tau_b, tau_t = solver.wall_shear(u, v)
    assert np.allclose(mod.wall_gradient(u, v, mesh, order=1, wall="bottom"),
                       tau_b, rtol=0, atol=1e-12)
    assert np.allclose(mod.wall_gradient(u, v, mesh, order=1, wall="top"),
                       tau_t, rtol=0, atol=1e-12)


def test_flat_wall_orders_reproduce_analytic_gradient():
    """Orders 2/3 are exact for the quadratic Poiseuille profile; order 1 is not."""
    cfg, mesh, u, v, exact = flat_setup()
    for wall in ("bottom", "top"):
        g1 = mod.wall_gradient(u, v, mesh, order=1, wall=wall)
        g2 = mod.wall_gradient(u, v, mesh, order=2, wall=wall)
        g3 = mod.wall_gradient(u, v, mesh, order=3, wall=wall)
        assert np.allclose(g2, exact, rtol=1e-10, atol=0)
        assert np.allclose(g3, exact, rtol=1e-10, atol=0)
        # order 1 carries the O(dn) profile-curvature error, relative size s0/h
        s0 = mod.wall_normal_distances(mesh, wall=wall, rows=1)[0]
        rel = np.abs(g1 - exact) / exact
        assert np.allclose(rel, s0 / cfg.channel_top_m, rtol=1e-8)


def test_order1_error_is_first_order_in_wall_distance():
    """Double ``ny`` -> the order-1 wall-gradient error halves."""
    errs = {}
    for ny in (12, 24):
        cfg, mesh, u, v, exact = flat_setup(ny=ny)
        g1 = mod.wall_gradient(u, v, mesh, order=1, wall="bottom")
        errs[ny] = float(np.max(np.abs(g1 - exact) / exact))
    ratio = errs[12] / errs[24]
    assert 1.7 < ratio < 2.3, (errs, ratio)


def test_cubic_profile_separates_order2_from_order3():
    """A cubic profile is outside order 2's exact class but inside order 3's."""
    cfg, mesh, u, v, _ = flat_setup()
    s = mod.wall_normal_distances(mesh, wall="bottom")
    a, b, c = 0.7, -0.3, 0.11
    u[:] = a * s + b * s ** 2 + c * s ** 3
    v[:] = 0.0
    g2 = mod.wall_gradient(u, v, mesh, order=2, wall="bottom")
    g3 = mod.wall_gradient(u, v, mesh, order=3, wall="bottom")
    assert np.allclose(g3, a, rtol=1e-10, atol=0)
    assert np.max(np.abs(g2 - a)) > 1e-8  # order 2 must NOT be exact here


def test_top_wall_row_ordering_is_mirrored():
    cfg = PeriodicHillConfig(nx=36, ny=24)
    mesh = PeriodicHillMesh(cfg)
    sb = mod.wall_normal_distances(mesh, wall="bottom")
    st = mod.wall_normal_distances(mesh, wall="top")
    assert sb.shape == st.shape
    # on a flat wall both ladders are equal; on the hill the top one starts at
    # the top face distance, so it must be strictly positive and increasing
    assert np.all(st > 0.0)
    assert np.all(np.diff(st, axis=0) > 0.0)


def test_scale_invariance_of_crossings_unit():
    """A pure rescaling of tau cannot move a crossing (unit-level statement)."""
    x = np.linspace(0.0, 9.0, 91)[:-1]
    tau = np.tanh((x - 2.5) / 0.6)  # negative in 0 < x < 2.5, positive after
    out = mod.crossing_scale_invariance(tau, x)
    assert out["identical"] is True
    assert out["unscaled"] == out["scaled"]


def test_invalid_arguments_are_rejected():
    cfg = PeriodicHillConfig(nx=20, ny=8)
    mesh = PeriodicHillMesh(cfg)
    u = np.zeros((8, 20))
    v = np.zeros((8, 20))
    with pytest.raises(ValueError):
        mod.wall_gradient(u, v, mesh, order=4)
    with pytest.raises(ValueError):
        mod.wall_gradient(u, v, mesh, order=0)
    with pytest.raises(ValueError):
        mod.wall_normal_distances(mesh, rows=0)
    with pytest.raises(ValueError):
        mod.wall_normal_distances(mesh, wall="middle")
    with pytest.raises(ValueError):
        mod.wall_gradient(np.zeros((8, 19)), v, mesh, order=1)


def test_no_crossings_reports_spread_as_none():
    """A field with no reversed flow must not invent a spread."""
    cfg, mesh, u, v, _ = flat_setup()
    out = mod.order_sensitivity(u, v, mesh)
    assert out["x_r"] == [None, None, None]
    assert out["spread"] is None and out["spread_rel"] is None


def _stored_mean_field():
    if not (LAMINAR_2800.with_suffix(".npz").exists()
            and LAMINAR_2800.with_suffix(".json").exists()):
        pytest.skip("laminar Re=2800 rung artefact not present")
    payload = json.loads(LAMINAR_2800.with_suffix(".json").read_text())
    cfg = PeriodicHillConfig(nx=payload["config"]["nx"],
                            ny=payload["config"]["ny"],
                            reynolds=payload["config"]["reynolds"])
    mesh = PeriodicHillMesh(cfg)
    with np.load(LAMINAR_2800.with_suffix(".npz")) as z:
        return payload, cfg, mesh, np.asarray(z["mean_u"]), np.asarray(z["mean_v"]), \
            np.asarray(z["mean_tau_b"])


def test_stored_artefact_order1_reproduces_published_tau():
    """The rebuilt geometry + one-sided operator == the stored published profile."""
    _, _, mesh, u, v, tau_stored = _stored_mean_field()
    tau1 = mod.wall_gradient(u, v, mesh, order=1, wall="bottom")
    assert np.abs(tau1 - tau_stored).max() < 1e-10 * np.abs(tau_stored).max()


def test_stored_artefact_order_spread_within_case_tolerance():
    """x_r/H may depend on the stencil, but not by more than the 10 % tolerance."""
    payload, _, mesh, u, v, _ = _stored_mean_field()
    out = mod.order_sensitivity(u, v, mesh)
    assert all(val is not None for val in out["x_r"]), out
    assert out["spread_rel"] is not None and out["spread_rel"] < 0.10, out
    # order 1 must reproduce the crossing the certified operator finds on the
    # *stored* profile, so the audit is bound to the same numbers rounds 17-23 used
    from tensorfvm.verification.periodic_hill_field_metrics import \
        bubble_object_report
    x_n = mesh.x_n[0] / mesh.config.height_m
    certified = bubble_object_report(
        mod.wall_gradient(u, v, mesh, order=1, wall="bottom"), x_n, primary="first")
    assert out["orders"][1]["x_r"] == pytest.approx(
        certified["crest"]["reattachment"], abs=1e-9)
    # the artefact's own published value measures a *different* bubble (round 23:
    # the legacy window rule 1.9-7.1 selects the valley bubble) -- an audit of the
    # stencil must never be confused with that 口径 change
    published = payload["time_averaged"]["x_r_over_H"]
    assert abs(published - out["orders"][1]["x_r"]) > 1.0