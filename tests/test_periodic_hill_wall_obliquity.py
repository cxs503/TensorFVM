"""Round-25 audit: is the published ``x_r/H`` a *mesh-obliquity* artefact?

Every claim in :mod:`tensorfvm.verification.periodic_hill_wall_obliquity` is
anchored on a result that is exact, not "close enough":

* the wall frame decomposes the wall-to-cell vector to round-off,
  ``d0 = alpha n + beta t``, on both walls and through the periodic seam;
* for a velocity field that is affine in ``(x, y)`` the two-point eta-line
  derivative obeys the *closed-form* identity
  ``raw = grad(u.t).n + (beta/alpha) grad(u.t).t``, and the 2-D least-squares
  operator returns ``grad(u.t).n`` exactly -- so the obliquity bias is proved,
  not inferred, and the correction is proved to remove it;
* on a flat wall (``hill_amplitude = 0``) the obliquity is identically zero and
  all three estimators collapse onto each other, which is the control that says
  the difference on the hill is obliquity and not a coding slip;
* the estimator binding: ``raw`` must reproduce the solver's stored
  ``mean_tau_b`` of the laminar ``Re = 2800`` rung to round-off, and the
  published ``x_r/H`` must survive the switch to the obliquity-free operator.
"""

from pathlib import Path

import numpy as np
import pytest

from tensorfvm.periodic_hill import (
    PeriodicHillConfig,
    PeriodicHillMesh,
    analytic_flat_channel,
)
from tensorfvm.verification import periodic_hill_wall_obliquity as obl
from tensorfvm.verification import periodic_hill_wall_shear_order as order_mod

CASE_DIR = Path(__file__).resolve().parents[1] / "docs" / "case-periodic-hills"
ANCHOR = CASE_DIR / "round20_laminar_Re2800_96x48"


def hill_mesh(nx=96, ny=48):
    return PeriodicHillMesh(PeriodicHillConfig(nx=nx, ny=ny))


def affine_field(mesh, a=0.7, b=-1.3, c=0.4):
    """``u = a x + b y + c``, ``v = 0``: the exact-gradient calibration field."""
    u = a * mesh.xc + b * mesh.yc + c
    v = np.zeros_like(u)
    return u, v


def affine_closed_forms(mesh, wall="bottom", a=0.7, b=-1.3, c=0.4):
    """Analytic ``d(u.t)/dn``, ``d(u.t)/dt`` and wall value of ``u.t``.

    ``u . t_i = (a x + b y + c) t_i.x`` is affine in ``(x, y)``, so its gradient
    is ``(a t_i.x, b t_i.x)`` exactly and both directional derivatives follow in
    closed form -- no discrete differentiation anywhere in the reference.
    """
    geom = obl.wall_geometry(mesh, wall)
    t, n = geom["tangent"], geom["normal"]
    level = 0 if wall == "bottom" else mesh.config.ny
    xw, yw = mesh.x_n[level], mesh.y_n[level]
    exact_normal = a * t[:, 0] * n[:, 0] + b * t[:, 0] * n[:, 1]
    exact_tangent = a * t[:, 0] * t[:, 0] + b * t[:, 0] * t[:, 1]
    ut_wall = (a * xw + b * yw + c) * t[:, 0]
    return exact_normal, exact_tangent, ut_wall


# --------------------------------------------------------------------------
# geometry: the decomposition is exact, and the hill really is oblique
# --------------------------------------------------------------------------


def test_wall_frame_decomposes_d0_to_round_off_on_both_walls():
    mesh = hill_mesh(nx=48, ny=24)
    for wall in ("bottom", "top"):
        geom = obl.wall_geometry(mesh, wall)
        assert np.all(geom["alpha"] > 0.0)
        rebuilt = (geom["alpha"][:, None] * geom["normal"]
                   + geom["beta"][:, None] * geom["tangent"])
        assert np.allclose(rebuilt, geom["d0"], atol=1e-14)
        assert np.allclose(np.hypot(geom["normal"][:, 0], geom["normal"][:, 1]), 1.0,
                           atol=1e-14)
        assert np.allclose(np.hypot(geom["tangent"][:, 0], geom["tangent"][:, 1]), 1.0,
                           atol=1e-14)
        assert np.allclose((geom["normal"] * geom["tangent"]).sum(axis=1), 0.0, atol=1e-14)


def test_alpha_is_the_operator_distance_used_by_the_solver():
    """``alpha`` must be the solver's own ``dn_0``/``s_0``, not a look-alike."""
    mesh = hill_mesh(nx=48, ny=24)
    for wall in ("bottom", "top"):
        s0 = order_mod.wall_normal_distances(mesh, wall=wall, rows=1)[0]
        assert np.allclose(obl.wall_geometry(mesh, wall)["alpha"], s0, atol=0.0)


def test_hill_wall_is_strongly_oblique_and_top_deck_is_not():
    """Documents the mesh fact round 24 could only cite as a caveat."""
    mesh = hill_mesh(nx=96, ny=48)
    bottom = obl.obliquity_ratio(mesh, "bottom")
    top = obl.obliquity_ratio(mesh, "top")
    assert 0.80 < bottom["max"] < 0.90      # measured 0.852
    assert bottom["mean"] > 0.2
    assert top["max"] < 1e-12               # flat deck: no obliquity at all
    assert 0.017 < bottom["alpha_min"] < 0.020      # measured 0.01804
    assert 0.030 < bottom["alpha_max"] < 0.035      # measured 0.031625


# --------------------------------------------------------------------------
# analytic anchors: the identity holds exactly, and the correction removes it
# --------------------------------------------------------------------------


@pytest.mark.parametrize("wall", ["bottom", "top"])
def test_affine_obliquity_identity_is_exact(wall):
    mesh = hill_mesh(nx=96, ny=48)
    u, v = affine_field(mesh)
    geom = obl.wall_geometry(mesh, wall)
    exact_n, exact_t, ut_wall = affine_closed_forms(mesh, wall)
    ut = obl.wall_tangential_velocity(u, v, mesh, wall=wall, rows=1)[0]
    raw = (ut - ut_wall) / geom["alpha"]

    implied = exact_n + (geom["beta"] / geom["alpha"]) * exact_t
    # the periodic seam carries a non-periodic manufactured field, so the
    # identity is asserted on the interior columns only
    interior = slice(1, mesh.config.nx - 1)
    scale = np.abs(raw[interior]).max()
    assert np.abs(raw[interior] - implied[interior]).max() < 1e-11 * max(scale, 1.0)
    # and the bias is not cosmetic on the hill
    if wall == "bottom":
        assert np.abs(raw[interior] - exact_n[interior]).max() > 0.2 * scale


def test_lsq_operator_is_exact_for_an_affine_field():
    mesh = hill_mesh(nx=96, ny=48)
    u, v = affine_field(mesh)
    exact_n, _, ut_wall = affine_closed_forms(mesh, "bottom")
    lsq = obl.wall_gradient_lsq(u, v, mesh, wall="bottom", ut_wall=ut_wall)
    interior = slice(1, mesh.config.nx - 1)
    assert np.abs(lsq[interior] - exact_n[interior]).max() < 1e-11 * np.abs(
        exact_n[interior]).max()


def test_deferred_correction_removes_most_of_the_obliquity_bias():
    mesh = hill_mesh(nx=96, ny=48)
    u, v = affine_field(mesh)
    geom = obl.wall_geometry(mesh, "bottom")
    exact_n, _, ut_wall = affine_closed_forms(mesh, "bottom")
    ut = obl.wall_tangential_velocity(u, v, mesh, wall="bottom", rows=1)[0]
    raw = (ut - ut_wall) / geom["alpha"]
    deferred = obl.wall_gradient_deferred(u, v, mesh, wall="bottom", ut_wall=ut_wall)
    interior = slice(1, mesh.config.nx - 1)
    raw_err = np.abs(raw[interior] - exact_n[interior]).max()
    def_err = np.abs(deferred[interior] - exact_n[interior]).max()
    # measured: the chord-projected tangential derivative leaves ~1.6 % of the
    # raw bias (the chord is not exactly parallel to the wall tangent)
    assert def_err < 0.05 * raw_err


# --------------------------------------------------------------------------
# control: on a flat wall there is no obliquity, so nothing may move
# --------------------------------------------------------------------------


def test_flat_wall_has_zero_obliquity_and_identical_estimators():
    conf = PeriodicHillConfig(nx=36, ny=24, hill_amplitude=0.0)
    mesh = PeriodicHillMesh(conf)
    h, nu, body_force = conf.channel_top_m, conf.nu, 0.01
    u = np.stack([analytic_flat_channel(mesh.yc[j], h, body_force, nu)
                  for j in range(conf.ny)])
    v = np.zeros_like(u)
    exact = body_force * h / (2.0 * nu)

    geom = obl.wall_geometry(mesh, "bottom")
    assert np.abs(geom["beta"]).max() < 1e-14
    raw = obl.wall_tangential_velocity(u, v, mesh, wall="bottom", rows=1)[0] / geom["alpha"]
    deferred = obl.wall_gradient_deferred(u, v, mesh, wall="bottom")
    lsq = obl.wall_gradient_lsq(u, v, mesh, wall="bottom")
    assert np.abs(deferred - raw).max() < 1e-13 * exact
    # both operators carry the same first-order truncation of a quadratic profile
    assert np.abs(np.abs(raw - exact) - np.abs(lsq - exact)).max() < 1e-12 * exact
    assert np.abs(raw - exact).max() / exact < 0.05


# --------------------------------------------------------------------------
# binding to the published numbers
# --------------------------------------------------------------------------


def _load_anchor():
    import json

    payload = json.loads(ANCHOR.with_suffix(".json").read_text())
    cfg = payload["config"]
    conf = PeriodicHillConfig(nx=cfg["nx"], ny=cfg["ny"],
                              hill_amplitude=cfg["hill_amplitude"])
    mesh = PeriodicHillMesh(conf)
    with np.load(ANCHOR.with_suffix(".npz")) as z:
        return payload, mesh, z["mean_u"], z["mean_v"], z["x_over_h"], z["mean_tau_b"]


def test_raw_estimator_reproduces_the_stored_published_tau():
    _, mesh, u, v, _, stored = _load_anchor()
    geom = obl.wall_geometry(mesh, "bottom")
    raw = obl.wall_tangential_velocity(u, v, mesh, wall="bottom", rows=1)[0] / geom["alpha"]
    assert np.abs(raw - stored).max() < 1e-11 * np.abs(stored).max()


def test_published_x_r_survives_the_obliquity_free_operator():
    """The anchor's metric is not an operator artefact (honest FAIL stays)."""
    _, mesh, u, v, x_over_h, _ = _load_anchor()
    report = obl.obliquity_sensitivity(u, v, mesh, primary="first",
                                       x_over_h=x_over_h)
    est = report["estimators"]
    for name in ("raw", "deferred", "lsq"):
        assert est[name]["x_r"] is not None
        assert est[name]["n_bubbles"] == est["raw"]["n_bubbles"]
    assert report["x_r_spread_rel"] is not None
    assert report["x_r_spread_rel"] < 0.02          # measured 0.9 %
    # the headline honest FAIL is untouched by the operator switch
    assert est["lsq"]["x_r"] < 2.0 < 5.41


def test_obliquity_bias_in_tau_is_material_for_magnitude_claims():
    """Round 24 said "use >= 2nd order for magnitudes"; obliquity is a peer term."""
    _, mesh, u, v, _, _ = _load_anchor()
    geom = obl.wall_geometry(mesh, "bottom")
    ut = obl.wall_tangential_velocity(u, v, mesh, wall="bottom", rows=1)[0]
    tau = ut / geom["alpha"]
    bias = (geom["beta"] / geom["alpha"]) * obl.wall_tangential_derivative(
        u, v, mesh, wall="bottom")
    assert np.abs(bias).max() / np.abs(tau).max() > 0.01       # measured 3.2 %
    assert np.abs(bias).max() / np.sqrt(np.mean(tau ** 2)) > 0.05  # measured 11 %


# --------------------------------------------------------------------------
# guards
# --------------------------------------------------------------------------


def test_invalid_arguments_are_rejected():
    mesh = hill_mesh(nx=36, ny=24)
    u, v = affine_field(mesh)
    with pytest.raises(ValueError):
        obl.wall_geometry(mesh, "side")
    with pytest.raises(ValueError):
        obl.wall_tangential_velocity(u[:, :-1], v, mesh)
    with pytest.raises(ValueError):
        obl.obliquity_sensitivity(u, v, mesh, estimators=("raw", "spectral"))


def test_top_wall_deferred_matches_raw_on_the_flat_deck():
    mesh = hill_mesh(nx=48, ny=24)
    u, v = affine_field(mesh)
    raw_geom = obl.wall_geometry(mesh, "top")
    raw = obl.wall_tangential_velocity(u, v, mesh, wall="top", rows=1)[0] / raw_geom["alpha"]
    deferred = obl.wall_gradient_deferred(u, v, mesh, wall="top")
    assert np.abs(deferred - raw).max() < 1e-10 * np.abs(raw).max()