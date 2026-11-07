"""Acceptance tests for the ALE moving / morphing mesh capability.

Machine-checkable criteria (each test states the tolerance it asserts):

* **G1 space-conservation law** - ``sum_faces dV_face == V_new - V_old`` to
  roundoff for rigid translation, rigid rotation and morphing.
* **G2 free-stream preservation** - the ALE convective operator has *exactly*
  zero residual for a uniform field on a moving mesh.
* **G3 Galilean invariance** - a Gaussian advected in a uniformly moving frame
  matches the fixed-frame solution to roundoff.
* **G4 impermeable moving wall** - the ALE mass flux through the body patch of
  a morphing mesh vanishes to roundoff (no spurious mass source).
* **G5 hull geometry gate** - the KVLCC2 double-body block reproduces the
  calibrated hull displacement (Cb) within the discretisation error.
* **G6 PMM kinematics** - pure sway / pure yaw poses obey the prescribed
  amplitude, period and phase relations, and a full KVLCC2 oscillation keeps
  the mesh valid (positive volumes) with bounded GCL error.
"""
from __future__ import annotations

import numpy as np
import pytest

from tensorfvm.moving_mesh import (
    PMMKinematics,
    ale_scalar_step,
    build_kvlcc2_double_body_ogrid,
    morph_nodes,
    wall_ale_flux,
)
from tensorfvm.pmm_cfd import KVLCC2Hull


@pytest.fixture(scope="module")
def cylinder():
    from tensorfvm.moving_mesh import build_cylinder_ogrid
    return build_cylinder_ogrid(radius=0.05, n_theta=32, n_radial=10, span=0.2,
                                n_span=3, far_radius=0.5)


def _weights(mesh):
    nj = mesh.shape[1]
    w = np.linspace(1.0, 0.0, nj + 1) ** 2
    return np.broadcast_to(w[None, :, None], tuple(n + 1 for n in mesh.shape)).copy()


POSE = dict(sway=0.01, sway_rate=0.0, yaw=0.05, yaw_rate=0.0)


def test_g1_space_conservation_translation(cylinder):
    new = cylinder.nodes + np.array([1.0e-3, -2.0e-3, 5.0e-4])
    dv = cylinder.cell_volume_change(cylinder.nodes, new)
    v0 = cylinder.metrics().volume
    v1 = cylinder.metrics(new).volume
    assert np.abs(dv - (v1 - v0)).max() < 1e-15
    assert np.abs(dv).max() < 1e-15


def test_g1_space_conservation_rotation(cylinder):
    th = 0.07
    rot = np.array([[np.cos(th), -np.sin(th), 0.0],
                    [np.sin(th), np.cos(th), 0.0], [0.0, 0.0, 1.0]])
    new = cylinder.nodes @ rot.T
    dv = cylinder.cell_volume_change(cylinder.nodes, new)
    v0 = cylinder.metrics().volume
    v1 = cylinder.metrics(new).volume
    assert np.abs(dv - (v1 - v0)).max() < 1e-15


def test_g1_space_conservation_morphing(cylinder):
    new = morph_nodes(cylinder.nodes, _weights(cylinder), POSE, midpoint_x=0.0)
    dv = cylinder.cell_volume_change(cylinder.nodes, new)
    v0 = cylinder.metrics().volume
    v1 = cylinder.metrics(new).volume
    assert np.abs(v1 - v0).max() > 1e-9          # the mesh really deformed
    assert np.abs(dv - (v1 - v0)).max() < 1e-14  # ... and GCL still holds


def test_g2_freestream_preservation(cylinder):
    """Uniform flow, moving mesh, zero residual for the divergence-form flux."""
    new = morph_nodes(cylinder.nodes, _weights(cylinder), POSE, midpoint_x=0.0)
    dt = 1e-3
    sweep = cylinder.swept_volumes(cylinder.nodes, new)
    g_old, g_new = cylinder.metrics(), cylinder.metrics(new)
    m = cylinder
    n = m.n_cells
    uniform = np.tile(np.array([0.3, -0.2, 0.1]), (n, 1))
    face_velocity = np.zeros((m.n_faces, 3))
    face_velocity[m.interior] = uniform[0]
    face_velocity[~m.interior] = uniform[0]
    up_flux = np.einsum("fi,fi->f", face_velocity, g_new.face_area) - sweep / dt
    residual = ((g_new.volume * 1.0 - g_old.volume * 1.0) / dt
                + g_new.divergence(up_flux * 1.0))
    # GCL: sum_f sweep = dV so the transient term cancels the flux exactly
    assert np.abs(residual).max() < 1e-12 * np.abs(up_flux).max()


def test_g3_galilean_invariance(cylinder):
    """Advect a Gaussian in the body frame; the field must translate rigidly."""
    m = cylinder
    dt = 2e-4
    speed = np.array([0.0, 0.25, 0.0])
    phi = np.exp(-((m.metrics().centers - m.metrics().centers.mean(axis=0)) ** 2).sum(axis=1)
                 / 2e-3)
    fixed = phi.copy()
    moving = phi.copy()
    nodes0 = m.nodes.copy()
    for k in range(25):
        # (a) fixed frame: fluid moves at +speed, mesh static
        g_old = m.metrics(nodes0)
        fv = np.tile(speed, (m.n_faces, 1))
        fixed = ale_scalar_step(g_old, g_old, field_old=fixed, face_velocity=fv,
                                sweep=np.zeros(m.n_faces), dt=dt, diffusivity=0.0)
        # (b) body frame: fluid at rest, mesh translates at -speed
        nodes1 = nodes0 - speed * dt
        g_new = m.metrics(nodes1)
        sweep = m.swept_volumes(nodes0, nodes1)
        moving = ale_scalar_step(g_old, g_new, field_old=moving,
                                 face_velocity=np.zeros((m.n_faces, 3)), sweep=sweep,
                                 dt=dt, diffusivity=0.0)
        nodes0 = nodes1
    assert np.abs(fixed - moving).max() < 1e-10 * np.abs(fixed).max()
    assert np.abs(fixed - phi).max() > 1e-3     # the field really advected


def test_g4_impermeable_moving_wall(cylinder):
    new = morph_nodes(cylinder.nodes, _weights(cylinder), POSE, midpoint_x=0.0)
    sweep = cylinder.swept_volumes(cylinder.nodes, new)
    um = sweep[:, None] * cylinder.metrics(new).face_area / np.maximum(
        cylinder.metrics(new).area_mag ** 2, 1e-300)[:, None]
    flux = wall_ale_flux(cylinder.metrics(), cylinder.metrics(new),
                         face_velocity=um, sweep=sweep)
    assert np.abs(flux).max() < 1e-15


def test_g6_pmm_kinematics_relations():
    sway = PMMKinematics(mode="sway", amplitude=0.05, period=10.0)
    w = sway.omega
    assert sway.pose(0.0)["sway"] == pytest.approx(0.0)
    assert sway.pose(0.25 * 10.0)["sway"] == pytest.approx(0.05, rel=1e-12)
    assert sway.pose(0.25 * 10.0)["sway_rate"] == pytest.approx(0.0, abs=1e-12)
    assert sway.pose(0.0)["sway_rate"] == pytest.approx(0.05 * w, rel=1e-12)
    yaw = PMMKinematics(mode="yaw", amplitude=3.0, period=10.0)
    assert np.rad2deg(yaw.pose(2.5)["yaw"]) == pytest.approx(3.0, rel=1e-12)
    drift = PMMKinematics(mode="drift", beta_deg=10.0)
    assert np.rad2deg(drift.pose(1.23)["yaw"]) == pytest.approx(10.0, rel=1e-12)
    assert drift.pose(1.23)["yaw_rate"] == 0.0


@pytest.fixture(scope="module")
def hull_mesh():
    hull = KVLCC2Hull()
    hull.calibrate()
    mesh = build_kvlcc2_double_body_ogrid(hull, n_station=24, n_around=32, n_radial=10)
    return hull, mesh, _weights(mesh)


def test_g5_hull_geometry_gate(hull_mesh):
    """Discrete body volume from the mesh must reproduce the calibrated Cb.

    ``V_outer`` is taken from the mesh surface itself (far field + caps) so the
    check is free of bookkeeping between the analytic box and its polygonal
    discretisation.  The residual ~4 % is the tip truncation at
    ``end_half_breadth_fraction = 0.03`` plus the section polygonalisation; it
    is *reported*, not tuned away.
    """
    hull, mesh, _ = hull_mesh
    g = mesh.metrics()
    lab = np.array([l[1] for l in mesh.face_label])
    outer = (lab == "far") | (lab == "cap_low") | (lab == "cap_high")
    v_outer = float(np.einsum("fi,fi->f", g.face_center[outer], g.face_area[outer]).sum() / 3.0)
    xs = np.unique(mesh.nodes[:, 0, 0, 0])
    length = float(xs[-1] - xs[0])
    v_body = (v_outer - g.volume.sum()) / 2.0           # double body -> half body
    cb_mesh = v_body / (length * hull.B * hull.T)
    cb_analytic = hull.integrals().Cb
    assert abs(cb_analytic - 0.708) < 0.005             # calibrated hull gate
    assert abs(cb_mesh - cb_analytic) < 0.06 * cb_analytic
    assert abs(length - hull.L) < 0.02 * hull.L


def test_g6_hull_pmm_cycle_keeps_mesh_valid(hull_mesh):
    hull, mesh, weight = hull_mesh
    base = mesh.nodes.copy()
    kin = PMMKinematics(mode="yaw", amplitude=3.0, period=10.0, midpoint_x=0.5 * hull.L)
    v0 = mesh.metrics().volume
    worst_gcl, min_ratio = 0.0, np.inf
    total_dv = total_dv_exact = 0.0
    dt = 10.0 / 20.0
    nodes_old = base
    for k in range(21):
        pose = kin.pose(-10.0 + k * dt)
        nodes_new = morph_nodes(base, weight, pose, midpoint_x=0.5 * hull.L)
        g_new = mesh.metrics(nodes_new)
        dv = mesh.cell_volume_change(nodes_old, nodes_new)
        d_vol = g_new.volume - mesh.metrics(nodes_old).volume
        worst_gcl = max(worst_gcl, np.abs(dv - d_vol).max())
        total_dv += dv.sum()
        total_dv_exact += d_vol.sum()
        min_ratio = min(min_ratio, (g_new.volume / v0).min())
        nodes_old = nodes_new
    # exact to roundoff for translation/rotation/morphing on the cylinder block;
    # on the KVLCC2 block with a 3 deg yaw morph the warped-quad triangulation
    # leaves a residual that is tiny relative to the cell volume
    assert worst_gcl < 1e-3 * v0.max()
    # global conservation is exact even where individual warped cells disagree
    assert abs(total_dv - total_dv_exact) < 1e-12
    assert min_ratio > 0.2            # no layer collapsed over the cycle