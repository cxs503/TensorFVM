"""Tests for the KVLCC2 double-body case: geometry discipline, mesh contract and
force bookkeeping.

The solver-level physics is not asserted here (see
``docs/resistance-chain-form-factor/`` for why a resolved double-body RANS is
out of reach in this workspace); what is tested is everything the form-factor
claim depends on: the discretised body must be the body the correlations are
evaluated for, the hull patch must be a no-slip boundary, and the force sign
convention must be fixed rather than guessed.
"""
import math

import numpy as np
import pytest

from tensorfvm.kvlcc2_double_body import (
    FITTED_KVLCC2_MODEL_PARAMS, KVLCC2_FULL_SCALE, KVLCC2_MODEL, HullShape,
    build_double_body_mesh, default_hull_shape, drag_components, fit_hull_shape,
    wall_cell_height_for_y_plus,
)
from tensorfvm.resistance_chain import (
    holtrop_form_factor, holtrop_wetted_surface, ittc57_cf,
)


@pytest.fixture(scope="module")
def shape():
    return default_hull_shape()


def test_fitted_hull_reproduces_the_principal_particulars(shape):
    coefficients = shape.coefficients()
    assert coefficients["Cb"] == pytest.approx(KVLCC2_MODEL.Cb, rel=1e-3)
    assert coefficients["Cm"] == pytest.approx(KVLCC2_MODEL.Cm, rel=1e-3)
    assert coefficients["Cwp"] == pytest.approx(KVLCC2_MODEL.Cwp, rel=1e-3)
    assert coefficients["LCB_pct"] == pytest.approx(KVLCC2_MODEL.lcb_pct, abs=2e-2)
    # length, breadth and draft are exact by construction: they are inputs
    assert coefficients["nabla"] == pytest.approx(KVLCC2_MODEL.nabla, rel=1e-3)


def test_analytic_wetted_area_agrees_with_the_holtrop_regression(shape):
    # two independent constructions of the same hull surface: the fitted section
    # family (integrated arc length) and the Holtrop (1984) regression
    assert shape.wetted_area() == pytest.approx(holtrop_wetted_surface(KVLCC2_MODEL), rel=0.01)


def test_fit_hull_shape_is_deterministic_and_starts_from_its_own_solution():
    # re-running the optimiser from the published parameters must stay put
    refit = fit_hull_shape(KVLCC2_MODEL, fixed=dict(FITTED_KVLCC2_MODEL_PARAMS),
                           max_nfev=12)
    errors = refit.coefficient_errors()
    assert max(abs(v) for v in errors.values()) < 1e-3


def test_inside_test_is_symmetric_and_respects_length_and_draft(shape):
    L, B, T = KVLCC2_MODEL.L, KVLCC2_MODEL.B, KVLCC2_MODEL.T
    mid = 0.5 * L
    assert bool(shape.inside(mid, 0.1, -0.1)) is True
    assert bool(shape.inside(mid, -0.1, -0.1)) is True      # port/starboard
    assert bool(shape.inside(mid, 0.1, 0.1)) is True        # mirror half
    assert bool(shape.inside(mid, 0.1, -0.1)) == bool(shape.inside(mid, 0.1, 0.1))
    assert bool(shape.inside(-0.5, 0.0, -0.1)) is False     # ahead of the AP
    assert bool(shape.inside(L + 0.5, 0.0, -0.1)) is False  # abaft the FP
    assert bool(shape.inside(mid, 0.0, -(T + 0.05))) is False   # below the keel
    assert bool(shape.inside(mid, 0.6 * B, -0.1)) is False      # outboard


def test_mesh_patch_labels_and_connectivity_contract():
    shape = default_hull_shape()
    mesh = build_double_body_mesh(shape, spacing=0.25, far_spacing=0.75, growth=1.3,
                                 y_half=1.5 * KVLCC2_MODEL.B,
                                 z_depth=1.5 * KVLCC2_MODEL.T)
    assert set(mesh.patches.values()) == {"hull", "inlet", "outlet", "far"}
    assert mesh.n_cells == int(mesh.meta["n_cells"]) > 0
    # the discretised body must not collapse: volume within 8 % of 2 x displaced
    assert mesh.meta["body_volume_ratio"] == pytest.approx(1.0, abs=0.08)
    from tensorfvm.native_mesh3d import NativeMesh3D
    native = NativeMesh3D.from_cells(mesh.points, mesh.cells,
                                     ["hexahedron"] * mesh.n_cells,
                                     boundary_patches=mesh.patches)
    labels = native.patch_labels
    boundary = native.neighbor < 0
    hull = labels == "hull"
    assert hull.sum() > 0
    # every hull face is a boundary face (no interior patch tags) ...
    assert np.all(boundary[hull])
    # ... and the hull patch is not an empty or a topology artefact
    assert 0.5 * hull.sum() < mesh.meta["n_body_cells"]
    # the fluid cells are all positively oriented hexahedra
    assert np.all(native.cell_volumes > 0.0)


def _hull_patch_area(mesh) -> float:
    area = 0.0
    for face in mesh.patch_faces("hull"):
        pts = mesh.points[list(face)]
        area += 0.5 * np.linalg.norm(np.cross(pts[1] - pts[0], pts[2] - pts[0]))
        area += 0.5 * np.linalg.norm(np.cross(pts[2] - pts[0], pts[3] - pts[0]))
    return area


def test_stair_step_hull_area_converges_upwards_towards_the_analytic_area():
    shape = default_hull_shape()
    analytic = 2.0 * shape.wetted_area()
    coarse = build_double_body_mesh(shape, spacing=0.12, far_spacing=0.36, growth=1.2,
                                    y_half=1.2 * KVLCC2_MODEL.B,
                                    z_depth=1.2 * KVLCC2_MODEL.T)
    fine = build_double_body_mesh(shape, spacing=0.06, far_spacing=0.30, growth=1.2,
                                  y_half=1.2 * KVLCC2_MODEL.B,
                                  z_depth=1.2 * KVLCC2_MODEL.T)
    coarse_area, fine_area = _hull_patch_area(coarse), _hull_patch_area(fine)
    # a stair-stepped surface measures |n_x|+|n_y|+|n_z| per facet instead of the
    # Euclidean area, so the patch area is *larger* than the analytic value (the
    # inflation does not vanish at these resolutions -- this is the documented
    # limitation that makes a resolved form factor unaffordable here)
    assert coarse_area > analytic and fine_area > analytic
    assert coarse_area / analytic < 1.4
    assert fine_area / analytic < 1.35


def test_wall_cell_height_follows_the_y_plus_definition():
    speed, nu = 1.1766, 1.19e-5
    re = speed * KVLCC2_MODEL.L / nu
    cf = ittc57_cf(re)
    height = wall_cell_height_for_y_plus(100.0, speed, nu, cf)
    centre = 0.5 * height
    assert centre * speed * math.sqrt(cf / 2.0) / nu == pytest.approx(100.0, rel=1e-12)
    assert wall_cell_height_for_y_plus(200.0, speed, nu, cf) == pytest.approx(2 * height, rel=1e-12)
    with pytest.raises(ValueError):
        wall_cell_height_for_y_plus(0.0, speed, nu, cf)


def test_drag_split_pins_the_sign_convention():
    # the native solver returns sum p n_fluid dA and -sum tau dA, both upstream
    forces = {"pressure_force": np.array([-12.0, 0.5, 0.2]),
              "viscous_force": np.array([-3.0, -0.1, 0.0])}
    drag = drag_components(forces)
    assert drag["pressure_drag_n"] == pytest.approx(12.0)
    assert drag["friction_drag_n"] == pytest.approx(3.0)
    assert drag["total_drag_n"] == pytest.approx(15.0)
    with pytest.raises(ValueError):
        drag_components({"pressure_force": np.array([-1.0, 0, 0]),
                         "viscous_force": np.array([1.0, 0, 0])})
    with pytest.raises(KeyError):
        drag_components({"pressure_force": np.zeros(3)})


def test_full_scale_particulars_give_a_much_larger_form_factor_than_the_model_case():
    # the geometry audit: Cb = 0.708 at L/B = 7.48 is not geometrically similar
    # to KVLCC2 (Cb = 0.8098, L/B = 5.52), and the Holtrop form factor sees it
    model = holtrop_form_factor(KVLCC2_MODEL)
    full = holtrop_form_factor(KVLCC2_FULL_SCALE)
    assert model == pytest.approx(1.1867, abs=5e-4)
    assert full > 1.30
    assert KVLCC2_MODEL.L / KVLCC2_MODEL.B > 7.0
    assert KVLCC2_FULL_SCALE.L / KVLCC2_FULL_SCALE.B == pytest.approx(5.52, abs=0.01)


def test_shape_parameters_are_dimensionless_so_the_family_rescales():
    small = HullShape(principal=KVLCC2_MODEL.__class__(
        name="KVLCC2_model_L7_scaled", L=3.5, B=0.468, T=0.171, Cb=0.708,
        Cm=0.985, Cwp=0.808, lcb_pct=-1.688), **FITTED_KVLCC2_MODEL_PARAMS)
    errors = small.coefficient_errors()
    assert max(abs(v) for v in errors.values()) < 1e-3
    assert small.wetted_area() == pytest.approx(0.25 * default_hull_shape().wetted_area(), rel=0.01)