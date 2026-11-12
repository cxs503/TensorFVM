"""Verification tests for the inviscid lifting panel method (NACA0012-INVISCID)."""
import math

import numpy as np

from tensorfvm.panel_airfoil import (
    _influence,
    cl_thin_aerofoil,
    naca4_panels,
    solve_lifting_panels,
)


def test_geometry_closes_and_normals_point_out():
    geo = naca4_panels(0.12, 50)
    loop = geo["a"]
    assert np.allclose(loop[0], loop[-1] + (geo["b"][-1] - geo["a"][-1]))  # closed loop
    # upper surface panels (first half) must have outward normal pointing up
    assert np.all(geo["normal"][:50, 1] > 0.0)
    assert np.all(geo["normal"][50:, 1] < 0.0)
    # chord of the panelised section is unity and the section is symmetric
    assert abs((geo["mid"][:, 0].max() - geo["mid"][:, 0].min()) - 1.0) < 1e-3
    assert abs(geo["mid"][0, 1] + geo["mid"][-1, 1]) < 1e-12


def test_zero_lift_at_zero_incidence_and_antisymmetry():
    """Symmetry is exact, so Cl(0) and Cl(a)+Cl(-a) must vanish to round-off."""
    zero = solve_lifting_panels(0.0, 100)
    plus = solve_lifting_panels(2.0, 100)
    minus = solve_lifting_panels(-2.0, 100)
    assert abs(zero.cl_pressure) < 1e-10
    assert abs(plus.cl_pressure + minus.cl_pressure) < 1e-10
    assert plus.normal_residual < 1e-9


def test_kutta_condition_is_satisfied_on_the_two_trailing_edge_panels():
    r = solve_lifting_panels(2.0, 200)
    assert r.kutta_residual < 1e-9
    # the two trailing-edge panels carry the same pressure (equal-speed Kutta)
    assert abs(r.cp[0] - r.cp[-1]) < 1e-8


def test_cylinder_cp_converges_to_the_analytic_solution():
    """Source-sheet kernel check: first-order convergence of Cp on a cylinder."""
    errors = []
    for n in (32, 64, 128, 256):
        th = np.linspace(0.0, 2.0 * math.pi, n, endpoint=False)
        pts = np.column_stack([np.cos(th), np.sin(th)])
        a = pts
        b = np.roll(pts, -1, axis=0)
        t = b - a
        arc = np.linalg.norm(t, axis=1)
        t = t / arc[:, None]
        nrm = np.column_stack([t[:, 1], -t[:, 0]])
        geo = dict(a=a, b=b, t=t, normal=nrm, arc=arc, mid=0.5 * (a + b), length=float(arc.sum()))
        An, At = _influence(geo["mid"], nrm, t, geo)
        U = np.array([1.0, 0.0])
        sigma = np.linalg.solve(An[:, :n], -(nrm @ U))
        vt = t @ U + At[:, :n] @ sigma
        errors.append(float(np.abs(1.0 - vt ** 2 - (1.0 - 4.0 * np.sin(th) ** 2)).max()))
    # monotone decrease and halving per refinement (first order in panel size)
    assert all(errors[i + 1] < errors[i] for i in range(len(errors) - 1))
    assert errors[-1] < 0.1
    for i in range(len(errors) - 1):
        assert 1.6 < errors[i] / errors[i + 1] < 2.4


def test_lift_is_linear_in_incidence_and_above_thin_aerofoil_theory():
    """The NACA 0012 has 12 % thickness, so the inviscid slope exceeds 2*pi.

    This is the *finding* of the case, not a tolerance: the panel method is
    checked here only for internal consistency (linearity) and for the sign of
    the thickness effect, so the test states the measured ratio band explicitly.
    """
    cl1 = solve_lifting_panels(1.0, 200).cl_pressure
    cl2 = solve_lifting_panels(2.0, 200).cl_pressure
    assert abs(cl2 - 2.0 * cl1) < 1e-3          # linear in alpha
    ratio = cl2 / cl_thin_aerofoil(2.0)
    assert 1.05 < ratio < 1.15                  # thickness raises the inviscid slope


def test_panel_convergence_monotone():
    values = [solve_lifting_panels(2.0, n).cl_pressure for n in (50, 100, 200, 400)]
    assert all(values[i + 1] > values[i] for i in range(len(values) - 1))
    assert abs(values[-1] - values[-2]) < 2e-3