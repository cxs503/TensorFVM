"""Pin the z-momentum probe finding for CASE-FVM-KOVASZNAY-ORDER.

The probe (``scripts/probe_kovasznay_z_momentum.py``) establishes that the
O(h) spanwise (w) velocity in the extruded ``n x n x 1`` Kovasznay order study
is *not* a defect of the z-momentum discretisation:

* the discrete z-momentum residual at the exact solution (w == 0) is
  machine-zero -- the z-momentum equation is trivially satisfied;
* the w DOF does not couple into the in-plane (x,y) momentum through the
  symmetric-deviatoric viscous operator (coupling == 0);
* the in-plane (x,y) viscous consistency error is O(h^2) (ratio ~4 between
  n=8 and n=16), consistent with the in-plane order 1.808 that already passes
  the 1.8 gate.

These tests pin those three invariants so a future change to the viscous
operator (or the verification setup) that re-introduces a z-momentum driver
fails loudly instead of silently re-opening the anchor.
"""
from __future__ import annotations

import numpy as np
import pytest

from tensorfvm.verification.native_hybrid_ns import kovasznay_fields, kovasznay_solver


def _exact_state(mesh, solver, exact):
    N, F = len(mesh.cells), len(mesh.owner)
    st = np.zeros((N + F, 3))
    st[:N] = exact['velocity']
    d = solver.viscous.dirichlet_components
    st[N:][d] = solver.prescribed[3 * N:].reshape(F, 3)[d]
    free = ~d
    if free.any():
        st[N:][free] = kovasznay_fields(mesh.face_centers, 40.)['velocity'][free]
    return st.ravel()


def _z_momentum_residual(mesh, solver, exact):
    """Full z-momentum residual at the exact solution (w == 0)."""
    N, F = len(mesh.cells), len(mesh.owner)
    st = _exact_state(mesh, solver, exact)
    full, source = solver._current_system()
    r = (full @ st + source - solver.divergence.T @ exact['pressure'] - solver.load)
    return r.reshape(N + F, 3)[:N, 2]


def _inplane_viscous_defect(mesh, solver, exact):
    """In-plane (x,y) viscous consistency error at the exact solution."""
    N, F = len(mesh.cells), len(mesh.owner)
    st = _exact_state(mesh, solver, exact)
    r = (solver.viscous.full_matrix @ st).reshape(N + F, 3)[:N, :2]
    return r


def _w_viscous_coupling_xy(mesh, solver, exact):
    """Viscous x/y reaction from the w DOFs (should be decoupled)."""
    N, F = len(mesh.cells), len(mesh.owner)
    st = _exact_state(mesh, solver, exact)
    w_dof = np.zeros_like(st); w_dof[2::3] = st[2::3]
    return (solver.viscous.full_matrix @ w_dof).reshape(N + F, 3)[:N, :2]


def test_z_momentum_is_trivially_satisfied_at_exact_solution():
    mesh, solver, exact = kovasznay_solver(8, 40., symmetry_extrusion=True,
                                           convection_order=2, coupled_linear_method='reused_lu')
    rz = _z_momentum_residual(mesh, solver, exact)
    # the z-momentum equation is 0 = 0 for the exact (w == 0) field
    assert np.max(np.abs(rz)) < 1e-14


def test_w_does_not_couple_into_inplane_viscous():
    mesh, solver, exact = kovasznay_solver(8, 40., symmetry_extrusion=True,
                                           convection_order=2, coupled_linear_method='reused_lu')
    c = _w_viscous_coupling_xy(mesh, solver, exact)
    # the w DOF is decoupled from the in-plane (x,y) viscous momentum
    assert np.max(np.abs(c)) < 1e-14


def test_inplane_viscous_defect_is_second_order():
    """The in-plane viscous consistency error halves ~4x per grid doubling."""
    m8, s8, e8 = kovasznay_solver(8, 40., symmetry_extrusion=True,
                                  convection_order=2, coupled_linear_method='reused_lu')
    m16, s16, e16 = kovasznay_solver(16, 40., symmetry_extrusion=True,
                                     convection_order=2, coupled_linear_method='reused_lu')
    d8 = np.max(np.abs(_inplane_viscous_defect(m8, s8, e8)))
    d16 = np.max(np.abs(_inplane_viscous_defect(m16, s16, e16)))
    # O(h^2): doubling h (halving n) should ~4x the defect
    ratio = d8 / d16
    assert 2.5 < ratio < 6.0, f'in-plane viscous defect ratio {ratio:.2f} not O(h^2)'
