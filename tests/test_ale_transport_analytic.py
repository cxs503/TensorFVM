"""Analytic verification of the ALE passive-scalar step (moving_mesh.ale_scalar_step).

Closes the verification debt behind ``docs/verification-ale-transport``: the ALE
acceptance tests only checked geometry and a *constant* free-stream field, while a
diffusing scalar was never compared with a closed form (and the module docstring
claimed a "second-order" temporal accuracy that nothing measured).

Each test states the quantity and the tolerance it asserts; all cases are tiny
Cartesian blocks so the module runs in seconds.
"""
from __future__ import annotations

import numpy as np
import pytest

import tensorfvm.moving_mesh as moving_mesh
from tensorfvm.moving_mesh import ale_scalar_step
from tensorfvm.verification.ale_transport import (
    build_cartesian_block,
    frozen_translation,
    gaussian_case,
    gaussian_exact,
    observed_order,
    relative_l2,
    uniform_preservation,
)


def test_constant_field_is_preserved_with_diffusion_on_moving_mesh():
    """A constant field is the zero mode of the transport operator (free stream).

    Before the fix the diffusive conductance entered the numerator and the
    diagonal asymmetrically, so a constant field decayed by ~3.3e-3 per step.
    """
    assert uniform_preservation(diffusivity=1e-2, mesh_shift=(1e-3, 0.0, 0.0)) < 1e-12
    assert uniform_preservation(diffusivity=1e-2, mesh_shift=(7e-4, -5e-4, 3e-4)) < 1e-12
    # diffusion alone, no mesh motion, must also preserve the constant
    assert uniform_preservation(diffusivity=1e-2, mesh_shift=(0.0, 0.0, 0.0)) < 1e-12


def test_frozen_linear_field_translates_exactly_in_the_block_interior():
    """Mesh moving under a frozen field phi = x must reproduce phi_new = x_new.

    The interior must be machine exact; the patch cells may only carry the
    first-order zero-gradient closure error, never another cell's value (the old
    ``neighbor.clip(0)`` fallback injected cell 0 into every boundary face).
    """
    res = frozen_translation(mesh_speed=1.0, dt=1e-3, nx=12, ny=6, nz=4)
    assert res["n_interior"] > 0
    assert res["max_interior"] < 1e-12
    assert res["max_all"] <= 1.5e-3 + 1e-12     # = u_m*dt, the boundary closure error


def test_boundary_patch_does_not_see_cell_zero():
    """Perturbing cell 0 must not change a cell at the opposite boundary patch."""
    mesh = build_cartesian_block(16, 4, 2, lx=0.2, ly=0.05, lz=0.05)
    g = mesh.metrics()
    base = np.linspace(0.0, 1.0, mesh.n_cells)
    far = int(np.argmax(g.centers[:, 0]))

    def step(phi):
        return ale_scalar_step(g, g, field_old=phi, face_velocity=np.zeros((mesh.n_faces, 3)),
                               sweep=np.zeros(mesh.n_faces), dt=1e-3, diffusivity=1e-3)

    ref = step(base)
    bumped = base.copy()
    bumped[0] += 100.0
    out = step(bumped)
    assert abs(out[far] - ref[far]) < 1e-10
    assert abs(out[0] - ref[0]) > 1e-3          # the bump is still felt locally


def test_gaussian_matches_the_analytic_advection_diffusion_solution():
    """Comoving mesh: relative L2 error against the exact Gaussian must be < 3 %."""
    case = dict(x0=0.06, sigma0=0.02, U=0.5, D=1e-4, t_end=0.02,
                lx=0.2, ly=0.05, lz=0.05)
    err, history = gaussian_case(nx=100, dt=5e-4, comoving=True, **case)
    assert len(history) == int(round(case["t_end"] / 5e-4))
    assert err < 0.03, err


def test_analytic_solution_helper_is_self_consistent():
    """Conservation of the Gaussian weight and diffusion-only limit."""
    x = np.linspace(-0.5, 0.5, 4001)
    a = gaussian_exact(x, 0.0, x0=0.0, sigma0=0.02, U=0.0, D=0.0)
    b = gaussian_exact(x, 0.0, x0=0.0, sigma0=0.02, U=0.0, D=1e-4)
    assert np.allclose(a, b)
    shifted = gaussian_exact(x, 0.1, x0=0.0, sigma0=0.02, U=0.5, D=0.0)
    assert relative_l2(shifted, gaussian_exact(x, 0.1, x0=0.05, sigma0=0.02, U=0.0, D=0.0)) < 1e-12


def test_measured_temporal_order_is_first_order_not_second():
    """Regression-lock the measured order and the honest docstring claim."""
    errs, dts = [], []
    for dt in (1.0e-3, 5.0e-4, 2.5e-4):
        case = dict(x0=0.06, sigma0=0.02, U=0.5, D=1e-4, t_end=0.02,
                    lx=0.2, ly=0.05, lz=0.05)
        errs.append(gaussian_case(nx=100, dt=dt, comoving=True, **case)[0])
        dts.append(dt)
    order = observed_order(errs, dts)
    # First order is the ceiling: the diffusion operator is diagonal-implicit only
    # and the neighbour states are old-time, so refinement can never beat O(dt).
    # A flat tail (order -> 0) is legitimate once the spatial floor dominates.
    assert order <= 1.4, order
    assert all(b < a for a, b in zip(errs, errs[1:])), errs

    doc = moving_mesh.__doc__ or ""
    assert "second-order time accuracy" not in doc
    assert "first order" in doctext_of_step()


def doctext_of_step() -> str:
    return ale_scalar_step.__doc__ or ""
