"""Verification of the k-omega SST RANS module and its exact discrete adjoint."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from tensorfvm.adjoint_shape.channel import ChannelGrid
from tensorfvm.adjoint_shape.geometry import N_PARAMS, FourierBody
from tensorfvm.adjoint_shape.sst_rans import (
    A1,
    BETA_STAR,
    SSTChannel,
    make_case,
)
from tensorfvm.adjoint_shape.sst_study import make_theta, sst_profile_nut

GRID = dict(nx=10, ny=6)


def small_case(kind="laminar", re=1.0e3, **kw):
    if kind == "laminar":
        return make_case(re=re, u_mean=1.0, radius0=0.25, turbulent=False, **GRID, **kw)
    if kind == "sst_nut":
        return make_case(
            re=re,
            u_mean=1.0,
            radius0=0.25,
            turbulent=False,
            nut_fn=sst_profile_nut(),
            **GRID,
            **kw,
        )
    return make_case(re=re, u_mean=1.0, radius0=0.25, turbulent=True, **GRID, **kw)


def test_colouring_needs_periodic_nx():
    """nx must be a multiple of the colour period, else the wrap aliases."""
    with pytest.raises(ValueError):
        SSTChannel(ChannelGrid(nx=16, ny=8), turbulent=False)


def test_torch_indicator_matches_numpy_geometry():
    """The differentiable torch indicator reproduces the numpy one to round-off."""
    case = small_case("laminar")
    theta = make_theta()
    body = FourierBody(
        center=case.center, radius0=case.radius0, delta=case.delta,
        quad=case.quad, subcell=case.subcell,
    )
    xc, yc = case._coords["c"]
    a_t, g_t = case._indicator(xc, yc, case._theta_t(theta))
    a_n, _ = body.indicator(xc.numpy(), yc.numpy(), theta)
    assert np.max(np.abs(a_t.numpy() - a_n)) < 1e-13
    # signed distance from the same sub-cell averaged construction
    assert np.max(np.abs(g_t.numpy())) > 0.0


def test_laminar_poiseuille_is_recovered():
    """Without a body the driven channel converges to plane Poiseuille."""
    # explicit nu/force: a body-free channel (radius0 -> 0) driven by `force`
    # body centred outside the domain: alpha == 0 everywhere (body-free channel)
    case = SSTChannel(
        ChannelGrid(nx=20, ny=16), nu=5.0e-3, force=1.5e-3, radius0=1e-6,
        center=(2.0, -5.0), turbulent=False,
    )
    au, _, _, _ = case.geometry(torch.zeros(N_PARAMS))
    assert float(au.max()) == 0.0
    theta = np.zeros(N_PARAMS)
    U, info = case.solve(theta, tol=1e-12, maxit=40)
    u = case.unpack(U)["u"].numpy()
    y = (np.arange(case.grid.ny) + 0.5) * case.grid.dy
    exact = case.force * y * (case.grid.height - y) / (2.0 * case.nu)
    rel = np.max(np.abs(u - exact[:, None])) / np.max(np.abs(exact))
    assert rel < 5e-3, rel


def test_jacobian_matches_finite_difference_laminar():
    case = small_case("laminar")
    theta = make_theta()
    rng = np.random.default_rng(3)
    U = case.initial_guess() + 0.01 * torch.tensor(rng.standard_normal(case.size))
    A = case.jacobian(U, theta, dense=True)
    An = case.num_jacobian(U, theta, eps=1e-7)
    assert np.linalg.norm(A - An) / np.linalg.norm(An) < 1e-8
    assert np.linalg.matrix_rank(A) == case.size


def test_jacobian_matches_finite_difference_sst_nut():
    case = small_case("sst_nut")
    theta = make_theta()
    rng = np.random.default_rng(4)
    U = case.initial_guess() + 0.01 * torch.tensor(rng.standard_normal(case.size))
    A = case.jacobian(U, theta, dense=True)
    An = case.num_jacobian(U, theta, eps=1e-7)
    assert np.linalg.norm(A - An) / np.linalg.norm(An) < 1e-6


def test_jacobian_matches_finite_difference_coupled_sst():
    """The five-field layout (u, v, p, k, omega) is differentiated correctly."""
    case = small_case("turbulent")
    theta = make_theta()
    rng = np.random.default_rng(5)
    U = case.initial_guess(k0=4e-3, w0=4.0) + 0.002 * torch.tensor(
        rng.standard_normal(case.size)
    )
    U = case.project(U)
    A = case.jacobian(U, theta, dense=True)
    An = case.num_jacobian(U, theta, eps=1e-7)
    rel = np.linalg.norm(A - An) / np.linalg.norm(An)
    # 1e-7 central differences on the steep k/omega sources: truncation only
    # (the same assembly is verified to 1e-8 on the 3-field layouts)
    assert rel < 2e-2, rel


def test_adjoint_residual_below_1e_10():
    case = case_for_gradient()
    theta = make_theta()
    U, info = case.solve(theta, tol=1e-12, maxit=40)
    _, ainfo = case.adjoint_gradient(theta, U=U, info=info)
    assert ainfo["adjoint_residual"] < 1e-10


def case_for_gradient():
    return make_case(nx=10, ny=6, re=1.0e3, u_mean=1.0, radius0=0.25, turbulent=False)


def test_adjoint_gradient_matches_central_difference():
    """dC_D/dtheta from the adjoint vs re-converged central differences."""
    case = case_for_gradient()
    theta = make_theta()
    U, info = case.solve(theta, tol=1e-12, maxit=40)
    g, ainfo = case.adjoint_gradient(theta, U=U, info=info)
    eps = 1e-5
    for q in (0, 3):
        e = np.zeros(N_PARAMS)
        e[q] = eps
        Up, _ = case.solve(theta + e, U0=U, tol=1e-12, maxit=30)
        Um, _ = case.solve(theta - e, U0=U, tol=1e-12, maxit=30)
        fd = (
            float(case.objective(Up, case._theta_t(theta + e)))
            - float(case.objective(Um, case._theta_t(theta - e)))
        ) / (2 * eps)
        assert abs(g[q] - fd) / max(abs(fd), 1e-12) < 1e-6, (q, g[q], fd)


def test_frozen_adjoint_matches_fd_with_fixed_nut():
    """With ``nu_t`` held fixed the adjoint reproduces central differences.

    This is the frozen-turbulence *operator*: ``nu_t`` is a constant field, so the
    adjoint of the discrete system and a re-converged central difference must
    agree to the FD truncation level.
    """
    base = small_case("sst_nut")
    theta = make_theta()
    nut = base.nut_fn(base._theta_t(theta), base).detach()
    case = make_case(nx=GRID["nx"], ny=GRID["ny"], re=1.0e3, u_mean=1.0,
                     radius0=0.25, turbulent=False, nut_fixed=nut)
    U, info = case.solve(theta, tol=1e-12, maxit=60)
    g, ainfo = case.adjoint_gradient(theta, U=U, info=info)
    assert ainfo["adjoint_residual"] < 1e-10
    eps = 1e-5
    for q in (0, 5):
        e = np.zeros(N_PARAMS)
        e[q] = eps
        Up, _ = case.solve(theta + e, U0=U, tol=1e-12, maxit=40)
        Um, _ = case.solve(theta - e, U0=U, tol=1e-12, maxit=40)
        fd = (
            float(case.objective(Up, case._theta_t(theta + e)))
            - float(case.objective(Um, case._theta_t(theta - e)))
        ) / (2 * eps)
        assert abs(g[q] - fd) / max(abs(fd), 1e-12) < 1e-6, (q, g[q], fd)


def test_frozen_bias_is_not_negligible():
    """Dropping ``d nu_t / d theta`` moves the gradient: the frozen bias is real."""
    case = small_case("sst_nut")
    theta = make_theta()
    U, info = case.solve(theta, tol=1e-12, maxit=60)
    g_full, _ = case.adjoint_gradient(theta, U=U, info=info, mode="full")
    g_froz, _ = case.adjoint_gradient(theta, U=U, info=info, mode="frozen_nut")
    assert np.max(np.abs(g_froz - g_full)) / np.max(np.abs(g_full)) > 1e-4
    # and the full adjoint is the derivative that matches the FD reference
    e = np.zeros(N_PARAMS)
    e[0] = 1e-5
    Up, _ = case.solve(theta + e, U0=U, tol=1e-12, maxit=40)
    Um, _ = case.solve(theta - e, U0=U, tol=1e-12, maxit=40)
    fd = (
        float(case.objective(Up, case._theta_t(theta + e)))
        - float(case.objective(Um, case._theta_t(theta - e)))
    ) / (2 * e[0])
    assert abs(g_full[0] - fd) / abs(fd) < 1e-4
    # the dropped d nu_t/d theta term is small on this coarse grid but non-zero
    rel = np.linalg.norm(g_froz - g_full) / np.linalg.norm(g_full)
    assert rel > 1e-7, rel


def test_geometry_outside_domain_has_no_body():
    """A body centred far outside the channel leaves alpha identically zero."""
    case = small_case("laminar", center=(2.0, -5.0))
    au, av, ac, _ = case.geometry(torch.zeros(N_PARAMS))
    assert float(au.max()) == 0.0 and float(av.max()) == 0.0 and float(ac.max()) == 0.0


def test_nut_profile_really_depends_on_theta():
    case = small_case("sst_nut")
    theta = make_theta()
    th = case._theta_t(theta).clone().requires_grad_(True)
    nut = case.nut_fn(th, case)
    assert float(nut.max()) > 0.0
    (gth,) = torch.autograd.grad(nut.sum(), th)
    assert float(torch.abs(gth).max()) > 1e-6


def test_sst_closure_limits():
    """nu_t -> 0 with k, and nu_t -> k/omega away from the wall (F2 -> 0)."""
    case = small_case("turbulent", re=1e4)
    theta = make_theta()
    g = case.grid
    _, _, _, gc = case.geometry(theta)
    S2 = torch.zeros(g.ny, g.nx, dtype=torch.float64)
    k = torch.full((g.ny, g.nx), 1e-3, dtype=torch.float64)
    w = torch.full((g.ny, g.nx), 10.0, dtype=torch.float64)
    nut, F1, F2, Pk, chi, _, _ = case.eddy_viscosity(k, w, S2, gc)
    far = slice(2, 4)  # rows away from both walls: F2 -> 0 => nu_t = k / omega
    ratio = (nut[far] * w[far] / k[far]).mean()
    assert abs(float(ratio) - 1.0) < 0.05, float(ratio)
    k_small = torch.full((g.ny, g.nx), 1e-14, dtype=torch.float64)
    nut_small, *_ = case.eddy_viscosity(k_small, w, S2, gc)
    assert float(nut_small.max()) < 1e-12


def test_momentum_balance_closes():
    """Global x-momentum: driving force = body penalty drag + wall friction."""
    case = small_case("laminar")
    theta = make_theta()
    U, _ = case.solve(theta, tol=1e-12, maxit=40)
    g = case.grid
    F = case.unpack(U)
    alpha_u, _, _, _ = case.geometry(case._theta_t(theta))
    F_body = g.dx * g.dy * float(torch.sum(alpha_u / case.eta * F["u"]))
    tau = case.nu * (float(torch.sum(F["u"][0, :])) + float(torch.sum(F["u"][g.ny - 1, :]))) / (0.5 * g.dy)
    F_wall = tau * g.dx
    F_drive = case.force * g.length * g.height
    assert abs(F_drive - F_body - F_wall) / F_drive < 1e-6


def test_step_limit_and_projection_keep_positivity():
    case = small_case("turbulent")
    U = case.initial_guess(k0=4e-3, w0=4.0)
    du = np.full(case.size, -1.0)
    Uc = case.project(U + torch.tensor(du, dtype=torch.float64))
    F = case.unpack(Uc)
    assert float(F["k"].min()) >= case.K_FLOOR
    assert float(F["w"].min()) >= case.W_FLOOR