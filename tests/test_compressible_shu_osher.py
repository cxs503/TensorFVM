"""Tests for the Shu-Osher shock-entropy wave interaction case."""

from __future__ import annotations

import math

import torch

from tensorfvm.compressible.cases import ShuOsherConfig, shu_osher, GAMMA_AIR


def test_shu_osher_initial_state():
    """Verify the initial condition matches the Shu-Osher specification."""
    cfg = ShuOsherConfig(nx=200)
    solver, meta = shu_osher(cfg)

    xc = meta["xc"]
    # Post-shock state (left of shock_pos=4.0)
    left = xc < cfg.shock_pos
    post = cfg.post_shock

    # Use primitive() to get actual primitive variables
    prim = solver.primitive()

    # Check post-shock region
    rho_left = prim["rho"][left, 0]
    assert torch.allclose(rho_left, torch.full_like(rho_left, post["rho"]), atol=1e-10), \
        f"Post-shock rho should be {post['rho']}"

    # Check pre-shock region: rho = 1 + 0.2*sin(pi*x)
    right = ~left
    xc_right = xc[right]
    rho_expected = cfg.rho_pre + cfg.entropy_amp * torch.sin(cfg.entropy_wavenumber * xc_right)
    rho_right = prim["rho"][right, 0]
    assert torch.allclose(rho_right, rho_expected, atol=1e-10), \
        "Pre-shock rho should follow sinusoidal perturbation"

    # Check pre-shock velocity is zero
    u_right = prim["u"][right, 0]
    assert torch.allclose(u_right, torch.zeros_like(u_right), atol=1e-10), \
        "Pre-shock velocity should be zero"

    # Check pre-shock pressure is uniform
    p_right = prim["p"][right, 0]
    assert torch.allclose(p_right, torch.full_like(p_right, cfg.p_pre), atol=1e-10), \
        "Pre-shock pressure should be uniform"


def test_shu_osher_post_shock_from_rh():
    """Verify post-shock state is computed from Rankine-Hugoniot (zero hand-copying)."""
    cfg = ShuOsherConfig(nx=100)
    post = cfg.post_shock

    # For Mach 3, gamma=1.4, rho=1, p=1:
    # rho2/rho1 = (gamma+1)*M^2 / ((gamma-1)*M^2 + 2) = 2.4*9 / (0.4*9 + 2) = 21.6/5.6 = 3.857143
    expected_rho_ratio = (GAMMA_AIR + 1.0) * cfg.mach**2 / ((GAMMA_AIR - 1.0) * cfg.mach**2 + 2.0)
    expected_rho = cfg.rho_pre * expected_rho_ratio
    assert abs(post["rho"] - expected_rho) < 1e-6, \
        f"Post-shock rho {post['rho']} != expected {expected_rho}"

    # p2/p1 = 2*gamma*M^2/(gamma+1) - (gamma-1)/(gamma+1) = 2.8*9/2.4 - 0.4/2.4 = 10.5 - 0.1667 = 10.3333
    expected_p_ratio = 2.0 * GAMMA_AIR * cfg.mach**2 / (GAMMA_AIR + 1.0) - (GAMMA_AIR - 1.0) / (GAMMA_AIR + 1.0)
    expected_p = cfg.p_pre * expected_p_ratio
    assert abs(post["p"] - expected_p) < 1e-6, \
        f"Post-shock p {post['p']} != expected {expected_p}"


def test_shu_osher_conservation():
    """Mass and energy should be conserved (with boundary ledger) for outflow BCs."""
    cfg = ShuOsherConfig(nx=100, t_end=0.5)
    solver, meta = shu_osher(cfg)
    initial = solver.integrals()
    solver.run(0.5)
    drift = solver.budget_drift(initial)

    # With outflow boundaries, the boundary ledger accounts for what leaves
    assert abs(drift["mass"]) < 1e-10, f"Mass drift {drift['mass']} too large"
    assert abs(drift["energy"]) < 1e-10, f"Energy drift {drift['energy']} too large"


def test_shu_osher_runs_to_completion():
    """The solver should complete without positivity violations for reasonable CFL."""
    cfg = ShuOsherConfig(nx=200, t_end=1.0, cfl=0.4)
    solver, meta = shu_osher(cfg)
    solver.run(1.0)

    prim = solver.primitive()
    assert (prim["rho"] > 0).all(), "Density should remain positive"
    assert (prim["p"] > 0).all(), "Pressure should remain positive"
    assert solver.positivity_fixes == 0, "No positivity fixes should be needed"


def test_shu_osher_density_bounds():
    """Density should remain within physically reasonable bounds after shock passage."""
    cfg = ShuOsherConfig(nx=200, t_end=1.8, cfl=0.4)
    solver, meta = shu_osher(cfg)
    solver.run(1.8)

    prim = solver.primitive()
    rho = prim["rho"][:, 0]
    # After the shock exits, density should be close to post-shock value (~3.857)
    # with small oscillations from the entropy wave interaction
    assert rho.min() > 3.0, f"Min density {rho.min():.4f} too low"
    assert rho.max() < 5.0, f"Max density {rho.max():.4f} too high"
    # Mean density should be close to post-shock value
    mean_rho = float(rho.mean())
    assert abs(mean_rho - 3.857) < 0.5, f"Mean density {mean_rho:.4f} far from post-shock"
