"""Verification for the free-surface VOF boundary-flux witness.

The module :mod:`tensorfvm.free_surface_bc` advertises ``outflow_vof_flux`` as
the quantity that closes the global liquid-volume ledger, but it used to be
pinned only by a zero-velocity test.  The old formula clamped the *column sum*
of the boundary velocity and multiplied by the *mean* boundary volume fraction,
which (a) cancels co-located inflow against outflow and (b) mis-weights a
non-uniform profile.  These tests pin the per-face witness:

* exact agreement with an independent scalar per-face reference loop,
* ledger closure against the repository's own advection operator,
* analytic closure for a uniform stream,
* a negative control showing the retired column-sum formula stays far off,
* the transfer/shape contract, including the staggered MAC shapes.
"""

from __future__ import annotations

import math

import pytest
import torch

from tensorfvm.free_surface_bc import (
    FreeSurfaceBC,
    net_vof_flux,
    outflow_vof_flux,
)
from tensorfvm.two_phase_vof import VOFTransport

DTYPE = torch.float64
SCALED_GATE = 1e-13


def _random_alpha(ny, nx, seed=0, lo=0.05, hi=0.95):
    g = torch.Generator().manual_seed(seed + 977)
    return lo + (hi - lo) * torch.rand(ny, nx, generator=g, dtype=DTYPE)


def _divergence_free(ny, nx, dx, dy, seed, target=0.5):
    """MAC velocity from a corner streamfunction: cell divergence telescopes to 0."""
    g = torch.Generator().manual_seed(seed)
    jj = torch.arange(ny + 1, dtype=DTYPE).view(-1, 1)
    ii = torch.arange(nx + 1, dtype=DTYPE).view(1, -1)
    psi = (
        torch.sin(math.pi * jj / ny) * torch.sin(math.pi * ii / nx)
        + 0.2 * (torch.rand(ny + 1, nx + 1, generator=g, dtype=DTYPE) - 0.5)
    )
    u = -(psi[1:, :] - psi[:-1, :]) / dy
    v = (psi[:, 1:] - psi[:, :-1]) / dx
    scale = target / max(float(u.abs().max()), float(v.abs().max()))
    return u * scale, v * scale


def _reference(alpha, u, v, dx, dy, dt):
    """Scalar per-face loop, signed positive out of the domain."""
    ny, nx = alpha.shape
    pairs = []
    for j in range(ny):
        pairs.append((+float(u[j, nx].item()), float(alpha[j, nx - 1].item()), dy))
        pairs.append((-float(u[j, 0].item()), float(alpha[j, 0].item()), dy))
    for i in range(nx):
        pairs.append((+float(v[ny, i].item()), float(alpha[ny - 1, i].item()), dx))
        pairs.append((-float(v[0, i].item()), float(alpha[0, i].item()), dx))
    gross = 0.0
    net = 0.0
    for velocity, a, length in pairs:
        flux = dt * velocity * length * a
        net += flux
        gross += max(flux, 0.0)
    return gross, net


def _retired_column_sum(alpha, u, v, dx, dy, dt):
    """The formula this module used to ship; kept as a negative control."""
    east_out = max(0.0, float(u[:, -1].sum().item())) * dy
    west_out = max(0.0, float(-u[:, 0].sum().item())) * dy
    north_out = max(0.0, float(v[-1, :].sum().item())) * dx
    south_out = max(0.0, float(-v[0, :].sum().item())) * dx
    a_e = float(alpha[:, -1].mean().item())
    a_w = float(alpha[:, 0].mean().item())
    a_n = float(alpha[-1, :].mean().item())
    a_s = float(alpha[0, :].mean().item())
    return dt * (east_out * a_e + west_out * a_w + north_out * a_n + south_out * a_s)


@pytest.mark.parametrize("seed", [1, 2, 3, 4])
def test_witness_matches_scalar_per_face_reference(seed):
    """Vectorised witness must equal an independent scalar per-face loop."""
    ny, nx, dx, dy, dt = 6, 8, 0.1, 0.1, 0.02
    alpha = _random_alpha(ny, nx, seed)
    g = torch.Generator().manual_seed(seed)
    u = torch.rand(ny, nx + 1, generator=g, dtype=DTYPE) - 0.5
    v = torch.rand(ny + 1, nx, generator=g, dtype=DTYPE) - 0.5

    ref_gross, ref_net = _reference(alpha, u, v, dx, dy, dt)
    assert outflow_vof_flux(alpha, u, v, dx, dy, dt) == pytest.approx(ref_gross, rel=1e-14, abs=1e-18)
    assert net_vof_flux(alpha, u, v, dx, dy, dt) == pytest.approx(ref_net, rel=1e-14, abs=1e-18)


def test_net_flux_closes_the_ledger_against_upwind_advection():
    """(sum(a_new) - sum(a_ref))*dx*dy + net_vof_flux == 0 to round-off."""
    ny, nx, dx, dy, dt = 6, 8, 0.1, 0.1, 0.02
    alpha = _random_alpha(ny, nx, 7)
    u, v = _divergence_free(ny, nx, dx, dy, 7)

    alpha_new = VOFTransport(ny, nx, dx, dy, scheme="upwind").advect(alpha, u, v, dt)
    delta_volume = float((alpha_new.sum() - alpha.sum()).item()) * dx * dy
    net = net_vof_flux(alpha, u, v, dx, dy, dt)

    # dV is a difference of O(1) sums, so judge the residual against the total.
    total_volume = float(alpha.sum().item()) * dx * dy
    assert abs(delta_volume + net) / total_volume < SCALED_GATE

    bc = FreeSurfaceBC(ny, nx, dx, dy)
    assert abs(bc.ledger_residual(alpha, alpha_new, u, v, dt)) / total_volume < SCALED_GATE


def test_uniform_stream_translation_closes_analytically():
    """Uniform stream U: dV must equal dt*U*dy*(sum(a_inlet) - sum(a_outlet))."""
    ny, nx, dx, dy, dt, U = 6, 8, 0.1, 0.1, 0.02, 0.4
    alpha = _random_alpha(ny, nx, 11)
    u = torch.full((ny, nx + 1), U, dtype=DTYPE)
    v = torch.zeros((ny + 1, nx), dtype=DTYPE)

    exact = dt * U * dy * (float(alpha[:, 0].sum().item()) - float(alpha[:, -1].sum().item()))
    alpha_new = VOFTransport(ny, nx, dx, dy, scheme="upwind").advect(alpha, u, v, dt)
    measured = float((alpha_new.sum() - alpha.sum()).item()) * dx * dy

    total_volume = float(alpha.sum().item()) * dx * dy
    assert abs(measured - exact) / total_volume < SCALED_GATE
    # And the witness reproduces the analytic volume change, not just the operator.
    assert net_vof_flux(alpha, u, v, dx, dy, dt) == pytest.approx(-exact, rel=0.0, abs=1e-12)


def test_co_located_inflow_does_not_cancel_outflow():
    """The retired bug in one line: a single boundary column with net zero flow.

    Half the east column leaves, half enters, so the column *sum* is zero and the
    retired formula reports no outflow at all even though liquid leaves.
    """
    ny, nx, dx, dy, dt = 4, 4, 0.1, 0.1, 0.01
    alpha = torch.full((ny, nx), 0.6, dtype=DTYPE)
    u = torch.zeros(ny, nx + 1, dtype=DTYPE)
    v = torch.zeros(ny + 1, nx, dtype=DTYPE)
    u[:, -1] = torch.tensor([0.5, 0.5, -0.5, -0.5], dtype=DTYPE)  # 2 out, 2 in

    gross = outflow_vof_flux(alpha, u, v, dx, dy, dt)
    expected = dt * dy * (2 * 0.5 * 0.6)
    assert gross == pytest.approx(expected, rel=1e-14)
    assert gross > 0.0

    retired = _retired_column_sum(alpha, u, v, dx, dy, dt)
    assert retired == pytest.approx(0.0, abs=1e-18)  # column sum cancels exactly
    assert _reference(alpha, u, v, dx, dy, dt)[0] == pytest.approx(expected, rel=1e-14)


def test_retired_column_sum_formula_is_rejected_as_a_ledger_witness():
    """Negative control: the retired formula must stay far outside the gate."""
    ny, nx, dx, dy, dt = 6, 8, 0.1, 0.1, 0.02
    for seed in (1, 2, 3):
        alpha = _random_alpha(ny, nx, seed)
        u, v = _divergence_free(ny, nx, dx, dy, seed)
        ref_gross, _ = _reference(alpha, u, v, dx, dy, dt)
        retired = _retired_column_sum(alpha, u, v, dx, dy, dt)
        assert abs(retired - ref_gross) / abs(ref_gross) > 1e-2


def test_zero_velocity_gives_zero_flux():
    """Backwards-compatible contract: no flow, no boundary flux."""
    alpha = _random_alpha(5, 5, 3)
    u = torch.zeros(5, 6, dtype=DTYPE)
    v = torch.zeros(6, 5, dtype=DTYPE)
    assert outflow_vof_flux(alpha, u, v, 0.1, 0.1, 0.01) == 0.0
    assert net_vof_flux(alpha, u, v, 0.1, 0.1, 0.01) == 0.0


def test_inflow_only_gives_zero_gross_and_negative_net():
    """Liquid only enters: gross outflow 0, net flux negative (domain gains)."""
    ny, nx = 4, 4
    alpha = torch.full((ny, nx), 0.5, dtype=DTYPE)
    u = torch.zeros(ny, nx + 1, dtype=DTYPE)
    u[:, 0] = 0.3  # west boundary inflow
    v = torch.zeros(ny + 1, nx, dtype=DTYPE)
    net = net_vof_flux(alpha, u, v, 0.1, 0.1, 0.01)
    assert outflow_vof_flux(alpha, u, v, 0.1, 0.1, 0.01) == 0.0
    assert net == pytest.approx(-0.01 * 0.1 * (ny * 0.3 * 0.5), rel=1e-14)


def test_staggered_shape_contract_is_enforced():
    """u must be (ny, nx+1) and v (ny+1, nx); silent broadcasting is a defect."""
    alpha = torch.full((4, 5), 0.5, dtype=DTYPE)
    good_u = torch.zeros(4, 6, dtype=DTYPE)
    good_v = torch.zeros(5, 5, dtype=DTYPE)
    assert net_vof_flux(alpha, good_u, good_v, 0.1, 0.1, 0.01) == 0.0
    with pytest.raises(ValueError, match="u must be the east-face field"):
        net_vof_flux(alpha, torch.zeros(4, 5, dtype=DTYPE), good_v, 0.1, 0.1, 0.01)
    with pytest.raises(ValueError, match="v must be the north-face field"):
        net_vof_flux(alpha, good_u, torch.zeros(4, 5, dtype=DTYPE), 0.1, 0.1, 0.01)
    with pytest.raises(ValueError, match="2-D cell field"):
        net_vof_flux(torch.zeros(4, 5, 1, dtype=DTYPE), good_u, good_v, 0.1, 0.1, 0.01)
    with pytest.raises(ValueError, match="finite"):
        net_vof_flux(alpha * float("nan"), good_u, good_v, 0.1, 0.1, 0.01)


def test_ledger_residual_helper_shape_and_value():
    """ledger_residual validates shapes and returns the unaccounted volume."""
    bc = FreeSurfaceBC(4, 5, 0.1, 0.1)
    alpha = torch.full((4, 5), 0.5, dtype=DTYPE)
    u = torch.zeros(4, 6, dtype=DTYPE)
    v = torch.zeros(5, 5, dtype=DTYPE)
    assert bc.ledger_residual(alpha, alpha, u, v, 0.01) == 0.0
    with pytest.raises(ValueError, match="alpha_new must have shape"):
        bc.ledger_residual(alpha, torch.full((4, 4), 0.5, dtype=DTYPE), u, v, 0.01)
    # A step that loses liquid without a matching boundary flux must be flagged.
    alpha_new = alpha - 0.1
    residual = bc.ledger_residual(alpha, alpha_new, u, v, 0.01)
    assert residual == pytest.approx(-0.1 * 4 * 5 * 0.1 * 0.1, rel=1e-13)