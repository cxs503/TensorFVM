"""Verification tests for solid-body-rotation VOF advection.

These tests close a verification debt of :mod:`tensorfvm.two_phase_vof`: the
donor-cell transport claimed bounded, conservative behaviour but was only
tested for trivial fields (all-zero / all-one velocity).  Against an exact
advection benchmark (rigid rotation: the initial condition *is* the analytic
solution after a whole number of revolutions) the production scheme leaves
``[0, 1]`` before its final clamp and therefore loses mass.  The flux-limited
scheme added in :mod:`tensorfvm.vof_fct` is bounded without clipping and
conserves exactly.
"""

import math

import torch

from tensorfvm.two_phase_vof import VOFTransport
from tensorfvm.vof_fct import LimitedVOFTransport
from tensorfvm.verification.vof_rotation import (
    cosine_hill,
    divergence_check,
    rotate,
    rotation_velocity,
    unbounded_evidence,
    zalesak_disk,
)


def _interior_blob(n=32, radius=0.12):
    """Smooth blob with a two-cell zero margin, so nothing touches the boundary."""
    dx = 1.0 / n
    centres = (torch.arange(n, dtype=torch.float64) + 0.5) * dx
    x, y = torch.meshgrid(centres, centres, indexing="xy")
    r = torch.sqrt((x - 0.5) ** 2 + (y - 0.5) ** 2)
    alpha = torch.zeros_like(r)
    inner = r < radius
    alpha[inner] = 0.5 * (1 + torch.cos(math.pi * r[inner] / radius))
    return alpha


def _one_step(transport, alpha, u, v, dx, cfl=0.4):
    dt = cfl * dx / float(max(u.abs().max(), v.abs().max()))
    return transport.advect(alpha, u, v, dt), dt


def test_rotation_velocity_is_discretely_divergence_free():
    assert divergence_check(32) == 0.0


def test_donor_cell_raw_update_leaves_unit_interval():
    evidence = unbounded_evidence(32)
    assert evidence["cells_out_of_range"] > 0
    assert evidence["raw_minimum"] < 0.0
    assert evidence["clipped_mass_gain_per_step"] > 0.0


def test_flux_limited_scheme_conserves_mass_to_machine_precision():
    n, dx = 32, 1.0 / 32
    alpha = _interior_blob(n)
    u, v, _ = rotation_velocity(n)
    transport = LimitedVOFTransport(n, n, dx, dx)
    new, _ = _one_step(transport, alpha, u, v, dx)
    drift = abs(float(new.sum()) - float(alpha.sum()))
    assert drift < 1e-12, drift


def test_donor_cell_scheme_loses_mass_on_the_benchmark_hill():
    n, dx = 32, 1.0 / 32
    alpha = cosine_hill(n)
    u, v, _ = rotation_velocity(n)
    transport = VOFTransport(n, n, dx, dx)
    new, _ = _one_step(transport, alpha, u, v, dx)
    assert abs(float(new.sum()) - float(alpha.sum())) > 1e-12


def test_flux_limited_scheme_stays_bounded():
    n, dx = 32, 1.0 / 32
    alpha0 = zalesak_disk(n)
    u, v, _ = rotation_velocity(n)
    alpha, _, _ = rotate(LimitedVOFTransport(n, n, dx, dx), alpha0, u, v, dx)
    assert float(alpha.min()) >= 0.0
    assert float(alpha.max()) <= 1.0


def test_smooth_hill_error_decreases_with_refinement():
    rows = []
    for n in (32, 64):
        alpha0 = cosine_hill(n)
        u, v, dx = rotation_velocity(n)
        alpha, _, _ = rotate(VOFTransport(n, n, dx, dx), alpha0, u, v, dx)
        rows.append(float((alpha - alpha0).abs().sum()) * dx * dx)  # L1 volume error
    assert rows[1] < rows[0]
    order = math.log(rows[0] / rows[1]) / math.log(2.0)
    # Honest bound: first-order transport is heavily smeared at CFL 0.4 and
    # shows sub-first-order convergence on this benchmark.
    assert 0.2 < order < 1.5, order