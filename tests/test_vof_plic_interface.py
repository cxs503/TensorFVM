"""Tests for the PLIC geometric VOF transport and its minimal verification surface.

The suite is deliberately cheap (tens of milliseconds per case): large-grid
studies live in :mod:`tensorfvm.verification.vof_plic_interface`.  About half of
the cases are **negative evidence** — they assert that the input guards reject
invalid data instead of silently repairing it.
"""

import pytest
import torch

from tensorfvm.two_phase_vof import VOFTransport
from tensorfvm.verification.vof_plic_interface import (
    donor_advect_periodic,
    exact_step_field,
    negative_evidence,
    zalesak_case,
)
from tensorfvm.verification.vof_rotation import rotation_velocity, zalesak_disk
from tensorfvm.vof_plic import (
    advect_1d_step,
    advect_plic,
    cut_area,
    interface_width_cells,
    plane_constant,
    validate_alpha,
    validate_directional_speed,
    validate_velocity,
    youngs_normal,
)

torch.set_grad_enabled(False)


# ---------------------------------------------------------------------------
# geometry
# ---------------------------------------------------------------------------
def _numeric_cut_area(n1: float, n2: float, c: float, samples: int = 400) -> float:
    """Reference area by midpoint quadrature over the unit square."""
    h = 1.0 / samples
    x = (torch.arange(samples, dtype=torch.float64) + 0.5) * h
    xx, yy = torch.meshgrid(x, x, indexing="xy")
    inside = (n1 * xx + n2 * yy) <= c
    return float(inside.to(torch.float64).mean().item())


@pytest.mark.parametrize(
    "n1,n2,c",
    [(1.0, 1.0, 0.5), (1.0, 1.0, 1.5), (2.0, 0.0, 1.0), (-1.0, 2.0, 0.6), (0.6, -1.4, 0.3)],
)
def test_cut_area_matches_quadrature(n1, n2, c):
    """The closed-form cut area agrees with direct quadrature to quadrature error."""
    exact = float(cut_area(torch.tensor(n1), torch.tensor(n2), torch.tensor(c)).item())
    ref = _numeric_cut_area(n1, n2, c)
    assert abs(exact - ref) < 5e-3, (exact, ref)


def test_reconstruction_is_volume_consistent():
    """Inverting the plane constant reproduces the cell volume fraction."""
    torch.manual_seed(7)
    alpha = torch.rand(12, 9, dtype=torch.float64)
    n = torch.randn(12, 9, 2, dtype=torch.float64)
    n = n / n.norm(dim=-1, keepdim=True)
    c = plane_constant(alpha, n)
    d = c - torch.where(n[..., 0] < 0, n[..., 0], torch.zeros_like(c))
    d = d - torch.where(n[..., 1] < 0, n[..., 1], torch.zeros_like(c))
    area = cut_area(n[..., 0].abs(), n[..., 1].abs(), d)
    assert torch.allclose(area, alpha, atol=1e-12)


def test_youngs_normal_points_from_liquid_to_gas():
    """A horizontal interface (liquid below) must give n = (0, +1)."""
    alpha = torch.zeros(8, 8, dtype=torch.float64)
    alpha[:4, :] = 1.0
    n = youngs_normal(alpha)
    nx_comp, ny_comp = n[3, 3].tolist()
    assert abs(nx_comp) < 1e-12 and abs(ny_comp - 1.0) < 1e-12
    # and the mirror image gives (0, -1)
    n2 = youngs_normal(1.0 - alpha)
    assert abs(n2[3, 3, 0].item()) < 1e-12
    assert abs(n2[3, 3, 1].item() + 1.0) < 1e-12


# ---------------------------------------------------------------------------
# 1-D step transport
# ---------------------------------------------------------------------------
def test_one_dimensional_translation_is_exact():
    """At CFL = 1 a geometric 1-D step is a pure shift (no diffusion at all)."""
    n = 32
    a = torch.zeros(n, dtype=torch.float64)
    a[: n // 2] = 1.0
    dx = 1.0 / n
    moved = a.clone()
    for _ in range(5):
        moved = advect_1d_step(moved, 1.0, dx, dx)  # CFL = 1 → pure shift
    assert torch.allclose(moved, torch.roll(a, 5), atol=1e-15)
    back = a.clone()
    for _ in range(3):
        back = advect_1d_step(back, 1.0, dx, dx, direction=-1)
    assert torch.allclose(back, torch.roll(a, -3), atol=1e-15)


def test_plic_step_interface_stays_sharp():
    """PLIC keeps a sub-cell transition where the donor-cell scheme diffuses."""
    n, cfl, shift = 64, 0.4, 0.3
    dx = 1.0 / n
    nsteps = int(round(shift / (cfl * dx)))
    dt = shift / nsteps
    u = torch.ones((1, n + 1), dtype=torch.float64)
    v = torch.zeros((2, n), dtype=torch.float64)
    a0 = torch.zeros(1, n, dtype=torch.float64)
    a0[0, : n // 2] = 1.0
    exact = exact_step_field(n, nsteps * dt, dx)

    a = a0.clone()
    for k in range(nsteps):
        a = advect_plic(a, u, v, dt, dx, dx, order="xy" if k % 2 == 0 else "yx", clip=False)
    d = a0.clone()
    for k in range(nsteps):
        d = donor_advect_periodic(d, u, v, dt, dx, dx, order="xy" if k % 2 == 0 else "yx")

    # L1 is the sub-cell offset the step has travelled (0.4 Δx), not diffusion
    assert float((a - exact).abs().mean()) < 0.01
    assert interface_width_cells(a) <= 2
    assert interface_width_cells(d) >= 5
    assert float((d - exact).abs().mean()) > 10 * float((a - exact).abs().mean())


def test_one_dimensional_restriction_matches_two_dimensional_path():
    """The dedicated 1-D step is the 1-D restriction of the 2-D geometric sweep.

    Equivalence holds for a monotone interface, where the Youngs normal of the
    donor cell points downstream — the configuration the 1-D formula encodes.  An
    isolated partial cell between a full and an empty neighbour has an ambiguous
    reconstruction (the normal can point either way) and is covered separately by
    the conservation/boundedness tests.
    """
    n = 24
    dx = 1.0 / n
    dt = 0.4 * dx
    a = torch.zeros(1, n, dtype=torch.float64)
    a[0, : n // 3] = 1.0
    u = torch.ones((1, n + 1), dtype=torch.float64)
    v = torch.zeros((2, n), dtype=torch.float64)
    two_d = advect_plic(a, u, v, dt, dx, dx, order="xy", clip=False)
    one_d = advect_1d_step(a, 1.0, dt, dx).reshape(1, n)
    assert torch.allclose(two_d, one_d, atol=1e-13)


# ---------------------------------------------------------------------------
# two-dimensional advection: conservation, boundedness, shape
# ---------------------------------------------------------------------------
def test_mass_conservation_per_step_below_tolerance():
    """Worst per-step relative volume change must stay below 1e-6."""
    n = 32
    alpha0 = zalesak_disk(n)
    u, v, dx = rotation_velocity(n)
    dt = 0.4 * dx / float(u.abs().max())
    a = alpha0.clone()
    worst = 0.0
    for k in range(20):
        vol0 = float(a.sum().item())
        a = advect_plic(a, u, v, dt, dx, dx, order="xy" if k % 2 == 0 else "yx", clip=False)
        worst = max(worst, abs(float(a.sum().item()) - vol0) / vol0)
    assert worst < 1e-6
    assert abs(float(a.sum().item()) - float(alpha0.sum().item())) / float(
        alpha0.sum().item()
    ) < 1e-12


def test_geometric_transport_is_bounded():
    """No compressive limiter is needed: the raw field stays inside [0, 1]."""
    n = 32
    alpha0 = zalesak_disk(n)
    u, v, dx = rotation_velocity(n)
    dt = 0.4 * dx / float(u.abs().max())
    a = alpha0.clone()
    for k in range(20):
        a = advect_plic(a, u, v, dt, dx, dx, order="xy" if k % 2 == 0 else "yx", clip=False)
    assert float(a.min().item()) > -1e-12
    assert float(a.max().item()) < 1.0 + 1e-12


def test_zalesak_shape_and_area_error():
    """One full revolution: area error < 5 %, shape preserved, peak retained."""
    res = zalesak_case(32, cfl=0.8, revolutions=1)
    plic = res["plic"]
    donor = res["donor_cell"]
    assert plic["area_error"] < 0.05
    assert plic["peak_retention"] > 0.95
    assert plic["peak_retention"] > 3.0 * donor["peak_retention"]
    assert plic["l1"] < 0.5 * donor["l1"]
    assert plic["distorted_fraction"] < 0.5 * donor["distorted_fraction"]
    # volume conservation alone is NOT the test: the donor-cell scheme conserves
    # volume exactly while destroying the shape.
    assert donor["area_error"] < 0.05


def test_zalesak_refuses_partial_revolution():
    """A non-integer revolution has no exact discrete reference — refuse it."""
    with pytest.raises(ValueError, match="positive integer"):
        zalesak_case(16, revolutions=0.5)


def test_shipped_donor_cell_costs_shape():
    """Provenance: the algebraic scheme's peak retention is the known deficiency."""
    n = 32
    alpha0 = zalesak_disk(n)
    u, v, dx = rotation_velocity(n)
    dt = 0.4 * dx / float(u.abs().max())
    a = alpha0.clone()
    for _ in range(10):
        a = donor_advect_periodic(a, u, v, dt, dx, dx)
    assert float(a.max().item()) < 0.8 * float(alpha0.max().item())


# ---------------------------------------------------------------------------
# negative evidence: invalid input must be rejected
# ---------------------------------------------------------------------------
def test_rejects_alpha_above_one():
    bad = torch.zeros(4, 4, dtype=torch.float64)
    bad[0, 0] = 1.2
    with pytest.raises(ValueError, match="above 1"):
        validate_alpha(bad)


def test_rejects_alpha_below_zero():
    bad = torch.zeros(4, 4, dtype=torch.float64)
    bad[1, 2] = -0.1
    with pytest.raises(ValueError, match="below 0"):
        validate_alpha(bad)


def test_rejects_non_finite_alpha():
    bad = torch.zeros(2, 2, dtype=torch.float64)
    bad[0, 0] = float("inf")
    with pytest.raises(ValueError, match="non-finite"):
        validate_alpha(bad)


def test_rejects_negative_directional_speed():
    with pytest.raises(ValueError, match=">= 0"):
        validate_directional_speed(-0.5)
    assert validate_directional_speed(0.0) == 0.0
    assert validate_directional_speed(0.4) == pytest.approx(0.4)


def test_rejects_non_positive_dt_and_oversized_cfl():
    u = torch.zeros((2, 5), dtype=torch.float64)
    v = torch.zeros((3, 4), dtype=torch.float64)
    with pytest.raises(ValueError, match="dt must be > 0"):
        validate_velocity(u, v, 0.0, 0.1, 0.1)
    with pytest.raises(ValueError, match="dt must be > 0"):
        validate_velocity(u, v, -1e-3, 0.1, 0.1)
    with pytest.raises(ValueError, match="CFL"):
        validate_velocity(u + 10.0, v, 0.1, 0.1, 0.1)
    with pytest.raises(ValueError, match="shape mismatch"):
        validate_velocity(u, v[:, :3], 0.01, 0.1, 0.1)


def test_negative_face_velocity_is_physical_and_mirrors():
    """Signed velocities are *not* rejected: leftward advection mirrors rightward."""
    n = 16
    dx = 1.0 / n
    dt = 0.4 * dx
    a = torch.zeros(1, n, dtype=torch.float64)
    a[0, 3:6] = 1.0
    right = advect_plic(
        a,
        torch.ones((1, n + 1), dtype=torch.float64),
        torch.zeros((2, n), dtype=torch.float64),
        dt,
        dx,
        dx,
        clip=False,
    )
    left = advect_plic(
        a,
        -torch.ones((1, n + 1), dtype=torch.float64),
        torch.zeros((2, n), dtype=torch.float64),
        dt,
        dx,
        dx,
        clip=False,
    )
    # the mirrored problem: mirror both the field and the velocity
    mirror = advect_plic(
        a.flip(1),
        -torch.ones((1, n + 1), dtype=torch.float64),
        torch.zeros((2, n), dtype=torch.float64),
        dt,
        dx,
        dx,
        clip=False,
    )
    assert torch.allclose(mirror, right.flip(1), atol=1e-13)
    assert abs(float(left.sum().item()) - float(a.sum().item())) < 1e-14


def test_negative_evidence_report_is_complete():
    """Every guard named in the verification report must actually reject."""
    report = negative_evidence()
    for name, entry in report.items():
        if name == "negative_velocity_accepted":
            assert entry["rejected"] is False
            assert abs(entry["mass_drift"]) < 1e-12
        else:
            assert entry["rejected"] is True, name


# ---------------------------------------------------------------------------
# consistency with the shipped transport
# ---------------------------------------------------------------------------
def test_donor_reference_matches_shipped_upwind_stencil():
    """The periodic donor reference reproduces the library stencil on a periodic field."""
    n = 16
    dx = 1.0 / n
    dt = 0.2 * dx
    torch.manual_seed(3)
    a = torch.rand(1, n, dtype=torch.float64) * 0.2  # smooth, so the clip is inactive
    u = torch.full((1, n + 1), 1.0, dtype=torch.float64)
    v = torch.zeros((2, n), dtype=torch.float64)
    ref = donor_advect_periodic(a, u, v, dt, dx, dx)
    lib = VOFTransport(1, n, dx, dx, scheme="upwind").advect(a, u, v, dt)
    # interior cells agree; the boundary cells differ by construction
    # (periodic halo vs zero-gradient ghost), which is exactly the measured
    # transmissive-BC drift reported in the artefacts.
    assert torch.allclose(ref[0, 1:-1], lib[0, 1:-1], atol=1e-13)


def test_interface_width_metric_sanity():
    """The sharpness metric is 0 for a pure step and the band count for a ramp."""
    step = torch.zeros(1, 16, dtype=torch.float64)
    step[0, :8] = 1.0
    assert interface_width_cells(step) == 0.0
    ramp = torch.linspace(0.0, 1.0, 16, dtype=torch.float64).reshape(1, 16)
    assert interface_width_cells(ramp) == pytest.approx(12.0)
    # a periodic step has two transitions; only the widest band is reported
    one_step = torch.zeros(1, 16, dtype=torch.float64)
    one_step[0, :8] = 1.0
    assert interface_width_cells(one_step) == 0.0
    # the band test is strict, so cells sitting exactly on 0 or 1 never count
    assert interface_width_cells(step, lo=0.0, hi=1.0) == 0.0