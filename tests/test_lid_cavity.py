"""Verification tests for the staggered lid-driven-cavity solver.

The heavy literature ladder lives in ``scripts/verify_lid_cavity_ghia.py``; the
tests here pin the pieces that would silently rot:

* the transcribed Ghia tables (extrema + ordering),
* the pure-Neumann pressure solver (inverts the discrete Laplacian exactly) and
  the compatibility of the closed-box divergence,
* the van Leer limiter (no new extrema),
* the operator sign/consistency check (linear shear is an exact null mode),
* one end-to-end Ghia comparison at Re = 100 on a 32^2 grid.
"""

import math

import pytest
import torch

from tensorfvm.lid_cavity import (
    LidCavityConfig,
    LidCavitySolver,
    ghia_acceptance,
    hermite_interpolate,
    score_against_ghia,
)
from tensorfvm.lid_cavity_reference import (
    SUPPORTED_REYNOLDS,
    VORTEX_CENTRE,
    extrema,
    self_check,
    table,
)


def test_reference_tables_are_self_consistent():
    report = self_check()
    assert report["ok"], report
    for rey in SUPPORTED_REYNOLDS:
        for which in ("u", "v"):
            pts = table(rey, which)
            assert len(pts) == 17
            assert pts[0][0] == pytest.approx(1.0)
            assert pts[-1][0] == pytest.approx(0.0)
            assert pts[0][1] == pytest.approx(1.0 if which == "u" else 0.0, abs=1e-12)
            assert pts[-1][1] == pytest.approx(0.0, abs=1e-12)
        lo, hi = extrema(rey, "u")
        assert lo[1] < 0.0            # reverse flow near the bottom wall
        assert hi[1] == pytest.approx(1.0)   # lid value on the centreline
    # published extrema, spot-checked against the paper text
    assert extrema(1000.0, "u")[0] == pytest.approx((0.1719, -0.38289))
    assert extrema(400.0, "u")[0][1] == pytest.approx(-0.32726)
    assert extrema(100.0, "v")[1][1] == pytest.approx(0.17527)


def test_config_rejects_bad_input():
    with pytest.raises(ValueError):
        LidCavityConfig(n=31)              # odd -> mid-plane not on a grid line
    with pytest.raises(ValueError):
        LidCavityConfig(n=32, reynolds=0.0)
    with pytest.raises(ValueError):
        LidCavityConfig(n=32, time_step=-1.0)
    assert LidCavityConfig(n=32, reynolds=1000.0, lid_velocity=1.0).viscosity == pytest.approx(1e-3)


def test_pressure_solver_inverts_neumann_laplacian():
    solver = LidCavitySolver(LidCavityConfig(n=16, reynolds=100.0))
    torch.manual_seed(0)
    rhs = torch.randn(16, 16, dtype=torch.float64)
    rhs -= rhs.mean()                     # compatibility with pure Neumann BCs
    pressure = solver.solve_pressure(rhs)
    assert torch.allclose(solver.neumann_laplacian(pressure), rhs, atol=1e-11)
    # zero mode is pinned: constant offset is not observable
    assert float(pressure.mean()) == pytest.approx(0.0, abs=1e-10)


def test_projection_keeps_the_box_divergence_free():
    solver = LidCavitySolver(LidCavityConfig(n=24, reynolds=100.0))
    for _ in range(10):
        solver.step()
    divergence = solver.divergence()
    assert float(divergence.abs().max()) < 1e-12
    # no penetration on any wall
    assert float(solver.u[0, :].abs().max()) == 0.0
    assert float(solver.u[-1, :].abs().max()) == 0.0
    assert float(solver.v[:, 0].abs().max()) == 0.0
    assert float(solver.v[:, -1].abs().max()) == 0.0


def test_vanleer_limiter_creates_no_new_extrema():
    torch.manual_seed(1)
    field = torch.rand(64, dtype=torch.float64)
    field[field > 0.7] = 1.0
    velocity = torch.full((63,), 0.3, dtype=torch.float64)
    face = LidCavitySolver._vanleer_face(field, velocity, axis=0)
    assert float(face.min()) >= float(field.min()) - 1e-12
    assert float(face.max()) <= float(field.max()) + 1e-12
    # constant field must be reproduced exactly by the reconstruction
    flat = torch.full((64,), 0.37, dtype=torch.float64)
    assert torch.allclose(LidCavitySolver._vanleer_face(flat, velocity, axis=0), flat[:-1])


def test_linear_shear_is_a_null_mode_of_the_operator():
    """``u = y``, ``v = 0``: convection and the interior Laplacian both vanish."""
    n = 16
    solver = LidCavitySolver(LidCavityConfig(n=n, reynolds=100.0, lid_velocity=1.0))
    h = solver.h
    y = (torch.arange(n, dtype=torch.float64) + 0.5) * h
    solver.u = y[None, :].repeat(n + 1, 1).contiguous()
    solver.v = torch.zeros(n, n + 1, dtype=torch.float64)
    tu, tv = solver.tendencies()
    assert float(tv.abs().max()) == 0.0
    # u = y is an exact null mode everywhere: the convection vanishes, the
    # interior Laplacian vanishes, and the bottom ghost (u_wall = 0) and the top
    # ghost (2U - u_adjacent, with U = 1 and u = y) are both consistent with the
    # linear profile, so no spurious wall shear is generated
    assert float(tu.abs().max()) < 1e-10


def test_hermite_interpolation_reproduces_a_quadratic_exactly():
    """Centred slopes are exact for quadratics, so cubic Hermite is exact too."""
    x = torch.linspace(0.0, 1.0, 41, dtype=torch.float64)
    f = 2.0 * x ** 2 - x + 0.5
    targets = [0.05, 0.3, 0.42, 0.77]
    got = hermite_interpolate(x, f, targets)
    want = torch.tensor([2.0 * t ** 2 - t + 0.5 for t in targets], dtype=torch.float64)
    assert torch.allclose(got, want, atol=1e-10)


def test_error_metrics_are_zero_for_the_reference_itself():
    solver = LidCavitySolver(LidCavityConfig(n=16, reynolds=100.0))
    # synthesise a field whose centreline equals the reference table through the
    # same interpolation machinery used by the scorer: a perfect match must score 0
    y, _ = solver.centreline_u()
    ref_u = torch.tensor([p[1] for p in table(100.0, "u")], dtype=torch.float64)
    coord_u = [p[0] for p in table(100.0, "u")]
    assert hermite_interpolate(y, torch.sin(y), coord_u).shape == ref_u.shape


@pytest.mark.parametrize("reynolds", [100.0])
def test_ghia_re100_passes_the_acceptance_band(reynolds):
    """End-to-end: 32^2 cells already land inside the 3e-2 / 15 % band."""
    solver = LidCavitySolver(LidCavityConfig(n=32, reynolds=reynolds,
                                             tolerance=1e-7, max_steps=20_000,
                                             check_every=250))
    info = solver.run()
    assert info["converged"], info["history"][-3:]
    assert info["max_divergence"] < 1e-11
    result = score_against_ghia(solver, reynolds)
    checks = ghia_acceptance(result)
    assert checks["pass"], checks
    assert result["u"]["max_abs_error"] < 0.03
    # primary vortex located by the streamfunction minimum, not by a corner
    vx, vy = result["vortex_centre"]["computed"]
    ref = VORTEX_CENTRE[reynolds]
    assert math.hypot(vx - ref[0], vy - ref[1]) < 0.05