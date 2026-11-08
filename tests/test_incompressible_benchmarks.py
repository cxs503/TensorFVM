"""Tests for the incompressible verification surface (backward-facing step,
sudden expansion, natural convection).

The gates are deliberately of three kinds:

* **operator identity** -- the momentum tendency must reproduce
  :class:`tensorfvm.lid_cavity.LidCavitySolver` bit-for-bit on the same fields,
  so the new module cannot silently drift away from the operator that is already
  validated against Ghia et al. (1982);
* **discrete invariants** -- post-projection divergence, inlet/outlet mass
  balance, boundedness of the buoyant solution;
* **reference distance** -- one cheap natural-convection run against the de Vahl
  Davis (1983) Nusselt number with a loose (15 %) band, so the physics is
  checked without making CI pay for a production grid.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from tensorfvm.incompressible_benchmarks import (
    BORDA_CARNOT_AR4,
    DE_VAHL_DAVIS,
    StaggeredFlow,
    backward_facing_step,
    mean_nusselt,
    natural_convection,
    parabolic_profile,
    reattachment_length,
    sudden_expansion,
)

ROOT = Path(__file__).resolve().parent.parent


def test_borda_carnot_value():
    assert BORDA_CARNOT_AR4 == pytest.approx(0.5625, rel=1e-12)


def test_tendency_matches_validated_lid_cavity_operator():
    """The momentum operator is the Ghia-validated lid-cavity operator."""
    torch = pytest.importorskip("torch")
    from tensorfvm.lid_cavity import LidCavityConfig, LidCavitySolver

    n = 12
    reference = LidCavitySolver(LidCavityConfig(n=n, reynolds=100.0))
    rng = np.random.default_rng(0)
    u = rng.standard_normal((n + 1, n))
    v = rng.standard_normal((n, n + 1))
    reference.u = torch.tensor(u)
    reference.v = torch.tensor(v)
    tu, tv = reference.tendencies()

    flow = StaggeredFlow(n, n, 1.0 / n, reference.nu, outlet="wall", wall_hi=1.0)
    flow.u = u.copy()
    flow.v = v.copy()
    mu, mv = flow.tendencies()
    assert np.abs(mu - tu.numpy()).max() < 1e-12
    assert np.abs(mv - tv.numpy()).max() < 1e-12


def test_closed_box_is_divergence_free_and_bounded():
    n = 12
    flow = StaggeredFlow(n, n, 1.0 / n, nu=0.01, outlet="wall", wall_hi=1.0, cfl=0.2)
    for _ in range(400):
        flow.step()
    assert np.abs(flow.divergence()).max() < 1e-10
    assert np.abs(flow.u).max() < 1.5          # bounded by the lid speed


def test_open_channel_conserves_mass():
    nx, ny, h = 40, 16, 1.0 / 16
    inlet = parabolic_profile(ny, 1.0, 0.0, h, mean_velocity=1.0)
    flow = StaggeredFlow(nx, ny, h, nu=0.02, inlet=inlet, outlet="open", cfl=0.2)
    for _ in range(800):
        flow.step()
    assert np.abs(flow.divergence()).max() < 1e-10
    inflow = float(np.sum(flow.u[0, :]))
    outflow = float(np.sum(flow.u[-1, :]))
    assert abs(inflow - outflow) / inflow < 1e-8
    assert np.abs(flow.u[1:-1, :] - inlet[None, :]).max() < 0.05


def test_reattachment_length_recovers_a_synthetic_crossing():
    x = np.linspace(0.0, 10.0, 101)
    tau = -(x - 2.0) * (6.4 - x)              # reversed shear between 2 and 6.4
    assert reattachment_length(x, tau) == pytest.approx(6.4, abs=1e-2)
    # a bubble that is still open at the outlet has no reattachment point
    assert math.isnan(reattachment_length(x, 1.0 - x / 6.4))
    # a strictly one-signed shear has no reattachment point
    assert math.isnan(reattachment_length(x, 1.0 + x / 10.0))
    assert math.isnan(reattachment_length(x, -1.0 - x / 10.0))


def test_backward_facing_step_geometry_and_inlet():
    flow = backward_facing_step(re_h=800.0, cells_per_step=8)
    assert flow.solid.shape == (flow.nx, flow.ny)
    # the step blocks the lower half of the upstream (4 H long) section only
    upstream = flow.x_coordinates() - 4.0 < 0.0
    assert flow.solid[upstream, : flow.ny // 2].all()
    assert not flow.solid[~upstream, :].any()
    assert float(np.sum(flow.inlet)) * flow.h == pytest.approx(1.0, rel=2e-2)
    assert flow.nu == pytest.approx(1.0 / 800.0)


def test_sudden_expansion_geometry_and_area_ratio():
    flow = sudden_expansion(area_ratio=4.0, re_d=100.0, cells_per_diameter=8)
    upstream = flow.x_coordinates() - 3.0 < 0.0
    blocked = flow.solid[upstream, :].sum(axis=1)
    assert blocked.max() == flow.ny * 3 // 4          # upstream area is 1/4
    assert not flow.solid[~upstream, :].any()
    assert float(np.sum(flow.inlet)) * flow.h == pytest.approx(1.0, rel=2e-2)


def test_natural_convection_stays_bounded():
    """Boundedness gate on a deliberately under-resolved buoyant run.

    Accuracy is *not* asserted here -- the reference comparison lives in the
    artefact table ``docs/benchmarks/incompressible-verification.md`` because a
    production grid costs minutes of wall clock.  On a 12^2 grid the explicit
    buoyancy term is stiff, so the gate is (i) finite fields, (ii) machine-zero
    divergence and (iii) a Nusselt number in the physical window between pure
    conduction (Nu = 1) and twice the de Vahl Davis value.
    """
    flow = natural_convection(1e4, cells=12, cfl=0.1)
    for _ in range(600):
        flow.step()
    nu = mean_nusselt(flow)
    assert np.isfinite(flow.u).all() and np.isfinite(flow.theta).all()
    assert np.abs(flow.u).max() < 5.0e2
    assert np.abs(flow.divergence()).max() < 1e-9
    assert 1.0 < nu < 2.0 * DE_VAHL_DAVIS[1e4]["nu"], nu

def test_stable_time_step_bounds_energy_diffusion():
    """The step must respect *every* diffusive bound, not just the viscous one.

    Regression for the m6444 defect: ``stable_time_step`` used ``0.25 h^2/nu``
    only.  With ``nu = Pr = 0.71`` and the energy diffusivity ``alpha = 1`` that
    admits ``dt = 0.352 h^2``, i.e. 1.41x the 2-D explicit bound ``h^2/(4
    alpha)``, so the thermal mode grows whenever the advective limit is
    inactive (slow low-Ra start-up).
    """
    flow = natural_convection(1e4, cells=8, cfl=0.5)
    assert flow.alpha == 1.0 and flow.nu < flow.alpha
    dt = flow.stable_time_step()
    assert dt <= 0.25 * flow.h ** 2 / flow.alpha, (dt, flow.h)
    # a larger thermal diffusivity must shrink the step (energy on) ...
    flow.alpha = 4.0
    assert flow.stable_time_step() < dt
    # ... but momentum-only flows keep the historical viscous-only cap
    momentum = sudden_expansion(area_ratio=4.0, re_d=100.0, cells_per_diameter=8)
    momentum.nu = 4.0
    h = momentum.h
    momentum.cfl = 1.0e6                       # make the advective limit inactive
    assert momentum.stable_time_step() == pytest.approx(0.25 * h ** 2 / 4.0)


def test_natural_convection_square_cavity_ra1e4_verified():
    """Ra = 1e4, Pr = 0.71 against de Vahl Davis (1983), default cfl.

    Previously WITHHELD: the N = 64 run diverged at step 84 000.  The cause was
    the missing thermal-diffusion step bound (see the test above); with the fix
    the case converges to a steady state on every grid and the N = 32 Nusselt
    number is within 3 % of the reference.
    """
    ref = DE_VAHL_DAVIS[1e4]
    flow = natural_convection(1e4, cells=16)
    ledger = flow.run_steady(tolerance=1e-7, max_steps=200_000)
    assert ledger.converged
    assert np.abs(flow.divergence()).max() < 1e-9
    nu = mean_nusselt(flow)
    assert 1.0 < nu < 2.0 * ref["nu"]
    assert abs(nu / ref["nu"] - 1.0) < 0.06, nu          # 16^2: +4.0 %
    psi_max = float(np.abs(flow.streamfunction()).max())
    assert abs(psi_max / ref["psi_max"] - 1.0) < 0.05, psi_max


def test_natural_convection_ra1e4_artifact_meets_criterion():
    """The committed N = 32 artefact must satisfy the 3 % Nusselt criterion."""
    root = ROOT / "fvm_natural_convection"
    summary_path = root.parent / "fvm_incompressible" / "summary.json"
    raw = root / "raw_natural_convection_Ra1e+04_N32.json"
    if not raw.exists() or not summary_path.exists():
        pytest.skip("natural-convection artefacts are not present in this checkout")
    payload = json.loads(raw.read_text())
    metrics = payload["metrics"]
    assert payload["ledger"]["converged"]
    assert metrics["nusselt_relative_error"] < 0.03, metrics["nusselt_hot"]
    assert metrics["psi_max_relative_error"] < 0.01
    assert metrics["nusselt_energy_imbalance"] < 1e-6


def test_streamfunction_is_consistent_with_vorticity():
    flow = natural_convection(1e4, cells=16, cfl=0.3)
    for _ in range(300):
        flow.step()
    psi = flow.streamfunction()
    assert np.isfinite(psi).all()
    # psi vanishes on the walls by construction (Dirichlet data at j = -1, n)
    assert np.abs(psi).max() > 0.0


def test_summary_is_consistent_with_raw_artifacts():
    """Every criterion number in summary.json must be re-derivable from a raw file."""
    root = ROOT
    summary_path = root / "fvm_incompressible" / "summary.json"
    if not summary_path.exists():
        pytest.skip("benchmark artifacts are not present in this checkout")
    summary = json.loads(summary_path.read_text())
    raws = list(root.glob("fvm_*/raw_*.json"))
    assert raws, "summary present but no raw evidence files"
    payloads = [json.loads(p.read_text()) for p in raws]
    for payload in payloads:
        assert payload["ledger"]["max_divergence"] < 1e-6
        assert payload["parameters"]
        assert payload["metrics"]
    nusselt = [p["metrics"]["nusselt_hot"] for p in payloads
               if p["case"] == "natural_convection"]
    if nusselt:
        stated = [c["value"] for key, c in summary["criteria"].items()
                  if key.startswith("convection_nu_")]
        for value in stated:
            assert any(abs(value - candidate) < 1e-9 for candidate in nusselt)

# ---------------------------------------------------------------------------
# m6383: mirror symmetry of the expansion, wall-shear convention, outflow guard
# ---------------------------------------------------------------------------


def test_mirror_symmetry_is_an_operator_invariant():
    """The discrete operators of a symmetric case preserve mirror symmetry.

    ``docs/benchmarks/incompressible-verification.md`` §2.2 attributed an
    ``L_r/D`` of 0.20 (lower) against 11.86 (upper) to a *solver* defect in the
    step-corner/wake treatment. This test pins the opposite: on a mirror
    symmetric field the momentum tendencies keep ``u`` symmetric and ``v``
    antisymmetric to round-off, and the pressure solve maps a symmetric
    right-hand side to a symmetric pressure.
    """
    flow = sudden_expansion(4.0, 100.0, 4)          # 60 x 16, cheap
    rng = np.random.default_rng(0)
    u = rng.standard_normal(flow.u.shape)
    u = 0.5 * (u + u[:, ::-1])
    v = rng.standard_normal(flow.v.shape)
    v = 0.5 * (v - v[:, ::-1])
    flow.u, flow.v = u, v
    tu, tv = flow.tendencies()
    assert np.abs(tu - tu[:, ::-1]).max() < 1e-12 * np.abs(tu).max()
    assert np.abs(tv + tv[:, ::-1]).max() < 1e-12 * np.abs(tv).max()
    div = ((u[1:, :] - u[:-1, :]) / flow.h + (v[:, 1:] - v[:, :-1]) / flow.h)
    p = flow.solve_pressure(div / flow.h)
    assert np.abs(p - p[:, ::-1]).max() < 1e-10 * max(np.abs(p).max(), 1.0)


def test_wall_shear_convention_is_shared_by_both_walls():
    """One orientation for both walls: ``tau < 0`` must mean *reversed* twice.

    The pre-m6383 accessors returned ``-2 nu u/h`` below and ``+2 nu u/h``
    above, i.e. ``tau_lower == -tau_upper`` on a mirror-symmetric field, so the
    same reversal test flipped its meaning from one wall to the other.
    """
    flow = sudden_expansion(4.0, 100.0, 4)
    rng = np.random.default_rng(1)
    u = rng.standard_normal(flow.u.shape)
    flow.u = 0.5 * (u + u[:, ::-1])
    tau_lo, tau_hi = flow.wall_shear_lower(), flow.wall_shear_upper()
    assert np.allclose(tau_lo, tau_hi, rtol=0.0, atol=1e-12 * np.abs(tau_lo).max())
    # attached forward flow is positive on both walls ...
    forward = flow.u.copy()
    forward[:, :] = 0.5
    flow.u = forward
    assert np.all(flow.wall_shear_lower() > 0.0)
    assert np.all(flow.wall_shear_upper() > 0.0)
    # ... and a recirculation is negative on both walls
    flow.u = -forward
    assert np.all(flow.wall_shear_lower() < 0.0)
    assert np.all(flow.wall_shear_upper() < 0.0)


def test_reattachment_length_rejects_a_crossing_in_the_outflow_zone():
    """A crossing in the last ``min_attached`` length units is *not resolved*."""
    x = np.linspace(0.0, 12.0, 121)
    tau = np.where(x < 11.8, -1.0, +1.0)           # bubble open up to the outflow
    assert math.isnan(reattachment_length(x, tau, start=0.02))
    assert reattachment_length(x, tau, start=0.02, min_attached=0.0) == pytest.approx(
        11.75, abs=0.06)
    # a genuinely closed bubble with attached wall downstream is still reported
    tau2 = np.where(x < 5.0, -1.0, +1.0)
    assert reattachment_length(x, tau2, start=0.02) == pytest.approx(5.0, abs=0.05)


def test_committed_expansion_fields_are_mirror_symmetric():
    """Regression on the committed snapshots behind the §2.2 asymmetry claim."""
    root = ROOT / "src" / "tensorfvm" / "verification" / "artifacts" / "incompressible"
    path = root / "fvm_sudden_expansion" / "fields_sudden_expansion_D10.npz"
    if not path.exists():
        pytest.skip("expansion field snapshots are not present in this checkout")
    fields = np.load(path)
    u, v, p = fields["u"], fields["v"], fields["p"]
    assert np.abs(u - u[:, ::-1]).max() / np.abs(u).max() < 1e-10
    assert np.abs(v + v[:, ::-1]).max() / np.abs(v).max() < 1e-10
    assert np.abs(p - p[:, ::-1]).max() / np.abs(p).max() < 1e-10
    flow = sudden_expansion(4.0, 100.0, 10)
    flow.u, flow.v, flow.p = u, v, p
    x = flow.x_face_coordinates() - 3.0
    tau_lo, tau_hi = flow.wall_shear_lower(), flow.wall_shear_upper()
    assert np.allclose(tau_lo, tau_hi, rtol=0.0, atol=1e-10)
    # the bubble is still open where the 12 D domain ends -> no reattachment
    assert math.isnan(reattachment_length(x, tau_lo, start=0.02))
    assert math.isnan(reattachment_length(x, tau_hi, start=0.02))
    # ... whereas the fabricated crossing appears as soon as the guard is off
    assert reattachment_length(x, tau_lo, start=0.02, min_attached=0.0) > 11.0
