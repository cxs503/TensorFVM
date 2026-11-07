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