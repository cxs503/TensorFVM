"""Tests for the Shu-Osher shock--entropy-wave interaction case (CASE-SHU-OSHER).

The case is pinned by three kinds of evidence, none of which needs digitised
literature data:

* the canonical parameters of Shu & Osher (1988) / Jiang & Shu (1996),
* the Rankine-Hugoniot post-shock state and shock trajectory (analytic),
* the *undisturbed* upstream profile ``1 + 0.2 sin(5x)`` ahead of the shock,
  which the scheme must leave untouched at ``t = 1.8``.

The grid-convergence verdict lives in ``docs/CASE-SHU-OSHER/shu_osher.json``
(produced by ``python -m tensorfvm.compressible.shu_osher_benchmark``); the
last test checks that artifact instead of re-running a 5-minute sweep.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
import torch

from tensorfvm.compressible.cases import GAMMA_AIR, ShuOsherConfig, shu_osher

REPO_ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = REPO_ROOT / "docs" / "CASE-SHU-OSHER" / "shu_osher.json"


def test_canonical_setup_and_rh_state():
    """Defaults must be the canonical Shu-Osher/Jiang-Shu configuration."""
    cfg = ShuOsherConfig()
    assert cfg.x_min == -5.0 and cfg.lx == 10.0
    assert cfg.x_max == 5.0
    assert cfg.shock_pos == -4.0
    assert cfg.entropy_wavenumber == 5.0
    assert cfg.entropy_amp == 0.2
    assert cfg.t_end == 1.8
    assert cfg.mach == 3.0

    post = cfg.post_shock
    # Rankine-Hugoniot for M=3, gamma=1.4 (computed, not hand-copied)
    ratio = (GAMMA_AIR + 1.0) * 9.0 / ((GAMMA_AIR - 1.0) * 9.0 + 2.0)
    p_ratio = 2.0 * GAMMA_AIR * 9.0 / (GAMMA_AIR + 1.0) - (GAMMA_AIR - 1.0) / (GAMMA_AIR + 1.0)
    assert post["rho"] == pytest.approx(ratio)
    assert post["p"] == pytest.approx(p_ratio)
    assert post["u"] == pytest.approx(2.0 / (GAMMA_AIR + 1.0) * (3.0 - 1.0 / 3.0) * math.sqrt(GAMMA_AIR))
    # shock speed D = M c1, and the shock must still be inside the domain at t=1.8
    assert post["shock_speed"] == pytest.approx(3.0 * math.sqrt(GAMMA_AIR))
    assert cfg.shock_position(1.8) == pytest.approx(-4.0 + post["shock_speed"] * 1.8)
    assert -4.0 < cfg.shock_position(1.8) < 5.0


def test_initial_profiles_and_grid_extent():
    """Cell centres span [-5, 5]; left = post-shock state, right = 1 + 0.2 sin(5x)."""
    cfg = ShuOsherConfig(nx=200)
    solver, meta = shu_osher(cfg)
    xc = meta["xc"]
    assert float(xc[0]) > -5.0 and float(xc[-1]) < 5.0
    assert float(xc[0]) == pytest.approx(-5.0 + 0.5 * cfg.dx)
    assert float(xc[-1]) == pytest.approx(5.0 - 0.5 * cfg.dx)

    prim = solver.primitive()
    left = xc < cfg.shock_pos
    right = ~left
    post = cfg.post_shock

    assert torch.allclose(prim["rho"][left, 0], torch.full((int(left.sum()),), post["rho"], dtype=torch.float64))
    assert torch.allclose(prim["u"][left, 0], torch.full((int(left.sum()),), post["u"], dtype=torch.float64))
    assert torch.allclose(prim["p"][left, 0], torch.full((int(left.sum()),), post["p"], dtype=torch.float64))

    expected = 1.0 + 0.2 * torch.sin(5.0 * xc[right])
    assert torch.allclose(prim["rho"][right, 0], expected, atol=1e-12)
    assert torch.allclose(prim["u"][right, 0], torch.zeros(int(right.sum()), dtype=torch.float64))
    assert torch.allclose(prim["p"][right, 0], torch.ones(int(right.sum()), dtype=torch.float64))


def test_shock_track_and_undisturbed_region():
    """At t=1.8 the shock sits on the RH track and the gas ahead is untouched."""
    cfg = ShuOsherConfig(nx=100)
    solver, meta = shu_osher(cfg)
    solver.run(cfg.t_end)
    xc = meta["xc"]
    rho = solver.primitive()["rho"][:, 0]

    # steepest density gradient == shock front
    i = int(torch.argmax(torch.diff(rho).abs()).item())
    x_front = float(xc[i]) + 0.5 * float(xc[i + 1] - xc[i])
    x_pred = cfg.shock_position(cfg.t_end)
    assert abs(x_front - x_pred) < 0.02 * cfg.lx, (
        f"shock front {x_front:.4f} strays from the RH track {x_pred:.4f}"
    )

    # ahead of the shock the analytic entropy wave is undisturbed
    ahead = xc > x_front + 3.0 * cfg.dx
    exact = 1.0 + 0.2 * torch.sin(5.0 * xc)
    assert int(ahead.sum()) > 10
    assert float((rho[ahead] - exact[ahead]).abs().max()) < 1e-3 * cfg.entropy_amp

    # behind the shock the entropy wave is compressed into an oscillation train
    # that is *not* the uniform post-shock plateau (3.857)
    behind = xc < x_front - 1.0
    amp_behind = float((rho[behind] - cfg.post_shock["rho"]).abs().max())
    assert amp_behind > 0.1, f"compressed entropy wave missing (amp={amp_behind:.3f})"
    assert float(rho.min()) > 0.0 and float(solver.primitive()["p"].min()) > 0.0
    assert solver.positivity_fixes == 0


def test_budget_accounting_with_open_boundaries():
    """Mass/energy budget (interior + boundary ledger) must close to round-off."""
    cfg = ShuOsherConfig(nx=100, t_end=1.0)
    solver, meta = shu_osher(cfg)
    initial = solver.integrals()
    solver.run(cfg.t_end)
    drift = solver.budget_drift(initial)
    assert abs(drift["mass"]) < 1e-10
    assert abs(drift["energy"]) < 1e-10


@pytest.mark.skipif(not ARTIFACT.exists(), reason="benchmark artifact not generated")
def test_convergence_artifact_meets_acceptance():
    """The committed benchmark artifact must pass the 5% acceptance criterion."""
    data = json.loads(ARTIFACT.read_text())
    assert data["reference_run"]["nx"] >= 3200
    verdicts = data["verdicts"]
    for key, ok in verdicts.items():
        assert ok, f"verdict {key} failed in the committed artifact"

    rows = data["results"]
    res = {r["nx"]: r["rho_error"] for r in rows}
    # the finest swept grid is the one that carries the acceptance claim
    finest = max(res)
    assert res[finest]["L1_rel"] < 0.05
    # L1 errors must decrease monotonically with refinement
    errs = [res[nx]["L1_abs"] for nx in sorted(res)]
    assert all(a > b for a, b in zip(errs[:-1], errs[1:]))
    # one-dimensional problem: rho range of the reference must be physical
    ref = data["reference_run"]
    assert 0.2 < ref["rho_min"] < 1.5
    assert 3.0 < ref["rho_max"] < 6.0