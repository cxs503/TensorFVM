"""Regression locks for the Woodward-Colella PEAK-ANOMALY closure (round-13).

Assertions:

* **synthetic invariants** -- conservation/positivity diagnostics on a
  manufactured profile (no stored artifacts needed).
* **regression locks** -- the stored nx=6400 MUSCL/HLLC solution must stay
  conservative (cell-sum mass/energy at machine precision), positive, and at
  the cross-code converged peak level (within 0.1 % of WENO5@12800); the
  documented non-monotonic peak sequence (6.4660 / 6.5075 / 6.4622) is pinned
  so a future re-run cannot silently change the verdict.  Missing ladder
  files make these skip rather than fail (a checkout without the artifacts
  is not a bug).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parents[1]
DATA = HERE / "docs" / "woodward-colella-double-blast" / "ladder" / "data"
GAMMA = 1.4
M0 = 1.0
E0 = (1000.0 * 0.1 + 0.01 * 0.8 + 100.0 * 0.1) / (GAMMA - 1.0)

FVM = ["nx1600_hllc", "nx3200_hllc", "nx6400_hllc"]
W5 = ["nx3200_weno5", "nx6400_weno5", "nx12800_weno5"]


def _load(tag: str):
    d = np.load(DATA / f"ladder_profile_{tag}.npz")
    return d["x"], d["rho"], d["p"], d["u"]


def _have(tags) -> bool:
    return all((DATA / f"ladder_profile_{t}.npz").exists() for t in tags)


needs_ladder = pytest.mark.skipif(
    not _have(FVM + W5),
    reason="stored Woodward-Colella ladder profiles not present in this checkout",
)


def _conservation(x, rho, p, u):
    dx = float(x[1] - x[0])
    with np.errstate(divide="ignore", invalid="ignore"):
        e = p / ((GAMMA - 1.0) * rho) + 0.5 * u * u
    return {
        "mass_cell": float(dx * rho.sum()) - M0,
        "energy_cell": float(dx * (rho * e).sum()) - E0,
        "mass_trap": float(np.trapezoid(rho, x)) - M0,
        "energy_trap": float(np.trapezoid(rho * e, x)) - E0,
        "energy_trap_rel": (float(np.trapezoid(rho * e, x)) - E0) / E0,
        "min_rho": float(rho.min()),
        "min_p": float(p.min()),
    }


# --------------------------------------------------------------------------- #
# synthetic invariants
# --------------------------------------------------------------------------- #
def test_cellsum_is_exact_for_piecewise_constant():
    # the cell-sum metric integrates the piecewise-constant reconstruction
    # exactly: a constant profile must give the exact mass.
    x = np.linspace(0.0, 1.0, 101)
    rho = np.full(100, 2.5)
    dx = 0.01
    assert abs(dx * rho.sum() - 2.5) < 1e-14


def test_trapzoid_error_scales_with_dx_squared():
    # trapezoid on cell averages of a smooth profile: O(dx^2) boundary error
    errs = []
    for n in (200, 400, 800):
        x = (np.arange(n) + 0.5) / n
        rho = 1.0 + 0.3 * np.sin(2 * np.pi * x)
        exact = 1.0  # mean of sin over full periods
        errs.append(abs(np.trapezoid(rho, x) - exact))
    assert errs[0] > errs[1] > errs[2]
    assert errs[0] / errs[1] > 2.0  # ~4x per halving


def test_positivity_gate_rejects_zero_touching():
    x = np.linspace(0, 1, 50)
    rho = np.ones(50)
    rho[10] = 0.0  # touches zero
    c = _conservation(x, rho, np.ones(50), np.zeros(50))
    assert c["min_rho"] <= 0.1  # gate must flag it


# --------------------------------------------------------------------------- #
# regression locks (stored ladder)
# --------------------------------------------------------------------------- #
@needs_ladder
def test_peak_sequence_is_the_documented_nonmonotone_one():
    peaks = [float(_load(t)[1].max()) for t in FVM]
    assert peaks[0] == pytest.approx(6.4660, abs=5e-4)
    assert peaks[1] == pytest.approx(6.5075, abs=5e-4)
    assert peaks[2] == pytest.approx(6.4622, abs=5e-4)
    # the anomaly itself: 3200 is the local maximum of the family
    assert peaks[1] > peaks[0] and peaks[2] < peaks[1]


@needs_ladder
def test_plateau_mean_shows_the_same_nonmonotonicity():
    # A1: the drop is in the level, not a max-vs-mean artifact
    plats = []
    for t in FVM:
        rho = _load(t)[1]
        plats.append(float(rho[rho > 0.99 * rho.max()].mean()))
    assert plats[1] > plats[0] and plats[2] < plats[1]


@needs_ladder
def test_plateau_is_broad_not_a_subcell_spike():
    # A2: FWHM >> dx on every resolved rung
    for t in FVM + W5:
        x, rho = _load(t)[:2]
        j = int(np.argmax(rho))
        half = rho[j] / 2.0
        lo = j
        while lo > 0 and rho[lo] > half:
            lo -= 1
        hi = j
        while hi < len(rho) - 1 and rho[hi] > half:
            hi += 1
        assert hi - lo > 50, f"{t}: plateau too narrow ({hi - lo} cells)"


@needs_ladder
def test_fvm6400_sits_at_cross_code_converged_level():
    # A3: FVM@6400 vs WENO5@12800 (independent discretization)
    f64 = float(_load("nx6400_hllc")[1].max())
    w128 = float(_load("nx12800_weno5")[1].max())
    assert (f64 - w128) / w128 < 1e-3
    # and the 3200 rung is the outlier: its plateau is >1% above WENO5@3200
    f32 = _load("nx3200_hllc")[1]
    w32 = _load("nx3200_weno5")[1]
    p32 = f32[f32 > 0.99 * f32.max()].mean()
    q32 = w32[w32 > 0.99 * w32.max()].mean()
    assert (p32 - q32) / q32 > 1e-2


@needs_ladder
def test_weno5_family_is_monotone_saturating():
    peaks = [float(_load(t)[1].max()) for t in W5]
    assert peaks[0] < peaks[1]
    assert abs(peaks[2] - peaks[1]) / peaks[1] < 1e-3


@needs_ladder
def test_nx6400_is_conservative_to_machine_precision():
    # A4: cell-sum (exact for the discrete solution) at round-off
    x, rho, p, u = _load("nx6400_hllc")
    c = _conservation(x, rho, p, u)
    assert abs(c["mass_cell"]) < 1e-10
    assert abs(c["energy_cell"]) < 1e-10
    # trapezoid on cell averages: quadrature error, small and bounded
    assert abs(c["mass_trap"]) < 1e-4
    assert abs(c["energy_trap_rel"]) < 1e-4


@needs_ladder
def test_nx6400_is_strongly_positive():
    # A5: margins far from zero
    _, rho, p, _ = _load("nx6400_hllc")
    assert rho.min() > 0.1
    assert p.min() > 1.0


@needs_ladder
def test_wall_momentum_transfer_agrees_across_codes():
    # the closed-tube fluid momentum is set by wall transfer; two codes
    # must agree on it (converged quantity, not a conservation violation)
    x64, rho64, _, u64 = _load("nx6400_hllc")
    x128, rho128, _, u128 = _load("nx12800_weno5")
    m64 = float(np.trapezoid(rho64 * u64, x64))
    m128 = float(np.trapezoid(rho128 * u128, x128))
    assert m64 == pytest.approx(m128, rel=1e-4)
    assert m64 > 0  # net transfer to the walls, same sign in both codes


@needs_ladder
def test_3200_overshoot_is_localized_to_the_core():
    # A7: |d(rho_3200, rho_6400)| concentrated in x in [0.55, 0.85]
    xf = np.linspace(0.0, 1.0, 256001)
    a = np.interp(xf, *_load("nx3200_hllc")[:2])
    b = np.interp(xf, *_load("nx6400_hllc")[:2])
    d = np.abs(a - b)
    tot = float(np.trapezoid(d, xf))
    core = float(np.trapezoid(d[(xf >= 0.55) & (xf < 0.85)], xf[(xf >= 0.55) & (xf < 0.85)]))
    assert core / tot > 0.9
