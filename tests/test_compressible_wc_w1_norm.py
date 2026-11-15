"""Regression gate for the transport-norm (W1 / negative-norm s=-1) certification.

``R4-REFERENCE-UNCERTAINTY-CLOSURE.md`` could only claim ``upper_bound_only`` for
``<1 %``: the pointwise L1 metric is dominated by shock smearing and shrinks like
``h**0.83``, so a provable ``<1 %`` bound needs ``nx_ref ~ 1.33e5``.  ``wc_w1_norm.py``
re-measures the *same stored ladder* in the transport norm (equivalently the W1
distance between normalised cumulative mass distributions), where the shock-band
sawtooth is integrated away.

This gate pins the things that make the resulting number trustworthy, and it
pins the *honest failures* too:

1. the transport norm is always below the pointwise L1 (``int|D| <= L int|e|``);
2. it is a convergent functional (monotone decreasing with refinement);
3. its response to *injected* errors matches closed-form values (rigid shift,
   global amplitude error, displaced mass) -- so ``0.01 %`` cannot be a metric
   that measures nothing;
4. a known-different solution (the digitised published curve) lands well above
   our own floor (negative control);
5. the predeclared verdict is what the report quotes: P4 (<1 % in transport norm,
   provable) passes, while **P1 (order gain) and P7 (interpolation-direction
   robustness) fail** -- they must keep failing unless the numbers change;
6. the pointwise L1 stays above 0.5 %, i.e. the weaker norm is *not* a free
   lunch.
"""
from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
XVAL = REPO / "docs" / "woodward-colella-double-blast" / "xval"
HERE = REPO / "docs" / "XVAL-FVM-M18737-ROUND2"
for p in (str(XVAL), str(HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)

import wc_w1_norm as w1  # noqa: E402


@pytest.fixture(scope="module")
def budget():
    return w1.build(verbose=False)


@pytest.fixture(scope="module")
def profiles():
    prof = {}
    for fam, ladder in (("hllc", w1.MUSCL_LADDER), ("weno5", w1.WENO_LADDER)):
        for nx in ladder:
            p = w1.load(nx, fam)
            if p is not None:
                prof[(fam, nx)] = p
    return prof


# --------------------------------------------------------------------------- #
# 1. the metric must be a weaker norm, always
# --------------------------------------------------------------------------- #
def test_transport_norm_never_exceeds_pointwise_l1(profiles):
    pairs = []
    ref = profiles[("weno5", w1.REF_FINEST)]
    for nx in w1.MUSCL_LADDER:
        if ("hllc", nx) in profiles:
            pairs.append((ref, profiles[("hllc", nx)]))
    for lo, hi in zip(w1.WENO_LADDER, w1.WENO_LADDER[1:]):
        pairs.append((profiles[("weno5", hi)], profiles[("weno5", lo)]))
    assert pairs
    for fine, coarse in pairs:
        for var in w1.VARS:
            l1 = w1.d_l1_percent(fine, coarse, var)
            d1 = w1.d_w1_percent(fine, coarse, var)
            assert math.isfinite(d1) and d1 > 0.0
            assert d1 <= l1 * (1.0 + 1e-9), (var, fine["nx"], coarse["nx"], d1, l1)


# --------------------------------------------------------------------------- #
# 2. it must converge
# --------------------------------------------------------------------------- #
def test_transport_norm_decreases_monotonically(budget):
    seq = [r["d_w1"] for r in budget["graded_family"]["rows"]]
    assert len(seq) >= 3
    assert all(b < a for a, b in zip(seq, seq[1:])), seq
    # and the observed order is ~1, not the hoped-for ~2: pinned honestly
    assert 0.8 < budget["graded_family"]["p_w1"] < 1.3
    assert 0.7 < budget["reference_family"]["p_w1"] < 1.3


def test_reference_family_self_convergence(budget):
    d = [p["d_w1"] for p in budget["reference_family"]["pairs"]]
    assert all(b < a for a, b in zip(d, d[1:])), d
    assert budget["reference_family"]["pairs"][-1]["d_w1"] < 0.01


# --------------------------------------------------------------------------- #
# 3. injected errors must be read back at their closed-form value
# --------------------------------------------------------------------------- #
def test_probes_match_closed_form(budget):
    probes = budget["probes_post_hoc"]
    for key in ("shift_1cell_nx3200", "bump_shift_1cell_exact", "mass_1pct_misplaced"):
        assert probes[key]["detects"], (key, probes[key])
        assert 0.95 <= probes[key]["ratio"] <= 1.05, (key, probes[key])
    # a global amplitude error is *not* cosmetic under the common denominator
    assert 0.9 <= probes["amplitude_1pct"]["ratio"] <= 1.1, probes["amplitude_1pct"]
    assert probes["amplitude_1pct"]["measured"] > 0.1


def test_negative_control_published_curve_flagged(budget):
    ctl = budget["probes_post_hoc"]["published_curve_control"]
    assert ctl["detects"], ctl
    assert ctl["ratio_to_floor"] >= 3.0


# --------------------------------------------------------------------------- #
# 4. the predeclared verdict, including its failures
# --------------------------------------------------------------------------- #
def test_predeclared_verdict_is_reported_honestly(budget):
    v = budget["verdict"]
    # the certification claim itself
    assert v["P4"]["pass"] is True
    assert v["P4"]["honest_bound"] < 1.0
    assert v["P2"]["pass"] and v["P3"]["pass"]
    # the mathematical invariant
    assert v["P6"]["pass"] is True and v["P6"]["n_pairs"] >= 20
    # honest FAILs: the order-gain hypothesis and the direction robustness
    assert v["P1"]["pass"] is False, "P1 (order gain) must not silently pass"
    assert v["P7"]["pass"] is False, "P7 (direction sensitivity) must not silently pass"
    assert v["P7"]["n_violating"] == 1, v["P7"]
    assert budget["all_pass"] is False


def test_weaker_norm_is_not_a_free_lunch(budget):
    v = budget["verdict"]
    assert v["P5"]["pass"] is True
    rows = {r["nx"]: r for r in budget["graded_family"]["rows"]}
    assert rows[max(rows)]["d_l1"] > 0.5
    # the constant factor between the two norms is ~100x and stable: it is a
    # property of the solution, so quoting d_W1 alone would be misleading
    ratios = [r["d_w1"] / r["d_l1"] for r in budget["graded_family"]["rows"]]
    assert all(0.5e-2 < r < 2e-2 for r in ratios), ratios


def test_mass_conservation_of_every_rung(budget):
    for key, row in budget["mass"].items():
        assert abs(row["drift_percent"]) < 0.1, (key, row)


def test_golden_values(budget):
    """Pin the numbers the report quotes (per cent of window span)."""
    ref = budget["reference_family"]
    assert ref["pairs"][-1]["d_w1"] == pytest.approx(0.005753, rel=2e-3)
    assert ref["p_w1"] == pytest.approx(0.98739, rel=2e-3)
    assert ref["eps_ref_w1"] == pytest.approx(0.0058545, rel=2e-3)
    graded = {r["nx"]: r for r in budget["graded_family"]["rows"]}
    assert graded[6400]["d_w1"] == pytest.approx(0.0097812, rel=2e-3)
    assert graded[6400]["d_l1"] == pytest.approx(0.90039, rel=2e-3)
    assert budget["verdict"]["P4"]["honest_bound"] == pytest.approx(0.0156357, rel=2e-3)