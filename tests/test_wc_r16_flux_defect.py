"""Regression gate for round-16: the pre-registered E2 (h^1 shrinkage of
max|D| at the 25600 rung) was **falsified** -- this gate pins that honest
negative result and the invariants that make it trustworthy.

Frozen in ``PREDECLARATION-r16.md`` (E2 band reused verbatim from
``PREDECLARATION-r7.md`` / ``PREDECLARATION-r10.md``):

    p_D = log2( max|D|(6400->12800) / max|D|(12800->25600) )  in [1.85, 2.15]

Measured on the same-instrument CFL 0.8 reference ladder: p_D = 0.6116,
i.e. the reference family's self-difference shrinks *slower* than h^1 and the
ratio trends down (1.852 -> 1.565 -> 1.528).  The defect stays at the
collision (x_peak ~ 0.7648) and is not a conservation defect
(rel_net_mass ~ 2e-6).  Per the predeclared negative branch the verdict is
recorded as-is; the band is NOT re-thresholded.
"""
from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
HERE = REPO / "docs" / "XVAL-FVM-M18737-ROUND2"
for p in (str(HERE),):
    if p not in sys.path:
        sys.path.insert(0, p)

import wc_w1_r16_flux_defect as r16  # noqa: E402


@pytest.fixture(scope="module")
def out():
    return r16.build(verbose=False)


def test_e0_artifact_gate(out):
    e0 = out["e0_artifact_gate"]
    assert e0["pass"] is True
    assert e0["nx"] == 25600 and e0["cfl"] == 0.8 and e0["status"] == "ok"
    assert e0["profile_length"] == 25600


def test_e2_falsified_is_pinned(out):
    """The frozen E2 band is violated; the verdict must stay FALSIFIED."""
    e2 = out["criteria"]["E2"]
    m_prev = e2["max_abs_D_6400_12800"]
    m_new = e2["max_abs_D_12800_25600"]
    assert m_prev > m_new > 0
    p_D = math.log2(m_prev / m_new)
    assert abs(p_D - e2["p_D"]) < 1e-12
    lo, hi = e2["band"]
    assert not (lo <= p_D <= hi), "if E2 ever passes, this gate must be re-derived"
    assert e2["verdict"] == "FALSIFIED"
    assert out["verdict"] == "FALSIFIED"


def test_e2b_amplitude_ratio(out):
    e2b = out["criteria"]["E2b"]
    assert e2b["ratio"] > 0.5, "new pair larger than the h^1 prediction"
    assert e2b["verdict"] == "FAIL"


def test_e2c_localization_at_collision(out):
    e2c = out["criteria"]["E2c"]
    assert 0.70 <= e2c["x_peak"] <= 0.85
    assert e2c["verdict"] == "PASS"


def test_e2d_not_a_conservation_defect(out):
    e2d = out["criteria"]["E2d"]
    assert e2d["rel_net_mass"] < 1e-4
    assert e2d["verdict"] == "PASS"


def test_e2e_monotone(out):
    seq = out["criteria"]["E2e"]["sequence"]
    assert all(b < a for a, b in zip(seq, seq[1:]))
    assert out["criteria"]["E2e"]["verdict"] == "PASS"


def test_cfl04_sequence_matches_r7(out):
    """The CFL 0.4 reference sequence must reproduce R7's defects bit-for-bit
    (same implementation, same stored profiles)."""
    vals = [p["max_abs_D"] for p in out["crosscheck_cfl04_sequence"]]
    ref = [0.002086920898378841, 0.0011269515470658312, 0.0007197911559912067]
    assert len(vals) == 3
    for a, b in zip(vals, ref):
        assert abs(a - b) / b < 1e-9


def test_graded_sequence_matches_r5(out):
    """The R5 'next hand' numbers are the GRADED family (hllc vs weno5@12800)
    -- pin them so the category distinction cannot be lost again."""
    vals = [g["max_abs_D"] for g in out["crosscheck_graded_sequence"]]
    ref = [0.0045086517053647005, 0.002351750115192417, 0.0010784520482430362]
    assert len(vals) == 3
    for a, b in zip(vals, ref):
        assert abs(a - b) / b < 1e-9


def test_cfl_mismatch_is_not_the_cause(out):
    """CFL 0.4 vs 0.8 at 6400->12800 differ by <1 %; the mixed-instrument
    probe (12800_cfl0.4 -> 25600_cfl0.8) is within 3 % of the same-instrument
    value, so the falsification is not a CFL-caliber artefact."""
    cfl04_6400_12800 = out["crosscheck_cfl04_sequence"][-1]["max_abs_D"]
    cfl08_6400_12800 = out["pairs_cfl08"][0]["max_abs_D"]
    assert abs(cfl04_6400_12800 - cfl08_6400_12800) / cfl04_6400_12800 < 0.01
    mixed = out["crosscheck_mixed_pair"]["max_abs_D"]
    same = out["pairs_cfl08"][1]["max_abs_D"]
    assert abs(mixed - same) / same < 0.03


def test_w1_below_l1_on_new_pairs(out):
    """Cauchy-style bound: d_W1 <= d_L1 on every new pair (implementation bug
    if violated)."""
    for p in out["pairs_cfl08"]:
        assert p["d_w1_percent"] <= p["d_l1_percent"] * (1 + 1e-9)


def test_reference_family_ratio_trends_down(out):
    """The falsification is a trend, not a single-pair fluke: the reference
    family's max|D| ratio decreases rung over rung (1.852 -> 1.565 -> ~1.53)."""
    r04 = [p["max_abs_D"] for p in out["crosscheck_cfl04_sequence"]]
    m_new = out["pairs_cfl08"][1]["max_abs_D"]
    ratios = [r04[0] / r04[1], r04[1] / r04[2],
              out["pairs_cfl08"][0]["max_abs_D"] / m_new]
    assert all(b < a for a, b in zip(ratios, ratios[1:]))
