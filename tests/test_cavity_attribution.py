"""Acceptance tests for :mod:`tensorfvm.cavity_attribution` and the audit it backs.

The tests are split in two groups:

* **synthetic** - exact invariants of the arithmetic (a second-order ladder must
  give ``p = 2``, an oscillatory ladder must be refused, a tabulated parabola
  must recover its own extremum), so a regression in the estimators shows up
  without depending on stored artefacts;
* **stored** - the verdicts the audit
  (``docs/CASE-FVM-CAVITY-RE1000/ATTRIBUTION_AUDIT.md``) claims, recomputed from
  ``docs/CASE-FVM-CAVITY-RE1000/runs/`` so the document cannot drift away from
  its data.

The stored group doubles as a *negative control*: ``test_verdict_flips_when_
residual_leaves_the_band`` perturbs the solution until the verdict changes, which
is what proves the classifier is a test and not a rubber stamp.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tensorfvm.cavity_attribution import (
    attribution_verdict,
    contraction_ratios,
    extremum_floor,
    finest_grid_disagreement,
    gci,
    is_asymptotic,
)
from tensorfvm.lid_cavity_reference import U_CENTRELINE, V_CENTRELINE

RUNS = Path("docs/CASE-FVM-CAVITY-RE1000/runs")


# --------------------------------------------------------------------------- #
# synthetic: the arithmetic itself
# --------------------------------------------------------------------------- #
def test_gci_recovers_the_order_of_a_second_order_ladder() -> None:
    """A sequence built as ``f + C h^2`` must return ``p = 2`` to round-off."""
    exact, C = 0.25, -0.4
    for r in (2.0, 1.5, 3.0):
        h = 1.0
        vals = []
        for _ in range(3):
            vals.append(exact + C * h**2)
            h /= r
        out = gci(vals[0], vals[1], vals[2], refinement=r)
        assert out["order"] == pytest.approx(2.0, abs=1e-9)
        # and the band must actually straddle the exact value
        lo, hi = out["interval"]
        assert lo <= exact <= hi


def test_gci_refuses_an_oscillatory_ladder() -> None:
    """Sign-changing increments mean there is no asymptotic order to report."""
    with pytest.raises(ValueError, match="oscillatory"):
        gci(-0.30, -0.38, -0.37)


def test_gci_refuses_a_flat_ladder() -> None:
    with pytest.raises(ValueError):
        gci(1.0, 1.0, 1.0)


def test_contraction_ratios_of_a_geometric_sequence_are_constant() -> None:
    rho = 0.25
    v = [1.0]
    for _ in range(4):
        v.append(v[-1] - 0.1 * rho ** (len(v) - 1))
    got = contraction_ratios(v)
    assert len(got) == 3
    assert all(g == pytest.approx(rho, rel=1e-9) for g in got)


def test_is_asymptotic_rejects_a_drifting_sequence() -> None:
    """Ratios differing by an order of magnitude must not pass as asymptotic."""
    # increments 0.38, 0.016, 0.00037 -> ratios 0.042, 0.023 (fine)
    ok = is_asymptotic([-0.7845, -0.4049, -0.3890, -0.38861])
    assert ok["informative"] is True
    assert ok["asymptotic"] is True
    # increments 0.38, 0.016, 0.005 -> ratios 0.042, 0.31 (drifts by ~7x)
    bad = is_asymptotic([-0.7845, -0.4049, -0.3890, -0.3840])
    assert bad["informative"] is True
    assert bad["asymptotic"] is False


def test_three_level_ladder_cannot_verify_the_asymptotic_range() -> None:
    """``informative`` must be False with one ratio: the test is then vacuous."""
    out = is_asymptotic([-0.3276, -0.3731, -0.3853])
    assert out["n_ratios"] == 1
    assert out["informative"] is False
    assert out["asymptotic"] is False


def test_extremum_floor_recovers_an_exactly_representable_extremum() -> None:
    """A table sampled from a parabola has a known continuous extremum."""
    x = [0.0, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0]
    y = [-((xi - 0.5) ** 2) + 0.25 for xi in x]  # min at x=0.5, value 0 -> use max
    out = extremum_floor(zip(x, y), mode="max")
    # max is 0.25 at x=0.5, a tabulated node: every interpolant must return it
    assert out["tabulated"] == pytest.approx(0.25)
    assert out["floor_upper"] == pytest.approx(0.0, abs=1e-12)


def test_extremum_floor_is_a_bracket_lower_le_upper() -> None:
    for tab, mode in ((U_CENTRELINE[1000.0], "min"), (V_CENTRELINE[1000.0], "max")):
        out = extremum_floor(tab, mode=mode)
        assert out["floor_lower"] <= out["floor_mid"] <= out["floor_upper"] + 1e-15


def test_extremum_floor_is_largest_where_the_table_is_coarsest() -> None:
    """The floor is a property of the table: a wide bracket must give a bigger floor.

    Re = 1000 Table I brackets its minimum over 0.18 L; Table II brackets its
    maximum over 0.13 L.  A hand-made, deliberately coarse table must come out
    worse than any real one.
    """
    real = extremum_floor(U_CENTRELINE[1000.0], mode="min")
    coarse = extremum_floor(
        [(0.0, 0.0), (0.05, -0.18), (0.35, -0.38), (0.75, -0.05), (1.0, 1.0)], mode="min"
    )
    assert coarse["bracket_gap"] > real["bracket_gap"]
    assert coarse["floor_upper"] > real["floor_upper"]


# --------------------------------------------------------------------------- #
# stored: what the audit claims
# --------------------------------------------------------------------------- #
def _uncertainty() -> dict:
    return json.loads((RUNS / "uncertainty.json").read_text())


def test_reference_floor_of_ghia_table_at_re1000() -> None:
    """The table alone pins u_min to <= 0.75 % and v_max to <= 0.09 %."""
    u = extremum_floor(U_CENTRELINE[1000.0], mode="min")
    v = extremum_floor(V_CENTRELINE[1000.0], mode="max")
    assert u["tabulated"] == pytest.approx(-0.38289)
    assert v["tabulated"] == pytest.approx(0.37095)
    assert u["floor_upper"] < 0.0075
    assert v["floor_upper"] < 0.0009


@pytest.mark.parametrize("quantity,ref", [("u_min", -0.38289), ("v_max", 0.37095)])
def test_ghia_lies_inside_the_solvers_own_gci_band(quantity: str, ref: float) -> None:
    """The audit's core finding: the residual is NOT significant.

    If this ever fails, the "consistent within its own discretisation
    uncertainty" statement in the report is no longer true and the attribution
    question re-opens.
    """
    q = _uncertainty()["quantities"][quantity]
    v = attribution_verdict(
        label=quantity,
        coarse=q["values"][0],
        medium=q["values"][1],
        fine=q["values"][2],
        reference=ref,
    )
    assert v["verdict"] == "within_own_uncertainty"
    assert v["reference_inside_gci_band"] is True
    assert 0.0 < v["reference_position_in_band"] < 1.0
    assert v["gci_assumption_tested"] is False  # only three levels stored


def test_verdict_flips_when_residual_leaves_the_band() -> None:
    """Negative control: the classifier must be able to say "not significant" false."""
    q = _uncertainty()["quantities"]["u_min"]
    # push the reference just outside the band -> attribution question re-opens
    lo, hi = gci(q["values"][0], q["values"][1], q["values"][2])["interval"]
    outside = hi * 1.05
    v = attribution_verdict(
        label="u_min",
        coarse=q["values"][0],
        medium=q["values"][1],
        fine=q["values"][2],
        reference=outside,
    )
    assert v["reference_inside_gci_band"] is False
    assert v["verdict"] == "unresolved"
    assert lo <= q["values"][2] <= hi  # sanity on the band itself


def test_verdict_can_attribute_when_the_reference_floor_is_large_enough() -> None:
    """The third verdict must be reachable, otherwise the classifier is a polarity."""
    q = _uncertainty()["quantities"]["u_min"]
    lo, hi = gci(q["values"][0], q["values"][1], q["values"][2])["interval"]
    outside = hi * 1.05
    v = attribution_verdict(
        label="u_min",
        coarse=q["values"][0],
        medium=q["values"][1],
        fine=q["values"][2],
        reference=outside,
        reference_floor=10.0,  # absurdly large floor: reference can explain anything
    )
    assert v["verdict"] == "reference_attributable"


def test_finest_grid_disagreement_exceeds_the_extrapolated_agreement() -> None:
    """0.84 % raw vs 0.26 % extrapolated - the report quoted the extrapolated one."""
    fvm_fine = _uncertainty()["quantities"]["u_min"]["values"][2]
    rep = json.loads((RUNS / "spectral" / "spectral_report.json").read_text())
    spectral_fine = rep["extrapolation"]["u_min"]["pairs"][-1][1]
    raw = finest_grid_disagreement(fvm_fine, spectral_fine)
    quoted = rep["verdict"]["u_min"]["spectral_vs_fvm_rel"]
    assert raw == pytest.approx(0.00837, abs=5e-4)
    assert raw > 3.0 * quoted


def test_spectral_extrapolation_assumed_a_contraction_ratio_the_data_refute() -> None:
    """The stored limit used rho = 1/3.21 = 0.31; the ladder's own rho is 0.023-0.042."""
    rep = json.loads((RUNS / "spectral" / "spectral_report.json").read_text())
    pairs = rep["extrapolation"]["u_min"]["pairs"]
    assert [p[0] for p in pairs] == [24, 32, 48, 64]
    observed = contraction_ratios([p[1] for p in pairs])
    assumed = 1.0 / rep["extrapolation"]["u_min"]["decay_ratio"]
    assert all(r < 0.05 for r in observed)
    assert assumed > 5.0 * max(observed)


def test_re100_control_is_a_converged_solution_with_the_same_offset_sign() -> None:
    """Re = 100 is the converged control: N=48 and N=64 agree to 1e-7, still ~1.5 % off.

    Reproducing the offset on a case whose boundary layers are thick rules out
    "the tables are simply under-resolved at high Re" as a *sufficient* story.
    """
    s48 = json.loads((RUNS / "spectral" / "cavity_spectral_N048_re100.json").read_text())
    s64 = json.loads((RUNS / "spectral" / "cavity_spectral_N064_re100.json").read_text())
    assert abs(s64["metrics"]["u_min"] - s48["metrics"]["u_min"]) < 1e-6
    # converged, yet 1.5 % away from the tabulated value
    assert s64["metrics"]["u_min"] / -0.21090 - 1.0 == pytest.approx(0.0149, abs=2e-4)
    # and the table's own extremum floor at Re = 100 is *small* here (0.02 %),
    # so the table cannot be blamed by the interpolation argument at this Re
    u100 = extremum_floor(U_CENTRELINE[100.0], mode="min")
    assert u100["floor_upper"] < 0.0005