"""Attribution of the Re = 400 v-centreline discrepancy to one published entry.

The committed Ghia ladder (``docs/verification-lid-cavity-ladder/summary.json``)
left one point of the Re = 400 v centreline failing while everything around it
converges, and handed on the question "reference table or extraction?".
``scripts/audit_ghia_re400_v_attribution.py`` answers it; the tests here keep the
answer pinned and keep the audit honest:

* the solver value at that point is grid converged at clean second order and the
  gap *survives* Richardson extrapolation, while every other point of both
  centrelines stays inside a few percent;
* the metric path cannot manufacture a gap of that size (interpolation error
  bound is two orders of magnitude smaller);
* in all six published tables the only place where the sampled profile has a
  corner is that entry (and its immediate neighbour);
* a corruption injected anywhere else is still detected, so the audit has teeth;
* :func:`tensorfvm.lid_cavity.ghia_acceptance` still fails by default and can
  only ignore a point that is in the register, and it reports it when it does.
"""
from __future__ import annotations

import importlib.util
import os

import pytest

from tensorfvm.lid_cavity import ghia_acceptance
from tensorfvm.lid_cavity_reference import (
    DISPUTES, V_CENTRELINE, table, smoothness_audit,
)

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPT = os.path.join(REPO, "scripts", "audit_ghia_re400_v_attribution.py")


def _audit_module():
    spec = importlib.util.spec_from_file_location("ghia_attribution_audit", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _bundle():
    module = _audit_module()
    path = os.path.join(REPO, "docs", "verification-lid-cavity-ladder", "summary.json")
    try:
        return module, module.load_bundle(path)
    except SystemExit as error:  # pragma: no cover - evidence checkout policy
        pytest.skip(f"ladder bundle unavailable: {error}")


# --------------------------------------------------------------------------- #
# reference-side audit (no solver run needed)
# --------------------------------------------------------------------------- #
def test_the_only_corner_in_all_six_tables_is_the_registered_dispute():
    audit = smoothness_audit()
    flagged = {item["key"] for item in audit["flagged"]}
    assert audit["ok"], audit
    assert "400/v/0.9063" in flagged
    # the corrupted entry flattens the slope of its neighbour too; nothing else
    assert flagged <= {"400/v/0.9063", "400/v/0.9453"}
    assert audit["unexplained"] == []
    assert (400.0, "v", 0.9063) in DISPUTES


def test_a_corruption_anywhere_else_is_still_flagged(monkeypatch):
    """Negative control: the audit must not be a rubber stamp for one known key."""
    # a flat spot injected where the published profile has a steep slope
    rey, which, x = 1000.0, "v", 0.8594
    table_rows = list(table(rey, which))
    index = [row[0] for row in table_rows].index(x)
    table_rows[index] = (x, -0.51550)
    assert table_rows[index][1] != V_CENTRELINE[rey][index][1]
    monkeypatch.setitem(V_CENTRELINE, rey, tuple(table_rows))
    audit = smoothness_audit()
    assert f"{rey:g}/{which}/{x:.4f}" in {item["key"] for item in audit["flagged"]}
    assert not audit["ok"]
    assert f"{rey:g}/{which}/{x:.4f}" in audit["unexplained"]


# --------------------------------------------------------------------------- #
# evidence-side audit: convergence, extrapolation, extraction bound
# --------------------------------------------------------------------------- #
def test_the_gap_survives_extrapolation_while_its_neighbours_do_not():
    module, bundle = _bundle()
    cases = module.cases_by_grid(bundle)
    rows = module.pointwise(cases, 400.0, "v")
    disputed = module.observed_order(rows[0.9063]["grid"])
    assert disputed["order"] == pytest.approx(2.0, abs=0.05)
    assert disputed["richardson_limit"] == pytest.approx(-0.38990, abs=5e-4)
    # the disputed entry is the only large deviation on the profile: its three
    # neighbours stay an order of magnitude closer at the finest grid
    finest = max(rows[0.9063]["grid"])
    gaps = {x: abs(rows[x]["grid"][finest] - rows[x]["reference"])
            for x in (0.9453, 0.8594, 0.8047)}
    disputed_gap = abs(rows[0.9063]["grid"][finest] - rows[0.9063]["reference"])
    assert max(gaps.values()) < 0.03, gaps
    assert disputed_gap > 5 * max(gaps.values()), (disputed_gap, gaps)


def test_the_metric_path_cannot_manufacture_the_gap():
    module, bundle = _bundle()
    rows = module.pointwise(module.cases_by_grid(bundle), 400.0, "v")
    finest = max(rows[0.9063]["grid"])
    bound = module.interpolation_bound(rows, 0.9063, finest)
    gap = abs(rows[0.9063]["grid"][finest] - rows[0.9063]["reference"])
    assert bound["target_inside_sampled_range"] is True  # no extrapolation involved
    assert gap / bound["interpolation_error_bound"] > 100


def test_register_record_agrees_with_the_committed_evidence():
    module, bundle = _bundle()
    record = DISPUTES[(400.0, "v", 0.9063)]
    rows = module.pointwise(module.cases_by_grid(bundle), 400.0, "v")
    assert set(record["computed_n"]) == set(rows[0.9063]["grid"])
    for n, value in record["computed_n"].items():
        assert rows[0.9063]["grid"][n] == pytest.approx(value, abs=5e-5), n
    converged = module.observed_order(rows[0.9063]["grid"])
    assert record["observed_order"] == pytest.approx(converged["order"], abs=0.05)
    assert record["richardson_limit"] == pytest.approx(converged["richardson_limit"], abs=5e-4)
    assert record["evidence"]


# --------------------------------------------------------------------------- #
# gate behaviour: fail by default, ignore only what is registered, never hide it
# --------------------------------------------------------------------------- #
def _synthetic_result(reynolds=400.0, extra_bad=None):
    """A perfect score_against_ghia-shaped result with the disputed point bad."""
    out = {"reynolds": reynolds}
    for which in ("u", "v"):
        rows = table(reynolds, which)
        coord = [row[0] for row in rows]
        reference = [row[1] for row in rows]
        computed = list(reference)
        if which == "v":
            index = coord.index(0.9063)
            computed[index] = reference[index] - 0.15
            if extra_bad is not None:
                other = coord.index(extra_bad)
                computed[other] = reference[other] - 0.2
        delta = [c - r for c, r in zip(computed, reference)]
        jmin = reference.index(min(reference))
        jmax = reference.index(max(reference))
        out[which] = {
            "profiles": {"coord": coord, "reference": reference,
                         "computed": computed, "delta": delta},
            "max_abs_error": max(abs(d) for d in delta),
            "rms_error": 0.0,
            "extremum": {"min_reference": [coord[jmin], reference[jmin]],
                         "min_computed": [coord[jmin], computed[jmin]],
                         "max_reference": [coord[jmax], reference[jmax]],
                         "max_computed": [coord[jmax], computed[jmax]]},
        }
    out["vortex_centre"] = {"computed": [0.5547, 0.6055]}
    return out


def test_default_gate_keeps_failing_on_the_disputed_point():
    result = _synthetic_result()
    verdict = ghia_acceptance(result)
    assert verdict["pass"] is False
    assert verdict["registered_disputes"] == []
    assert verdict["criteria"]["v_max_abs_error"]["value"] > 0.03


def test_registered_dispute_can_be_ignored_but_is_reported():
    verdict = ghia_acceptance(_synthetic_result(), allow_registered_disputes=True)
    assert verdict["pass"] is True
    ignored = verdict["registered_disputes"]
    assert [item["x"] for item in ignored] == [0.9063]
    assert ignored[0]["published"] == -0.23827
    assert ignored[0]["evidence"]
    assert verdict["criteria"]["v_max_abs_error"]["ignored"] == 1
    assert verdict["disputes_declared"] is True


def test_a_second_bad_point_still_fails_the_dispute_aware_gate():
    verdict = ghia_acceptance(_synthetic_result(extra_bad=0.2344),
                              allow_registered_disputes=True)
    assert verdict["pass"] is False
    assert verdict["criteria"]["v_max_abs_error"]["value"] == pytest.approx(0.2, abs=1e-9)
    assert [item["x"] for item in verdict["registered_disputes"]] == [0.9063]