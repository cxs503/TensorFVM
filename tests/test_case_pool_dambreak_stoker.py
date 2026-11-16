"""Case-pool registration gate for ``CASE-FVM-DAMBREAK-STOKER``.

The ledger acceptance is "wave front ``x_f(t)`` and the still-water level step vs
the analytic solution at ``t* = 2``, waveform monotone without spurious
oscillation, 5 %".  That acceptance was *graded* (``dam_break.py::_verdict``) and
*reported* (``docs/case-fvm-dambreak-stoker/``), but the case was **not in the
pool**: no ``case_pool_entry.json`` and no catalog entry, so its numbers could
drift from the artifact and nothing indexed it.  These tests close that loop:

1. the committed ``case_pool_entry.json`` must be byte-identical to what
   ``tensorfvm.verification.case_pool_dambreak`` derives from the committed
   artifact -- no hand-typed numbers;
2. every ledger quantity must appear as a *primary* graded criterion and pass at
   the pool grid, at ``t* = 2``, against the 5 % gate;
3. coarse grids that fail must stay on the record (``honest_fail``), and the
   grid table must be internally consistent (``pass`` == "no failing criterion");
4. the catalog entry must point at live tests so the catalog gate keeps
   auditing it.
"""

from __future__ import annotations

import json
import os

import pytest

from tensorfvm.verification import catalog_gate
from tensorfvm.verification.case_pool_dambreak import (
    ENTRY_PATH,
    SLUG,
    TOLERANCE,
    T_STAR,
    build_entry,
    render_entry_json,
)

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CATALOG = os.path.join(REPO_ROOT, "docs", "verification-catalog.md")


@pytest.fixture(scope="module")
def entry():
    return build_entry()


def test_pool_entry_is_derived_from_the_committed_artifact(entry):
    """Regeneration must reproduce the committed file exactly (drift gate)."""

    path = os.path.join(REPO_ROOT, ENTRY_PATH)
    with open(path, encoding="utf-8") as handle:
        committed = handle.read()
    assert committed == render_entry_json(entry), (
        "docs/case-fvm-dambreak-stoker/case_pool_entry.json is stale; "
        "re-run: PYTHONPATH=src python -m tensorfvm.verification.case_pool_dambreak"
    )
    # the entry is a registration record: it must not bake in machine-local paths
    assert "/root/" not in committed
    assert entry["slug"] == SLUG


def test_entry_grades_every_ledger_quantity_at_t_star_2(entry):
    """The ledger wording is mapped to primary criteria, and they pass."""

    assert entry["ledger_contract"]["t_star"] == pytest.approx(T_STAR)
    assert entry["ledger_contract"]["tolerance_percent"] == pytest.approx(100.0 * TOLERANCE)

    cases = {case["key"]: case for case in entry["cases"]}
    assert set(cases) == {"ritter-dry", "stoker-wet"}
    for case in cases.values():
        assert case["t_star"] == pytest.approx(T_STAR)
        assert case["pass"] is True
        primaries = {c["criterion"] for c in case["criteria"] if c["primary"]}
        assert primaries, f"{case['key']} has no primary criterion"

    # ledger: 波前位置 x_f(t) -- both the end state and the series
    dry = {c["criterion"]: c for c in cases["ritter-dry"]["criteria"]}
    assert {"front_fan<5%", "front_series<5%"}.issubset(dry)
    # ledger: 静水区水位阶跃
    wet = {c["criterion"]: c for c in cases["stoker-wet"]["criteria"]}
    assert {"step_ratio<5%", "shock_position<5%", "front_series<5%"}.issubset(wet)
    # ledger: 波形须单调无伪振荡
    for case in cases.values():
        by_name = {c["criterion"]: c for c in case["criteria"]}
        assert by_name["monotone(no_extrema)"]["pass"] is True
        assert by_name["monotone(no_extrema)"]["value"] == 0.0  # zero interior extrema

    for case in cases.values():
        for crit in case["criteria"]:
            if crit["primary"]:
                assert crit["pass"] is True, (case["key"], crit["criterion"])
                if crit["unit"] == "%":
                    assert crit["value"] <= crit["threshold"] == pytest.approx(5.0)


def test_coarse_grid_failures_stay_on_the_record(entry):
    """Honest FAILs are recorded, and the grid table is self-consistent."""

    for row in entry["grid_study"]["rows"]:
        for case_key, cell in row["cases"].items():
            assert cell["pass"] is (not cell["verdict_failing"]), (row["n"], case_key)

    fails = {(f["case"], f["n"]) for f in entry["honest_fail"]}
    # coarse grids still miss, but on *different* criteria than they used to:
    # the dry-bed front *series* no longer fails anywhere (its n = 200/400
    # 6.61 % reading was a three-cell-fan, half-cell artefact -- the samples are
    # now excluded by SERIES_MIN_FAN_CELLS and the defect itself is locked in
    # tests/test_dambreak_stoker_series.py::test_dry_three_cell_fan_artefact_is_locked
    # and docs/case-fvm-dambreak-stoker/ANALYSIS-series-fan-cell-floor.md), while
    # the n = 100 dry bed still fails the *legacy* wetted-edge diagnostic and the
    # wet bed still shows interior extrema at n = 100/200 (real coarse-grid
    # behaviour, unchanged).
    assert fails, "no honest coarse-grid miss recorded"
    for record in entry["honest_fail"]:
        assert record["n"] < entry["accepted_grid_min"]
        assert record["failing_criteria"]
    assert ("stoker-wet", 100) in fails
    assert ("stoker-wet", 200) in fails
    for row in entry["grid_study"]["rows"]:
        assert row["cases"]["ritter-dry"]["front_series_error_percent"] < 5.0, row["n"]
    assert entry["accepted_grid_min"] >= 400
    assert entry["verdict"]["accepted_grid_min"] == entry["accepted_grid_min"]
    assert entry["verdict"]["honest_fail_count"] == len(entry["honest_fail"])
    assert entry["verdict"]["all_primary_pass_at_pool_grid"] is True


def test_catalog_entry_is_live_and_the_gate_still_passes():
    """The case must be indexed in the pool, with live test pointers."""

    with open(CATALOG, encoding="utf-8") as handle:
        text = handle.read()
    assert SLUG in text
    entries, _ = catalog_gate.parse_catalog(text, root=REPO_ROOT)
    mine = [e for e in entries if SLUG in e.body]
    assert len(mine) == 1, "expected exactly one catalog entry for the dam-break case"
    registered = mine[0]
    assert registered.passing, "catalog entry must be marked ✅"
    live = [p for p in registered.test_pointers if os.path.exists(os.path.join(REPO_ROOT, p))]
    assert live, registered.test_pointers

    audit = catalog_gate.audit_catalog(CATALOG, root=REPO_ROOT)
    assert audit.passed, json.dumps(audit.to_dict(), indent=1)


def test_markdown_section_numbers_come_from_the_entry(entry):
    """The rendered catalog text must quote the entry's numbers, not literals."""

    from tensorfvm.verification.case_pool_dambreak import render_catalog_section

    section = render_catalog_section(entry)
    dry = {c["criterion"]: c for c in entry["cases"][0]["criteria"]}
    wet = {c["criterion"]: c for c in entry["cases"][1]["criteria"]}
    assert f"{dry['front_fan<5%']['value']:.4f}" in section
    assert f"{wet['step_ratio<5%']['value']:.4f}" in section
    assert f"n >= {entry['accepted_grid_min']}" in section
    # the legacy estimator's failure mode stays visible next to the graded value
    assert "212.500 %" in section


def _recompute_order(rows, field, sub=None):
    """Least-squares slope of log(err) vs log(dx) -- the test's own estimator."""

    import math

    pts = []
    for row in rows:
        value = row.get(field) if sub is None else (row.get(field) or {}).get(sub)
        if value:
            pts.append((math.log(float(row["dx"])), math.log(float(value))))
    assert len(pts) >= 2, "refinement ladder too short to fit an order"
    n = float(len(pts))
    mx = sum(x for x, _ in pts) / n
    my = sum(y for _, y in pts) / n
    den = sum((x - mx) ** 2 for x, _ in pts)
    return sum((x - mx) * (y - my) for x, y in pts) / den


def test_order_claim_is_recomputed_from_the_ladder(entry):
    """``order = 2`` must travel with the order the committed ladder measures.

    The entry used to expose the nominal scheme order alone; a re-user reading
    only ``case_pool_entry.json`` could mistake it for the accuracy of the case.
    The claim is now derived from the convergence artifact, so the test both
    recomputes it (independent estimator) and requires the honest direction:
    on this discontinuous case the observed order is *below* the nominal one.
    """

    import json

    art = json.load(
        open(
            os.path.join(
                REPO_ROOT,
                "src/tensorfvm/verification/artifacts/shallow-water-dam-break/convergence.json",
            ),
            encoding="utf-8",
        )
    )
    assert entry["solver"]["order"] == 2
    assert entry["solver"]["order_evidence"] in entry["reports"]
    assert os.path.exists(os.path.join(REPO_ROOT, entry["solver"]["order_evidence"]))

    for case in entry["cases"]:
        claim = case["order_claim"]
        blob = art[case["key"]]
        assert claim["nominal"] == case["order"] == blob["order"] == 2
        assert claim["grids"] == [row["n"] for row in blob["rows"]]
        # 1. the pool number is reproduced by the test's own fit, not copied
        assert claim["observed_depth_l1"] == pytest.approx(
            _recompute_order(blob["rows"], "depth_l1"), abs=1e-12
        )
        # 2. ... and it agrees with the artifact's own declared order
        assert claim["observed_depth_l1_declared"] == pytest.approx(
            blob["observed_order_l1"], abs=1e-12
        )
        assert claim["observed_headline"] == pytest.approx(
            _recompute_order(blob["rows"], "headline", claim["observed_headline_quantity"]),
            abs=1e-12,
        )
        # 3. the honest direction is recorded, with a scope note, and the report exists
        assert claim["observed_below_nominal"] is True
        assert claim["observed_depth_l1"] < claim["nominal"] - 0.1
        assert claim["scope"] and "accuracy" in claim["scope"]
        assert claim["evidence_report"] == entry["solver"]["order_evidence"]


def test_order_claim_follows_the_artifact_not_a_literal():
    """Power check: perturbing the ladder must move the claim (no hard-coded digit).

    A green ``test_order_claim_is_recomputed_from_the_ladder`` would be vacuous
    if the number were typed in.  Corrupt the finest row by a factor of 8 (one
    half-refinement's worth of second order) and require the derived claim to
    follow it.
    """

    import copy

    from tensorfvm.verification.case_pool_dambreak import load_artifacts, render_entry_json

    art = load_artifacts()
    corrupted = copy.deepcopy(art)
    finest = max(corrupted["convergence"]["ritter-dry"]["rows"], key=lambda r: r["n"])
    finest["depth_l1"] *= 8.0
    claim = [
        case["order_claim"]
        for case in build_entry(corrupted)["cases"]
        if case["key"] == "ritter-dry"
    ][0]
    base = build_entry(art)["cases"][0]["order_claim"]["observed_depth_l1"]
    moved = claim["observed_depth_l1"] - base
    # one corner of a 5-point ladder: the fitted slope drops by (delta * lever arm),
    # i.e. ~0.6 here -- far above the 1e-12 agreement the derived claim has
    assert moved < -0.4, moved  # a coarser-looking ladder fits a lower order
    assert render_entry_json(build_entry(corrupted)) != render_entry_json(build_entry(art))


def test_catalog_section_states_the_order_claim(entry):
    """Catalog readers see the measured order next to the nominal one."""

    from tensorfvm.verification.case_pool_dambreak import render_catalog_section

    section = render_catalog_section(entry)
    assert "**Order claim**" in section
    for case in entry["cases"]:
        claim = case["order_claim"]
        assert f"{claim['observed_depth_l1']:.3f}" in section
    assert "scheme" in section and "docs/case-fvm-dambreak-order-audit/REPORT.md" in section


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-q"]))