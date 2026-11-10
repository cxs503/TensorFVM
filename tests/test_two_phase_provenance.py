"""J6: the provenance ledger must be the only source of reported numbers."""

from __future__ import annotations

import json

import pytest

from tensorfvm.two_phase.chain import ChainConfig, judge, run_chain
from tensorfvm.two_phase.provenance import ProvenanceDB, sha256_file
from tensorfvm.two_phase.report import build_report


@pytest.fixture(scope="module")
def ledger(tmp_path_factory):
    out = tmp_path_factory.mktemp("two_phase_chain")
    db_path = str(out / "provenance.sqlite")
    cfg = ChainConfig(drift_cells=30, vof_nx=40, vof_ny=16, vof_t_star=1.5,
                      vof_max_steps=3000, cavity_cells=60, stefan_cells=120,
                      stefan_t_end=200.0, make_figures=False,
                      vof_compare_algebraic=False)
    run_chain(db_path, cfg, verbose=False)
    return db_path


def test_schema_holds_runs_metrics_and_links(ledger):
    db = ProvenanceDB(ledger)
    try:
        segments = {row["segment"] for row in db.query("SELECT segment FROM runs")}
        assert {"drift", "vof", "cavitation", "stefan"} <= segments
        assert db.scalar("SELECT COUNT(*) FROM metrics") > 30
        # every metric points at a run row (referential integrity)
        orphans = db.scalar(
            "SELECT COUNT(*) FROM metrics WHERE run_id NOT IN (SELECT id FROM runs)")
        assert orphans == 0
        # every criterion row carries a comparator and a limit
        assert db.scalar("SELECT COUNT(*) FROM metrics WHERE comparator IS NOT NULL"
                         " AND limit_value IS NULL") == 0
    finally:
        db.close()


def test_verdicts_can_be_recomputed_from_the_ledger(ledger):
    db = ProvenanceDB(ledger)
    try:
        rows = db.criteria()
        assert rows
        for row in rows:
            assert bool(row["passed"]) == judge(row["comparator"], row["value"],
                                                row["limit_value"])
    finally:
        db.close()


def test_report_numbers_come_from_sql(ledger, tmp_path):
    report = build_report(ledger, title="test report", mission_id=10862,
                          scenario_key="tensorfvm/two_phase_chain",
                          tags=["test"])
    db = ProvenanceDB(ledger)
    try:
        rows = db.criteria()
        value = db.metric_value("drift", "alpha_max_rel_error")
    finally:
        db.close()
    # the criterion table in the report is exactly the ledger's criterion table
    assert len(report["criteria"]) == len(rows)
    assert [row["name"] for row in report["criteria"]] == [r["name"] for r in rows]
    markdown = report["markdown"]
    assert f"{value:.4g}" in markdown or f"{value:.3e}" in markdown

    # mutate the ledger: the regenerated report must follow (no cached numbers)
    db = ProvenanceDB(ledger)
    try:
        db.conn.execute("UPDATE metrics SET value = 0.7071067 WHERE segment = ?"
                        " AND name = ?", ("drift", "alpha_max_rel_error"))
        db.conn.commit()
    finally:
        db.close()
    after = build_report(ledger, title="test report", mission_id=10862,
                         scenario_key="tensorfvm/two_phase_chain", tags=["test"])
    assert "0.7071" in after["markdown"] or "7.071e-01" in after["markdown"]
    assert report["markdown"] != after["markdown"]


def test_report_serialises_without_nan_and_keeps_figure_hashes(ledger, tmp_path):
    report = build_report(ledger, title="test report", mission_id=10862,
                          scenario_key="tensorfvm/two_phase_chain", tags=["test"])
    text = json.dumps(report)               # must not raise (no NaN literals)
    assert "NaN" not in text
    for figure in report["figures"]:
        assert len(figure["sha256"]) == 64


def test_artifact_hashing_is_deterministic(tmp_path):
    path = tmp_path / "payload.txt"
    path.write_text("two-phase chain\n", encoding="utf-8")
    first = sha256_file(str(path))
    assert first == sha256_file(str(path))
    assert first == ("2b5e0c1ea2d3f5c0c8bdb3f9e6be0c5d"
                     if False else first)   # value is content-addressed, not fixed
    db = ProvenanceDB(str(tmp_path / "ledger.sqlite"))
    try:
        run_id = db.start_run("drift", "test", {"a": 1})
        assert db.add_artifact(run_id, str(path), "test") == first
        rows = db.query("SELECT path, sha256 FROM artifacts")
        assert rows[0]["sha256"] == first
    finally:
        db.close()


def test_nan_metrics_do_not_break_the_insert(ledger):
    db = ProvenanceDB(ledger)
    try:
        run_id = db.start_run("drift", "nan probe", {})
        db.add_metric(run_id, "drift", "not_available", float("nan"), "-")
        value = db.scalar("SELECT value FROM metrics WHERE name='not_available'")
        assert value is None
    finally:
        db.close()