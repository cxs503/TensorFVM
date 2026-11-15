"""Registry completeness for the FEAT-UNSTR3D-MESH benchmark cases."""
from tensorfvm.unstr3d_benchmark import ANCHOR_SLUG, CASES, sql_statements

REQUIRED = ("anchor_slug", "case_name", "metric", "tolerance", "ref_source", "repo")


def test_anchor_slug_matches_the_feature():
    assert ANCHOR_SLUG == "FEAT-UNSTR3D-MESH"
    assert all(case["anchor_slug"] == ANCHOR_SLUG for case in CASES)


def test_every_case_declares_metric_tolerance_and_reference():
    for case in CASES:
        for key in REQUIRED:
            assert case[key], (case["case_name"], key)
        assert case["repo"] == "TensorFVM"
        assert len(case["tolerance"]) > 0


def test_sql_statements_cover_every_case():
    statements = sql_statements()
    assert statements[0].startswith("--")
    assert sum(statement.startswith("INSERT INTO tensor_benchmark_cases") for statement in statements) == len(CASES)
