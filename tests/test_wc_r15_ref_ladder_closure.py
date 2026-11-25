"""Round-15 regression gates: R15-CLOSURE.md, r15_certified_rung.json, figures, publish script."""
import json
import os
import subprocess
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent.parent / "docs" / "XVAL-FVM-M18737-ROUND2"
JSON_PATH = HERE / "r15_certified_rung.json"
MD_PATH = HERE / "R15-CLOSURE.md"
PNG_PATH = HERE / "figures" / "fig_r15_ref_ladder.png"
PUBLISH_SCRIPT = HERE / "publish_r15_report.py"


@pytest.fixture
def rung():
    if not JSON_PATH.exists():
        pytest.skip("r15_certified_rung.json not found")
    return json.loads(JSON_PATH.read_text())


@pytest.fixture
def md():
    if not MD_PATH.exists():
        pytest.skip("R15-CLOSURE.md not found")
    return MD_PATH.read_text()


def test_r15_json_exists():
    assert JSON_PATH.exists(), "r15_certified_rung.json missing"


def test_r15_json_structure(rung):
    assert rung["status"] == "measured"
    assert rung["gates_all_pass"] is True
    assert rung["cfl_admission_pass"] is True
    assert rung["missing_rungs"] == []
    assert "E1r6400_cfl_reference" in rung
    assert rung["E1r6400_cfl_reference"]["admitted"] is True
    assert rung["E4_honest_ceiling"]["licensed"] is True


def test_r15_md_exists():
    assert MD_PATH.exists(), "R15-CLOSURE.md missing"


def test_r15_md_content(md):
    assert "Round-15 闭合" in md
    assert "E1r6400" in md
    assert "licensed" in md.lower()


def test_r15_figure_exists():
    assert PNG_PATH.exists(), "fig_r15_ref_ladder.png missing"


def test_r15_publish_script_exists():
    assert PUBLISH_SCRIPT.exists(), "publish_r15_report.py missing"


def test_r15_md_rendered_from_json(rung, md):
    """Check that MD contains key numbers from JSON."""
    e1r = rung["E1r6400_cfl_reference"]
    assert f"{e1r['l1_percent']:.4f}" in md


def test_r15_invariants_vs_r14(rung):
    """Check that invariants_vs_r14 is present and ok."""
    assert "invariants_vs_r14" in rung
    assert rung["invariants_vs_r14"]["ok"] is True
