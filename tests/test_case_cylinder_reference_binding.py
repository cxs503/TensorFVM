"""Binding test for the CASE-CYLINDER-DRAG-FVM case-level reference file.

``scripts/case_cylinder_drag_fvm.py analyze`` reads an external reference file
(``artifacts/case-cylinder-drag-fvm/reference/reference.json``).  That file was
missing from the repository, so the analyze/report steps of the documented
reproduction command failed with ``FileNotFoundError``.  This test keeps the
committed reference file (a) parseable, (b) complete for every quantity the
harness judges, and (c) numerically identical to the solver-side constant
``tensorfvm.verification.dfg.REFERENCE`` so the two cannot drift apart.
"""
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
REFERENCE_FILE = REPO / "artifacts" / "case-cylinder-drag-fvm" / "reference" / "reference.json"
QUANTITIES = ("drag", "lift", "pressure_difference_pa")


def test_reference_file_exists_and_is_complete():
    reference = json.loads(REFERENCE_FILE.read_text())
    for key in QUANTITIES:
        assert key in reference["values"], key
        assert isinstance(reference["values"][key], float), key
    for key in ("case", "citation", "source_url", "provenance"):
        assert reference.get(key), key


def test_reference_file_matches_solver_constant():
    from tensorfvm.verification.dfg import REFERENCE

    values = json.loads(REFERENCE_FILE.read_text())["values"]
    for key in QUANTITIES:
        assert values[key] == REFERENCE[key], (key, values[key], REFERENCE[key])