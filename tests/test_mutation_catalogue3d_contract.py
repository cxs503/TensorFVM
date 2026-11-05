"""Contract test for the 3-D mutation catalogue.

The catalogue is only meaningful if every anchor still identifies exactly one
line of the current source: a drifted anchor would either raise at audit time or
silently mutate the wrong code.  These checks are cheap and run on every suite
invocation, so drift is caught long before the audit is rerun.
"""
from pathlib import Path

import pytest

from tensorfvm.verification.mutation_catalogue3d import CATALOGUE3D

REPO = Path(__file__).resolve().parents[1]


def test_ids_are_unique_and_expected_values_declared():
    ids = [entry["id"] for entry in CATALOGUE3D]
    assert len(ids) == len(set(ids))
    assert {entry["expected"] for entry in CATALOGUE3D} <= {"killed", "survived"}


@pytest.mark.parametrize("entry", CATALOGUE3D, ids=[e["id"] for e in CATALOGUE3D])
def test_anchor_is_unique_and_mutation_is_semantic(entry):
    path = REPO / entry["file"]
    text = path.read_text()
    assert text.count(entry["find"]) == 1, f"{entry['id']}: anchor not unique in {entry['file']}"
    mutated = text.replace(entry["find"], entry["replace"])
    assert mutated != text, f"{entry['id']}: mutation is a no-op"
    compile(mutated, str(path), "exec")
    assert entry["tests"], f"{entry['id']}: no targeted test"


def test_targeted_test_files_exist():
    for entry in CATALOGUE3D:
        for target in entry["tests"]:
            assert (REPO / target).exists(), f"{entry['id']}: missing {target}"