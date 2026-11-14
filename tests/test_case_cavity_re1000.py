"""Machine gate for the committed ``CASE-FVM-CAVITY-RE1000`` evidence bundle.

The tests never trust a stored verdict: they re-derive the numbers from the
artefacts and, for the profiles, from the stored velocity fields themselves.

    PYTHONPATH=src python -m pytest tests/test_case_cavity_re1000.py -q
"""

from __future__ import annotations

import json
import pathlib

import numpy as np
import pytest
import torch

from tensorfvm.case_cavity_re1000 import CRITERIA, REYNOLDS
from tensorfvm.lid_cavity import hermite_interpolate
from tensorfvm.lid_cavity_reference import VORTEX_CENTRE, table
from tensorfvm.verification.ladder import lid_cavity_verdict

RUNS = (pathlib.Path(__file__).resolve().parents[1]
        / "docs" / "CASE-FVM-CAVITY-RE1000" / "runs")
TOLERANCE = 0.05


def _json(name: str) -> dict:
    return json.loads((RUNS / name).read_text())


@pytest.fixture(scope="module")
def summary() -> dict:
    return _json("summary.json")


@pytest.fixture(scope="module")
def cases(summary) -> dict[int, dict]:
    return {int(c["n"]): c for c in summary["cases"]}


def test_evidence_files_exist() -> None:
    for name in ("summary.json", "benchmark.json", "fields.npz"):
        path = RUNS / name
        assert path.is_file() and path.stat().st_size > 0, name
    assert summary_has_grids()


def summary_has_grids() -> bool:
    payload = _json("summary.json")
    grids = set(int(n) for n in payload["resolutions"])
    assert {32, 64, 128} <= grids, grids
    for n in grids:
        path = RUNS / f"lid_cavity_Re1000_n{n}.json"
        assert path.is_file() and path.stat().st_size > 0, path.name
    return True


def test_profiles_are_reproducible_from_stored_fields(cases) -> None:
    """The stored centreline must be what the stored field actually contains."""
    fields = np.load(RUNS / "fields.npz")
    for n, case in cases.items():
        h = float(case["h"])
        coords = (np.arange(n) + 0.5) * h
        u_field = torch.as_tensor(fields[f"u_{n}"])
        v_field = torch.as_tensor(fields[f"v_{n}"])
        assert tuple(u_field.shape) == (n + 1, n)
        assert tuple(v_field.shape) == (n, n + 1)

        y, u_line = coords, u_field[n // 2, :]
        x, v_line = coords, v_field[:, n // 2]
        for which, coord, line in (("u", y, u_line), ("v", x, v_line)):
            pts = table(REYNOLDS, which)
            got = hermite_interpolate(torch.as_tensor(coord), line, [p[0] for p in pts])
            stored = np.asarray(case[which]["profiles"]["computed"], dtype=float)
            assert np.allclose(np.asarray(got, dtype=float), stored, atol=1e-12, rtol=0.0), (n, which)


def test_criteria_recomputed_from_stored_profiles(cases) -> None:
    """Recompute the profile criteria from ref/got pairs, not from the verdict."""
    for n, case in cases.items():
        for cid, which in (("C1_u_profile_max_abs", "u"), ("C2_v_profile_max_abs", "v")):
            ref = np.asarray(case[which]["profiles"]["reference"], dtype=float)
            got = np.asarray(case[which]["profiles"]["computed"], dtype=float)
            measured = float(np.abs(got - ref).max())
            assert measured == pytest.approx(case["criteria"][cid]["value"], abs=1e-12), (n, cid)
        for cid, which in (("C3_u_min_relative", "u"), ("C4_v_max_relative", "v")):
            ref = np.asarray(case[which]["profiles"]["reference"], dtype=float)
            got = np.asarray(case[which]["profiles"]["computed"], dtype=float)
            j = int(np.argmin(ref)) if which == "u" else int(np.argmax(ref))
            measured = abs(got[j] - ref[j]) / abs(ref[j])
            assert measured == pytest.approx(case["criteria"][cid]["value"], abs=1e-12), (n, cid)
            basis = case["criteria"]["_extremum_basis"][which]
            assert basis["reference"] == pytest.approx(float(ref[j]))
            assert basis["computed"] == pytest.approx(float(got[j]))


def test_finest_grid_passes_all_criteria(cases, summary) -> None:
    finest = max(cases)
    assert finest >= 128
    case = cases[finest]
    assert case["converged"] is True
    for spec in CRITERIA:
        cid = str(spec["id"])
        entry = case["criteria"][cid]
        assert entry["value"] <= entry["threshold"], (cid, entry)
    assert case["verdict"] == "PASS"
    assert summary["finest_verdict"] == "PASS"

    # the two numbers the anchor actually asks for, recomputed here
    u_err = max(abs(a - b) for a, b in zip(case["u"]["profiles"]["computed"],
                                           case["u"]["profiles"]["reference"]))
    v_err = max(abs(a - b) for a, b in zip(case["v"]["profiles"]["computed"],
                                           case["v"]["profiles"]["reference"]))
    assert u_err < TOLERANCE and v_err < TOLERANCE

    # primary vortex centre against the Ghia value
    ref = VORTEX_CENTRE[REYNOLDS]
    got = case["vortex_centre"]["computed"]
    assert max(abs(got[0] - ref[0]), abs(got[1] - ref[1])) < TOLERANCE


def test_coarse_grid_is_allowed_to_fail(cases) -> None:
    """n = 32 is reported, not hidden: it must be a real (failed) number."""
    assert 32 in cases
    assert cases[32]["criteria"]["C1_u_profile_max_abs"]["value"] > 0.05


def test_benchmark_table_matches_summary(summary) -> None:
    bench = _json("benchmark.json")
    assert bench["case"] == summary["case"]
    assert len(bench["rows"]) == len(summary["resolutions"])
    for row, case in zip(sorted(bench["rows"], key=lambda r: r["n"]),
                         sorted(summary["cases"], key=lambda c: c["n"])):
        assert row["n"] == case["n"]
        assert row["u_min"]["ref"] == pytest.approx(
            case["criteria"]["_extremum_basis"]["u"]["reference"])
        assert row["v_max"]["got"] == pytest.approx(
            case["criteria"]["_extremum_basis"]["v"]["computed"])
        assert row["u_profile_max_abs"]["err"] == pytest.approx(
            case["criteria"]["C1_u_profile_max_abs"]["value"])


def test_ladder_consumes_the_bundle(cases) -> None:
    """Main verification pipeline (ladder C2) evaluated on this bundle's artefacts.

    The ladder criterion is stricter than the case pool (2 % of the lid speed, not
    5 %), so the assertion binds the *measured value* to the stored artefacts and
    checks that the ladder's own verdict is the mechanical consequence of it.  A
    FAIL there is reported, not hidden: it would mean the finest grid does not meet
    the ladder's tighter band.
    """
    verdicts = lid_cavity_verdict(artifact_dir=RUNS)
    evaluated = [v for v in verdicts if v.detail.get("evaluated_on")]
    assert evaluated, verdicts
    verdict = evaluated[0]
    finest = max(cases)
    expected = max(cases[finest]["u"]["max_abs_error_consistent"],
                   cases[finest]["v"]["max_abs_error_consistent"])
    assert verdict.measured == pytest.approx(expected, abs=1e-12)
    assert verdict.threshold == pytest.approx(0.02)
    assert verdict.verdict == ("PASS" if expected <= 0.02 else "FAIL"), verdict.note
    assert cases[finest]["u"]["max_abs_error_consistent"] < TOLERANCE
    assert cases[finest]["v"]["max_abs_error_consistent"] < TOLERANCE
