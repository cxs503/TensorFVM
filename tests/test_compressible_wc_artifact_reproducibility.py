"""Byte-level reproducibility gate for the CASE-WOODWARD-COLELLA logbook artefacts.

``ladder/data/asymptotic_analysis.json`` and ``ladder/data/plateau_convergence.json``
are the machine-readable evidence behind
``docs/CASE-WOODWARD-COLELLA-nx3200-asymptotic.md``: every number quoted in that
report is a field of one of them.  The sibling gates
(``test_compressible_wc_asymptotic.py``, ``test_compressible_wc_plateau_metrics.py``)
re-derive the *judged* quantities and compare them to the committed ones, which is
the right thing to do but can only ever cover the fields somebody remembered to
assert on.

Both artefacts used to embed the **absolute** path of the work tree that produced
them (``/root/.../reference_density.csv``, ``/root/.../ladder_profile_nx3200_hllc.npz``,
...).  An honest re-run in any other checkout therefore rewrote the same numbers
with a different prefix, so the JSON could not be pinned byte-for-byte - drift in
an unasserted field stayed invisible - and the committed file leaked the local
directory layout.  The generators now emit repo-relative provenance
(``wc_asymptotic_analysis._case_relative`` / ``wc_plateau_convergence._repo_relative``),
which makes the strictest possible gate available: *regenerate and diff the bytes*.

The generators are pure numpy over committed ``.npz`` profiles (no solver), so the
whole module costs ~10 s.
"""
from __future__ import annotations

import json
import os
import struct
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
CASE = ROOT / "docs" / "woodward-colella-double-blast"
LADDER = CASE / "ladder"
DATA = LADDER / "data"
ARTIFACTS = CASE / "artifacts"

#: generator script -> artefacts it writes (all relative to the ladder/data dir)
GENERATORS = {
    "wc_asymptotic_analysis.py": ("asymptotic_analysis.json", "asymptotic_tables.md"),
    "wc_plateau_convergence.py": ("plateau_convergence.json", "plateau_tables.md"),
}


def _run(script: str, outdir: Path) -> None:
    """Run one generator into ``outdir`` exactly as the report documents it."""
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        [str(ROOT / "src")] + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else [])
    )
    cmd = [sys.executable, str(LADDER / script)]
    if script == "wc_asymptotic_analysis.py":
        cmd += ["--outdir", str(outdir),
                "--json", str(outdir / "asymptotic_analysis.json"),
                "--markdown", str(outdir / "asymptotic_tables.md")]
    else:
        cmd += ["--outdir", str(outdir)]
    proc = subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True, text=True,
                          timeout=900)
    assert proc.returncode == 0, (
        f"{script} failed:\nSTDOUT\n{proc.stdout[-2000:]}\nSTDERR\n{proc.stderr[-2000:]}")


@pytest.mark.parametrize("script", sorted(GENERATORS))
def test_committed_artifacts_are_byte_reproducible(script, tmp_path):
    """Regeneration into a scratch dir must reproduce the committed bytes exactly.

    Byte identity is the strongest no-drift statement available: it catches a
    stale artefact, an edit that only touched a descriptive field, a changed
    rounding path, and - the defect this gate exists for - a provenance field
    that depends on where the script ran.
    """
    _run(script, tmp_path)
    for name in GENERATORS[script]:
        committed = DATA / name
        fresh = tmp_path / name
        assert fresh.exists(), f"{script} did not write {name}"
        assert fresh.read_text() == committed.read_text(), (
            f"{name} is stale or environment-dependent: re-run {script} and commit "
            f"the result (diff: {committed} vs {fresh})"
        )


@pytest.mark.parametrize("name", ["asymptotic_analysis.json", "plateau_convergence.json"])
def test_committed_evidence_carries_no_absolute_paths(name):
    """Provenance must be repo-relative: no checkout path may enter evidence.

    Absolute paths make the artefact machine-specific (see module docstring) and
    expose the layout of whatever work tree produced it, so this is a standing
    property of the committed files, checked without re-running anything.
    """
    doc = json.loads((DATA / name).read_text())
    offenders: list[str] = []

    def walk(node, where=""):
        if isinstance(node, dict):
            for key, value in node.items():
                walk(value, f"{where}.{key}" if where else str(key))
        elif isinstance(node, list):
            for i, value in enumerate(node):
                walk(value, f"{where}[{i}]")
        elif isinstance(node, str) and node.startswith("/"):
            offenders.append(f"{where} = {node}")

    walk(doc)
    assert offenders == [], f"{name} embeds absolute paths: {offenders}"

    # the provenance fields that used to be absolute now name the file portably
    if name == "asymptotic_analysis.json":
        assert doc["reference"]["committed_raw"] == "reference/reference_density.csv"
        assert doc["reference"]["calibrated"] == "reference/reference_density_calibrated.csv"
    else:
        assert doc["reference"]["path"] == \
            "docs/woodward-colella-double-blast/reference/reference_density.csv"
        for tag, row in doc["rows"].items():
            assert row["path"].startswith("docs/woodward-colella-double-blast/"), tag
            assert Path(row["path"]).exists(), tag


def test_asymptotic_figure_artifact_present():
    """The report's figure is a deliverable: it must exist and stay a real plot.

    A PNG's IHDR is checked rather than its pixels: the byte stream legitimately
    depends on the matplotlib build, but the figure must still be a wide,
    high-resolution multi-panel plot and not a stub that was committed to satisfy
    a file list.
    """
    fig = ARTIFACTS / "fig_wc_asymptotic_nx3200.png"
    blob = fig.read_bytes()
    assert blob[:8] == b"\x89PNG\r\n\x1a\n", "not a PNG"
    width, height = struct.unpack(">II", blob[16:24])
    assert width >= 1600 and height >= 400, (width, height)
    assert len(blob) > 50_000, len(blob)
    # and it is driven by the committed JSON, which is what got regenerated above
    assert json.loads((DATA / "asymptotic_analysis.json").read_text())["families"]["hllc"]