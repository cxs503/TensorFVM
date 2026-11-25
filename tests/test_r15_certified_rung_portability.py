"""Portability gate for the R15 certified-rung record (anchor XVAL-FVM-M18737-ROUND2).

PR #1113 (m25906) was closed by the owner because it "fixed" the figure ``path``
in ``r15_certified_rung.json`` by swapping one *machine-specific absolute* path
(``/srv/tensor/agent/work/m23274/...``) for another
(``/root/tp-agent/work/m25906/...``).  A machine-specific absolute path in a
committed artifact is broken by construction: it points at a scratch worktree
that no longer exists on any other node.

This gate pins the correct repair:

1. every ``figures[*].path`` in the committed record is repo-relative
   (no leading ``/``, no ``/srv/tensor`` / ``/root/tp-agent`` worktree prefix);
2. the referenced figure actually exists at that location in the repo, so the
   record is self-contained and re-checkable on any checkout;
3. the certification invariants of the record survive the edit untouched
   (admission, licensing, adjudication) -- the edit must touch *only* the path.
"""
from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
RECORD = REPO / "docs" / "XVAL-FVM-M18737-ROUND2" / "r15_certified_rung.json"

# prefixes that betray a machine-specific scratch worktree
_MACHINE_PREFIXES = ("/srv/tensor/", "/root/tp-agent/", "/home/", "/tmp/")


def _record() -> dict:
    return json.loads(RECORD.read_text())


def test_record_is_valid_json_with_figures():
    v = _record()
    assert v["round"] == 15
    assert isinstance(v["figures"], list) and v["figures"], "record must list its figures"


def test_figure_paths_are_repo_relative():
    v = _record()
    for fig in v["figures"]:
        p = fig["path"]
        assert not p.startswith("/"), f"figure path must be repo-relative, got {p!r}"
        assert not p.startswith(_MACHINE_PREFIXES), (
            f"figure path embeds a machine-specific worktree prefix: {p!r}"
        )
        assert "work/m" not in p, f"figure path embeds a mission worktree: {p!r}"


def test_figure_exists_in_repo():
    v = _record()
    for fig in v["figures"]:
        target = REPO / fig["path"]
        assert target.is_file(), f"figure referenced by the record is missing: {fig['path']}"
        assert target.stat().st_size > 0, f"figure is empty: {fig['path']}"
        # name and path must agree
        assert target.name == fig["name"], f"path/name mismatch: {fig}"


def test_certification_invariants_survive():
    """The path repair must not perturb the certified accounting."""
    v = _record()
    assert v["r15_admitted"] is True
    assert v["E1r6400_cfl_reference"]["admitted"] is True
    assert v["E4_honest_ceiling"]["licensed"] is True
    assert v["invariants_vs_r14"]["ok"] is True
    assert "CERTIFIED_UNDER_1PCT_AT_NX12800" in v["adjudication"]
    assert "HONEST_UNDER_1PCT_B0_B1_B2_B4" in v["adjudication"]
