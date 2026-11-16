#!/usr/bin/env python3
"""Register the Ritter/Stoker dam-break case in the archived-source staleness net.

Why this exists
---------------
``docs/VERIFICATION_DEBT_LEDGER.md`` D31 recorded that ``docs/shallow-water-dam-break/``
carried **no** ``manifest.json``, so the dam-break case was the one verification case
whose producer source was not bound by ``tests/test_source_staleness.py``: editing
``shallow_water/dam_break.py`` (or the driver, or the solver) left the archived numbers
in ``src/tensorfvm/verification/artifacts/shallow-water-dam-break/`` unverifiable with no
gate noticing.  This script writes that manifest and freezes the producer source the
archive was produced from, which is the repository's sanctioned fix for the
"unfrozen producer bindings" violation (``source_staleness.freeze_violations``).

Two different questions, deliberately answered by two different mechanisms:

* **reproducibility** (this manifest): the numbers quoted by the pool entry can be
  re-derived from a byte-identical copy of the code that produced them, forever,
  because the copy lives in-tree under ``source-snapshot/``;
* **drift** (``scripts/audit_dambreak_pool_provenance.py`` P1): the *current* source
  still reproduces the archived numbers.  That comparison needs the driver to run and
  therefore is not a function of a digest; ``--check`` here reports drift as a
  readable diff so a stale archive is noticed immediately rather than silently.

Usage::

    PYTHONPATH=src python scripts/register_dam_break_staleness.py --write
    PYTHONPATH=src python scripts/register_dam_break_staleness.py --check
    PYTHONPATH=src python scripts/register_dam_break_staleness.py --check --json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import shutil
import sys
from typing import Dict, Iterable, List

ROOT = pathlib.Path(__file__).resolve().parents[1]

CASE_DIR = ROOT / "docs" / "shallow-water-dam-break"
MANIFEST_PATH = CASE_DIR / "manifest.json"
SNAPSHOT_ROOT = "source-snapshot"

ARTIFACT_DIR = ROOT / "src" / "tensorfvm" / "verification" / "artifacts" / "shallow-water-dam-break"
POOL_ENTRY = ROOT / "docs" / "case-fvm-dambreak-stoker" / "case_pool_entry.json"

#: Producer code whose changes can move a graded number.  ``swe1d`` is the solver,
#: ``exact`` owns the closed forms (Ritter/Stoker), ``dam_break`` the case setup and
#: the instruments (front / bore / step readings), the driver the study protocol.
PRODUCER_FILES = (
    "pyproject.toml",
    "src/tensorfvm/shallow_water/__init__.py",
    "src/tensorfvm/shallow_water/swe1d.py",
    "src/tensorfvm/shallow_water/exact.py",
    "src/tensorfvm/shallow_water/dam_break.py",
    "scripts/run_shallow_water_dam_break.py",
)

#: Artefacts whose bytes the manifest records (informational: ``source_staleness``
#: audits ``SOURCE_KEYS`` only, but a reader comparing digests sees an edited archive).
ARTIFACT_FILES = (
    "metrics.json",
    "convergence.json",
    "exactness.json",
    "sensitivity.json",
    "front_definitions.json",
)

SCHEMA = "tensorfvm.case-shallow-water-dam-break/1"

TOLERANCE = 0.05
T_STAR = 2.0

ACCEPTANCE = (
    "wave-front position x_f(t) (dry bed, Ritter) and the still-water level step h2/h1 "
    "plus the bore position x_s(t) (wet bed, Stoker) within 5 % of the closed form at "
    "t* = 2, waveform monotone (no interior local extremum above 1e-4 h0)"
)


def digest(path: pathlib.Path) -> str:
    """Hex SHA-256 of *path*'s bytes."""

    return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()


def _snapshot_path(relative: str) -> pathlib.Path:
    return CASE_DIR / SNAPSHOT_ROOT / relative


def _artifact_digests() -> Dict[str, str]:
    out = {}
    for name in ARTIFACT_FILES:
        path = ARTIFACT_DIR / name
        if path.is_file():
            out[name] = digest(path)
    return out


def build_manifest(use_snapshot: bool = False) -> Dict[str, object]:
    """Render the manifest from the producer source (no hand-typed digests).

    ``use_snapshot=False`` reads the live tree (what ``--write`` freezes);
    ``use_snapshot=True`` reads the frozen copies, so ``--check`` answers "is the
    committed manifest still the manifest these frozen bytes describe?" without
    confusing that question with live-tree drift (reported separately).
    """

    source = {}
    missing: List[str] = []
    for relative in PRODUCER_FILES:
        path = _snapshot_path(relative) if use_snapshot and _snapshot_path(relative).is_file() else ROOT / relative
        if not path.is_file():
            missing.append(relative)
            continue
        source[relative] = digest(path)
    if missing:
        raise SystemExit(f"producer files missing, refusing to register: {missing}")
    return {
        "schema": SCHEMA,
        "case": "CASE-FVM-DAMBREAK-STOKER",
        "title": "Shallow-water dam break vs the Ritter (1892) / Stoker (1957) closed forms",
        "entry": "src/tensorfvm/shallow_water/dam_break.py",
        "impl_files": list(PRODUCER_FILES),
        "pool_entry": POOL_ENTRY.relative_to(ROOT).as_posix(),
        "repro_cmd": (
            "PYTHONPATH=src python scripts/run_shallow_water_dam_break.py "
            "--out src/tensorfvm/verification/artifacts/shallow-water-dam-break"
        ),
        "acceptance": ACCEPTANCE,
        "tolerance": TOLERANCE,
        "t_star": T_STAR,
        "reference": {
            "dry": "Ritter (1892) Z. Ver. Deutsch. Ing. 36:947-954",
            "wet": "Stoker (1957) Water Waves, ch. 10",
        },
        "source_snapshot_root": SNAPSHOT_ROOT,
        "source_sha256": source,
        "artifacts_sha256": _artifact_digests(),
    }


def render(manifest: Dict[str, object]) -> str:
    """Deterministic manifest text (sorted keys, trailing newline)."""

    return json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False) + "\n"


def write_snapshot(manifest: Dict[str, object]) -> List[str]:
    """Copy every bound producer file into ``source-snapshot/``; return the written paths."""

    written = []
    for relative, want in manifest["source_sha256"].items():
        source = ROOT / relative
        target = _snapshot_path(relative)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        got = digest(target)
        if got != want:
            raise SystemExit(f"snapshot copy of {relative} is not byte-identical ({got} != {want})")
        written.append(relative)
    return written


def drift(manifest: Dict[str, object]) -> Dict[str, Dict[str, str]]:
    """Bound producer files whose *live* bytes differ from the frozen snapshot."""

    out = {}
    for relative, frozen in manifest["source_sha256"].items():
        live = ROOT / relative
        if not live.is_file():
            out[relative] = {"frozen": frozen, "live": "missing"}
            continue
        got = digest(live)
        if got != frozen:
            out[relative] = {"frozen": frozen, "live": got}
    return out


def snapshot_integrity(manifest: Dict[str, object]) -> Dict[str, str]:
    """Snapshot copies that are absent or no longer match the recorded digest."""

    bad: Dict[str, str] = {}
    for relative, frozen in manifest["source_sha256"].items():
        path = _snapshot_path(relative)
        if not path.is_file():
            bad[relative] = "missing"
        elif digest(path) != frozen:
            bad[relative] = "digest-mismatch"
    return bad


def report(manifest: Dict[str, object], check_snapshot: bool = True) -> Dict[str, object]:
    """Machine-readable state of the registration (used by ``--json`` and the tests)."""

    blocked: Dict[str, object] = {}
    committed = MANIFEST_PATH.read_text() if MANIFEST_PATH.is_file() else None
    return {
        "manifest_path": MANIFEST_PATH.relative_to(ROOT).as_posix(),
        "manifest_present": committed is not None,
        "manifest_is_derived": committed == render(manifest),
        "snapshot_integrity": snapshot_integrity(manifest) if check_snapshot else {},
        "drift": drift(manifest),
        "producer_files": list(manifest["source_sha256"]),
        "artifact_digests": dict(manifest["artifacts_sha256"]),
        "blocked": blocked,
    }


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true", help="write manifest + source-snapshot")
    parser.add_argument("--check", action="store_true", help="verify manifest/snapshot, report drift")
    parser.add_argument("--json", action="store_true", help="print the report as JSON")
    parser.add_argument(
        "--allow-drift",
        action="store_true",
        help="exit 0 even when the live source drifted from the snapshot",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    if not args.write and not args.check:
        parser.error("choose --write or --check")

    manifest = build_manifest()
    if args.write:
        written = write_snapshot(manifest)
        CASE_DIR.mkdir(parents=True, exist_ok=True)
        MANIFEST_PATH.write_text(render(manifest), encoding="utf-8")
        print(f"wrote {MANIFEST_PATH.relative_to(ROOT).as_posix()}")
        print(f"froze {len(written)} producer files under "
              f"{(CASE_DIR / SNAPSHOT_ROOT).relative_to(ROOT).as_posix()}/")
        for relative in written:
            print(f"  - {relative}")

    state = report(build_manifest(use_snapshot=True) if args.check else manifest)
    if args.json:
        print(json.dumps(state, indent=2, ensure_ascii=False))
    else:
        print(f"manifest present           : {state['manifest_present']}")
        print(f"manifest is derived        : {state['manifest_is_derived']}")
        print(f"snapshot integrity problems: {state['snapshot_integrity'] or 'none'}")
        print(f"live-vs-frozen drift       : {state['drift'] or 'none'}")

    problems = bool(state["snapshot_integrity"]) or not state["manifest_is_derived"]
    if not args.allow_drift and state["drift"]:
        problems = True
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())