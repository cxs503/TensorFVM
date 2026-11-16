"""Bind / refresh the numbers the ``CASE-FVM-DAMBREAK-STOKER`` pool entry quotes.

The pool entry carries two derived blocks and a prose verdict that quotes
numbers from two artifact bundles.  This script is the only writer of the blocks
and the only checker of the prose:

    PYTHONPATH=src python scripts/audit_dambreak_pool_claims.py --write
    PYTHONPATH=src python scripts/audit_dambreak_pool_claims.py --check
    PYTHONPATH=src python scripts/audit_dambreak_pool_claims.py --out <dir>

``--write``  re-derives ``probe_ladder`` (from
             ``docs/case-fvm-dambreak-stoker/artifacts/reproduction.json``) and
             ``gate_margins`` (from ``metrics.json``) and writes them into the
             pool entry, so no number in either block is ever typed by hand.
``--check``  verifies the committed blocks *and* every number still written in
             prose (``verdict_qualified``, ``graded_set_note``); exit code 1 on
             any problem, so it can gate a PR.
``--out``    additionally writes ``claims.json`` (the derived blocks plus the
             margin table) for the report/figures, so a reader can trace every
             quoted number without opening the entry.

The prose numbers are not generated: a sentence like "Ritter 5.58 %" is a
*reading* of the evidence, and the checker's job is to make a stale reading fail
loudly (this is how the 2026-11-16 overstatement of the margin bounds was
caught).
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tensorfvm.verification import dambreak_pool_claims as claims  # noqa: E402

DEFAULT_OUT = ROOT / "docs" / "CASE-FVM-DAMBREAK-STOKER-POOL-CLAIMS"


def _entry_path() -> pathlib.Path:
    return ROOT / claims.POOL_ENTRY


def _dump(entry: dict) -> str:
    """The entry's on-disk formatting (indent=1, UTF-8, trailing newline)."""
    return json.dumps(entry, indent=1, ensure_ascii=False) + "\n"


def refresh(entry: dict, metrics: dict, bundle: dict) -> dict:
    """Return ``entry`` with both derived blocks replaced by fresh derivations."""
    updated = dict(entry)
    updated["probe_ladder"] = claims.probe_ladder(bundle)
    updated["gate_margins"] = claims.gate_margins(metrics)
    return updated


def write(out_dir: pathlib.Path) -> int:
    entry = claims.load_json(claims.POOL_ENTRY)
    metrics = claims.load_json(claims.METRICS_ARTIFACT)
    bundle = claims.load_json(claims.PROBE_BUNDLE)
    updated = refresh(entry, metrics, bundle)
    changed = _dump(updated) != _entry_path().read_text(encoding="utf-8")
    _entry_path().write_text(_dump(updated), encoding="utf-8")
    payload = {
        "pool_entry": claims.POOL_ENTRY,
        "entry_changed": changed,
        "probe_ladder": updated["probe_ladder"],
        "gate_margins": updated["gate_margins"],
        "prose_problems": claims.claim_problems(updated, metrics, bundle),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "claims.json").write_text(
        json.dumps(payload, indent=1, ensure_ascii=False, sort_keys=False) + "\n",
        encoding="utf-8")
    print(f"{'rewrote' if changed else 'unchanged'}: {claims.POOL_ENTRY}")
    print(f"wrote {out_dir / 'claims.json'}")
    return 0 if not payload["prose_problems"] else 1


def check(out_dir: pathlib.Path | None = None) -> int:
    entry = claims.load_json(claims.POOL_ENTRY)
    metrics = claims.load_json(claims.METRICS_ARTIFACT)
    bundle = claims.load_json(claims.PROBE_BUNDLE)
    problems = claims.claim_problems(entry, metrics, bundle)
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "claims.json").write_text(
            json.dumps({
                "pool_entry": claims.POOL_ENTRY,
                "probe_ladder": claims.probe_ladder(bundle),
                "gate_margins": claims.gate_margins(metrics),
                "prose_problems": problems,
            }, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"wrote {out_dir / 'claims.json'}")
    if problems:
        print(f"FAIL: {len(problems)} unbound claim(s)")
        for problem in problems:
            print(f"  - {problem}")
        return 1
    print("PASS: every quoted number re-derives from its artifact")
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--write", action="store_true",
                      help="refresh the entry's derived blocks, then check the prose")
    mode.add_argument("--check", action="store_true",
                      help="verify the committed entry against its artifacts")
    parser.add_argument("--out", default=None,
                        help=f"directory for claims.json (default {DEFAULT_OUT})")
    args = parser.parse_args(argv)
    out_dir = pathlib.Path(args.out) if args.out else DEFAULT_OUT
    return write(out_dir) if args.write else check(out_dir)


if __name__ == "__main__":
    raise SystemExit(main())