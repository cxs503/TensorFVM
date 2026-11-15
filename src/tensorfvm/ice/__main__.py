"""Command-line entry point for the ice-domain guardrail.

Usage
-----
    python -m tensorfvm.ice                        # benchmark, writes artifacts
    python -m tensorfvm.ice --out docs/ICE-STRUCT-GUARD-REPAIR/artifacts
    python -m tensorfvm.ice --case case.json       # guard a 12-parameter case
    python -m tensorfvm.ice --case case.json --report
    python -m tensorfvm.ice --spec                 # dump the guarded table

Exit status is 0 when every graded benchmark row passes and, in ``--case``
mode, when the case is admissible; 1 otherwise.  This makes the module usable
as a gate from a shell pipeline.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from .benchmark import REPRO_CMD, run_benchmark, write_artifacts
from .guardrails import get_parameter_report, validate_ice_parameters
from .load import guarded_level_ice_resistance, known_parameters
from .parameters import ICE_RESISTANCE_INPUTS


def _load_case(path: Path) -> dict[str, float]:
    payload: dict[str, Any] = json.loads(path.read_text())
    # Accept either a flat 12-parameter mapping or {"ice": {...}, "ship": {...}}
    flat: dict[str, float] = {}
    for key in ("ice", "ship"):
        block = payload.get(key)
        if isinstance(block, dict):
            flat.update({k: v for k, v in block.items()})
    if not flat:
        flat = {k: v for k, v in payload.items() if k in ICE_RESISTANCE_INPUTS}
    return flat


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m tensorfvm.ice",
        description="Ice-domain parameter guardrail: validate, clamp, benchmark.",
    )
    parser.add_argument("--out", default="docs/ICE-STRUCT-GUARD-REPAIR/artifacts",
                        help="artifact directory (default: docs/ICE-STRUCT-GUARD-REPAIR/artifacts)")
    parser.add_argument("--case", type=Path, default=None,
                        help="JSON case file with the 12-parameter input set")
    parser.add_argument("--report", action="store_true",
                        help="print the human-readable parameter status report")
    parser.add_argument("--spec", action="store_true",
                        help="dump the guarded parameter table as JSON")
    parser.add_argument("--mode", choices=("reject", "clamp"), default="reject",
                        help="guarding mode for --case (default: reject)")
    parser.add_argument("--bow-angle-deg", type=float, default=30.0,
                        help="bow angle from horizontal for the resistance formula")
    args = parser.parse_args(argv)

    if args.spec:
        print(json.dumps(known_parameters(), indent=2))
        return 0

    if args.case is not None:
        params = _load_case(args.case)
        res = guarded_level_ice_resistance(
            params, mode=args.mode, bow_angle_deg=args.bow_angle_deg
        )
        verdict = validate_ice_parameters(params)
        payload = {
            "case": str(args.case),
            "params": params,
            "guard_status": verdict.status.value,
            "is_valid": verdict.is_valid,
            "errors": verdict.errors,
            "warnings": verdict.warnings,
            "resistance_kN": res.resistance_kN,
            "unguarded_resistance_kN": res.unguarded_resistance_kN,
            "components_kN": res.components_kN,
            "mode": args.mode,
        }
        if args.report:
            payload["parameter_report"] = get_parameter_report(params)
        print(json.dumps(payload, indent=2))
        return 0 if verdict.is_valid else 1

    if args.report:
        print(get_parameter_report({}))
        bundle = run_benchmark()
        print(f"\nrepro: {REPRO_CMD}")
        return 0 if bundle["all_graded_rows_pass"] else 1

    summary = write_artifacts(args.out)
    print(json.dumps(summary, indent=2))
    return 0 if summary["all_graded_rows_pass"] else 1


if __name__ == "__main__":      # pragma: no cover
    sys.exit(main())