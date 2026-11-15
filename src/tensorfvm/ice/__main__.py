"""CLI entry point for ice-domain guardrail validation.

Usage
-----
# Validate parameters from command line
tensorfvm-ice-guard --thickness 1.5 --strength 3.0 --friction 0.1

# Validate from JSON file
tensorfvm-ice-guard --input params.json

# Clamp extreme values
tensorfvm-ice-guard --thickness 50.0 --clamp

# Generate full report
tensorfvm-ice-guard --thickness 50.0 --strength 100.0 --report

# Run demo with all test cases
tensorfvm-ice-guard --demo --output artifacts/ice_guardrails
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

from .guardrails import (
    clamp_ice_parameters,
    get_parameter_report,
    validate_ice_parameters,
)
from .parameters import ALL_ICE_PARAMETERS


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tensorfvm-ice-guard",
        description="Ice-domain physical guardrail validation for ship-ice interaction parameters.",
    )
    parser.add_argument(
        "--input", type=Path, default=None,
        help="JSON file with ice parameters to validate",
    )
    parser.add_argument("--thickness", type=float, default=None, help="Ice thickness h (m)")
    parser.add_argument("--strength", type=float, default=None, help="Compressive strength σ_c (MPa)")
    parser.add_argument("--friction", type=float, default=None, help="Friction coefficient μ")
    parser.add_argument("--speed", type=float, default=None, help="Ship speed v (m/s)")
    parser.add_argument("--density", type=float, default=None, help="Ice density ρ_ice (kg/m³)")
    parser.add_argument("--elastic-modulus", type=float, default=None, help="Elastic modulus E (GPa)")
    parser.add_argument("--poisson-ratio", type=float, default=None, help="Poisson's ratio ν")
    parser.add_argument("--temperature", type=float, default=None, help="Ice temperature T (°C)")
    parser.add_argument(
        "--clamp", action="store_true",
        help="Clamp out-of-bounds values instead of rejecting",
    )
    parser.add_argument(
        "--allow-extrapolation", action="store_true",
        help="Accept parameters outside calibration domain (with warnings)",
    )
    parser.add_argument(
        "--report", action="store_true",
        help="Generate human-readable parameter report",
    )
    parser.add_argument(
        "--output", type=Path, default=None,
        help="Output directory for artifacts (implies --demo if no input given)",
    )
    parser.add_argument(
        "--demo", action="store_true",
        help="Run demonstration with predefined test cases",
    )
    parser.add_argument(
        "--json", action="store_true",
        help="Output results as JSON",
    )
    return parser


def collect_params(args: argparse.Namespace) -> dict[str, float]:
    """Collect ice parameters from command-line arguments."""
    params = {}
    mapping = {
        "thickness": "ice_thickness",
        "strength": "ice_strength",
        "friction": "ice_friction",
        "speed": "ice_speed",
        "density": "ice_density",
        "elastic_modulus": "ice_elastic_modulus",
        "poisson_ratio": "ice_poisson_ratio",
        "temperature": "ice_temperature",
    }
    for cli_name, param_name in mapping.items():
        value = getattr(args, cli_name.replace("-", "_"), None)
        if value is not None:
            params[param_name] = value
    return params


def run_demo(output_dir: Path) -> dict:
    """Run demonstration with predefined test cases and generate artifacts."""
    output_dir.mkdir(parents=True, exist_ok=True)

    demo_cases = [
        {
            "name": "typical_arctic",
            "params": {"ice_thickness": 1.5, "ice_strength": 3.0, "ice_friction": 0.1, "ice_speed": 2.5},
            "description": "Typical Arctic shipping conditions",
        },
        {
            "name": "impossible_thickness",
            "params": {"ice_thickness": 50.0},
            "description": "Physically impossible ice thickness (h=50 m)",
        },
        {
            "name": "impossible_strength",
            "params": {"ice_strength": 100.0},
            "description": "Unrealistic compressive strength (σ_c=100 MPa)",
        },
        {
            "name": "extrapolation_risk",
            "params": {"ice_thickness": 6.0, "ice_strength": 2.0},
            "description": "Outside calibration domain (h=6 m)",
        },
        {
            "name": "combined_extremes",
            "params": {"ice_thickness": 50.0, "ice_strength": 100.0, "ice_speed": 20.0, "ice_friction": 0.8},
            "description": "Multiple physically impossible values",
        },
    ]

    results = []
    for case in demo_cases:
        validation = validate_ice_parameters(case["params"])
        clamping = clamp_ice_parameters(case["params"])
        results.append({
            "name": case["name"],
            "description": case["description"],
            "input": case["params"],
            "validation": {
                "is_valid": validation.is_valid,
                "status": validation.status.value,
                "errors": validation.errors,
                "warnings": validation.warnings,
                "extrapolation_risk": validation.extrapolation_risk,
            },
            "clamping": {
                "clamped_params": clamping.clamped_params,
                "warnings": clamping.warnings,
            },
        })

    # Save results
    json_path = output_dir / "demo_results.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # Generate parameter report for the failure case
    failure_params = {"ice_thickness": 50.0, "ice_strength": 100.0, "ice_friction": 0.8, "ice_speed": 10.0}
    report = get_parameter_report(failure_params)
    report_path = output_dir / "failure_case_report.txt"
    with open(report_path, "w") as f:
        f.write(report)

    # Generate bounds summary
    bounds_summary = {}
    for name, param in ALL_ICE_PARAMETERS.items():
        b = param.bounds
        bounds_summary[name] = {
            "symbol": b.symbol,
            "unit": b.unit,
            "hard_bounds": [b.hard_min, b.hard_max],
            "soft_bounds": [b.soft_min, b.soft_max],
            "typical_range": [b.typical_min, b.typical_max],
            "reference": b.reference,
        }
    bounds_path = output_dir / "parameter_bounds.json"
    with open(bounds_path, "w") as f:
        json.dump(bounds_summary, f, indent=2, ensure_ascii=False)

    return {
        "demo_cases": len(demo_cases),
        "artifacts": {
            "demo_results": str(json_path),
            "failure_report": str(report_path),
            "parameter_bounds": str(bounds_path),
        },
    }


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    # Demo mode
    if args.demo or args.output:
        output_dir = args.output or Path("artifacts/ice_guardrails")
        result = run_demo(output_dir)
        if args.json:
            print(json.dumps(result, indent=2, ensure_ascii=False))
        else:
            print(f"Demo completed. {result['demo_cases']} test cases processed.")
            print(f"Artifacts saved to: {output_dir}")
            for name, path in result["artifacts"].items():
                print(f"  {name}: {path}")
        return 0

    # Load from file
    if args.input:
        with open(args.input) as f:
            params = json.load(f)
    else:
        params = collect_params(args)

    if not params:
        parser.error("No parameters provided. Use --input, --demo, or individual parameter flags.")

    # Clamp mode
    if args.clamp:
        result = clamp_ice_parameters(params)
        if args.json:
            output = {
                "mode": "clamp",
                "input": params,
                "clamped_params": result.clamped_params,
                "warnings": result.warnings,
                "errors": result.errors,
            }
            print(json.dumps(output, indent=2, ensure_ascii=False))
        else:
            print("Clamped parameters:")
            for name, value in result.clamped_params.items():
                print(f"  {name}: {value}")
            if result.warnings:
                print("\nWarnings:")
                for w in result.warnings:
                    print(f"  ⚠ {w}")
        return 0 if result.is_valid else 1

    # Validation mode
    result = validate_ice_parameters(params, allow_extrapolation=args.allow_extrapolation)

    if args.json:
        output = {
            "mode": "validate",
            "input": params,
            "is_valid": result.is_valid,
            "status": result.status.value,
            "errors": result.errors,
            "warnings": result.warnings,
            "extrapolation_risk": result.extrapolation_risk,
            "validated_params": result.validated_params,
        }
        print(json.dumps(output, indent=2, ensure_ascii=False))
    elif args.report:
        print(get_parameter_report(params))
    else:
        if result.is_valid:
            print("✅ PASS: All parameters within acceptable bounds")
            if result.warnings:
                print("\nWarnings:")
                for w in result.warnings:
                    print(f"  ⚠ {w}")
        else:
            print("❌ FAIL: Parameter validation failed")
            print("\nErrors:")
            for e in result.errors:
                print(f"  ✗ {e}")

    return 0 if result.is_valid else 1


if __name__ == "__main__":
    raise SystemExit(main())
