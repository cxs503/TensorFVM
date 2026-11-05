"""Mutation survival check for the native_sections3d audit (negative control).

The audit of ``tensorfvm.native_sections3d.cell_section_polygons`` is only
useful if its assertions actually detect a degraded kernel. This harness
applies four semantic mutations to the production source *in memory* (the file
on disk is never modified), reruns the audit probes against each mutant and
reports which assertion family fires.

Run order for the published report::

    python3 scripts/audit_native_sections3d_mutations.py   # writes mutation.json
    python3 scripts/audit_native_sections3d.py             # injects the table
    python3 -m pytest tests/test_native_sections3d.py

Writes ``scripts/native_sections3d/mutation.json``.
"""

import importlib.util
import json
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "scripts" / "native_sections3d"
SOURCE = ROOT / "src" / "tensorfvm" / "native_sections3d.py"

MUTANTS = {
    # the coplanar diagonal cancellation of the docstring
    "coplanar_diagonal_not_cancelled": ("if count%2", "if count"),
    # the hard failure on a non-planar facet cut in more than two points
    "nonplanar_facet_accepted": (
        "elif len(hits)>2:raise ValueError('nonplanar facet plane intersection')",
        "elif False:raise ValueError('nonplanar facet plane intersection')",
    ),
    # the counter-clockwise normalisation of the returned ring
    "sign_flip_removed": (
        "if area<0:polygon=polygon[::-1]",
        "if False:polygon=polygon[::-1]",
    ),
    # the tolerance snap that turns a near-tangent plane into a coplanar one
    "plane_snap_removed": ("d[np.abs(d)<=tolerance]=0.", "pass"),
}


def load_audit():
    spec = importlib.util.spec_from_file_location(
        "native_sections_audit", ROOT / "scripts" / "audit_native_sections3d.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_mutant(source, name):
    module = types.ModuleType(name)
    module.__dict__["__file__"] = str(SOURCE)
    exec(compile(source, "<mutant:{}>".format(name), "exec"), module.__dict__)
    return module


def main():
    source = SOURCE.read_text()
    baseline = load_audit()
    summary = baseline.build_report()["machine"]["summary"]
    results = dict(
        baseline_all_verified=bool(summary["all_verified"]),
        baseline_max_area_error=summary["maximum_relative_area_error"],
        mutants={},
    )
    for name, (old, new) in MUTANTS.items():
        assert old in source, name
        mutant = load_mutant(source.replace(old, new, 1), name)
        audit = load_audit()
        audit.cell_section_polygons = mutant.cell_section_polygons
        entry = {}
        try:
            report = audit.build_report()
            record = report["machine"]["summary"]
            entry.update(
                all_verified=bool(record["all_verified"]),
                max_area_error=record["maximum_relative_area_error"],
                worst_case=record["worst_case"],
                orientation_failures=record["orientation_failures"],
                rejections_accepted=record["rejections_accepted"],
                polygons_expected_mismatch=record["polygons_expected_mismatch"],
                caught=not record["all_verified"],
            )
        except Exception as error:  # a mutant may crash the kernel
            entry.update(
                caught=True, exception="{}: {}".format(type(error).__name__, error)
            )
        results["mutants"][name] = entry
        print(
            "{:32s} caught={} {}".format(
                name, entry["caught"], entry.get("exception", "")
            )
        )
    results["survivors"] = [k for k, v in results["mutants"].items() if not v["caught"]]
    results["kill_rate"] = 1.0 - len(results["survivors"]) / len(MUTANTS)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "mutation.json").write_text(json.dumps(results, indent=1, sort_keys=True))
    print(
        "kill_rate={:.0%} survivors={}".format(
            results["kill_rate"], results["survivors"]
        )
    )
    return results


if __name__ == "__main__":
    main()
