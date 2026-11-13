"""Reproduce and gate the ``CASE-FVM-CHANNEL-KMM180`` artifact set.

The case logic lives in :mod:`tensorfvm.benchmark_channel_kmm1987` and
:mod:`tensorfvm.verification.kmm180_artifacts`; this script is the operator
entry point.  It either

* **regenerates** the artifacts from the solver and then gates them
  (``--run``, the default), or
* **gates the committed artifacts only** (``--check``, no solve, seconds).

The gate is the part that keeps the case honest over time.  It cross-checks

1. ``metrics.csv`` / ``profiles.csv`` / ``stations.csv`` against
   ``benchmark.json`` (same run, same numbers);
2. every graded verdict against the declared tolerance (a PASS with a
   deviation above the tolerance is a hard failure, not a warning);
3. every recorded ``sha256`` in ``manifest.json`` against the file it names, so
   a hand edit of an artifact cannot survive;
4. ``docs/oss_solutions/CASE-FVM-CHANNEL-KMM180.json`` against the artifacts:
   the catalog is computed at the *reference* Re_tau = 182.088 while this
   benchmark solves at the nominal KMM1987 Re_tau = 180, so the catalog numbers
   are compared with an explicit substitution band (``CATALOG_RMS_BAND``),
   measured rather than assumed (see the case report).

Examples
--------
Gate the committed artifacts (fast, no solve)::

    PYTHONPATH=src python scripts/run_case_fvm_channel_kmm180.py --check

Regenerate the full 3 models x 3 grids matrix, rewrite the artifacts, then
gate them::

    PYTHONPATH=src python scripts/run_case_fvm_channel_kmm180.py --run

Artifacts (default ``docs/case-fvm-channel-kmm180``)::

    benchmark.json   every graded number per case, both grading windows
    metrics.csv      one row per (case, station, metric)
    profiles.csv     per-cell y+, model u+, DNS u+, relative error
    stations.csv     the grade stations and their verdicts
    results.json     the legacy matrix (unchanged by this script)
    results.csv      the legacy matrix, CSV
    REPORT.md        prose report rendered from the same numbers
    manifest.json    implementation files, repro command, artifact sha256
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import pathlib
import sys

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

CASE_DIR = REPO / "docs" / "case-fvm-channel-kmm180"
CATALOG = REPO / "docs" / "oss_solutions" / "CASE-FVM-CHANNEL-KMM180.json"

REQUIRED = (
    "benchmark.json",
    "metrics.csv",
    "profiles.csv",
    "stations.csv",
    "results.json",
    "results.csv",
    "REPORT.md",
    "manifest.json",
    "fig_velocity_profiles.png",
    "fig_log_region_error.png",
)

#: Numbers quoted by the catalog were produced at the *reference* Re_tau =
#: 182.088 (== the DNS), the benchmark solves at the nominal KMM1987 Re_tau =
#: 180.  The band below is above the measured substitution cost (recorded per
#: run in ``benchmark.json`` -> ``reference_substitution``: max 0.00080 on the
#: RMS, i.e. 0.080 pp) and far below a real regression, which moves these
#: numbers by >= 0.01.
CATALOG_RMS_BAND = 0.005
CATALOG_ERROR_BAND = 0.01
TOL = 1e-9


def _load_json(path):
    return json.loads(pathlib.Path(path).read_text())


def _rows(path):
    with pathlib.Path(path).open(newline="") as handle:
        return list(csv.DictReader(handle))


def _f(value):
    return float(value)


def check_artifacts(case_dir=CASE_DIR, catalog_path=CATALOG):
    """Return the list of problems found in the artifact set (empty == clean)."""
    case_dir = pathlib.Path(case_dir)
    problems = []

    missing = [name for name in REQUIRED if not (case_dir / name).is_file()]
    if missing:
        return [f"missing artifact(s) in {case_dir}: {', '.join(missing)}"]

    benchmark = _load_json(case_dir / "benchmark.json")
    manifest = _load_json(case_dir / "manifest.json")
    metrics = _rows(case_dir / "metrics.csv")
    profiles = _rows(case_dir / "profiles.csv")
    stations = _rows(case_dir / "stations.csv")

    if benchmark.get("schema") != "tensorfvm.case-fvm-channel-kmm180/1":
        problems.append(f"unexpected benchmark schema {benchmark.get('schema')!r}")
    if manifest.get("case") != benchmark.get("anchor"):
        problems.append("manifest case does not match benchmark anchor")

    cases = {case["case"]: case for case in benchmark.get("cases", [])}
    if benchmark.get("cases_total") != len(cases):
        problems.append("benchmark cases_total does not match the case list")
    if len(cases) != len({(c["model"], c["n"]) for c in cases.values()}):
        problems.append("duplicate (model, n) records in benchmark.json")
    tolerance = benchmark.get("tolerance")
    if tolerance is None:
        problems.append("benchmark.json carries no tolerance")

    # -- 2. verdicts must follow the declared tolerance ---------------------- #
    for key, case in cases.items():
        if not case.get("converged"):
            continue
        for station in ("legacy_full", "log_layer"):
            window = case.get(station)
            if window is None:
                problems.append(f"{key}: benchmark.json has no {station} window")
                continue
            if not window.get("graded"):
                problems.append(f"{key}/{station}: window is not graded ({window})")
                continue
            expected = "PASS" if window["rms_relative"] < tolerance else "FAIL"
            if window["verdict"] != expected:
                problems.append(
                    f"{key}/{station}: verdict {window['verdict']} contradicts "
                    f"RMS {window['rms_relative']:.6g} against tolerance {tolerance}")
            if window["cells"] < 3:
                problems.append(f"{key}/{station}: window holds {window['cells']} cells")

    # -- 1a. metrics.csv vs benchmark.json ---------------------------------- #
    metric_cases = {}
    for row in metrics:
        metric_cases.setdefault(row["case"], []).append(row)
    if set(metric_cases) != set(cases):
        problems.append(
            "metrics.csv cases and benchmark.json cases differ: "
            f"{sorted(set(metric_cases) ^ set(cases))}")
    for key, rows in metric_cases.items():
        case = cases.get(key)
        if case is None or not case.get("converged"):
            continue
        by_name = {(r["station"], r["metric"]): r for r in rows}
        for station in ("legacy_full", "log_layer"):
            for metric, field in (("log_rms_relative", "rms_relative"),
                                  ("log_max_relative", "max_relative")):
                row = by_name.get((station, metric))
                if row is None:
                    problems.append(f"{key}: metrics.csv lacks {station}/{metric}")
                    continue
                if abs(_f(row["computed"]) - case[station][field]) > TOL:
                    problems.append(
                        f"{key}: metrics.csv {station}/{metric} disagrees with "
                        "benchmark.json")
                if row["graded"] == "true":
                    expected = "PASS" if _f(row["computed"]) < _f(row["tolerance"]) else "FAIL"
                    if row["verdict"] != expected:
                        problems.append(f"{key}: metrics.csv verdict {row['verdict']} "
                                        f"contradicts {station}/{metric}")
        cf_row = by_name.get(("", "Cf"))
        if cf_row is None or abs(_f(cf_row["computed"]) - case["Cf"]) > TOL:
            problems.append(f"{key}: metrics.csv Cf row missing or inconsistent")

    # -- 1b. profiles.csv vs benchmark.json -------------------------------- #
    profile_cases = {}
    for row in profiles:
        profile_cases.setdefault(row["case"], []).append(row)
    if set(profile_cases) != {k for k, c in cases.items() if c.get("converged")}:
        problems.append("profiles.csv cases and converged benchmark cases differ")
    for key, rows in profile_cases.items():
        case = cases.get(key)
        if case is None:
            continue
        if len(rows) != case.get("cells"):
            problems.append(f"{key}: profiles.csv has {len(rows)} rows, "
                            f"benchmark.json records cells={case.get('cells')}")
        y_plus = [_f(r["y_plus"]) for r in rows]
        if any(b <= a for a, b in zip(y_plus, y_plus[1:])):
            problems.append(f"{key}: profiles.csv y+ is not strictly increasing")
        for row in rows[:3] + rows[-3:]:
            u, dns = _f(row["u_plus"]), _f(row["u_plus_dns"])
            if abs((u - dns) / dns - _f(row["error_relative"])) > 1e-7:
                problems.append(f"{key}: profiles.csv error_relative is inconsistent "
                                f"at y+={row['y_plus']}")

    # -- 1c. stations.csv vs benchmark.json -------------------------------- #
    station_cases = {}
    for row in stations:
        station_cases.setdefault(row["case"], []).append(row)
    for key, rows in station_cases.items():
        case = cases.get(key)
        if case is None:
            continue
        if sorted(r["station"] for r in rows) != ["legacy_full", "log_layer"]:
            problems.append(f"{key}: stations.csv must hold legacy_full and log_layer")
        for row in rows:
            window = case.get(row["station"])
            if window is None:
                continue
            if int(row["cells"]) != window["cells"]:
                problems.append(f"{key}/{row['station']}: stations.csv cell count "
                                "disagrees with benchmark.json")
            value = row["rms_relative"]
            if value and abs(_f(value) - window["rms_relative"]) > TOL:
                problems.append(f"{key}/{row['station']}: stations.csv RMS disagrees "
                                "with benchmark.json")
            if row["verdict"] != window["verdict"]:
                problems.append(f"{key}/{row['station']}: stations.csv verdict "
                                "disagrees with benchmark.json")

    # -- 2b. reference substitution must be measured and inside its band ---- #
    substitution = benchmark.get("reference_substitution")
    if substitution is None:
        problems.append("benchmark.json carries no reference_substitution block")
    else:
        if not substitution.get("cases"):
            problems.append("reference_substitution holds no measured case")
        for row in substitution.get("cases", []):
            for nominal, reference, delta in (
                ("rms_legacy_nominal", "rms_legacy_reference", "delta_pp_legacy"),
                ("rms_log_layer_nominal", "rms_log_layer_reference",
                 "delta_pp_log_layer"),
            ):
                expected = abs(row[nominal] - row[reference]) * 100.0
                if abs(expected - row[delta]) > 1e-6:
                    problems.append(f"{row['case']}: substitution {delta} is not "
                                    "the measured delta")
                if row[nominal] < 0 or row[reference] < 0:
                    problems.append(f"{row['case']}: negative RMS")
        if substitution.get("verdict_flips"):
            problems.append("reference substitution flips a verdict: "
                            f"{substitution['verdict_flips']}")
        if substitution.get("max_delta_pp", 0.0) > substitution.get("band_pp", 0.0):
            problems.append(
                f"reference substitution delta {substitution['max_delta_pp']} pp "
                f"exceeds the declared band {substitution['band_pp']} pp")
        if substitution.get("passed") is not True:
            problems.append("reference substitution did not pass its own band")

    # -- 3. manifest sha256 ------------------------------------------------- #
    for name, digest in (manifest.get("sha256") or {}).items():
        path = case_dir / name
        if not path.is_file():
            problems.append(f"manifest hashes a file that does not exist: {name}")
            continue
        if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            problems.append(f"{name} does not match its manifest sha256")
    for name in (manifest.get("artifacts") or []):
        if not (case_dir / name).is_file():
            problems.append(f"manifest lists a missing artifact: {name}")

    # -- 4. catalog cross-check --------------------------------------------- #
    catalog_path = pathlib.Path(catalog_path)
    if catalog_path.is_file():
        catalog = _load_json(catalog_path)
        if catalog.get("tolerance") != tolerance:
            problems.append("catalog tolerance differs from the benchmark tolerance")
        for name in catalog.get("artifacts", []):
            if not (REPO / name).is_file():
                problems.append(f"catalog lists a missing artifact: {name}")
        for key, quoted in (catalog.get("key_results") or {}).items():
            model, _, cells = key.rpartition("_n")
            record = cases.get(f"{model}_n{cells}")
            if record is None:
                problems.append(f"catalog quotes a case the benchmark does not run: {key}")
                continue
            for quoted_name, station, field in (
                ("log_rms_legacy", "legacy_full", "rms_relative"),
                ("log_rms_log_layer", "log_layer", "rms_relative"),
            ):
                if quoted_name not in quoted:
                    continue
                delta = abs(quoted[quoted_name] - record[station][field])
                if delta > CATALOG_RMS_BAND:
                    problems.append(
                        f"catalog {key}.{quoted_name}={quoted[quoted_name]} vs "
                        f"artifact {record[station][field]:.6g} (delta {delta:.4g} "
                        f"> band {CATALOG_RMS_BAND})")
            for quoted_name, field in (("cf_error", "cf_relative_error"),
                                       ("bulk_u_plus_error", "bulk_relative_error")):
                if quoted_name not in quoted:
                    continue
                delta = abs(quoted[quoted_name] - record[field])
                if delta > CATALOG_ERROR_BAND:
                    problems.append(
                        f"catalog {key}.{quoted_name}={quoted[quoted_name]} vs "
                        f"artifact {record[field]:.6g} (delta {delta:.4g} "
                        f"> band {CATALOG_ERROR_BAND})")
        finest = max(c["n"] for c in cases.values())
        sst = cases.get(f"sst_n{finest}")
        windows = catalog.get("grading_windows") or {}
        if sst is not None:
            for station in ("legacy_full", "log_layer"):
                quoted = (windows.get(station) or {}).get("sst_rms_relative")
                if quoted is None:
                    problems.append(f"catalog grading_windows lacks {station}")
                    continue
                if abs(quoted - sst[station]["rms_relative"]) > CATALOG_RMS_BAND:
                    problems.append(
                        f"catalog grading_windows.{station}.sst_rms_relative={quoted} "
                        f"vs artifact {sst[station]['rms_relative']:.6g}")
                expected = sst[station]["verdict"].lower()
                if (windows[station].get("sst_verdict") or "").lower() != expected:
                    problems.append(
                        f"catalog grading_windows.{station}.sst_verdict disagrees "
                        f"with the artifact ({expected.upper()})")
    else:
        problems.append(f"catalog not found: {catalog_path}")

    return problems


def summarise(case_dir=CASE_DIR):
    """One screen of headline numbers straight from the artifacts."""
    case_dir = pathlib.Path(case_dir)
    benchmark = _load_json(case_dir / "benchmark.json")
    lines = [f"{benchmark['anchor']}  tolerance={benchmark['tolerance']:.0%}"]
    lines.append(f"{'case':14s} {'Cf err':>8s} {'Ub+ err':>8s} "
                 f"{'rms full':>9s} {'rms log':>9s}  verdict(full/log)")
    for case in benchmark["cases"]:
        if not case.get("converged"):
            lines.append(f"{case['case']:14s} {'-':>8s} {'-':>8s} {'-':>9s} {'-':>9s}  FAIL/FAIL")
            continue
        lines.append(
            f"{case['case']:14s} {case['cf_relative_error']:8.2%} "
            f"{case['bulk_relative_error']:8.2%} "
            f"{case['legacy_full']['rms_relative']:9.2%} "
            f"{case['log_layer']['rms_relative']:9.2%}  "
            f"{case['legacy_full']['verdict']}/{case['log_layer']['verdict']}")
    lines.append(benchmark["headline"])
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=pathlib.Path, default=CASE_DIR,
                        help="artifact directory (default: docs/case-fvm-channel-kmm180)")
    parser.add_argument("--catalog", type=pathlib.Path, default=CATALOG)
    parser.add_argument("--check", action="store_true",
                        help="gate the committed artifacts without re-running the solver")
    parser.add_argument("--quick", action="store_true",
                        help="regenerate with the single SST n=128 case (smoke run)")
    args = parser.parse_args(argv)

    if not args.check:
        from tensorfvm.benchmark_channel_kmm1987 import main as run_benchmark

        run_argv = ["--out", str(args.out)] + (["--quick"] if args.quick else [])
        # The benchmark exits non-zero when the anchor verdict is FAIL; that is
        # a scientific result, not a script error, so the artifact gate below
        # decides the exit code.
        run_benchmark(run_argv)

    print(summarise(args.out))
    problems = check_artifacts(args.out, args.catalog)
    if problems:
        print(f"\nARTIFACT GATE: {len(problems)} problem(s)")
        for problem in problems:
            print(f"  - {problem}")
        return 1
    print("\nARTIFACT GATE: clean "
          "(benchmark/metrics/profiles/stations/manifest/catalog consistent)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())