"""CASE-CYLINDER-RE100 case harness: grid ladder, aggregation, referee rows.

The physics lives in :mod:`tensorfvm.verification.cylinder_re100`; this script is
the *case-level* driver that

``ladder``     runs the production grids of the refinement ladder (one job),
``aggregate``  reads the per-grid ``result.json``/``benchmark.json`` artifacts,
               writes the ladder table, the refinement figure and the case
               README (all numbers injected from the artifacts, none typed in),
``referee-sql`` emits the SQL that registers the anchor and the measured rows in
               the platform referee tables (``tensor_benchmark_cases`` /
               ``tensorbench_results``) from the same artifacts.

Usage
  OMP_NUM_THREADS=4 PYTHONPATH=src python scripts/case_cylinder_re100.py ladder \\
      --grids 12 20 32 --dt 0.02 --t-end 120 --root runs/case-cylinder-re100
  PYTHONPATH=src python scripts/case_cylinder_re100.py aggregate \\
      --root runs/case-cylinder-re100 --out docs/case-cylinder-re100
  PYTHONPATH=src python scripts/case_cylinder_re100.py referee-sql \\
      --root runs/case-cylinder-re100 --mission 21018 --job cj3129 --sql-out /tmp/ref.sql
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

ANCHOR = "GAP-DIFFLBM-EXT-LIT-CYL"


def ladder(args) -> int:
    from tensorfvm.verification import cylinder_re100 as case

    root = Path(args.root)
    for cells in args.grids:
        out = root / f"cpd{cells}"
        if (out / "result.json").is_file() and not args.force:
            print(f"[ladder] {out} exists, skipping (use --force to redo)")
            continue
        print(f"[ladder] cells_per_diameter={cells} -> {out}", flush=True)
        spec = case.ImmersedSpec(cells_per_diameter=cells, dt=args.dt,
                                 t_end=args.t_end)
        result, _flow, snapshot = case.run_immersed(spec, log_every=args.log_every)
        errors = case.compute_errors(result)
        case.publish(out, result, errors, snapshot=snapshot)
        print(f"[ladder] {out}: {case.verdict(errors)[1]}", flush=True)
    return 0


def _load_runs(root: Path) -> list[dict]:
    runs = []
    for result_file in sorted(root.glob("cpd*/result.json")):
        result = json.loads(result_file.read_text())
        benchmark_file = result_file.parent / "benchmark.json"
        benchmark = json.loads(benchmark_file.read_text()) if benchmark_file.is_file() else {}
        runs.append(dict(dir=str(result_file.parent), result=result, benchmark=benchmark,
                         cpd=result["grid"]["cells_per_diameter"]))
    runs.sort(key=lambda item: item["cpd"])
    return runs


def aggregate(args) -> int:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from tensorfvm.verification.cylinder_re100 import load_reference, verdict

    root = Path(args.root)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    runs = _load_runs(root)
    if not runs:
        raise SystemExit(f"no runs found under {root}")
    ref = load_reference()
    values, bands = ref["values"], ref["tolerance_band"]

    rows = []
    for item in runs:
        result = item["result"]
        errors = (item["benchmark"] or {}).get("errors", {})
        cd_err = errors.get("cd_mean", {}).get("relative_error")
        st_err = errors.get("st", {}).get("relative_error")
        rows.append(dict(
            cells_per_diameter=item["cpd"], grid=result["grid"],
            steps=result.get("steps"), elapsed_s=result.get("elapsed_s"),
            s_per_step=result.get("s_per_step"),
            cd_mean=result.get("cd_mean"), cd_error=cd_err,
            st=result.get("st"), st_fft=result.get("st_fft"), st_error=st_err,
            cl_amplitude=result.get("cl_amplitude"),
            periods=result.get("statistics_periods"),
            shedding_detected=result.get("shedding_detected"),
            gates=result.get("gate_reasons"),
            verdict=(item["benchmark"] or {}).get("verdict", "unknown"),
            passed=(item["benchmark"] or {}).get("passed", False)))

    deltas = []
    for a, b in zip(rows, rows[1:]):
        entry = dict(between=[a["cells_per_diameter"], b["cells_per_diameter"]])
        for key in ("cd_mean", "st"):
            if a[key] is not None and b[key] is not None:
                entry[f"{key}_delta_pct"] = 100 * (b[key] - a[key]) / b[key]
        deltas.append(entry)

    finest = rows[-1]
    payload = dict(anchor_slug=ANCHOR, reference=values, tolerance_band=bands,
                   citation=ref["citation"], source_url=ref["source_url"],
                   ladder=rows, successive_differences=deltas,
                   finest=finest["cells_per_diameter"],
                   verdict=finest["verdict"], passed=finest["passed"])
    (out / "ladder.json").write_text(json.dumps(payload, indent=2, default=float))

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    cpds = [r["cells_per_diameter"] for r in rows]
    for ax, key, label in ((axes[0], "cd_mean", "Cd_mean"), (axes[1], "st", "St")):
        got = [r[key] for r in rows]
        ax.plot(cpds, got, "o-", label="computed")
        ax.axhline(values[key], color="r", ls="--", label="literature")
        ax.axhspan(values[key] - bands[key], values[key] + bands[key], color="r",
                   alpha=0.12, label="literature band")
        ax.set_xlabel("cells per diameter")
        ax.set_ylabel(label)
        ax.set_title(f"{label} vs mesh resolution (Re = 100)")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "grid_ladder.png", dpi=150)
    plt.close(fig)

    lines = [
        "# CASE-CYLINDER-RE100: unbounded cylinder at Re = 100 (Cd_mean, St)",
        "",
        f"Anchor: `{ANCHOR}` (JOUR-DIFFLBM literature anchor 3, FVM side).",
        f"Reference: Cd_mean = {values['cd_mean']} +/- {bands['cd_mean']}, "
        f"St = {values['st']} +/- {bands['st']} ({ref['citation']}).",
        "",
        "## Method",
        "",
        "`tensorfvm.immersed` masked MAC fractional step (geometric-multigrid",
        "preconditioned Poisson, `projection='masked'`, staircase cylinder mask from",
        "`tensorfvm.immersed.geometry.circular_cylinder`) with uniform freestream",
        "inlet, freestream top/bottom, zero-gradient outlet. The hydrodynamic force",
        "is read from a control-volume momentum budget",
        "(`tensorfvm.immersed.forces`); no surface interpolation of the staircase",
        "body is involved. Statistics: `tensorfvm.verification.cylinder_re100.",
        "shedding_statistics` - shedding gate, then the last 10 shedding periods,",
        "St from band-passed zero crossings and an independent zero-padded FFT",
        "cross-check.",
        "",
        "## Grid ladder (numbers injected from the run artifacts)",
        "",
        "| cells/D | grid | steps | Cd_mean | err | St | err | St (FFT) | Cl amp | periods | verdict |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    def fmt(value, spec="{:.4f}"):
        return "-" if value is None else spec.format(value)

    def pct(value):
        return "-" if value is None else "{:.2f} %".format(100 * value)

    for r in rows:
        lines.append(
            "| {} | {}x{} | {} | {} | {} | {} | {} | {} | {} | {} | {} |".format(
                r["cells_per_diameter"], r["grid"]["nx"], r["grid"]["ny"], r["steps"],
                fmt(r["cd_mean"]), pct(r["cd_error"]), fmt(r["st"]), pct(r["st_error"]),
                fmt(r["st_fft"]), fmt(r["cl_amplitude"]),
                fmt(r["periods"], "{:.0f}"), r["verdict"]))
    lines += ["", "## Successive grid differences", ""]
    if deltas:
        for entry in deltas:
            pieces = [f"{key} {entry[key]:+.2f} %" for key in
                      ("cd_mean_delta_pct", "st_delta_pct") if key in entry]
            lines.append(f"- {entry['between'][0]} -> {entry['between'][1]} cells/D: "
                         + ", ".join(pieces))
    else:
        lines.append("- single grid: no refinement evidence")
    lines += ["", f"Finest grid verdict: **{finest['verdict']}**", "",
              "![grid ladder](grid_ladder.png)", "",
              "## Reproduction", "",
              "```",
              "# one grid",
              "OMP_NUM_THREADS=4 PYTHONPATH=src python -m tensorfvm.verification.cylinder_re100 \\",
              "    --backend immersed --cells-per-diameter 20 --dt 0.02 --t-end 120 \\",
              "    --output runs/case-cylinder-re100/cpd20",
              "# the whole ladder + this document",
              "OMP_NUM_THREADS=4 PYTHONPATH=src python scripts/case_cylinder_re100.py ladder \\",
              "    --grids 12 20 32 --dt 0.02 --t-end 120 --root runs/case-cylinder-re100",
              "PYTHONPATH=src python scripts/case_cylinder_re100.py aggregate \\",
              "    --root runs/case-cylinder-re100 --out docs/case-cylinder-re100",
              "```", ""]
    (out / "README.md").write_text("\n".join(lines))
    print(f"[aggregate] wrote {out/'ladder.json'} and {out/'README.md'}")
    for r in rows:
        print(f"  cpd={r['cells_per_diameter']:>3} Cd={r['cd_mean']} St={r['st']} "
              f"-> {r['verdict']}")
    return 0 if finest["passed"] else 2


def referee_sql(args) -> int:
    """Emit SQL that registers the anchor and every measured ladder row."""
    root = Path(args.root)
    runs = _load_runs(root)
    if not runs:
        raise SystemExit(f"no runs found under {root}")
    reference = runs[0]["benchmark"]["reference"]
    q = lambda text: "'" + str(text).replace("'", "''") + "'"

    statements = [
        "-- referee rows generated by scripts/case_cylinder_re100.py referee-sql",
        f"DELETE FROM tensor_benchmark_cases WHERE anchor_slug = {q(ANCHOR)};",
    ]
    for metric, tolerance in (("cd_mean", "5%"), ("st", "3%")):
        statements.append(
            "INSERT INTO tensor_benchmark_cases "
            "(anchor_slug, case_name, ref_source, metric, tolerance, repo, status) VALUES ("
            f"{q(ANCHOR)}, {q('cylinder-re100-unbounded')}, "
            f"{q(reference['citation'])}, {q(metric)}, {q(tolerance)}, "
            f"{q('TensorFVM')}, {q('pending')});")
    for item in runs:
        result, benchmark = item["result"], item["benchmark"]
        if not benchmark:
            continue
        for index, metric in enumerate(("cd_mean", "st")):
            entry = benchmark["errors"][metric]
            if entry.get("computed") is None:
                continue
            cpd = item["cpd"]
            run_id = "m{}-cpd{}-{}".format(args.mission, cpd, metric)
            statements.append(
                "INSERT INTO tensorbench_results (result_id, mission_id, benchmark_id, "
                "benchmark_version, workload_id, workload_version, metric, unit, "
                "ref_value, got_value, abs_error, rel_error, tolerance_rel, passed, "
                "run_id, implementation_project, implementation_revision, "
                "execution_status, cluster_job_id, source, spec_sha256) VALUES ("
                f"{q(run_id)}, {args.mission}, {q('GAP-DIFFLBM-EXT-LIT-CYL:' + metric)}, "
                f"{q('1')}, {q('cylinder-re100-cpd{}'.format(cpd))}, {q('1')}, "
                f"{q(metric)}, {q('-')}, {entry['reference']!r}, {entry['computed']!r}, "
                f"{abs(entry['computed'] - entry['reference'])!r}, "
                f"{entry['relative_error']!r}, {entry['tolerance_relative']!r}, "
                f"{'true' if entry['passed'] else 'false'}, {q(run_id)}, "
                f"{q('TensorFVM')}, {q(args.revision)}, {q('completed')}, "
                f"{args.job}, {q('case_cylinder_re100.py')}, {q(args.spec_sha)});")
    text = "\n".join(statements) + "\n"
    if args.sql_out:
        Path(args.sql_out).write_text(text)
        print(f"[referee-sql] wrote {args.sql_out} ({len(statements) - 1} statements)")
    else:
        print(text)
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p_lad = sub.add_parser("ladder")
    p_lad.add_argument("--grids", nargs="+", type=int, default=[12, 20, 32])
    p_lad.add_argument("--dt", type=float, default=0.02)
    p_lad.add_argument("--t-end", type=float, default=120.0)
    p_lad.add_argument("--root", default="runs/case-cylinder-re100")
    p_lad.add_argument("--log-every", type=float, default=10.0)
    p_lad.add_argument("--force", action="store_true")
    p_lad.set_defaults(func=ladder)

    p_agg = sub.add_parser("aggregate")
    p_agg.add_argument("--root", default="runs/case-cylinder-re100")
    p_agg.add_argument("--out", default="docs/case-cylinder-re100")
    p_agg.set_defaults(func=aggregate)

    p_sql = sub.add_parser("referee-sql")
    p_sql.add_argument("--root", default="runs/case-cylinder-re100")
    p_sql.add_argument("--mission", type=int, required=True)
    p_sql.add_argument("--job", type=int, default=0)
    p_sql.add_argument("--revision", default="unknown")
    p_sql.add_argument("--spec-sha", default="unknown")
    p_sql.add_argument("--sql-out", default="")
    p_sql.set_defaults(func=referee_sql)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())