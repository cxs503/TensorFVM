"""Run CASE-CAVITY-RE100 (lid-driven cavity, Re = 100) against Ghia et al. (1982).

The case logic lives in :mod:`tensorfvm.case_cavity_re100`; this script only
drives it and writes the artefacts/figures the report is generated from, so the
report can quote machine-produced numbers only.

Examples
--------
Run the three ladder resolutions and rebuild everything::

    PYTHONPATH=src python scripts/case_cavity_re100.py --resolutions 32 64 128

Add one more resolution without discarding the others (each resolution is
written as its own file, so a long run can be resumed)::

    PYTHONPATH=src python scripts/case_cavity_re100.py --resolutions 256 --merge-only-existing

Artefacts (default ``docs/case-cavity-re100``)::

    artifacts/cavity_re100_n{n}.json     one run, scored point by point
    artifacts/case_cavity_re100.json     case-level verdict + convergence orders
    metrics.csv                          the report tables, machine written
    figures/*.png                        profiles, error vs grid, field
"""

from __future__ import annotations

import argparse
import csv
import json
import pathlib
import sys

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from tensorfvm import case_cavity_re100 as case  # noqa: E402

DEFAULT_OUT = REPO / "docs" / "case-cavity-re100"


def _load_existing(art_dir: pathlib.Path) -> dict[int, dict]:
    out: dict[int, dict] = {}
    for path in sorted(art_dir.glob("cavity_re100_n*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        out[int(payload["n"])] = payload
    return out


def _write_csv(path: pathlib.Path, summary: dict) -> None:
    fields = ["n", "h", "steps", "residual", "max_divergence", "wall_seconds",
              "u_max_abs", "u_rms", "v_max_abs", "v_rms", "u_min", "v_max",
              "vortex_x", "vortex_y", "psi_min", "verdict"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for row in summary["rows"]:
            writer.writerow({
                "n": row["n"], "h": f"{row['h']:.6g}", "steps": row["steps"],
                "residual": f"{row['residual']:.6e}",
                "max_divergence": f"{row['max_divergence']:.3e}",
                "wall_seconds": f"{row['wall_seconds']:.1f}",
                "u_max_abs": f"{row['u_max_abs']:.6f}",
                "u_rms": f"{row['u_rms']:.6f}",
                "v_max_abs": f"{row['v_max_abs']:.6f}",
                "v_rms": f"{row['v_rms']:.6f}",
                "u_min": f"{row['u_min']:.6f}",
                "v_max": f"{row['v_max']:.6f}",
                "vortex_x": f"{row['vortex_centre'][0]:.4f}",
                "vortex_y": f"{row['vortex_centre'][1]:.4f}",
                "psi_min": f"{row['psi_min']:.6e}",
                "verdict": row["verdict"],
            })


def _figures(summary: dict, artifacts: list[dict], fig_dir: pathlib.Path) -> list[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    fig_dir.mkdir(parents=True, exist_ok=True)
    written: list[str] = []
    finest = max(artifacts, key=lambda a: a["n"])

    # 1. mid-plane profiles vs the transcribed Ghia points
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    for ax, which, label in ((axes[0], "u", "u/U on x = 0.5"),
                             (axes[1], "v", "v/U on y = 0.5")):
        block = finest["profiles"][which]
        ax.plot(block["reference"], block["coord"], "o", ms=4.5, mfc="none",
                label=f"Ghia et al. (1982) Re=100, {which}")
        order = np.argsort(block["computed"])
        ax.plot([block["computed"][i] for i in order],
                [block["coord"][i] for i in order], "-", lw=1.6,
                label=f"TensorFVM n={finest['n']}")
        ax.axvline(0.0, color="k", lw=0.6, alpha=0.4)
        ax.set_xlabel(label, fontsize=9)
        ax.set_ylabel("x/L" if which == "v" else "y/L", fontsize=9)
        ax.grid(alpha=0.3, ls=":")
        ax.legend(fontsize=7.5, loc="best")
    fig.suptitle("CASE-CAVITY-RE100: mid-plane velocity profiles", fontsize=10)
    fig.tight_layout()
    p = fig_dir / "fig_profiles.png"
    fig.savefig(p, dpi=140)
    plt.close(fig)
    written.append(str(p))

    # 2. error vs grid with the 5 % (pool) and 3 % (paper-grade) bands
    ns = [r["n"] for r in summary["rows"]]
    fig, ax = plt.subplots(figsize=(5.6, 4.0))
    ax.plot(ns, [r["u_max_abs"] for r in summary["rows"]], "o-", label="max|Δu| (Table I)")
    ax.plot(ns, [r["v_max_abs"] for r in summary["rows"]], "s-", label="max|Δv| (Table II)")
    ax.axhline(0.05, color="tab:red", ls="--", lw=1,
               label="case-pool tolerance 5 % of U")
    ax.axhline(0.03, color="tab:orange", ls=":", lw=1, label="paper-grade 3 % of U")
    ax.set_xscale("log", base=2)
    ax.set_xticks(ns)
    ax.set_xticklabels([str(n) for n in ns])
    ax.set_xlabel("grid n (cells per side)")
    ax.set_ylabel("max pointwise deviation / U")
    ax.grid(alpha=0.3, ls=":")
    ax.legend(fontsize=7.5)
    ax.set_title("CASE-CAVITY-RE100: error vs resolution", fontsize=10)
    fig.tight_layout()
    p = fig_dir / "fig_error_vs_n.png"
    fig.savefig(p, dpi=140)
    plt.close(fig)
    written.append(str(p))

    # 3. field: streamfunction contours + velocity vectors
    psi = np.array(finest["field"]["psi"], dtype=float)
    uc = np.array(finest["field"]["u"], dtype=float)
    vc = np.array(finest["field"]["v"], dtype=float)
    g = psi.shape[0]
    xs = (np.arange(g) + 0.5) / g
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    cs = ax.contour(xs, xs, psi, levels=21, cmap="viridis")
    ax.clabel(cs, inline=True, fontsize=6, fmt="%.4f")
    step = max(1, g // 16)
    ax.quiver(xs[::step], xs[::step], uc[::step, ::step], vc[::step, ::step],
              color="0.25", width=0.0025)
    ax.plot(*finest["vortex_centre"]["computed"], "r+", ms=11, mew=1.6,
            label="primary vortex (computed)")
    ax.plot(*finest["vortex_centre"]["reference"], "wx", ms=7, mew=1.4,
            label="Ghia (1982) centre")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.set_xlabel("x/L")
    ax.set_ylabel("y/L")
    ax.legend(fontsize=7, loc="lower left")
    ax.set_title(f"CASE-CAVITY-RE100: ψ contours, n={finest['n']}", fontsize=10)
    fig.tight_layout()
    p = fig_dir / "fig_field.png"
    fig.savefig(p, dpi=140)
    plt.close(fig)
    written.append(str(p))
    return written


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resolutions", type=int, nargs="+", default=[32, 64, 128])
    parser.add_argument("--out", type=pathlib.Path, default=DEFAULT_OUT)
    parser.add_argument("--cfl", type=float, default=0.15)
    parser.add_argument("--tolerance", type=float, default=1e-07)
    parser.add_argument("--max-steps", type=int, default=200000)
    parser.add_argument("--field-grid", type=int, default=32,
                        help="cell fields are block-averaged onto this grid for "
                             "cross-resolution comparison")
    parser.add_argument("--merge-only-existing", action="store_true",
                        help="do not run, only rebuild the case-level documents")
    parser.add_argument("--no-figures", action="store_true")
    args = parser.parse_args(argv)

    art_dir = args.out / "artifacts"
    art_dir.mkdir(parents=True, exist_ok=True)
    artifacts = _load_existing(art_dir)

    if not args.merge_only_existing:
        for n in args.resolutions:
            print(f"[case] running n={n} ...", flush=True)
            payload = case.run_resolution(n, cfl=args.cfl, tolerance=args.tolerance,
                                          max_steps=args.max_steps,
                                          field_grid=args.field_grid)
            case.write_json(art_dir / f"cavity_re100_n{n}.json", payload)
            artifacts[n] = payload
            print(f"[case] n={n}: verdict={payload['verdict']} "
                  f"max|du|={payload['errors']['u_max_abs']:.5f} "
                  f"max|dv|={payload['errors']['v_max_abs']:.5f} "
                  f"steps={payload['solver']['steps']} "
                  f"wall={payload['solver']['wall_seconds']:.1f}s", flush=True)

    if not artifacts:
        print("[case] no artefacts to merge", file=sys.stderr)
        return 2

    rows = [artifacts[n] for n in sorted(artifacts)]
    summary = case.summarise(rows)
    case.write_json(art_dir / "case_cavity_re100.json", summary)
    _write_csv(args.out / "metrics.csv", summary)
    print(f"[case] case verdict={summary['verdict']} "
          f"resolutions={summary['resolutions']}", flush=True)
    if not args.no_figures:
        for path in _figures(summary, rows, args.out / "figures"):
            print(f"[case] figure {path}", flush=True)
    return 0 if summary["finest_verdict"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())