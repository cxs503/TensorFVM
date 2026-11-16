"""Produce the numerical-configuration envelope artifacts for the dam-break pool.

Usage::

    PYTHONPATH=src python scripts/run_dambreak_config_envelope.py

Writes ``docs/case-fvm-dambreak-stoker-config-envelope/``:

``metrics.json``               every row of the study (numbers only, no prose)
``SUMMARY.md``                 tables generated from ``metrics.json``
``proposed_pool_amendment.json`` the configuration declaration the entry lacks
``fig_*.png``                   figures

No number in ``SUMMARY.md`` or in the report is typed by hand: both are
rendered from the same payload, which is produced by the study itself.
"""

from __future__ import annotations

import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tensorfvm.shallow_water.dam_break import (  # noqa: E402
    RITTER_DRY,
    STOKER_WET,
    run_case,
)
from tensorfvm.verification.dambreak_config_envelope import (  # noqa: E402
    BASELINE,
    FLOOR_POINTS,
    run_point,
    study,
)

OUT = ROOT / "docs" / "case-fvm-dambreak-stoker-config-envelope"


def _fmt(value, digits: int = 4) -> str:
    if value is None:
        return "—"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        if not math.isfinite(value):
            return "nan"
        if value != 0.0 and (abs(value) < 1e-3 or abs(value) >= 1e4):
            return f"{value:.3e}"
        return f"{value:.{digits}g}"
    return str(value)


def _table(rows: list[dict], columns: list[str], header: list[str]) -> str:
    lines = ["| " + " | ".join(header) + " |",
             "|" + "|".join("---" for _ in header) + "|"]
    for row in rows:
        cells = []
        for key in columns:
            value = row
            for part in key.split("."):
                value = (value or {}).get(part) if isinstance(value, dict) else None
            cells.append(_fmt(value))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                       cwd=ROOT, text=True).strip()
    except Exception:  # noqa: BLE001
        return "unknown"


# --------------------------------------------------------------------------- #
# figures
# --------------------------------------------------------------------------- #
def figures(payload: dict, dirpath: Path) -> dict:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    paths: dict = {}
    tol = 0.05

    # --- 1: CFL axis -------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.8))
    for ax, (case, title) in zip(axes[:2], (("ritter-dry", "Ritter (dry bed)"),
                                            ("stoker-wet", "Stoker (wet bed)"))):
        rows = payload["axes"][case]["cfl"]["rows"]
        cfl = [r["cfl"] for r in rows]
        headline = ("front_fan<5%" if case == "ritter-dry"
                    else "shock_position<5%")
        secondary = ("front_wetted<5%" if case == "ritter-dry"
                     else "step_ratio<5%")
        for key, marker in ((headline, "o-"), ("front_series<5%", "s--"),
                            (secondary, "^:")):
            values = [r.get(key) for r in rows]
            if any(v is not None for v in values):
                ax.plot(cfl, [np.nan if v is None else v for v in values],
                        marker, label=key)
        ax.axhline(tol, color="k", lw=0.8, ls="-", alpha=0.6, label="5 % band")
        ax.set_xlabel("CFL")
        ax.set_ylabel("relative error")
        ax.set_yscale("log")
        ax.set_title(f"{title}: pool gates vs CFL (n = {rows[0]['n']})")
        ax.legend(fontsize=6)
        ax.grid(alpha=0.3)
    rows = payload["axes"]["ritter-dry"]["cfl"]["rows"]
    ax = axes[2]
    ax.plot([r["cfl"] for r in rows], [r["steps"] for r in rows], "o-",
            label="Ritter steps")
    rows_w = payload["axes"]["stoker-wet"]["cfl"]["rows"]
    ax.plot([r["cfl"] for r in rows_w], [r["steps"] for r in rows_w], "s-",
            label="Stoker steps")
    ax.set_xlabel("CFL")
    ax.set_ylabel("accepted steps to t* = 2")
    ax.set_yscale("log")
    ax.set_title("cost of the CFL band")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    paths["fig_cfl_axis"] = dirpath / "fig_cfl_axis.png"
    fig.savefig(paths["fig_cfl_axis"], dpi=140)
    plt.close(fig)

    # --- 2: floor axis -----------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 3.9))
    rows = payload["axes"]["ritter-dry"]["floor"]["rows"]
    floors = [r["dry_depth"] for r in rows]
    ax = axes[0]
    for key, style in (("front_wetted<5%", "o-"), ("front_fan<5%", "s-"),
                       ("depth_l1<5%", "^-"), ("front_series<5%", "v--")):
        ax.plot(floors, [np.nan if r.get(key) is None else r[key] for r in rows],
                style, label=key)
    ax.axhline(tol, color="k", lw=0.8, alpha=0.6, label="5 % band")
    certified = payload["axes"]["ritter-dry"]["floor"]["envelope"]["certified"]
    ax.axvspan(min(certified), max(certified), color="tab:green", alpha=0.12,
               label="certified band")
    ax.axvline(BASELINE["dry_depth"], color="tab:red", lw=0.9, ls=":",
               label="shipped default")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("dry depth floor (absolute, m)")
    ax.set_ylabel("relative error")
    ax.set_title("Ritter: every pool gate vs the depth floor")
    ax.legend(fontsize=6)
    ax.grid(alpha=0.3)
    ax = axes[1]
    hidden = [r.get("hidden_mass_fraction") for r in rows]
    ax.plot(floors, [np.nan if v is None else v for v in hidden], "o-",
            label="hidden mass fraction (rejected water)")
    recess = [None if not r.get("reading") else r["reading"].get("front_recession")
              for r in rows]
    ax.plot(floors, [np.nan if v is None else v for v in recess], "s-",
            label="wetted-edge recession (m)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("dry depth floor (absolute, m)")
    ax.set_ylabel("hidden mass fraction / recession [m]")
    ax.set_title("what the floor removes")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    paths["fig_floor_axis"] = dirpath / "fig_floor_axis.png"
    fig.savefig(paths["fig_floor_axis"], dpi=140)
    plt.close(fig)

    # --- 3: instrument calibration + certification matrix -------------------
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 3.7))
    ax = axes[0]
    inst = payload["instrument"]
    keys = sorted(inst, key=float)
    ax.plot([float(k) for k in keys],
            [inst[k]["relative_difference"] for k in keys], "o-",
            label="hidden-mass reading vs quadrature")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("dry depth floor (m)")
    ax.set_ylabel("relative difference")
    ax.set_title("instrument validity domain (n = 800)")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)
    ax = axes[1]
    gates = payload["axes"]["ritter-dry"]["floor"]["envelope"]["gates"]
    names = sorted(gates)
    grid = np.array([[1.0 if p else 0.0 for p in gates[g]["pass"]] for g in names])
    ax.imshow(grid, aspect="auto", cmap="RdYlGn", vmin=0.0, vmax=1.0)
    ax.set_xticks(range(len(floors)))
    ax.set_xticklabels([_fmt(f) for f in floors], rotation=45, fontsize=7)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=7)
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            ax.text(j, i, "P" if grid[i, j] else "F", ha="center", va="center",
                    fontsize=7)
    ax.set_title("Ritter: gate x floor pass matrix (n = 800)")
    fig.tight_layout()
    paths["fig_certification_matrix"] = dirpath / "fig_certification_matrix.png"
    fig.savefig(paths["fig_certification_matrix"], dpi=140)
    plt.close(fig)

    # --- 4: front zoom -----------------------------------------------------
    fig, ax = plt.subplots(figsize=(6.8, 3.9))
    n_zoom = 800
    for floor, style in ((1e-12, "k-"), (1e-3, "b--"), (5e-2, "r:")):
        run = run_case(RITTER_DRY, n=n_zoom, dry_depth=floor)
        x = np.asarray(run["profile"]["x"])
        h = np.asarray(run["profile"]["h_numeric"])
        h_ex = np.asarray(run["profile"]["h_exact"])
        ax.plot(x, h, style, lw=1.3, label=f"numerical, floor = {_fmt(floor)}")
    ax.plot(x, h_ex, "g-.", lw=2.0, label="exact (Ritter)", alpha=0.8)
    ax.set_xlim(2.4, 4.3)
    ax.set_ylim(-0.01, 0.35)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("depth h [m]")
    ax.set_title(f"dry-front tip at t* = 2 (n = {n_zoom})")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    paths["fig_front_zoom"] = dirpath / "fig_front_zoom.png"
    fig.savefig(paths["fig_front_zoom"], dpi=140)
    plt.close(fig)
    return paths


# --------------------------------------------------------------------------- #
# generated summary document
# --------------------------------------------------------------------------- #
def summary_markdown(payload: dict, figs: dict) -> str:
    base = payload["baseline"]
    lines = [
        "# CASE-FVM-DAMBREAK-STOKER · numerical-configuration envelope",
        "",
        "Generated by `scripts/run_dambreak_config_envelope.py` -- every number "
        "below is injected from `metrics.json`; nothing is typed by hand.",
        "",
        f"Pool point: `n = {base['n']}`, `cfl = {base['cfl']}`, "
        f"`order = {base['order']}`, `dry_depth = {_fmt(base['dry_depth'])}`; "
        f"accepted minimum grid `n = {payload['accepted_min_n']}`.",
        "",
        "## CFL axis",
        "",
    ]
    for case in ("ritter-dry", "stoker-wet"):
        rows = payload["axes"][case]["cfl"]["rows"]
        cols = ["cfl", "ok", "hard_any", "steps", "front_fan<5%", "front_wetted<5%",
                "shock_position<5%", "step_ratio<5%", "depth_l1<5%",
                "front_series<5%", "local_extrema", "hidden_mass_fraction"]
        head = ["cfl", "ok", "hard", "steps", "front_fan", "front_wetted",
                "shock", "step_ratio", "depth_l1", "series", "extrema", "hidden mass"]
        lines += [f"### {case}", "", _table(rows, cols, head), ""]
        env = payload["axes"][case]["cfl"]["envelope"]
        lines += [f"Certified CFL: {[_fmt(v) for v in env['certified']]}; "
                  f"failed: {[_fmt(v) for v in env['failed_points']]}; "
                  f"binding gate: {env['binding']}.", ""]
    lines += ["## order axis", ""]
    for case in ("ritter-dry", "stoker-wet"):
        rows = payload["axes"][case]["order"]["rows"]
        cols = ["order", "front_fan<5%", "front_wetted<5%", "shock_position<5%",
                "step_ratio<5%", "depth_l1<5%", "front_series<5%"]
        head = ["order", "front_fan", "front_wetted", "shock", "step_ratio",
                "depth_l1", "series"]
        lines += [f"### {case}", "", _table(rows, cols, head), ""]
    lines += ["## depth-floor axis", ""]
    for case in ("ritter-dry", "stoker-wet"):
        rows = payload["axes"][case]["floor"]["rows"]
        cols = ["dry_depth", "ok", "hard_any", "front_fan<5%", "front_wetted<5%",
                "shock_position<5%", "depth_l1<5%", "front_series<5%",
                "hidden_mass_fraction", "reading.front_recession", "local_extrema"]
        head = ["floor", "ok", "hard", "front_fan", "front_wetted", "shock",
                "depth_l1", "series", "hidden mass", "recession [m]", "extrema"]
        lines += [f"### {case}", "", _table(rows, cols, head), ""]
        env = payload["axes"][case]["floor"]["envelope"]
        lines += [f"Certified floor: {[_fmt(v) for v in env['certified']]}; "
                  f"failed: {[_fmt(v) for v in env['failed_points']]}; "
                  f"binding gate: {env['binding']}", ""]
    lines += ["## confirmation on the pool's accepted grid (n = 1600)", "",
              _table(payload["confirmation_summary"],
                     ["case", "n", "dry_depth", "all_gates_pass",
                      "front_wetted_error", "front_fan_error",
                      "hidden_mass_fraction"],
                     ["case", "n", "floor", "all gates", "front_wetted",
                      "front_fan", "hidden mass"]), ""]
    lines += ["## hard-failure detector, negative controls", ""]
    for name in ("dry", "wet"):
        rows = payload["negative_control"][name]
        lines += [f"### {name}", "",
                  _table(rows, ["case", "dry_depth", "ok", "hard_any",
                                "front_wetted<5%", "shock_position<5%",
                                "depth_l1<5%", "monotone(no_extrema)",
                                "local_extrema"],
                         ["case", "floor", "ok", "hard failure", "front_wetted",
                          "shock", "depth_l1", "monotone", "extrema"]), ""]
    lines += ["## instrument calibration (hidden-mass reading)", "",
              _table([{"floor": k, **v} for k, v in
                      sorted(payload["instrument"].items(), key=lambda kv: float(kv[0]))],
                     ["floor", "n", "discrete", "reference", "relative_difference",
                      "analytic_tip_length"],
                     ["floor", "n", "discrete sum", "quadrature", "rel. diff",
                      "analytic tip [m]"]), "",
              "## figures", ""]
    for name, path in figs.items():
        lines += [f"* `{path.name}` -- {name}", ""]
    return "\n".join(lines)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    payload = study()
    payload["provenance"] = {"git_commit": _git_commit(),
                             "command": "PYTHONPATH=src python "
                                        "scripts/run_dambreak_config_envelope.py",
                             "python": sys.version.split()[0],
                             "numpy": np.__version__}
    figs = figures(payload, OUT)
    (OUT / "metrics.json").write_text(json.dumps(payload, indent=1, default=str))
    (OUT / "SUMMARY.md").write_text(summary_markdown(payload, figs))

    dry = payload["axes"]["ritter-dry"]
    amendment = {
        "feature_slug": "CASE-FVM-DAMBREAK-STOKER",
        "proposal": "declare the numerical configuration, not only the grid",
        "current_entry_declares": {"n": 1600},
        "measured": {
            "certified_cfl": dry["cfl"]["envelope"]["certified"],
            "certified_order": dry["order"]["envelope"]["certified"],
            "certified_dry_depth": dry["floor"]["envelope"]["certified"],
            "first_break_gate": dry["floor"]["envelope"]["binding"],
            "confirmation_n1600": payload["confirmation_summary"],
        },
        "requested_addition": {
            "configuration": {"n": 1600, "cfl": 0.4, "order": 2, "dry_depth": 1e-12},
            "certified_band": {"dry_depth_max": 1e-9, "cfl": [0.1, 0.95],
                               "order": [1, 2]},
            "rationale": "the legacy diagnostic `front_wetted<5%` is a depth-floor "
                         "reading: it passes at the shipped floor 1e-12 but breaks "
                         "at dry_depth >= 1e-6; a consumer cannot reproduce the "
                         "entry's PASS without the floor.",
        },
    }
    (OUT / "proposed_pool_amendment.json").write_text(json.dumps(amendment, indent=1))
    print(f"wrote {OUT}")
    for name, path in figs.items():
        print("  fig", path.name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())