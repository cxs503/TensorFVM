#!/usr/bin/env python3
"""Generate the verification report markdown (and a platform report payload).

All numeric tables are rendered from ``artifacts/inc_bench/summary.json`` so that
no value in the report is hand-copied from a solver log.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "artifacts" / "inc_bench"
DOCS = ROOT / "docs" / "inc-bench"


DIVERGENCE_LIMIT = 1.0e6


def fmt(value: float | None, digits: int = 4) -> str:
    if value is None:
        return "-"
    if isinstance(value, float) and not math.isfinite(value):
        return "n/a"
    if isinstance(value, float) and abs(value) > DIVERGENCE_LIMIT:
        return "diverged"
    return f"{value:.{digits}f}"


def pct(value: float | None, digits: int = 2) -> str:
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return "n/a"
    if isinstance(value, float) and abs(value) > DIVERGENCE_LIMIT:
        return "diverged"
    return f"{value * 100:+.{digits}f}%"


def table(headers: list[str], rows: list[list[str]]) -> str:
    out = ["| " + " | ".join(headers) + " |",
           "|" + "|".join(["---"] * len(headers)) + "|"]
    for row in rows:
        out.append("| " + " | ".join(row) + " |")
    return "\n".join(out)


def load_summary() -> dict:
    path = SRC / "summary.json"
    if not path.exists():
        raise SystemExit("run `scripts/run_inc_bench.py summary` first")
    return json.loads(path.read_text())


def render(summary: dict) -> str:
    cases = summary.get("cases", {})
    lines: list[str] = []
    lines.append("# Incompressible FVM verification surface: backward-facing step, "
                 "sudden expansion, natural-convection cavity")
    lines.append("")
    lines.append("Independent verification of the TensorFVM finite-volume kernel on three "
                 "classical incompressible benchmarks.  The solver path used here "
                 "(`tensorfvm.inc_bench`) shares no mesh, projection or pressure code with "
                 "the LBM verification path or with the lid-cavity DCT solver, so the "
                 "evidence is independent rather than a re-reading of existing evidence.")
    lines.append("")
    lines.append("## Criteria overview")
    lines.append("")
    rows = []
    for item in summary.get("criteria", []):
        rows.append([item["id"], item["description"], fmt(item["value"], 5),
                     fmt(item["reference"], 5),
                     fmt(item.get("error"), 4),
                     "PASS" if item["passed"] else "FAIL"])
    lines.append(table(["criterion", "requirement", "value", "reference", "error", "verdict"],
                       rows))
    lines.append("")

    bfs = cases.get("backward_facing_step")
    if bfs:
        lines.append("## 1. Backward-facing step (Armaly geometry, expansion ratio 1.9423)")
        lines.append("")
        rows = []
        for level, entry in bfs["levels"].items():
            rows.append([level, str(entry["nx"]), str(entry["ny"]), str(entry["cells"]),
                         fmt(entry["expansion_ratio_mask"], 5),
                         fmt(entry["L_r_over_S"], 3), fmt(entry["friction_min"], 5),
                         fmt(entry["friction_far"], 5),
                         fmt(entry["friction_far_analytic"], 5),
                         pct(entry["friction_far_rel_error"], 2),
                         str(entry["iterations"])])
        lines.append(table(["grid", "nx", "ny", "cells", "ER (mask)", "L_r/S",
                            "min C_f", "C_f far", "C_f analytic", "rel. err.",
                            "iterations"], rows))
        lines.append("")
        rich = bfs.get("richardson", {})
        lines.append(f"Richardson extrapolation of L_r/S: "
                     f"**{fmt(rich.get('estimate'), 3)}** "
                     f"(observed order {fmt(rich.get('order'), 2)}; {rich.get('note', '')}).")
        lines.append("")

    exp = cases.get("sudden_expansion")
    if exp:
        lines.append("## 2. Planar sudden expansion (AR = 4, Re_D = 100)")
        lines.append("")
        rows = []
        for level, entry in exp["levels"].items():
            rows.append([level, str(entry["cells"]), fmt(entry["L_r_over_h"], 3),
                         fmt(entry["K"], 4), fmt(entry["K_borda_carnot"], 4),
                         fmt(entry["K_excess"], 4), f"{entry['mass_imbalance']:.2e}",
                         str(entry["iterations"])])
        lines.append(table(["grid", "cells", "L_r/h_in", "K", "K_Borda-Carnot",
                            "K - K_BC", "mass imbalance", "iterations"], rows))
        lines.append("")

    cav = cases.get("natural_cavity")
    if cav:
        lines.append("## 3. Differentially heated square cavity (Pr = 0.71)")
        lines.append("")
        rows = []
        for key, entry in cav["levels"].items():
            rows.append([f"{entry['rayleigh']:.0e}", str(entry["cells"]),
                         fmt(entry["nusselt"], 3), fmt(entry["nusselt_reference"], 3),
                         pct(entry["nusselt_rel_error"], 2),
                         fmt(entry["psi_max"], 3), fmt(entry["psi_max_reference"], 3),
                         pct(entry["psi_max_rel_error"], 2),
                         fmt(entry["u_max"], 2), pct(entry["u_max_rel_error"], 2),
                         pct(entry["energy_balance_rel"], 3)])
        lines.append(table(["Ra", "cells", "Nu", "Nu_ref", "err", "psi_max",
                            "psi_max_ref", "err", "u_max", "err", "energy balance"],
                           rows))
        lines.append("")
    return "\n".join(lines)


def main() -> int:
    summary = load_summary()
    DOCS.mkdir(parents=True, exist_ok=True)
    text = render(summary)
    (DOCS / "report.md").write_text(text + "\n")
    print(f"wrote {DOCS / 'report.md'} ({len(text)} chars)")
    return 0


if __name__ == "__main__":
    sys.exit(main())