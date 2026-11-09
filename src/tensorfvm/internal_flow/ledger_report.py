"""Assemble the internal-flow ledger campaign report from the case records.

Nothing in the report is typed by hand: every table cell and every figure is
derived from the per-case JSON records written by
:mod:`tensorfvm.internal_flow.pipeline` (and the record list is hashed into
``results.json`` so a test can prove the numbers were not edited afterwards).
"""

from __future__ import annotations

import hashlib
import json
import math
import os

from . import metrics, reference

LEVELS = ("coarse", "medium", "fine")
GRID_RATIO = 2.0


def _load(out_dir: str) -> dict[str, dict]:
    records = {}
    for name in sorted(os.listdir(out_dir)):
        if not name.endswith(".json") or name == "results.json":
            continue
        with open(os.path.join(out_dir, name)) as handle:
            rec = json.load(handle)
        rec["_file"] = name
        records[name] = rec
    return records


def _pick(records: dict[str, dict], family: str, prefix: str, level: str,
          mode: str = "dirichlet") -> dict | None:
    for rec in records.values():
        c = rec["case"]
        if c["family"] == family and c["level"] == level and c["outlet_mode"] == mode \
                and c["case"].startswith(prefix):
            return rec
    return None


def _families(records: dict[str, dict], family: str) -> dict[str, dict[str, dict]]:
    """``{case name: {level: record}}`` for the baseline outlet mode."""
    out: dict[str, dict[str, dict]] = {}
    for rec in records.values():
        c = rec["case"]
        if c["family"] != family or c["outlet_mode"] not in ("dirichlet", "closed"):
            continue
        out.setdefault(c["case"], {})[c["level"]] = rec
    return out


def _finest(by_level: dict[str, dict]) -> tuple[str, dict]:
    """The finest grid level actually available for a case."""
    for level in reversed(LEVELS):
        if level in by_level:
            return level, by_level[level]
    raise KeyError("no grid level available")


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _crit(value, reference_value, error, threshold: float, target: float | None = None
          ) -> dict:
    entry = {"value": value, "reference": reference_value, "error": error,
             "threshold": threshold, "passed": bool(error is not None
                                                    and error == error
                                                    and error <= threshold)}
    if target is not None:
        entry["target"] = target
        entry["meets_target"] = bool(entry["passed"] and error <= target)
    return entry


def assemble(records: dict[str, dict]) -> dict:
    """All criteria and tables of the campaign, as a JSON-able dict."""
    criteria: list[dict] = []
    tables: dict[str, object] = {}

    # ------------------------------------------------------- 1. duct ledgers
    duct_rows = []
    for rec in records.values():
        case = rec["case"]
        if case["family"] not in ("expansion", "step"):
            continue
        for box_name, led in rec.get("ledgers", {}).items():
            rel = led.get("relative_inlet_momentum")
            duct_rows.append({"case": case["case"], "level": case["level"],
                              "outlet": case["outlet_mode"], "box": box_name,
                              "residual": led["residual"],
                              "convective": led["convective"],
                              "pressure": led["pressure"],
                              "viscous": led["viscous"],
                              "relative_inlet_momentum": rel,
                              "relative_max_term": led["relative_residual"]})
            if case["outlet_mode"] == "dirichlet" and box_name == "full" \
                    and case["level"] == _finest(_families(records, case["family"])
                                                 [case["case"]])[0]:
                criteria.append({
                    "id": f"ledger.momentum_residual.{case['case']}.full",
                    "description": "control-volume x-momentum residual of the "
                                   "inlet->exit box, normalised by rho U_in^2 A_in",
                    **_crit(rel, 0.0, rel, 0.01)})
    tables["duct_ledgers"] = duct_rows

    # ------------------------------------------------------ 2. cavity ledgers
    cavity_rows = []
    for rec in records.values():
        case = rec["case"]
        if case["family"] != "cavity":
            continue
        for box_name in ("ledger_y_full", "ledger_y_left_half"):
            led = rec[box_name]
            rel = led.get("relative_to_driving")
            cavity_rows.append({"case": case["case"], "level": case["level"],
                                "box": box_name, "residual": led["residual"],
                                "convective": led["convective"],
                                "pressure": led["pressure"],
                                "viscous": led["viscous"],
                                "buoyancy": led["buoyancy"],
                                "relative_to_driving": rel})
            if box_name == "ledger_y_left_half" and case["level"] == _finest(
                    _families(records, "cavity")[case["case"]])[0]:
                criteria.append({
                    "id": f"ledger.momentum_residual.{case['case']}.left_half",
                    "description": "y-momentum residual of the hot-wall half of the "
                                   "cavity, normalised by the integral buoyancy force",
                    **_crit(rel, 0.0, rel, 0.01)})
    tables["cavity_ledgers"] = cavity_rows

    # ------------------------------------------- 3. Cp recovery (expansion)
    cp_rows = []
    orders = {}
    for name, by_level in _families(records, "expansion").items():
        finest_level, fine = _finest(by_level)
        cp = fine["cp"]
        cp_rows.append({"case": name, "level": finest_level,
                        "cp_measured": cp["cp_measured"],
                        "cp_ledger": cp["cp_ledger_prediction"],
                        "cp_borda_carnot": cp["cp_borda_carnot"],
                        "dcp_profile": cp["dcp_profile"],
                        "dcp_step_face": cp["dcp_step_face"],
                        "dcp_viscous": cp["dcp_viscous"],
                        "area_ratio": cp["area_ratio"],
                        "closure": cp["decomposition_closure"]})
        criteria.append({
            "id": f"expansion.cp_two_routes.{cp['area_ratio']:g}",
            "description": f"pressure-tap Cp vs momentum-ledger Cp over the same "
                           f"control volume (inlet side of the step -> 4 h_out), "
                           f"grid level {finest_level}",
            "level": finest_level,
            **_crit(cp["cp_measured"], cp["cp_ledger_prediction"],
                    cp["cp_relative_error"], 0.05, target=0.03)})
        if not all(level in by_level for level in LEVELS):
            continue
        rich = metrics.richardson(by_level["coarse"]["cp"]["cp_measured"],
                                  by_level["medium"]["cp"]["cp_measured"],
                                  by_level["fine"]["cp"]["cp_measured"], GRID_RATIO)
        orders[f"expansion.cp.{cp['area_ratio']:g}"] = rich
        criteria.append({
            "id": f"grid.observed_order.expansion.cp.{cp['area_ratio']:g}",
            "description": "observed order of Cp from the three grids (>= 1.8)",
            **_crit(rich["observed_order"], 2.0,
                    abs(rich["observed_order"] - 2.0)
                    if rich["observed_order"] == rich["observed_order"] else None, 0.2)})
    tables["expansion_cp"] = cp_rows

    # ------------------------------------------- 4. reattachment (step + exp)
    re_rows = []
    for family in ("step", "expansion"):
        for name, by_level in _families(records, family).items():
            finest_level, finest_rec = _finest(by_level)
            fine = finest_rec["reattachment"]
            rich = {"observed_order": float("nan"), "extrapolated": float("nan"),
                    "fine_vs_extrapolated": float("nan")}
            if all(level in by_level for level in LEVELS):
                rich = metrics.richardson(by_level["coarse"]["reattachment"]["shear"],
                                          by_level["medium"]["reattachment"]["shear"],
                                          by_level["fine"]["reattachment"]["shear"],
                                          GRID_RATIO)
                orders[f"{family}.reattachment.{name}"] = rich
            re_rows.append({"case": name, "level": finest_level,
                            "shear": fine["shear"], "velocity": fine["velocity"],
                            "richardson": rich["extrapolated"],
                            "observed_order": rich["observed_order"],
                            "fine_vs_extrapolated": rich["fine_vs_extrapolated"],
                            "diagnostic_gap": abs(fine["shear"] - fine["velocity"])
                                              / max(abs(fine["shear"]), 1e-30)})
            if not all(level in by_level for level in LEVELS):
                continue
            if family == "step":
                rec = by_level["fine"]
                anchor = reference.armaly_reattachment(rec["case"]["params"]["reynolds"])
                criteria.append({
                    "id": f"step.reattachment.{anchor.name}",
                    "description": "fine-grid reattachment vs the Richardson "
                                   "extrapolate of the same grid family (<= 2 %)",
                    **_crit(fine["shear"], rich["extrapolated"],
                            rich["fine_vs_extrapolated"], 0.02)})
                criteria.append({
                    "id": f"step.reattachment_band.{anchor.name}",
                    "description": "orientation only: |x_r/S - Armaly| / Armaly "
                                   "(grade WITHHELD, +/-10 % band, not a gate)",
                    **_crit(fine["shear"], anchor.value,
                            abs(fine["shear"] - anchor.value) / anchor.value, 0.10)})
            criteria.append({
                "id": f"grid.observed_order.{family}.reattachment.{name}",
                "description": "observed order of the reattachment length (>= 1.8)",
                **_crit(rich["observed_order"], 2.0,
                        abs(rich["observed_order"] - 2.0)
                        if rich["observed_order"] == rich["observed_order"] else None,
                        0.2)})
    tables["reattachment"] = re_rows

    # ------------------------------------------------- 5. cavity Nusselt
    nu_rows = []
    for name, by_level in _families(records, "cavity").items():
        finest_level, fine = _finest(by_level)
        ra = fine["case"]["params"]["rayleigh"]
        nu = fine["nusselt"]["nusselt_hot"]
        nu_rows.append({"case": name, "level": finest_level, "rayleigh": ra,
                        "nusselt": nu,
                        "nusselt_order1": fine["nusselt"]["nusselt_hot_order1"],
                        "reference": fine["nusselt_reference"],
                        "error": fine["nusselt_rel_error"],
                        "error_order1": fine["nusselt_rel_error_order1"],
                        "psi_max": fine["psi_max_scaled"],
                        "psi_max_reference": fine["psi_max_reference"],
                        "u_max": fine["u_max_scaled"],
                        "u_max_reference": fine["u_max_reference"],
                        "v_max": fine["v_max_scaled"],
                        "v_max_reference": fine["v_max_reference"],
                        "energy_balance_rel": fine["nusselt"]["energy_balance_rel"],
                        # records written before the cold-wall sign fix carry
                        # Nu_cold = -Nu_hot; flag them instead of showing 200 %
                        "postprocessing_stale": bool(
                            fine["nusselt"]["energy_balance_rel"] > 0.5)})
        criteria.append({
            "id": f"cavity.nusselt.{ra:g}",
            "description": f"mean hot-wall Nu on grid level {finest_level} vs "
                           "de Vahl Davis (1983), threshold 15 %, target 3 %",
            "level": finest_level,
            **_crit(nu, fine["nusselt_reference"], fine["nusselt_rel_error"],
                    0.15, target=0.03)})
        if all(level in by_level for level in LEVELS):
            rich = metrics.richardson(by_level["coarse"]["nusselt"]["nusselt_hot"],
                                      by_level["medium"]["nusselt"]["nusselt_hot"],
                                      nu, GRID_RATIO)
            orders[f"cavity.nusselt.{ra:g}"] = rich
            criteria.append({
                "id": f"grid.observed_order.cavity.nusselt.{ra:g}",
                "description": "observed order of Nu from three grids (>= 1.8)",
                **_crit(rich["observed_order"], 2.0,
                        abs(rich["observed_order"] - 2.0)
                        if rich["observed_order"] == rich["observed_order"] else None,
                        0.2)})
    tables["cavity_nusselt"] = nu_rows

    # ------------------------------------------------------ 6. outlet effects
    outlet_rows = []
    for family, quantity in (("expansion", "cp"), ("step", "reattachment")):
        base = None
        for rec in records.values():
            c = rec["case"]
            if c["family"] != family or c["level"] != "medium":
                continue
            value = (rec["cp"]["cp_measured"] if quantity == "cp"
                     else rec["reattachment"]["shear"])
            row = {"family": family, "outlet": c["outlet_mode"], "value": value,
                   "quantity": quantity}
            outlet_rows.append(row)
            if c["outlet_mode"] == "dirichlet":
                base = value
        if base is None:
            continue
        rows_family = [row for row in outlet_rows if row["family"] == family]
        for row in rows_family:
            if base:
                row["drift"] = abs(row["value"] - base) / abs(base)
                row["identical_to_dirichlet"] = bool(row["drift"] == 0.0)
        distinct = [row for row in rows_family if not row.get("identical_to_dirichlet")]
        evaluated = len(distinct) >= 1 and len(rows_family) >= 2
        drift = (max(row["drift"] for row in distinct) if evaluated
                 else float("nan"))
        criteria.append({
            "id": f"outlet.drift.{quantity}.{family}",
            "description": "max drift of the quantity over the outlet treatments "
                           "actually evaluated differently from the Dirichlet "
                           "baseline (dirichlet / extended domain / traction-free); "
                           f"{len(rows_family)} treatment(s) recorded, "
                           f"{len(distinct)} distinct",
            "treatments_recorded": [row["outlet"] for row in rows_family],
            "treatments_distinct": [row["outlet"] for row in distinct],
            "evaluated": evaluated,
            **_crit(drift, 0.0, drift, 0.05)})
    tables["outlet_sensitivity"] = outlet_rows

    # ------------------------------------------ 7. grid-triplet coverage guard
    for family in ("expansion", "step", "cavity"):
        for name, by_level in _families(records, family).items():
            missing = [level for level in LEVELS if level not in by_level]
            if missing:
                criteria.append({
                    "id": f"grid.triplet.{family}.{name}",
                    "description": "three grid levels are required for the observed-order "
                                   "fit; missing levels make the case incomplete",
                    "value": float(len(LEVELS) - len(missing)),
                    "reference": float(len(LEVELS)),
                    "error": float(len(missing)), "threshold": 0.0,
                    "passed": False, "missing_levels": missing})

    fail = [c["id"] for c in criteria if not c["passed"]]
    targets_missed = [c["id"] for c in criteria
                      if c.get("target") is not None and not c.get("meets_target")]
    return {"criteria": criteria, "orders": orders, "tables": tables,
            "verdict": {"passed": not fail, "failed": fail,
                        "targets_missed": targets_missed,
                        "n_criteria": len(criteria)}}


def _fmt(x, digits: int = 4) -> str:
    if isinstance(x, float):
        if not math.isfinite(x):
            return "n/a"
        return f"{x:.{digits}f}"
    return str(x)


def _table(rows: list[dict], columns: list[str]) -> str:
    head = "| " + " | ".join(columns) + " |\n|" + "|".join(["---"] * len(columns)) + "|\n"
    body = ""
    for row in rows:
        body += "| " + " | ".join(_fmt(row.get(c)) for c in columns) + " |\n"
    return head + body


def figures(records: dict[str, dict], payload: dict, fig_dir: str) -> list[dict]:
    """Generate the comparison figures; returns ``[{name, caption, path}]``."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(fig_dir, exist_ok=True)
    out = []
    tables = payload["tables"]

    # ---- Cp decomposition
    rows = tables["expansion_cp"]
    if rows:
        labels = [f"AR={r['area_ratio']:g}" for r in rows]
        terms = ["dcp_profile", "dcp_step_face", "dcp_viscous"]
        fig, ax = plt.subplots(figsize=(6.4, 4.0))
        bottom = [r["cp_borda_carnot"] for r in rows]
        ax.bar(labels, bottom, color="#4477aa", label="Cp Borda-Carnot (inviscid)")
        for term in terms:
            vals = [r[term] for r in rows]
            ax.bar(labels, vals, bottom=bottom, label=term)
            bottom = [b + v for b, v in zip(bottom, vals)]
        ax.plot(labels, [r["cp_measured"] for r in rows], "ko", label="Cp measured")
        ax.axhline(0.0, color="k", lw=0.6)
        ax.set_ylabel("Cp = (p2 - p1) / (0.5 rho U_in^2)")
        ax.set_title("Sudden expansion: Cp recovery ledger decomposition (fine grid)")
        ax.legend(fontsize=7)
        fig.tight_layout()
        path = os.path.join(fig_dir, "fig_cp_decomposition.png")
        fig.savefig(path, dpi=140)
        plt.close(fig)
        out.append({"name": "fig_cp_decomposition.png",
                    "caption": "Cp recovery decomposed into the inviscid Borda-Carnot "
                               "value plus the profile, step-face and viscous terms "
                               "(black dot = measured pressure-tap Cp).",
                    "path": path})

    # ---- ledger residual convergence
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    plotted = False
    for family in ("expansion", "step"):
        for name in _families(records, family):
            by_level = _families(records, family)[name]
            xs, ys = [], []
            for level in LEVELS:
                rec = by_level.get(level)
                if rec is None:
                    continue
                h = rec["case"]["dx"]
                led = rec["ledgers"].get("full")
                if led is None:
                    continue
                xs.append(h)
                ys.append(max(led["relative_inlet_momentum"], 1e-12))
            if len(xs) >= 2:
                ax.loglog(xs, ys, "o-", label=name)
                plotted = True
    for name in _families(records, "cavity"):
        by_level = _families(records, "cavity")[name]
        xs, ys = [], []
        for level in LEVELS:
            rec = by_level.get(level)
            if rec is None:
                continue
            xs.append(rec["case"]["dx"])
            ys.append(max(rec["ledger_y_left_half"]["relative_to_driving"], 1e-12))
        if len(xs) >= 2:
            ax.loglog(xs, ys, "s--", label=name)
            plotted = True
    if plotted:
        ax.axhline(0.01, color="r", ls=":", label="1 % gate")
        ax.set_xlabel("cell size h")
        ax.set_ylabel("|momentum residual| / scale")
        ax.set_title("Control-volume momentum residual vs grid")
        ax.legend(fontsize=7)
        fig.tight_layout()
        path = os.path.join(fig_dir, "fig_ledger_convergence.png")
        fig.savefig(path, dpi=140)
        out.append({"name": "fig_ledger_convergence.png",
                    "caption": "Momentum-ledger residual versus cell size for the "
                               "duct (solid) and cavity (dashed) cases; the dotted "
                               "line is the 1 % acceptance gate.", "path": path})
    plt.close(fig)

    # ---- wall shear / reattachment
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    plotted = False
    for name in _families(records, "step"):
        rec = _families(records, "step")[name].get("fine")
        if rec is None:
            continue
        prof = rec["reattachment"]
        ax.plot(prof["profile_x"], prof["profile_tau"], label=name)
        ax.axhline(0.0, color="k", lw=0.6)
        ax.axvline(prof["shear"], ls=":", lw=0.8)
        plotted = True
    if plotted:
        ax.set_xlabel("x (step heights)")
        ax.set_ylabel("wall shear tau_w (bottom wall)")
        ax.set_xlim(0, 14)
        ax.set_title("Backward-facing step: wall shear and reattachment (fine grid)")
        ax.legend(fontsize=8)
        fig.tight_layout()
        path = os.path.join(fig_dir, "fig_step_wall_shear.png")
        fig.savefig(path, dpi=140)
        out.append({"name": "fig_step_wall_shear.png",
                    "caption": "Bottom-wall shear stress versus streamwise distance; "
                               "the zero crossing (dotted) is the reattachment point.",
                    "path": path})
    plt.close(fig)

    # ---- Nusselt vs Rayleigh
    rows = tables["cavity_nusselt"]
    if rows:
        fig, ax = plt.subplots(figsize=(6.0, 4.0))
        ra = [r["rayleigh"] for r in rows]
        ax.loglog(ra, [r["nusselt"] for r in rows], "o-", label="this solver")
        ax.loglog(ra, [r["reference"] for r in rows], "s--",
                  label="de Vahl Davis 1983")
        for r in rows:
            ax.annotate(f"{100 * r['error']:+.1f}%", (r["rayleigh"], r["nusselt"]),
                        fontsize=7, xytext=(4, 4), textcoords="offset points")
        ax.set_xlabel("Rayleigh number")
        ax.set_ylabel("mean Nu")
        ax.set_title("Natural convection cavity: Nu vs Ra")
        ax.legend(fontsize=8)
        fig.tight_layout()
        path = os.path.join(fig_dir, "fig_cavity_nusselt.png")
        fig.savefig(path, dpi=140)
        out.append({"name": "fig_cavity_nusselt.png",
                    "caption": "Mean Nusselt number versus Rayleigh number compared "
                               "with de Vahl Davis (1983); annotations are the "
                               "relative errors.", "path": path})
    plt.close(fig)

    # ---- mesh convergence of the physical quantities
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    plotted = False
    for name, by_level in _families(records, "step").items():
        xs, ys = [], []
        for level in LEVELS:
            rec = by_level.get(level)
            if rec is None:
                continue
            xs.append(rec["case"]["dx"])
            ys.append(rec["reattachment"]["shear"])
        if len(xs) == 3:
            ax.plot(xs, ys, "o-", label=f"{name} x_r/S")
            plotted = True
    for name, by_level in _families(records, "expansion").items():
        xs, ys = [], []
        for level in LEVELS:
            rec = by_level.get(level)
            if rec is None:
                continue
            xs.append(rec["case"]["dx"])
            ys.append(rec["cp"]["cp_measured"])
        if len(xs) == 3:
            ax.plot(xs, ys, "s--", label=f"{name} Cp")
            plotted = True
    if plotted:
        ax.set_xscale("log")
        ax.set_xlabel("cell size h")
        ax.set_ylabel("quantity")
        ax.set_title("Grid convergence of the reported quantities")
        ax.legend(fontsize=7)
        fig.tight_layout()
        path = os.path.join(fig_dir, "fig_grid_convergence.png")
        fig.savefig(path, dpi=140)
        out.append({"name": "fig_grid_convergence.png",
                    "caption": "Reattachment length and Cp recovery versus cell size "
                               "on the three grids used for the observed-order fit.",
                    "path": path})
    plt.close(fig)

    # ---- outlet sensitivity
    rows = tables["outlet_sensitivity"]
    if rows:
        fig, ax = plt.subplots(figsize=(6.0, 3.6))
        labels = sorted({r["family"] for r in rows})
        width = 0.25
        modes = ["dirichlet", "extended", "traction"]
        for k, mode in enumerate(modes):
            vals = [next((r["value"] for r in rows
                          if r["family"] == fam and r["outlet"] == mode), float("nan"))
                    for fam in labels]
            base = [next((r["value"] for r in rows
                          if r["family"] == fam and r["outlet"] == "dirichlet"), 1.0)
                    for fam in labels]
            ax.bar([i + k * width for i in range(len(labels))],
                   [v / b for v, b in zip(vals, base)], width, label=mode)
        ax.axhline(1.05, color="r", ls=":", lw=0.8)
        ax.axhline(0.95, color="r", ls=":", lw=0.8)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels)
        ax.set_ylabel("value / dirichlet value")
        ax.set_title("Outlet-boundary sensitivity (medium grid)")
        ax.legend(fontsize=7)
        fig.tight_layout()
        path = os.path.join(fig_dir, "fig_outlet_sensitivity.png")
        fig.savefig(path, dpi=140)
        out.append({"name": "fig_outlet_sensitivity.png",
                    "caption": "Cp (expansion) and reattachment (step) for the three "
                               "outlet treatments, normalised by the Dirichlet case; "
                               "dotted lines are the +/-5 % band.", "path": path})
    plt.close(fig)

    for rec in records.values():
        fig_path = rec.get("figure")
        if fig_path and os.path.exists(fig_path):
            out.append({"name": os.path.basename(fig_path),
                        "caption": f"Velocity magnitude, {rec['case']['case']} "
                                   f"{rec['case']['level']} "
                                   f"(outlet={rec['case']['outlet_mode']}).",
                        "path": fig_path})
    return out

def build(out_dir: str, *, docs_dir: str | None = None) -> dict:
    """Write ``results.json`` (+ figures and markdown) and return the payload."""
    records = _load(out_dir)
    if not records:
        raise SystemExit(f"no case records in {out_dir}")
    payload = assemble(records)
    payload["source_records"] = {
        name: _sha256(os.path.join(out_dir, name)) for name in records}
    payload["cases"] = {name: rec["case"] for name, rec in records.items()}
    payload["solver_log"] = {name: rec["solver"] for name, rec in records.items()}
    fig_dir = os.path.join(out_dir, "figures")
    payload["figures"] = figures(records, payload, fig_dir)
    results_path = os.path.join(out_dir, "results.json")
    with open(results_path, "w") as handle:
        json.dump(payload, handle, indent=1, default=float)
    if docs_dir:
        os.makedirs(docs_dir, exist_ok=True)
        markdown = render_markdown(payload)
        with open(os.path.join(docs_dir, "REPORT.md"), "w") as handle:
            handle.write(markdown)
        table_path = os.path.join(docs_dir, "results.json")
        with open(table_path, "w") as handle:
            json.dump(payload, handle, indent=1, default=float)
    return payload


def render_markdown(payload: dict) -> str:
    tables = payload["tables"]
    verdict = payload["verdict"]
    lines = []
    add = lines.append
    add("# Internal-flow momentum-ledger closure: sudden expansion, "
        "backward-facing step, natural-convection cavity\n")
    add("Generated by `python scripts/run_internal_flow_ledger.py all`, then "
        "`report`. Every number below comes from the per-case JSON records whose "
        "sha256 digests are stored in `results.json.source_records`.\n")
    add(f"**Verdict: {'PASS' if verdict['passed'] else 'FAIL'}** — "
        f"{verdict['n_criteria']} criteria, failed: "
        f"{', '.join(verdict['failed']) if verdict['failed'] else 'none'}; "
        f"targets missed: "
        f"{', '.join(verdict['targets_missed']) if verdict['targets_missed'] else 'none'}.\n")
    add("## Criteria\n")
    add(_table(payload["criteria"],
               ["id", "value", "reference", "error", "threshold", "passed"]))
    add("\n## Sudden expansion: Cp recovery ledger (fine grid)\n")
    add(_table(tables["expansion_cp"],
               ["case", "cp_borda_carnot", "dcp_profile", "dcp_step_face",
                "dcp_viscous", "cp_ledger", "cp_measured", "closure"]))
    add("\n## Control-volume momentum residuals (duct cases)\n")
    add(_table(tables["duct_ledgers"],
               ["case", "level", "outlet", "box", "convective", "pressure",
                "viscous", "residual", "relative_inlet_momentum"]))
    add("\n## Reattachment lengths\n")
    add(_table(tables["reattachment"],
               ["case", "shear", "velocity", "richardson", "observed_order",
                "fine_vs_extrapolated", "diagnostic_gap"]))
    add("\n## Natural-convection cavity\n")
    add(_table(tables["cavity_nusselt"],
               ["case", "rayleigh", "nusselt", "reference", "error", "psi_max",
                "psi_max_reference", "u_max", "u_max_reference", "energy_balance_rel"]))
    add("\n## Cavity momentum ledger\n")
    add(_table(tables["cavity_ledgers"],
               ["case", "level", "box", "convective", "pressure", "viscous",
                "buoyancy", "residual", "relative_to_driving"]))
    add("\n## Outlet-boundary sensitivity\n")
    add(_table(tables["outlet_sensitivity"], ["family", "quantity", "outlet", "value",
                                              "drift"]))
    add("\n## Observed orders\n")
    add(_table([{"quantity": k, "observed_order": v["observed_order"],
                 "extrapolated": v["extrapolated"]} for k, v in
                payload["orders"].items()],
               ["quantity", "observed_order", "extrapolated"]))
    add("\n## Figures\n")
    for fig in payload["figures"]:
        add(f"* `{os.path.basename(fig['path'])}` — {fig['caption']}")
    return "\n".join(lines) + "\n"
