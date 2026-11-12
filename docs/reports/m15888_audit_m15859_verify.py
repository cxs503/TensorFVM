#!/usr/bin/env python3
"""Independent audit recomputation for m15859 (CASE-CHANNEL-PERIODIC, TensorFVM).

Deliberately does NOT reuse the delivered benchmark harness for the reference
numbers: the exact solution, the wall shear, the momentum balance and the error
norms are re-derived here from scratch (own numpy code), then compared with
(a) the committed artifacts docs/case-channel-periodic/*, (b) a fresh CLI replay
in /tmp/m15888-repro, and (c) a fresh solver configuration that is not part of
the delivered plan (ny=48, Re_H=250, nx=12).

Run:  PYTHONPATH=src python docs/reports/m15888_audit_m15859_verify.py
"""
import csv
import hashlib
import json
import math
import re
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
ART = ROOT / "docs" / "case-channel-periodic"
REPLAY = Path("/tmp/m15888-repro")
OUT = HERE / "m15888_audit_m15859_verify.json"

TOL = 0.02  # only used for reporting; the delivered TOLERANCE const is checked separately


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_profiles(path):
    rows = []
    with open(path) as fh:
        for r in csv.DictReader(fh):
            rows.append({k: (float(v) if k not in ("case", "wall_flux", "resolution") else v) for k, v in r.items()})
    return rows


def rel(computed, reference):
    if reference == 0.0:
        return abs(computed)
    return abs(computed - reference) / abs(reference)


def main():
    report = {}

    # ---------------------------------------------------------------- 1. artifacts present
    artifacts = ["REPORT.md", "benchmark.json", "metrics.csv", "profiles.csv", "manifest.json",
                 "figures/profile.png", "figures/convergence.png", "figures/perturbation.png"]
    report["artifacts_present"] = {p: (ART / p).exists() and (ART / p).stat().st_size > 0
                                   for p in artifacts}
    manifest = json.loads((ART / "manifest.json").read_text())
    report["manifest_sha256_match"] = {
        a["path"]: sha256(ART / a["path"]) == a["sha256"] for a in manifest["artifacts"]}
    bench = json.loads((ART / "benchmark.json").read_text())

    # ---------------------------------------------------------------- 2. own reference algebra
    # u(y) = G/(2 nu) y (H-y); Ubar = G H^2/(12 nu); tau_w = rho G H/2; f*Re_Dh = 96
    H, rho, mu = 1.0, 1.0, 0.01
    nu = mu / rho
    ubar_target = 1.0                       # Re_H = 100 with H=1, nu=0.01
    G = 12.0 * nu * ubar_target / H ** 2
    stated_G = bench["cases"][0]["body_force_m_s2"]
    report["own_algebra"] = dict(
        body_force_G=G, delivered_G=stated_G,
        G_rel_diff=rel(G, stated_G),
        peak=1.5 * ubar_target,
        tau_w=rho * G * H / 2.0,
        poiseuille=96.0,
    )

    # ---------------------------------------------------------------- 3. profile error, recomputed
    prof = load_profiles(ART / "profiles.csv")
    ref_rows = [r for r in prof if r["case"] == "reference"]
    y = np.array([r["y_over_H"] for r in ref_rows])
    u = np.array([r["u_m_s"] for r in ref_rows])
    u_exact = 6.0 * ubar_target * y * (1.0 - y)          # own formula, not theirs
    peak = 1.5 * ubar_target
    l2 = float(np.sqrt(np.mean((u - u_exact) ** 2)) / peak)
    linf = float(np.abs(u - u_exact).max() / peak)
    delivered_ref_metrics = {m["name"]: m for m in
                             next(c for c in bench["cases"] if c["role"] == "reference")["metrics"]}
    report["recomputed_reference_profile"] = dict(
        n_rows=len(ref_rows),
        velocity_l2_recomputed=l2,
        velocity_l2_delivered=delivered_ref_metrics["velocity_l2"]["computed"],
        velocity_l2_rel_diff=rel(l2, delivered_ref_metrics["velocity_l2"]["computed"]),
        velocity_linf_recomputed=linf,
        velocity_linf_delivered=delivered_ref_metrics["velocity_linf"]["computed"],
        velocity_linf_rel_diff=rel(linf, delivered_ref_metrics["velocity_linf"]["computed"]),
        csv_error_column_max_deviation=float(
            np.abs(np.array([r["error_m_s"] for r in ref_rows]) - (u - u_exact)).max()),
        under_2pct=(l2 < TOL),
    )

    # ---------------------------------------------------------------- 3b. other headline numbers
    bulk_csv = float(np.mean(u))
    dy = H / len(u)
    # per-wall shear (sum over the two walls = rho G H by the discrete balance)
    tau_flux = 0.5 * (mu * 2.0 * float(u[0]) / dy + mu * 2.0 * float(u[-1]) / dy)
    tau_quad = 0.5 * (mu * (9.0 * float(u[0]) - float(u[1])) / (3.0 * dy)
                      + mu * (9.0 * float(u[-1]) - float(u[-2])) / (3.0 * dy))
    f_darcy = 8.0 * tau_quad / (rho * bulk_csv ** 2)
    poiseuille = f_darcy * (2.0 * bulk_csv * H / nu)
    report["recomputed_headline"] = dict(
        bulk_velocity_recomputed=bulk_csv,
        bulk_rel_error=rel(bulk_csv, ubar_target),
        tau_wall_flux_estimator=tau_flux,
        tau_wall_flux_rel_error=rel(tau_flux, rho * G * H / 2.0),
        tau_wall_quadratic_estimator=tau_quad,
        tau_wall_quadratic_rel_error=rel(tau_quad, rho * G * H / 2.0),
        poiseuille_recomputed=poiseuille,
        poiseuille_rel_error=rel(poiseuille, 96.0),
        delivered_poiseuille=delivered_ref_metrics["poiseuille"]["computed"],
        poiseuille_rel_diff_vs_delivered=rel(poiseuille, delivered_ref_metrics["poiseuille"]["computed"]),
        all_under_2pct=all(rel(v, t) < TOL for v, t in
                           ((bulk_csv, ubar_target), (tau_flux, rho * G * H / 2),
                            (tau_quad, rho * G * H / 2), (poiseuille, 96.0))),
    )

    # ---------------------------------------------------------------- 4. gate arithmetic re-check
    recomputed_pass, mismatches = {}, []
    for c in bench["cases"]:
        for m in c["metrics"]:
            ok = bool(rel(m["computed"], m["reference"]) < m["tolerance"])
            recomputed_pass[(c["role"], c["resolution"], m["name"])] = ok
            if ok != m["passed"]:
                mismatches.append((c["role"], c["resolution"], m["name"], m["passed"], ok))
    gated = [c for c in bench["cases"] if c.get("gate", True)]
    report["gate_recheck"] = dict(
        n_cases=len(bench["cases"]), n_gated=len(gated),
        flag_mismatches=mismatches,
        all_gated_passed_recomputed=bool(all(recomputed_pass[(c["role"], c["resolution"], m["name"])]
                                             for c in gated for m in c["metrics"])),
        delivered_all_gated_passed=bench["all_gated_passed"],
        delivered_all_passed=bench["all_passed"],
        tolerance_const=bench["tolerance"],
        delivered_verdict=manifest["verdict"],
        coarse_grids_outside_gate=sorted({c["resolution"] for c in bench["cases"]
                                          if not c.get("gate", True)}),
        observed_orders=bench["observed_orders"],
        order_bands=bench["order_band"],
    )

    # ---------------------------------------------------------------- 5. delivered report sections
    md = (ART / "REPORT.md").read_text()
    marks = ["①", "②", "③", "④", "⑤", "⑥"]
    report["report_sections"] = {m: (m in md) for m in marks}
    report["report_sections"]["error_statement"] = ("误差判定" in md)
    report["report_sections"]["figure_refs"] = {f: (f in md) for f in
                                                ("profile.png", "convergence.png", "perturbation.png")}
    # numbers quoted in REPORT.md §4 vs benchmark.json
    quoted = re.findall(r"\|\s*(velocity_l2|velocity_linf|peak_velocity|bulk_velocity|"
                        r"wall_shear_quadratic|wall_shear_flux|poiseuille|momentum_balance)\s*\|"
                        r"[^|]*\|[^|]*\|\s*([0-9.eE+-]+)\s*\|", md)
    report["report_number_crosscheck"] = {
        name: dict(quoted_error=float(err),
                   benchmark_error=delivered_ref_metrics[name]["relative_error"],
                   match=rel(float(err), delivered_ref_metrics[name]["relative_error"]) < 0.02)
        for name, err in quoted}

    # ---------------------------------------------------------------- 6. fresh replay (repro_cmd)
    if (REPLAY / "benchmark.json").exists():
        rp = json.loads((REPLAY / "benchmark.json").read_text())
        pairs = []
        for c_del, c_rep in zip(bench["cases"], rp["cases"]):
            for m_del, m_rep in zip(c_del["metrics"], c_rep["metrics"]):
                pairs.append((c_del["role"] + "/" + c_del["resolution"] + "/" + m_del["name"],
                              m_del["computed"], m_rep["computed"]))
        worst = max((rel(a, b), k) for k, a, b in pairs)
        report["replay"] = dict(
            ran=True, cases=len(rp["cases"]),
            verdict=rp["all_passed"],
            max_rel_diff_vs_published=worst[0], worst_metric=worst[1],
            n_metrics=len(pairs),
            bit_identical=bool(worst[0] == 0.0),
            re1000_steps={c["resolution"]: c["steps"] for c in rp["cases"]
                          if c["role"] == "sensitivity-re"},
            all_cases_converged=bool(all(c["converged"] for c in rp["cases"]
                                         if c["role"] != "physics")),
        )
    else:
        report["replay"] = dict(ran=False)

    # ---------------------------------------------------------------- 7. fresh config not in plan
    import torch
    from tensorfvm.channel_periodic import PeriodicChannelConfig, PeriodicChannelSolver

    fresh = {}
    for wall_flux in ("half-cell", "quadratic"):
        cfg = PeriodicChannelConfig(nx=12, ny=48, time_step_s=5.0,
                                    wall_flux=wall_flux).with_re_h(250.0)
        solver = PeriodicChannelSolver(cfg)
        steady = solver.solve_steady()
        uu = solver.u.mean(dim=1).numpy()
        yy = (np.arange(cfg.ny) + 0.5) / cfg.ny
        ubar_fresh = 250.0 * cfg.nu / cfg.height_m
        exact = 6.0 * ubar_fresh * yy * (1.0 - yy)
        peak_fresh = 1.5 * ubar_fresh          # algebraic peak (cell centres miss H/2)
        # two independent wall-shear estimators, both from the raw u field
        tau_lo = mu * 2.0 * float(uu[0]) / cfg.dy
        tau_hi = mu * 2.0 * float(uu[-1]) / cfg.dy
        cap_lo = mu * (9.0 * float(uu[0]) - float(uu[1])) / (3.0 * cfg.dy)
        cap_hi = mu * (9.0 * float(uu[-1]) - float(uu[-2])) / (3.0 * cfg.dy)
        target = rho * cfg.body_force_m_s2 * cfg.height_m
        g_fresh = 12.0 * cfg.nu * (250.0 * cfg.nu / cfg.height_m) / cfg.height_m ** 2
        fresh[wall_flux] = dict(
            converged=steady["converged"], steps=steady["steps"],
            body_force_rel_diff_vs_own_algebra=rel(cfg.body_force_m_s2, g_fresh),
            velocity_l2_rel_peak=float(np.sqrt(np.mean((uu - exact) ** 2)) / peak_fresh),
            velocity_linf_rel_peak=float(np.abs(uu - exact).max() / peak_fresh),
            bulk_velocity=solver.bulk_velocity(),
            bulk_rel_diff_vs_algebra=rel(solver.bulk_velocity(), 250.0 * cfg.nu / cfg.height_m),
            tau_wall_sum_flux_estimator=tau_lo + tau_hi,
            tau_wall_sum_quadratic_estimator=cap_lo + cap_hi,
            tau_wall_target=target,
            momentum_balance_rel_residual=abs(tau_lo + tau_hi - target) / target,
            quadratic_estimator_rel_error=rel(cap_lo + cap_hi, target),
            profile_is_machine_exact=bool(float(np.sqrt(np.mean((uu - exact) ** 2)))
                                          / peak_fresh < 1e-9),
            max_divergence_scaled=float(solver.max_divergence() * cfg.height_m
                                        / (250.0 * cfg.nu / cfg.height_m)),
        )
    report["fresh_case_outside_plan"] = fresh
    # Discrete invariants, evaluated with the estimator each wall treatment actually
    # applies (half-cell -> mu 2u0/dy; quadratic -> mu (9u0-u1)/(3dy); the audit does
    # not borrow the solver's own bookkeeping, both estimators are recomputed from u).
    for mode, v in fresh.items():
        v["scheme_closure_estimator"] = "flux" if mode == "half-cell" else "quadratic"
        v["scheme_closure_rel_residual"] = (v["momentum_balance_rel_residual"] if mode == "half-cell"
                                            else v["quadratic_estimator_rel_error"])
    report["verdict_fresh_case"] = bool(
        all(v["scheme_closure_rel_residual"] < 1e-9 and v["max_divergence_scaled"] < 1e-9
            for v in fresh.values())
        and fresh["half-cell"]["velocity_l2_rel_peak"] < TOL
        and fresh["quadratic"]["profile_is_machine_exact"])

    # Steady state must not depend on the pseudo time step (guards against a
    # solution that was tuned by dt rather than converged).
    dt_free = {}
    base = None
    for dt in (1.0, 5.0, 25.0):
        cfg = PeriodicChannelConfig(nx=12, ny=48, time_step_s=dt).with_re_h(250.0)
        solver = PeriodicChannelSolver(cfg)
        solver.solve_steady()
        uu = solver.u.mean(dim=1).numpy()
        dt_free[str(dt)] = dict(steps=solver.history and len(solver.history),
                               bulk_velocity=solver.bulk_velocity(),
                               max_abs_u=float(np.abs(uu).max()))
        if base is None:
            base = uu.copy()
        else:
            dt_free[str(dt)]["max_abs_diff_vs_dt1"] = float(np.abs(uu - base).max())
    report["dt_independence"] = dt_free
    report["verdict_dt_independent"] = all(
        v.get("max_abs_diff_vs_dt1", 0.0) < 1e-10 for v in dt_free.values())

    # ---------------------------------------------------------------- 8. benchmark registry
    report["registry"] = dict(
        note="tensor_benchmark_cases / tensor_case_pool checked separately via psql; "
             "see audit markdown",
    )

    OUT.write_text(json.dumps(report, indent=2))
    print(json.dumps({k: report[k] for k in
                      ("manifest_sha256_match", "recomputed_reference_profile", "gate_recheck",
                       "report_sections", "replay", "verdict_fresh_case")}, indent=2)[:6000])
    print("written", OUT)


if __name__ == "__main__":
    main()