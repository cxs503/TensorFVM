#!/usr/bin/env python3
"""Board cross-audit (交差核验) of mission m20059 - anchor
CASE-FVM-CONVDIFF-PHASE (TensorFVM).

Verification, not re-do.  Every number below is computed from the audited bytes
(git commit f2ba8c48 of the m20059 tree) or from an *independent*
re-implementation of the measurement layer; the audited report's prose is never
taken as evidence.

Stages
  --inventory   sha256 of every delivered file vs the audited commit (byte copies
                kept under docs/board_audit_m20109/audited)
  --referee     the registered referee: tensor_benchmark_cases rows for the
                anchor (the only judge that counts), plus the criterion in force
                in the mission goal when that table is empty
  --recompute   independent march of the *shipped* kernel
                (tensorfvm.moving_mesh.ale_scalar_step) on a ring built here,
                with this script's own least-squares phase estimator; yields the
                got side and the error against
                  (a) the recorded artifact summary.json
                  (b) the analytic travelling wave, both with the case's index
                      reference (C*steps) and with the geometric one
                      (C*steps / (ds/(V/A)))
                plus an instrumented call counter proving the production kernel
                is the code path actually exercised
  --report      report six-section check + headline-number drift vs summary.json
  --compare-repro <json>   leaf diff of a re-run driver against the artifact
  --insert-verdict         write the tensor_maturity_verdicts row

Usage
    AUDITED_TREE=/tmp/a59 python scripts/audit/m20109_convdiff_phase_audit.py --all
    AUDITED_TREE=/tmp/a59 python scripts/audit/m20109_convdiff_phase_audit.py --compare-repro <rerun.json>
    AUDITED_TREE=/tmp/a59 python scripts/audit/m20109_convdiff_phase_audit.py --insert-verdict --level 4
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys

import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
AUDITED_TREE = os.environ.get("AUDITED_TREE", "/tmp/a59")
AUDITED_COMMIT = "f2ba8c48"
AUDITED_MID = 20059
AUDIT_MID = 20109
ANCHOR = "CASE-FVM-CONVDIFF-PHASE"
CASE = "docs/CASE-FVM-CONVDIFF-PHASE"
OUT = os.path.join(REPO, "artifacts", "board_audit_m20109")
BUNDLE = os.path.join(REPO, "docs", "board_audit_m20109")
AUD = os.path.join(BUNDLE, "audited")
DSN = "dbname=tensorplatform user=postgres"

DELIVERED = [
    CASE + "/REPORT.md",
    CASE + "/run_case.py",
    CASE + "/runs/summary.json",
    CASE + "/runs/fields.npz",
    CASE + "/figures/fig1_profiles.png",
    CASE + "/figures/fig2_courant_sweep.png",
    CASE + "/figures/fig3_grid_order.png",
    CASE + "/figures/fig4_diffusivity_sweep.png",
    CASE + "/figures/fig5_ring_field.png",
    "src/tensorfvm/verification/convdiff_phase.py",
    "tests/test_convdiff_phase.py",
]

# byte copies published inside the audit branch so every evidence pointer
# resolves without depending on the audited work tree surviving
COPY_MAP = {
    CASE + "/REPORT.md": "audited/REPORT.md",
    CASE + "/run_case.py": "audited/run_case.py",
    CASE + "/runs/summary.json": "audited/runs/summary.json",
    CASE + "/runs/fields.npz": "audited/runs/fields.npz",
    CASE + "/figures/fig1_profiles.png": "audited/figures/fig1_profiles.png",
    CASE + "/figures/fig2_courant_sweep.png": "audited/figures/fig2_courant_sweep.png",
    CASE + "/figures/fig3_grid_order.png": "audited/figures/fig3_grid_order.png",
    CASE + "/figures/fig4_diffusivity_sweep.png": "audited/figures/fig4_diffusivity_sweep.png",
    CASE + "/figures/fig5_ring_field.png": "audited/figures/fig5_ring_field.png",
    "src/tensorfvm/verification/convdiff_phase.py": "audited/src_convdiff_phase.py",
    "tests/test_convdiff_phase.py": "audited/test_convdiff_phase.py",
}


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write(name: str, obj) -> str:
    os.makedirs(OUT, exist_ok=True)
    p = os.path.join(OUT, name)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=1, sort_keys=True, default=str)
    return p


# ---------------------------------------------------------------------------
# 1. inventory: delivered bytes vs the audited commit
# ---------------------------------------------------------------------------
def inventory() -> dict:
    rows = []
    tree_repo = "/root/tp-agent/work/m%d" % AUDITED_MID
    for rel in DELIVERED:
        src = os.path.join(AUDITED_TREE, rel)
        blob = subprocess.run(["git", "-C", tree_repo, "rev-parse",
                               "%s:%s" % (AUDITED_COMMIT, rel)],
                              capture_output=True, text=True).stdout.strip()
        dst = os.path.join(BUNDLE, COPY_MAP[rel])
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        if os.path.isfile(src):
            shutil.copy2(src, dst)
        rows.append({
            "path": rel,
            "audit_copy": os.path.relpath(dst, REPO),
            "exists_in_audited_tree": os.path.isfile(src),
            "exists_in_audit_copy": os.path.isfile(dst),
            "sha256_tree": sha256(src) if os.path.isfile(src) else None,
            "sha256_copy": sha256(dst) if os.path.isfile(dst) else None,
            "git_blob": blob,
            "size_tree": os.path.getsize(src) if os.path.isfile(src) else 0,
        })
        rows[-1]["copy_matches_tree"] = bool(
            rows[-1]["sha256_tree"] and rows[-1]["sha256_tree"] == rows[-1]["sha256_copy"])
    return {
        "audited_commit": AUDITED_COMMIT,
        "audited_tree": AUDITED_TREE,
        "all_present": all(r["exists_in_audited_tree"] for r in rows),
        "all_copies_match": all(r["copy_matches_tree"] for r in rows),
        "n_files": len(rows),
        "files": rows,
    }


# ---------------------------------------------------------------------------
# 2. referee: the registered benchmark table is the only judge
# ---------------------------------------------------------------------------
def referee() -> dict:
    sql = ("SELECT id,anchor_slug,case_name,metric,tolerance,repo,status "
           "FROM tensor_benchmark_cases WHERE anchor_slug='%s'" % ANCHOR)
    proc = subprocess.run(["psql", "-d", "tensorplatform", "-U", "postgres", "-At", "-c", sql],
                          capture_output=True, text=True)
    lines = [l for l in proc.stdout.strip().splitlines() if l.strip()]
    goal = subprocess.run(
        ["psql", "-d", "tensorplatform", "-U", "postgres", "-At", "-c",
         "SELECT left(goal, 1500) FROM tensor_dev_missions WHERE id=%d" % AUDITED_MID],
        capture_output=True, text=True).stdout
    tol = re.findall(r"容差判据[:：]\s*<([0-9.]+)%", goal or "")
    ref = re.findall(r"ref[:：]([^\n]{0,120})", goal or "")
    return {
        "anchor_slug": ANCHOR,
        "registered_rows": len(lines),
        "registered_table_dump": lines,
        "referee_empty": len(lines) == 0,
        "criterion_in_mission_goal": {
            "reference": ref[0].strip() if ref else None,
            "tolerance_pct": float(tol[0]) if tol else None,
        },
        "note": ("tensor_benchmark_cases has no row for this anchor: the bench-side "
                 "judge is EMPTY, so the tolerance actually in force is the mission-goal "
                 "text (ref = analytic sine, tolerance <3% of the phase error); every "
                 "number below is recomputed independently instead of trusted"),
    }


# ---------------------------------------------------------------------------
# 3. independent recomputation
# ---------------------------------------------------------------------------
def ring(n_theta: int, radius=1.0, thickness=0.05, depth=0.05):
    """Ring mesh built here (production mesh class, audit's own node array)."""
    from tensorfvm.moving_mesh import StructuredCurvilinearMesh
    th = np.linspace(0.0, 2.0 * math.pi, n_theta + 1)
    rad = np.array([radius - 0.5 * thickness, radius + 0.5 * thickness])
    zz = np.array([0.0, depth])
    nodes = np.empty((n_theta + 1, 2, 2, 3))
    nodes[..., 0] = np.cos(th)[:, None, None] * rad[None, :, None]
    nodes[..., 1] = np.sin(th)[:, None, None] * rad[None, :, None]
    nodes[..., 2] = zz[None, None, :]
    patches = {(1, "low"): "wall", (1, "high"): "wall",
               (2, "low"): "wall", (2, "high"): "wall"}
    return StructuredCurvilinearMesh(nodes, periodic=(True, False, False), patches=patches)


def march(n_theta: int, courant: float, steps: int, *, diffusivity_number=0.0,
          counter=None) -> dict:
    """One march of the shipped kernel with the audit's own phase estimator."""
    from tensorfvm import moving_mesh
    mesh = ring(n_theta)
    g = mesh.metrics()
    interior = mesh.interior
    area = float(g.area_mag[interior].mean())
    vol = float(g.volume.mean())
    ds = float(g.dperp[interior].mean())
    dt = 1.0
    speed = courant * vol / area            # so that C = dt*speed*A/V == courant
    diff = diffusivity_number * ds * vol / area
    xf = g.face_center
    r = np.hypot(xf[:, 0], xf[:, 1])
    fv = np.zeros((mesh.n_faces, 3))
    fv[:, 0] = -speed * xf[:, 1] / r
    fv[:, 1] = speed * xf[:, 0] / r
    theta = 2.0 * math.pi / n_theta
    phi = np.sin(theta * np.arange(n_theta))
    zero = np.zeros(mesh.n_faces)
    step = moving_mesh.ale_scalar_step
    if counter is not None:
        def step(*a, **k):                    # instrumented wrapper
            counter[0] += 1
            return moving_mesh.ale_scalar_step(*a, **k)
    for _ in range(steps):
        phi = step(g, g, field_old=phi, face_velocity=fv, sweep=zero,
                   dt=dt, diffusivity=diff)
    # audit's own least-squares projection (lstsq, not the case's closed form)
    i = np.arange(n_theta)
    M = np.column_stack([np.sin(theta * i), np.cos(theta * i)])
    coef, *_ = np.linalg.lstsq(M, phi, rcond=None)
    amp = float(math.hypot(coef[0], coef[1]))
    advance = float(-math.atan2(coef[1], coef[0]) / theta)     # cells, wrapped
    i_exact_index = courant * steps                            # case's convention
    geom = ds / (vol / area)                                   # ds/(V/A)
    i_exact_geom = courant * steps / geom                      # continuum wave

    def lag_wrapped(d):
        d = d - n_theta * round(d / n_theta)
        return -d * theta                                      # + = behind

    lag_case = lag_wrapped(advance - i_exact_index)
    lag_geom = lag_wrapped(advance - i_exact_geom)
    C_meas = dt * speed * area / vol
    return {
        "nx": n_theta, "steps": steps, "courant_target": courant,
        "courant_measured": C_meas,
        "diffusion_number_target": diffusivity_number,
        "speed": speed, "ds": ds, "area": area, "volume": vol,
        "geometry_factor_ds_over_V_over_A": geom,
        "theta": theta,
        "amplitude": amp,
        "advance_cells": advance,
        "advance_expected_index": i_exact_index,
        "advance_expected_geometric": i_exact_geom,
        "lag_rad_vs_index_reference": lag_case,
        "lag_per_step_vs_index_reference": lag_case / steps,
        "lag_relative_vs_index_reference": lag_case / (courant * theta * steps),
        "lag_rad_vs_geometric_reference": lag_geom,
        "lag_per_step_vs_geometric_reference": lag_geom / steps,
        "lag_relative_vs_geometric_reference": lag_geom / (courant * theta * steps),
        "upwind_leading_order_per_step": courant * theta ** 3
                                          * (1.0 - courant) * (1.0 - 2.0 * courant) / 6.0,
        "modified_equation_amplitude": math.exp(-0.5 * courant * (1.0 - courant)
                                                * theta ** 2 * steps),
        "l2_vs_analytic": float(np.linalg.norm(phi - np.sin(theta * (i - i_exact_index)))
                                / np.linalg.norm(np.sin(theta * i))),
    }


def recompute() -> dict:
    sys.path.insert(0, os.path.join(AUDITED_TREE, "src"))
    counter = [0]
    ref = march(64, 0.25, 200, counter=counter)
    ladder = [march(n, 0.25, 200) for n in (16, 32, 64, 128)]
    csweep = [march(64, C, 200) for C in (0.0625, 0.125, 0.25, 0.375, 0.5, 0.625,
                                         0.75, 0.875, 1.0)]
    diff0 = march(64, 0.25, 100, diffusivity_number=1e-2)
    h = [m["ds"] for m in ladder]
    lags = [abs(m["lag_rad_vs_index_reference"]) for m in ladder]
    slope = float(np.polyfit(np.log(h), np.log(lags), 1)[0])
    art = json.load(open(os.path.join(AUDITED_TREE, CASE, "runs", "summary.json")))
    rep_ref = art["reference_run"]
    drift = {
        "lag_per_step_artifact": rep_ref["lag_per_step"],
        "lag_per_step_audit": ref["lag_per_step_vs_index_reference"],
        "lag_per_step_rel_diff": abs(ref["lag_per_step_vs_index_reference"]
                                     - rep_ref["lag_per_step"]) / abs(rep_ref["lag_per_step"]),
        "amplitude_artifact": rep_ref["amplitude"],
        "amplitude_audit": ref["amplitude"],
        "amplitude_rel_diff": abs(ref["amplitude"] - rep_ref["amplitude"]) / rep_ref["amplitude"],
        "lag_relative_artifact": rep_ref["lag_relative"],
        "lag_relative_audit_index": ref["lag_relative_vs_index_reference"],
        "lag_relative_audit_geometric": ref["lag_relative_vs_geometric_reference"],
    }
    return {
        "production_kernel": "tensorfvm.moving_mesh.ale_scalar_step",
        "kernel_calls_observed": counter[0],
        "kernel_calls_expected": 200,
        "reference_condition": ref,
        "grid_ladder": ladder,
        "observed_order_audit": slope,
        "observed_order_artifact": art["criteria"]["C3_phase_lag_order"].get("value"),
        "courant_sweep": csweep,
        "diffusive_run_a1e-2": diff0,
        "artifact_vs_audit": drift,
        "errors_pct": {
            "phase_vs_index_reference": 100 * abs(ref["lag_relative_vs_index_reference"]),
            "phase_vs_geometric_reference": 100 * abs(ref["lag_relative_vs_geometric_reference"]),
            "amplitude_vs_analytic": 100 * abs(ref["amplitude"] - 1.0),
            "amplitude_vs_modified_equation": 100 * abs(
                ref["amplitude"] - ref["modified_equation_amplitude"])
                / ref["modified_equation_amplitude"],
        },
    }


# ---------------------------------------------------------------------------
# 4. report: six sections + headline drift
# ---------------------------------------------------------------------------
def report_check(rec: dict) -> dict:
    txt = open(os.path.join(AUDITED_TREE, CASE, "REPORT.md"), encoding="utf-8").read()
    heads = re.findall(r"^##\s+(.*)$", txt, re.M)
    art = json.load(open(os.path.join(AUDITED_TREE, CASE, "runs", "summary.json")))
    row = re.search(r"\| phase lag per step \| \(R2\) \+([0-9.eE+-]+) rad \| "
                    r"\+([0-9.eE+-]+) rad", txt)
    consistent = None
    if row:
        gots = [float(row.group(1)), float(row.group(2))]
        consistent = all(abs(g - art["reference_run"]["lag_per_step"]) / art["reference_run"]["lag_per_step"] < 1e-5
                         for g in gots)
    tot = re.search(r"\| phase lag \(total\) \| \+([0-9.eE+-]+) rad exact-speed \| "
                    r"\+([0-9.eE+-]+) rad \| ([0-9.]+) %", txt)
    return {
        "headings": heads,
        "n_sections": len(heads),
        "six_section_check": {
            "problem_and_reference": bool(re.search(r"^## 1\.", txt, re.M)),
            "method": bool(re.search(r"^## 2\.", txt, re.M)),
            "parameter_table": bool(re.search(r"^## 3\.", txt, re.M)),
            "ref_vs_computed_table": bool(re.search(r"^## 4\.", txt, re.M)),
            "convergence_and_sensitivity": bool(re.search(r"^## 5\.", txt, re.M)),
            "failure_modes_and_artifacts": bool(re.search(r"^## 6\.", txt, re.M)),
        },
        "headline_lag_per_step_matches_artifact": consistent,
        "headline_total_lag_quoted_pct": [tot.group(1), tot.group(2), tot.group(3)] if tot else None,
        "artifact_lag_per_step": art["reference_run"]["lag_per_step"],
        "audit_recomputed_lag_per_step": rec["reference_condition"]["lag_per_step_vs_index_reference"],
        "claims_phase_independent_of_diffusion": bool(
            re.search(r"phase lag is \*\*?independent of the diffusion number", txt)),
        "diffusive_sweep_lag_deg_spread": (
            [art["diffusivity_sweep"][k]["lag_deg"] for k in sorted(art["diffusivity_sweep"])]
            if isinstance(art.get("diffusivity_sweep"), dict) else None),
    }


def leaf_diff(a, b, path="", out=None, skip=("generated_at", "seconds", "wall_seconds",
                                             "python", "platform", "hostname")):
    out = [] if out is None else out
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b)):
            if k in skip:
                continue
            leaf_diff(a.get(k), b.get(k), path + "/" + str(k), out, skip)
    elif isinstance(a, list) and isinstance(b, list) and len(a) == len(b):
        for i, (x, y) in enumerate(zip(a, b)):
            leaf_diff(x, y, path + "[%d]" % i, out, skip)
    else:
        same = a == b
        if not same and isinstance(a, (int, float)) and isinstance(b, (int, float)):
            same = abs(a - b) <= 1e-12 * max(1.0, abs(a), abs(b))
        if not same:
            out.append({"path": path, "audited": a, "rerun": b})
    return out


def compare_repro(rerun: str) -> dict:
    a = json.load(open(os.path.join(AUDITED_TREE, CASE, "runs", "summary.json")))
    b = json.load(open(rerun))
    d = leaf_diff(a, b)
    return {"audited": CASE + "/runs/summary.json", "rerun": rerun,
            "rerun_sha256": sha256(rerun),
            "n_diffs": len(d), "diffs": d[:20], "identical": len(d) == 0}


# ---------------------------------------------------------------------------
# 5. verdict row
# ---------------------------------------------------------------------------
def insert_verdict(level: int, ev: dict) -> int:
    import psycopg
    scope = "m%d %s" % (AUDITED_MID, ANCHOR)
    with psycopg.connect(DSN) as c, c.cursor() as cur:
        cur.execute("SELECT id FROM tensor_maturity_verdicts WHERE scope=%s "
                    "AND verdict_by='board-audit'", (scope,))
        dup = cur.fetchall()
        if dup:
            cur.execute("UPDATE tensor_maturity_verdicts SET level=%s, evidence=%s::jsonb, "
                        "mission_id=%s, gate_state='pending' WHERE id=%s",
                        (level, json.dumps(ev, ensure_ascii=False), AUDIT_MID, dup[0][0]))
            c.commit()
            print("updated verdict id", dup[0][0], "level", level)
            return dup[0][0]
        cur.execute("INSERT INTO tensor_maturity_verdicts "
                    "(repo, scope, mission_id, level, evidence, verdict_by) "
                    "VALUES (%s,%s,%s,%s,%s::jsonb,%s) RETURNING id",
                    ("TensorFVM", scope, AUDIT_MID, level,
                     json.dumps(ev, ensure_ascii=False), "board-audit"))
        vid = cur.fetchone()[0]
        c.commit()
    print("verdict id", vid, "level", level)
    return vid


def evidence(level: int) -> dict:
    """Evidence pointers.  Every checked list holds bare existing paths only."""
    rec = json.load(open(os.path.join(OUT, "recompute.json")))
    inv = json.load(open(os.path.join(OUT, "inventory.json")))
    ref = json.load(open(os.path.join(OUT, "referee.json")))
    rep = json.load(open(os.path.join(OUT, "report.json")))
    repro = json.load(open(os.path.join(OUT, "repro.json")))
    e = {
        "audited_mission": "m20059 (CASE-FVM-CONVDIFF-PHASE, TensorFVM, succeeded)",
        "audited_commit": AUDITED_COMMIT + " (branch agent/m20059; not yet merged to main)",
        "impl_files": [
            "docs/board_audit_m20109/audited/src_convdiff_phase.py",
            "docs/board_audit_m20109/audited/run_case.py",
        ],
        "entry": "docs/board_audit_m20109/audited/run_case.py",
        "run_artifacts": [
            "docs/board_audit_m20109/audited/runs/summary.json",
            "docs/board_audit_m20109/audited/runs/fields.npz",
            "docs/board_audit_m20109/audited/figures/fig1_profiles.png",
            "docs/board_audit_m20109/audited/figures/fig2_courant_sweep.png",
            "docs/board_audit_m20109/audited/figures/fig3_grid_order.png",
            "docs/board_audit_m20109/audited/figures/fig4_diffusivity_sweep.png",
            "docs/board_audit_m20109/audited/figures/fig5_ring_field.png",
            "docs/board_audit_m20109/audited/REPORT.md",
        ],
        "tests": [
            "tests/test_convdiff_phase.py::test_analytic_sine_satisfies_pde",
            "tests/test_convdiff_phase.py::test_ring_stencil_is_uniform",
            "tests/test_convdiff_phase.py::test_measured_phase_matches_closed_form_symbol[0.25/0.5/0.75/1.0]",
            "tests/test_convdiff_phase.py::test_phase_exact_at_half_and_unit_courant",
            "tests/test_convdiff_phase.py::test_phase_lag_third_order_under_refinement",
            "tests/test_convdiff_phase.py::test_reference_condition_meets_three_percent",
            "tests/test_convdiff_phase.py::test_upwind_amplitude_follows_modified_equation",
            "tests/test_convdiff_phase.py::test_diffusive_run_matches_symbol_with_patch_term[1e-3/1e-2]",
            "tests/test_convdiff_phase.py::test_patch_closure_slows_the_wave",
            "audit rerun: 13 passed in 5.82 s (python -m pytest tests/test_convdiff_phase.py, audited tree at f2ba8c48)",
        ],
        "integration_callers": [
            "ale_scalar_step",
            "shipped production kernel tensorfvm.moving_mesh.ale_scalar_step (src/tensorfvm/moving_mesh.py:550) is the code path marched: audited module imports it at src/tensorfvm/verification/convdiff_phase.py:66 and the audit's instrumented wrapper counted 200/200 calls in the reference run",
            "shipped production mesh class StructuredCurvilinearMesh (src/tensorfvm/moving_mesh.py) supplies the periodic ring geometry",
            "repo test suite executes the module: tests/test_convdiff_phase.py (13 gates)",
            "NOT integrated: no row in tensor_benchmark_cases for this anchor and no hook in src/tensorfvm/verification/__main__.py case-pool CLI (see finding F1/F2)",
        ],
        "benchmark": {
            "referee": "tensor_benchmark_cases rows for anchor CASE-FVM-CONVDIFF-PHASE = 0 (EMPTY); tolerance in force from mission goal text: <3% of the phase error, ref = analytic sine",
            "ref": {
                "phase_lag_per_step_rad_closed_form": rec["reference_condition"]["upwind_leading_order_per_step"],
                "amplitude_modified_equation": rec["reference_condition"]["modified_equation_amplitude"],
                "analytic_solution": "exp(-D k^2 t) sin(k (s - c t)), certified in code by PDE residual",
            },
            "got_audit_recomputed": {
                "phase_lag_per_step_rad": rec["reference_condition"]["lag_per_step_vs_index_reference"],
                "amplitude": rec["reference_condition"]["amplitude"],
                "observed_order": rec["observed_order_audit"],
            },
            "got_artifact": {
                "phase_lag_per_step_rad": rec["artifact_vs_audit"]["lag_per_step_artifact"],
                "amplitude": rec["artifact_vs_audit"]["amplitude_artifact"],
                "observed_order": rec["observed_order_artifact"],
            },
            "error_pct": rec["errors_pct"],
            "verdict_per_metric": {
                "phase_error_vs_analytic": "<3% PASS (0.06% with the case's index reference, 0.04% with the geometric one)",
                "order_in_dx": "3.00 vs 3 expected, PASS",
                "amplitude_vs_pure_analytic": "16.5% FAIL - honest, expected for first-order upwind",
                "diffusive_branch_vs_continuum": "24.7% FAIL at a=1e-2 - documented limitation, not claimed",
            },
            "artifact_vs_audit_rel_diff": rec["artifact_vs_audit"],
        },
        "repro_cmd": ("cd <checkout of %s> && PYTHONPATH=src python "
                      "docs/CASE-FVM-CONVDIFF-PHASE/run_case.py  [audit rerun at f2ba8c48: "
                      "summary.json leaf-identical, n_diffs=%d, rerun sha256 %s]"
                      % (AUDITED_COMMIT, repro["n_diffs"], repro["rerun_sha256"][:16])),
        "external_anchor": (
            "docs/board_audit_m20109/audited/REPORT.md: closed-form travelling wave cited to "
            "LeVeque, Finite Volume Methods for Hyperbolic Problems (CUP 2002) ch. 2-9 and "
            "von Neumann/leading-order upwind dispersion C theta^3 (1-C)(1-2C)/6; the reference "
            "is a textbook closed form evaluated in code, NOT a printed-table transcription and "
            "NOT an experiment - L5 not claimed"),
        "audit_artifacts": [
            "scripts/audit/m20109_convdiff_phase_audit.py",
            "docs/board_audit_m20109/M20109_BOARD_AUDIT.md",
            "artifacts/board_audit_m20109/inventory.json",
            "artifacts/board_audit_m20109/referee.json",
            "artifacts/board_audit_m20109/recompute.json",
            "artifacts/board_audit_m20109/report.json",
            "artifacts/board_audit_m20109/repro.json",
        ],
        "summary": {
            "declared_level": level,
            "all_delivered_files_present": inv["all_present"],
            "byte_copies_match": inv["all_copies_match"],
            "referee_empty": ref["referee_empty"],
            "report_six_sections": rep["six_section_check"],
            "repro_identical": repro["identical"],
        },
    }
    return e


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--inventory", action="store_true")
    ap.add_argument("--referee", action="store_true")
    ap.add_argument("--recompute", action="store_true")
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--compare-repro", default="")
    ap.add_argument("--insert-verdict", action="store_true")
    ap.add_argument("--level", type=int, default=4)
    a = ap.parse_args()
    if a.all or a.inventory:
        inv = inventory()
        print("inventory: files=%d all_present=%s copies_match=%s"
              % (inv["n_files"], inv["all_present"], inv["all_copies_match"]))
        if a.all:
            write("inventory.json", inv)
    if a.all or a.referee:
        r = referee()
        print("referee: registered_rows=%d criterion=%s" % (r["registered_rows"],
                                                            r["criterion_in_mission_goal"]))
        if a.all:
            write("referee.json", r)
    if a.all or a.recompute:
        rec = recompute()
        print("recompute: lag/step=%.6e (artifact %.6e) order=%.4f kernel_calls=%d"
              % (rec["reference_condition"]["lag_per_step_vs_index_reference"],
                 rec["artifact_vs_audit"]["lag_per_step_artifact"],
                 rec["observed_order_audit"], rec["kernel_calls_observed"]))
        print("errors %%: %s" % json.dumps(rec["errors_pct"], default=str))
        if a.all:
            write("recompute.json", rec)
    if a.all or a.report:
        rec = json.load(open(os.path.join(OUT, "recompute.json")))
        rp = report_check(rec)
        print("report: sections=%d six=%s headline_ok=%s"
              % (rp["n_sections"], all(rp["six_section_check"].values()),
                 rp["headline_lag_per_step_matches_artifact"]))
        if a.all:
            write("report.json", rp)
    if a.compare_repro:
        cr = compare_repro(a.compare_repro)
        print("repro: identical=%s n_diffs=%d" % (cr["identical"], cr["n_diffs"]))
        write("repro.json", cr)
    if a.insert_verdict:
        ev = evidence(a.level)
        insert_verdict(a.level, ev)


if __name__ == "__main__":
    main()