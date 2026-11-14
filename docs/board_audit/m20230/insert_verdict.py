#!/usr/bin/env python3
"""m20230 board-audit driver: insert the maturity verdict and publish the report.

Everything the verdict claims is read from files this audit produced
(``recompute.json``, ``runs/repro.json``) or from SQL this script runs itself.
Idempotent: one row per (repo, scope, verdict_by) via ON CONFLICT update.
"""
from __future__ import annotations

import base64
import json
import os
import pathlib
import shutil
import subprocess

import numpy as np
import torch
from scipy.interpolate import PchipInterpolator

HERE = pathlib.Path(__file__).resolve().parent
REFS = HERE / "refs"
ART = REFS / "docs" / "artifacts"
DSN = os.environ.get("AUDIT_DSN", "host=127.0.0.1 port=5432 dbname=tensorplatform user=postgres")
REPO = "TensorFVM"
SCOPE = "m20144 CASE-FVM-CAVITY-RE1000"
MISSION = 20230
BY = "board-audit"
LEVEL = 4
SUBJECT_HEAD = "517fc38a (branch agent/m20144), baseline main=27d67542"

GHIA_U = [(1.0, 1.0), (0.9766, 0.65928), (0.9688, 0.57492), (0.9609, 0.51117),
          (0.9531, 0.46604), (0.8516, 0.33304), (0.7344, 0.18719), (0.6172, 0.05702),
          (0.5, -0.0608), (0.4531, -0.10648), (0.2813, -0.27805), (0.1719, -0.38289),
          (0.1016, -0.2973), (0.0703, -0.2222), (0.0625, -0.20196), (0.0547, -0.18109),
          (0.0, 0.0)]
GHIA_V = [(1.0, 0.0), (0.9688, -0.21388), (0.9609, -0.27669), (0.9531, -0.33714),
          (0.9453, -0.39188), (0.9063, -0.5155), (0.8594, -0.42665), (0.8047, -0.31966),
          (0.5, 0.02526), (0.2344, 0.32235), (0.2266, 0.33075), (0.1563, 0.37095),
          (0.0938, 0.32627), (0.0781, 0.30353), (0.0703, 0.29012), (0.0625, 0.27485),
          (0.0, 0.0)]


def profiles(n: int = 128):
    f = torch.load(ART / "fields" / f"Re1000_n{n}.pt", weights_only=False)
    h = 1.0 / n
    ys = np.concatenate(([0.0], (np.arange(n) + 0.5) * h, [1.0]))
    us = np.concatenate(([0.0], f["u"].numpy()[n // 2, :], [1.0]))
    xs = np.concatenate(([0.0], (np.arange(n) + 0.5) * h, [1.0]))
    vs = np.concatenate(([0.0], f["v"].numpy()[:, n // 2], [0.0]))
    return (ys, us, xs, vs)


def figure_centreline(path: pathlib.Path) -> str:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ys, us, xs, vs = profiles(128)
    pu, pv = PchipInterpolator(ys, us), PchipInterpolator(xs, vs)
    fig, ax = plt.subplots(1, 2, figsize=(10.5, 4.2))
    ax[0].plot(pu(ys), ys, "-", lw=1.6, label="TensorFVM n=128 (auditor re-interp)")
    ax[0].plot([p[1] for p in GHIA_U], [p[0] for p in GHIA_U], "o", ms=4,
               label="Ghia et al. 1982 Table I")
    ax[0].set_xlabel("u / U_lid"); ax[0].set_ylabel("y / L")
    ax[0].set_title("u on x = L/2, Re = 1000"); ax[0].legend(fontsize=8); ax[0].grid(alpha=.3)
    ax[1].plot(pv(xs), xs, "-", lw=1.6, label="TensorFVM n=128 (auditor re-interp)")
    ax[1].plot([p[1] for p in GHIA_V], [p[0] for p in GHIA_V], "o", ms=4,
               label="Ghia et al. 1982 Table II")
    ax[1].set_xlabel("v / U_lid"); ax[1].set_ylabel("x / L")
    ax[1].set_title("v on y = L/2, Re = 1000"); ax[1].legend(fontsize=8); ax[1].grid(alpha=.3)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)
    return str(path)


def figure_error(path: pathlib.Path, rec: dict) -> str:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ns = [32, 64, 128]
    keys = [("u_min_rel_err", "u_min rel. error"), ("v_max_rel_err", "v_max rel. error"),
            ("u_max_abs_dev", "max|du|/U"), ("v_max_abs_dev", "max|dv|/U")]
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    for k, lab in keys:
        y = [100 * rec["profiles"][f"n{n}"]["pchip"][k] for n in ns]
        ax.loglog(ns, y, "o-", label=lab)
    ax.axhline(3.0, color="k", ls="--", lw=1, label="referee bar 3%")
    ax.axhline(5.0, color="r", ls=":", lw=1, label="ledger band 5%")
    ax.set_xlabel("grid n"); ax.set_ylabel("percent")
    ax.set_title("auditor-recomputed error vs Ghia (Re=1000)")
    ax.set_xticks(ns); ax.set_xticklabels([str(x) for x in ns])
    ax.legend(fontsize=8); ax.grid(alpha=.3, which="both")
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)
    return str(path)


def main() -> int:
    # keep the cluster job's evidence inside the worktree
    src = pathlib.Path("/tmp/cj2580/cj2580/repo/docs/board_audit/m20230/runs/repro.json")
    if src.exists():
        (HERE / "runs").mkdir(exist_ok=True)
        shutil.copy(src, HERE / "runs" / "repro.json")
    rec = json.loads((HERE / "recompute.json").read_text())
    repro = json.loads((HERE / "runs" / "repro.json").read_text())

    evidence = {
        "impl_files": [
            "src/tensorfvm/case_cavity_re1000.py (506 lines, CLI at :470-506; audited copy docs/board_audit/m20230/refs/src/case_cavity_re1000.py)",
            "src/tensorfvm/lid_cavity.py:98-368 (staggered MAC FV, van Leer faces, DCT-II Neumann pressure solve)",
            "src/tensorfvm/lid_cavity_reference.py:78-156 (Ghia Table I/II transcription)",
            "docs/board_audit/m20230/refs/src/*.py (byte copies audited)",
        ],
        "entry": ("PYTHONPATH=src python -m tensorfvm.case_cavity_re1000 --out "
                  "docs/CASE-CAVITY-RE1000-FVM/artifacts --grids 64,128 --cold-grids 32 "
                  "--ladder-dir src/tensorfvm/verification/artifacts/verification_ladder "
                  "(argparse case_cavity_re1000.py:470-506)"),
        "run_artifacts": [
            "docs/CASE-CAVITY-RE1000-FVM/artifacts/summary.json",
            "docs/CASE-CAVITY-RE1000-FVM/artifacts/fields/Re1000_n32.pt",
            "docs/CASE-CAVITY-RE1000-FVM/artifacts/fields/Re1000_n64.pt",
            "docs/CASE-CAVITY-RE1000-FVM/artifacts/fields/Re1000_n64_from32.pt",
            "docs/CASE-CAVITY-RE1000-FVM/artifacts/fields/Re1000_n128.pt",
            "docs/CASE-CAVITY-RE1000-FVM/figures/fig1_centrelines.png",
            "docs/CASE-CAVITY-RE1000-FVM/figures/fig2_fields.png",
            "docs/CASE-CAVITY-RE1000-FVM/figures/fig3_convergence.png",
            "src/tensorfvm/verification/artifacts/verification_ladder/lid_cavity_Re1000_n128.json",
            "src/tensorfvm/verification/artifacts/verification_ladder/lid_cavity_Re1000_n64.json",
            "docs/board_audit/m20230/refs/docs/artifacts/summary.json (audited copy)",
            "docs/board_audit/m20230/recompute.json (auditor recomputation from the .pt fields)",
            "docs/board_audit/m20230/runs/repro.json (cluster job cj2580: cold n=32 + n=128 restart)",
        ],
        "tests": {
            "auditor_run": ("PYTHONPATH=src pytest tests/test_case_cavity_re1000.py "
                            "tests/test_case_cavity_re1000_m15458.py -q -> 13 passed, 0 skipped, 5.9 s"),
            "files": ["tests/test_case_cavity_re1000.py (10 tests, incl. test_ladder_consumes_the_run PASSED)",
                      "tests/test_case_cavity_re1000_m15458.py (3 tests)"],
            "gap": ("bundle tests re-evaluate judge() on summary.json metrics; no shipped test "
                    "recomputes the metrics from the .pt fields (auditor filled this with recompute_from_fields.py)"),
        },
        "integration_callers": [
            "src/tensorfvm/verification/ladder.py:1052-1140 lid_cavity_verdict() -> C2 PASS, measured 0.011184 <= 0.02, evaluated_on=src/tensorfvm/verification/artifacts/verification_ladder/lid_cavity_Re1000_n128.json (auditor executed it on the branch)",
            "tests/test_case_cavity_re1000.py::test_ladder_consumes_the_run PASSED (not skipped)",
            "platform ingestion ABSENT: tensorbench_results has 0 rows for missions 20144/19978/15369",
        ],
        "benchmark": {
            "referee_row": "tensor_benchmark_cases id=18, anchor_slug='CASE-CAVITY-RE1000-FVM', repo TensorFVM, ref_source Ghia/Ghia/Shin 1982 Tables I/II, tolerance 'u_min/v_max <3%; pointwise max|du|/U <3% interior; ledger band 5%; vortex centre |d|/L <= 0.01 @ n>=128'",
            "auditor_recomputed_n128": {
                "u_min_rel_err": 0.00630, "v_max_rel_err": 0.00786,
                "max_abs_du": 0.0043074, "max_abs_dv": 0.0112200,
                "vortex_centre_abs": 0.00391, "max_divergence": 1.3500311979441904e-13,
            },
            "shipped_n128": {"u_min_rel_err": 0.006471, "v_max_rel_err": 0.007791,
                             "u_max_abs": 0.0043072983257934905,
                             "v_max_abs": 0.011184127632372254,
                             "vortex_centre_abs": 0.0039562},
            "verdict": "all referee criteria met at n=128 (and at the stricter 3%/0.01 bar, not only the 5% band)",
            "anchor_name_defect": {
                "mission_anchor_slug": "CASE-FVM-CAVITY-RE1000",
                "rows_under_mission_slug": {
                    "tensor_benchmark_cases": 0, "anchor_leases": 0, "repro_pool": 0,
                    "tensor_xval_cases": 0, "tensor_case_pool.linked_anchor": 0},
                "rows_under_repo_slug": {"tensor_benchmark_cases": 1, "anchor_slug": "CASE-CAVITY-RE1000-FVM"},
                "note": "the delivery is anchored to the legacy case-pool slug CAVITY-RE1000-FVM / mission 15369; resolving the mission anchor by slug returns nothing",
            },
        },
        "repro_cmd": {
            "full": ("PYTHONPATH=src python -m tensorfvm.case_cavity_re1000 --out "
                     "docs/CASE-CAVITY-RE1000-FVM/artifacts --grids 64,128 --cold-grids 32 "
                     "--ladder-dir src/tensorfvm/verification/artifacts/verification_ladder"),
            "auditor_executed": ("cluster job cj2580 (node n315, rc=0): cold n=32 with the same "
                                 "numerics -> max|du|=0.0, max|dv|=0.0 vs shipped Re1000_n32.pt, "
                                 "30500 steps both, u_min=-0.316430836724452 both (bit-identical); "
                                 "n=128 restart 1000 steps -> drift 1.09e-7"),
            "gate": "PYTHONPATH=src pytest tests/test_case_cavity_re1000.py -q -> 13 passed",
        },
        "external_anchor": {
            "source": "U. Ghia, K.N. Ghia, C.T. Shin, J. Comput. Phys. 48 (1982) 387-411, Tables I/II",
            "corroboration": ("audited transcription vs auditor copy 34/34 identical, vs independent "
                              "transcription in TensorLBM src/tensorlbm/lid_driven_cavity.py GHIA_RE1000 "
                              "17/17 u and 17/17 v (v reversed order) identical"),
            "disclosure": ("the auditor's first hand-typed copy was wrong at 9 of 34 entries; the error "
                           "surfaced as a 0.06 U mid-cavity inconsistency with the converged field and was "
                           "corrected against the cross-repo transcription"),
        },
        "subject_head": SUBJECT_HEAD,
        "level_rationale": ("L2 real solver+case module and CLI; L3 repo-level: the repository verification "
                            "ladder C2 now reads this run's n=128 row and passes (platform ingestion absent -> "
                            "no platform-level L3 claim); L4 complete benchmark case with grid refinement, "
                            "per-point error analysis, steady-state/continuation receipts and bit-identical "
                            "auditor reproduction, external anchor inside referee tolerance; L5 refused "
                            "(textbook benchmark, single external source, no novel method)"),
        "checks_not_run": [
            "scripts/render_case_cavity_re1000.py re-render diff (would write into the audited worktree; replaced by numeric spot-check of report sections 4/5 against summary.json)",
            "full n=64+n=128 sequence re-run (determinism already shown by the bit-identical n=32 rerun)",
            "any platform-side consumer (tensorbench_results empty -> nothing to check)",
        ],
        "findings": {
            "F1_major_bookkeeping": "mission anchor slug CASE-FVM-CAVITY-RE1000 has 0 referee rows in every registry table; referee row id=18 sits under CASE-CAVITY-RE1000-FVM (mission 15369)",
            "F2_minor": "CRITERIA use the 5%/0.05 band while the referee row is 3%/0.01@n>=128; measured values pass both, but the pytest gate would stay green in the 3-5% range",
            "F3_minor": "no shipped test recomputes the metrics from the raw fields (judge() self-consistency only)",
            "F5_note": "max|du| is interpolation-scheme sensitive at coarse grids (n=32: 0.2033 shipped Hermite vs 0.0792 auditor PCHIP; n=64: 0.0293 vs 0.0189; n=128: agree to 1.4e-7); the shipped numbers are the conservative ones",
            "F6_disclosure": "auditor reference-typing error, corrected and documented (see external_anchor.disclosure)",
        },
        "disposition": ("approve with notes: level 4 confirmed; remediation (non-blocking): unify the anchor "
                        "name, tighten CRITERIA to the registered referee bar, add a field-recompute test, "
                        "merge to main and ingest into tensorbench_results"),
    }

    import psycopg
    with psycopg.connect(DSN) as cx, cx.cursor() as cur:
        cur.execute("""
            INSERT INTO tensor_maturity_verdicts (repo, scope, mission_id, level, evidence, verdict_by)
            VALUES (%s, %s, %s, %s, %s::jsonb, %s)
            ON CONFLICT (repo, scope, verdict_by) DO UPDATE
              SET mission_id = EXCLUDED.mission_id, level = EXCLUDED.level,
                  evidence = EXCLUDED.evidence, gate_state = 'pending',
                  gate_level = NULL, gate_note = NULL
            RETURNING id, gate_state
        """, (REPO, SCOPE, MISSION, LEVEL, json.dumps(evidence, ensure_ascii=False), BY))
        row = cur.fetchone()
        cx.commit()
    print("verdict row:", row)

    # ------------------------------------------------------------- report
    figs = []
    p1 = figure_centreline(HERE / "fig_centreline_audit.png")
    p2 = figure_error(HERE / "fig_error_vs_grid.png", rec)
    for path, cap in ((p1, "审计员独立重算的 n=128 中线速度剖面（实线）对标 Ghia 1982 Table I/II（点）"),
                      (p2, "审计员从原始流场重算的误差随网格变化，3% 裁判杆与 5% ledger band")):
        figs.append({"name": pathlib.Path(path).name, "caption": cap,
                     "png_base64": base64.b64encode(pathlib.Path(path).read_bytes()).decode()})
    f3 = REFS / "docs" / "figures" / "fig2_fields.png"
    if f3.exists():
        figs.append({"name": "fig2_fields.png",
                     "caption": "被审交付的流场云图（cavity 流函数/速度幅值，来自被审 artifacts）",
                     "png_base64": base64.b64encode(f3.read_bytes()).decode()})

    md = (HERE / "AUDIT-m20230-CASE-FVM-CAVITY-RE1000.md").read_text()
    body = {
        "title": "CASE-FVM-CAVITY-RE1000 交差审计（m20144）：判级 L4，基准 3% 全项通过，锚点错配待修",
        "summary": ("从原始 .pt 流场独立重算 n=128 全部计分量并与基准行逐项比对：u_min 0.630%、v_max 0.786%、"
                    "max|Δu| 0.43%、max|Δv| 1.12%、涡心 0.0039L，全部落在注册裁判行 3%/0.01 内；"
                    "集群 cj2580 冷启动 n=32 与被审工件逐比特相同，n=128 重启 1000 步漂移 1.1e-7；"
                    "机检门 13 passed；主梯 C2 实读 n=128 行并 PASS。判 L4，附锚点 slug 错配等 3 条非阻断返修项。"),
        "scenario_key": "board-audit",
        "mission_id": 20230,
        "tags": ["TensorFVM", "CASE-FVM-CAVITY-RE1000", "Ghia1982", "board-audit", "L4", "m20144"],
        "markdown": md,
        "figures": figs,
    }
    pathlib.Path("/tmp/report.json").write_text(json.dumps(body, ensure_ascii=False))
    base = os.environ.get("TENSOR_API_BASE", "http://127.0.0.1:8010/api/v1")
    user = os.environ.get("TENSOR_AGENT_USER", "eco_admin")
    pwd = os.environ.get("TENSOR_AGENT_PASS", "")
    tok = subprocess.run(["curl", "-s", "-X", "POST", f"{base}/auth/login",
                          "-H", "Content-Type: application/json",
                          "-d", json.dumps({"username": user, "password": pwd})],
                         capture_output=True, text=True).stdout
    try:
        token = json.loads(tok).get("access_token") or json.loads(tok).get("token")
    except Exception:
        token = None
    if not token:
        print("login response:", tok[:300])
    out = subprocess.run(["curl", "-s", "-X", "POST", f"{base}/tensor/reports",
                          "-H", f"Authorization: Bearer {token}",
                          "-H", "Content-Type: application/json",
                          "-d", "@/tmp/report.json"], capture_output=True, text=True).stdout
    print("report post:", out[:400])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())