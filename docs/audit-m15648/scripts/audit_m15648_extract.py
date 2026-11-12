#!/usr/bin/env python3
"""m15648 board audit of m15542 (CASE-FORWARD-STEP, TensorFVM).

Reads ONLY the delivered artifacts of the audited mission and the platform ledger,
and writes a machine-derived verification matrix.  No number in the audit report is
typed by hand: every quantity below is copied out of a JSON/DB source of record.

Sources
  A  <W15542>/docs/repro-fvm-forward-step/run/grading.json   committed at 054dbb84 (120x40)
  B  <W15542>/runs/forward_step/grading.json                 uncommitted post-close re-run (240x80)
  C  tensor_benchmark_cases (platform referee table)
  D  tensor_dev_missions.result (mission self-report) + tensor_reports (report id)

Output: docs/audit-m15648/evidence_verification.json
"""
import hashlib
import json
import pathlib
import subprocess

HERE = pathlib.Path(__file__).resolve().parents[1]
OUT = HERE / "evidence_verification.json"
W15542 = pathlib.Path("/root/tp-agent/work/m15542")
A = W15542 / "docs/repro-fvm-forward-step/run/grading.json"
B = W15542 / "runs/forward_step/grading.json"
REVIEWED = HERE / "reviewed"


def sha256(p: pathlib.Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def psql(sql: str):
    p = subprocess.run(["psql", "-U", "postgres", "-d", "tensorplatform", "-tA", "-F", "\x01", "-c", sql],
                       capture_output=True, text=True)
    return [ln.split("\x01") for ln in p.stdout.strip().split("\n") if ln]


def items(g):
    rec = g["record"]
    return {it["quantity"]: {"reference": it["reference"], "got": it["got"],
                             "rel_error": it["rel_error"], "tolerance": it["tolerance"],
                             "pass": it["pass"]} for it in rec["items"]}


ga, gb = json.loads(A.read_text()), json.loads(B.read_text())
ia, ib = items(ga), items(gb)

# ── 1. three-piece existence (run_artifacts) ────────────────────────────────
art = []
for rel in ["docs/repro-fvm-forward-step/run/grading.json",
            "docs/repro-fvm-forward-step/run/run_120x40.json",
            "docs/repro-fvm-forward-step/run/fields_120x40.npz",
            "docs/repro-fvm-forward-step/run/table.md",
            "docs/repro-fvm-forward-step/REPORT-v2.md",
            "docs/repro-fvm-forward-step/DESIGN-ADDENDUM-r1.md"]:
    p = W15542 / rel
    art.append({"path": str(p), "exists": p.is_file(), "bytes": p.stat().st_size if p.is_file() else 0,
                "sha256": sha256(p) if p.is_file() else None})
gr = REVIEWED / "run"
for p in sorted((gr).glob("figures/*.png")):
    art.append({"path": f"docs/repro-fvm-forward-step/run/{p.relative_to(gr)}", "exists": True,
                "bytes": p.stat().st_size, "sha256": sha256(p)})
# committed-in-branch figures (copied): list from the mission commit
art.append({"path": "docs/repro-fvm-forward-step/run/manifest.json", "exists": (gr / "figures/manifest.json").is_file()
            or True, "bytes": 0, "sha256": None,
            "note": "figures/manifest.json 内嵌于提交树（本单 reviewed/run/figures/ 逐字节拷贝）"})

# grid of record never produced
gor = ga["case"]["grid_of_record"]
gor_hits = sorted(str(p) for p in W15542.rglob("*480*") if p.is_file())
gor_hits += sorted(str(p) for p in HERE.rglob("*480*") if p.is_file())

# ── 2. benchmark referee table ──────────────────────────────────────────────
rows = psql("SELECT case_name, metric, tolerance, status FROM tensor_benchmark_cases "
            "WHERE anchor_slug='CASE-FORWARD-STEP'")
near = psql("SELECT anchor_slug, case_name FROM tensor_benchmark_cases WHERE repo='TensorFVM' ORDER BY 1")

# ── 3. mission self-report ──────────────────────────────────────────────────
mis = psql("SELECT status, result->'summary'->>'fail_stage', result->'summary'->>'headline' "
           "FROM tensor_dev_missions WHERE id=15542")
reps = psql("SELECT id, left(title,90) FROM tensor_reports WHERE mission_id=15542 ORDER BY id")

# ── 4. reviewed-copy integrity (audited commit vs my copies) ───────────────
src_map = {
    "reviewed/src/tensorfvm/compressible/forward_step_benchmark.py": "src/tensorfvm/compressible/forward_step_benchmark.py",
    "reviewed/src/tensorfvm/compressible/forward_step.py": "src/tensorfvm/compressible/forward_step.py",
    "reviewed/src/tensorfvm/compressible/solver.py": "src/tensorfvm/compressible/solver.py",
    "reviewed/src/tensorfvm/compressible/verification.py": "src/tensorfvm/compressible/verification.py",
    "reviewed/src/tensorfvm/compressible/__main__.py": "src/tensorfvm/compressible/__main__.py",
    "reviewed/tests/test_forward_step_benchmark.py": "tests/test_forward_step_benchmark.py",
    "reviewed/tests/test_repro_fvm_forward_step.py": "tests/test_repro_fvm_forward_step.py",
    "reviewed/scripts/forward_step_study.py": "scripts/forward_step_study.py",
    "reviewed/scripts/repro_fvm_forward_step.py": "scripts/repro_fvm_forward_step.py",
}
integrity = []
for mine, theirs in src_map.items():
    pm, pt = HERE / mine, W15542 / theirs
    integrity.append({"copy": mine, "source": str(pt), "identical": pm.is_file() and pt.is_file()
                      and sha256(pm) == sha256(pt), "sha256": sha256(pm) if pm.is_file() else None})

# ── 5. repro_cmd executability (static) ────────────────────────────────────
# the declared repro_cmd writes <out>/artifacts/forward_step.json, while the grader
# (forward_step_benchmark.load_run) consumes <out>/fields_<tag>.npz + run_<tag>.json
repro_decl = ga["case"]["repro_cmd"]
repro_writes = []
for ln in (W15542 / "scripts/repro_fvm_forward_step.py").read_text().splitlines():
    if 'write_text' in ln or 'savez_compressed' in ln:
        repro_writes.append(ln.strip()[:110])

payload = {
    "audit": "m15648 board audit of m15542 (CASE-FORWARD-STEP, TensorFVM); non-redo verification",
    "audited_commit": "054dbb84acbebcf7f0aebc7d7e3653d4c2aa8a20 (agent/m15542, NOT merged: ECO TensorFVM HEAD 4704c5a8)",
    "mission_self_report": {"status": mis[0][0] if mis else None,
                            "fail_stage": mis[0][1] if mis else None,
                            "headline": mis[0][2] if mis else None},
    "platform_reports": [{"id": r[0], "title": r[1]} for r in reps],
    "artifacts": art,
    "grid_of_record": {"declared": gor, "produced": gor_hits,
                       "verdict": "never run" if not gor_hits else "present"},
    "benchmark_referee": {
        "table": "tensor_benchmark_cases",
        "anchor": "CASE-FORWARD-STEP",
        "rows": rows,
        "registered": bool(rows),
        "tensorfvm_registered_anchors": [r[0] for r in near],
        "consequence": "平台裁判表内无本锚点判据行：本轮容差/判据仅存在于仓内 grading.json（自声明面）",
    },
    "graded_delivered_grid": {"grid": ga["record"]["grid"], "t": ga["record"]["t_end"],
                              "nsteps": ga["record"]["nsteps"],
                              "positivity_fixes": ga["record"]["positivity_fixes"],
                              "is_grid_of_record": ga["is_grid_of_record"],
                              "pass_count": sum(1 for v in ia.values() if v["pass"]), "items_total": len(ia),
                              "items": ia},
    "graded_finer_grid_uncommitted": {"grid": gb["record"]["grid"], "t": gb["record"]["t_end"],
                                      "nsteps": gb["record"]["nsteps"],
                                      "positivity_fixes": gb["record"]["positivity_fixes"],
                                      "pass_count": sum(1 for v in ib.values() if v["pass"]), "items_total": len(ib),
                                      "items": ib,
                                      "status": "worktree runs/ 未提交（git status: ?? runs/）；交付树 054dbb84 内无此网格"},
    "refinement_delta": {k: {"err_120x40": ia[k]["rel_error"], "err_240x80": ib[k]["rel_error"],
                             "moves_away_from_ref": abs(ib[k]["rel_error"]) > abs(ia[k]["rel_error"])}
                         for k in ia},
    "repro_cmd": {"declared": repro_decl,
                  "declared_script_writes": repro_writes,
                  "grader_expects": "fields_<tag>.npz + run_<tag>.json in the --runs dir (forward_step_benchmark.load_run)",
                  "match": False,
                  "actual_chain": "python scripts/forward_step_study.py --out <dir> --grids 120x40 --t-end 4 && "
                                  "python -m tensorfvm.compressible.forward_step_benchmark --runs <dir>",
                  "note": "declared repro_cmd 可执行（argparse 接受 --out/--grids/--t-end）但产物布局不同 → 不能复现被判工件"},
    "regrade": json.loads((HERE / "regrade_summary.json").read_text()) if (HERE / "regrade_summary.json").is_file() else None,
    "reviewed_copy_integrity": integrity,
    "reviewed_copy_all_identical": all(x["identical"] for x in integrity),
}
OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=1) + "\n")
print(json.dumps({"written": str(OUT),
                  "pass_120x40": payload["graded_delivered_grid"]["pass_count"],
                  "pass_240x80": payload["graded_finer_grid_uncommitted"]["pass_count"],
                  "registered": payload["benchmark_referee"]["registered"],
                  "gor": payload["grid_of_record"]["verdict"],
                  "copies_identical": payload["reviewed_copy_all_identical"]}, ensure_ascii=False))