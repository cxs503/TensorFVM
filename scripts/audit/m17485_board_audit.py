#!/usr/bin/env python3
"""Board cross-audit (交差核验) of mission m17360 — anchor
XVAL-FVM-CONVERGENCE-DIAGNOSIS-V2 (TensorFVM).

Verification, not re-do.  Everything below is computed from the committed bytes
of the audited tree, never from the audited report's prose:

  1. inventory + sha256 of the delivered files (byte copies kept)
  2. the *registered* referee (tensor_benchmark_cases) for the anchor, plus the
     criterion that is actually in force (mission goal text) when it is empty
  3. independent recomputation of p_ref (this script's own three-grid GCI code,
     not importing gci_reference.py nor convergence_crosscheck.py) and of the
     got side from the production estimators, then the 15 % agreement verdict
     per functional quantity -> judge.json
  4. report-vs-artifact drift check: every headline number quoted in
     REPORT.md section 3 recomputed from the artifact JSON
  5. integration scan: who imports the new module, and does the new payload
     reach the repo study pipeline output
  6. repro: JSON-leaf diff against a re-run of the delivered driver

Usage
    python scripts/audit/m17485_board_audit.py --recompute
    python scripts/audit/m17485_board_audit.py --compare-repro <rerun.json>
    python scripts/audit/m17485_board_audit.py --insert-verdict
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

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SRC = "/root/tp-agent/work/m17360"
COMMIT = "98ab85edf309b9225c82b51befad5d5054997b8d"
BRANCH = "agent/m17360"
ANCHOR = "XVAL-FVM-CONVERGENCE-DIAGNOSIS-V2"
AUDITED_MID = 17360
AUDIT_MID = 17485
OUT = os.path.join(REPO, "artifacts", "audit_m17485")
DOC = os.path.join(REPO, "docs", "board_audit_m17485", "M17485_BOARD_AUDIT.md")
XVAL = "docs/xval/convergence-diagnosis-v2"
ART = XVAL + "/artifacts/xval_convergence_diagnosis.json"

DELIVERED = [
    XVAL + "/REPORT.md",
    XVAL + "/artifacts/xval_convergence_diagnosis.json",
    XVAL + "/artifacts/xval_table.md",
    XVAL + "/figures/xval_committed_artifact.png",
    XVAL + "/figures/xval_error_vs_h.png",
    XVAL + "/figures/xval_order_agreement.png",
    XVAL + "/gci_reference.py",
    XVAL + "/xval_driver.py",
    "src/tensorfvm/verification/convergence_crosscheck.py",
    "src/tensorfvm/verification/sod1978_convergence.py",
    "tests/test_convergence_diagnosis_xval.py",
]

# functional quantities the decision rule is applied to (REPORT 3.1 / 3.2)
ASYMPTOTIC = ["smooth_wave_l1", "l1_rho_full", "l1_rho_cellavg", "contact_x"]
NON_ASYMPTOTIC = ["shock_x", "l1_rho_plateaus"]
SWEEP = "combined_100_200_400_800"
TOL = 0.15


# --------------------------------------------------------------------------- #
# 1. inventory
# --------------------------------------------------------------------------- #
def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def inventory() -> dict:
    os.makedirs(os.path.join(OUT, "delivered"), exist_ok=True)
    rows = {}
    for rel in DELIVERED:
        src = os.path.join(SRC, rel)
        row = {"exists": os.path.isfile(src)}
        if row["exists"]:
            dst = os.path.join(OUT, "delivered", rel)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copy2(src, dst)
            row.update(bytes=os.path.getsize(src), sha256=sha256(src),
                       copy=os.path.relpath(dst, REPO))
            if src.endswith(".md") and not rel.endswith("xval_table.md"):
                row["lines"] = sum(1 for _ in open(src, encoding="utf-8"))
        rows[rel] = row
    st = subprocess.run(["git", "-C", SRC, "status", "--short"], capture_output=True, text=True).stdout
    head = subprocess.run(["git", "-C", SRC, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    in_main = subprocess.run(["git", "-C", SRC, "merge-base", "--is-ancestor", COMMIT, "main"],
                             capture_output=True).returncode == 0
    branches = subprocess.run(["git", "-C", SRC, "branch", "--contains", COMMIT],
                              capture_output=True, text=True).stdout.split()
    man = {"audited_tree": SRC, "head": head, "expected_commit": COMMIT,
           "head_matches": head == COMMIT,
           "branch": BRANCH, "in_main": in_main,
           "branches_containing": [b.lstrip("*") for b in branches],
           "uncommitted_status": [l for l in st.splitlines() if l.strip()],
           "files": rows,
           "all_present": all(r["exists"] for r in rows.values())}
    json.dump(man, open(os.path.join(OUT, "manifest.json"), "w"), indent=1)
    return man


# --------------------------------------------------------------------------- #
# 2. referee
# --------------------------------------------------------------------------- #
def referee(dsn: str) -> dict:
    import psycopg
    with psycopg.connect(dsn) as c, c.cursor() as cur:
        cur.execute("SELECT id, anchor_slug, case_name, metric, tolerance, repo, status, ref_source "
                    "FROM tensor_benchmark_cases WHERE anchor_slug=%s ORDER BY id", (ANCHOR,))
        cols = [d.name for d in cur.description]
        rows = [dict(zip(cols, r)) for r in cur.fetchall()]
        cur.execute("SELECT goal, status, repo FROM tensor_dev_missions WHERE id=%s", (AUDITED_MID,))
        goal, status, repo = cur.fetchone()
        cur.execute("SELECT count(*) FROM tensor_maturity_verdicts WHERE scope LIKE %s",
                    ("m%d %%" % AUDITED_MID,))
        prior = cur.fetchone()[0]
    m = re.search(r"判据：([^。]+)", goal)
    out = {"anchor_slug": ANCHOR, "registered_rows": rows, "registered_rows_count": len(rows),
           "criterion_in_mission_goal": m.group(1) if m else None,
           "mission_status": status, "mission_repo": repo,
           "prior_verdicts_for_scope": prior}
    json.dump(out, open(os.path.join(OUT, "referee.json"), "w"), indent=1, ensure_ascii=False)
    return out


# --------------------------------------------------------------------------- #
# 3. independent recomputation of the 15 % agreement table
# --------------------------------------------------------------------------- #
def _gci_ref_p(phi, h):
    """This script's own three-grid observed order (Celik et al. 2008 eq. 5).

    Deliberately written here from the formula, not imported from the audited
    tree or from gci_reference.py, so the referee face is a third derivation.
    """
    e21, e32 = phi[1] - phi[0], phi[2] - phi[1]
    r21, r32 = h[0] / h[1], h[1] / h[2]
    if e21 == 0 or e32 == 0 or min(abs(e21), abs(e32)) < 1e-300:
        return float("nan")
    return abs(math.log(abs(e32 / e21))) / math.log(r32 if abs(r32 - r21) < 1e-12 else math.sqrt(r21 * r32))


def _classify(phi) -> str:
    d = [phi[i + 1] - phi[i] for i in range(len(phi) - 1)]
    if any(x == 0 for x in d):
        return "no_change"
    s = [1 if x > 0 else -1 for x in d]
    if len(set(s)) > 1:
        return "oscillatory"
    if abs(d[-1]) >= abs(d[-2]):
        return "divergent"
    return "monotonic"


def _lsq(err, h):
    import numpy as np
    n = min(3, len(err))
    lx = np.log(np.asarray(h[-n:], dtype=float))
    ly = np.log(np.asarray(err[-n:], dtype=float))
    return float(np.polyfit(lx, ly, 1)[0])


def recompute() -> dict:
    sys.path.insert(0, os.path.join(SRC, "src"))
    from tensorfvm.verification.sod1978_convergence import fitted_exponent, successive_orders

    d = json.load(open(os.path.join(SRC, ART)))
    sw = d["sweeps"][SWEEP]
    rows = []
    for key, family in [(k, "asymptotic") for k in ASYMPTOTIC] + \
                       [(k, "non_asymptotic") for k in NON_ASYMPTOTIC]:
        s = sw[key]
        phi, h, nx = s["phi"], s["h"], s["nx"]
        exact = s.get("exact_limit")
        zero_limit = s.get("limit_form") == "zero_limit"
        # got side: the production estimators, applied to exactly the same data
        err = list(phi) if zero_limit else [abs(p - exact) for p in phi]
        got_rows = [{"nx": n, "q": e} for n, e in zip(nx, err)]
        p_succ = successive_orders(got_rows, "q")[-1]["rate"]
        p_lsq = fitted_exponent(got_rows, "q")
        # ref side (independent): fine triplet, differences of the functional
        p_ref = _gci_ref_p(phi[-3:], h[-3:])
        rec = {"functional": key, "family": family, "limit_form": s.get("limit_form"),
               "class_ref_audit": _classify(phi),
               "class_ref_artifact": s["ref"].get("convergence_class"),
               "p_ref_audit": p_ref, "p_ref_artifact": s["ref"].get("observed_order"),
               "p_got_lsq_audit": p_lsq, "p_got_succ_audit": p_succ,
               "p_got_lsq_artifact": s["ref"].get("loglog_v2_estimator", {}).get("exponent")
               if False else None,
               "rel_diff_lsq": abs(p_lsq - p_ref) / abs(p_ref) if p_ref else float("nan"),
               "rel_diff_succ": abs(p_succ - p_ref) / abs(p_ref) if p_ref else float("nan"),
               "pass_15pct_lsq": bool(p_ref and abs(p_lsq - p_ref) / abs(p_ref) <= TOL),
               "pass_15pct_succ": bool(p_ref and abs(p_succ - p_ref) / abs(p_ref) <= TOL)}
        rows.append(rec)

    # artifact's own agreement block, for the drift check
    agr = d["sweeps"][SWEEP]
    art = {}
    for key in agr:
        a = agr[key].get("agreement_ref_vs_got", {})
        art[key] = {k: a.get(k) for k in ("p_ref", "p_got_loglog", "p_got_successive",
                                          "rel_diff_loglog", "rel_diff_successive")}
    drift = []
    for r in rows:
        a = art.get(r["functional"], {})
        r["p_ref_artifact"] = a.get("p_ref")
        r["p_got_lsq_artifact"] = a.get("p_got_loglog")
        r["p_got_succ_artifact"] = a.get("p_got_successive")
        for name in ("p_ref", "p_got_lsq", "p_got_succ"):
            mine, theirs = r[name + "_audit"], r[name + "_artifact"]
            if theirs is None:
                drift.append({"functional": r["functional"], "field": name, "note": "artifact_missing"})
                continue
            rel = abs(mine - theirs) / max(abs(theirs), 1e-300)
            r.setdefault("artifact_drift", {})[name] = rel
            if not (rel < 1e-9):
                drift.append({"functional": r["functional"], "field": name,
                              "audit": mine, "artifact": theirs, "rel": rel})

    # every one of the 30 (sweep x functional) rows recomputed independently
    sys.path.insert(0, os.path.join(SRC, "src"))
    all_rows, mism = {}, []
    for swname, swtmp in d["sweeps"].items():
        for key, s in swtmp.items():
            phi, h, nx = s["phi"], s["h"], s["nx"]
            if len(phi) < 3:
                continue
            zero = s.get("limit_form") == "zero_limit"
            err = list(phi) if zero else [abs(p - s["exact_limit"]) for p in phi]
            gr = [{"nx": n, "q": e} for n, e in zip(nx, err)]
            p_succ = successive_orders(gr, "q")[-1]["rate"]
            p_lsq = fitted_exponent(gr, "q")
            p_ref = _gci_ref_p(phi[-3:], h[-3:])
            cl = _classify(phi)
            a = s.get("agreement_ref_vs_got", {})
            rel = {"p_ref": abs(p_ref - a.get("p_ref", float("nan"))) / abs(p_ref),
                   "p_got_loglog": abs(p_lsq - a.get("p_got_loglog", float("nan"))) / abs(p_lsq),
                   "p_got_successive": abs(p_succ - a.get("p_got_successive", float("nan"))) / abs(p_succ)}
            key2 = "%s/%s" % (swname, key)
            all_rows[key2] = {"p_ref_audit": p_ref, "p_got_lsq_audit": p_lsq,
                              "p_got_succ_audit": p_succ, "class_audit": cl,
                              "class_artifact": s["ref"].get("convergence_class"),
                              "artifact": a, "rel_drift": rel}
            if any(v > 1e-9 for v in rel.values() if v == v):
                mism.append(key2)
            if cl != s["ref"].get("convergence_class"):
                mism.append(key2 + ":class " + cl + " vs " + str(s["ref"].get("convergence_class")))
    out_all = {"n_rows": len(all_rows), "mismatches": mism, "rows": all_rows}

    asym = [r for r in rows if r["family"] == "asymptotic"]
    # overlapping-triplet drift for the non-asymptotic functionals (REPORT 3.2)
    drift_trip = {}
    for key in NON_ASYMPTOTIC:
        s2 = sw[key]
        p_coarse = _gci_ref_p(s2["phi"][:3], s2["h"][:3])
        p_fine = _gci_ref_p(s2["phi"][-3:], s2["h"][-3:])
        drift_trip[key] = {"p_coarse_triplet_audit": p_coarse, "p_fine_triplet_audit": p_fine,
                           "rel_diff_audit": abs(p_fine - p_coarse) / abs(p_coarse),
                           "p_coarse_triplet_artifact": s2["ref"].get("triplet_consistency", {})
                           .get("p_triplet_coarse"),
                           "p_fine_triplet_artifact": s2["ref"].get("triplet_consistency", {})
                           .get("p_triplet_fine"),
                           "artifact_rel_diff": s2["ref"].get("triplet_consistency", {}).get("rel_diff"),
                           "rel_diff_denominator_fine": abs(p_fine - p_coarse) / abs(p_fine)}
    out = {"sweep": SWEEP, "tolerance": TOL, "rows": rows,
           "non_asymptotic_triplet_drift": drift_trip,
           "asymptotic_pass_lsq": sum(r["pass_15pct_lsq"] for r in asym),
           "asymptotic_total": len(asym),
           "failing": [r["functional"] for r in asym if not r["pass_15pct_lsq"]],
           "artifact_ref_vs_audit_drift": drift,
           "artifact_verdicts": d["verdicts"],
           "committed_artifact": d["committed_v2_artifact"],
           "two_reference_implementations": {
               "n_numeric_comparisons": d["verdicts"].get("n_numeric_comparisons_two_references"),
               "agree": d["verdicts"].get("both_reference_implementations_agree"),
               "class_disagreements": d["verdicts"].get("reference_class_disagreements")},
           "sweeps_wall_seconds": {k: v.get("wall_seconds") for k, v in d["sweeps_config"].items()},
           "repro_cmd": d.get("repro_cmd"),
           "all_rows_recomputed": out_all}
    json.dump(out, open(os.path.join(OUT, "recompute.json"), "w"), indent=1, default=str)
    return out


# --------------------------------------------------------------------------- #
# 4. report vs artifact drift (six sections + headline numbers)
# --------------------------------------------------------------------------- #
SECTIONS = ["① 案例介绍", "② 方法与实现", "③ 计算结果", "④ 结果图",
            "⑤ 收敛性", "⑥ 失败模式"]


def report_check(rec: dict) -> dict:
    txt = open(os.path.join(SRC, XVAL, "REPORT.md"), encoding="utf-8").read()
    present = {s: (s in txt) for s in SECTIONS}
    # headline numbers claimed in 3.1 / 3.2, recomputed above
    claims = {"2.1116": "smooth_wave_l1 p_ref", "0.9386": "l1_rho_full p_ref",
              "0.9049": "l1_rho_cellavg p_ref", "0.7868": "contact_x p_ref",
              "80.72": "contact_x rel_diff_lsq %", "128.1": "shock triplet drift %",
              "-1.2494": "published shock loglog exponent"}
    got_p_ref = {r["functional"]: r["p_ref_artifact"] for r in rec["rows"]}
    got_rel = {r["functional"]: r["rel_diff_lsq"] for r in rec["rows"]}
    checks = []
    for lit, what in claims.items():
        if lit == "2.1116":
            val = got_p_ref["smooth_wave_l1"]; ok = round(val, 4) == 2.1116
        elif lit == "0.9386":
            val = got_p_ref["l1_rho_full"]; ok = round(val, 4) == 0.9386
        elif lit == "0.9049":
            val = got_p_ref["l1_rho_cellavg"]; ok = round(val, 4) == 0.9049
        elif lit == "0.7868":
            val = got_p_ref["contact_x"]; ok = round(val, 4) == 0.7868
        elif lit == "80.72":
            val = got_rel["contact_x"] * 100; ok = round(val, 2) == 80.72
        elif lit == "128.1":
            val = rec["non_asymptotic_triplet_drift"]["shock_x"]["rel_diff_denominator_fine"] * 100
            ok = abs(val - 128.1) < 0.1 and "128.1" in txt
        else:
            val = (rec["committed_artifact"].get("before") or {}).get("shock_exponent_finest3")
            ok = abs(val + 1.2494149735724402) < 1e-9
        checks.append({"claim": lit, "meaning": what, "recomputed": val, "agrees": bool(ok)})
    # <3% statement present?
    out = {"sections": present, "all_sections": all(present.values()),
           "headline_checks": checks, "all_headlines_agree": all(c["agrees"] for c in checks),
           "declares_lt3pct": "<3 %" in txt or "<3%" in txt,
           "declares_honest_fail": "❌" in txt and "80.7" in txt,
           "difference_list_present": "差异清单" in txt,
           "porting_tradeoff_present": "移植取舍" in txt}
    json.dump(out, open(os.path.join(OUT, "report_check.json"), "w"), indent=1, default=str)
    return out


# --------------------------------------------------------------------------- #
# 5. integration scan
# --------------------------------------------------------------------------- #
def integration(study_json: str | None) -> dict:
    def rg(pat: str) -> list[str]:
        r = subprocess.run(["grep", "-rn", "--include=*.py", "--include=*.md", "--include=*.toml", pat,
                            os.path.join(SRC, "src"), os.path.join(SRC, "scripts"),
                            os.path.join(SRC, "tests"), os.path.join(SRC, "docs")],
                           capture_output=True, text=True)
        return [l.replace(SRC + "/", "") for l in r.stdout.splitlines()
                if "__pycache__" not in l][:12]

    out = {"importers_of_crosscheck": rg("convergence_crosscheck"),
           "module_def_lines": None, "study_pipeline": None}
    f = os.path.join(SRC, "src/tensorfvm/verification/convergence_crosscheck.py")
    out["module_def_lines"] = sum(1 for _ in open(f, encoding="utf-8"))
    out["functions"] = re.findall(r"^def (\w+)", open(f, encoding="utf-8").read(), re.M)
    if study_json and os.path.isfile(study_json):
        s = open(study_json, encoding="utf-8").read()
        out["study_pipeline"] = {"study_json": study_json,
                                 "crosscheck_field_occurrences": s.count("crosscheck"),
                                 "effective_exponent_occurrences": s.count("effective_exponent"),
                                 "loglog_guard_occurrences": s.count("loglog_guard")}
    json.dump(out, open(os.path.join(OUT, "integration.json"), "w"), indent=1)
    return out


# --------------------------------------------------------------------------- #
# 6. repro comparison
# --------------------------------------------------------------------------- #
def leaf_diff(a, b, path="", out=None, skip=("generated_at", "seconds", "wall_seconds")):
    if out is None:
        out = {"n_compared": 0, "diffs": [], "skipped": list(skip)}
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b)):
            if k in skip:
                continue
            if k not in a or k not in b:
                out["diffs"].append({"path": f"{path}/{k}", "note": "key only in one side"})
                continue
            leaf_diff(a[k], b[k], f"{path}/{k}", out, skip)
    elif isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            out["diffs"].append({"path": path, "note": f"len {len(a)} vs {len(b)}"})
            return out
        for i, (x, y) in enumerate(zip(a, b)):
            leaf_diff(x, y, f"{path}[{i}]", out, skip)
    elif isinstance(a, (int, float)) and isinstance(b, (int, float)):
        out["n_compared"] += 1
        if not (a == b or (isinstance(a, float) and math.isnan(a) and isinstance(b, float) and math.isnan(b))):
            rel = abs(a - b) / max(abs(a), abs(b), 1e-300)
            if rel > 1e-12:
                out["diffs"].append({"path": path, "audit": a, "rerun": b, "rel": rel})
    else:
        out["n_compared"] += 1
        if a != b:
            out["diffs"].append({"path": path, "audit": a, "rerun": b})
    return out


def compare_repro(rerun: str) -> dict:
    a = json.load(open(os.path.join(SRC, ART)))
    b = json.load(open(rerun))
    r = leaf_diff(a, b)
    r.update({"audited": ART, "rerun": rerun,
              "rerun_sha256": sha256(rerun),
              "audited_sha256": sha256(os.path.join(SRC, ART)),
              "identical_leaves": len(r["diffs"]) == 0})
    json.dump(r, open(os.path.join(OUT, "repro.json"), "w"), indent=1, default=str)
    return r


# --------------------------------------------------------------------------- #
# 7. verdict row
# --------------------------------------------------------------------------- #
def insert_verdict(dsn: str, level: int, ev: dict) -> int:
    import psycopg
    scope = "m%d %s" % (AUDITED_MID, ANCHOR)
    with psycopg.connect(dsn) as c, c.cursor() as cur:
        cur.execute("SELECT id FROM tensor_maturity_verdicts WHERE scope=%s AND verdict_by='board-audit'",
                    (scope,))
        dup = cur.fetchall()
        if dup:
            print("already audited:", dup)
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dsn", default="dbname=tensorplatform user=postgres")
    ap.add_argument("--inventory", action="store_true")
    ap.add_argument("--referee", action="store_true")
    ap.add_argument("--recompute", action="store_true")
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--integration", default="")
    ap.add_argument("--compare-repro", default="")
    ap.add_argument("--insert-verdict", action="store_true")
    ap.add_argument("--level", type=int, default=3)
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    if a.inventory:
        m = inventory(); print("inventory: all_present=%s head_matches=%s in_main=%s"
                               % (m["all_present"], m["head_matches"], m["in_main"]))
    if a.referee:
        r = referee(a.dsn); print("referee rows:", r["registered_rows_count"], "| criterion:",
                                  (r["criterion_in_mission_goal"] or "")[:120])
    if a.recompute:
        rc = recompute()
        print("recompute: %d/%d asymptotic rows pass 15%%; failing=%s; drift=%d"
              % (rc["asymptotic_pass_lsq"], rc["asymptotic_total"], rc["failing"],
                 len(rc["artifact_ref_vs_audit_drift"])))
    if a.report:
        rec = json.load(open(os.path.join(OUT, "recompute.json")))
        rp = report_check(rec)
        print("report: sections=%s headlines=%s" % (rp["all_sections"], rp["all_headlines_agree"]))
    if a.integration is not None and a.integration != "":
        it = integration(a.integration); print("integration:", it["study_pipeline"])
    if a.compare_repro:
        cr = compare_repro(a.compare_repro)
        print("repro: identical=%s compared=%d diffs=%d"
              % (cr["identical_leaves"], cr["n_compared"], len(cr["diffs"])))
    if a.insert_verdict:
        ev = json.load(open(os.path.join(OUT, "evidence.json")))
        insert_verdict(a.dsn, a.level, ev)


if __name__ == "__main__":
    main()