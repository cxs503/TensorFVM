"""Provenance + reproduction audit of the anchored dam-break case-pool entry.

Why this exists
---------------
``CASE-FVM-DAMBREAK-STOKER`` has an unusually thick evidence chain
(exact solution, 6-gate pool entry, 2-D/oblique/hires/long-time/similarity,
oracle independence, configuration envelope).  Everything in it answers
*physics* questions.  Nothing answers the *provenance* questions a reader of
``docs/case-fvm-dambreak-stoker/case_pool_entry.json`` has to ask before
re-using the numbers:

P1  Does the committed artifact still come out of the committed source?
    (i.e. is the evidence chain frozen, and does it reproduce today?)
P2  Is the committed pool entry really the *derived* payload, or has it drifted
    from the artifact (hand edits, unit slips, threshold drift)?
P3  Can a third party re-run the graded case from the entry alone -- are all
    the inputs that move the numbers actually recorded?
P4  Do the entry's own verdicts (``pass``, ``accepted_grid_min``,
    ``honest_fail``, ``passing_grids``) follow from its own rows, and which
    criteria are binding?
P5  Is the case covered by the repository's archival-staleness firewall at all?

Every check below is mechanical and every number written to
``docs/CASE-FVM-DAMBREAK-STOKER-PROVENANCE/`` is injected from this script.
The auditor is itself tested for *power*: :func:`negative_controls` corrupts a
copy of the committed data in five ways and requires the corresponding check to
turn red, so a green audit is not vacuous.

Usage::

    PYTHONPATH=src python scripts/audit_dambreak_pool_provenance.py
    PYTHONPATH=src python scripts/audit_dambreak_pool_provenance.py --out <dir>
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import pathlib
import subprocess
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tensorfvm.shallow_water.dam_break import CASES, run_case  # noqa: E402
from tensorfvm.verification import case_pool_dambreak as pool  # noqa: E402

ENTRY_PATH = ROOT / "docs" / "case-fvm-dambreak-stoker" / "case_pool_entry.json"
ARTIFACT_DIR = ROOT / "src" / "tensorfvm" / "verification" / "artifacts" / "shallow-water-dam-break"
STALE_LEDGER = ROOT / "docs" / "verification-archived-source-staleness" / "staleness.json"

#: Tolerance for "same number" between the artifact and a fresh run.  The
#: solver is a deterministic serial loop, so anything above rounding means the
#: source moved the physics.
REPRO_TOL = 1e-12

#: ``criterion -> (raw metric path in the artifact, scale)``.  The scales are the
#: documented conversions of the entry (fractions -> percent; ``tv_ratio - 1``
#: -> percent; counts stay counts).  Deliberately re-implemented here instead of
#: calling ``case_pool_dambreak._criteria``: an audit that reuses the builder
#: cannot detect a builder bug.
CRITERION_SOURCES = {
    "ritter-dry": {
        "front_fan<5%": (("metrics", "front_fan_error"), 100.0),
        "front_wetted<5%": (("metrics", "front_wetted_error"), 100.0),
        "depth_l1<5%": (("metrics", "depth_l1"), 100.0),
        "monotone(no_extrema)": (("metrics", "local_extrema"), 1.0),
        "tv_no_growth<0.5%": (("metrics", "tv_ratio"), "tv"),
        "front_series<5%": (("metrics", "front_series", "max_error"), 100.0),
    },
    "stoker-wet": {
        "shock_position<5%": (("metrics", "shock_error"), 100.0),
        "step_ratio<5%": (("metrics", "step_ratio_error"), 100.0),
        "plateau_h2<5%": (("metrics", "h2_error"), 100.0),
        "depth_l1<5%": (("metrics", "depth_l1"), 100.0),
        "monotone(no_extrema)": (("metrics", "local_extrema"), 1.0),
        "tv_no_growth<0.5%": (("metrics", "tv_ratio"), "tv"),
        "bore_position_modes<5%": (("metrics", "modes", "bore_x"), "bore_x"),
        "step_ratio_modes<5%": (("metrics", "modes", "step_ratio"), "step_ratio"),
        "front_series<5%": (("metrics", "front_series", "max_error"), 100.0),
    },
}

#: Inputs that move the graded numbers.  Anything listed here must be readable
#: from the entry for the entry to be re-runnable on its own.
REQUIRED_INPUTS = (
    "grid",
    "dx",
    "cfl",
    "order",
    "t_star",
    "x_min",
    "x_max",
    "x_dam",
    "h0",
    "h1",
    "gravity",
    "dry_depth",
)


def _get(blob, path):
    cur = blob
    for key in path:
        cur = cur[key]
    return cur


def derived_criterion_value(case_key, criterion, metrics):
    """Recompute one entry criterion from raw artifact numbers."""

    path, scale = CRITERION_SOURCES[case_key][criterion]
    if path and path[0] == "metrics" and "metrics" not in metrics:
        path = path[1:]  # callers may already hand in the ``metrics`` subtree
    raw = _get(metrics, path)
    if scale == "tv":
        return 100.0 * (raw - 1.0)
    if scale == "bore_x":
        exact = _get(metrics, ("shock_exact",))
        return 100.0 * abs(raw - exact) / exact
    if scale == "step_ratio":
        exact = _get(metrics, ("step_ratio_exact",))
        return 100.0 * abs(raw - exact) / exact
    return float(scale) * float(raw)


def _rel(a, b):
    if a == b:
        return 0.0
    return abs(a - b) / max(abs(a), abs(b), 1e-300)


# ---------------------------------------------------------------------------
# P1  reproduction from current source
# ---------------------------------------------------------------------------
def reproduction(artifact, cases=None):
    """Re-run every graded case at the entry's declared configuration."""

    cases = CASES if cases is None else cases
    rows = []
    for key, case in cases.items():
        old = artifact["cases"][key]
        disc = old["discretisation"]
        run = run_case(
            case,
            n=int(disc["n"]),
            cfl=float(disc["cfl"]),
            order=int(disc["order"]),
            t_star=float(disc["t_star_end"]),
        )
        new = run["metrics"]
        graded = {}
        for name in CRITERION_SOURCES[key]:
            try:
                graded[name] = (
                    derived_criterion_value(key, name, old["metrics"]),
                    derived_criterion_value(key, name, new),
                )
            except KeyError:
                continue
        worst = max(
            ((_rel(a, b), name) for name, (a, b) in graded.items()), default=(0.0, None)
        )
        rows.append(
            {
                "case": key,
                "n": int(disc["n"]),
                "steps_committed": int(disc["steps"]),
                "steps_rerun": int(run["discretisation"]["steps"]),
                "steps_match": int(disc["steps"]) == int(run["discretisation"]["steps"]),
                "worst_relative_difference": worst[0],
                "worst_criterion": worst[1],
                "criteria": {
                    name: {"committed": a, "rerun": b, "reproduced": _rel(a, b) <= REPRO_TOL}
                    for name, (a, b) in graded.items()
                },
                "recorded_keys_missing_in_artifact": sorted(
                    set(run["discretisation"]) - set(disc)
                ),
            }
        )
    return {
        "tolerance": REPRO_TOL,
        "rows": rows,
        "all_reproduced": all(r["steps_match"] and r["worst_relative_difference"] <= REPRO_TOL for r in rows),
    }


# ---------------------------------------------------------------------------
# P2  entry <-> artifact derivation
# ---------------------------------------------------------------------------
def entry_derivation(entry, artifact):
    rows = []
    for case in entry["cases"]:
        key = case["key"]
        metrics = artifact["cases"][key]["metrics"]
        seen = set()
        for criterion in case["criteria"]:
            name = criterion["criterion"]
            seen.add(name)
            if name not in CRITERION_SOURCES[key]:
                rows.append({"case": key, "criterion": name, "status": "unmapped"})
                continue
            expect = derived_criterion_value(key, name, metrics)
            rel = _rel(expect, criterion["value"])
            rows.append(
                {
                    "case": key,
                    "criterion": name,
                    "artifact_value": expect,
                    "entry_value": criterion["value"],
                    "relative_difference": rel,
                    "status": "derived" if rel <= 1e-12 else "drift",
                }
            )
        for name in sorted(set(CRITERION_SOURCES[key]) - seen):
            rows.append({"case": key, "criterion": name, "status": "missing_from_entry"})
    drift = [r for r in rows if r["status"] in ("drift", "missing_from_entry")]
    unmapped = [r for r in rows if r["status"] == "unmapped"]
    return {
        "rows": rows,
        "drifted": drift,
        "unmapped": unmapped,
        "clean": not drift and not unmapped,
    }


# ---------------------------------------------------------------------------
# P3  can the entry be re-run on its own?
# ---------------------------------------------------------------------------
def config_completeness(entry):
    rows = []
    for case in entry["cases"]:
        have = {k for k in REQUIRED_INPUTS if case.get(k) is not None}
        rows.append(
            {
                "case": case["key"],
                "present": sorted(have),
                "missing": sorted(set(REQUIRED_INPUTS) - have),
            }
        )
    missing = sorted({m for r in rows for m in r["missing"]})
    return {"rows": rows, "missing_anywhere": missing, "complete": not missing}


# ---------------------------------------------------------------------------
# P4  verdict algebra + which criteria bind
# ---------------------------------------------------------------------------
def verdict_algebra(entry):
    problems = []
    for case in entry["cases"]:
        flags = [c["pass"] for c in case["criteria"]]
        all_pass = all(flags)
        if bool(case["pass"]) != all_pass:
            problems.append(f"{case['key']}: case.pass={case['pass']} but all(criteria)={all_pass}")
        for c in case["criteria"]:
            value, limit = c["value"], c["threshold"]
            lt = c["comparator"] == "lt"
            ok = (value < limit) if lt else (value <= limit)
            if bool(c["pass"]) != bool(ok):
                problems.append(
                    f"{case['key']}/{c['criterion']}: pass={c['pass']} vs {value} {c['comparator']} {limit}"
                )
    verdict = entry["verdict"]
    if verdict["all_criteria_pass_at_pool_grid"] != all(c["pass"] for c in entry["cases"]):
        problems.append("verdict.all_criteria_pass_at_pool_grid != all(case.pass)")
    primary_ok = all(
        c["pass"] for case in entry["cases"] for c in case["criteria"] if c.get("primary")
    )
    if bool(verdict["all_primary_pass_at_pool_grid"]) != bool(primary_ok):
        problems.append("verdict.all_primary_pass_at_pool_grid != all primary criteria pass")
    if verdict["honest_fail_count"] != len(entry["honest_fail"]):
        problems.append("verdict.honest_fail_count != len(honest_fail)")

    rows = entry["grid_study"]["rows"]
    passing = sorted({r["n"] for r in rows if all(d["pass"] for d in r["cases"].values())})
    if passing != entry["grid_study"]["passing_grids"]:
        problems.append(f"passing_grids {entry['grid_study']['passing_grids']} != rows-derived {passing}")
    if passing and entry["accepted_grid_min"] != passing[0]:
        problems.append("accepted_grid_min != min(passing_grids)")
    if verdict["pool_grid"] != max(c["grid"] for c in entry["cases"]):
        problems.append("verdict.pool_grid != max(case.grid)")

    # honest_fail must be exactly the failing grid rows.
    from_rows = sorted(
        (r["n"], k) for r in rows for k, d in r["cases"].items() if not d["pass"]
    )
    from_entry = sorted((h["n"], h["case"]) for h in entry["honest_fail"])
    if from_rows != from_entry:
        problems.append("honest_fail is not the set of failing grid rows")

    # Which criteria bind the accepted-grid boundary, and are they primary?
    primary = {
        (case["key"]): {c["criterion"]: c.get("primary", False) for c in case["criteria"]}
        for case in entry["cases"]
    }
    binding = []
    if passing:
        boundary = passing[0]
        for r in rows:
            if r["n"] >= boundary:
                continue
            for case_key, data in r["cases"].items():
                for crit in data.get("verdict_failing", []):
                    binding.append(
                        {
                            "n": r["n"],
                            "case": case_key,
                            "criterion": crit,
                            "primary": bool(primary[case_key].get(crit, False)),
                        }
                    )
    primary_only_passing = sorted(
        {
            r["n"]
            for r in rows
            if all(
                d["pass"] or all(not primary[k].get(c, True) for c in d["verdict_failing"])
                for k, d in r["cases"].items()
            )
        }
    )
    return {
        "problems": problems,
        "consistent": not problems,
        "binding_failures": binding,
        "passing_grids": passing,
        "passing_grids_primary_only": primary_only_passing,
        "accepted_grid_min": entry["accepted_grid_min"],
        "accepted_grid_min_primary_only": primary_only_passing[0] if primary_only_passing else None,
    }


# ---------------------------------------------------------------------------
# P5  staleness coverage + builder byte identity
# ---------------------------------------------------------------------------
def builder_identity(entry):
    derived = pool.render_entry_json()
    committed = ENTRY_PATH.read_text()
    return {
        "identical": derived == committed,
        "committed_bytes": len(committed),
        "derived_bytes": len(derived),
    }


def staleness_coverage():
    from tensorfvm.verification.source_staleness import audit, unfrozen_names

    report = audit(str(ROOT))
    names = [c["case"] for c in report["cases"]]
    frozen = [n for n in names if "dam-break" in n or "shallow-water" in n]
    unfrozen = unfrozen_names(report) if callable(unfrozen_names) else []
    return {
        "cases_audited": len(names),
        "dam_break_cases_registered": frozen,
        "covered": bool(frozen),
        "unfrozen_count": len(unfrozen),
        "unfrozen_contains_shallow_water": any("shallow" in str(u) for u in unfrozen),
    }


# ---------------------------------------------------------------------------
# Power of the auditor: five corruptions, each must be caught
# ---------------------------------------------------------------------------
def negative_controls(entry, artifact):
    controls = []

    # 1. numerically: a +10 % front error must break reproduction.
    bad = copy.deepcopy(artifact)
    bad["cases"]["ritter-dry"]["metrics"]["front_fan_error"] *= 1.10
    rep = reproduction(bad, cases={"ritter-dry": CASES["ritter-dry"]})
    controls.append(
        {
            "name": "artifact front_fan_error +10%",
            "check": "P1 reproduction",
            "caught": not rep["all_reproduced"],
        }
    )

    # 2. bookkeeping: a hand-edited entry value must break derivation.
    bad_entry = copy.deepcopy(entry)
    bad_entry["cases"][0]["criteria"][0]["value"] *= 1.10
    der = entry_derivation(bad_entry, artifact)
    controls.append(
        {
            "name": "entry front_fan value +10% (hand edit)",
            "check": "P2 derivation",
            "caught": not der["clean"],
        }
    )

    # 3. self-contradiction: a flipped pass flag must break the verdict algebra.
    bad_entry = copy.deepcopy(entry)
    bad_entry["cases"][1]["criteria"][0]["pass"] = not bad_entry["cases"][1]["criteria"][0]["pass"]
    alg = verdict_algebra(bad_entry)
    controls.append(
        {
            "name": "one criteria pass flag flipped",
            "check": "P4 verdict algebra",
            "caught": not alg["consistent"],
        }
    )

    # 4. a dropped input must break completeness.
    bad_entry = copy.deepcopy(entry)
    bad_entry["cases"][0]["x_dam"] = None
    comp = config_completeness(bad_entry)
    controls.append(
        {
            "name": "x_dam blanked in the entry",
            "check": "P3 config completeness",
            "caught": not comp["complete"],
        }
    )

    # 5. a shifted accepted grid must break the algebra.
    bad_entry = copy.deepcopy(entry)
    bad_entry["accepted_grid_min"] = bad_entry["accepted_grid_min"] * 2
    alg = verdict_algebra(bad_entry)
    controls.append(
        {
            "name": "accepted_grid_min doubled",
            "check": "P4 verdict algebra",
            "caught": not alg["consistent"],
        }
    )
    return {"controls": controls, "all_caught": all(c["caught"] for c in controls)}


def run_audit():
    started = time.time()
    entry = json.loads(ENTRY_PATH.read_text())
    artifact = json.loads((ARTIFACT_DIR / "metrics.json").read_text())
    out = {
        "slug": entry["slug"],
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "command": "PYTHONPATH=src python scripts/audit_dambreak_pool_provenance.py",
        "entry": ENTRY_PATH.relative_to(ROOT).as_posix(),
        "artifact": (ARTIFACT_DIR / "metrics.json").relative_to(ROOT).as_posix(),
        "P1_reproduction": reproduction(artifact),
        "P2_entry_derivation": entry_derivation(entry, artifact),
        "P3_config_completeness": config_completeness(entry),
        "P4_verdict_algebra": verdict_algebra(entry),
        "P5_provenance": {
            "builder_identity": builder_identity(entry),
            "staleness": staleness_coverage(),
        },
    }
    out["P6_negative_controls"] = negative_controls(entry, artifact)
    out["verdict"] = {
        "P1_reproduces": out["P1_reproduction"]["all_reproduced"],
        "P2_entry_is_derived": out["P2_entry_derivation"]["clean"],
        "P3_entry_is_self_contained": out["P3_config_completeness"]["complete"],
        "P4_verdicts_follow_from_rows": out["P4_verdict_algebra"]["consistent"],
        "P5_entry_is_generated": out["P5_provenance"]["builder_identity"]["identical"],
        "P5_case_is_frozen": bool(out["P5_provenance"]["staleness"]["covered"]),
        "P6_auditor_has_power": out["P6_negative_controls"]["all_caught"],
    }
    out["wall_time_s"] = time.time() - started
    return out


# ---------------------------------------------------------------------------
# rendering
# ---------------------------------------------------------------------------
def render_summary(audit):
    p1 = audit["P1_reproduction"]
    p2 = audit["P2_entry_derivation"]
    p3 = audit["P3_config_completeness"]
    p4 = audit["P4_verdict_algebra"]
    p5 = audit["P5_provenance"]
    p6 = audit["P6_negative_controls"]
    L = []
    a = L.append
    a("# CASE-FVM-DAMBREAK-STOKER · 算例池条目的溯源与可复现性审计")
    a("")
    a(f"生成的工件：`{audit['entry']}` 对 `{audit['artifact']}`。本文件所有数字由")
    a("`scripts/audit_dambreak_pool_provenance.py` 注入，无手抄。")
    a("")
    a(f"生成时间：`{audit['generated_at']}` · 用时 `{audit['wall_time_s']:.1f} s`")
    a("")
    a("## 判定汇总")
    a("")
    a("| 编号 | 问的问题 | 结果 |")
    a("|---|---|---|")
    labels = {
        "P1_reproduces": "P1 存档数字是否还能从当前源码复现",
        "P2_entry_is_derived": "P2 条目是否真是工件的派生（无手改/无单位滑移）",
        "P3_entry_is_self_contained": "P3 能否只凭条目重跑（输入是否记全）",
        "P4_verdicts_follow_from_rows": "P4 判定是否由条目自己的行推得",
        "P5_entry_is_generated": "P5 条目是否与生成器逐字节一致",
        "P5_case_is_frozen": "P5 该算例是否被仓内 staleness 冻结网覆盖",
        "P6_auditor_has_power": "P6 审计器自身是否可失败（5 个注入探针）",
    }
    for key, label in labels.items():
        a(f"| `{key}` | {label} | **{'PASS' if audit['verdict'][key] else 'FAIL'}** |")
    a("")
    a("## P1 复现（当前源码 vs 存档工件）")
    a("")
    a("| 算例 | n | 存档步数 | 重跑步数 | 最差判据相对差 | 该判据 |")
    a("|---|---:|---:|---:|---:|---|")
    for r in p1["rows"]:
        a(
            f"| `{r['case']}` | {r['n']} | {r['steps_committed']} | {r['steps_rerun']} | "
            f"{r['worst_relative_difference']:.3e} | `{r['worst_criterion']}` |"
        )
    a("")
    a(f"容差 `{p1['tolerance']:.0e}`（串行确定性求解器，超出即是源码改了物理）。")
    a("")
    a("存档 `discretisation` 相对当前 `run_case` 缺失的键（**答案**）:")
    for r in p1["rows"]:
        a(f"* `{r['case']}`: {r['recorded_keys_missing_in_artifact'] or '无'}")
    a("")
    a("## P2 条目 vs 工件重派生")
    a("")
    a(f"判据重派生 {len(p2['rows'])} 条：drift {len(p2['drifted'])} 条、未映射 {len(p2['unmapped'])} 条。")
    if p2["drifted"] or p2["unmapped"]:
        for r in p2["drifted"] + p2["unmapped"]:
            a(f"* `{r['case']}/{r['criterion']}`: {r['status']}")
    a("")
    a("## P3 条目的输入完备性")
    a("")
    a("| 算例 | 已记录 | 缺失 |")
    a("|---|---|---|")
    for r in p3["rows"]:
        a(f"| `{r['case']}` | {', '.join(r['present'])} | {', '.join(r['missing']) or '无'} |")
    a("")
    a("## P4 判定代数与约束判据")
    a("")
    a(f"自洽性问题 {len(p4['problems'])} 条。")
    for p in p4["problems"]:
        a(f"* {p}")
    a("")
    a(f"`accepted_grid_min` = **{p4['accepted_grid_min']}**（条目自己给出的）；"
      f"若只用 primary 判据则为 **{p4['accepted_grid_min_primary_only']}**。")
    a("")
    if p4["binding_failures"]:
        a("| 网格 n | 算例 | 失效判据 | 是否 primary |")
        a("|---:|---|---|---|")
        for b in p4["binding_failures"]:
            a(f"| {b['n']} | `{b['case']}` | `{b['criterion']}` | {'是' if b['primary'] else '**否（诊断量）**'} |")
        a("")
    a("## P5 溯源")
    a("")
    bi = p5["builder_identity"]
    st = p5["staleness"]
    a(f"* 生成器逐字节一致：`{bi['identical']}`（{bi['committed_bytes']} / {bi['derived_bytes']} 字节）")
    a(f"* staleness 审计覆盖的算例数：{st['cases_audited']}；命中 dam-break 的算例：{st['dam_break_cases_registered'] or '**无**'}")
    a(f"* 未冻结名字（unfrozen）共 {st['unfrozen_count']}，其中含 shallow-water：{st['unfrozen_contains_shallow_water']}")
    a("")
    a("## P6 审计器的可失败性（注入探针）")
    a("")
    a("| 注入 | 应对检查 | 是否被发现 |")
    a("|---|---|---|")
    for c in p6["controls"]:
        a(f"| {c['name']} | {c['check']} | {'是' if c['caught'] else '**否**'} |")
    a("")
    return "\n".join(L) + "\n"


def figure(audit, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    rows = audit["P1_reproduction"]["rows"]
    names = [r["case"] for r in rows]
    vals = [max(r["worst_relative_difference"], 1e-18) for r in rows]
    ax[0].bar(names, vals, color=["#3b7dd8", "#d85a3b"])
    ax[0].axhline(audit["P1_reproduction"]["tolerance"], color="k", ls="--", lw=1)
    ax[0].set_yscale("log")
    ax[0].set_ylabel("worst |Δ| / max|·| vs archived artifact")
    ax[0].set_title("P1 reproduction from current source")
    for x, v in zip(names, vals):
        ax[0].annotate(f"{v:.1e}", (x, v), ha="center", va="bottom", fontsize=8)

    keys = list(audit["verdict"])
    ys = np.arange(len(keys))
    ax[1].barh(ys, [1 if audit["verdict"][k] else 0 for k in keys],
               color=["#2e9e5b" if audit["verdict"][k] else "#c0392b" for k in keys])
    ax[1].set_yticks(ys, [k for k in keys], fontsize=7)
    ax[1].set_xlim(0, 1.1)
    ax[1].set_xlabel("1 = PASS, 0 = FAIL")
    ax[1].set_title("provenance verdicts")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="docs/CASE-FVM-DAMBREAK-STOKER-PROVENANCE")
    args = parser.parse_args(argv)
    out_dir = ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    audit = run_audit()
    (out_dir / "summary.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    )
    (out_dir / "SUMMARY.md").write_text(render_summary(audit))
    figure(audit, out_dir / "fig_provenance.png")
    print(json.dumps(audit["verdict"], indent=2))
    print(f"wrote {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())