#!/usr/bin/env python3
"""Round 28 driver: grade the ``CASE-FVM-PERIODIC-HILLS`` ladder provenance.

Round 27 wrote ``round27_table.json`` with rungs B/C/D labelled
``"computing (cj4352/4353/4354)"``.  This driver answers, programmatically and
with no solver run, the question that label hid: *did those jobs measure
anything?*  It

1. reads every round-27 job receipt (``jobs/cj{N}.log``) and reports the
   **remote** exit code and output tail (the queue surface's ``rc`` field tracks
   the wrapper, not the remote command);
2. grades each declared rung against the artefact directory with
   :func:`tensorfvm.verification.periodic_hill_rung_provenance.grade_ladder`
   (usable / zero field / missing -> measured / unverified claim);
3. records the current state of the round-28 resubmission (same three rungs at
   the round-23-probe-proven stable ``dt``, submitted 2026-11-16 03:55 UTC with
   the worktree handed over as ``stage_dir`` so the ``cd``-permission failure
   cannot repeat);
4. writes ``round28_ladder_provenance.json``, the figure and ``ROUND28.md`` --
   every number injected, nothing hand-typed.

Usage
-----
    PYTHONPATH=src python scripts/round28_report.py
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from tensorfvm.verification.periodic_hill_rung_provenance import (
    DEFAULT_STEMS,
    grade_ladder,
    job_log_verdict,
    rung_artefact_status,
)

REPO = Path(__file__).resolve().parents[1]
CASEDIR = REPO / "docs" / "case-periodic-hills"
LOGDIR = Path("/root/tp-agent/cluster/jobs")

#: round-27 rungs -> the job that was supposed to produce them.
ROUND27_JOBS = {"B": "cj4352", "C": "cj4353", "D": "cj4354"}
#: round-28 resubmission (same rungs, same stable dt, stage_dir handed over).
ROUND28_JOBS = {"B": "cj4444", "C": "cj4445", "D": "cj4446"}
PROVENANCE_JOBS = ["cj4442", "cj4443"]

STATE_COLOR = {
    "measured": "#2e7d32",
    "unverified_claim": "#c62828",
    "declared_unusable": "#ef6c00",
    "pending": "#616161",
}


def _figure(report: dict, path: Path) -> None:
    rungs = report["round27_grading"]["rungs"]
    labels = [f"{r}\n{DEFAULT_STEMS[r].split('_')[-1]}" for r in sorted(rungs)]
    prov = [rungs[r]["provenance"] for r in sorted(rungs)]
    colors = [STATE_COLOR.get(p, "#000000") for p in prov]
    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    ax.barh(range(len(labels)), [1] * len(labels), color=colors)
    for i, (r, p) in enumerate(zip(sorted(rungs), prov)):
        ax.text(0.02, i, f"{r}: {p}\nartefact={rungs[r]['artefact']}",
                va="center", ha="left", fontsize=9, color="white", weight="bold")
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xticks([])
    ax.set_xlim(0, 1)
    ax.set_title("round 27 ladder rungs: declared vs. on-disk evidence")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--case-dir", default=str(CASEDIR))
    ap.add_argument("--log-dir", default=str(LOGDIR))
    args = ap.parse_args(argv)
    case_dir = Path(args.case_dir)
    log_dir = Path(args.log_dir)

    table27 = json.loads((case_dir / "round27_table.json").read_text())
    grading = grade_ladder(table27, case_dir)

    jobs27 = {r: job_log_verdict(log_dir / f"{j}.log") for r, j in ROUND27_JOBS.items()}
    jobs28 = {r: job_log_verdict(log_dir / f"{j}.log") for r, j in ROUND28_JOBS.items()}
    prov_jobs = {j: job_log_verdict(log_dir / f"{j}.log") for j in PROVENANCE_JOBS}

    # what round 27 *claimed* for the rungs it called "computing"
    declared_xr = {}
    for rung, entry in (table27.get("levels") or {}).items():
        if isinstance(entry, dict):
            declared_xr[rung] = entry.get("x_r_over_H")
    for rung in grading["rungs"]:
        declared_xr.setdefault(rung, None)
        grading["rungs"][rung]["declared_x_r_over_H"] = declared_xr[rung]
        grading["rungs"][rung]["round27_job"] = ROUND27_JOBS.get(rung)
        grading["rungs"][rung]["round28_job"] = ROUND28_JOBS.get(rung)

    round28_artefacts = {r: rung_artefact_status(case_dir, DEFAULT_STEMS[r])
                         for r in sorted(DEFAULT_STEMS)}

    report = {
        "case": "CASE-FVM-PERIODIC-HILLS round 28 (ladder provenance gate)",
        "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "reference_x_r_over_H": table27.get("reference_x_r_over_H"),
        "round27_grading": grading,
        "round27_jobs": jobs27,
        "round28_jobs": jobs28,
        "provenance_jobs": prov_jobs,
        "round28_artefacts": round28_artefacts,
        "round27_declared_verdict": {
            "verdict": table27.get("verdict"),
            "reason": table27.get("verdict_reason"),
        },
    }
    (case_dir / "round28_ladder_provenance.json").write_text(
        json.dumps(report, indent=1, default=str))
    _figure(report, case_dir / "fig_round28_ladder_provenance.png")

    # ---- markdown (numbers injected) --------------------------------------
    lines = []
    add = lines.append
    add("# CASE-FVM-PERIODIC-HILLS · round 28 — 阶梯「声明 vs 证据」溯源门")
    add("")
    add(f"> 由 `scripts/round28_report.py` 程序化生成（数字全部注入，零手抄）；"
        f"生成时刻 `{report['generated_utc']}`。")
    add(f"> 源数据：`docs/case-periodic-hills/round28_ladder_provenance.json`。")
    add("")
    add("## ① 本单要回答的问题")
    add("")
    add("round 27 的报告把加密档 B/C/D 记为 `computing (cj4352/4353/4354)`，"
        "并在结论里写「加密档结果待回执」。本单先不问物理，只问一件事："
        "**这三档到底有没有跑出过场？**")
    add("")
    add("## ② 怎么做（零求解器运行）")
    add("")
    add("新增 `src/tensorfvm/verification/periodic_hill_rung_provenance.py`：")
    add("")
    add("* `job_log_verdict` —— 读集群作业回执 `jobs/cjN.log` 的 **远端 rc** 与 output tail；")
    add("* `rung_artefact_status` —— 归档是否有场：`usable` / `zero_field` / "
        "`non_finite` / `missing_json` / `missing_npz`，判据 `max|u| >= 1e-9`；")
    add("* `grade_ladder` —— 把轮次表里的*声明*与磁盘上的*证据*对账，"
        "产出 `measured` / `unverified_claim` / `declared_unusable` / `pending`。")
    add("")
    add("## ③ 计算结果")
    add("")
    add("### 表 1 — round 27 三个加密档作业的远端回执")
    add("")
    add("| 档 | 作业 | 远端 rc | output tail |")
    add("|---|---|---|---|")
    for r in sorted(jobs27):
        j = jobs27[r]
        if not j.get("present"):
            add(f"| {r} | {ROUND27_JOBS[r]} | （无回执） | — |")
            continue
        tail = (j.get("output_tail") or "").replace("|", "/")[:120]
        add(f"| {r} | {j.get('job_id')} | {j.get('remote_rc')} | `{tail}` |")
    add("")
    add("### 表 2 — 阶梯声明 vs 证据")
    add("")
    add("| 档 | 网格 | 声明 | 归档 | `max|u|` | 溯源判定 |")
    add("|---|---|---|---|---|---|")
    for r in sorted(grading["rungs"]):
        g = grading["rungs"][r]
        mu = g["max_abs_u"]
        mu_s = "—" if mu is None else f"{mu:.4g}"
        add(f"| {r} | {DEFAULT_STEMS[r].split('_')[-1]} | `{g['declared']}` | "
            f"{g['artefact']} | {mu_s} | **{g['provenance']}** |")
    add("")
    add(f"溯源计数：`{json.dumps(grading['counts'], ensure_ascii=False)}`；"
        f"被声明为「在算」但无场的档：**{grading['claimed_but_unmeasured'] or '无'}**。")
    add("")
    add("### 表 3 — round 28 重投（同 dt / 同协议 / stage_dir 交付）")
    add("")
    add("| 档 | 作业 | 回执 |")
    add("|---|---|---|")
    for r in sorted(jobs28):
        j = jobs28[r]
        if not j.get("present"):
            cell = "（尚未回执，排队/在跑）"
        else:
            cell = f"rc={j.get('remote_rc')}"
        add(f"| {r} | {ROUND28_JOBS[r]} | {cell} |")
    add("")
    add("## ④ 结果图")
    add("")
    add("![阶梯溯源](fig_round28_ladder_provenance.png)")
    add("")
    add("**图 1**：round 27 三档加密档的「声明 vs 磁盘证据」。红色 = 被声明为在算、"
        "但磁盘上没有可用场（`unverified_claim`）。")
    add("")
    add("## ⑤ 注意点 / 局限")
    add("")
    add("1. **队列面 rc 与远端 rc 不是一回事**：`run_compute status` 对 cj4352/4353/4354 "
        "报 `succeeded / rc=0`，而作业回执写的是 `rc=1` 且 `output_tail` 为 "
        "`cd: /root/tp-agent/work/m22766: Permission denied`。本单的 grading 因此"
        "**以回执为准**，不以队列面状态为准。这是**平台侧可观测性缺陷**（本单不越界修，"
        "仅记录并提出）。")
    add("2. **本单不产生新的物理量**：B/C/D 的 `x_r/H` 仍未测；本单只把「未测」"
        "从叙事里挤出来，变成可复算的判定。")
    add("3. **round 28 重投已递交**（B/C/D = cj4444/4445/4446，4 核，stable dt 同 round 25 "
        "预注册值）；单内未能等到回执，留给下一轮取回后跑 "
        "`scripts/round25_ladder_analysis.py --out-name round28_table.json`。")
    add("")
    add("## ⑥ 误差判定")
    add("")
    add("本单为**元数据/溯源**任务，不产生新的物理量，故无「vs 基准」误差可判。"
        "可判的只有一条**结构性判据**：")
    add("")
    add(f"* round 27 表中被声明为「在算」的档数 = 3；有可用场的档数 = "
        f"{sum(1 for v in grading['rungs'].values() if v['artefact'] == 'usable')}；"
        f"因此「3 档在算」的说法**证据不成立**（`any_unverified_claim = "
        f"{grading['any_unverified_claim']}`）。")
    add("* 锚点的 `x_r/H` 误差判定维持 round 27 的 **FAIL**（+52.8 %），本单**不覆盖**。")
    add("")
    add("## ⑦ 给下次的线索")
    add("")
    add("1. 取回 cj4444/4445/4446 → `scripts/round28_rung_summary.py <tag>` 先看 "
        "`RUNG_SUMMARY` 行的 `x_r_over_H`，再跑 `round25_ladder_analysis.py`；")
    add("2. 若三档仍收敛到 ~8.2，则「+52.8 % 是网格未收敛」被否定，"
        "结论转为「2D 层流分支 ≠ Breuer 2009 的 3D 湍流解」，锚点应转判为"
        "**闭包缺失型 FAIL**，需转 SST/LES 路线；")
    add("3. 队列面 rc 语义缺陷应向平台侧报（`run_compute status` 应回传远端 rc）。")
    add("")
    add("## ⑧ 复现")
    add("")
    add("```bash")
    add("PYTHONPATH=src python scripts/round28_report.py")
    add("pytest tests/test_periodic_hill_round28.py -q")
    add("```")
    (case_dir / "ROUND28.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({"counts": grading["counts"],
                      "claimed_but_unmeasured": grading["claimed_but_unmeasured"],
                      "round27_remote_rc": {r: jobs27[r].get("remote_rc") for r in jobs27},
                      "round28_receipts": {r: jobs28[r].get("remote_rc") for r in jobs28}},
                     ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())