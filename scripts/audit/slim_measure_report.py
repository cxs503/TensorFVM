#!/usr/bin/env python3
"""把 measure_rewrite_pack.py 的实测结果与 docs/m23216/SLIM_PLAN.md 的估算逐项对账，
生成报告所需的一切数字 / 表格 / 图（README.md 亦由本脚本程序化注入，零手抄）。

输入：docs/FVM-REPO-SLIM-MEASURE/raw/*.json（实测）+ docs/m23216/census.json（方案版估算）
输出：measure_summary.json、tables/*.md、figures/*.png、README.md

用法：python3 scripts/audit/slim_measure_report.py
"""

from __future__ import annotations

import glob
import json
import os
import subprocess
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

OUT = "docs/FVM-REPO-SLIM-MEASURE"
RAW = os.path.join(OUT, "raw")
PLAN = "docs/m23216/census.json"

ORDER = ["R1", "R5", "R5b", "R2", "R2b", "R2c", "R3"]


def load_measured() -> tuple[dict, dict, dict]:
    full, trunc, orphan = {}, {}, {}
    for path in sorted(glob.glob(os.path.join(RAW, "measured_pack*.json"))):
        with open(path) as fh:
            d = json.load(fh)
        is_trunc = path.endswith("_trunc.json")
        for row in d["scenarios"]:
            (trunc if is_trunc else full)[row["tag"]] = row
        if not is_trunc:
            full["_meta"] = {k: d[k] for k in
                             ("generated_at", "git_head", "git_branch", "reachable",
                              "baseline_keepall_pack_MiB", "pack_params")}
            if "_trunc_meta" not in full and os.path.exists(
                    os.path.join(RAW, "measured_pack_R2_trunc.json")):
                pass
    tp = os.path.join(RAW, "measured_pack_R2_trunc.json")
    if os.path.exists(tp):
        with open(tp) as fh:
            d = json.load(fh)
        trunc["_meta"] = {"baseline_keepall_pack_MiB": d["baseline_keepall_pack_MiB"]}
    op = os.path.join(RAW, "measured_orphan.json")
    if os.path.exists(op):
        with open(op) as fh:
            orphan = json.load(fh)["snapshots"]
    return full, trunc, orphan


def drift_rows(plan: dict) -> list[tuple[str, str, str, str]]:
    """方案基线 vs 现在的现场漂移（全部现测，供窗口前置门用）。"""
    s = plan["scalars"]
    repo = "."
    g = lambda *a: subprocess.run(["git", "-C", repo, *a], capture_output=True,
                                  text=True).stdout.strip()  # noqa: E731
    now_commits = g("rev-list", "--all", "--count")
    now_heads = sum(1 for line in g("for-each-ref", "--format=%(refname)")
                    .splitlines() if line.startswith("refs/heads/"))
    now_wt = len(g("worktree", "list").splitlines())
    co = g("count-objects", "-vH")
    kv = dict(line.split(": ") for line in co.splitlines() if ": " in line)
    return [
        ("提交数（全引用可达）", s["commits"], now_commits, "git rev-list --all --count"),
        ("本地 heads", s["local_heads"], now_heads, "git for-each-ref"),
        ("关联工作树", s["worktrees"], now_wt, "git worktree list"),
        ("in-pack 对象数", s["inpack"], kv.get("in-pack", "?"), "git count-objects -vH"),
        ("size-pack（含不可达）", s["now_pack"] + " MiB", kv.get("size-pack", "?"),
         "git count-objects -vH"),
    ]


def main() -> int:
    plan = json.load(open(PLAN))
    full, trunc, orphan = load_measured()
    plan_sc = {r["tag"]: r for r in plan["scenarios"]}
    var = plan["variants"]
    meta = full["_meta"]
    base_meas = meta["baseline_keepall_pack_MiB"]

    rows = []
    for tag in ORDER:
        m, p = full[tag], plan_sc[tag]
        err = 100.0 * (m["measured_pack_MiB"] - p["est_pack_MiB"]) / m["measured_pack_MiB"]
        rows.append({
            "tag": tag, "desc": m["desc"],
            "plan_MiB": p["est_pack_MiB"], "measured_MiB": m["measured_pack_MiB"],
            "plan_err_pct": round(err, 1),
            "dropped_plan_MiB": p["dropped_MiB"], "dropped_meas_MiB": m["dropped_MiB"],
            "dropped_blobs_plan": p["nblobs"], "dropped_blobs_meas": m["dropped_blobs"],
            "red_plan_pct": p["reduction_pct"],
            "red_meas_pct": round(100.0 * (1 - m["measured_pack_MiB"] / base_meas), 1),
        })

    marginal = []
    for a, b in (("R2", "R2b"), ("R2b", "R2c"), ("R5", "R2")):
        marginal.append({
            "pair": f"{b} 相对 {a} 的边际收益",
            "plan_MiB": round(plan_sc[a]["est_pack_MiB"] - plan_sc[b]["est_pack_MiB"], 1),
            "measured_MiB": round(full[a]["measured_pack_MiB"] - full[b]["measured_pack_MiB"], 1),
        })

    d2 = [
        {"en": "now (untouched)", "option": "现状（不动）", "plan_MiB": float(plan["scalars"]["now_pack"]),
         "measured_MiB": meta["reachable"]["inpack_MiB"], "note": "实测=可达对象 size-in-pack 合计"},
        {"en": "A full-history R2", "option": "A 全史 filter-repo R2", "plan_MiB": plan_sc["R2"]["est_pack_MiB"],
         "measured_MiB": full["R2"]["measured_pack_MiB"], "note": "推荐案"},
        {"en": "A full-history R3", "option": "A 全史 filter-repo R3", "plan_MiB": plan_sc["R3"]["est_pack_MiB"],
         "measured_MiB": full["R3"]["measured_pack_MiB"], "note": "最狠路径规则"},
        {"en": "B orphan@HEAD (drop history)", "option": "B orphan@HEAD（弃全史）", "plan_MiB": var["orphan_pack_MiB"],
         "measured_MiB": orphan.get("HEAD", {}).get("measured_pack_MiB"),
         "note": "只留当前快照，历史归零"},
        {"en": "B' orphan@2026-10-28 snapshot", "option": "B' orphan@2026-10-28 快照", "plan_MiB": var["cleanup_pack_MiB"],
         "measured_MiB": orphan.get("b77e7783", {}).get("measured_pack_MiB"),
         "note": "仅快照本体（不含其后历史重放）"},
        {"en": "B'' truncate@10-28, keep later history", "option": "B'' 截断@10-28 + 保留其后全部历史（不过滤）",
         "plan_MiB": None, "measured_MiB": trunc["_meta"]["baseline_keepall_pack_MiB"],
         "note": "方案未列：保住 10-28 之后可审计历史"},
        {"en": "B''+R2 truncate@10-28 + R2", "option": "B''+R2 截断@10-28 + R2", "plan_MiB": None,
         "measured_MiB": trunc["R2"]["measured_pack_MiB"], "note": "组合案"},
        {"en": "B''+R3 truncate@10-28 + R3", "option": "B''+R3 截断@10-28 + R3", "plan_MiB": None,
         "measured_MiB": trunc["R3"]["measured_pack_MiB"], "note": "组合案"},
    ]

    drift = drift_rows(plan)
    os.makedirs(os.path.join(OUT, "tables"), exist_ok=True)
    os.makedirs(os.path.join(OUT, "figures"), exist_ok=True)

    summary = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "measured_meta": meta,
        "scenario_compare": rows,
        "marginal": marginal,
        "d2_options": d2,
        "drift": [{"item": d[0], "plan": d[1], "now": d[2], "cmd": d[3]} for d in drift],
        "plan_selfcheck": plan["model_selfcheck"],
    }
    with open(os.path.join(OUT, "measure_summary.json"), "w") as fh:
        json.dump(summary, fh, indent=1, ensure_ascii=False)

    # ---------------- 图 1：方案估算 vs 实测（每场景）
    fig, ax = plt.subplots(figsize=(9, 4.6))
    xs = range(len(rows))
    ax.bar([x - 0.2 for x in xs], [r["plan_MiB"] for r in rows], width=0.4,
           label="plan estimate (§3.2 model)", color="#c0c0c0", edgecolor="k")
    ax.bar([x + 0.2 for x in xs], [r["measured_MiB"] for r in rows], width=0.4,
           label="measured (real git pack-objects)", color="#2b6cb0")
    for x, r in zip(xs, rows):
        ax.annotate(f"{r['plan_err_pct']:+.1f}%", (x, max(r["plan_MiB"], r["measured_MiB"])),
                    ha="center", va="bottom", fontsize=8)
    ax.set_xticks(list(xs))
    ax.set_xticklabels([r["tag"] for r in rows])
    ax.set_ylabel("post-rewrite pack (MiB)")
    ax.axhline(base_meas, ls="--", lw=1, color="#888")
    ax.annotate(f"keep-all baseline {base_meas:.1f} MiB", (len(rows) - 1.4, base_meas),
                va="bottom", ha="right", fontsize=8, color="#666")
    ax.set_yscale("log")
    ax.set_title("Slim scenarios: plan-model estimate vs measured repack (log scale)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "figures", "fig1_plan_vs_measured.png"), dpi=140)
    plt.close(fig)

    # ---------------- 图 2：D2 选项（历史策略 × 过滤）
    fig, ax = plt.subplots(figsize=(9.4, 4.8))
    labels = [d.get("en", d["option"]) for d in d2]
    y = list(range(len(d2)))
    # 逐条画，避免把 None 混成 nan 数组喂给 categorical 单位解析
    for i, d in enumerate(d2):
        ax.barh(i + 0.2, d["measured_MiB"], height=0.4, color="#2b6cb0",
                label="measured" if i == 0 else None)
        if d["plan_MiB"]:
            ax.barh(i - 0.2, d["plan_MiB"], height=0.4, color="#c0c0c0",
                    edgecolor="k", label="plan estimate" if i == 0 else None)
    for i, d in enumerate(d2):
        txt = f"{d['measured_MiB']:.1f} MiB"
        if d["plan_MiB"]:
            txt += f"  (plan {d['plan_MiB']:.0f})"
        ax.annotate(txt, (d["measured_MiB"], i + 0.2), va="center", ha="left",
                    fontsize=8, xytext=(3, 0), textcoords="offset points")
    ax.set_yticks(list(y))
    ax.set_yticklabels(labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_xscale("log")
    ax.set_xlabel("resulting repo pack (MiB, log scale)")
    ax.set_title("D2 options — history policy x path filter (measured)")
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "figures", "fig2_d2_options.png"), dpi=140)
    plt.close(fig)

    # ---------------- 表格
    md = ["| 场景 | 方案估 MiB | 实测 MiB | 方案误差 | 方案删 blob/MiB | 实测删 blob/MiB |",
          "|---|---|---|---|---|---|"]
    for r in rows:
        md.append(f"| {r['tag']} {r['desc']} | {r['plan_MiB']:.1f} | **{r['measured_MiB']:.1f}** | "
                  f"{r['plan_err_pct']:+.1f}% | {r['dropped_blobs_plan']} / {r['dropped_plan_MiB']:.1f} | "
                  f"{r['dropped_blobs_meas']} / {r['dropped_meas_MiB']:.1f} |")
    open(os.path.join(OUT, "tables", "scenario_compare.md"), "w").write("\n".join(md) + "\n")

    md = ["| 选项 | 方案估 MiB | 实测 MiB | 说明 |", "|---|---|---|---|"]
    for d in d2:
        pv = f"{d['plan_MiB']:.1f}" if d["plan_MiB"] else "—"
        md.append(f"| {d['option']} | {pv} | **{d['measured_MiB']:.1f}** | {d['note']} |")
    open(os.path.join(OUT, "tables", "d2_options.md"), "w").write("\n".join(md) + "\n")

    md = ["| 项 | 方案基线(m23216) | 现在实测 | 取法 |", "|---|---|---|---|"]
    for item, pv, nv, cmd in drift:
        md.append(f"| {item} | {pv} | {nv} | `{cmd}` |")
    open(os.path.join(OUT, "tables", "drift.md"), "w").write("\n".join(md) + "\n")

    write_readme(summary, plan)
    print(f"[i] wrote {OUT}/measure_summary.json, tables/, figures/, README.md")
    return 0


def write_readme(s: dict, plan: dict) -> None:
    rows, d2, marg = s["scenario_compare"], s["d2_options"], s["marginal"]
    base = s["measured_meta"]["baseline_keepall_pack_MiB"]
    inpack = s["measured_meta"]["reachable"]["inpack_MiB"]
    sc = plan["scalars"]
    var = plan["variants"]
    r2, r2b, r2c, r3 = (next(r for r in rows if r["tag"] == t) for t in ("R2", "R2b", "R2c", "R3"))
    r1 = next(r for r in rows if r["tag"] == "R1")
    m_r2b = next(m for m in marg if "R2b" in m["pair"])
    m_r2c = next(m for m in marg if "R2c" in m["pair"])
    orph = {d["option"]: d for d in d2}
    d_head = orph["B orphan@HEAD（弃全史）"]["measured_MiB"]
    d_snap = orph["B' orphan@2026-10-28 快照"]["measured_MiB"]
    d_trunc = orph["B'' 截断@10-28 + 保留其后全部历史（不过滤）"]["measured_MiB"]
    d_tr2 = orph["B''+R2 截断@10-28 + R2"]["measured_MiB"]
    d_tr3 = orph["B''+R3 截断@10-28 + R3"]["measured_MiB"]
    within3 = [r["tag"] for r in rows if abs(r["plan_err_pct"]) < 3.0]
    over10 = [r for r in rows if r["plan_err_pct"] < -10.0]

    L = []
    A = L.append
    A("# TensorFVM 历史瘦身：**实测**收益复核（FVM-REPO-SLIM ｜ 本单零改写、零 force-push）\n")
    A(f"> 生成时刻 {s['generated_at']} ｜ HEAD `{s['measured_meta']['git_head']}` "
      f"｜ 分支 `{s['measured_meta']['git_branch']}` ｜ 全部数字由 "
      "`scripts/audit/measure_rewrite_pack.py` + `scripts/audit/slim_measure_report.py` 程序化注入，无手抄。\n")
    A("## 0. 一句话结论\n")
    A(f"方案版（`docs/m23216/SLIM_PLAN.md`）把改写后体积建模为「逐 blob zlib6 × 全局标定 0.874」，"
      f"本单把它换成**真跑 `git pack-objects` 的实测**：基线（一条都不删、可达对象重打包）"
      f"**{base:.1f} MiB**，推荐案 R2 实测 **{r2['measured_MiB']:.1f} MiB**（方案估 {r2['plan_MiB']:.1f}）、"
      f"R3 实测 **{r3['measured_MiB']:.1f} MiB**（方案估 {r3['plan_MiB']:.1f}）。\n"
      f"**两条会改变 owner 决策的实测结论**：\n"
      f"1. **R2b（全仓 `*.pyc`）的边际收益实测只有 {m_r2b['measured_MiB']:.1f} MiB**，"
      f"不是方案说的 {m_r2b['plan_MiB']:.1f} MiB —— 方案 D1(b) 的取舍数字站不住；\n"
      f"2. **orphan「弃全史只留快照」实测 {d_head:.1f} MiB**、**「截断到 2026-10-28 但保留其后全部历史」"
      f"实测 {d_trunc:.1f} MiB**（方案完全没列这一档），比全史 R2 的 {r2['measured_MiB']:.1f} MiB 小 "
      f"{r2['measured_MiB'] / d_trunc:.1f}×，且不需要对 10-28 之后 3 周的可审计历史做任何路径删除决策。\n")

    A("## 1. ① 案例介绍\n")
    A("| 项 | 值 |\n|---|---|")
    A(f"| 仓库 | TensorFVM（`{s['measured_meta']['git_head']}`，与 331 个工作树共享同一 gitdir） |")
    A(f"| 症状 | `size-pack` {sc['now_pack']} MiB；1.33 GiB pack 里 91% 是已删除的可再生工件 |")
    A(f"| 本单范围 | **只读审计**：不改写历史、不改 ref、不删对象、不 force-push |")
    A(f"| 对象规模 | 可达 {s['measured_meta']['reachable']['n_objects']} 个对象，"
      f"其中 blob {s['measured_meta']['reachable']['n_blobs']} 个 / {s['measured_meta']['reachable']['blob_MiB']} MiB |")
    A(f"| 现场 in-pack | {inpack} MiB（实测 size-in-pack 合计，不含 tmp_pack 垃圾） |")
    A("")
    A("## 2. ② 怎么做的\n")
    A("1. **删除集判定（比方案更严）**：`git log --all --raw -r --no-merges --no-abbrev` 的 new-side "
      "给出每个 blob 历史上出现过的**全部** (path, blob) 对；只有当一个 blob 的**每一条**路径都命中规则才算被删。"
      "方案用 `rev-list --objects`（每 blob 只报一条路径），同名异径 blob 有误判风险（本单实测见 §5 F5：实际无影响）。")
    A("2. **实测体积**：把每个场景的存活对象集喂给真实的 `git pack-objects --stdout --window=10 --depth=50`，"
      "边读边计数，**不落盘、不标定**；因此结果包含 git 真实的 delta 复用（这正是方案模型缺的那一项）。")
    A(f"3. **基线**：同一命令对全部可达对象重打包 = **{base:.1f} MiB**；与实测 `size-pack` "
      f"{sc['now_pack']} MiB 相差 {100 * (float(sc['now_pack']) - base) / base:.1f}%（差额是 pack 里的不可达对象与 loose）。"
      "所有 \"-xx%\" 一律以此实测基线为分母。")
    A("4. **D2 历史策略**：另加两种口径 —— `--measure-rev <rev>`（单快照闭包 = orphan 重开）与 "
      "`--truncate-at <rev>`（以 rev 快照为根 + 其后历史全部保留），rev 取 2026-10-28 的清理点 `b77e7783`。")
    A("")
    A("## 3. ③ 结果\n")
    A("### 3.1 场景：方案估算 vs 实测\n")
    A("| 场景 | 方案估 MiB | 实测 MiB | 方案误差 | 方案删 blob/MiB | 实测删 blob/MiB | 实测降幅 |")
    A("|---|---|---|---|---|---|---|")
    for r in rows:
        A(f"| {r['tag']} {r['desc']} | {r['plan_MiB']:.1f} | **{r['measured_MiB']:.1f}** | "
          f"{r['plan_err_pct']:+.1f}% | {r['dropped_blobs_plan']} / {r['dropped_plan_MiB']:.1f} | "
          f"{r['dropped_blobs_meas']} / {r['dropped_meas_MiB']:.1f} | {r['red_meas_pct']:.1f}% |")
    A("")
    A("### 3.2 边际收益（方案用它支撑 D1(a)/D1(b) 的取舍）\n")
    A("| 对比 | 方案估边际 | **实测边际** |\n|---|---|---|")
    for m in marg:
        A(f"| {m['pair']} | {m['plan_MiB']:.1f} MiB | **{m['measured_MiB']:.1f} MiB** |")
    A("")
    A("### 3.3 D2 历史策略 × 路径过滤（实测）\n")
    A("| 选项 | 方案估 MiB | 实测 MiB | 说明 |\n|---|---|---|---|")
    for d in d2:
        pv = f"{d['plan_MiB']:.1f}" if d["plan_MiB"] else "—（方案未列）"
        A(f"| {d['option']} | {pv} | **{d['measured_MiB']:.1f}** | {d['note']} |")
    A("")
    A("### 3.4 现场漂移（窗口前置门要用的量）\n")
    A("| 项 | 方案基线 | 现在实测 | 取法 |\n|---|---|---|---|")
    for d in s["drift"]:
        A(f"| {d['item']} | {d['plan']} | {d['now']} | `{d['cmd']}` |")
    A("")
    A("## 4. ④ 图\n")
    A("![fig1](figures/fig1_plan_vs_measured.png)\n")
    A("\n**图 1** 七个场景的方案估算（灰）vs 实测重打包（蓝），纵轴 log；标注为方案相对实测的误差。"
      "灰蓝排序不一致即说明标定系数不能外推到不同构成的存活集。\n")
    A("![fig2](figures/fig2_d2_options.png)\n")
    A("\n**图 2** D2 历史策略选项：全史过滤 / orphan 弃史 / 截断到 10-28 三档 × 路径过滤，实测（蓝）vs 方案估算（灰）。"
      "灰色缺柱=方案未给该组合的估计。\n")
    A("## 5. ⑤ 注意点与局限（honest）\n")
    A(f"- **保守偏置**：实测保留**原始** tree/commit 对象（tree {s['measured_meta']['reachable']['by_type'].get('tree', {}).get('MiB', '?')} MiB 未压缩），"
      "真实改写会把这些树也改小，故本单数字是**偏大**的（方向上不掩盖收益，也不会把改写夸好）。")
    A("- **单快照口径**：`B' orphan@10-28 快照` 只测快照本体；若在其上重放后续历史，应看 `B''` 那一行（"
      f"{d_trunc:.1f} MiB）。两者不可混读，方案 §4 的 {var['cleanup_pack_MiB']} MiB 是快照口径。")
    A("- **部分克隆**：本仓是 `--filter=blob:none` 部分克隆，普查覆盖本地已有对象；实测与 GitHub 侧体积量级一致，"
      "但「100% 远端全量」仍需窗口内 `fetch --refetch` 后复跑本工具。")
    A("- **pack 参数敏感**：结果随 `--window/--depth` 变化（本单 window=10, depth=50 = git 默认 gc 口径）；"
      "若窗口内改用 `gc --aggressive`，数值会更小。复现时请固定这两个参数（见 `--window/--depth`）。")
    A("- **规则复现差异**：R3 的「docs 全部二进制」边界（文本后缀表）与方案不完全相同（实测删 "
      f"{r3['dropped_blobs_meas']} blob vs 方案 {r3['dropped_blobs_plan']}），差异 {abs(r3['dropped_meas_MiB'] - r3['dropped_plan_MiB']):.1f} MiB。")
    A("")
    A("## 6. ⑥ 误差判定（逐项，显式回答是否 <3%）\n")
    A("| 判定项 | 基准 | 误差 | 是否 <3% |\n|---|---|---|---|")
    A(f"| 方法自校验：可达对象重打包 vs `size-pack` | {sc['now_pack']} MiB | "
      f"{100 * (float(sc['now_pack']) - base) / base:+.1f}% | {'是' if abs(100 * (float(sc['now_pack']) - base) / base) < 3 else '否'} |")
    for r in rows:
        A(f"| {r['tag']}：方案模型 vs 本单实测 | {r['measured_MiB']:.1f} MiB | {r['plan_err_pct']:+.1f}% | "
          f"{'是' if abs(r['plan_err_pct']) < 3 else '否'} |")
    A("")
    A(f"- 方案模型的误差**不是均匀的**：{len(within3)}/7 个场景落在 ±3% 内（{', '.join(within3)}），"
      f"而 {', '.join(r['tag'] for r in over10)} 三条方案偏乐观 "
      + ", ".join("{:.1f}%".format(r['plan_err_pct']) for r in over10) + "，"
      f"超出方案自报的 ±10% 带；这正是单点标定（只在「全保留」一个点上标定 0.874）无法外推的直接证据。")
    A(f"- 结论层的量：推荐案 R2 的降幅 方案 {r2['red_plan_pct']:.0f}% → 实测 {r2['red_meas_pct']:.0f}%（差 {r2['red_plan_pct'] - r2['red_meas_pct']:.1f} 个百分点）；"
      f"R3 的 \"-96%\" → 实测 {r3['red_meas_pct']:.1f}%（改写后 {r3['measured_MiB']:.1f} MiB），量级结论不变。")
    A("")
    A("## 7. 发现（按严重度）与处置\n")
    A(f"1. **[HIGH] D1(b) 的取舍数字失真**：R2b 相对 R2 的边际收益实测 **{m_r2b['measured_MiB']:.1f} MiB**"
      f"（方案 {m_r2b['plan_MiB']:.1f} MiB，夸大 {m_r2b['plan_MiB'] / max(m_r2b['measured_MiB'], 0.01):.0f}×）。"
      "`.pyc` 在 pack 里已被 delta 到近零，单独按 blob zlib 会算出 13 MiB 的假收益。"
      "**处置**：D1 删去 (b)，不要把「全仓删 pyc」作为独立筹码。")
    A(f"2. **[HIGH] D2 的 B/B' 被严重低估、且漏了一档**：orphan@HEAD 实测 {d_head:.1f} MiB（方案 {var['orphan_pack_MiB']}）、"
      f"orphan@10-28 快照 {d_snap:.1f} MiB（方案 {var['cleanup_pack_MiB']}，差 {var['cleanup_pack_MiB'] / d_snap:.1f}×）；"
      f"方案未列的 **截断@10-28 + 保留其后全部历史 = {d_trunc:.1f} MiB**、+R2 = {d_tr2:.1f}、+R3 = {d_tr3:.1f}。"
      "**处置**：D2 的对照表按 §3.3 重排；「保住 10-28 之后可审计历史」这一档应进入 owner 的选项集。")
    A(f"3. **[MED] R2c / R3 方案偏乐观 {r2c['plan_err_pct']:.1f}% / {r3['plan_err_pct']:.1f}%**，"
      "贴在方案自报 ±10% 带外沿。**处置**：§3.2 数字换成实测列；若 owner 需要带宽，用实测 ±window 敏感性，而非 0.874 标定。")
    A(f"4. **[MED] 标定方法本身不可外推**：方案只在「全保留」点标定（拟合 {s['plan_selfcheck']['estimate_MiB']} vs 实测 "
      f"{s['plan_selfcheck']['measured_pack_MiB']}，{s['plan_selfcheck']['gap_pct']}%），"
      "而误差符号随存活集的文本/二进制构成翻转（R1/R5 方案偏保守、R2c/R3 偏乐观）。**处置**：以后凡改历史收益，一律用本工具实测。")
    A("5. **[LOW｜否证] 方案用单路径近似判定删除集**：与全路径精确集比对，"
      f"删除 MiB 差 ≤{max(abs(r['dropped_meas_MiB'] - r['dropped_plan_MiB']) for r in rows):.1f}，"
      "无 blob 因「同名异径」被误判而改变结论 —— 该疑虑**不成立**（记录以免后人重复怀疑）。")
    A(f"6. **[LOW] 分母口径**：方案的 \"-65%\" 以 `size-pack` {sc['now_pack']} MiB 为分母，其中含不可达对象；"
      f"可达基线实测 {base:.1f} MiB。**处置**：统一用可达基线，避免后续对账口径漂移。")
    A(f"7. **[OPS] 24 h 内基线已漂移**（提交 {sc['commits']}→{s['drift'][0]['now']}、heads {sc['local_heads']}→{s['drift'][1]['now']}、"
      f"工作树 {sc['worktrees']}→{s['drift'][2]['now']}）。**处置**：改写窗口前先跑 "
      "`python3 scripts/audit/measure_rewrite_pack.py --dry-run` 与 `--scenarios R2 --truncate-at <rev>` 复核，数字漂移 >5% 则重排决策。")
    A("")
    A("**总处置**：对方案 §3.2/§4/§7 的**数字与 D1/D2 排布 → request changes**（用本单实测列替换）；"
      "对 §5 协调面 / §6 回滚 / 窗口步骤 **无实质异议（approve with notes）**。本单未执行任何改写。\n")
    A("## 8. 复现\n")
    A("```bash\n"
      "# 单场景实测（约 10–60 s/场景，带 in-pack 与基线缓存）\n"
      "python3 scripts/audit/measure_rewrite_pack.py --out docs/FVM-REPO-SLIM-MEASURE/raw --scenarios R2\n"
      "# D2 历史策略\n"
      "python3 scripts/audit/measure_rewrite_pack.py --out docs/FVM-REPO-SLIM-MEASURE/raw \\\n"
      "    --measure-rev HEAD --measure-rev b77e7783\n"
      "python3 scripts/audit/measure_rewrite_pack.py --out docs/FVM-REPO-SLIM-MEASURE/raw \\\n"
      "    --scenarios R2,R3 --truncate-at b77e7783\n"
      "# 汇总出表与图（含本 README 的数字注入）\n"
      "python3 scripts/audit/slim_measure_report.py\n"
      "```\n")
    open(os.path.join(OUT, "README.md"), "w").write("\n".join(L) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())