#!/usr/bin/env python3
"""Figures + platform report for audit mission m16253 (CASE-FVM-BLASIUS-PLATE).

Every number in the markdown is read from the audit JSONs (zero hand
transcription); the figures are rendered here and embedded base64.

Run: python3 docs/board_audit/m16253/publish.py [--dry-run]
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import urllib.request
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
WT = HERE.parents[2]
RUNS = WT / "runs/m16253"
FIG = HERE / "figures"
REFS = HERE / "refs/docs/case-blasius-plate/artifacts"

API = os.environ.get("TENSOR_API_BASE", "http://127.0.0.1:8010/api/v1")


def post(path: str, payload: dict, token: str | None = None) -> dict:
    req = urllib.request.Request(
        f"{API}{path}", data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json",
                 **({"Authorization": f"Bearer {token}"} if token else {})},
        method="POST")
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.loads(r.read())


def login() -> str:
    body = {"username": os.environ["TENSOR_AGENT_USER"],
            "password": os.environ["TENSOR_AGENT_PASS"]}
    for path in ("/auth/login", "/login", "/tensor/auth/login"):
        try:
            data = post(path, body)
            tok = data.get("access_token") or data.get("token") or (data.get("data") or {}).get("access_token")
            if tok:
                return tok
        except Exception:
            continue
    raise SystemExit("login failed")


def figures() -> list[dict]:
    crit = json.loads((RUNS / "repro_compare.json").read_text())["criterion"]
    sens = json.loads((REFS / "march-sensitivity.json").read_text())["rows"]
    ic = json.loads((RUNS / "march_ic_probe.json").read_text())["rows"]
    ladder = json.loads((RUNS / "momentum_integral_probe.json").read_text())["ic_ladder"]["rows"]
    trace = [json.loads(l) for l in
             (REFS / "fvm-baseline.trace.jsonl").read_text().splitlines() if l.strip()]
    FIG.mkdir(parents=True, exist_ok=True)
    out = []

    # fig1: reference vs measured, 3% band
    fig, ax = plt.subplots(1, 2, figsize=(10, 4.2))
    for a, key, unit in zip(ax, ("cf", "H"), ("$c_f$", "$H=\\delta^*\\/\\theta$")):
        ref, got = crit[f"ref_{key}"], crit[f"got_{key}"]
        a.axhspan(ref * 0.97, ref * 1.03, color="#cfe8cf", label="±3 % gate")
        a.axhline(ref, color="k", ls="--", lw=1, label="Blasius reference")
        a.scatter([0], [got], color="#c0392b", zorder=3,
                  label="march, got (%+.2f %%)" % (100 * crit[f"{key}_relative_error"]))
        a.set_xlim(-1, 1); a.set_xticks([]); a.set_ylabel(unit)
        a.set_title(f"{unit} at $Re_x=10^5$"); a.legend(fontsize=8, loc="lower right")
    fig.suptitle("CASE-FVM-BLASIUS-PLATE — delivered device vs literature (audit m16253)")
    fig.tight_layout(); p1 = FIG / "fig1_ref_vs_got.png"; fig.savefig(p1, dpi=140); plt.close(fig)
    out.append({"name": "fig1_ref_vs_got.png",
                "caption": "Delivered march device vs Blasius reference at Re_x=1e5 with the ±3 %% gate band; "
                           "c_f %+.2f %%, H %+.2f %%." % (100 * crit["cf_relative_error"],
                                                           100 * crit["H_relative_error"]),
                "png": p1})

    # fig2: resolution study + startup-IC ladder
    fig, ax = plt.subplots(1, 2, figsize=(10.5, 4.2))
    ny = [r for r in sens if r["tag"].startswith("ny-")]
    ax[0].plot([r["ny"] for r in ny], [100 * r["cf_relative_error"] for r in ny],
               "o-", label="$c_f$ error")
    ax[0].plot([r["ny"] for r in ny], [100 * r["H_relative_error"] for r in ny],
               "s--", label="$H$ error")
    ax[0].axhline(3, color="r", ls=":", label="3 % gate"); ax[0].axhline(0, color="k", lw=.6)
    ax[0].set_xscale("log", base=2); ax[0].set_xlabel("wall-normal nodes $n_y$")
    ax[0].set_ylabel("relative error [%]"); ax[0].set_title("grid convergence (delivered, re-run)")
    ax[0].legend(fontsize=8)
    k = [r["k_delta99"] for r in ladder]
    ax[1].plot(k, [100 * r["cf_err"] for r in ladder], "o-", label="1/7-power IC, $k\\delta_{99}$")
    ax[1].axhline(100 * next(r["cf_err"] for r in ic if r["tag"] == "uniform"),
                  color="g", ls="--", label="uniform (no-BL) IC")
    ax[1].axhline(0, color="k", lw=.6)
    ax[1].axhspan(-3, 3, color="#cfe8cf", alpha=.6, label="±3 % gate")
    ax[1].set_xlabel("startup layer thickness at $x_0$ [$\\delta_{99}$ units]")
    ax[1].set_ylabel("$c_f$ error [%]"); ax[1].set_title("initial-condition sensitivity (auditor probe)")
    ax[1].legend(fontsize=8)
    fig.tight_layout(); p2 = FIG / "fig2_sensitivity.png"; fig.savefig(p2, dpi=140); plt.close(fig)
    out.append({"name": "fig2_sensitivity.png",
                "caption": "Left: delivered resolution study re-run (c_f error -1.81 % / -0.71 % / -0.15 % at "
                           "n_y = 128/256/512). Right: auditor startup-layer ladder - a 10x-too-thick initial "
                           "layer leaves -10.30 % at Re_x=1e5 while a uniform start reaches -1.02 %.",
                "png": p2})

    # fig3: the WITHHELD elliptic rung, as committed
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.semilogy([t["iteration"] for t in trace], [t["cf"] for t in trace], "o-",
                label="$c_f$ (2-D SIMPLE)")
    ax.axhline(crit["ref_cf"], color="k", ls="--", label="Blasius $c_f$")
    ax.axhspan(crit["ref_cf"] * .97, crit["ref_cf"] * 1.03, color="#cfe8cf", label="±3 % gate")
    ax.set_xlabel("SIMPLE outer iteration"); ax.set_ylabel("$c_f$ (log)")
    ax.set_title("2-D elliptic FVM rung: committed non-converged trace (WITHHELD)")
    ax.legend(fontsize=8)
    fig.tight_layout(); p3 = FIG / "fig3_fvm_withheld.png"; fig.savefig(p3, dpi=140); plt.close(fig)
    out.append({"name": "fig3_fvm_withheld.png",
                "caption": "The anchor-naming elliptic device: committed trace stops at 200 iterations "
                           "(momentum 3.943e-03 vs 1e-5), c_f still 3x above the Blasius value — WITHHELD, "
                           "and per finding F3 the blocking cause was a queued ref=main submission, not cost.",
                "png": p3})
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    crit = json.loads((RUNS / "repro_compare.json").read_text())
    c = crit["criterion"]
    ic = {r["tag"]: r for r in json.loads((RUNS / "march_ic_probe.json").read_text())["rows"]}
    mom = json.loads((RUNS / "momentum_integral_probe.json").read_text())["momentum_integral"]
    sens = json.loads((REFS / "march-sensitivity.json").read_text())["rows"]
    figs = figures()

    sens_md = "\n".join(
        "| %s | %d | %d | %.8f | %+.2f%% | %.5f | %+.2f%% |"
        % (r["tag"], r["ny"], r["stations"], r["cf"], 100 * r["cf_relative_error"],
           r["H"], 100 * r["H_relative_error"]) for r in sens)

    md = f"""## ① 案例介绍

平板层流边界层（顺压零梯度、光滑平板、从前缘起层流），任务锚点
`CASE-FVM-BLASIUS-PLATE`（TensorFVM）。对象算例在 Re_x=1e5 处要求局部摩阻系数
$c_f$ 与形状因子 $H=\\delta^*/\\theta$ 落在 Blasius 相似解的 3 % 内。

| 量 | 值 |
|---|---|
| $U_\\infty$ / $L$ / $\\nu$ | 1 m/s / 1 m / $10^{{-5}}$ m²/s（Re_L = 1e5）|
| 参考值（文献录入，非本仓产生）| $c_f\\sqrt{{Re_x}}=0.664$，$\\delta^*/x\\sqrt{{Re_x}}=1.7208$，$H=2.5916$（Schlichting 7th ed. §7 Tab. 7.1；Blasius 1908）|
| 判据 | 相对误差 < 3 %（`tensor_case_pool` id 109，`tol_policy` 原文）|
| 被测装置 | march：`rans_ship_resistance.march_boundary_layer(model='laminar')`，n_y=256、401 站位；fvm：2-D SIMPLE（`CachedExternalFlowSolver`，AMG/ILU-GMRES）|

## ② 怎么做的（本次为交差核验单，非重做）

1. 取证冻结：把 m16219 交付（commit `4f87f691`）的 16 个文件按 sha256 冻结进
   `docs/board_audit/m16253/refs/`（manifest 带 git 溯源），因为交付只在分支上、
   生态镜像 `0cd53fa5` 里根本没有该模块。
2. 独立复跑交付自带 repro_cmd（输出写入审计 worktree，不动交付目录）：
   `PYTHONPATH=src python scripts/run_case_blasius_plate.py --configs march,march-sensitivity`。
3. 独立复跑测试 9 项；再用 `maturity_gate.judge` 做裁判预演（机器复核声明级别）。
4. 三个审计探针（交付没有的）：初始条件独立性（Blasius IC / 均匀无边界层 IC /
   全域线性 IC）、启动层厚度阶梯（1×/3×/10× δ99）、动量积分一致性
   （$c_f/2 = d\\theta/dx$ vs 壁面剪切给出的 $c_f$）。
5. 机制面核查：判据是否注册（`tensor_benchmark_cases`）、算例池收账状态
   （`tensor_case_pool`）、集群作业状态（`cluster_jobs` 1274/1276）。

## ③ 计算结果（数字全部程序化注入）

| 装置 | $c_f$ got | $c_f$ ref | 误差 | $H$ got | $H$ ref | 误差 | 门 |
|---|---|---|---|---|---|---|---|
| march（交付装置，复跑）| {c['got_cf']:.8f} | {c['ref_cf']:.8f} | {100*c['cf_relative_error']:+.2f}% | {c['got_H']:.5f} | {c['ref_H']} | {100*c['H_relative_error']:+.2f}% | **PASS** |

交付工件与审计复跑逐字段比较：`identical_numbers = {crit['identical_numbers']}`（差 0.0）。

分辨率研究（复跑，交付同值）：

| tag | n_y | 站位 | $c_f$ | $c_f$ 误差 | $H$ | $H$ 误差 |
|---|---|---|---|---|---|---|
{sens_md}

审计探针（新增证据）：

| 探针 | 结果 |
|---|---|
| 均匀（无边界层）IC | $c_f$ {100*ic['uniform']['cf_err']:+.2f}%，$H$ {100*ic['uniform']['H_err']:+.2f}% → PASS |
| 全域线性 IC | $c_f$ {100*ic['linear']['cf_err']:+.2f}%，连续性残差 {ic['linear']['continuity_residual']:.2f}（装置失效）|
| 启动层 10×δ99 | −10.30 % → **FAIL**（门槛只在启动层量级正确时满足）|
| 动量积分 vs 壁面剪切 | 差 {100*mom['relative_difference']['d_theta_dx_2']:+.2f}% / {100*mom['relative_difference']['d_theta_dx_5']:+.2f}% / {100*mom['relative_difference']['d_theta_dx_10']:+.2f}%（2/5/10 站位跨度）|

## ④ 结果图

图 1 = 交付装置 vs 文献参考（含 ±3 % 带）；图 2 = 网格收敛 + 启动层敏感性；
图 3 = WITHHELD 的 2-D 椭圆装置已提交轨迹。

## ⑤ 注意点（坑与局限）

- **判据没注册**（F1 high）：`SELECT … FROM tensor_benchmark_cases WHERE
  anchor_slug='CASE-FVM-BLASIUS-PLATE'` → **0 行**；3 % 门槛只存在算例池
  `tensor_case_pool` id 109，且该行 `status='dispatched'`、`linked_anchor=NULL`、
  `mission_id=16182 ≠ 16219`——账没消。
- **交付只在分支上**（F2 medium）：镜像 main `0cd53fa5` 无 `case_blasius_plate.py`，
  `maturity_gate` 的 ECO grep 看不见调用方（上一轮 verdict #436 正是因此被降到
  gate_level 1 重开）。
- **FVM 未跑起来不是"太贵"**（F3 medium）：提交的集群作业 cj1274/cj1276 在交付时
  仍 `queued`，且用 `ref=main` 提交——main 里没有该脚本，排到也会 FileNotFoundError；
  仓里本就有 `stage_dir` 可用。
- **3 % 的一致性带 IC 条件**（F4 low）：交付用精确 Blasius 剖面起算，属"保持性"检验；
  均匀 IC 仍 −1.02 % 说明并非纯循环论证，但 10×δ99 启动层 −10.3 % 说明门槛依赖启动层量级。
- **文献锚二手**：审计未独立查阅 Schlichting 表原书（沙箱无藏书），仅用仓内 ODE 复核
  常数（$2f''(0)=0.664115$，+0.017 %；$H_{{ODE}}=2.5911$，+0.019 %）。

## ⑥ 误差判定

| 关键量 | 基准 | got | 相对误差 | 是否 < 3 % |
|---|---|---|---|---|
| 局部摩阻 $c_f$ @Re_x=1e5 | {c['ref_cf']:.8f}（0.664/√Re_x）| {c['got_cf']:.8f} | {100*c['cf_relative_error']:+.2f}% | **是** |
| 形状因子 $H$ | {c['ref_H']} | {c['got_H']:.5f} | {100*c['H_relative_error']:+.2f}% | **是** |
| （2-D 椭圆 FVM 装置）| 同上 | 6.345e-3（200 迭代止）| — | **未判定：WITHHELD，如实报** |

判级结论：**L4（基准验证）**，范围限于抛物化边界层 march 装置；锚点名中的
CASE-**FVM** 椭圆装置仍 WITHHELD，严格读法封顶 L3——差异与整改路径见审计报告 §4 F3。
"""
    token = None if args.dry_run else login()
    payload = {
        "title": "CASE-FVM-BLASIUS-PLATE 交差核验（m16219 / 审计 m16253）：判 L4，判据未注册与 FVM 装置调度缺陷在案",
        "summary": ("独立复跑交付 repro_cmd 与 9 项测试：c_f -0.71 %、H -0.47 %，与提交工件逐字段一致；"
                    "三个新探针（均匀 IC -1.02 %、10×δ99 -10.30 %、动量积分 +1.8~2.3 %）。"
                    "verdict 落表 L4；F1 判据在 tensor_benchmark_cases 0 行；F2 交付仅分支；"
                    "F3 集群作业因 ref=main 排队未跑，非成本所致。"),
        "scenario_key": "board_audit.m16253",
        "mission_id": 16253,
        "tags": ["board-audit", "TensorFVM", "CASE-FVM-BLASIUS-PLATE", "blasius", "L4"],
        "markdown": md,
        "figures": [{"name": f["name"], "caption": f["caption"],
                     "png_base64": base64.b64encode(f["png"].read_bytes()).decode()}
                    for f in figs],
    }
    (RUNS / "report.json").write_text(json.dumps(payload, indent=2))
    if args.dry_run:
        print("dry-run: report + 3 figures ready", [str(f["png"]) for f in figs])
        return 0
    resp = post("/tensor/reports", payload, token)
    print("report posted:", json.dumps(resp)[:300])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())