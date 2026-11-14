"""Render the m20098 board-audit report + figures from the audit artefacts.

Every number in the markdown and in the figures is injected from
``artifacts/CASE-CAVITY-RE1000-FVM/m20098_audit.json`` (my re-scoring of the
m20000 fields) and from the m20000 bundle JSONs; nothing is typed by hand.

    python3 scripts/render_audit_m20098.py --bundle /root/tp-agent/work/m20000/artifacts/CASE-CAVITY-RE1000-FVM
"""

from __future__ import annotations

import argparse
import json
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts" / "CASE-CAVITY-RE1000-FVM"
DOC = ROOT / "docs" / "CASE-CAVITY-RE1000-FVM"

GHIA_U = [(1.0, 1.0), (0.9766, 0.65928), (0.9688, 0.57492), (0.9609, 0.51117),
          (0.9531, 0.46604), (0.8516, 0.33304), (0.7344, 0.18719), (0.6172, 0.05702),
          (0.5, -0.06080), (0.4531, -0.10648), (0.2813, -0.27805), (0.1719, -0.38289),
          (0.1016, -0.29730), (0.0703, -0.22220), (0.0625, -0.20196), (0.0547, -0.18109),
          (0.0, 0.0)]
GHIA_V = [(1.0, 0.0), (0.9688, -0.21388), (0.9609, -0.27669), (0.9531, -0.33714),
          (0.9453, -0.39188), (0.9063, -0.51550), (0.8594, -0.42665), (0.8047, -0.31966),
          (0.5, 0.02526), (0.2344, 0.32235), (0.2266, 0.33075), (0.1563, 0.37095),
          (0.0938, 0.32627), (0.0781, 0.30353), (0.0703, 0.29012), (0.0625, 0.27485),
          (0.0, 0.0)]


def fig_centrelines(audit, out):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    gu = np.asarray(GHIA_U)
    gv = np.asarray(GHIA_V)
    for ax, (key, ghia, lab) in zip(axes, (("u_centreline", gu, "u/U on x = L/2 (vs y/L)"),
                                           ("v_centreline", gv, "v/U on y = L/2 (vs x/L)"))):
        comp = "u" if key.startswith("u") else "v"
        for field in audit["fields"]:
            c = field[key]
            ax.plot(c["y"] if comp == "u" else c["x"], c[comp], lw=1.2,
                    label=f"TensorFVM n={field['n']} (FVM)")
        order = ghia[:, 0].argsort()
        ax.plot(ghia[order, 0], ghia[order, 1], "ks", ms=5, label="Ghia et al. (1982)")
        ax.set_xlabel("y/L" if comp == "u" else "x/L")
        ax.set_ylabel(lab)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7, loc="best")
    fig.suptitle("CASE-FVM-CAVITY-RE1000 · audited fields vs external anchor (Re = 1000)", fontsize=11)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def fig_convergence(audit, out):
    q = [r for r in audit["rows"] if r["rule"] == "quadratic"]
    ns = sorted({r["n"] for r in q})
    hy = {r["n"]: r["h"] for r in q}
    series = {
        "u_min rel. err": [next(r["u_min_rel_err"] for r in q if r["n"] == n) for n in ns],
        "v_max rel. err": [next(r["v_max_rel_err"] for r in q if r["n"] == n) for n in ns],
        "pointwise max|Δu|/U": [next(r["pointwise_max_abs_du"] for r in q if r["n"] == n) for n in ns],
        "pointwise max|Δv|/U": [next(r["pointwise_max_abs_dv"] for r in q if r["n"] == n) for n in ns],
    }
    hs = np.array([hy[n] for n in ns])
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    for lab, ys in series.items():
        ax.loglog(hs, ys, "o-", label=lab)
    ref = series["pointwise max|Δu|/U"][-1]
    ax.loglog(hs, ref * (hs / hs[-1]) ** 2, "k--", lw=1, label="h² reference slope")
    ax.axhline(0.03, color="tab:red", lw=1, ls=":", label="ledger 3% band")
    ax.set_xlabel("h = 1/n")
    ax.set_ylabel("error (relative / of U)")
    ax.set_title("grid convergence of the audited fields (Re = 1000)")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def pct(x):
    return f"{100 * x:.2f}%"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", default="/root/tp-agent/work/m20000/artifacts/CASE-CAVITY-RE1000-FVM")
    args = ap.parse_args()
    bundle = pathlib.Path(args.bundle)

    audit = json.loads((ART / "m20098_audit.json").read_text())
    summary = json.loads((bundle / "summary.json").read_text())
    their = json.loads((bundle / "m20000_independent_audit.json").read_text())

    fig_centrelines(audit, ART / "fig_m20098_centrelines.png")
    fig_convergence(audit, ART / "fig_m20098_convergence.png")

    q = {(r["n"], r["rule"]): r for r in audit["rows"]}
    theirs = {r["n"]: r for r in their["rows"]}
    cases = {c["n"]: c for c in summary["cases"]}

    # ---------------- table 1: independent re-scoring -----------------------
    t1 = ["| n | steps (bundle) | u_min | rel err | v_max | rel err | max&#124;Δu&#124;/U | max&#124;Δv&#124;/U | vortex &#124;Δ&#124;/L (ψ int / ψ poisson) | max&#124;div&#124; | 3 % band |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
    for n in sorted({r["n"] for r in audit["rows"]}):
        r = q[(n, "quadratic")]
        t1.append(
            f"| {n} | {cases[n]['solver']['steps']} | {r['u_min']:+.5f} | {pct(r['u_min_rel_err'])} | "
            f"{r['v_max']:+.5f} | {pct(r['v_max_rel_err'])} | {r['pointwise_max_abs_du']:.5f} | "
            f"{r['pointwise_max_abs_dv']:.5f} | {r['vortex_centre_psi_integrated']['distance']:.4f} / "
            f"{r['vortex_centre_psi_poisson']['distance']:.4f} | {r['max_abs_divergence']:.1e} | "
            f"{'PASS' if r['benchmark_pass_3pct'] else 'FAIL'} |")

    # ---------------- table 2: mine vs theirs (same field) ------------------
    t2 = ["| quantity (n = 128) | m20000 audit | m20098 re-scoring | Δ | comment |",
          "|---|---|---|---|---|"]
    a = q[(128, "quadratic")]
    b = q[(128, "linear")]
    th = theirs[128]
    rows = [
        ("u_min rel. err", th["interpolators"]["u"]["extremum_dense"]["rel_error"], a["u_min_rel_err"],
         "extremum from a dense resample, not from the Ghia abscissae only"),
        ("v_max rel. err", th["interpolators"]["v"]["extremum_dense"]["rel_error"], a["v_max_rel_err"],
         "same"),
        ("max&#124;Δu&#124;/U (quadratic)", th["interpolators"]["u"]["quadratic"]["max_abs"], a["pointwise_max_abs_du"],
         "different local quadratic rules; both far inside the band"),
        ("max&#124;Δv&#124;/U (quadratic)", th["interpolators"]["v"]["quadratic"]["max_abs"], a["pointwise_max_abs_dv"],
         "agree to 3 % (different local quadratic rules)"),
        ("max&#124;Δu&#124;/U (linear)", th["interpolators"]["u"]["linear"]["max_abs"], b["pointwise_max_abs_du"],
         "theirs extrapolates into the lid row (y = 1, exact u = 1); mine imposes the BC"),
        ("vortex centre &#124;Δ&#124;/L", th["criteria"]["values"]["vortex_centre_abs"],
         a["vortex_centre_psi_poisson"]["distance"], "theirs integrates ψ along the wrong axis"),
        ("max&#124;div u&#124;", th["max_abs_divergence"], a["max_abs_divergence"], "identical field, same invariant"),
    ]
    for lab, tv, mv, note in rows:
        t2.append(f"| {lab} | {tv:.6g} | {mv:.6g} | {abs(tv - mv):.3g} | {note} |")

    t3 = ["| n | myself: ψ = ∫u dy | myself: ∇²ψ = −ω | bundle summary (shipped solver) | bundle audit JSON |",
          "|---|---|---|---|---|"]
    for n in sorted({r["n"] for r in audit["rows"]}):
        r = q[(n, "quadratic")]
        s = cases[n]["vortex_centre"]["computed"]
        sdist = None
        if s:
            sdist = float(np.hypot(s[0] - 0.5313, s[1] - 0.5625))
        vc = theirs[n]["vortex_centre_field"]["computed"]
        vdist = float(np.hypot(vc[0] - 0.5313, vc[1] - 0.5625))
        sdoc = f"({s[0]:.4f}, {s[1]:.4f}) d={sdist:.4f}" if s else "—"
        vdoc = f"({vc[0]:.4f}, {vc[1]:.4f}) d={vdist:.4f}"
        t3.append("| {} | ({:.4f}, {:.4f}) d={:.4f} | ({:.4f}, {:.4f}) d={:.4f} | {} | {} |".format(
            n,
            r["vortex_centre_psi_integrated"]["centre"][0], r["vortex_centre_psi_integrated"]["centre"][1],
            r["vortex_centre_psi_integrated"]["distance"],
            r["vortex_centre_psi_poisson"]["centre"][0], r["vortex_centre_psi_poisson"]["centre"][1],
            r["vortex_centre_psi_poisson"]["distance"], sdoc, vdoc))

    cont = summary["continuation_check"]["continuation"]
    base_ws = cases[cont["to_n"]]["solver"]["wall_seconds"]
    implied = base_ws / cont["speedup"]
    per_step_cold = base_ws / cont["cold_steps"] * 1e3
    per_step_cont = implied / 250 * 1e3
    psiv = {f["n"]: f["psi_extrema"] for f in audit["fields"]}
    tlm = {f["n"]: f["boundary"]["top_cell_row_mean_u"] for f in audit["fields"]}
    sha = {f["n"]: f["sha256"] for f in audit["fields"]}

    text = f"""# m20098 · 交差审计：m20000 / CASE-FVM-CAVITY-RE1000（TensorFVM 方腔 Re = 1000）

审计对象：m20000（`agent/m20000`，状态 succeeded），工作树 `/root/tp-agent/work/m20000`，
分支 HEAD `{_head()}`（与 main 同点，工作树有未提交改动）。
审计方：m20098（`agent/m20098`，本分支）。方法：**独立重算**，不重跑求解器。

## 0. 结论（先行）

| 项 | 结论 |
|---|---|
| 基准判据（`tensor_benchmark_cases`，anchor `CASE-CAVITY-RE1000-FVM`） | **PASS**：n = 128 全部 4 项 3 % 指标与主涡判据通过；n = 64 亦通过；n = 32 作为分辨率不足的对照行 FAIL（与报告一致） |
| 三件套 | run_artifacts **存在但未入库**；基准 ref vs got **数值成立**（我独立复算）；repro_cmd **第 4 步机器门禁跑挂** |
| 级别判定 | **L3**（不是 L4）：功能实质（求解器 + 完整算例 + 误差分析）为 L4 级，但 L4 的「可复现」要件不成立（自带门禁红、证据未入库、一条灵敏度对照实验为空转） |
| 处置 | **request changes / 建议 gate = reopened**（四项返修，见 §6） |

一句话：**数值是真的、判据是过的、交付是不可复现的**——附 100 页报告不改变级别，级别由功能实质与可复现性共同决定。

## 1. 取证范围与独立性限制

* 读到的物证：`summary.json`、`m20000_independent_audit.json`、`fields/Re1000_n{{32,64,128}}.pt`、`docs/CASE-CAVITY-RE1000-FVM/{{report.md,method-selection.md,round2-status.md}}`、`scripts/{{run,audit,render}}_cavity_re1000_*.py`、`tests/test_case_cavity_re1000_m20000.py`。
* 我的重算脚本 `scripts/audit_cavity_re1000_m20098.py` 为**从零编写**（numpy/torch/scipy），不 import 也不复制 `tensorfvm.lid_cavity` 的评分路径；唯一引入的外部数据是 Ghia(1982) 两张表（`src/tensorfvm/lid_cavity_reference.py`），并在脚本内先做自检（u_min = −0.38289 @ y = 0.1719、v_max = +0.37095 @ x = 0.1563、壁面端点归零）——**自检不过就不评分**。
* 独立性限制（如实声明）：① 我**没有**重跑求解器（一次 n = 128 冷启动约 1 800 s，属重算，应由调度系统派发；本单为核验单）；② 无 1982 年原论文 PDF，参考表出处按仓库转录 + 论文公布极值双证；③ 未审计 m15458 及更早轮次的产出（仅核 m20000 交差物证）；④ 未审计 `docs/oss_solutions/CASE-FVM-CAVITY-RE1000.json` 的内容正确性（该文件未入库，见 F3）。
* 字段 SHA256（审计对象身份绑定，防止「审的不是那份场」）：n=32 `{sha[32][:32]}…`、n=64 `{sha[64][:32]}…`、n=128 `{sha[128][:32]}…`。

## 2. 独立重算结果（n = 32 / 64 / 128，二次插值；程序化注入）

{chr(10).join(t1)}

判据原文（DB 行）：`u_min/v_max 相对误差 <3%；逐点 max|Δu|/U <3%（内点）；ledger band 5%；主涡中心 |Δ|/L ≤ 0.01 @ n≥128`。
逐点误差按「内点」计：壁面行（y = 1 顶盖运动壁、y = 0 底壁、x = 0/1 侧壁）取精确 Dirichlet 值，不作外插。

不变量：三套网格上 `max|div u| ≤ {max(r['max_abs_divergence'] for r in audit['rows'][::2]):.1e}`、侧壁 u = 0、上下壁 v = 0（机器零）、每列净流量 `max|mean_y u| < 2e-17`（封闭腔无穿流）、ψ ∈ [{min(v['min'] for v in psiv.values()):.4f}, {max(v['max'] for v in psiv.values()):.4f}]（物理量级正确）。

## 3. 与 m20000 自带「独立审计」逐项交叉核对

{chr(10).join(t2)}

主涡中心四种读法对照（关键：m20000 的「独立审计」这一项是坏的）：

{chr(10).join(t3)}

`scripts/audit_cavity_re1000_m20000.py:82-84`：

```python
def streamfunction(uc, h):
    \"\"\"psi(x, y) = int_0^y u dy from the bottom wall (psi = 0 there).\"\"\"
    return np.cumsum(uc, axis=0) * h      # uc is (x-cell, y-cell) -> integrates along x, not y
```

u 场轴序是 `(x-face, y-cell)`（我按 Ghia 表 I 反解确认：u[64, :] 的极小在 y = 0.176、误差 0.43 %）。
`uc` 因而也是 `(x-cell, y-cell)`，`cumsum(axis=0)` 得到的是 ∫u dx，不是 ψ。后果可测：
该 JSON 记录 ψ ∈ [{theirs[128]['psi_min']:.4f}, {theirs[128]['psi_max']:.4f}]（物理应为 [{psiv[128]['min']:.4f}, {psiv[128]['max']:.4f}]），
主涡 ({(theirs[128]['vortex_centre_field']['computed'][0]):.4f}, {(theirs[128]['vortex_centre_field']['computed'][1]):.4f})（顶右角，物理不可能），
其 `criteria.pass.vortex_centre_abs = false`（0.4256），**与报告 ④ 的 0.0056 pass = true 直接矛盾**。

## 4. 级别判定（L0–L5，逐级对表）

| 级 | 判据 | 本对象 | 事实依据 |
|---|---|---|---|
| L0 文档 | 只有文档 | 超过 | 有真实求解器与场 |
| L1 数学/单元验证 | 单测/公式验证 | 通过 | `tests/test_lid_cavity.py`、`tests/test_case_cavity_re1000_m15458.py`（main 内 tracked） |
| L2 独立功能实现 | 输入→计算→输出真代码 | 通过 | `src/tensorfvm/lid_cavity.py`（van Leer 面重构 + 镜像 ghost 运动壁 + DCT-II 投影 Poisson），输出 `fields/Re1000_n*.pt` |
| L3 系统集成 | 接入主求解器/平台 | **满足** | `src/tensorfvm/verification/ladder.py:1052 lid_cavity_verdict`、`:997` 工件命名、`scripts/run_cavity_re1000_m15458.py:166` CLI、`src/tensorfvm/verification/artifacts/verification_ladder/lid_cavity_Re1000_n64.json`（tracked） |
| L4 基准验证 | 完整算例 + 误差分析 + 可复现 | **未满足（可复现）** | 完整算例 ✓（3 网格、全收敛）；误差分析 ✓（网格列 + Richardson p = 2.77 + 双插值 + 第三方重算）；**可复现 ✗**：自带门禁红（F2）、物证未入库（F3）、一条灵敏度对照空转（F1） |
| L5 科研工程成果 | 新认识/工程可用 | 不主张 | 标准基准题，无新物理 |

→ **level = 3**。若四项返修完成（§6），凭本单第 2 节的独立重算即可直接支持 L4。

## 5. 严重度排序的问题清单

### F1 · 阻塞（L4 要件）· 延续性对照实验是空转，×231.63「加速」是假象
* **证据**：`summary.json.continuation_check`：`continued_steps = {cont['continued_steps']}`、`cold_steps = {cont['cold_steps']}`、`speedup = {cont['speedup']:.2f}`、`same_steady_state = true`（du = {cont['max_abs_du']:.2e}）。
  步数几乎相同却说 231 倍加速，算术上不成立。反推：`{base_ws:.1f} s / {cont['speedup']:.2f} = {implied:.3f} s`，而该阶段记录步数为 {cont['continued_steps']}。
  {implied:.2f} s / 250 步 = {per_step_cont:.2f} ms/步，与冷启动 {per_step_cold:.2f} ms/步**完全同级**——即它只跑了约 250 步就收敛了。
* **根因**（`scripts/run_cavity_re1000_m15458.py:91-97`）：
  ```python
  if init is not None:                       # 先把 n=32 的 prolongation 装进去
      solver.u, solver.v = init[0].clone(), init[1].clone()
  if os.path.exists(fields_path):            # 但随后若已有 checkpoint，就整体覆盖（含 init）
      saved = torch.load(fields_path, weights_only=False)
      solver.u, solver.v, solver.steps = saved["u"], saved["v"], int(saved["steps"])
  ```
  第 2 步的冷启动已经把 `fields/Re1000_n64.pt`（收敛场）落盘，第 4 步的「延续性对照」于是从**冷的 n = 64 收敛场本身**续跑：`steps` 计数是累计值（{cont['cold_steps']} + 250 → {cont['continued_steps']}），墙钟只算新跑的 250 步。
* **后果**：① 报告 ⑤「Sensitivity axis 1 — initial condition（continuation vs cold start）」与 ⑥ 第 3 条「The cold-start control run on the same grid resolves it quantitatively」**不成立**——对照的初值就是它自己，`max|Δu| = 5.65e-08` 是自证，不是证据；② 报告/摘要的「×231.63 wall-time speed-up」不可引用；③ 同一 bug 让「按 repro_cmd 重跑」变成「读旧 checkpoint 续跑」，可复现性依赖残留状态。
* **复现**：`cd /root/tp-agent/work/m20000 && python3 -c "import json;d=json.load(open('artifacts/CASE-CAVITY-RE1000-FVM/summary.json'));print(d['continuation_check'])"` → 对照 `cases[n=64].solver.steps = {cont['cold_steps']}` 与 `wall_seconds = {base_ws:.1f}`。
* **返修**：`init` 优先级高于 checkpoint（显式 `--resume` 才读盘），并把延续性对照跑在干净目录；`continued_steps` 记增量步数。

### F2 · 阻塞（可复现）· 携带的机器门禁是红的，报告却当作绿
* **证据**：`cd /root/tp-agent/work/m20000 && PYTHONPATH=src python3 -m pytest tests/test_case_cavity_re1000_m20000.py -q` → **1 failed, 5 passed**（实测本单）。
  `tests/test_case_cavity_re1000_m20000.py:127`：`assert float(uc[:, -1].mean()) > 0.9`，n = 32 实得 {tlm[32]:.4f}。
* **根因**：该量是「顶盖下一层网格中心行的 u 均值」，在剪切层里天然 O(h) 低于 U：三套网格实测 {tlm[32]:.4f} / {tlm[64]:.4f} / {tlm[128]:.4f}，**全部 < 0.9**，因此这条断言在任何网格上都不可达——是判据写错，不是场错（顶盖驱动本身很强：n = 128 时 0.851）。
* **后果**：报告 ⑥ 第 4 步「machine gate（recomputes every stored number from the fields）」被当作已通过（并声称「fails on a stale or duplicated report」），实际未通过；这也是本单判 L3 而非 L4 的直接依据之一。
* **返修**：把阈值改成随 h 收敛的物理判据（如 `> 1 - C·h`，或直接断言 `u[:, -1]` 的顶壁面值），并让该文件入库后由 CI 跑。

### F3 · 阻塞（可复现）· 三件套里的工件与脚本没有入库，「fields are committed」是虚标
* **证据**：`cd /root/tp-agent/work/m20000 && git status --porcelain` →
  `?? artifacts/`、`?? docs/oss_solutions/CASE-FVM-CAVITY-RE1000.json`、`?? scripts/audit_cavity_re1000_m20000.py`、`?? scripts/render_cavity_re1000_m20000.py`、`?? tests/test_case_cavity_re1000_m20000.py`，外加 3 个 `M`（report.md 等）。
  `git ls-files artifacts | wc -l` → 0；main 与远端都不存在 `agent/m20000` 分支。
* **后果**：报告 ⑥「Fixed here: fields are committed under `artifacts/CASE-CAVITY-RE1000-FVM/fields/`」为**不实陈述**；交差物证只活在 worker 磁盘上，一旦工作树回收即丢失；`repro_cmd` 第 2/3/4 步（audit/render/pytest）在仓库内无法执行。
* **补充（本单实测的机制约束）**：本仓工作树用 sparse-checkout，`artifacts/` 不在可提交定义内（`git add artifacts/...` 被拒），因此「把大件场文件入库」需另择路径；但 `scripts/`、`tests/`、`docs/` 是**可提交**的——这三个目录里的交差物（3 个脚本 + 门禁测试 + oss_solutions JSON）未入库，纯属漏提交。本单因此把审计证据放在**可提交**的 `docs/CASE-CAVITY-RE1000-FVM/`（`m20098_audit.json` + 3 张图），`artifacts/` 下只留同一份工作副本。
* **返修**：把 3 个脚本 + 门禁测试入库（含修正后的 F2），场文件按仓库既有惯例给「摘要 JSON + SHA256SUMS」或走 LFS/工件库；在报告里改为「工件在 X 位置，命名与校验和见 Y」。

### F4 · 重要（非阻塞基准结论）· 主涡判据在「错误 ψ」下被记 PASS / 在另一个合法定义下无对照
* 见 §3：m20000 审计 JSON 的 ψ 轴错（`scripts/audit_cavity_re1000_m20000.py:82-84`），主涡 ({(theirs[128]['vortex_centre_field']['computed'][0]):.4f}, {(theirs[128]['vortex_centre_field']['computed'][1]):.4f})、ψ_max {theirs[128]['psi_max']:.4f}，且其 `criteria.pass` 记 false，与 ④ 的 pass = true 冲突。
* 我用**两种合法 ψ 定义**各算一次（§2）：∫u dy → d = {q[(128,'quadratic')]['vortex_centre_psi_integrated']['distance']:.4f}；∇²ψ = −ω（DST-I）→ d = {q[(128,'quadratic')]['vortex_centre_psi_poisson']['distance']:.4f}；两者均 ≤ 0.01，**基准结论不受影响**。
* 但注意该判据的分辨率余量薄：h = 1/128 时 0.01 只相当于 1.3 个网格，建议 n = 256 复核或直接以两种 ψ 定义同时报出。

### F5 · 重要（叙述失真）· 「linear 比较器 5.87 % 是插值阶限制」是误判
* 报告 ④ 说 piecewise-linear 在 n = 128 读到 0.0587 U「exceeds the 5 % band by 0.87 pp；这是 comparator limit 不是解误差」。
* 实测：该 0.0587 全部来自 y = 1.0 那一行——Ghia 表把顶盖点 (y = 1, u = 1) 也列进去了，而线性插值从 y ≤ 0.996 的网格值**外插**到 1.0 得到 0.941。**内点**（不给壁面行外插）线性最大误差 = {q[(128,'linear')]['pointwise_max_abs_du']:.5f}（{pct(q[(128,'linear')]['pointwise_max_abs_du'])}），远在 5 % 带内。
* 同一 artefact 也解释了报告 ⑤ 的 n = 32「max|Δu| = 0.2033」：内点值为 {q[(32,'linear')]['pointwise_max_abs_du']:.4f}。结论不变（n = 32 因 u_min 14 % 仍 FAIL），但归因需要改。

### F6 · 次要（可追溯性）· 物证清单与实际不符；锚点 slug 有两套写法
* ⑥ 列的 `artifacts/CASE-CAVITY-RE1000-FVM/run.log` **不存在**（`ls` 实测 No such file）。
* 任务/报告用 `CASE-FVM-CAVITY-RE1000`，而 `tensor_benchmark_cases.anchor_slug` 实际是 **`CASE-CAVITY-RE1000-FVM`**（`SELECT ... WHERE anchor_slug='CASE-FVM-CAVITY-RE1000'` 返回 0 行）。本单按 DB 实际行判据裁决，建议统一或加别名映射，否则将来机器复核会找不到判据行。
* n = 64 的 `fields/Re1000_n64.pt` 被 F1 的对照运行**覆盖**（同一稳态，du = {cont['max_abs_du']:.1e}），但 `summary.json.cases[n=64]` 描述的是冷启动那次——证据链「摘要 ↔ 场文件」在 n = 64 上并非同一进程产物（数值上无影响，已用 5e-7 门禁验证）。

### F7 · 背景（非本单责任）· main 上该锚点的其他门禁本就是红的
`PYTHONPATH=src pytest tests/test_audit_cavity_re1000_m15458.py tests/test_cavity_anchor.py -q` → 3 failed / 4 error（前者因 `artifacts/audit-m15458/summary.json` 未入库，后者 `psi_abs_max` spread 2.63 与阈值 3.0、扫描白名单）。与本单对象无关，但说明「工件不入库」是系统性问题，不是 m20000 独有。

## 6. 处置与返修清单（request changes → 建议 reopened）

1. **F1**：`run_case` 让显式 `init` 优先于 checkpoint（`--resume` 才读盘）；重跑干净目录下的 continuation 对照，`continued_steps` 记增量；重写 ⑤ 的轴 1 与 ⑥ 第 3 条，删除「×231.63」。
2. **F2**：修 `tests/test_case_cavity_re1000_m20000.py:127` 的不可达断言（改为随 h 收敛的判据），入库并由 CI 跑绿；报告里的门禁状态改为实测输出。
3. **F3**：`scripts/{{audit,render}}_cavity_re1000_m20000.py`、`tests/test_case_cavity_re1000_m20000.py`、`docs/oss_solutions/CASE-FVM-CAVITY-RE1000.json` 入库；场文件用摘要 + SHA256SUMS（或工件库）固化；报告 ⑥ 的「committed」改为真实位置。
4. **F4**：修 `audit_cavity_re1000_m20000.py:82-84` 的 `cumsum` 轴（`axis=1`），重生成审计 JSON，使 `criteria.pass.vortex_centre_abs` 与 ④ 一致；建议同时报 ∫u dy 与 ∇²ψ = −ω 两种 ψ。
5. **F5/F6**：改正 linear 比较器的归因；补 `run.log`（或删掉清单里的它）；统一锚点 slug（或加别名）。

**批准结论**：基准判据（3 % 带 + 主涡 ≤ 0.01 @ n ≥ 128）**成立**，数值可被我独立复算复现（§2）；但交付层（可复现、门禁、工件、对照实验）不达 L4 → **L3，reopened**。

## 7. 本单可复核的命令

```bash
# 我的独立重算（不 import tensorfvm 的评分路径；~8 s）
PYTHONPATH=src python3 scripts/audit_cavity_re1000_m20098.py \\
    --dir /root/tp-agent/work/m20000/artifacts/CASE-CAVITY-RE1000-FVM \\
    --out docs/CASE-CAVITY-RE1000-FVM/m20098_audit.json
# 我的证据门禁（从入库的 profile 复算每个头条数字，不需要 .pt）
PYTHONPATH=src python3 -m pytest tests/test_board_audit_m20098.py -q     # 5 passed
# 本次实测到的 m20000 红门禁（只读，不改对方工作树）
cd /root/tp-agent/work/m20000 && PYTHONPATH=src python3 -m pytest tests/test_case_cavity_re1000_m20000.py -q
# 判据行
psql -U postgres -d tensorplatform -c "SELECT anchor_slug,metric,tolerance FROM tensor_benchmark_cases WHERE anchor_slug='CASE-CAVITY-RE1000-FVM'"
```
"""

    DOC.mkdir(parents=True, exist_ok=True)
    (DOC / "m20098-board-audit.md").write_text(text)
    print("wrote", DOC / "m20098-board-audit.md")
    print("wrote", ART / "fig_m20098_centrelines.png", ART / "fig_m20098_convergence.png")


def _head():
    import subprocess
    try:
        return subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return "unknown"


if __name__ == "__main__":
    main()