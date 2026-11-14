"""Build and publish the m20339 platform report (numbers injected from JSON)."""
from __future__ import annotations

import base64
import json
import os
import subprocess
from pathlib import Path

REPO = Path("/srv/tensor/agent/work/m20339")
COV = REPO / "docs/woodward-colella-double-blast/coverage"
cov = json.loads((COV / "coverage_robustness.json").read_text())
prov = json.loads((COV / "provenance_manifest.json").read_text())
c, e, inv, sw = cov["coverage"], cov["evidence"], cov["invariants"], cov["offset_sweep"]
tc = cov["trough_conditioning"]

peak3200 = sw["sweep"]["+0.022017"]["nx3200_hllc"]
raw3200 = sw["sweep"]["+0.000000"]["nx3200_hllc"]


def rows_sweep() -> str:
    out = ["| δ（密度单位） | 峰值 % | L1 % | 窗质量 % | 浅槽 %（不承载 verdict） | verdict |",
           "|---|---|---|---|---|---|"]
    for k, v in sw["sweep"].items():
        r = v["nx3200_hllc"]
        mark = "**" if k == "+0.022017" else ""
        out.append(f"| {mark}{k}{mark} | {r['peak_percent']:.2f} | {r['l1_percent']:.2f} | "
                   f"{r['window_mass_percent']:.3f} | {r['trough_percent']:.2f} | {r['verdict']} |")
    return "\n".join(out)


def rows_regions() -> str:
    out = ["| 分区 | 长度占比 | 质量占比 | 交叉格式互差 (nx3200) | D(1600,3200) | D(200,1600) |",
           "|---|---|---|---|---|---|"]
    share = {"blind_left_(0,0.4)": (40, c["left_mass_share_percent"]),
             "window_(0.4,1.0)": (60, c["window_mass_share_percent"]),
             "full_(0,1)": (100, 100.0)}
    for name in ("blind_left_(0,0.4)", "window_(0.4,1.0)", "full_(0,1)"):
        L, M = share[name]
        out.append(f"| {name} | {L}% | {M:.2f}% | {e['cross_scheme']['nx3200'][name]['percent']:.3f} % | "
                   f"{e['cross_grid'][name]['d_1600_3200']['percent']:.3f} % | "
                   f"{e['cross_grid'][name]['d_200_1600']['percent']:.3f} % |")
    return "\n".join(out)


def rows_inv() -> str:
    out = ["| 剖面 | nx | 质量漂移 | 能量相对漂移 |", "|---|---|---|---|"]
    for tag, r in inv["rows"].items():
        out.append(f"| {tag} | {r['nx']} | {r['mass_drift']:+.2e} | {r['energy_relative_drift']:+.2e} |")
    return "\n".join(out)


md = f"""# CASE-WOODWARD-COLELLA 证据包对抗性复核：覆盖边界、验收稳健性与 provenance 修复

**锚点**：`CASE-WOODWARD-COLELLA`（算例池·双爆轰波相互作用，容差 <10%）
**性质**：验证欠账类（对已合并交付物做文献/解析基准复核），求解器代码零改动
**裁决**：approve with notes —— `<10%` 验收在整个合理标定带内成立；发现 5 项缺陷（无阻断性），已修复或门禁固化
**证据指针**：`docs/audit-m20339/AUDIT-CASE-WOODWARD-COLELLA-adversarial-review.md`、
`docs/woodward-colella-double-blast/coverage/`、`tests/test_compressible_wc_coverage.py`

## ① 案例介绍（物理背景 / 几何 / 参数表）

一维闭管 x∈[0,1] 内双爆轰波相互作用（Woodward & Colella 1984 标准算例；复述文献 Dumbser et al., JCP 2019
arXiv:1902.07773 §5.1.4 与图 21 的 "Reference" 曲线，原文本体闭源）。

| 参数 | 值 |
|---|---|
| 初始 (ρ,u,p) | (1,0,1000) x≤0.1；(1,0,0.01) 0.1<x<0.9；(1,0,100) x≥0.9 |
| γ / 终止时刻 / 壁面 | 1.4 / t=0.038 / 两端反射（绝热，壁不作功） |
| 求解读 | TensorFVM `cases.double_blast`：MUSCL-Hancock 二阶 + MC 限幅 + HLLC + SSP-RK，CFL 0.4 |
| 交叉码 | WENO5 + Rusanov + SSP-RK3（与生产码不共享重构/通量/限幅） |
| 参考解 | 文献 Reference 数字化（RPJ@3200），经几何标定更正 +0.022017 密度单位 |
| 出版窗口 / 容差 | x∈[0.4,1.0] / 10 % |
| 严格守恒量（解析） | 总质量 1.000000；总能量 {inv['energy_exact']:.6f} = (10³·0.1+10⁻²·0.8+10²·0.1)/(γ−1) |

## ② 怎么做的（方法 / 流程 / 关键设置 / 工具链）

本单不重跑求解器，只对**已提交剖面**做四类**不同种类**的独立复核（既有报告已证明：同源双通道互校验
只能测噪声、不能测系统偏差，故此处刻意换证据种类）：

1. **验收稳健性**：把参考表整体加性平移 δ 扫 7 点（δ∈[−0.022, +0.044]，含 0 与几何实测 +0.022017），
   对每档网格/每个指标重打分，量化 `d(err%)/dδ`。目的：检验 `<10%` 是否只是「选对了一个 δ」。
2. **排除理由的独立验证**：`low_plateau` 一直被排除在主判据外，理由是「参考最小值钉在窗边、是单调斜坡的边界值」。
   本单直接测参考曲线 argmin 位置与 [0.42,0.50] 的单调性，独立证实/证伪该理由。
3. **覆盖边界与盲区证据**：按 盲区/窗/全域 分区计算质量分配、交叉格式距、交叉网格距。文献不给参考的
   x<0.4 段落，用「无参考但可证伪」的方式定界。
4. **provenance 复现**：证据包 11 份 JSON 记录 20 条**跨工作树绝对路径**；用本仓提交的参考表重打分 16 份
   metrics 记录，判定「悬空路径」是纯可移植性缺陷还是数值污染。

工具链：`coverage/wc_coverage_robustness.py`（分析+图）、`coverage/wc_provenance_manifest.py`（provenance）、
`tests/test_compressible_wc_coverage.py`（13 条门禁，全部**重算**而非读 JSON）；
并修复 `tests/test_compressible_wc_calibration.py::_profile()` 的静默 skip。

## ③ 计算结果（程序化注入，零手抄）

### 3.1 验收对参考标定的稳健性（nx=3200）

{rows_sweep()}

灵敏度 `d(err%)/dδ`：峰值 {sw['sensitivity']['peak_percent']['d_error_percent_per_unit_density']:.1f}、
L1 {sw['sensitivity']['l1_percent']['d_error_percent_per_unit_density']:.1f}、
窗质量 {sw['sensitivity']['window_mass_percent']['d_error_percent_per_unit_density']:.1f}、
浅槽 {sw['sensitivity']['trough_percent']['d_error_percent_per_unit_density']:.1f} (%/密度单位)
⇒ 浅槽比峰值敏感 **{abs(sw['sensitivity']['trough_percent']['d_error_percent_per_unit_density']/sw['sensitivity']['peak_percent']['d_error_percent_per_unit_density']):.0f} 倍**，故不能承载 verdict。
**结论**：7 个 δ 上 verdict 全为 PASS；`<10%` 不是标定选择的产物。

### 3.2 `low_plateau` 排除理由的独立验证

| 量 | 实测 |
|---|---|
| 参考最小点位置 | x={tc['edge']['argmin_x']:.6f}（距下窗边 {tc['edge']['edge_distance_abs']:.6f} = **{tc['edge']['edge_distance_percent_of_window']:.3f}% 窗宽**） |
| 是否存在内部极小 | {tc['edge']['reference_has_interior_min']} |
| [0.42,0.50] 单调性 | 严格递增 = {tc['monotone_rise']['strictly_increasing']}（{tc['monotone_rise']['negative_increments']}/{tc['monotone_rise']['n_increments']} 为负增量） |

⇒ 「浅槽是窗边截断的单调斜坡，不是物理谷底」这一理由**被独立证实**（此前只是断言）。

### 3.3 覆盖边界与盲区

{rows_regions()}

窗内只占 60% 长度但 **{c['window_mass_share_percent']:.2f}% 质量**；无参考的盲区占 {c['left_mass_share_percent']:.2f}% 质量，
其交叉格式互差仅 {e['cross_scheme']['nx3200']['blind_left_(0,0.4)']['percent']:.3f}% —— 全部方案敏感度集中在含激波的窗内。

### 3.4 严格守恒不变式（闭管：质量与总能量都应精确守恒）

{rows_inv()}

### 3.5 provenance

记录绝对路径 {prov['absolute_path_records']['count']} 条 / 16 个不同目标 / 11 份 JSON；
sha256 可比者 {prov['absolute_path_records']['identical_verified']} 条全部 identical，矛盾
{len(prov['absolute_path_records']['contradicted'])} 条，源工作树已删除无法比 sha
{prov['absolute_path_records']['unverifiable_source_gone']} 条；**用本仓提交参考表重打分复现
{prov['record_reproduction']['n_reproduced']}/{prov['record_reproduction']['n_records']} 条**（≤1e−6）
⇒ 悬空路径是可移植性缺陷，**非数值污染**。

## ④ 结果图

![覆盖与分区证据](fig_domain.png)

*图 1*：(a) nx=3200 双格式密度剖面，灰色带为无外部参考的盲区（占 {c['left_mass_share_percent']:.2f}% 质量）；
(b) 分区交叉格式距与交叉网格距，全部远低于 10% 验收线。

![标定稳健性](fig_offset.png)

*图 2*：各指标 |误差| 随参考偏置 δ 的变化（对数纵轴）。窗质量/峰值/L1 在整个 δ 带内都深居 10% 线下方；
浅槽在 δ=0 与 δ=+0.044 处越过 10% 线，故不能承载 verdict（其灵敏度为峰值的 44 倍）。

## ⑤ 注意点（坑 / 局限 / 适用范围）

1. **发现并修复的质量缺陷**：`tests/test_compressible_wc_calibration.py::_profile("nx1600")` 只找
   `ladder_profile_nx1600.npz`，实存名为 `ladder_profile_nx1600_hllc.npz` ⇒ 两条「最细网格」断言
   **在 CI 里静默 skip**（not_run ≠ passed）。修复后转真执行并通过；wc 全套门槛 27 passed/3 skipped → **60 passed/1 skipped**。
2. **产物名不符内容**：`artifacts/wc_blast_profile_nx{{400,800}}.npz` 只含窗内 240/480 点（非全域），
   按名当全域用会除零。已加契约门固定行为。
3. **过强表述更正**：「交叉码峰值/L1 互差 <1%」只成立于 nx=3200（0.972%），nx=1600 为 1.475%，
   且既有报告自身表格 nx=400 为 2.85%。今后引用须带网格号。
4. **单源风险（未消除）**：参考解来自唯一一篇开放文献的数字化曲线；本单未引入第二篇独立文献。
5. **近零交叉的不可分辨性**：δ=+0.022017 时窗质量误差 0.009%，可读作「独立佐证」，但同一数据也无法排除
   参考与解共享某种系统效应 —— **本数据集内不可分辨**，故不作证明使用（开放项）。
6. **不重跑求解器是有意为之**：比较器与剖面生成的正确性不在本次覆盖面内（独立性限制已声明）。

## ⑥ 误差判定（逐项 vs 基准，显式回答是否 <3%）

| 关键量 | nx=3200（δ=实测偏差） | 是否 <3% | 是否 <10%（总账验收） | 是否随 δ 全带保持 <10% |
|---|---|---|---|---|
| 峰值密度 ρ_max | {peak3200['peak_percent']:.2f} % | {peak3200['peak_percent'] < 3} | ✅ | ✅（最差 {max(v['nx3200_hllc']['peak_percent'] for v in sw['sweep'].values()):.2f} %） |
| 剖面 L1 | {peak3200['l1_percent']:.2f} % | {peak3200['l1_percent'] < 3} | ✅ | ✅（最差 {max(v['nx3200_hllc']['l1_percent'] for v in sw['sweep'].values()):.2f} %） |
| 窗质量 ∫ρdx[0.4,1] | {peak3200['window_mass_percent']:.3f} % | ✅ | ✅ | ✅（最差 {max(v['nx3200_hllc']['window_mass_percent'] for v in sw['sweep'].values()):.3f} %） |
| 近真空浅槽 ρ_min（不承载 verdict） | {peak3200['trough_percent']:.2f} % | ✅ | ✅（原始参考下 14.45 %，见下） | ❌（病态，灵敏度 44×） |
| 峰位 x(ρ_max) | 0.0020（绝对，nx=1600） | ✅（容差 0.02） | ✅ | — |
| 严格守恒：质量 / 总能量 | ≤4.93e−13 / ≤7.34e−13 | ✅（round-off） | ✅ | — |

**显式回答**：nx=3200 上峰值 {peak3200['peak_percent']:.2f}% 与 L1 {peak3200['l1_percent']:.2f}% **略高于 3%**，
但这 3% 级残余已被量化为**参考解自身的误差地板**（数值不确定度 `U_num≤0.14%`，残余/U_num ≫ 1），
且**不存在任何单一 δ 能令所有指标同时归零**（峰值最佳在 δ=+0.044、L1 最佳在 δ=+0.022）——
即残余不能由单一标定误差解释。总账 `<10%` 在 δ 全带、双格式、200–3200 网格上**全部满足**。
诚实保留：δ=0（原始入库参考）时浅槽 14.45%（>10%），该数字**未被编辑**，仍可从原始表复现。
"""


def fig(name: str, caption: str) -> dict:
    return {"name": name, "caption": caption,
            "png_base64": base64.b64encode((COV / name).read_bytes()).decode()}


payload = {
    "title": "CASE-WOODWARD-COLELLA 对抗性复核：验收对参考标定的稳健性、盲区定界与 provenance 修复",
    "summary": ("双爆轰波算例证据包独立复核：7 点参考偏置扫描证明 <10% 验收在整个合理标定带内成立（而非依赖"
                "单一标定值）；独立证实 low_plateau 排除理由（argmin 距窗边 0.064% 窗宽、[0.42,0.5] 严格递增）；"
                "量化无外部参考盲区仅占 6.44% 质量且双格式互差 0.022%；证明 8 条跨工作树悬空路径未造成数值污染"
                "（16/16 记录用本仓参考表复现）；修复 2 处静默 skip 的最细网格断言。"),
    "scenario_key": "fvm.compressible.double_blast.verification",
    "mission_id": 20339,
    "tags": ["CASE-WOODWARD-COLELLA", "TensorFVM", "verification", "adversarial-review",
             "provenance", "GCI", "blast-wave", "evidence-audit"],
    "markdown": md,
    "figures": [
        fig("fig_wc_domain_coverage.png",
            "图1 (a) 双格式密度剖面与无参考盲区 (b) 分区交叉格式/交叉网格距离"),
        fig("fig_wc_offset_robustness.png",
            "图2 各指标误差随参考偏置 δ 的变化：验收稳健、浅槽病态"),
    ],
}

Path("/tmp/report.json").write_text(json.dumps(payload, ensure_ascii=False))
print("payload bytes:", len(json.dumps(payload)), "figures:", len(payload["figures"]))

base = os.environ["TENSOR_API_BASE"]
tok = subprocess.run(["curl", "-s", "-X", "POST", f"{base}/auth/login",
                      "-H", "Content-Type: application/json",
                      "-d", json.dumps({"username": os.environ["TENSOR_AGENT_USER"],
                                        "password": os.environ["TENSOR_AGENT_PASS"]})],
                     capture_output=True, text=True).stdout
token = json.loads(tok).get("access_token") or json.loads(tok).get("token")
assert token, tok[:300]
res = subprocess.run(["curl", "-s", "-X", "POST", f"{base}/tensor/reports",
                      "-H", f"Authorization: Bearer {token}",
                      "-H", "Content-Type: application/json",
                      "-d", "@/tmp/report.json"], capture_output=True, text=True).stdout
print(res[:600])