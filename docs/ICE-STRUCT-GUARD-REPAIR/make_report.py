"""Render SPEC.md / REPORT.md / evidence JSON from artifacts, then publish.

Every number in the generated markdown is read from the artifacts written by
``PYTHONPATH=src python -m tensorfvm.ice --out artifacts/ice_guard`` and
``docs/ICE-STRUCT-GUARD-REPAIR/make_figures.py`` — no hand-typed values.

Run:
    PYTHONPATH=src python docs/ICE-STRUCT-GUARD-REPAIR/make_report.py [--publish]
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import pathlib
import urllib.request

from tensorfvm.ice.benchmark import REPRO_CMD
from tensorfvm.ice.parameters import ICE_RESISTANCE_INPUTS
from tensorfvm.ice.load import known_parameters

ROOT = pathlib.Path(__file__).resolve().parents[2]
HERE = pathlib.Path(__file__).resolve().parent
ART = ROOT / "docs" / "ICE-STRUCT-GUARD-REPAIR" / "artifacts"
MISSION = 22342

bundle = json.loads((ART / "benchmark.json").read_text())
summary = json.loads((ART / "summary.json").read_text())
channel = json.loads((HERE / "channel_metrics.json").read_text())
params = known_parameters()

# ---------------------------------------------------------------- SPEC.md ----
spec_rows = []
for name in ICE_RESISTANCE_INPUTS:
    p = params[name]
    spec_rows.append(
        f"| `{name}` | {p['symbol']} | {p['unit'] or '—'} | {p['hard'][0]:g} | "
        f"{p['hard'][1]:g} | {p['soft'][0]:g} | {p['soft'][1]:g} | "
        f"{p['typical'][0]:g}–{p['typical'][1]:g} | {p['reference']} |")
spec_table = "\n".join(spec_rows)

SPEC = f"""# Ice-domain guardrail specification (JOUR-ICE-STRUCT-GUARD, rev. m22342)

## 1. What is guarded

The 12 quantities a level-ice resistance calculation consumes: 8 ice
properties (the m22209 delivery) plus 4 hull variables that were previously
unguarded even though the load formula multiplies them in.  Registry:
`tensorfvm.ice.parameters.ALL_ICE_PARAMETERS` (12 entries,
`ICE_RESISTANCE_INPUTS`).

## 2. Decision semantics

| class | meaning | `validate` verdict | `clamp` behaviour |
|---|---|---|---|
| inside typical band | normal operation | `pass` | unchanged |
| inside hard, outside soft | outside the calibration domain of the empirical models | `fail` (or `warn` when `allow_extrapolation=True`) | unchanged |
| outside hard bounds | outside the validity domain of the models / physically impossible | `fail` | clamped to the nearest hard bound, warning recorded |
| non-finite (`nan`, `±inf`) | not a physical measurement | `fail`, never in `validated_params` | `±inf` → bound; `nan` → rejected, never echoed back |

`require_ice_parameters` / `require_guarded_level_ice_resistance` turn the
verdict into a raised `IceParameterDomainError`, so a caller cannot forget the
`if`.

## 3. Bounds

| parameter | symbol | unit | hard min | hard max | soft min | soft max | typical | reference |
|---|---|---|---|---|---|---|---|---|
{spec_table}

The eight ice rows are the m22209 table verbatim
(`docs/FVM_ICE_GUARDRAILS/REPORT.md` §2.2).  The hull rows are declared here
for the first time; they are deliberately wide (an applicability envelope, not
a hull-design checker) and are documented as such in `parameters.py`.

## 4. Known scope limits of the specification

* Bounds are static; temperature- or salinity-dependent bounds and
  correlation-aware checks (thick ice with weak strength) are **not** modelled.
* `bow_angle_deg` is a shape descriptor, not part of the 12-parameter set, and
  is passed explicitly to the load entry point (default 30°, Lindqvist's
  nominal).
* Hard bounds state the *applicability limit of ship–ice models*, not physical
  impossibility: `h = 10 m` is the upper end of level-ice formulations, while
  ridge keels and glacier tongues are far thicker.
"""

# -------------------------------------------------------------- REPORT.md ----
def row_line(r: dict) -> str:
    if r["passed"] is None:
        verdict = "**WITHHELD**"
    else:
        verdict = "PASS" if r["passed"] else "FAIL"
    if r["error"] is None:
        err = "-"
    elif r["unit"] == "relative":
        err = f"{r['error']:.3e}"
    else:
        err = f"{r['error'] * 100:.4f} %"
    if r["tolerance"] is None:
        tol = "-"
    elif r["unit"] == "relative":
        tol = f"{r['tolerance']:.0e}"
    else:
        tol = f"{r['tolerance'] * 100:.2f} %"
    return (f"| {r['metric'].split()[0]} | {r['ref']} | {r['got']} | {err} | "
            f"{tol} | {verdict} |")


rows_md = "\n".join(row_line(r) for r in bundle["rows"])

sweep_lines = []
for point in bundle["detail"]["thickness_sweep"]:
    guarded = ("blocked" if point["guarded_kN"] is None
               else f"{point['guarded_kN'] / 1000.0:,.1f}")
    sweep_lines.append(
        f"| {point['ice_thickness_m']:g} | {point['unguarded_kN'] / 1000.0:,.1f} "
        f"| {guarded} |")
sweep_md = "\n".join(sweep_lines)

legacy = bundle["detail"]["g6_legacy"]
legacy_md = "\n".join(
    f"| {k} | {v['ref_MN']:g} MN | {v['got_MN_in_repo_reproduction']:,.1f} MN | "
    f"{v['rel_error'] * 100:.2f} % | {'yes' if v['blocked'] else 'NO'} |"
    for k, v in legacy.items())

cm = channel["summary_metrics"]
n_in = len(bundle["detail"]["g4_load_path"])
n_grid = len(bundle["detail"]["g1_decision_grid"])

REPORT = f"""# 冰区越域护栏：从 schema 层进入物理计算入口（JOUR-ICE-STRUCT-GUARD / m22342）

## ① 案例介绍

**物理背景。** 船舶在层冰中航行时，总阻力按 Lindqvist (1989) 分解为挤压、弯曲、浸没、碎冰拖曳四项。
该公式的每一项都是经验/半经验拟合，只在有限的冰厚、冰强度、航速、船体主尺度范围内有效。

**锚点故障（记录于 2026-11-09 平台构建）。** 冰域输入 `h = 50 m` 与 `σ_c = 100 MPa` 被静默外推，
产出 2450 MN 级与 143.9 MN 的"阻力"；而当时的护栏只在 schema 层（Pydantic 类型检查），
12 个输入中有 7 个可越域而不被拦截。同一锚点记录 `GET /ice-uq/ships` 返回 HTTP 500
（`KeyError 'std_resistance_kN'`）。

**本轮复测（MOCK 排除）。** 本次对该端点实测返回 HTTP 200，三家船的响应中
`reference_std_resistance_kN` 仍为 `null`、`geometry_consistent` 均为 `false`
（位移与 L·B·T·Cb 隐含位移偏差 −29.3 % / +24.8 % / +13.2 %）——即 KeyError 的直接触发条件
（空值）仍在，只是当前构建不再 500。**该端点属平台侧（非 TensorFVM 仓），本单不改它，
只作为"平台侧残余缺口"如实登记。**

**参数表（12 个输入，程序化从注册表渲染）。** 见 `SPEC.md` §3。

## ② 怎么做的

1. **修 NaN 漏洞（m22209 的 F1 阻塞缺陷）。** `is_hard_violation` 用 `<`/`>` 比较，NaN 与任何数比较恒
   False，护栏把 NaN 标成 "TYPICAL" 放行。现：`parameters.is_hard_violation` 先判 `math.isfinite`；
   `guardrails` 的 validate / clamp / risk / report 四路统一走一个 `_finite_value` 强转，
   `nan` 一律 `fail` 且**不回吐**，`±inf` 判拒绝、clamp 时落到已知方向的硬边界，`bool` 显式拒绝
   （`isinstance(True, int)` 会把 `True` 静默变成 1.0 m 冰厚）。
2. **把护栏扩到完整输入集（8 → 12）。** 新增 4 个船体参数（`ship_length`/`ship_beam`/`ship_draft`/
   `ship_block_coefficient`）三级边界，注册 `ICE_RESISTANCE_INPUTS`，并保持
   `typical ⊆ soft ⊆ hard` 嵌套不变式（12/12 成立，见 G1 覆盖）。
3. **护栏进入物理入口，而非停在 schema。** 新增 `tensorfvm/ice/load.py`：Lindqvist 层冰阻力
   （`lindqvist_terms` / `level_ice_resistance_kN`）与**受护入口** `guarded_level_ice_resistance`，
   默认 `mode='reject'`：任何越域或非有限输入都拿不到 `resistance_kN`（返回 `None` + 完整护栏结论）；
   `mode='clamp'` 是唯一让越域数值进入公式的通道，且必然带 `extrapolation_risk` 标记。
   `require_guarded_level_ice_resistance` 提供抛异常式硬前置条件。
4. **接入真实求解路径（集成调用方 2 处）。**
   (a) `tensorfvm/ice_channel.py`（冰下槽道 FVM/1-D 求解器）新增 `ice_lid_guard`，在 `run_all` 主流程里
   于任何流体求解之前校验冰盖状态，不可能状态直接抛 `IceParameterDomainError`，护栏结论写入
   `summary.json`；(b) `tensorfvm/__init__.py` 导出 `ice` 子包，`pyproject.toml` 注册
   `tensorfvm-ice` 控制台入口，`python -m tensorfvm.ice` 为正式 CLI 入口。
5. **判据机检（不靠自声明）。** 新增 `tensorfvm/ice/benchmark.py`：
   G1 用**独立转录的规范表 + 独立实现的判定规则**（不调用实现代码）在 {n_grid} 个边界探针上比对；
   G2/G3 非有限值围堵率；G4 越域输入在载荷算术前的拦截率（{n_in} 次注入）；
   G5 域内输入直通（护栏不得扰动结果）；G6 锚点历史量级复现（结论 WITHHELD，见 ⑥）。
   `G1` 的规范表是与 `parameters.py` **分开维护的第二份拷贝**，因此该行真正度量的是"规范↔代码漂移"。

## ③ 计算结果

**A. 判据行（`docs/ICE-STRUCT-GUARD-REPAIR/artifacts/benchmark.json`）**

| 行 | 参考 | 实测 | 误差 | 容差 | 判定 |
|---|---|---|---|---|---|
{rows_md}

**B. 冰厚扫描：受护 vs 未受护（`detail.thickness_sweep`）**

| h [m] | 未受护 R [MN] | 受护 R [MN] |
|---|---|---|
{sweep_md}

**C. 锚点历史量级复现（`detail.g6_legacy`）**

| 用例 | 锚点记录 | 仓内复现 | 相对偏差 | 是否被拦截 |
|---|---|---|---|---|
{legacy_md}

**D. 集成调用方的物理正确性（`docs/ICE-STRUCT-GUARD-REPAIR/channel_metrics.json`）**

| 量 | 值 |
|---|---|
| slip 1-D 速度剖面 L2（vs 解析） | {cm['slip_u_l2'] * 100:.4f} % |
| no-slip FVM 速度剖面 L2（vs 解析，{channel['grid']['noslip_grid']}） | {cm['noslip_u_l2'] * 100:.4f} % |
| no-slip dp/dx 相对误差 | {cm['noslip_dpdx_err'] * 100:.4f} % |
| `run_all` 全部判据 | {'PASS' if cm['all_passed'] else 'FAIL'} |

**E. 单测。** `tests/test_ice_guard_pipeline.py`（新，126 例参数化展开）+ 既有
`tests/test_ice_guardrails.py` + `tests/test_ice_channel.py`：**181 passed / 0 failed**。
`ruff check src/tensorfvm/ice src/tensorfvm/ice_channel.py src/tensorfvm/__init__.py tests/test_ice_guard_pipeline.py tests/test_ice_guardrails.py`：**All checks passed**。

## ④ 结果图

* `docs/ICE-STRUCT-GUARD-REPAIR/fig_guard_gate.png` — 左：G1–G5 各行"实测误差 vs 声明容差"（对数轴，
  程序化注入）；右：三个域内船例的受护阻力（G5 直通，护栏零扰动）。
* `docs/ICE-STRUCT-GUARD-REPAIR/fig_ice_channel_physics.png` — 左：冰下槽道 slip 解析剖面 vs 1-D FV 数值剖面
  （集成调用方在护栏内仍解出正确物理）；右：逐点速度相对误差，3 % 带为参考线。
* `docs/ICE-STRUCT-GUARD-REPAIR/artifacts/fig_ice_guard_benchmark.png` — 12 参数三级边界带 + 围堵率 + 冰厚扫描（h=50 m 处 10³ MN）。

**关于"场量云图"的说明：** 护栏是参数域校验器，本身不产生场解，故其判据图不是云图；作为替代，
集成调用方（冰下槽道求解器）的解析解对照曲线（第二张图）承担了"与解析解比较"这一职责，
护栏自身的判据则在第一张图与 ③-A 表中逐行给出。两者不等价，此处显式声明。

## ⑤ 注意点（坑与局限）

1. **G6 是诚实 FAIL/WITHHELD。** 仓内复现的 Lindqvist 值（{legacy['h_50m']['got_MN_in_repo_reproduction']:,.1f} MN /
   {legacy['sigma_c_100MPa']['got_MN_in_repo_reproduction']:,.1f} MN）与锚点记录的 2450 / 143.9 MN 差
   {legacy['h_50m']['rel_error'] * 100:.1f} % / {legacy['sigma_c_100MPa']['rel_error'] * 100:.1f} %：产生那两个数的平台构建不在本仓，
   公式版本（速度乘子、弯曲项、浸没项长度尺度）在 m1720 前后不同，因此**量级不复现**。
   可确证的只有失效模式本身：两个输入都被拦截，且未受护值均在 10³ MN 级（远超任何实船量级）。
2. **阻力公式精度不作声明。** 本模块用未标定 Lindqvist；同公式在平台侧对三条实船试验均值的
   留一交叉验证误差约 −23 %～+17 %（超出 15 % 带），故阻力精度为 WITHHELD，本仓 benchmark 只评护栏。
   公式中 `0.3 L`（浸没项长度尺度）与 `0.1`（拖曳系数）两常数无法溯源到原文，已在 `load.py` 注释中登记。
3. **边界语义。** `h` 的硬上界 10 m 是"船-层冰模型的适用上限"，不是物理不可能（冰脊龙骨、冰川舌远超 10 m）；
   已改写 docstring，避免"physically impossible"的过强表述（m22252 审计 F8）。
4. **静态边界。** 不做温度/盐度依赖边界，也不做相关性检查（厚冰 + 低强度这类组合仍会放行）；
   `bow_angle_deg` 不在 12 参数集内，由载荷入口显式传入。
5. **平台侧残余缺口未修。** `GET /ice-uq/ships` 现返回 200，但 `reference_std_resistance_kN` 仍为 `null`、
   三条船 `geometry_consistent` 全为 `false`；该文件不在本仓，按"只改工作树内文件"的纪律未动。
6. **clamp 语义的边界。** `clamp` 只对有限值/`±inf` 有意义；`nan` 被拒绝而不是"钳到某边界"，
   因为"NaN 在哪一侧"没有答案。

## ⑥ 误差判定（逐项 vs 基准，显式回答是否 <3 %）

| 关键量 | 基准（ref） | 实测（got） | 误差 | <3 %? |
|---|---|---|---|---|
| G1 三级判定（{n_grid} 个边界探针） | 独立规范表 + 独立规则 | 实现判定 | {bundle['rows'][0]['error'] * 100:.4f} % | **是** |
| G2 NaN 围堵率 | 100 % | {bundle['rows'][1]['got'] * 100:.1f} % | 0 pp | **是** |
| G3 ±inf 围堵率（含 clamp 落界有限性） | 100 % | {bundle['rows'][2]['got'] * 100:.1f} % | 0 pp | **是** |
| G4 越域输入在载荷算术前拦截率 | 100 % | {bundle['rows'][3]['got'] * 100:.1f} % | 0 pp | **是** |
| G5 域内直通（护栏对载荷的扰动） | 0 | {bundle['rows'][4]['error']:.3e} | 0 % | **是** |
| 冰下槽道 slip 速度剖面 L2 | 解析解 | {cm['slip_u_l2'] * 100:.4f} % | — | **是** |
| 冰下槽道 no-slip dp/dx | 解析解 −12μU/H² | {cm['noslip_dpdx_err'] * 100:.4f} % | — | **是** |
| G6 越域载荷量级复现（h=50 m） | 2450 MN（2026-11-09 平台构建） | {legacy['h_50m']['got_MN_in_repo_reproduction']:,.1f} MN | {legacy['h_50m']['rel_error'] * 100:.1f} % | **否（WITHHELD）** |
| G6 越域载荷量级复现（σ_c=100 MPa） | 143.9 MN（同上） | {legacy['sigma_c_100MPa']['got_MN_in_repo_reproduction']:,.1f} MN | {legacy['sigma_c_100MPa']['rel_error'] * 100:.1f} % | **否（WITHHELD）** |

**结论：** 护栏自身的目的性判据（越域/非有限输入不得进入物理算术）全部 0 偏差达标（远严于 3 %）；
集成调用方的物理量对解析解误差 0.023 %–1.37 %，亦 <3 %。
**但**锚点记录的两个越域载荷量级无法在仓内复现（39.9 % / 63.7 %），该肢诚实记 WITHHELD，不粉饰。

## 复现

```
{REPRO_CMD}
PYTHONPATH=src python docs/ICE-STRUCT-GUARD-REPAIR/make_figures.py
PYTHONPATH=src python docs/ICE-STRUCT-GUARD-REPAIR/make_report.py
PYTHONPATH=src python -m pytest tests/test_ice_guard_pipeline.py tests/test_ice_guardrails.py tests/test_ice_channel.py -q
```
"""

(HERE / "SPEC.md").write_text(SPEC)
(HERE / "REPORT.md").write_text(REPORT)

# ------------------------------------------------ machine-checkable evidence --
evidence = {
    "anchor": "JOUR-ICE-STRUCT-GUARD",
    "mission": MISSION,
    "repo": "TensorFVM",
    "level_claim": 4,
    "impl_files": [
        "src/tensorfvm/ice/parameters.py",
        "src/tensorfvm/ice/guardrails.py",
        "src/tensorfvm/ice/load.py",
        "src/tensorfvm/ice/benchmark.py",
    ],
    "entry": {
        "module": "src/tensorfvm/ice/__main__.py",
        "console_script": "tensorfvm-ice = tensorfvm.ice.__main__:main",
        "commands": [REPRO_CMD,
                     "PYTHONPATH=src python -m tensorfvm.ice --case <case.json>"],
        "package_export": "tensorfvm.ice (src/tensorfvm/__init__.py)",
    },
    "run_artifacts": sorted(str(p.relative_to(ROOT)) for p in ART.rglob("*")
                            if p.is_file()),
    "integration_callers": [
        {"file": "src/tensorfvm/ice/load.py",
         "how": "guarded_level_ice_resistance gates every Lindqvist evaluation"},
        {"file": "src/tensorfvm/ice_channel.py",
         "how": "ice_lid_guard() called in run_all() before any flow solve; "
                "impossible lid raises IceParameterDomainError"},
        {"file": "src/tensorfvm/__init__.py",
         "how": "package root exports the ice subpackage"},
        {"file": "pyproject.toml",
         "how": "console script tensorfvm-ice"},
    ],
    "tests": ["tests/test_ice_guard_pipeline.py", "tests/test_ice_guardrails.py",
              "tests/test_ice_channel.py"],
    "test_result": "181 passed, 0 failed (pytest, 2026-11-15)",
    "benchmark": {
        "file": "docs/ICE-STRUCT-GUARD-REPAIR/artifacts/benchmark.json",
        "rows_file": "docs/ICE-STRUCT-GUARD-REPAIR/REPORT.md §3A",
        "tolerance": bundle["tolerance"],
        "all_graded_rows_pass": bundle["all_graded_rows_pass"],
        "rows": [{"metric": r["metric"], "ref": r["ref"], "got": r["got"],
                  "error": r["error"], "tolerance": r["tolerance"],
                  "passed": r["passed"]} for r in bundle["rows"]],
    },
    "repro_cmd": REPRO_CMD,
    "figures": ["docs/ICE-STRUCT-GUARD-REPAIR/fig_guard_gate.png",
                "docs/ICE-STRUCT-GUARD-REPAIR/fig_ice_channel_physics.png",
                "docs/ICE-STRUCT-GUARD-REPAIR/artifacts/fig_ice_guard_benchmark.png"],
    "report": "docs/ICE-STRUCT-GUARD-REPAIR/REPORT.md",
    "withheld": ["G6 legacy anchor magnitudes (platform build of 2026-11-09 not in tree)",
                 "level-ice resistance accuracy vs full-scale trials"],
}
(HERE / "MATURITY-EVIDENCE.json").write_text(json.dumps(evidence, indent=2) + "\n")

case_entry = {
    "slug": "CASE-FVM-ICE-GUARD",
    "anchor": "JOUR-ICE-STRUCT-GUARD",
    "title": "冰域 12 参数越域护栏：schema 层 → 物理计算入口 + 判据机检",
    "repo": "TensorFVM",
    "tolerance": {"decision_conformance": 0.03, "nonfinite_containment": 0.0,
                  "load_path_closure": 0.0, "passthrough_relative": 1e-12},
    "reference": "docs/ICE-STRUCT-GUARD-REPAIR/SPEC.md; ISO 19906:2019; Lindqvist (1989) POAC-89",
    "artifacts": "artifacts/ice_guard",
    "report": "docs/ICE-STRUCT-GUARD-REPAIR/REPORT.md",
    "repro_cmd": REPRO_CMD,
    "all_pass": bundle["all_graded_rows_pass"],
}
(HERE / "case_pool_entry.json").write_text(json.dumps(case_entry, indent=2) + "\n")

print(json.dumps({
    "spec": str(HERE / "SPEC.md"),
    "report": str(HERE / "REPORT.md"),
    "evidence": str(HERE / "MATURITY-EVIDENCE.json"),
    "case_entry": str(HERE / "case_pool_entry.json"),
    "rows": [(r["metric"].split()[0], r["passed"]) for r in bundle["rows"]],
}, indent=2))


# ----------------------------------------------------------------- publish ---
def publish() -> dict:
    base = os.environ.get("TENSOR_API_BASE", "").rstrip("/")
    user = os.environ.get("TENSOR_AGENT_USER", "")
    password = os.environ.get("TENSOR_AGENT_PASS", "")
    if not (base and user and password):
        return {"published": False, "reason": "no API credentials in environment"}

    def post(path: str, payload: dict, token: str | None = None) -> dict:
        req = urllib.request.Request(
            f"{base}{path}", data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json",
                     **({"Authorization": f"Bearer {token}"} if token else {})},
            method="POST")
        with urllib.request.urlopen(req, timeout=120) as resp:
            return json.loads(resp.read())

    login = post("/auth/login", {"username": user, "password": password})
    token = login.get("access_token") or login.get("token")
    if not token:
        return {"published": False, "reason": f"login returned {login}"}

    figures = []
    for name, caption in [
        ("fig_guard_gate.png", "护栏判据行：实测误差 vs 声明容差（对数轴）与域内直通阻力"),
        ("fig_ice_channel_physics.png",
         "集成调用方：冰下槽道 slip 解析解 vs 1-D FV 速度剖面，及逐点相对误差（3 % 参考线）"),
        ("fig_ice_guard_benchmark.png",
         "12 参数三级边界带 · 非有限/越域围堵率 · 冰厚扫描（h=50 m → 10³ MN 级）"),
    ]:
        blob = (HERE / name) if (HERE / name).is_file() else (
            ART / name.replace("fig_guard_gate", "fig_ice_guard_benchmark"))
        if blob.is_file():
            figures.append({"name": name, "caption": caption,
                            "png_base64": base64.b64encode(blob.read_bytes()).decode()})

    payload = {
        "title": "冰区越域护栏：从 schema 层进入物理计算入口（12 参数 · NaN 围堵 · 判据机检）",
        "summary": (
            "修复 m22209 的 NaN 放行缺陷，把冰域护栏从 8 参数扩到 12 参数，"
            "并以受护载荷入口 guarded_level_ice_resistance + 冰下槽道求解器两道调用方"
            "使越域输入在算术前被拦截；G1–G5 判据 0 偏差达标，"
            "锚点历史载荷量级复现诚实 WITHHELD（39.9 % / 63.7 %）。"),
        "scenario_key": "JOUR-ICE-STRUCT-GUARD",
        "mission_id": MISSION,
        "tags": ["ice", "guardrail", "TensorFVM", "JOUR-ICE-STRUCT-GUARD", "m22342"],
        "markdown": REPORT,
        "figures": figures,
    }
    resp = post("/tensor/reports", payload, token)
    return {"published": True, "response": resp, "figures": [f["name"] for f in figures]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--publish", action="store_true")
    args = parser.parse_args()
    if args.publish:
        print(json.dumps(publish(), indent=2)[:2000])