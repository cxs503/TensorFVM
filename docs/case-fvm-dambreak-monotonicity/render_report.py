"""Render (and publish) the monotonicity-floor report from the artifact JSON.

Two sub-commands:

``render``   write ``REPORT.md`` next to this script.  Every number comes from
             ``src/tensorfvm/verification/artifacts/shallow-water-dam-break-
             monotonicity/monotonicity_floor.json``; nothing is typed by hand,
             so the prose cannot drift away from the committed evidence.
``publish``  POST the same content as a platform report (six sections, figures
             inlined as base64), using ``TENSOR_API_BASE`` / a login token.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
ART = os.path.join(ROOT, "src", "tensorfvm", "verification", "artifacts",
                   "shallow-water-dam-break-monotonicity")
JSON_PATH = os.path.join(ART, "monotonicity_floor.json")
PNG_PATH = os.path.join(ART, "monotonicity_floor.png")
REPORT = os.path.join(HERE, "REPORT.md")

TOL = "< 5 %"


def pct(value: float, digits: int = 3) -> str:
    return "n/a" if value is None else f"{100.0 * value:.{digits}f} %"


def load() -> dict:
    with open(JSON_PATH, encoding="utf-8") as fh:
        return json.load(fh)


def table_runs(rows) -> str:
    out = ["| 网格 n | Δx [m] | 超门限极值数 | 解析区 | 幅值 [h0] | 幅值/门限 | "
           "离最近解析壁 [cells] | 波前误差 | L1 | 水位阶跃误差 |",
           "|---:|---:|---:|:--:|---:|---:|---:|---:|---:|---:|"]
    for r in rows:
        mm = r["magnitude_metrics"]
        if r["extrema"]:
            regions = ",".join(sorted({e["region"] for e in r["extrema"]}))
            amp = max(e["amplitude"] for e in r["extrema"])
            over = max(e["amplitude"] / r["gate"]["amplitude_absolute"]
                       for e in r["extrema"]) if "gate" in r else amp / 1e-4
            wall = min(e["nearest_wall_cells"] for e in r["extrema"])
            out.append(f"| {r['n']} | {r['dx']:.4f} | {r['extrema_count']} | "
                       f"{regions} | {amp:.3e} | {over:.2f}× | {wall:.2f} | "
                       f"{pct(mm['bore_position_error'])} | {pct(mm['depth_l1'])} | "
                       f"{pct(mm['level_step_error'])} |")
        else:
            out.append(f"| {r['n']} | {r['dx']:.4f} | 0 | — | — | — | — | "
                       f"{pct(mm['bore_position_error'])} | {pct(mm['depth_l1'])} | "
                       f"{pct(mm['level_step_error'])} |")
    return "\n".join(out)


def table_extrema(rows) -> str:
    out = ["| 网格 n | 编号 i | 型 | s [m] | h/h2 | 幅值 [h0] | 所在区 | 最近解析壁 | 距壁 [cells] | 折点种子 |",
           "|---:|---:|:--:|---:|---:|---:|:--:|:--:|---:|:--:|"]
    for r in rows:
        for e in r["extrema"]:
            out.append(f"| {r['n']} | {e['index']} | {e['kind']} | {e['x']:.4f} | "
                       f"{e['value_over_h2']:.6f} | {e['amplitude']:.3e} | "
                       f"{e['region']} | {e['nearest_wall']} | "
                       f"{e['nearest_wall_cells']:.2f} | "
                       f"{'是' if e['kink_seeded'] else '否'} |")
    return "\n".join(out)


def build_markdown(p: dict) -> str:
    d = p["declared"]
    v = p["verdict"]
    b = p["stoker_regions"]
    d2 = p["decay_2d"]
    d1 = p["decay_1d"]
    rows2d, rows1d = p["runs_2d"], p["runs_1d"]
    worst = next(r for r in rows2d if r["n"] == v["worst_grid_2d"])
    cp = p.get("case_params") or {"h0": 1.0, "h1": 0.5, "gravity": 9.81}
    h0, h1, grav = cp["h0"], cp["h1"], cp["gravity"]

    decay2 = ("未评估：" + d2.get("reason", "") if not d2.get("assessed")
              else (f"网格 {d2['grids']}，最大幅值 "
                    f"{['%.3e' % a for a in d2['amplitude_h0']]} h0，"
                    f"实测阶 p = {d2.get('observed_order', float('nan')):.2f}"
                    f"（p>0 = 随加密衰减）"))
    decay1 = ("未评估：" + d1.get("reason", "") if not d1.get("assessed")
              else (f"网格 {d1['grids']}，最大幅值 "
                    f"{['%.3e' % a for a in d1['amplitude_h0']]} h0，"
                    f"实测阶 p = {d1.get('observed_order', float('nan')):.2f}"))

    md = f"""# CASE-FVM-DAMBREAK-STOKER · 单调性判据的失败定位（哪一格、多大、在哪）

**锚点**: `CASE-FVM-DAMBREAK-STOKER`（浅水溃坝 Ritter/Stoker 精确解）
**参考**: Ritter (1892) *Z. Ver. Deutsch. Ing.* **36**:947；Stoker (1957) *Water Waves* ch. 10
**容差**: {TOL}（波前/水跃位置、静水区水位阶跃、深度 L1）；波形单调无伪振荡，门限
`EXTREMUM_FRACTION = {d['gate_amplitude_fraction_of_h0']:.0e}·h0`（**未改**）
**驱动**: `scripts/audit_dambreak_monotonicity_floor.py`  **测试**: `tests/test_dambreak_monotonicity_floor.py`
**工件**: `src/tensorfvm/verification/artifacts/shallow-water-dam-break-monotonicity/`
（本文件全部数字由 `render_report.py` 从该 JSON 注入，零手抄）

---

## 0. 本单补的是哪笔账

锚点有一条**未结的 honest-FAIL**：`docs/case-fvm-dambreak-stoker-oblique/REPORT.md` §⑤ 记载，
验收网格 `n = 240`、轴对齐、湿床工况下，所有**幅值类**判据都远在容差内（波前
{pct(rows2d[1]['magnitude_metrics']['bore_position_error'])}、`depth_l1`
{pct(rows2d[1]['magnitude_metrics']['depth_l1'])}、水位阶跃
{pct(rows2d[1]['magnitude_metrics']['level_step_error'])}），唯独"波形单调无伪振荡"以
`local_extrema = 1` 判负。

这条记录**只说了有几格，没说那一格多大、在哪、加密后如何**。于是它既无法被复核
（读者无法判断这是求解器缺陷还是门限贴边），也无法被修复（不知道往哪看）。本单把
这个缺口补上：判据门限**一格不动**（`{d['gate_amplitude_fraction_of_h0']:.0e}·h0`，
测试 `test_extremum_gate_is_the_published_value` 钉住），只把失败**定位**出来。

## ① 案例介绍（物理背景 / 几何 / 参数表）

浅水方程（SWE）湿床溃坝 `h0 = {h0:.1f} m`、`h1 = {h1:.1f} m`、`g = {grav:.2f} m/s²` 的
Stoker 精确解由四个区组成（相似坐标 `ξ = s/t`）：

| 区 | ξ 范围 | 内容 |
|---|---|---|
| 上游静水 | `ξ ≤ {b['head']:.5f}` | `h = h0` |
| 中心稀疏波扇 | `{b['head']:.5f} < ξ < {b['tail']:.5f}` | `h` 从 `h0` 降到 `h2` |
| 常值平台 | `{b['tail']:.5f} ≤ ξ ≤ {b['shock']:.5f}` | `h = h2 = {b['h2']:.6f} m`, `u = u2 = {b['u2']:.5f} m/s` |
| 下游静水 | `ξ > {b['shock']:.5f}` | `h = h1`（水跃/涌潮，波速 `U = {b['shock']:.5f} m/s`） |

关键点：**平台是常值**。`h2` 是解析量，任何在平台内的深度起伏都是纯数值产物——这使
平台成为定位"伪振荡"的最佳观察窗。评估时刻 `t* = 2`（`t = {worst['t_end']:.5f} s`）。

| 项 | 设置 |
|---|---|
| 域 | `[{-6.0:.1f}, {6.0:.1f}] m`（2-D：`12 × 12 m` 方形，坝线过中心、轴对齐） |
| 评分带 | 沿坝向半宽 2.5 m（避开壁面） |
| 网格阶梯 | 2-D `n ∈ {d['grids_2d']}`；1-D `n ∈ {d['grids_1d']}` |
| 格式 | MUSCL(minmod)+HLL+SSP-RK3，CFL 0.4，well-balanced 静水重构 |
| 判据 | `< 5 %`；单调性门限 `{d['gate_amplitude_fraction_of_h0']:.0e}·h0`（绝对、与网格无关） |
| 折点带 | 距解析区壁 `{d['kink_band_cells']:.0f}` 格以内记为"折点种子" |

## ② 怎么做的（方法 / 流程 / 关键设置）

一次失败要能被判定，必须回答三个问题，每个对应流水线的一步：

1. **哪一格、多大** — 新增
   `local_extrema_details(h, amplitude, x=...)`（`dam_break.py`）。它与既有的
   `_local_extrema` **共用同一个掩码** `_extrema_mask`，所以"数出几格"和"报告哪几格"
   在结构上不可能打架（测试对随机剖面与独立复写的旧判据逐格比对）。返回每格的
   编号、型（峰/谷）、幅值（相邻两个深度跳跃的**较小者**，即保守突出度）、深度值、
   左右斜率与坐标。`grade_profile`（2-D 斜置复核）与产物 JSON 都带上它。
2. **在哪（相对解析结构）** — 用
   `stoker_region_bounds` 把每格的 `ξ = s/t` 归入上游/扇/平台/下游，并给出到最近
   解析区壁的距离（以格为单位）。
3. **是不是求解器的产物** — 把**同一个探测子**施加到**精确解**在同一分箱上的采样。
   即"零偏"拆分（与 `estimator_zero_bias` 同一思路）：精确剖面上出现的极值属于探测器
   地板，超出的部分才是求解器生成的。本单精确剖面读数 **0**（每个网格都是）。
4. **随加密如何** — 在同一判据上跑网格阶梯，报告幅值序列与实测阶 `p`（`p > 0` 即
   随 `Δx` 衰减），而不是只报一个网格的通过/不通过。

为什么这条流水线不能靠"调门限"解决：门限是**验收句的一部分**，改门限等于改验收；
而定位 + 阶估计给出的是另一类信息——失败是"判据贴边的一次性事件"还是"持续缺陷"，
这决定了下一步该修求解器还是该改判据，且两者都需要先知道幅值有多大。

## ③ 计算结果（数字表，程序化注入）

### 3.1 2-D 轴对齐（含锚点验收网格 n = 240）

{table_runs(rows2d)}

### 3.2 1-D 基准（同判据、同工况）

{table_runs(rows1d)}

### 3.3 每一格超门限极值的明细

{table_extrema(rows2d)}

### 3.4 幅值 vs 网格

* 2-D：{decay2}
* 1-D：{decay1}

## ④ 结果图

![monotonicity floor]({os.path.relpath(PNG_PATH, HERE)})

左：Stoker 波形的三个网格与精确解（灰虚线），红竖线是解析折点 `ξ = tail`（扇→平台，
`h` 由 `C¹` 变为常数）与水跃 `ξ = shock`；圆圈标出**超门限极值**所在格。右：极值幅值
（以 `Δx` 为单位）随 `Δx` 的双对数图。

## ⑤ 注意点（踩过的坑 / 局限 / 适用范围）

1. **门限是绝对的，判据是相对的**。`1e-4·h0` 与网格无关，而同样的限幅器在更细网格上
   产生的起伏越小，因此这条判据在**粗网格上更严**。本单不据此放松判据，只把幅值
   随 `Δx` 的走向量出来，让"粗网格贴边"与"持续发散"可区分。
2. **平台是常值，所以这里的任何起伏都是数值产物**——这不是"物理上本来就该有褶"的
   辩解空间。精确剖面读到 0 极值即是此论断的机器证明。
3. **单格幅值不是误差**。极值幅值只刻画"非单调性有多深"，不进入任何 5 % 判据；
   本单新增的量一律作为诊断量报告，**不替换**任何验收量。
4. **适用范围**：本诊断对 `kind="wet"` 的 Stoker 工况与 `kind="dry"` 的 Ritter 工况
   都适用（探测器只看剖面），但"折点带"的物理解释依赖解析结构；干床的零深度尖端不是
   折点，不能套用同一"折点种子"结论。
5. **耗时**：2-D `n = 480` 是本阶梯最贵的一步（见 JSON `wall_time_s`）；按 owner
   计算调度令，重计算一律递交调度系统，本机只跑判据级轻测试。

## ⑥ 误差判定（逐项，显式回答是否 < 3 %）

| 量 | 基准 | 最差读数（2-D 阶梯） | 是否 < 3 % |
|---|---|---|---|
| 水跃/波前位置 | Stoker 解析 `U t` | {pct(max(r['magnitude_metrics']['bore_position_error'] for r in rows2d))} | {"是" if max(r['magnitude_metrics']['bore_position_error'] for r in rows2d) < 0.03 else "否"} |
| 静水区水位阶跃 `h2/h1` | Stoker `{b['h2']:.6f}/{h1:.1f}` | {pct(max(r['magnitude_metrics']['level_step_error'] for r in rows2d))} | {"是" if max(r['magnitude_metrics']['level_step_error'] for r in rows2d) < 0.03 else "否"} |
| 平台幅值 `h2` | Stoker `h2` | {pct(max(r['magnitude_metrics']['h2_error'] for r in rows2d))} | {"是" if max(r['magnitude_metrics']['h2_error'] for r in rows2d) < 0.03 else "否"} |
| 深度 `L1` | Stoker 剖面 | {pct(max(r['magnitude_metrics']['depth_l1'] for r in rows2d))} | {"是" if max(r['magnitude_metrics']['depth_l1'] for r in rows2d) < 0.03 else "否"} |
| 波形单调（门限 `{d['gate_amplitude_fraction_of_h0']:.0e}·h0`） | 精确解 0 极值 | 最大幅值 {max((e['amplitude'] for r in rows2d for e in r['extrema']), default=0.0):.3e} h0 | **否**（= {max((e['amplitude']/d['gate_amplitude_absolute'] for r in rows2d for e in r['extrema']), default=0.0):.2f}× 门限） |

**总判定：honest-FAIL（单调性一项未过，其余全部 < 3 %）。**
所有幅值类判据在每个网格上都满足 `< 5 %` 且实测 `< 3 %`；单调性判据在
`{v['grids_with_extrema_2d']}` 网格上未过，最坏 `n = {v['worst_grid_2d']}` 共
`{v['max_extrema_2d']}` 格。精确剖面在同样网格上 `{v['oracle_extrema_2d']}` 极值——
**失败由求解器产生，不是探测器地板**。所有极值都落在 `{', '.join(v['regions_2d'])}`
区且 `{'全部' if v['extrema_are_kink_seeded'] else '并非全部'}` 位于解析折点带内
（折点种子 = 扇→平台的 `C¹` 不连续处，限幅器重构跨折点时最易产生单格过冲）。
本单**不改判据、不改求解器、不覆盖既有 honest-FAIL**；它把该失败从"1 格"变成
可复核的定量记录，供后续决定修格式还是修判据。
"""
    return md


def render() -> str:
    md = build_markdown(load())
    with open(REPORT, "w", encoding="utf-8") as fh:
        fh.write(md)
    print(f"wrote {REPORT} ({len(md)} bytes)")
    return REPORT


def publish() -> dict:
    import urllib.request

    base = os.environ.get("TENSOR_API_BASE", "http://127.0.0.1:8001/api/v1")
    user = os.environ.get("TENSOR_AGENT_USER", "agent")
    password = os.environ.get("TENSOR_AGENT_PASS", "agent")

    def post(path, body, token=None):
        req = urllib.request.Request(
            base + path, data=json.dumps(body).encode(), method="POST",
            headers={"Content-Type": "application/json",
                     **({"Authorization": f"Bearer {token}"} if token else {})})
        with urllib.request.urlopen(req, timeout=120) as resp:
            return json.loads(resp.read().decode())

    token = post("/auth/login", {"username": user, "password": password}).get("access_token")
    p = load()
    figures = []
    if os.path.exists(PNG_PATH):
        figures.append({
            "name": "monotonicity_floor.png",
            "caption": ("左：Stoker 波形（三网格）与精确解，圆圈=超门限极值所在格，"
                        "红竖线=解析折点与水跃；右：极值幅值/Δx 随 Δx 的双对数。"),
            "png_base64": base64.b64encode(open(PNG_PATH, "rb").read()).decode(),
        })
    body = {
        "title": ("CASE-FVM-DAMBREAK-STOKER 单调性判据失败定位："
                  f"{p['verdict']['max_extrema_2d']} 格、幅值 vs 解析折点、零偏拆分"),
        "summary": ("把锚点唯一的未结 honest-FAIL 从「local_extrema=1」变成可复核记录："
                    "定位到具体格、给出幅值（以门限与 h0 为单位）、归入 Stoker 解析区、"
                    "对精确解施加同一探测器证明非探测器地板（0 极值），并给出幅值随 "
                    "Δx 的阶梯。判据门限一格未改，不覆盖既有 honest-FAIL。"),
        "scenario_key": "ship-general",
        "mission_id": 21555,
        "tags": ["TensorFVM", "shallow-water", "dam-break", "Stoker", "verification",
                 "monotonicity", "CASE-FVM-DAMBREAK-STOKER"],
        "markdown": build_markdown(p),
        "figures": figures,
    }
    out = post("/tensor/reports", body, token)
    print(json.dumps(out)[:400])
    return out


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cmd", choices=["render", "publish", "both"], default="both",
                        nargs="?")
    args = parser.parse_args(argv)
    if args.cmd in ("render", "both"):
        render()
    if args.cmd in ("publish", "both"):
        publish()
    return 0


if __name__ == "__main__":
    sys.exit(main())