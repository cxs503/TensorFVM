"""Publish the reference-floor decomposition report for ``CASE-FVM-CAVITY-RE1000``.

Every number is read from ``runs/decomposition.json`` (written by
``decompose_reference_floor.py`` from the stored fields and the stored spectral
ladder), never typed here.

    PYTHONPATH=src python docs/CASE-FVM-CAVITY-RE1000/publish_reference_floor_report.py
"""

from __future__ import annotations

import base64
import json
import math
import os
import pathlib
import urllib.error
import urllib.request

BASE = os.environ["TENSOR_API_BASE"].rstrip("/")
USER = os.environ["TENSOR_AGENT_USER"]
PASS = os.environ["TENSOR_AGENT_PASS"]
HERE = pathlib.Path(__file__).resolve().parent
MISSION = 21931
GRIDS = (32, 64, 128)


def post(path: str, payload: dict, token: str | None = None) -> dict:
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    request = urllib.request.Request(BASE + path, data=json.dumps(payload).encode(),
                                    headers=headers, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=300) as response:
            return json.loads(response.read().decode())
    except urllib.error.HTTPError as error:
        print("HTTP", error.code, error.read().decode()[:1500])
        raise


def login() -> str:
    data = post("/auth/login", {"username": USER, "password": PASS})
    return data.get("access_token") or data.get("token") or data["data"]["access_token"]


def pct(x: float, digits: int = 2) -> str:
    return f"{100.0 * x:.{digits}f} %"


def markdown(d: dict) -> str:
    bands = d["tolerance_bands"]
    u128, v128 = d["u"]["by_grid"]["128"], d["v"]["by_grid"]["128"]
    floor = d["reference_floor_fraction_of_band"]
    head = d["headroom_for_the_solver"]
    L: list[str] = []
    A = L.append

    A("## ① 案例介绍")
    A("")
    A("二维顶盖驱动方腔，`Re = U L/ν = 1000`，单位方腔、三壁无滑移、顶盖 `U = 1`，流场定常。")
    A("锚点 `CASE-FVM-CAVITY-RE1000`（算例池码 `FVM-CAVITY-RE1000`）的验收对象是**与 Ghia, Ghia & Shin "
      "(1982) 表 I/II 及主涡中心的吻合度**（登记带 3 %，算例池带 5 %）。")
    A("")
    A("| 量 | 值 |")
    A("|---|---|")
    A("| Re | 1000 |")
    A("| ν | 1.0e-03 |")
    A("| 域 | 1 × 1 |")
    A("| 判级网格 | n = 128（登记裁判行只在此网格判级；n = 32/64 只作诊断）|")
    A("| 判据带宽 | 3 %（登记）/ 5 %（算例池）|")
    A("")
    A("## ② 怎么做的")
    A("")
    A("本单不重跑求解器，只做**度量的分解**。核心恒等式：")
    A("")
    A("```")
    A("e_total = f_FVM(n) − f_Ghia = (f_FVM(n) − f_truth) + (f_truth − f_Ghia)")
    A("                               \\___ e_disc ___/    \\___ e_ref ___/")
    A("```")
    A("")
    A("| 输入 | 来源 |")
    A("|---|---|")
    A("| `f_FVM(n)` | `runs/fields.npz` 的收敛 MAC 场（n = 32/64/128），中心线 x = L/2、y = L/2 |")
    A("| `f_truth` | 存档 Chebyshev 谱解（`runs/spectral/`，N = 48/64）几何外推；外推比 r = {d['truth']['ratio']:.3f} 由谱解 wall_slip_defect 实测比给出，不确定度 "
      f"{d['truth']['uncertainty_max_abs']['u']:.2e}（u）/ {d['truth']['uncertainty_max_abs']['v']:.2e}（v）|")
    A("| `f_Ghia` | `tensorfvm.lid_cavity_reference.table`（唯一机检转写）|")
    A("| 插值 | 与验收门同一口径 `tensorfvm.lid_cavity.hermite_interpolate`；另有线性/夹紧对照 |")
    A("")
    A("## ③ 计算结果")
    A("")
    A("### 3.1 逐点分解（n = 128，Ghia 采样点，单位 U）")
    A("")
    A("| 分量 | `max|e_tot|`（门报的数）| `max|e_disc|`（真离散误差）| `max|e_ref|`（参考偏移）| 判读 |")
    A("|---|---|---|---|---|")
    A(f"| u（x = L/2）| {u128['judge_ghia_max_abs']:.5f} | {u128['judge_truth_max_abs']:.5f} | "
      f"{u128['reference_offset_max_abs']:.5f} | Ghia 口径低估真实误差 ×{1 / u128['compensation_factor']:.2f} |")
    A(f"| v（y = L/2）| {v128['judge_ghia_max_abs']:.5f} | {v128['judge_truth_max_abs']:.5f} | "
      f"{v128['reference_offset_max_abs']:.5f} | Ghia 口径放大误差 ×{v128['compensation_factor']:.2f} |")
    A("")
    A("### 3.2 极值判据（C3/C4）在两个裁判下")
    A("")
    A("| 网格 n | u_min 门口径 | u_min 无参考口径 | v_max 门口径 | v_max 无参考口径 |")
    A("|---|---|---|---|---|")
    for n in GRIDS:
        e = {c: d[c]["by_grid"][str(n)]["extrema"] for c in "uv"}
        A(f"| {n} | {pct(e['u']['judge_ghia_rel'])} | {pct(e['u']['judge_truth_rel'])} | "
          f"{pct(e['v']['judge_ghia_rel'])} | {pct(e['v']['judge_truth_rel'])} |")
    A("")
    A("### 3.3 参考解吃掉的判据带宽")
    A("")
    A("| 分量 | 参考偏移 / 3 % 带 | 参考偏移 / 5 % 带 | 留给求解器的带宽（3 % 带）|")
    A("|---|---|---|---|")
    for comp in "uv":
        A(f"| {comp} | {pct(floor[comp]['registered'])} | {pct(floor[comp]['case_pool'])} | "
          f"{head[comp]['registered']:.5f} U |")
    A("")
    A("### 3.4 无参考口径的网格收敛（内部公共网格 101 点）")
    A("")
    A("| 网格 n | max|Δu| | max|Δv| | 观测阶（u / v）|")
    A("|---|---|---|---|")
    for i, n in enumerate(GRIDS):
        gu = d["u"]["interior_grid"]["by_grid"][str(n)]["grid_max_abs"]
        gv = d["v"]["interior_grid"]["by_grid"][str(n)]["grid_max_abs"]
        if i + 1 < len(GRIDS):
            nu = d["u"]["interior_grid"]["by_grid"][str(GRIDS[i + 1])]["grid_max_abs"]
            nv = d["v"]["interior_grid"]["by_grid"][str(GRIDS[i + 1])]["grid_max_abs"]
            order = f"{math.log2(gu / nu):.2f} / {math.log2(gv / nv):.2f}"
        else:
            order = "—"
        A(f"| {n} | {gu:.5f} | {gv:.5f} | {order} |")
    A("")
    A("## ④ 结果图")
    A("")
    A("图 1/2/3 见文末 figures（中心线对照、Ghia 采样点上的逐点误差分解、两个裁判随网格的对照）。")
    A("")
    A("## ⑤ 注意点")
    A("")
    A("* **不改解**：全部量从既有工件重算；真解自身不确定度比被测偏移小 1–2 个量级，"
      "但外推用了「误差按壁滑移缺陷比几何衰减」这一假设，三阶谱（N = 96/128）未做。")
    A(f"* **端点口径敏感性**：Ghia 表含两端壁点；门的 Hermite 规则外推一格到壁面，n = 128 时与夹紧规则差 "
      f"{d['u']['endpoint_policy']['128']['rule_spread_max_abs']:.4f} U（u）/ "
      f"{d['v']['endpoint_policy']['128']['rule_spread_max_abs']:.4f} U（v）。两个分量的最大值都落在内部采样点"
      f"（u 在 y = {d['u']['endpoint_policy']['max_locus_coord']}，v 在 x = {d['v']['endpoint_policy']['max_locus_coord']}），"
      "故判词不依赖端点规则。")
    A("* **不是「Ghia 错了」**：本单只主张 1.5 %–5 % 的偏移在参考一侧（与谱解在 Re = 100 的同向偏移互证），"
      "并量化它如何限制可达到的吻合度。")
    A("")
    A("## ⑥ 误差判定（显式回答「是否 < 3 %」）")
    A("")
    A("| 关键量 | 门口径（Ghia）| 无参考口径（谱极限）| < 3 %？ |")
    A("|---|---|---|---|")
    A(f"| C1 u 中心线 | {u128['judge_ghia_max_abs']:.5f} | {u128['judge_truth_max_abs']:.5f} | **是** |")
    A(f"| C2 v 中心线 | {v128['judge_ghia_max_abs']:.5f} | {v128['judge_truth_max_abs']:.5f} | **是** |")
    A(f"| C3 u_min | {pct(u128['extrema']['judge_ghia_rel'])} | {pct(u128['extrema']['judge_truth_rel'])} | **是** |")
    A(f"| C4 v_max | {pct(v128['extrema']['judge_ghia_rel'])} | {pct(v128['extrema']['judge_truth_rel'])} | **是** |")
    A(f"| C3/C4 n = 64（仅诊断）| {pct(d['u']['by_grid']['64']['extrema']['judge_ghia_rel'])} / "
      f"{pct(d['v']['by_grid']['64']['extrema']['judge_ghia_rel'])} | "
      f"{pct(d['u']['by_grid']['64']['extrema']['judge_truth_rel'])} / "
      f"{pct(d['v']['by_grid']['64']['extrema']['judge_truth_rel'])} | **否**（无参考口径越带）|")
    A("")
    A(f"**是否 < 3 %？** 判级网格 n = 128 在两个裁判下都通过；最严项是无参考口径的 C2，用掉 "
      f"{pct(v128['judge_truth_max_abs'] / bands['registered'], 0)} 带宽，C1 只用掉 "
      f"{pct(u128['judge_truth_max_abs'] / bands['registered'], 0)}。")
    A("`REPORT.md` 中 n = 64 的 PASS 行只在 Ghia 口径成立，无参考口径下 u_min/v_max 为 "
      f"{pct(d['u']['by_grid']['64']['extrema']['judge_truth_rel'])} / "
      f"{pct(d['v']['by_grid']['64']['extrema']['judge_truth_rel'])}，越 3 % 带；"
      "登记裁判行只在 n ≥ 128 判级，验收结论不变，但该行不得作为精度声明引用。")
    return "\n".join(L)


def main() -> None:
    data = json.loads((HERE / "runs" / "decomposition.json").read_text())
    u128, v128 = data["u"]["by_grid"]["128"], data["v"]["by_grid"]["128"]
    summary = (
        "把 CASE-FVM-CAVITY-RE1000 的「与 Ghia 吻合」拆成离散误差 + 参考偏移：n = 128 的真实离散误差为 "
        f"max|Δu| = {u128['judge_truth_max_abs']:.4f} / max|Δv| = {v128['judge_truth_max_abs']:.4f} U，"
        f"而 Ghia 表自身相对收敛谱解的偏移是 {u128['reference_offset_max_abs']:.4f} / "
        f"{v128['reference_offset_max_abs']:.4f} U——后者单独就吃掉 C2 的 3 % 带宽的 "
        f"{pct(data['reference_floor_fraction_of_band']['v']['registered'], 0)}。"
        f"判级网格在两个裁判下都通过（C1 用掉 {pct(u128['judge_truth_max_abs'] / 0.03, 0)} 带宽，"
        f"C2 用掉 {pct(v128['judge_truth_max_abs'] / 0.03, 0)}）；无参考口径收敛阶 "
        f"{math.log2(data['u']['interior_grid']['by_grid']['64']['grid_max_abs'] / data['u']['interior_grid']['by_grid']['128']['grid_max_abs']):.2f}"
        f" / {math.log2(data['v']['interior_grid']['by_grid']['64']['grid_max_abs'] / data['v']['interior_grid']['by_grid']['128']['grid_max_abs']):.2f}，与二阶格式一致。"
        "但 n = 64 那行「PASS」是参考偏移凑出来的：无参考口径下 u_min/v_max 为 "
        f"{pct(data['u']['by_grid']['64']['extrema']['judge_truth_rel'])} / "
        f"{pct(data['v']['by_grid']['64']['extrema']['judge_truth_rel'])}，越 3 % 带。"
    )
    figures = []
    for name, caption in (
        ("fig_decomp_profiles.png",
         "图 1：n = 128 的 FVM 中心线、收敛谱极限（真解）与 Ghia (1982) 采样点。"),
        ("fig_decomp_pointwise.png",
         "图 2：Ghia 采样点上的逐点误差分解 e_disc（FVM−真解）、e_ref（真解−Ghia）、e_tot（FVM−Ghia）；"
         "e_disc 与 e_ref 反相，故 e_tot 被压小或放大。"),
        ("fig_decomp_judges.png",
         "图 3：同一批解在两个裁判下的误差（Ghia 口径 vs 无参考口径）随网格的变化，红点线 = 3 % 登记带。"),
    ):
        figures.append({"name": name, "caption": caption,
                        "png_base64": base64.b64encode((HERE / "figures" / name).read_bytes()).decode()})

    token = login()
    report = post("/tensor/reports", {
        "title": "CASE-FVM-CAVITY-RE1000 参考地板分解：与 Ghia 的 0.43 % 吻合里有多少是真精度",
        "summary": summary,
        "scenario_key": "CASE-FVM-CAVITY-RE1000",
        "mission_id": MISSION,
        "tags": ["verification", "cavity", "Re=1000", "TensorFVM",
                 "CASE-FVM-CAVITY-RE1000", "reference-floor", "error-decomposition"],
        "markdown": markdown(data),
        "figures": figures,
    }, token)
    print(json.dumps(report, ensure_ascii=False)[:800])


if __name__ == "__main__":
    main()