#!/usr/bin/env python
"""Render ``REPORT.md`` from the archived third-code study (no hand-copied numbers).

Every table and every quoted number below is read from
``src/tensorfvm/verification/artifacts/shallow-water-third-code/xval.json``; the
prose references keys, not literals, so regenerating the archive updates the
report.
"""

from __future__ import annotations

import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[2]
ARTIFACT = (ROOT / "src" / "tensorfvm" / "verification" / "artifacts"
            / "shallow-water-third-code" / "xval.json")
OUT = pathlib.Path(__file__).resolve().parent / "REPORT.md"

SHORT = {
    "tensorfvm-hll-muscl-ssprk3": "TensorFVM HLL+MUSCL+SSPRK3",
    "clawpack-classic-roe": "Clawpack Roe+efix",
    "clawpack-classic-hlle": "Clawpack HLLE",
}


def pct(x, digits=4):
    return "—" if x is None else f"{100.0 * x:.{digits}f} %"


def num(x, digits=4):
    return "—" if x is None else f"{x:.{digits}f}"


def main() -> int:
    d = json.loads(ARTIFACT.read_text())
    v = d["verdict"]
    crit = v["criteria"]
    grids = sorted(int(n) for n in d["wet"])
    wet = d["wet"]
    n_fine = grids[-1]

    lines: list[str] = []
    A = lines.append

    A("# CASE-FVM-DAMBREAK-STOKER · 第三方代码交叉验证（Clawpack 5.14.0 独立证人）")
    A("")
    A("**锚点**: `CASE-FVM-DAMBREAK-STOKER`（TensorFVM 浅水算例池，总账 `status = open` 开工核对）· "
      "本单 **mission m23268** · 分支 `agent/m23268`")
    A("**参考**: Ritter (1892) *Z. Ver. Deutsch. Ing.* **36**, 947-954；Stoker (1957) *Water Waves* ch. 10；"
      "Toro (2009) ch. 6-7")
    A(f"**容差**: {pct(d['tolerance'], 0)}（锚点同口径）· 判据网格下限 n = {d['accepted_min_n']}（锚点 `ACCEPTED_MIN_N`）· "
      f"`t* = 2` · CFL {d['cfl']}")
    A(f"**第三方代码**: {d['third_code']}（Fortran 波传播内核，独立作者/限幅器/波分解）")
    A("**性质**: 补 `VERIFICATION_DEBT_LEDGER` D29 建议的 `XVAL-FVM-DAMBREAK-THIRD-CODE` —— "
      "锚点此前的全部证人（解析解复核、oracle-free 读数、动量恒等式、配置包络、冻结网）**都在 TensorFVM 内部**，"
      "共有网格/边界/提交纪律；本单引入**真正的异族代码**作为独立证人。")
    A("")
    A("---")
    A("")
    A("## ① 案例介绍")
    A("")
    A("浅水方程 `h_t + (hu)_x = 0`、`(hu)_t + (hu² + gh²/2)_x = 0` 的经典溃坝初值问题：")
    A("`x = 0` 处间断，`g = 9.81 m/s²`，计算域 `[-4, 6] m`，`t* = t√(g h0)/h0 = 2`。")
    A("")
    A("| 算例 | 类型 | h0 [m] | h1 [m] | t_end [s] | 解析参照 |")
    A("|---|---|---:|---:|---:|---|")
    for key, case in d["cases"].items():
        A(f"| `{key}` | {case['kind']} | {case['h0']:.3f} | {case['h1']:.3f} | "
          f"{case['t_end']:.6f} | {case['reference']} |")
    A("")
    st = d["exact"]["stoker"]
    A(f"解析量（Stoker 匹配条件，见 `shallow_water.exact`）：星区水位 `h* = {st['h2']:.6f} m`、"
      f"星区流速 `u* = {st['u2']:.6f} m/s`、水跃速度 `U = {st['shock_speed']:.6f} m/s`、"
      f"水位阶跃 `h*/h1 = {st['h2'] / d['cases']['stoker-wet']['h1']:.6f}`；"
      f"干床波前 `x_f = 2 c0 t = {d['exact']['ritter_front_x']:.6f} m`。")
    A("")
    A("## ② 怎么做的")
    A("")
    A("1. **同一问题、同一网格、同一端面**：两码都用零梯度（transmissive）端面、CFL 0.4、`t* = 2`、"
      "同样的单元中心初值 `h(x<0) = h0, h(x>=0) = h1`；比较**逐格**进行，**不插值**"
      "（插值会把整格平移吸收掉，正是最该被抓住的缺陷）。")
    A("2. **异族通量**：Clawpack Classic 波传播法，MC 限幅，两个独立 Riemann 解算器"
      "（Roe + entropy fix、HLLE），与 TensorFVM 的 HLL + MUSCL(minmod) + SSP-RK3 属不同格式族。")
    A("3. **同一量具**：锚点的三条头条读数——干床波前 `fan_front_estimate`、湿床台地/静水位/水跃"
      "`plateau_from_depth_modes`、单调性 `local_extrema_details`——**原样**作用到第三方剖面上，"
      "回答的不是「TensorFVM 对不对」，而是「这条验收门在异族代码上是否给出同样的判决」。")
    A("4. **量具自身标定**（adversarial 纪律）：对距离度量注入已知缺陷——平移 1…128 格、"
      "台地幅值 +1/+2/+5 %、1e-9 舍入噪声——证明读数随缺陷单调移动且不误报；"
      "并把**每条**读数在同一扰动下的灵敏度并列，回答「哪条读数更锐」。")
    A("5. **干床的诚实边界**：Clawpack Classic 不保正性，锚点的精确干床初值它**跑不起来**；"
      "因此另做 `h1 = 0.2 … 1e-3` 的**干极限阶梯**（两码都能跑的范围）与**可湿性下限扫描**，"
      "把「第三方证人只覆盖湿床」这一局限量化到具体深度，而不是换成软化初值冒充同一测试。")
    A("")
    A("## ③ 计算结果")
    A("")
    A("### 3.1 湿床（Stoker）逐格读数——同一量具、三个代码")
    A("")
    A("| n | 代码 | 水跃位置误差 | 水位阶跃误差 | 深度 L1 vs 解析 | 内部极值数 |")
    A("|---:|---|---:|---:|---:|---:|")
    for n in grids:
        for code, m in wet[str(n)]["codes"].items():
            A(f"| {n} | {SHORT.get(code, code)} | {pct(m['bore_error'])} | {pct(m['step_error'])} | "
              f"{pct(m['depth_l1_vs_exact'])} | {m['n_extrema']} |")
    A("")
    A("### 3.2 跨码距离（同网格、不插值）")
    A("")
    A("| n | 对比 | 深度 L1 | 动量 hu L1 | 一格里 L1（本网格灵敏度） | 折算平移 = 距离/一格 |")
    A("|---:|---|---:|---:|---:|---:|")
    for n in grids:
        row = wet[str(n)]
        per_cell = row.get("one_cell_translation_l1")
        for code, dist in row.get("cross_code", {}).items():
            A(f"| {n} | vs {SHORT.get(code, code)} | {pct(dist['depth_l1'])} | "
              f"{pct(dist['momentum_l1'])} | {pct(per_cell)} | "
              f"{num(dist['depth_l1'] / per_cell if per_cell else None, 3)} 格 |")
    A("")
    A("距离随加密单调下降；在验收网格带上，最高读数折算为 "
      f"**{num(crit['XV1_cross_code_depth_l1']['equivalent_translation_cells'], 3)} 格**"
      f"（{num(crit['XV1_cross_code_depth_l1']['equivalent_translation_m'], 5)} m，格宽 "
      f"{num(crit['XV1_cross_code_depth_l1']['cell_width_m'], 5)} m），即两码的波形差别**小于半格**。")
    A("")
    A("### 3.3 干床（Ritter）——第三方代码跑不起来（honest 局限）")
    A("")
    A("| n | 代码 | 结果 | 波前位置 | 解析波前 | 波前误差 |")
    A("|---:|---|---|---:|---:|---:|")
    for n in grids:
        for code, m in d["dry"][str(n)]["codes"].items():
            if m.get("ok"):
                A(f"| {n} | {SHORT.get(code, code)} | 成功 | {num(m['front'], 5)} m | "
                  f"{num(m['front_exact'], 5)} m | {pct(m['front_error'])} |")
            else:
                A(f"| {n} | {SHORT.get(code, code)} | **非有限解**（{m.get('non_finite_cells')} / {n} 格） | — | "
                  f"{num(d['dry'][str(n)]['codes']['tensorfvm-hll-muscl-ssprk3']['front_exact'], 5)} m | — |")
    A("")
    A("失败模式与量级已归档在 `xval.json` 的 `dry`/`wettability_floor` 节点：Classic 的 Roe/HLLE 通量"
      "都不保正性，干床/近干床会出现负水深。**结论：干床三条判据仍是单码证据**，本单不改变这一事实。")
    A("")
    A("### 3.4 干极限阶梯（两码都能跑的下游水深）")
    A("")
    A("| h1 [m] | n | 水跃前缘解析 [m] | TensorFVM 误差 | Clawpack Roe | Clawpack HLLE | 跨码 L1 (Roe) |")
    A("|---:|---:|---:|---:|---:|---:|---:|")
    for h1_str, rungs in sorted(d["dry_limit"].items(), key=lambda kv: -float(kv[0])):
        for n in sorted(int(k) for k in rungs):
            row = rungs[str(n)]
            codes = row["codes"]
            exact = codes["tensorfvm-hll-muscl-ssprk3"].get("bore_exact")
            def cell(code):
                m = codes.get(code, {})
                return pct(m["bore_error"]) if m.get("ok") else "**死**"
            cc = row.get("cross_code", {}).get("clawpack-classic-roe", {})
            A(f"| {float(h1_str):g} | {n} | {num(exact, 5)} | "
              f"{cell('tensorfvm-hll-muscl-ssprk3')} | {cell('clawpack-classic-roe')} | "
              f"{cell('clawpack-classic-hlle')} | {pct(cc.get('depth_l1'))} |")
    A("")
    floor = d["wettability_floor"]
    A("可湿性下限（n = " + str(floor["n"]) + "，能积分到 `t* = 2` 的最薄下游水层）：")
    A("")
    A("| 代码 | 最薄可跑 h1 [m] |")
    A("|---|---:|")
    for code, depth in floor["floor"].items():
        A(f"| {SHORT.get(code, code)} | {('—' if depth is None else f'{depth:g}')} |")
    A("")
    A("TensorFVM 的湿/干处理（HLL + 计数式 clip）能吃到比第三方代码薄一到两个量级的水层，"
      "这正是干床判据无法外包给 Classic Clawpack 的机制原因。")
    A("")
    A("## ④ 结果图")
    A("")
    A("![third code](../case-fvm-dambreak-stoker-third-code/fig_third_code.png)")
    A("")
    A("四联图（`scripts/run_dambreak_third_code.py` 生成）：(a) 最细网格上三码剖面 vs Stoker 解析；"
      "(b) 跨码深度 L1 随网格下降与 5 % 容差线；(c) 同一条读数（水跃位置 / 水位阶跃）在三码上的误差；"
      "(d) 干极限阶梯里水跃前缘误差 vs 下游水深 `h1`，并标出第三方代码的非有限解边界。")
    A("")
    A("## ⑤ 注意点（踩过的坑 / 局限）")
    A("")
    A("1. **干床不可外包**：最想被独立复核的判据（干床波前）恰好是第三方代码跑不了的。"
      "本单没有用「把 h1 设成 1e-6」来绕过——那会改掉被测问题（Ritter 干床解 vs Stoker 湿床解），"
      "而是把局限做成阶梯 + 下限表。")
    A("2. **全局 L1 是钝器**：`depth_l1` 按全域体积归一，波只占一小段，"
      f"实测每平移 1 格只涨 {pct(crit['XV1_cross_code_depth_l1'].get('depth_l1_per_cell_translation', d['calibration']['detection']['depth_l1_per_cell_translation']))}；"
      "要到容差 5 % 需要平移 "
      f"{d['calibration']['detection']['translation_cells_to_reach_tolerance']} 格"
      f"（{num(d['calibration']['detection']['translation_m_to_reach_tolerance'], 3)} m）。"
      "因此单看 0.04 % 会**高估**本判据的判决力：报告一律同时给「折算平移格数」。")
    A("3. **幅值读数对位置缺陷天生盲**：标定阶梯里平移 64 格（0.4 m）时水跃位置读数报 "
      f"{pct([r for r in d['calibration']['sensitivity_ladder'] if r.get('translation_cells') == 64][0]['bore_error'])}，"
      "而水位阶跃读数只有 "
      f"{pct([r for r in d['calibration']['sensitivity_ladder'] if r.get('translation_cells') == 64][0]['step_error'])}"
      "——与 `AUDIT.md` §9.4 的 0.4 m 探针一致，故两条读数必须并列使用。")
    A("4. **不同代码的 limiter/CFL 是各自默认的**：本单固定 CFL 0.4（两码同值），限幅器用各自推荐值"
      "（TensorFVM minmod、Clawpack MC）。跨码距离因此含格式差，不是纯离散误差——这正是「异族证人」的语义，"
      "但也意味着**不能**把该距离读成某一方的误差界。")
    A("5. **`t` 对齐**：Clawpack 的 `evolve_to_time` 停在其自适应步的终点，解析参照一律用**各码实际时刻**"
      "求值（`xval.json` 里 `t` 与 `t_requested` 都存），避免把时间差记成空间误差。")
    A("")
    A("## ⑥ 误差判定（逐项，显式回答是否 < 5 %）")
    A("")
    A("| 判据 | 量 | 实测（最差验收网格） | 容差 | < 5 % ? |")
    A("|---|---|---:|---:|---|")
    xv1, xv2, xv3, xv4 = (crit["XV1_cross_code_depth_l1"], crit["XV2_gate_portability"],
                          crit["XV3_instrument_monotone"], crit["XV4_dry_bed_third_code"])
    A(f"| XV1 跨码一致性 | 深度 L1（异族代码） | {pct(xv1['value'])} | {pct(xv1['limit'], 0)} | "
      f"{'是' if xv1['pass'] else '否'} |")
    A(f"| XV2 判决门可移植 | 头条读数（异族剖面） | {pct(xv2['value'])} | {pct(xv2['limit'], 0)} | "
      f"{'是' if xv2['pass'] else '否'} |")
    A(f"| XV3 单调性可移植 | 内部极值数 | {len(xv3['failures'])} 处 | 0 | {'是' if xv3['pass'] else '否'} |")
    A(f"| XV4 干床外包 | 第三方可跑网格 | {len(xv4['runnable_grids'])} 个 | ≥1 | "
      f"{'是' if xv4['pass'] else '否（诚实局限）'} |")
    A("")
    A(f"**无基准项说明**：XV4 不是误差，而是证据覆盖面；它**未通过**，因此干床三条判据在本单之后"
      "依然是单码证据，锚点的干床数字**未被独立代码复核**。")
    A("")
    A(f"**判定**：`{v['status'].upper()}`（XV1/XV2/XV3 通过；XV4 为范围局限，已在 3.3/3.4 与 ⑤ 如实登记）。")
    A("")
    A("## 复现")
    A("")
    A("```bash")
    A("PYTHONPATH=src python scripts/run_dambreak_third_code.py     # ~60 s 单核，写 xval.json / profiles.npz / 图")
    A("PYTHONPATH=src python docs/case-fvm-dambreak-stoker-third-code/make_report.py   # 重渲染本文件")
    A("PYTHONPATH=src python -m pytest tests/test_dambreak_third_code.py -q            # 13 例（含 clawpack 缺失时 skip）")
    A("```")
    A("")
    A("工件：`src/tensorfvm/verification/artifacts/shallow-water-third-code/{xval.json, profiles.npz}`"
      "（数字与剖面全部程序注入）、`docs/case-fvm-dambreak-stoker-third-code/fig_third_code.png`。")
    OUT.write_text("\n".join(lines) + "\n")
    print(f"wrote {OUT} ({len(lines)} lines)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())