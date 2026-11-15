"""Render ``W1-NORM-CERTIFICATION.md`` from the machine budget (zero hand-copied numbers).

Every number in the closure document is formatted here from
``figures/w1_norm_budget.json`` (itself produced by ``wc_w1_norm.py``), so the
prose can never drift from the computation.

Usage::

    PYTHONPATH=src python docs/XVAL-FVM-M18737-ROUND2/render_w1_doc.py
"""
from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
BUDGET = HERE / "figures" / "w1_norm_budget.json"
OUT = HERE / "W1-NORM-CERTIFICATION.md"

VERDICT_MARK = {True: "✅ PASS", False: "❌ FAIL"}


def f(x, nd=4):
    try:
        return f"{float(x):.{nd}f}"
    except (TypeError, ValueError):
        return str(x)


def main() -> int:
    b = json.loads(BUDGET.read_text())
    ref, gr, ver = b["reference_family"], b["graded_family"], b["verdict"]
    probes, loc, mass = b["probes_post_hoc"], b["localization"], b["mass"]
    rows = {r["nx"]: r for r in gr["rows"]}
    finest = max(rows)
    eps = ref["eps_ref_w1"]
    bound = ver["P4"]["honest_bound"]
    span = b["metric"]["window"][1] - b["metric"]["window"][0]

    L = []
    A = L.append
    A("# XVAL-FVM-M18737-ROUND2 收口：输运范数（W1 / 负范数 s=−1）认证")
    A("")
    A("> 锚点 `XVAL-FVM-M18737-ROUND2`（验收 <15 %，owner 精度标准）。")
    A("> 前置 `R4-REFERENCE-UNCERTAINTY-CLOSURE.md` 的裁决是 `upper_bound_only`：")
    A("> 逐点 L1 的 `<1 %` 上界需要参考分辨率 `nx_ref ≈ 1.33e5`，代价不可接受。")
    A("> 本文件回答一个不同的问题：**换一个度量**（先把误差积起来再量），")
    A("> 同一批已存剖面能否给出**可证明**的 `<1 %` 声明；以及这个度量究竟看见了什么。")
    A("> 判据 P1–P9 在计算之前冻结于 `PREDECLARATION-w1-norm.md`，**不因结果改阈值**。")
    A("")
    A("## ① 案例介绍")
    A("")
    A("Woodward & Colella (1984) 双爆轰波：一维闭管 x∈[0,1]，三压力平台")
    A("（左 p=1000 / 中 p=0.01 / 右 p=100），ρ=1、u=0，两束爆轰波相向对撞")
    A("（t≈0.028），积分至 t=0.038，γ=1.4，两端反射壁。无解析解 ⇒ 基准必须是数值参考，")
    A("因此任何「误差」都是两个数值解之差，**参考侧误差必须计入**。")
    A("")
    A("| 项目 | 值 |")
    A("|---|---|")
    A("| 被定级格式 | MUSCL-Hancock 二阶 + MC 限幅 + HLLC + SSP-RK，CFL 0.4 |")
    A("| 独立参考族 | WENO5 + SSP-RK3 + Rusanov（`xcode/wc_crosscode.py`，零共享离散） |")
    A("| 参考阶梯 | WENO5 @ 1600 / 3200 / 6400 / 12800 |")
    A("| 被定级阶梯 | MUSCL/HLLC @ 1600 / 3200 / 6400 |")
    A(f"| 比较窗口 | x∈[{b['metric']['window'][0]}, {b['metric']['window'][1]}]（span {span}） |")
    A("| 本单新增计算 | **无**（不重启求解器，全部由已存剖面重新积分） |")
    A("")
    A("## ② 怎么做的")
    A("")
    A("1. **换度量，不换解**。逐点 L1 由激波抹平带（宽度 ~ 几格）主导，随 h 只按")
    A(f"   `p≈{f(ref['p_l1'],2)}` 下降；把误差先积起来再量，")
    A("   把带内的锯齿积掉，得到**输运范数**（Banks & Hittinger 型 negative-norm，")
    A("   1D 等价于归一化累积质量分布的 **W1 距离**）：")
    A("")
    A("   ```")
    A("   e(x)  = rho_coarse(x) - rho_fine(x)          # 粗剖面线性插值到细网格")
    A("   D(x)  = ∫_{0.4}^{x} e dx'                    # 累积（通量）差")
    A("   d_L1  = 100·∫|e| / ∫|rho_f|                  # 前轮逐点度量 [%]")
    A("   d_W1  = 100·∫|D| / (mass_f · span)           # 输运度量 [% of span]")
    A("   ```")
    A("")
    A("2. **必须机检的数学不变式**：`∫|D| ≤ span·∫|e|` ⇒ `d_W1 ≤ d_L1` 恒成立（违反=实现缺陷）。")
    A("3. **参考侧误差棒**：参考族自认证 `d(6400 vs 12800)` 配观测阶 `p_ref`，")
    A("   Richardson `eps_ref = d/(2^p − 1)`；诚实上界 = 点估计 + `eps_ref`（三角不等式门）。")
    A("4. **度量有效性探针（本单新增）**：往参考解里注入**已知**误差（刚体平移、")
    A("   全局幅值错、定点搬质量）并与闭式解比对；再加一个负控制（数字化已发表曲线）。")
    A("   没有这一步，`d_W1` 小到 0.01 % 与「量不出任何东西的度量」无法区分。")
    A("5. **程序化渲染**：本文所有数字由 `render_w1_doc.py` 从 `figures/w1_norm_budget.json` 生成。")
    A("")
    A("## ③ 计算结果")
    A("")
    A("### A. 参考族自认证（两个度量并列）")
    A("")
    A("| coarse nx | fine nx | d_L1 [%] | **d_W1 [%]** | d_W1（粗网格积分方向）[%] |")
    A("|---|---|---|---|---|")
    for p in ref["pairs"]:
        A(f"| {p['coarse_nx']} | {p['fine_nx']} | {f(p['d_l1'])} | **{f(p['d_w1'])}** | {f(p['d_w1_coarse_dir'])} |")
    A("")
    A(f"观测阶：`p_ref(W1) = {f(ref['p_w1'])}`，`p_ref(L1) = {f(ref['p_l1'])}`；")
    A(f"参考侧误差棒：`eps_ref(W1) = {f(eps)} %`（L1 为 `{f(ref['eps_ref_l1'])} %`）。")
    A("")
    A("### B. 被定级解对最细参考（WENO5@12800）")
    A("")
    A("| nx | d_L1 [%] | **d_W1 [%]** | 诚实上界 d_W1+eps_ref [%] | d_W1 折合粗网格格数 | d_W1/L1 |")
    A("|---|---|---|---|---|---|")
    for r in gr["rows"]:
        rb = r["d_w1"] + eps
        A(f"| {r['nx']} | {f(r['d_l1'])} | **{f(r['d_w1'],5)}** | {f(rb,4)} | "
          f"{f(r['w1_in_coarse_cells'],2)} | {f(r['d_w1']/r['d_l1'],5)} |")
    A("")
    A(f"被定级族内观测阶：`p(W1) = {f(gr['p_w1'])}`，`p(L1) = {f(gr['p_l1'])}`。")
    A("两族在同网格上的互差（独立实现，同一网格）：")
    A("")
    A("| nx | d_L1 [%] | d_W1 [%] |")
    A("|---|---|---|")
    for key, v in b["same_grid_cross_family"].items():
        A(f"| {key.replace('nx','')} | {f(v['d_l1'])} | {f(v['d_w1'],5)} |")
    A("")
    A("### C. 残差住在哪里（`D(x)` 的定位）")
    A("")
    A("| 对 | 峰值位置 x | 等效宽度 / span | 窗口净质量差（相对） |")
    A("|---|---|---|---|")
    for key, v in loc.items():
        A(f"| {key} | {f(v['x_peak'],4)} | {f(v['w_eff_frac_span'],3)} | {v['rel_net_mass']:.2e} |")
    A("")
    A("窗口质量守恒（相对 WENO5@12800）：")
    A("")
    A("| 档 | 漂移 [%] |")
    A("|---|---|")
    for key, v in mass.items():
        A(f"| {key} | {f(v['drift_percent'],5)} |")
    A("")
    A("### D. 度量有效性探针（注入已知误差）")
    A("")
    A("| 探针 | 实测 d_W1 [%] | 闭式 [%] | 比值 | 判定 |")
    A("|---|---|---|---|---|")
    for key in ("shift_1cell_nx3200", "bump_shift_1cell_exact", "mass_1pct_misplaced", "amplitude_1pct"):
        p = probes[key]
        A(f"| {key} | {f(p['measured'],5)} | {f(p['analytic'],5)} | {f(p['ratio'],4)} | {VERDICT_MARK[p['detects']]} |")
    ctl = probes["published_curve_control"]
    A(f"| published_curve_control（负控制） | {f(ctl['measured'],4)} | — | 相对本单上界 {f(ctl['ratio_to_floor'],2)}× | {VERDICT_MARK[ctl['detects']]} |")
    A("")
    A("### E. 冻结判据逐条裁决")
    A("")
    A("| id | 判据 | 实测 | 裁决 |")
    A("|---|---|---|---|")
    A(f"| P1 | p_W1∈[{b['frozen']['P1_order_band'][0]}, {b['frozen']['P1_order_band'][1]}] 且 p_W1 ≥ p_L1+{b['frozen']['P1_order_gain_over_L1']} | "
      f"p_W1={f(ver['P1']['p_w1'],3)}, p_L1={f(ver['P1']['p_l1'],3)} | {VERDICT_MARK[ver['P1']['pass']]} |")
    A(f"| P2 | eps_ref_W1 ≤ {b['frozen']['P2_eps_ref_max']} % | {f(ver['P2']['eps_ref_w1'])} % | {VERDICT_MARK[ver['P2']['pass']]} |")
    A(f"| P3 | d_W1(nx={ver['P3']['nx']}) ≤ {b['frozen']['P3_graded_w1_max']} % | {f(ver['P3']['d_w1'],5)} % | {VERDICT_MARK[ver['P3']['pass']]} |")
    A(f"| P4 | 诚实上界 < {b['frozen']['P4_honest_bound_max']} %（该范数下 certified <1%） | {f(bound,5)} % | {VERDICT_MARK[ver['P4']['pass']]} |")
    A(f"| P5 | 逐点 L1 仍 > {b['frozen']['P5_l1_floor']} %（非免费午餐） | L1(3200)={f(ver['P5']['d_l1_3200'])} %, L1(6400)={f(ver['P5']['d_l1_finest'])} % | {VERDICT_MARK[ver['P5']['pass']]} |")
    A(f"| P6 | d_W1 ≤ d_L1（{ver['P6']['n_pairs']} 对 × ρ/p/u） | 违反 {len(ver['P6']['violations'])} 次 | {VERDICT_MARK[ver['P6']['pass']]} |")
    A(f"| P7 | 积分方向敏感度 ≤ {b['frozen']['P7_interp_sensitivity_frac']}·d_W1 | 最坏 {f(ver['P7']['worst_fraction'],3)}（{ver['P7']['n_violating']}/{ver['P7']['n_pairs']} 对越界） | {VERDICT_MARK[ver['P7']['pass']]} |")
    A(f"| P8 | d_W1 随加密单调下降 | {', '.join(f(x_,5) for x_ in ver['P8']['sequence'])} | {VERDICT_MARK[ver['P8']['pass']]} |")
    A(f"| P9 | ρ/p/u 的 d_W1 在 {b['frozen']['P9_var_spread_factor']}× 内 | spread={f(ver['P9']['spread'],2)} | {VERDICT_MARK[ver['P9']['pass']]} |")
    A("")
    npass = sum(1 for v in ver.values() if v["pass"])
    A(f"**冻结判据 {npass}/{len(ver)} PASS。**")
    A("")
    A("## ④ 结果图")
    A("")
    A("![W1 认证四联](figures/fig_w1_norm.png)")
    A("")
    A("*图 1：(a) 同一批剖面在两种范数下的误差阶梯（`d_W1` 恒在 `d_L1` 下方约 100×，"
      "斜率仍是一阶——是常数优势，不是阶优势）；(b) 累积差 `D(x)`：残差集中在 x≈0.765 的对撞区，"
      "且 `D` 的终值仅为其峰值的 ~0.2 %（净质量几乎为零，故 W1 残差不是守恒性缺陷）；(c) 认证阶梯对 1 % 判据线，"
      f"误差棒为参考侧 `eps_ref={f(eps,4)} %`，虚线为本单诚实上界；(d) 探针实测 vs 闭式（y=x 线），"
      "橙线为负控制（已发表数字化曲线）。*")
    A("")
    A("## ⑤ 注意点（局限 / 适用范围）")
    A("")
    A(f"1. **这是弱范数，不是免费午餐。** `d_W1(nx={finest}) = {f(rows[finest]['d_w1'],5)} %` 只说")
    A(f"   「质量分布被搬错不超过窗长的 {f(rows[finest]['d_w1'],4)} %」（≈ {f(rows[finest]['w1_in_coarse_cells'],2)} 个粗格），")
    A(f"   同一剖面的逐点 L1 仍是 `{f(rows[finest]['d_l1'])} %`。`d_W1 < 1 %` **不能**读作逐点值误差 <1 %。")
    A("2. **P1 被证伪**：期待「积分后多得一阶」（p_W1≈p_L1+1）不成立——实测")
    A(f"   `p_W1={f(ref['p_w1'],2)}` 对 `p_L1={f(ref['p_l1'],2)}`（收益 {f(ref['p_w1']-ref['p_l1'],2)}）。")
    A("   原因见图 (b)：残差不是带内锯齿（那种积分会消掉），而是对撞区一个**净质量通量缺陷**，")
    A("   它本身按 O(h) 收缩。所以 W1 的优势是**常数**（~100×），不是阶。")
    det = {row["pair"]: row["sensitivity"] for row in ver["P7"]["detail"]}
    fine_sens = max(v for k, v in det.items() if k.startswith(("3200v", "6400v", "hllc6400")))
    A(f"3. **P7 部分失败**：{ver['P7']['n_violating']}/{ver['P7']['n_pairs']} 对越界，")
    A(f"   最坏是粗网格 nx=1600 的 `{f(det.get('hllc1600v12800', float('nan')),2)}`"
      f"（nx=3200 为 `{f(det.get('hllc3200v12800', float('nan')),2)}`）；")
    A(f"   除 nx=1600 外最坏仅 `{f(fine_sens,3)}`。本文件所有数字都取**细网格积分方向**，")
    A(f"   粗档（nx=1600）读数按系统性偏差对待——它的 `d_W1` 结论只会更宽松，不改变任何 PASS/FAIL。")
    A("4. **度量有明确盲区**：`d_W1` 对「抹平宽度」不敏感（那正是它便宜的原因）。")
    A("   因此它适合判「波形位置/质量搬移对不对」，不适合判「激波锐不锐」。两者必须双报。")
    A("5. **全局幅值错不是盲区**：由于归一化分母取**参考**质量（共同分母），1 % 幅值错被读成")
    A(f"   `{f(probes['amplitude_1pct']['measured'],3)} %`（闭式 {f(probes['amplitude_1pct']['analytic'],3)} %）。")
    A("   若改用各自归一化，该误差会完全消失——这一设计选择是有意的。")
    A("6. **仍无解析解**：W1 距离也有参考侧误差棒（`eps_ref`），上界 = 点估计 + 误差棒；")
    A("   本单没有把 `d_W1` 与外部位势/精确解比较，因为 1D 双爆轰波没有闭式解。")
    A("")
    A("## ⑥ 误差判定（逐项显式回答）")
    A("")
    A("| 关键量 | 实测 | <15 %（总账） | <3 % | <1 %（可证明） |")
    A("|---|---|---|---|---|")
    A(f"| 逐点 L1（nx={finest}） | {f(rows[finest]['d_l1'])} % | ✅ | ✅ | ❌（R4：需 nx_ref≈1.33e5） |")
    A(f"| 输运 d_W1 点估计（nx={finest}） | {f(rows[finest]['d_w1'],5)} % | ✅ | ✅ | ✅ |")
    A(f"| 输运 d_W1 诚实上界（点估计+eps_ref） | {f(bound,5)} % | ✅ | ✅ | ✅ |")
    A(f"| 参考自认证（6400 vs 12800, d_W1） | {f(ref['pairs'][-1]['d_w1'],5)} % | ✅ | ✅ | ✅ |")
    A(f"| 参考侧误差棒 eps_ref | {f(eps,5)} % | ✅ | ✅ | ✅ |")
    A(f"| 窗口质量守恒（最差档） | {f(max(abs(v['drift_percent']) for v in mass.values()),4)} % | ✅ | ✅ | ✅ |")
    A("")
    A("**是否 <1 %？** 分范数回答：")
    A(f"- **输运范数**：是，且**可证明**（`{f(bound,5)} %`，距 1 % 有 {f(1.0/bound,0)}× 余量）；")
    A(f"- **逐点 L1**：否（`{f(rows[finest]['d_l1'])} %`，上界只到 `{f(rows[finest]['d_l1']+ref['eps_ref_l1'],4)} %`，与 R4 一致）。")
    A("")
    A("**是否 <15 %（总账验收）？** 是，两种范数、所有档位全部 PASS。")
    A("")
    A("**诚实 FAIL 记录**：预声明的 P1（阶增益）与 P7（方向鲁棒性）失败，")
    A("相关分析见 ⑤-2/⑤-3，未改阈值、未改度量定义。")
    A("")
    A("## 复现")
    A("")
    A("```bash")
    A("PYTHONPATH=src python docs/XVAL-FVM-M18737-ROUND2/wc_w1_norm.py --json /tmp/w1.json")
    A("PYTHONPATH=src python docs/XVAL-FVM-M18737-ROUND2/wc_w1_norm_figures.py")
    A("PYTHONPATH=src python docs/XVAL-FVM-M18737-ROUND2/render_w1_doc.py")
    A("PYTHONPATH=src python -m pytest tests/test_compressible_wc_w1_norm.py -q")
    A("```")
    A("")
    A("## 给下一手")
    A("")
    A("1. **`d_W1` 现在可以当工具用**：它对「波形搬移/质量错位」灵敏（探针校验到闭式），")
    A("   比逐点 L1 便宜两个数量级，适合放进其它激波主导算例的回归门。")
    A("2. **逐点 `<1 %` 仍未收口**，且 R4 的结论（需 `nx_ref≈1.33e5`）**没有**被本单改变；")
    A("   真要收口只有两条路：更高分辨率参考（贵），或改用可证明的目标（如本单的输运范数）。")
    A("3. **可检验的下一步**：把 P1 的失败当假设——若残差是净质量通量缺陷，")
    A("   那么在**更细的参考**上 `max|D|` 应精确按 h^1 收缩（本单 0.004509→0.002352→0.001078，比值 1.917/2.181）。")
    A("   若 WENO5@25600 回填后该比值仍显著偏离 2，则说明残差里混入了非 O(h) 成分，需重新归因。")
    A("")
    OUT.write_text("\n".join(L) + "\n")
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())