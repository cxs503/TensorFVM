"""Build /tmp/report.json for the platform report endpoint (numbers from the artifact)."""

import base64
import json

ROOT = __file__.rsplit("/docs/", 1)[0]
d = json.load(open(f"{ROOT}/docs/case-fvm-dambreak-stoker/artifacts/reproduction.json"))
png = base64.b64encode(
    open(f"{ROOT}/docs/case-fvm-dambreak-stoker/crosscheck.png", "rb").read()).decode()


def pct(v):
    return "—" if v is None else f"{100 * v:.3f} %"


L = []
L.append("## 1. 案例介绍\n")
L.append("浅水方程（深度平均、静水压、无粘）溃坝黎曼问题，节点 `CASE-FVM-DAMBREAK-STOKER`。"
         "几何为 1-D 平底（**无船体几何，故不适用 Cb/L/B 前置校验**），计算域 [-4, 6] m，两端透射边界。")
L.append("\n| 参数 | Ritter 干床 | Stoker 湿床 |\n|---|---|---|\n| h0 | 1.0 m | 1.0 m |\n"
         "| h1 | 0 m | 0.5 m |\n| g | 9.81 m/s² | 9.81 m/s² |\n| t* = t√(g/h0) | 2.0 | 2.0 |\n"
         "| t | 0.6386 s | 0.6386 s |")
b = d["runs"]["stoker-wet"]["1600"]["bounds"]
L.append(f"\nStoker 水跃态（闭式）：h2 = {b['h2']:.5f} m，u2 = {b['u2']:.5f} m/s，"
         f"U = {b['shock']:.5f} m/s，h2/h1 = {b['h2'] / 0.5:.5f}。\n")
L.append("## 2. 怎么做的\n")
L.append("被检对象：`swe1d.ShallowWater1D`（单元中心 FV、MUSCL+minmod、HLL、SSP-RK3、CFL=0.4、平板无摩阻）。"
         "本轮新增 `shallow_water/crosscheck.py`，给出与既有 `dam_break` **不同的第二条度量路径**：\n"
         "1. 水跃态用**三条互不相同的求解路径**互证：A 对 c2 二分（既有 `exact`）、"
         "B 对 r=h2/h1 的带括区间 Newton、C 对 u2 二分（驱动脚本内）；\n"
         "2. 波速/位置用**固定深度分数探针**：扇内闭式反演 ξ(h)=2c0−3√(gh)，探针深度取 "
         "h0−f(h0−h2)（f = 2 % / 50 %）与水跃落差中点，与 dx 无关；\n"
         "3. 误差按相似坐标分区（上游/扇/平台/下游）给相对 L1 与绝对 L∞；L∞ 只报告不判定"
         "（捕捉间断必然带 O(dx) 误差，用 5 % 判它只能凑系数）；\n"
         "4. 质量守恒用闭式解析积分（分段多项式，无求积误差）+ 求解器体积漂移双向核对。\n"
         "工具链：`verify_reference.py`（认证 + 交叉校验 + 写 `artifacts/reproduction.json` + 注入报告表格）、"
         "`make_figure.py`（仅由 JSON 作图）、`pytest tests/test_shallow_water_analytic_crosscheck.py`。\n")
L.append("## 3. 计算结果\n")
ref = d["reference"]
n_ref = (len(ref["cases"]["stoker-wet"]["checks_relative"])
         + len(ref["cases"]["ritter-dry"]["checks_relative"]))
worst3 = max(max(r["h2_newton_rel"], r["h2_velocity_rel"], r["shock_velocity_rel"])
             for r in ref["bore_sweep"])
L.append(f"参考自证（容差 1e-12）：两条闭式解共 {n_ref} 项校核，最大相对残差 "
         f"{ref['worst_relative']:.2e}；三路径（A/B/C）在 7 组 h1/h0 上一致到 {worst3:.1e}。\n")
L.append("| 情形 | n | 剖面 L1 | L∞ | fan-mid 探针 | 水跃/波前探针 | 平台深度 | 平台速度 | 体积漂移 | 5 % 判定 |")
L.append("|---|---|---|---|---|---|---|---|---|---|")
for key in ("stoker-wet", "ritter-dry"):
    for n in ("200", "400", "800", "1600"):
        r = d["runs"][key][n]
        m = r["metrics"]
        pl = m.get("plateau") or {}
        pr = m["probes"]
        hl = "bore" if "bore" in pr else "front"
        L.append(f"| {key} | {n} | {pct(m['depth_l1'])} | {pct(m['depth_linf'])} | "
                 f"{pct(pr['fan_mid']['rel_error'])} | {pct(pr[hl]['rel_error'])} | "
                 f"{pct(pl.get('depth_rel_error'))} | {pct(pl.get('velocity_rel_error'))} | "
                 f"{m['volume_drift']:.1e} | {'✅' if r['verdict']['pass'] else '❌'} |")
L.append("\n网格收敛（拟合观测阶）：" + "；".join(
    f"{k}: L1 {v['observed_order_l1']:.2f}, fan-mid {v['observed_order_fan_mid']:.2f}, "
    f"接受网格通过 {v['accepted_pass']}" for k, v in d["convergence"].items()) + "。\n")
L.append("## 4. 结果图\n")
L.append("crosscheck.png：**(a)** Stoker 深度剖面（数值 n=1600 vs 闭式解，四区着色 + 固定分数探针）；"
         "**(b)** 速度剖面与平台 u2；**(c)** Ritter/Stoker 的 L1 与 fan-mid 误差 log-log 收敛"
         "（标注观测阶与 5 % 门限）；**(d)** 相似坐标下的误差场 h_num − h_exact，"
         "误差集中在波头折点与水跃。\n")
L.append("## 5. 注意点\n")
L.append("(i) 扇内探针必须落在 (h2, h0)：初版取 0.5h0 时湿床扇内根本不穿过该深度，探针静默锁到水跃上"
         "（实测误差 1009 %），现由 h0−f(h0−h2) 生成并有专项测试守住；"
         "(ii) 跨水跃深度是**下降**穿越，方向写反会取到扇内同名深度；"
         "(iii) 探针是单元中心线性插值，光滑区下限 O(dx²)（n=400 约 3e-6），跨间断探针下限约 "
         "dx/(2Ut)（n=400 约 0.6 %），属测量下限而非格式误差，已在文档与测试中显式给出；"
         "(iv) 粗网格 n ≤ 400 明确失败，不插值绕过；"
         "(v) 适用范围仅 1-D 平底无粘无摩阻，2-D/底坡/摩阻另有专账。\n")
L.append("## 6. 误差判定（是否 < 3 %）\n")
rows, bad = [], []
for key in ("stoker-wet", "ritter-dry"):
    m = d["runs"][key]["800"]["metrics"]
    pl = m.get("plateau") or {}
    items = [("depth_l1", m["depth_l1"]), ("fan_head", m["probes"]["fan_head"]["rel_error"]),
             ("fan_mid", m["probes"]["fan_mid"]["rel_error"])]
    for nm in ("front", "bore"):
        if nm in m["probes"]:
            items.append((nm, m["probes"][nm]["rel_error"]))
    if pl.get("h_exact"):
        items += [("plateau_depth", pl["depth_rel_error"]),
                  ("plateau_velocity", pl["velocity_rel_error"])]
    for nm, v in items:
        ok = v < 0.03
        rows.append(f"| {key} | {nm} | {pct(v)} | {'✅ < 3 %' if ok else '❌ ≥ 3 %'} |")
        if not ok:
            bad.append(f"{key}/{nm} = {pct(v)}")
L.append("接受网格 n = 800 上逐项（基准为三路径互证的闭式解，参考值不确定度约 1e-15）：\n")
L.append("| 情形 | 关键量 | 相对误差 | 对 3 % 判据 |\n|---|---|---|---|")
L += rows
L.append("\n**回答**：不满足 3 % 的项只有 " + "；".join(bad) + "（仍满足 5 % 任务门限），"
         "该量在 n=1600 降到 "
         + pct(d["runs"]["stoker-wet"]["1600"]["metrics"]["probes"]["fan_head"]["rel_error"])
         + " 并满足 3 %。**因此不能笼统宣称“全部 < 3 %”**：其余关键量均优于 0.1 %。\n")

report = {
    "title": "TensorFVM 浅水溃坝 Stoker/Ritter 解析解的独立交叉校验（CASE-FVM-DAMBREAK-STOKER）",
    "summary": "新增 crosscheck 模块：三路径水跃态互证 + 固定深度分数探针 + 相似坐标分区误差；"
               "接受网格 n≥800 全部满足 5 % 门限，n=800 上 10 项关键量 9 项 <3 %"
               "（fan_head 3.05 % 破 3 %，n=1600 降到 1.56 %）。",
    "scenario_key": "case-fvm-dambreak-stoker",
    "mission_id": 18685,
    "tags": ["TensorFVM", "shallow-water", "dam-break", "Stoker", "Ritter",
             "analytic-crosscheck", "verification"],
    "markdown": "\n".join(L),
    "figures": [{"name": "crosscheck.png",
                 "caption": "(a) Stoker 深度剖面 vs 闭式解 + 固定分数探针；(b) 速度与平台 u2；"
                            "(c) L1/fan-mid 收敛与 5 % 门限；(d) 相似坐标误差场",
                 "png_base64": png}],
}
json.dump(report, open("/tmp/report.json", "w"), ensure_ascii=False)
print("wrote /tmp/report.json", len(open("/tmp/report.json", "rb").read()), "bytes")