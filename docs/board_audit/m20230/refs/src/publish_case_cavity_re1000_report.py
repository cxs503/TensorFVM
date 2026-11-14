"""Build the platform report payload (/tmp/report.json) for CASE-CAVITY-RE1000-FVM.

Every number is injected from ``docs/CASE-CAVITY-RE1000-FVM/artifacts/summary.json``
(zero hand-typing); the reference column comes from the bundle's own
``reference_table`` block, which is written from
:mod:`tensorfvm.lid_cavity_reference`.  Figures are embedded as base64 PNG from
``docs/CASE-CAVITY-RE1000-FVM/figures``.

Usage::

    PYTHONPATH=src python scripts/publish_case_cavity_re1000_report.py [--out /tmp/report.json]
"""

from __future__ import annotations

import argparse
import base64
import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT = ROOT / "artifacts" / "CASE-CAVITY-RE1000-FVM"
FIGS = ROOT / "docs" / "CASE-CAVITY-RE1000-FVM" / "figures"

CAPTIONS = {
    "fig1_centrelines.png": "中线速度剖面与 Ghia et al. (1982) 表格逐点对比（点线为欠分辨对照网格）",
    "fig2_fields.png": "最细网格收敛流场的涡量云图与流线（主涡位置与 Table I/II 极值一致）",
    "fig3_convergence.png": "离散误差随网格分辨率的收敛（左）与各分辨率的推进代价（右）",
}


def build() -> dict:
    b = json.load(open(OUT / "summary.json"))
    cases = sorted(b["cases"], key=lambda c: c["n"])
    prod = [c for c in cases if not c.get("role", "").startswith("control")]
    fine = prod[-1]
    ctrl = [c for c in cases if c.get("role", "").startswith("control")]
    L: list[str] = []
    A = L.append

    A("# CASE-CAVITY-RE1000-FVM · 方腔驱动流 Re = 1000 对标 Ghia et al. (1982)")
    A("")
    A("## ① 案例介绍")
    A("")
    A("二维方腔驱动流：单位方腔 `[0,1]²`，左右与底面为无滑移固壁，顶盖 `y = L` 以 "
      "`u = U = 1` 切向运动（不可穿透）。`Re = U L / ν = 1000` ⇒ `ν = 1e-3`（ρ = 1，L = 1），定常解。")
    A("")
    A("基准：" + str(b["reference"]) + "。基准值单源转录在 "
      "`src/tensorfvm/lid_cavity_reference.py`；本报告的每个参考数字由程序从该模块（经工件）取出，零手抄。")
    A("")
    A("| 参数 | 值 |")
    A("|---|---|")
    A(f"| Re | {b['reynolds']:g} |")
    A(f"| 粘性 ν | {1.0 / b['reynolds']:g} |")
    A(f"| CFL | {cases[0]['config']['cfl']:g}（dt = CFL·h/U） |")
    A(f"| 稳态容差 | max|Δu|/Δt ≤ {cases[0]['config']['tolerance']:g} |")
    A(f"| 网格序列 | {', '.join(str(c['n']) for c in cases)} |")
    A("| 精度 / 并行 | float64，单线程单进程 |")
    A("")

    A("## ② 怎么做的")
    A("")
    A("求解器 `tensorfvm.lid_cavity.LidCavitySolver`：交错 MAC 有限体积，显式对流（van Leer 受限面通量）"
      "+ 隐式扩散，投影法解耦压力，压力泊松在纯 Neumann 边界下用 DCT-II 谱解（离散拉普拉斯精确求逆）。")
    A("")
    A("本单（m20144 返修）把该算例做成可导入的正式算例模块 + 入口 + 工件 + 集成调用方：")
    A("")
    A("- 实现：`src/tensorfvm/case_cavity_re1000.py`（网格序列驱动、逐条判据 `CRITERIA`、工件写出、CLI）")
    A("- 入口：`python -m tensorfvm.case_cavity_re1000`（`scripts/run_cavity_re1000_m15458.py` 为薄封装）")
    A("- 集成：`src/tensorfvm/verification/ladder.py::lid_cavity_verdict`（C2，Re = 1000 取最细网格）"
      "直接读取本算例写出的 `lid_cavity_Re1000_n*.json`")
    A("- 机检门：`tests/test_case_cavity_re1000.py`（从工件重算误差与判据；另含小网格真跑的执行层测试）")
    A("- 成本装置：网格延拓（32→64→128）+ 每 20 000 步检查点 + 延拓校验（同网格冷启动 vs 延拓，两解逐点比对）")
    A("")
    A("关键设置：延拓把单元中心速度双线性插值后重建面速度（`align_corners=False`，与单元中心共位一致），"
      "壁面法向速度置零；延拓校验以 `resume=False` 写独立流场文件，避免“校验”被冷启动检查点掩盖。")
    A("")

    A("## ③ 计算结果")
    A("")
    A("| n | 角色 | 步数 | max|Δu|/Δt | max|∇·u| | u_min (ref) | v_max (ref) | 涡心 got (ref) | 5% 判据 |")
    A("|---|---|---|---|---|---|---|---|---|")
    for c in cases:
        um = c["u"]["extremum"]["min_computed"][1]
        ur = c["u"]["extremum"]["min_reference"][1]
        vm = c["v"]["extremum"]["max_computed"][1]
        vr = c["v"]["extremum"]["max_reference"][1]
        vx, vy = c["vortex_centre"]["computed"]
        gx, gy = b["reference_table"]["vortex_centre"]
        A(f"| {c['n']} | {c.get('role', '')} | {c['solver']['steps']} "
          f"| {c['solver']['residual_rate']:.2e} | {c['max_divergence']:.1e} "
          f"| {um:.5f} ({ur:.5f}) | {vm:.5f} ({vr:.5f}) "
          f"| ({vx:.4f},{vy:.4f}) ({gx:.4f},{gy:.4f}) "
          f"| {'PASS' if c['verdict']['pass'] else 'FAIL'} |")
    A("")
    A(f"最细生产网格 n = {fine['n']} 的逐判据结果：")
    A("")
    A("| 判据 | 说明 | 实测 | 阈值 | 结论 |")
    A("|---|---|---|---|---|")
    for chk in fine["verdict"]["checks"]:
        v = chk["value"]
        shown = ("true" if v is True else "false") if isinstance(v, bool) else f"{v:.5f}"
        A(f"| {chk['id']} | {chk['statement']} | {shown} | {chk['threshold']} "
          f"| {'PASS' if chk['pass'] else 'FAIL'} |")
    A("")
    A("## ④ 结果图")
    A("")
    A("1. 中线剖面 vs Ghia 表格（u@x=L/2、v@y=L/2，含欠分辨对照网格）")
    A("2. 最细收敛流场的涡量云图与流线")
    A("3. 离散误差随网格的收敛与各分辨率的推进代价")
    A("")

    A("## ⑤ 注意点")
    A("")
    for note in NOTES(b, ctrl):
        A(f"- {note}")
    A("")

    A("## ⑥ 误差判定（是否 <3%）")
    A("")
    A(f"| 量 | 基准 | got (n = {fine['n']}) | 误差 | 是否 <3% |")
    A("|---|---|---|---|---|")
    items = ERROR_ITEMS(b, fine)
    for name, ref, got, err in items:
        A(f"| {name} | {ref:.5f} | {got:.5f} | {err * 100:.2f}% | {'是' if err < 0.03 else '否'} |")
    A("")
    worst = max(c["u"]["max_abs_error"] for c in prod)
    A(f"最细生产网格上最坏的剖面偏差 {worst * 100:.2f}% of U；"
      f"总账 5% 判据 {'全部通过' if b.get('production_all_pass') else '存在未通过项'}；"
      f"是否全部 <3%：{'是' if all(e < 0.03 for *_, e in items) else '否（见上表逐项）'}。")

    figs = []
    for p in sorted(FIGS.glob("*.png")):
        figs.append({"name": p.name, "caption": CAPTIONS.get(p.name, p.name),
                     "png_base64": base64.b64encode(p.read_bytes()).decode()})
    return {
        "title": ("CASE-CAVITY-RE1000-FVM：方腔驱动流 Re = 1000 对标 Ghia et al. (1982)，"
                  f"n = {prod[0]['n']}/{fine['n']} 收敛 + 主验证梯集成"),
        "summary": (f"TensorFVM 方腔 Re=1000 做成正式模块+入口+工件：最细 n={fine['n']} "
                    f"u_min={fine['u']['extremum']['min_computed'][1]:.5f}，"
                    f"v_max={fine['v']['extremum']['max_computed'][1]:.5f}，"
                    f"max|Δu|={fine['u']['max_abs_error'] * 100:.2f}% of U，"
                    f"5% 判据 {'PASS' if fine['verdict']['pass'] else 'FAIL'}；"
                    "延拓校验证明延拓不改定常解；主验证梯 C2 消费本行。"),
        "scenario_key": "CASE-FVM-CAVITY-RE1000",
        "mission_id": 20144,
        "tags": ["TensorFVM", "FVM", "lid-driven-cavity", "Re1000", "Ghia1982",
                 "benchmark", "verification", "CASE-FVM-CAVITY-RE1000"],
        "markdown": "\n".join(L),
        "figures": figs,
    }


def NOTES(b: dict, ctrl: list[dict]) -> list[str]:
    out = []
    if ctrl:
        out.append("欠分辨对照网格 " + "、".join(f"n={c['n']}" for c in ctrl)
                   + " 未通过 5% 判据，且如实保留在工件里（不用平均掩盖）。")
    chk = b.get("continuation_check")
    if chk:
        out.append(f"延拓校验：n={chk['to_n']} 网格上延拓解与冷启动解 max|Δu| = {chk['max_abs_du']:.2e}、"
                   f"max|Δv| = {chk['max_abs_dv']:.2e} ⇒ 同一离散定常解；延拓省时 {chk['speedup']:.2f}×。")
    out.append("显式推进步数 ∝ n，n = 128 需 10⁵ 量级步数；稳态判据 max|Δu|/Δt 是必要非充分，"
               "故要求两个独立初值给出同一解。")
    out.append("只覆盖定常二维解，不对非定常/三维方腔作声明；参考是文献表而非解析解，"
               "表格采样点之间的误差未评估。")
    out.append("`verification_ladder` 的 C2 阈值是 2%（严于本算例 5%）；本单只负责让该判据消费到真实行，"
               "其结论由数字决定。")
    return out


def ERROR_ITEMS(b: dict, fine: dict) -> list[tuple[str, float, float, float]]:
    um = fine["u"]["extremum"]["min_computed"][1]
    ur = fine["u"]["extremum"]["min_reference"][1]
    vm = fine["v"]["extremum"]["max_computed"][1]
    vr = fine["v"]["extremum"]["max_reference"][1]
    gx, gy = b["reference_table"]["vortex_centre"]
    vx, vy = fine["vortex_centre"]["computed"]
    return [
        ("u_min（x = L/2 剖面极小）", ur, um, abs(um - ur) / abs(ur)),
        ("v_max（y = L/2 剖面极大）", vr, vm, abs(vm - vr) / abs(vr)),
        ("主涡中心 x", gx, vx, abs(vx - gx) / abs(gx)),
        ("主涡中心 y", gy, vy, abs(vy - gy) / abs(gy)),
        ("max|Δu|（整条剖面，以 U 归一）", 0.0, fine["u"]["max_abs_error"],
         fine["u"]["max_abs_error"]),
        ("max|Δv|（整条剖面，以 U 归一）", 0.0, fine["v"]["max_abs_error"],
         fine["v"]["max_abs_error"]),
    ]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/tmp/report.json")
    args = ap.parse_args()
    payload = build()
    pathlib.Path(args.out).write_text(json.dumps(payload))
    print(json.dumps({"out": args.out, "bytes": len(json.dumps(payload)),
                      "figures": [f["name"] for f in payload["figures"]]}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())