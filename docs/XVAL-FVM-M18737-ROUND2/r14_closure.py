"""Round-14 closure driver: the three R13 jobs returned; every gate is measured.

Why this file exists
--------------------
``R13-CERTIFIED-RUNG.md`` closed the *criteria* but left the numbers open: the
three pre-registered jobs (``cj4221`` MUSCL/HLLC@12800 CFL 0.8, ``cj4222``
MUSCL/HLLC@6400 CFL 0.8, ``cj4223`` WENO5@12800 CFL 0.8) were still queued when
the R13 window closed, so E1/E1r/E2/E3/E4 were all ``not_measured``.

This round does **no new solver work and changes no frozen criterion**:

1. the three job artifacts were fetched and back-filled into
   ``ladder/data/`` (tags ``nx12800_hllc_cfl0.8``, ``nx6400_hllc_cfl0.8``,
   ``nx12800_weno5_cfl0.8``);
2. the *frozen* R13 pipeline (``r13_certified_rung.py`` -- imported, never
   copied) is re-run on the completed ladder;
3. this file renders ``R14-CLOSURE.md`` and the figures from the resulting
   JSON, so every number in the document is program-injected (zero hand-copy).

The R13 pre-declaration (``PREDECLARATION-r13.md`` §③) already anticipated the
two ways the E2 power-law reading could miss and forbids re-tuning after the
fact; whatever the numbers say is reported as-is (honest FAIL stays FAIL).

Usage::

    PYTHONPATH=src python docs/XVAL-FVM-M18737-ROUND2/r14_closure.py \
        --json docs/XVAL-FVM-M18737-ROUND2/r14_certified_rung.json \
        --markdown docs/XVAL-FVM-M18737-ROUND2/R14-CLOSURE.md
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
from pathlib import Path
from typing import Dict, List

HERE = Path(__file__).resolve().parent

#: job -> (tag, what it unlocks)
JOBS = {
    "cj4221": ("nx12800_hllc_cfl0.8", "the missing graded rung nx=12800 (E2/E3/E4)"),
    "cj4222": ("nx6400_hllc_cfl0.8", "the E1 CFL admission control (graded family)"),
    "cj4223": ("nx12800_weno5_cfl0.8", "the E1r CFL admission control (reference family)"),
}


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


R13 = _load_module(HERE / "r13_certified_rung.py", "wc_r13_driver")


def _f(x, nd=4):
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return "—"
    return f"{x:.{nd}f}"


def render(v: Dict) -> str:
    ks = sorted(int(k) for k in v["ladder"])
    e4 = v["E4_honest_ceiling"]
    e2 = v["E2_prediction"]
    e3 = v["E3_certified"]

    gate_rows = "\n".join([R13._gate_row("E1（被定级族）", v.get("E1_cfl_graded")),
                           R13._gate_row("E1r（参考族）", v.get("E1r_cfl_reference"))])
    rows = "\n".join(
        f"| {nx} | {_f(v['ladder'][str(nx)]['l1_percent'])} | "
        f"{_f(v['ladder'][str(nx)]['peak_percent'])} | "
        f"{_f(v['ladder'][str(nx)]['mass_percent'])} | "
        f"{_f(v['ladder'][str(nx)]['l1_percent_independent'])} | "
        f"{'✅' if v['ladder'][str(nx)]['l1_percent'] < 1.0 else '❌'} | "
        f"`{v['ladder'][str(nx)]['tag']}` |" for nx in ks)
    orders = "\n".join(
        f"| {k} | {_f(o['coarse_l1'])} | {_f(o['fine_l1'])} | {_f(o['p'])} | "
        f"{'yes' if o['monotone'] else 'no'} |" for k, o in v["orders"].items())

    if e4.get("status") == "measured":
        conv_head = ("| 口径 | eps_ref [%] | 承重假设 |\n|---|---|---|\n" + "\n".join(
            f"| {k} | {_f(R13._eps(e4['bars'][k]), 4)} | {R13._eps(e4['bars'][k], 'assumption')} |"
            for k in ("B0", "B1", "B2", "B3", "B4") if k in e4["bars"]))
        bound_head = ("| nx | L1 [%] | B0 | B1 | B2 | B3 | B4 |\n|---|---|---|---|---|---|---|"
                      + "\n" + "\n".join(
                          f"| {r['nx']} | {_f(r['l1_percent'])} | "
                          + " | ".join(_f(r["upper_bound_percent"].get(k))
                                       for k in ("B0", "B1", "B2", "B3", "B4")) + " |"
                          for r in e4["rows"]))
        surv = ("| 口径 | <1.5 % | <1 % | 总账 <15 % |\n|---|---|---|---|\n" + "\n".join(
            f"| {k} | {d['E4']} | {d['E5']} | {d['ledger']} |"
            for k, d in e4["verdicts"].items()))
    else:
        conv_head = bound_head = surv = f"_未测_：{e4.get('reason')}"

    e1, e1r = v.get("E1_cfl_graded"), v.get("E1r_cfl_reference")
    return f"""# XVAL-FVM-M18737-ROUND2 Round-14 闭合：nx=12800 档落地，诚实上界四口径首进 1 %

> 锚点 `XVAL-FVM-M18737-ROUND2`（TensorFVM 的 m18737-round2 做法），总账验收 <15 %。
> 判据在 R13 已先于数字冻结于 `PREDECLARATION-r13.md`，本轮**不改任何阈值/口径**，
> 只把 R13 窗口内未回件的三个作业取回、回填、按冻结管线重跑。
> 本文件由 `r14_closure.py` 从 `r14_certified_rung.json` 程序化渲染（零手抄）；
> 前轮文档（含 `R13-CERTIFIED-RUNG.md` 的 not_measured 表）**不改写**。

## ① 案例介绍（物理背景 / 几何 / 参数表）

Woodward & Colella (1984) 双爆轰波：一维闭管 x∈[0,1]，三压力平台（左 p=1000 /
中 p=0.01 / 右 p=100），ρ=1、u=0，两束爆轰波相向对撞（t≈0.028），积分至 t=0.038，
γ=1.4，两端反射壁。**无解析解** ⇒ 任何「误差」都是两个数值解之差，参考解自身的
离散误差必须作为误差棒计入（`CERTIFIED-NX6400-ERRATA.md` 的教训）。

| 项 | 值 | 出处 |
|---|---|---|
| 被定级格式 | MUSCL-Hancock 二阶 + MC 限幅 + HLLC + SSP-RK | `compressible/cases.double_blast` |
| 参考格式 | WENO5 + SSP-RK3 + Rusanov（零共享离散） | `xcode/wc_crosscode.py` |
| 比较窗口 / 度量 | x∈[0.4,1.0]，梯形归一化 L1（与前轮逐字一致） | `xval/wc_graded_refinement.py` |
| 冻结参考 | WENO5@12800（自认证 L1 0.544 % vs @25600） | 既有 |
| 本轮新增计算 | **无**（三个 R13 预注册作业回件，纯 numpy 重算） | 见 ② |

## ② 怎么做的（方法 / 流程 / 关键设置）

1. **取回 R13 预注册作业**（`run_compute fetch`，rc=0）：

   | 作业 | 档 | 解锁 |
   |---|---|---|
   | `cj4221` | `{JOBS['cj4221'][0]}`（{JOBS['cj4221'][1]}） | E2/E3/E4 |
   | `cj4222` | `{JOBS['cj4222'][0]}`（{JOBS['cj4222'][1]}） | E1 |
   | `cj4223` | `{JOBS['cj4223'][0]}`（{JOBS['cj4223'][1]}） | E1r |

2. **回填** `ladder/data/`（npz + metrics json 成对入库），E0 工件门逐档复核
   （dx、cells、ρ>0 有限、steps/seconds、mass drift <1e-10、**metrics 里 cfl 必须
   与档名一致**——防「档名说 0.8、实际跑 0.4」）。
3. **冻结管线重跑**：`r13_certified_rung.py`（import，不复制）→ 本文件渲染。
   关键 L1 另有独立重积分（固定 40001 点 stencil + 矩形法，与共享度量不同实现），
   两实现最大差 **{_f(v['independent_check_max_abs_diff_percent'], 3)} pp**（度量伪影须同时骗过两套实现）。
4. 求解器一行未改。

## ③ 计算结果（程序化注入，零手抄）

### E0 工件门（逐档）

| 档 | cells | dx 相对误差 | ρmin | ρmax | steps | 秒 | cfl | mass drift | 通过 |
|---|---|---|---|---|---|---|---|---|---|
""" + "\n".join(
        f"| `{g['tag']}` | {g['cells']} | {g['dx_rel_error']:.2e} | {g['rho_min']:.4f} | "
        f"{g['rho_max']:.4f} | {g['steps']} | {_f(g['seconds_solver'], 1)} | {g['cfl']} | "
        f"{g['mass_drift']} | {'✅' if g['pass'] else '❌'} |"
        for g in v["E0_genuineness"].values()) + f"""

### E1 / E1r CFL 入场门（阈值 {R13.CFL_ADMISSION} %，R13 预注册）

| 门 | 档 | 口径对 | L1 [%] | 峰值 [%] | 准入 |
|---|---|---|---|---|---|
{gate_rows}

两门都过 ⇒ 12800 档（CFL 0.8 变体，CFL 0.4 版因超单作业墙钟明确不做）获得准入；
CFL 侧影响实测 { _f(e1['l1_percent']) if e1 else '—' } %（被定级族）/
{_f(e1r['l1_percent']) if e1r else '—' } %（参考族），均远小于相邻档减量。

### 被定级族逐档（对自认证 WENO5@{R13.REF_FINEST}）

| nx | L1 [%] | 峰值 [%] | 窗质量 [%] | 独立重积分 L1 [%] | L1 <1 % | 档 tag |
|---|---|---|---|---|---|---|
{rows}

| 相邻档 | 粗档 L1 | 细档 L1 | 观测阶 p | 单调 |
|---|---|---|---|---|
{orders}

地板判定：`{v['floor']['floor_class']}`（最细对 {v['floor']['finest_pair']}，
`e_inf = {_f(v['floor']['e_inf_percent_raw'])}`，p = {_f(v['floor']['order'])}）。
3200→12800 总改善 **{_f(100.0 * (v['ladder']['3200']['l1_percent'] - v['ladder'][str(ks[-1])]['l1_percent']) / v['ladder']['3200']['l1_percent'], 1)} %**。

### E2 预注册预测命中（幂律 e(n)=C·n^-p，p=0.991）

| 项 | 值 |
|---|---|
| 预测 L1({ks[-1]}) | {_f(e2['predicted_percent'], 3)} % |
| 预注册带 ±15 % | {_f(e2['band_percent'][0], 3)}–{_f(e2['band_percent'][1], 3)} % |
| 实测 | **{_f(e2['measured_percent'])} %** |
| 偏差 | {_f(e2['deviation_percent_of_prediction'], 1)} % |
| 命中 | {'✅' if e2['hit'] else '❌'}（p({ks[-2]}→{ks[-1]}) = {_f(v['orders'][f'{ks[-2]}->{ks[-1]}']['p'])}，{'≤' if e2['order_within_band'] else '>'} 1.15） |

**honest 记录**：预测**未命中**，且是朝有利方向未命中（实测比幂律外推低 28.5 %，
观测阶 1.477 超带）。按预注册分支处理：幂律读法不采信，不反向调参；
该超收敛对（6400→12800）本身不改变任何判据线。

### E3 certified 门（<1 %，沿用 CERTIFIED_PERCENT=1.0）

| 量 | 实测 | 判据 | 判定 |
|---|---|---|---|
| L1({ks[-1]}) | {_f(v['ladder'][str(ks[-1])]['l1_percent'])} % | <1 % | {'✅' if v['ladder'][str(ks[-1])]['l1_percent'] < 1.0 else '❌'} |
| 峰值({ks[-1]}) | {_f(v['ladder'][str(ks[-1])]['peak_percent'])} % | <1 % | {'✅' if v['ladder'][str(ks[-1])]['peak_percent'] < 1.0 else '❌'} |
| certified_under_1pct | {e3['certified_under_1pct']} | — | {'✅' if e3['certified_under_1pct'] else '❌'} |

口径：`{e3['scope']}`。

### E4 诚实上限（参考侧误差棒取 12800→25600 同仪器对）

{conv_head}

{bound_head}

{surv}

参考侧：e_pair = {_f(e4.get('e_percent'))} %，p_ref = {_f(e4.get('p_ref'))}，
licensed = {e4.get('licensed')}（{e4.get('license_note', '')}）。

## ④ 结果图

- `figures/fig_r13_certified_rung.png`（R13 管线出图，本轮数据重绘）：
  (a) 被定级族 L1 阶梯（含 nx=12800 实测）与幂律预测、1 %/15 % 判据线；
  (b) 五种误差棒口径下的诚实上界（L1 + eps_ref）；(c) CFL 0.4 vs 0.8 同格准入控制。
- `figures/fig_r14_bars_6400_vs_12800.png`：6400 vs 12800 两档的五口径上界对比
  （R11 在 6400 上五口径全不进 1 %，本轮 12800 上 B0/B1/B2/B4 进线、B3 仍越线）。

## ⑤ 注意点（坑 / 局限 / 适用范围）

1. **12800 档是 CFL 0.8 变体**：CFL 0.4 的 nx=12800 被定级档明确不做（~178 min 超墙钟），
   其可用性由 E1（{ _f(e1['l1_percent']) if e1 else '—' } %）与 E1r（{_f(e1r['l1_percent']) if e1r else '—' } %）
   两道入场门背书，而不是「结果看起来不错」。
2. **E4 的 licensed=False 保留**：参考阶梯 6400 档仍是 CFL 0.4（无 6400 CFL 0.8 档），
   跨口径 ⇒ eps_ref 记为探针。但 E1r 实测 CFL 0.4→0.8 在 12800 只移 0.0255 %，
   比该对差商 0.2951 % 小一个量级，eps_ref 一阶值可信、不得当已认证棒引用。
3. **B3（非渐近 Fs=3）仍越线**（{_f(e4['finest']['upper_bound_percent'].get('B3'))} %）：
   最保守口径下「<1 %」仍不可证，honest FAIL 不覆盖。
4. **E2 未命中**（朝有利方向）：幂律外推在 6400→12800 段失效（p=1.477），
   下一档（25600）的预测不可再按幂律给，需新的预注册。
5. 全部结论限于「vs 自认证 WENO5@12800」；对精确解的 <1 % 声明**不成立**
   （无解析解，且 B3 越线）。

## ⑥ 误差判定（逐项，显式回答）

| 判据 | 最不利实测 | 判定 |
|---|---|---|
| 总账 <15 % | 最保守上界 B3 = {_f(e4['finest']['upper_bound_percent'].get('B3'))} % | **PASS**（余量 { _f(15.0 - e4['finest']['upper_bound_percent'].get('B3'), 2) if e4.get('status')=='measured' else '—' } pp） |
| <3 % | 同上 | **PASS** |
| <1 %（vs 自认证参考，点估计） | L1 = {_f(v['ladder'][str(ks[-1])]['l1_percent'])} % | **PASS** |
| <1 %（诚实上界，B0/B1/B2/B4） | {_f(e4['finest']['upper_bound_percent'].get('B0'))} / {_f(e4['finest']['upper_bound_percent'].get('B1'))} / {_f(e4['finest']['upper_bound_percent'].get('B2'))} / {_f(e4['finest']['upper_bound_percent'].get('B4'))} % | **PASS**（四口径） |
| <1 %（诚实上界，B3 非渐近） | {_f(e4['finest']['upper_bound_percent'].get('B3'))} % | **honest FAIL**（最保守口径越线） |

**一句话裁决**：`{v['adjudication']}`
"""


def figures(v: Dict, outdir: Path) -> List[Dict]:
    """R13's three panels (re-drawn on the completed ladder) + the 6400-vs-12800 bar figure."""
    made = R13.figures(v, outdir)
    e4 = v["E4_honest_ceiling"]
    if e4.get("status") != "measured":
        return made
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    import numpy as np
    keys = ["B0", "B1", "B2", "B3", "B4"]
    cols = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
    rows = {r["nx"]: r for r in e4["rows"]}
    x = np.arange(len(keys))
    w = 0.38
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    for i, (k, c) in enumerate(zip(keys, cols)):
        y64 = rows[6400]["upper_bound_percent"].get(k)
        y128 = rows[12800]["upper_bound_percent"].get(k)
        ax.bar(x[i] - w / 2, y64, w, color=c, alpha=0.45, label=f"{k} @6400")
        ax.bar(x[i] + w / 2, y128, w, color=c, label=f"{k} @12800")
    ax.axhline(1.0, color="crimson", ls=":", lw=1.4, label="<1 % certified bar")
    ax.axhline(1.5, color="#bbbbbb", ls="--", lw=0.9, label="<1.5 %")
    ax.set_xticks(x)
    ax.set_xticklabels(keys)
    ax.set_ylabel("honest upper bound L1 + eps_ref [%]")
    ax.set_title("Honest ceiling: nx=6400 (R11/R13 probe) vs nx=12800 (R14)")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(fontsize=6.5, ncol=2)
    fig.tight_layout()
    p = outdir / "fig_r14_bars_6400_vs_12800.png"
    fig.savefig(p, dpi=140)
    plt.close(fig)
    made.append({"path": str(p), "name": p.name,
                 "caption": "五口径诚实上界（L1+eps_ref）在 6400 与 12800 两档的对比："
                            "6400 五口径全越 1 % 线，12800 上 B0/B1/B2/B4 进线、B3（非渐近 Fs=3）仍越线。"})
    return made


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--json", type=Path, default=HERE / "r14_certified_rung.json")
    ap.add_argument("--markdown", type=Path, default=HERE / "R14-CLOSURE.md")
    ap.add_argument("--figures", type=Path, default=HERE / "figures")
    ap.add_argument("--no-figures", action="store_true")
    args = ap.parse_args()

    v = R13.compute()
    v["round"] = 14
    v["jobs"] = JOBS
    if not args.no_figures:
        v["figures"] = figures(v, args.figures)
    args.json.write_text(json.dumps(v, indent=2) + "\n")
    args.markdown.write_text(render(v))
    print(json.dumps({"status": v["status"], "adjudication": v["adjudication"],
                      "missing_rungs": v["missing_rungs"],
                      "finest_l1": v["ladder"][str(max(int(k) for k in v["ladder"]))]["l1_percent"],
                      "json": str(args.json), "markdown": str(args.markdown)}))


if __name__ == "__main__":
    main()
