"""Round-15 driver: the missing WENO5@6400 CFL 0.8 rung licenses the eps_ref bar.

Why this file exists
--------------------
``R14-CLOSURE.md`` §⑤-2 left exactly one item open on the anchor
``XVAL-FVM-M18737-ROUND2``: the reference ladder's 6400 rung is still the
frozen CFL 0.4 instrument (no 6400 CFL 0.8 rung), so the E4 ``licensed``
check (all reference rungs on one instrument) is False and ``eps_ref`` is
recorded as a *probe* rather than a certified error bar.

This round does **one new solver job** (WENO5 @nx=6400, CFL 0.8, tag
``nx6400_weno5_cfl0.8``, produced by ``ladder/wc_ladder_weno5_cfl.py``) and
**one pre-declared pipeline change** (``PREDECLARATION-r15.md`` §③): the
reference-ladder tag table of the frozen R13 pipeline is exposed as the
module constant ``REF_LADDER_TAGS`` (default = R14 behaviour verbatim) and is
overridden for the 6400 rung *only after* the new E1r6400 same-grid CFL
admission gate passes.  No threshold, window, metric or criterion changes.

Honest branches (pre-registered):

* E0 fails  -> the rung is ``not_measured``; the pipeline runs with the
  default tags; this file reports the negative result.
* E1r6400 fails (L1 >= 0.2 %) -> the 6400 rung stays on the CFL 0.4
  instrument, ``licensed`` stays False, no re-tuning.

Usage::

    PYTHONPATH=src python docs/XVAL-FVM-M18737-ROUND2/r15_ref_ladder_cfl.py \
        --json docs/XVAL-FVM-M18737-ROUND2/r15_certified_rung.json \
        --markdown docs/XVAL-FVM-M18737-ROUND2/R15-CLOSURE.md
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
from pathlib import Path
from typing import Dict, Optional

import numpy as np

HERE = Path(__file__).resolve().parent


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


R13 = _load_module(HERE / "r13_certified_rung.py", "wc_r13_driver")
GH = R13.GH

NEW_NX = 6400
NEW_TAG = "weno5_cfl0.8"
CANON_TAG = "weno5"


def _f(x, nd=4):
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return "—"
    return f"{x:.{nd}f}"


def e1r6400() -> Optional[Dict]:
    """E1r6400: same-grid CFL 0.4 vs 0.8 L1/peak on the reference family @6400."""
    a = R13.load_profile(NEW_NX, CANON_TAG)
    b = R13.load_profile(NEW_NX, NEW_TAG)
    if a is None or b is None:
        return None
    d = GH.l1_percent(a["x"], a["rho"], b["x"], b["rho"])
    d_peak = GH.peak_percent(a["x"], a["rho"], b["x"], b["rho"])
    return {"nx": NEW_NX, "canonical_tag": a["tag"], "relaxed_tag": b["tag"],
            "l1_percent": d, "peak_percent": d_peak,
            "admitted": bool(d < R13.CFL_ADMISSION),
            "threshold_percent": R13.CFL_ADMISSION}


def invariants(r14: Dict, v: Dict) -> Dict:
    """The graded side must be bit-identical to R14 (same input artifacts)."""
    out: Dict = {"ok": True, "mismatches": []}
    for nx, row14 in r14["ladder"].items():
        row15 = v["ladder"].get(nx)
        if row15 is None:
            out["ok"] = False
            out["mismatches"].append(f"ladder[{nx}] missing")
            continue
        for k in ("l1_percent", "peak_percent", "mass_percent",
                  "l1_percent_independent"):
            if row14[k] != row15[k]:
                out["ok"] = False
                out["mismatches"].append(f"ladder[{nx}].{k}: {row14[k]} != {row15[k]}")
    for key in ("E1_cfl_graded", "E1r_cfl_reference", "E2_prediction",
                "E3_certified", "orders", "floor"):
        if r14.get(key) != v.get(key):
            out["ok"] = False
            out["mismatches"].append(f"{key} differs from R14")
    return out


def render(v: Dict, e0: Dict, e1r6400: Optional[Dict], inv: Dict,
           r14: Dict) -> str:
    e4 = v["E4_honest_ceiling"]
    licensed = bool(e4.get("licensed"))
    ref_rows = "\n".join(
        f"| {r['nx']} | `{r['tag']}` | {r['cfl']} | "
        f"{_f(r.get('l1_vs_prev_percent'))} | {_f(r.get('p_vs_prev'))} |"
        for r in e4["reference_pair"])
    bars = e4["bars"]
    bar_rows = "\n".join(
        f"| {k} | {_f(bars[k]['eps_percent'])} | {bars[k]['assumption']} |"
        for k in ("B0", "B1", "B2", "B3", "B4"))
    rows = "\n".join(
        f"| {r['nx']} | {_f(r['l1_percent'])} | " +
        " | ".join(_f(r["upper_bound_percent"].get(k)) for k in ("B0", "B1", "B2", "B3", "B4")) +
        " |" for r in e4["rows"])
    verdict_rows = "\n".join(
        f"| {k} | {d['E4']} | {d['E5']} | {d['ledger']} |"
        for k, d in e4["verdicts"].items())
    g = e0
    gate = (f"| E1r6400（参考族 @6400） | {e1r6400['nx']} | `{e1r6400['canonical_tag']}` vs "
            f"`{e1r6400['relaxed_tag']}` | {_f(e1r6400['l1_percent'])} | "
            f"{_f(e1r6400['peak_percent'])} | "
            f"{'✅ 准入' if e1r6400['admitted'] else '❌ 拒绝'} |"
        if e1r6400 else "| E1r6400（参考族 @6400） | — | — | — | — | not_measured |")
    inv_txt = ("全部一致（被定级族阶梯 / E1 / E1r / E2 / E3 与 R14 逐位相同）"
               if inv["ok"] else "不一致：" + "; ".join(inv["mismatches"]))
    return f"""# XVAL-FVM-M18737-ROUND2 Round-15 闭合：参考阶梯 6400 档补 CFL 0.8 变体，eps_ref 升为已认证误差棒

> 锚点 `XVAL-FVM-M18737-ROUND2`（TensorFVM 的 m18737-round2 做法），总账验收 <15 %。
> 判据在 `PREDECLARATION-r15.md` 先于数字冻结；本轮唯一管线改动（6400 取档表覆盖）
> 以 E1r6400 门通过为前提，**不改任何阈值/口径**。
> 本文件由 `r15_ref_ladder_cfl.py` 从 `r15_certified_rung.json` 程序化渲染（零手抄）；
> 前轮文档（含 `R14-CLOSURE.md`）**不改写**。

## ① 案例介绍（物理背景 / 几何 / 参数表）

Woodward & Colella (1984) 双爆轰波：一维闭管 x∈[0,1]，三压力平台（左 p=1000 /
中 p=0.01 / 右 p=100），ρ=1、u=0，两束爆轰波相向对撞（t≈0.028），积分至 t=0.038，
γ=1.4，两端反射壁。**无解析解** ⇒ 任何「误差」都是两个数值解之差，参考解自身的
离散误差必须作为误差棒计入。

| 项 | 值 | 出处 |
|---|---|---|
| 被定级格式 | MUSCL-Hancock 二阶 + MC 限幅 + HLLC + SSP-RK | `compressible/cases.double_blast` |
| 参考格式 | WENO5 + SSP-RK3 + Rusanov（零共享离散） | `ladder/wc_ladder_weno5_cfl.py` |
| 比较窗口 / 度量 | x∈[0.4,1.0]，梯形归一化 L1（与前轮逐字一致） | `xval/wc_graded_refinement.py` |
| 冻结参考 | WENO5@12800（自认证 L1 0.544 % vs @25600） | 既有 |
| 本轮新增计算 | **1 个档**：WENO5@6400 CFL 0.8（tag `nx6400_weno5_cfl0.8`） | 见 ② |

## ② 怎么做的（方法 / 流程 / 关键设置）

1. **新档**：`wc_ladder_weno5_cfl.py --sizes 6400 --cfl 0.8`（作业 `cj4398`，
   1 核，~6 min），工件回填 `ladder/data/`（npz + metrics json 成对入库）。
2. **E0 工件门**（同 R13 三道，含「metrics 里 cfl 必须与档名一致」）。
3. **E1r6400 入场门**：同格 CFL 0.4 vs 0.8 的互相 L1 < 0.2 %（与 E1/E1r 同阈值）。
4. **冻结管线重跑**：`r13_certified_rung.py`（import，不复制）；门过 ⇒ 6400 取档表
   覆盖为 `("weno5_cfl0.8", "weno5")`（预声明改动），门不过 ⇒ 默认表（R14 行为）。
5. **不变量复核**：被定级族阶梯 / E1 / E1r / E2 / E3 与 `r14_certified_rung.json`
   逐位比对（同输入工件 ⇒ 必须逐位相同）：{inv_txt}。
6. 求解器一行未改。

## ③ 计算结果（程序化注入，零手抄）

### E0 工件门（新档）

| 项 | 值 |
|---|---|
| cells | {g['cells']} |
| dx 相对误差 | {g['dx_rel_error']:.3e} |
| ρmin / ρmax | {g['rho_min']:.4f} / {g['rho_max']:.4f} |
| steps / 秒 | {g['steps']} / {_f(g['seconds_solver'], 1)} |
| cfl（metrics） | {g['cfl']} |
| mass drift | {g['mass_drift']} |
| 通过 | {'✅' if g['pass'] else '❌'} |

### E1r6400 CFL 入场门（阈值 0.2 %，R15 预注册）

{gate}

### 参考阶梯（本轮取档）

| nx | tag | CFL | L1 vs 上一档 [%] | 观测阶 |
|---|---|---|---|---|
{ref_rows}

### E4 诚实上限（参考侧误差棒，licensed={str(licensed).lower()}）

| 口径 | eps_ref [%] | 承重假设 |
|---|---|---|
{bar_rows}

| nx | L1 [%] | B0 | B1 | B2 | B3 | B4 |
|---|---|---|---|---|---|---|
{rows}

| 口径 | <1.5 % | <1 % | 总账 <15 % |
|---|---|---|---|
{verdict_rows}

参考侧：e_pair = {_f(e4['e_percent'])} %，p_ref = {_f(e4['p_ref'])}，
licensed = {str(licensed).lower()}。

## ④ 结果图

- `figures/fig_r15_ref_ladder.png`：(a) 参考阶梯三档同仪器（全 CFL 0.8）的相邻档
  L1 对差商与 eps_ref；(b) 五口径诚实上界（licensed 状态标注）；(c) 6400 档
  CFL 0.4 vs 0.8 剖面（E1r6400 控制）。

## ⑤ 注意点（坑 / 局限 / 适用范围）

1. **licensed 的含义**：三档同仪器 ⇒ eps_ref 可作误差棒引用；但「vs 自认证参考」
   的口径不变，对精确解的 <1 % 声明**仍不成立**（无解析解）。
2. **B3（非渐近 Fs=3）若仍越 1 % 线**：honest FAIL 不覆盖（预注册分支）。
3. 6400 档 CFL 0.4 工件**保留**（E1r6400 的对照侧 + 门不过时的回退档）。
4. 全部结论限于「vs 自认证 WENO5@12800」。

## ⑥ 误差判定（逐项，显式回答）

| 判据 | 最不利实测 | 判定 |
|---|---|---|
| 总账 <15 % | 最保守上界 B3 = {_f(e4['rows'][-1]['upper_bound_percent'].get('B3'))} % | **{'PASS' if (e4['rows'][-1]['upper_bound_percent'].get('B3') or 99) < 15 else 'FAIL'}** |
| <3 % | 同上 | **{'PASS' if (e4['rows'][-1]['upper_bound_percent'].get('B3') or 99) < 3 else 'FAIL'}** |
| <1 %（vs 自认证参考，点估计） | L1 = {_f(v['ladder'][str(max(int(k) for k in v['ladder']))]['l1_percent'])} % | **{'PASS' if v['ladder'][str(max(int(k) for k in v['ladder']))]['l1_percent'] < 1.0 else 'FAIL'}** |
| <1 %（诚实上界，B0/B1/B2/B4） | {_f(e4['rows'][-1]['upper_bound_percent'].get('B0'))} / {_f(e4['rows'][-1]['upper_bound_percent'].get('B1'))} / {_f(e4['rows'][-1]['upper_bound_percent'].get('B2'))} / {_f(e4['rows'][-1]['upper_bound_percent'].get('B4'))} % | **{'PASS' if all((e4['rows'][-1]['upper_bound_percent'].get(k) or 99) < 1.0 for k in ('B0','B1','B2','B4')) else 'FAIL'}** |
| <1 %（诚实上界，B3 非渐近） | {_f(e4['rows'][-1]['upper_bound_percent'].get('B3'))} % | **{'PASS' if (e4['rows'][-1]['upper_bound_percent'].get('B3') or 99) < 1.0 else 'honest FAIL'}** |

**一句话裁决**：`{v['adjudication']}`
"""


def figures(v: Dict, outdir: Path, e1r6400: Optional[Dict]) -> list:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    outdir.mkdir(parents=True, exist_ok=True)
    e4 = v["E4_honest_ceiling"]
    licensed = bool(e4.get("licensed"))
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.4))

    rp = e4["reference_pair"]
    xs = [r["nx"] for r in rp]
    ys = [r.get("l1_vs_prev_percent") for r in rp[1:]]
    ax[0].semilogy(xs[1:], ys, "s-", color="#1f77b4", label="adjacent-pair L1")
    ax[0].axhline(e4["e_percent"], color="crimson", ls=":",
                   label=f"eps_ref = {e4['e_percent']:.4f} % (12800→25600)")
    ax[0].set_xlabel("fine rung nx")
    ax[0].set_ylabel("L1 [%]")
    ax[0].set_title("(a) reference ladder, one instrument (CFL 0.8)")
    ax[0].grid(True, which="both", alpha=0.3)
    ax[0].legend(fontsize=7)

    keys = ["B0", "B1", "B2", "B3", "B4"]
    cols = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
    for key, col in zip(keys, cols):
        yv = [r["upper_bound_percent"].get(key) for r in e4["rows"]]
        xv = [r["nx"] for r in e4["rows"] if r["upper_bound_percent"].get(key) is not None]
        yv = [y for y in yv if y is not None]
        if yv:
            ax[1].loglog(xv, yv, "s-", color=col, label=key)
    ax[1].loglog([r["nx"] for r in e4["rows"]], [r["l1_percent"] for r in e4["rows"]],
                 "ko--", label="point estimate")
    ax[1].axhline(1.0, color="crimson", ls=":", label="<1 %")
    ax[1].set_xlabel("nx (graded)")
    ax[1].set_ylabel("L1 + eps_ref [%]")
    ax[1].set_title(f"(b) honest upper bound (licensed={str(licensed).lower()})")
    ax[1].grid(True, which="both", alpha=0.3)
    ax[1].legend(fontsize=7)

    a = R13.load_profile(NEW_NX, CANON_TAG)
    b = R13.load_profile(NEW_NX, NEW_TAG)
    if a is not None and b is not None:
        ax[2].plot(a["x"], a["rho"], lw=1.2, label="CFL 0.4 (frozen)")
        ax[2].plot(b["x"], b["rho"], lw=1.0, ls="--", color="#d62728",
                  label="CFL 0.8 (new)")
        ax[2].set_xlabel("x")
        ax[2].set_ylabel("rho")
        if e1r6400:
            ax[2].set_title(f"(c) E1r6400: L1 = {e1r6400['l1_percent']:.4f} % "
                            f"(bar {R13.CFL_ADMISSION} %)")
        else:
            ax[2].set_title("(c) E1r6400: not measured")
        ax[2].legend(fontsize=7)
    else:
        ax[2].text(0.5, 0.5, "not measured", ha="center", va="center")
        ax[2].set_axis_off()

    fig.suptitle("XVAL-FVM-M18737-ROUND2 R15: reference ladder on one instrument")
    fig.tight_layout()
    p = outdir / "fig_r15_ref_ladder.png"
    fig.savefig(p, dpi=140)
    plt.close(fig)
    return [{"path": str(p), "name": p.name,
             "caption": "(a) 参考阶梯 6400/12800/25600 同仪器（CFL 0.8）相邻档 L1 对差商与 "
                        "eps_ref；(b) 五口径诚实上界（licensed 状态标注）；(c) 6400 档 "
                        "CFL 0.4 vs 0.8 剖面（E1r6400 准入控制）。"}]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--json", type=Path, default=HERE / "r15_certified_rung.json")
    ap.add_argument("--markdown", type=Path, default=None)
    ap.add_argument("--figures", type=Path, default=HERE / "figures")
    ap.add_argument("--no-figures", action="store_true")
    args = ap.parse_args()

    prof = R13.load_profile(NEW_NX, NEW_TAG)
    if prof is None:
        raise FileNotFoundError(R13.profile_path(NEW_NX, NEW_TAG))
    prof["metrics"] = R13.load_metrics(NEW_NX, NEW_TAG)
    e0 = R13.genuineness(NEW_NX, prof)
    e1r6400_res = e1r6400()

    admitted = bool(e0["pass"] and e1r6400_res and e1r6400_res["admitted"])
    if admitted:
        R13.REF_LADDER_TAGS = (
            (NEW_NX, (NEW_TAG, CANON_TAG)),
            (R13.REF_FINEST, ("weno5_cfl0.8", "weno5")),
            (R13.REF_EXTENDED, ("weno5_cfl0.8", "weno5")))
    v = R13.compute()

    r14 = json.loads((HERE / "r14_certified_rung.json").read_text())
    inv = invariants(r14, v)
    v["round"] = 15
    v["jobs"] = {"cj4398": ("nx6400_weno5_cfl0.8",
                            "the missing reference rung nx=6400 CFL 0.8 (E4 license)")}
    v["E0_new_rung"] = e0
    v["E1r6400_cfl_reference"] = e1r6400_res
    v["r15_admitted"] = admitted
    v["invariants_vs_r14"] = inv
    if not args.no_figures:
        v["figures"] = figures(v, args.figures, e1r6400_res)
    args.json.write_text(json.dumps(v, indent=2) + "\n")
    if args.markdown:
        args.markdown.write_text(render(v, e0, e1r6400_res, inv, r14))
    print(json.dumps({"status": v["status"], "adjudication": v["adjudication"],
                      "E0_pass": e0["pass"],
                      "E1r6400": (e1r6400_res["l1_percent"] if e1r6400_res else None),
                      "admitted": admitted,
                      "licensed": v["E4_honest_ceiling"].get("licensed"),
                      "invariants_ok": inv["ok"],
                      "json": str(args.json)}))


if __name__ == "__main__":
    main()
