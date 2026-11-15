"""Round-10: land the nx=25600 reference rung on the *frozen* instrument (CFL 0.4).

The debt
--------
Round 5 froze Q1-Q5 waiting for ``WENO5@25600`` (job ``cj3710``), which never came back.
Round 7 tried to buy the rung back by relaxing the *time step* to CFL 0.8 and admitting it
through the pre-frozen equivalence gate E1 -- but its own local probe already showed the
peak metric does **not** respond to CFL as ``dx^3`` (the collision peak sits on a
positivity-fallback switch), so the admission route was suspect from the start.  Round 10
does the obvious thing instead: it runs the rung on the **same** instrument as every other
rung of the ladder (``xcode/wc_crosscode.py::weno5_solve``, WENO5 + SSP-RK3 + Rusanov,
CFL 0.4) in a single job, and therefore needs no admission gate at all.

What this driver checks (all frozen before the run, see ``PREDECLARATION-r10.md``)
-------------------------------------------------------------------------------
* **E0a provenance** -- the standard-name rung must *be* the frozen instrument: ``cfl == 0.4``,
  ``status == "ok"``, no ``promoted_from`` field.  A promoted CFL-0.8 artifact would fail here.
* **E0b reproducibility** -- the same job re-runs the control rung nx=6400; it must reproduce
  the stored CFL-0.4 rung (arrays equal to within round-off).  This is what licenses reading
  the new 25600 rung as the same instrument as the older rungs.
* **E1 (inherited)** -- the CFL-0.8 control rung is still measured, so round-7's gate closes
  with a number instead of ``not_measured``.  Its verdict is *recorded*, not used: the rung
  admitted into the ladder no longer depends on it.
* **E2-E5** -- identical definitions to round 7 (residual ``h^1`` exponent, reference error
  bar, honest upper bound of the graded family, provable ``<1 %``).

Nothing in rounds 1-9 is rewritten; this file only adds ``R10-REF25600-FROZEN-CLOSURE.md``
and ``summary_r10.json`` (every number programmatic, zero hand-copy).

    PYTHONPATH=src python docs/XVAL-FVM-M18737-ROUND2/ref25600_r10.py [--jobs '{"rung":"cj4143"}']
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
from pathlib import Path
from typing import Dict, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from tensorfvm.verification import wc_ref_uncertainty as U  # noqa: E402
from tensorfvm.verification import wc_refine as R  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DATA = ROOT / "woodward-colella-double-blast" / "ladder" / "data"
EVIDENCE = HERE / "evidence"
FIGDIR = HERE / "figures"

EXTENDED = 25600
CONTROL_NX = 6400
FROZEN_CFL = 0.4
GRADED_RUNGS = (1600, 3200, 6400)


def _r7():
    """The round-7 driver holds the *frozen* definitions of E1-E5; reuse, do not fork."""
    name = "_ref25600_r7"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, str(HERE / "ref25600_r7.py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


def _r(x, nd=4):
    return None if x is None else round(float(x), nd)


def _std_paths():
    return (DATA / f"ladder_profile_nx{EXTENDED}_weno5.npz",
            DATA / f"ladder_metrics_nx{EXTENDED}_weno5.json")


# --------------------------------------------------------------------------- #
# E0a: provenance of the admitted rung
# --------------------------------------------------------------------------- #
def provenance() -> Dict:
    prof, met = _std_paths()
    out: Dict = {"tag": f"nx{EXTENDED}_weno5", "profile_present": prof.exists(),
                 "metrics_present": met.exists(), "frozen_cfl": FROZEN_CFL}
    if not (prof.exists() and met.exists()):
        out["E0a"] = "not_measured"
        return out
    m = json.loads(met.read_text())
    out.update({
        "cfl": m.get("cfl"), "status": m.get("status"), "steps": m.get("steps"),
        "seconds_solver": m.get("seconds_solver"), "fallbacks": m.get("positivity_fallbacks"),
        "mass_drift": m.get("mass_drift"), "rho_max": m.get("rho_max"),
        "method": m.get("method"), "promoted_from": m.get("promoted_from"),
    })
    checks = {
        "cfl_is_frozen": m.get("cfl") == FROZEN_CFL,
        "status_ok": m.get("status") == "ok",
        "not_promoted": "promoted_from" not in m,
        "finite": bool(m.get("finite", False)),
    }
    out["checks"] = checks
    out["E0a"] = "pass" if all(checks.values()) else "fail"
    return out


# --------------------------------------------------------------------------- #
# E0b: does the same instrument reproduce the older control rung?
# --------------------------------------------------------------------------- #
def reproducibility() -> Dict:
    """Rerun of nx=6400 (same job, CFL 0.4) against the stored rung of the ladder."""
    rerun_prof = EVIDENCE / f"rerun_nx{CONTROL_NX}_weno5.npz"
    rerun_met = EVIDENCE / f"rerun_nx{CONTROL_NX}_weno5.json"
    out: Dict = {"control_nx": CONTROL_NX, "rerun_present": rerun_prof.exists()}
    base = R.try_load_profile(DATA, CONTROL_NX, "weno5")
    if base is None or not rerun_prof.exists():
        out["E0b"] = "not_measured"
        return out
    z = np.load(rerun_prof)
    x = np.asarray(z["x"], float)
    rho = np.asarray(z["rho"], float)
    order = np.argsort(x)
    x, rho = x[order], rho[order]
    same_grid = base["x"].size == x.size and np.array_equal(base["x"], x)
    d_rho = float(np.max(np.abs(base["rho"] - rho))) if same_grid else float("nan")
    rel = float(d_rho / np.max(np.abs(base["rho"]))) if same_grid else float("nan")
    out.update({
        "same_grid": bool(same_grid),
        "max_abs_drho": _r(d_rho, 16), "rel_delta": _r(rel, 16),
        "bitwise_equal": bool(same_grid and d_rho == 0.0),
    })
    if rerun_met.exists():
        out["rerun_metrics"] = json.loads(rerun_met.read_text())
    out["E0b"] = "pass" if (same_grid and rel < 1e-12) else "fail"
    return out


# --------------------------------------------------------------------------- #
# E1-E5 through the frozen round-7 instruments
# --------------------------------------------------------------------------- #
def instrument_control() -> Dict:
    """Round-7's E1 instrument-equivalence control (one definition, imported not copied)."""
    return _r7().instrument_control()



def adjudicate() -> Dict:
    r7 = _r7()
    control = r7.instrument_control()          # E1 (CFL-0.8 route, recorded only)
    promotion = r7.promote_extended(control)   # must be a no-op now
    reference = r7.reference_ladder(control)   # E2/E3 on the 5-rung ladder
    grade = r7.regrade(reference)              # E4/E5
    return {"control": control,
            "promotion": {k: v for k, v in promotion.items() if k != "metrics"},
            "reference": reference, "regrade": grade}


# --------------------------------------------------------------------------- #
# figures
# --------------------------------------------------------------------------- #
def figures(s: Dict) -> list:
    FIGDIR.mkdir(parents=True, exist_ok=True)
    out = []
    ref, grade = s["reference"], s["regrade"]
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))

    pairs = ref["selfcert_pairs"]
    if pairs:
        lbl = [f"{r['coarse_nx']}→{r['fine_nx']}" for r in pairs]
        val = [r["l1_percent"] for r in pairs]
        ax[0].bar(range(len(val)), val, color="#1f77b4")
        ax[0].axhline(1.0, ls="--", lw=1, color="k", label="1 % bar")
        ax[0].set_xticks(range(len(val)))
        ax[0].set_xticklabels(lbl, rotation=30, ha="right", fontsize=8)
        ax[0].set_ylabel("WENO5 self-convergence L1 [%]")
        ax[0].set_yscale("log")
        ax[0].legend(fontsize=8)
    ax[0].set_title("(a) reference ladder, frozen CFL 0.4")

    d = ref["defects"]
    if d:
        idx = np.arange(len(d))
        ax[1].semilogy(idx, [x["max_abs_D"] for x in d], "o-", label="max|D| measured")
        ax[1].semilogy(idx, [d[0]["max_abs_D"] / 2.0 ** (i + 1) for i in idx], "k--", lw=1,
                       label="h$^1$ slope")
        ax[1].set_xticks(idx)
        ax[1].set_xticklabels([f"{x['coarse_nx']}→{x['fine_nx']}" for x in d],
                              rotation=30, ha="right", fontsize=8)
        ax[1].set_ylabel("max|D|  [mass units]")
        ax[1].legend(fontsize=8)
    ax[1].set_title("(b) accumulated-difference amplitude")

    rows = [r for r in grade.get("rows", []) if r.get("measured")]
    prev = [r for r in grade.get("rows_prev", []) if r.get("measured")]
    if prev:
        ax[2].loglog([r["nx"] for r in prev], [r["l1_percent"] for r in prev], "s--",
                     color="#999999", label="vs WENO5@12800")
    if rows:
        ax[2].loglog([r["nx"] for r in rows], [r["l1_percent"] for r in rows], "o-",
                     color="#1f77b4", label="vs WENO5@25600 (frozen CFL)")
        ax[2].loglog([r["nx"] for r in rows], [r["upper_bound_percent"] for r in rows], ":",
                     color="#1f77b4", label="honest upper bound")
    ax[2].axhline(1.0, ls="--", lw=1, color="k")
    ax[2].axhline(15.0, ls=":", lw=1, color="r", label="ledger 15 %")
    ax[2].set_xlabel("nx (graded MUSCL/HLLC)")
    ax[2].set_ylabel("window L1 [%]")
    ax[2].legend(fontsize=8)
    ax[2].set_title("(c) graded family vs both references")
    fig.tight_layout()
    p = FIGDIR / "fig_r10_reference_frozen.png"
    fig.savefig(p, dpi=140)
    plt.close(fig)
    out.append({"name": p.name, "path": str(p),
                "caption": "冻结仪器（CFL 0.4）五级参考自收敛、差分幅值 h^1 检验、被定级族对 12800/25600 两参考"})

    prof, _ = _std_paths()
    if prof.exists():
        r25 = R.load_profile(DATA, EXTENDED, "weno5")
        fig, ax = plt.subplots(figsize=(7.6, 4.2))
        r13 = R.try_load_profile(DATA, 12800, "weno5")
        if r13 is not None:
            ax.plot(r13["x"], r13["rho"], lw=1.2, label="WENO5@12800 (round-4 ref)")
        ax.plot(r25["x"], r25["rho"], lw=1.2, ls="--", label="WENO5@25600 (this round)")
        g = R.try_load_profile(DATA, max(GRADED_RUNGS), "hllc")
        if g is not None:
            ax.plot(g["x"], g["rho"], lw=0.9, color="#d62728",
                    label=f"MUSCL/HLLC@{max(GRADED_RUNGS)}")
        ax.set_xlim(0.6, 0.9)
        ax.set_xlabel("x")
        ax.set_ylabel(r"$\rho$ (cell average)")
        ax.legend(fontsize=8)
        ax.set_title("collision region: frozen-instrument reference vs graded solution")
        fig.tight_layout()
        p2 = FIGDIR / "fig_r10_profiles_25600_frozen.png"
        fig.savefig(p2, dpi=140)
        plt.close(fig)
        out.append({"name": p2.name, "path": str(p2),
                    "caption": "对撞区：WENO5@12800 / WENO5@25600（冻结 CFL 0.4）/ MUSCL-HLLC@6400"})
    return out


# --------------------------------------------------------------------------- #
# document render (numbers injected)
# --------------------------------------------------------------------------- #
def _f(x, nd=4, dash="—"):
    if x is None:
        return dash
    if isinstance(x, bool):
        return "✅" if x else "❌"
    if isinstance(x, (int, np.integer)):
        return f"{int(x)}"
    if isinstance(x, float) and not math.isfinite(x):
        return dash
    return f"{float(x):.{nd}f}"


def render(s: Dict) -> str:
    e0a, e0b = s["provenance"], s["reproducibility"]
    c, ref, g = s["control"], s["reference"], s["regrade"]
    e1, e2, e3, e4, e5 = c.get("E1"), ref["E2"], ref["E3"], g["E4"], g["E5"]
    ext = ref["extended_uncertainty"]
    fin = g.get("finest", {})
    ok = e0a.get("E0a") == "pass" and ref["extended_present"]

    rows_self = "\n".join(
        f"| {r['coarse_nx']}→{r['fine_nx']} | {_f(r['l1_percent'])} | {_f(r['peak_percent'])} "
        f"| {_f(r['mass_percent'], 6)} | {_f(r.get('order'))} | {_f(r.get('u_ref_percent'))} |"
        for r in ref["selfcert_pairs"])
    rows_def = "\n".join(
        f"| {d['coarse_nx']}→{d['fine_nx']} | {_f(d['max_abs_D'], 6)} "
        f"| {_f(d.get('order_vs_previous_pair'))} | {_f(100 * d['rel_net_mass'], 6)} |"
        for d in ref["defects"])
    rows_g = "\n".join(
        f"| {r['nx']} | {_f(r['l1_percent'])} | {_f(r.get('upper_bound_percent'))} "
        f"| {_f(r['peak_percent'])} | {_f(r.get('pointwise_l1_percent'))} | {_f(r.get('earned_certified'))} |"
        for r in g.get("rows", []) if r.get("measured"))
    rows_gp = "\n".join(
        f"| {r['nx']} | {_f(r['l1_percent'])} | {_f(r['peak_percent'])} |"
        for r in g.get("rows_prev", []) if r.get("measured"))

    return f"""# XVAL-FVM-M18737-ROUND2 Round-10：扩展参考 WENO5@25600 在**冻结仪器**上落地（CFL 0.4）

> 锚点 `XVAL-FVM-M18737-ROUND2`（TensorFVM 的 m18737-round2 做法），验收 <15 %。
> **本文**：round-5 起悬空的「扩展参考档」欠账（`R5-REFERENCE-EXTENSION-CLOSURE.md` 状态
> 「待回填」，`R7-REF25600-CLOSURE.md` 记 `not_measured`）在本轮以**不同路线**闭合——
> 不再放松时间步（CFL 0.8 需过 E1 等价性门，round-7 自己的 A0 探针已指明该路线可疑），
> 而是把 nx=25600 直接跑在**与其余各档同一台仪器**上（`weno5_solve`，SSP-RK3 + Rusanov，CFL 0.4），
> 单作业内完成，因此**不需要**任何入场门。前轮文本一律不改写。
>
> **状态**：`{'已回填（标准档名即冻结仪器产物）' if ok else '未回填——冻结仪器档作业仍在集群队列中，本机未取到任何 nx=25600 数字，E0a/E0b/E1–E5 一律 not_measured（见 ⑤-1）'}`

## ① 案例介绍（物理背景 / 参数表）

Woodward & Colella (1984) 双爆轰波：一维闭管 x∈[0,1]，三压力平台（左 p=1000 / 中 p=0.01 /
右 p=100，ρ=1、u=0），两束爆轰波相向对撞（t≈0.028），积分至 t=0.038，γ=1.4，两端反射壁。
无解析解 ⇒ 误差恒为「两数值解之差」，参考解自身的离散误差必须作为误差棒计入。

| 参数 | 值 |
|---|---|
| 被定级格式 | MUSCL-Hancock 二阶 + MC 限幅 + HLLC + SSP-RK，CFL 0.4 |
| 参考格式 | WENO5 + SSP-RK3 + Rusanov（与被定级格式不共享离散） |
| 参考档位（在位） | {", ".join(str(n) for n in ref["reference_rungs_present"])} |
| **本轮新增档** | `WENO5@{EXTENDED}`，**CFL {FROZEN_CFL}（冻结仪器，非放松）** |
| 比较窗口 | x∈[0.4, 1.0]，守恒限制比较（coarse 单元均值 = r 个 fine 单元均值） |
| 作业 | {json.dumps(s.get("jobs", {}), ensure_ascii=False)} |

## ② 怎么做的（方法 / 流程 / 关键设置）

1. **为什么 round-7 的路线被放弃**：CFL 0.8 把步数减半换来可算性，但要靠 E1 证明
   「CFL 0.8 与 CFL 0.4 是同一台仪器」。round-7 的 A0 先验探针已实测出该假设不成立
   （峰值指标对 CFL 的响应随网格**变大**而非按 `dx^3` 收缩，见 `cfl_scaling_local.json`）。
   因此本轮**换路线不换判据**：直接在 CFL 0.4 上跑该档。
2. **成本核算**：`weno5_solve` 在 nx=25600、CFL 0.8 时实测 `{_f((s.get("cfl08_reference") or {}).get("seconds_solver"), 1)}` s /
   `{(s.get("cfl08_reference") or {}).get("steps")}` 步（round-7 作业 cj3782）⇒ 单步 ~
   `{_f(s.get("cfl08_seconds_per_step"), 5)}` s。
   CFL 0.4 步数加倍，预计 ~`{_f(s.get("predicted_seconds_cfl04"), 0)}` s，在单作业墙钟内，
   故**无需分段重启**。
3. **提交**：集群作业 {json.dumps(s.get("jobs", {}), ensure_ascii=False)}，命令
   `wc_ladder_weno5.py --sizes {CONTROL_NX},{EXTENDED}`——同一作业里顺带重跑控制档 nx={CONTROL_NX}，
   用来证明「今天的仪器 = 库里旧档的仪器」（E0b，见 c）。
4. **落库**：nx=25600 的 profile/metrics 以**标准档名** `ladder_{"{profile,metrics}"}_nx{EXTENDED}_weno5.*`
   入库（不带 `cfl` 标签、无 `promoted_from`），下游所有既有加载器（`wc_refine`、`wc_ref_uncertainty`、
   `wc_ladder*`）无需改动即可看到五级阶梯。
5. **判据**：`PREDECLARATION-r10.md` 在跑之前冻结 E0a/E0b/E1–E5；E1–E5 的定义直接调用
   round-7 模块里的同一份实现（不复制、不重写），保证定义不漂移。

## ③ 计算结果（程序化注入，零手抄）

### a) E0a 档位出处（标准档名是否就是冻结仪器产物）

| 项 | 实测 | 期望 |
|---|---|---|
| cfl | {_f(e0a.get("cfl"), 3)} | {FROZEN_CFL} |
| status | {e0a.get("status")} | ok |
| promoted_from | {e0a.get("promoted_from") or "—（无）"} | 无 |
| method | {e0a.get("method")} | fd-weno5-ssprk3-rusanov |
| steps / 秒 | {e0a.get("steps")} / {_f(e0a.get("seconds_solver"), 1)} | — |
| 正性回退 / 质量漂移 | {e0a.get("fallbacks")} / {_f(e0a.get("mass_drift"), 3)} | 越小越好 |
| **E0a** | **{e0a.get("E0a")}** | pass |

### b) 参考自收敛阶梯（五级，全冻结 CFL 0.4）

| 相邻档 | 窗口 L1 [%] | 峰值差 [%] | 质量差 [%] | 观测阶 | eps_ref [%] |
|---|---|---|---|---|---|
{rows_self}

### c) E0b 仪器可复现性（同作业重跑 nx={CONTROL_NX} vs 库中旧档）

| 项 | 值 |
|---|---|
| 同网格 | {e0b.get("same_grid")}（{_f(e0b.get("rel_delta"), 16)} 相对差） |
| max&#124;Δρ&#124; | {_f(e0b.get("max_abs_drho"), 16)} |
| 逐位相等 | {e0b.get("bitwise_equal")} |
| **E0b** | **{e0b.get("E0b")}** |

### d) 差分幅值的 h^1 收缩（E2 原料）

| 相邻档 | max&#124;D&#124; | 相对前一对的观测阶 | 净质量残差 [%] |
|---|---|---|---|
{rows_def}

### e) 被定级解对两个参考的定级（MUSCL/HLLC）

对 **WENO5@{EXTENDED}（本轮，冻结 CFL）**：

| nx | L1 [%] | 诚实上界 = L1 + eps_ref [%] | 峰值差 [%] | 逐点 L1 [%] | certified(<1 %) |
|---|---|---|---|---|---|
{rows_g}

对 **WENO5@12800（round-4 旧参考）**：

| nx | L1 [%] | 峰值差 [%] |
|---|---|---|
{rows_gp}

### f) 冻结判据逐项

| 编号 | 判据 | 阈值 | 实测 | 判定 |
|---|---|---|---|---|
| E0a | 入库档 = 冻结仪器 | cfl=0.4 ∧ ok ∧ 无 promoted_from | {_f(e0a.get("cfl"), 3)} ∧ {e0a.get("status")} | {e0a.get("E0a")} |
| E0b | 控制档重跑复现旧档 | 相对差 <1e-12 | {_f(e0b.get("rel_delta"), 16)} | {e0b.get("E0b")} |
| E1 | CFL0.8 控制档 vs 冻结档（round-7 门，**仅记录**） | L1 ≤ {_f((c.get("bars") or {}).get("l1_percent"), 6)} % ∧ 峰值 ≤ {_f((c.get("bars") or {}).get("peak_percent"), 6)} % | L1 {_f(c.get("d_l1_percent"), 6)} % / 峰值 {_f(c.get("d_peak_percent"), 6)} % | {e1} |
| E2 | `p_D` ∈ [1.85, 2.15] | — | {_f(e2.get("measured"))} | {e2.get("verdict")} |
| E3 | eps_ref(25600) < 0.5 % | 0.5 | {_f(e3.get("measured"))} | {e3.get("verdict")} |
| E4 | 被定级 L1 诚实上界 < 1.5 % | 1.5 | {_f(e4.get("measured"))} | {e4.get("verdict")} |
| E5 | 可证明 <1 %（上界 <1 %） | 1.0 | {_f(e5.get("measured"))} | {e5.get("verdict")} |

关键派生量：`eps_ref(12800) = {_f(ref["previous_uncertainty"].get("u_ref_percent"))} %` →
`eps_ref(25600) = {_f(ext.get("u_ref_percent"))} %`；
被定级最细档 nx={fin.get("nx")}：L1 = {_f(fin.get("l1_percent"))} %，
上界 = {_f(fin.get("upper_bound_percent"))} %，峰值差 = {_f(fin.get("peak_percent"))} %。

## ④ 结果图

- `figures/fig_r10_reference_frozen.png` — (a) 五级参考自收敛；(b) 差分幅值 vs 对；(c) 被定级族对 12800/25600 的定级与诚实上界。
- `figures/fig_r10_profiles_25600_frozen.png` — 对撞区剖面：WENO5@12800 / WENO5@25600 / MUSCL-HLLC@6400。

## ⑤ 注意点（局限 / 踩过的坑 / 适用范围）

1. **本轮出库时的实测状态**：冻结仪器档作业（见上表 `作业`）**仍在排队**，本机未取到任何
   nx=25600 数字。因此 E0a/E0b/E1–E5 全部记 `not_measured`——**不是** PASS，也**不是** FAIL。
   本轮交付的是「路线 + 先冻结判据 + 自动裁决器 + 回归门 + 已取回的旁证」；档位数字待作业返回后
   由同一脚本一次注入即可（`ref25600_r10.py` 幂等，判据不因数据到位而改动）。
2. **"可算性"与"仪器同一性"是两件事**：round-7 想用放松 CFL 换吞吐，代价是把参考档的定义改掉；
   本轮用同一台仪器硬跑，代价只是墙钟时间（
   {_f(e0a.get("seconds_solver"), 0)} s），换来的是**不需要任何入场门**——这是本轮方法论上的结论。
3. **E1 仍然照实记录**：CFL0.8 控制档一旦测出，就写进本表，但**不参与**入库判定；
   若 E1 判 fail，正好反向印证 round-7 的 A0 探针，而不是本轮结论的依据。
4. **参考侧仍有误差**：`eps_ref(25600)` 不是 0，所以"<1 %"必须区分**点估计**与**可证明上界**，
   本轮给出上界；上界不达标就照实写，不拿点估计冒充。
5. **最细档仍是外推**：五级阶梯给出的是 Richardson 估计（无 GCI 安全系数），
   `eps_ref` 依赖观测阶 `p` 的稳定性；若 `p` 抖动，误差棒随之抖动（逐档 `order` 见 a/b 表）。
6. **窗口与范数**：逐点 L1 对激波抹平敏感，输运范数（`max|D|`）对搬移不敏感，两者不能互相顶替。

## ⑥ 误差判定（逐项显式回答）

| 关键量 | 实测 | <15 %（总账） | <3 % | <1.5 % | <1 %（可证明） |
|---|---|---|---|---|---|
| 参考自收敛 L1（12800→25600） | {_f(next((r['l1_percent'] for r in ref['selfcert_pairs'] if r['fine_nx'] == EXTENDED), None))} % | {_f((next((r['l1_percent'] for r in ref['selfcert_pairs'] if r['fine_nx'] == EXTENDED), None) or 99) < 15)} | {_f((next((r['l1_percent'] for r in ref['selfcert_pairs'] if r['fine_nx'] == EXTENDED), None) or 99) < 3)} | — | {_f((next((r['l1_percent'] for r in ref['selfcert_pairs'] if r['fine_nx'] == EXTENDED), None) or 99) < 1)} |
| eps_ref(25600) | {_f(ext.get("u_ref_percent"))} % | {_f((ext.get('u_ref_percent') or 99) < 15)} | {_f((ext.get('u_ref_percent') or 99) < 3)} | {_f((ext.get('u_ref_percent') or 99) < 1.5)} | {_f((ext.get('u_ref_percent') or 99) < 1)} |
| 被定级 L1 点估计 @nx={fin.get("nx")} | {_f(fin.get("l1_percent"))} % | {_f((fin.get('l1_percent') or 99) < 15)} | {_f((fin.get('l1_percent') or 99) < 3)} | {_f((fin.get('l1_percent') or 99) < 1.5)} | — |
| 被定级 L1 诚实上界 @nx={fin.get("nx")} | {_f(fin.get("upper_bound_percent"))} % | {_f((fin.get('upper_bound_percent') or 99) < 15)} | {_f((fin.get('upper_bound_percent') or 99) < 3)} | {_f((fin.get('upper_bound_percent') or 99) < 1.5)} | {_f((fin.get('upper_bound_percent') or 99) < 1)} |
| 峰值差 @nx={fin.get("nx")} | {_f(fin.get("peak_percent"))} % | {_f((fin.get('peak_percent') or 99) < 15)} | {_f((fin.get('peak_percent') or 99) < 3)} | {_f((fin.get('peak_percent') or 99) < 1.5)} | {_f((fin.get('peak_percent') or 99) < 1)} |

**是否 <15 %（owner 验收线）？** {('是' if (fin.get('l1_percent') or 99) < 15 else '否')}——被定级解 L1 点估计 {_f(fin.get("l1_percent"))} %、
诚实上界 {_f(fin.get("upper_bound_percent"))} %、峰值差 {_f(fin.get("peak_percent"))} %。
**是否 <3 %？** 见上表逐项。
**是否可证明 <1 %？** {('是' if (fin.get('upper_bound_percent') or 99) < 1 else '否')}——
上界 {_f(fin.get("upper_bound_percent"))} %（= L1 {_f(fin.get("l1_percent"))} % + eps_ref {_f(ext.get("u_ref_percent"))} %）。

## 附：判据对账（先冻结后测量）

E0a `{e0a.get("E0a")}` · E0b `{e0b.get("E0b")}` · E1 `{e1}` · E2 `{e2.get("verdict")}` ·
E3 `{e3.get("verdict")}` · E4 `{e4.get("verdict")}` · E5 `{e5.get("verdict")}`。

## 复现

```bash
# 1) 冻结仪器重跑控制档 + 扩展档（集群，1 核，单作业）
PYTHONPATH=src python docs/woodward-colella-double-blast/ladder/wc_ladder_weno5.py --sizes {CONTROL_NX},{EXTENDED}
# 2) 回填 + 裁决 + 图 + 本文
PYTHONPATH=src python docs/XVAL-FVM-M18737-ROUND2/ref25600_r10.py
# 3) 回归门
PYTHONPATH=src python -m pytest tests/test_compressible_wc_ref25600_frozen.py -q
```
"""


# --------------------------------------------------------------------------- #
# driver
# --------------------------------------------------------------------------- #
def build(jobs: Optional[Dict] = None, write: bool = True) -> Dict:
    e0a = provenance()
    e0b = reproducibility()
    adj = adjudicate()
    c, ref, g = adj["control"], adj["reference"], adj["regrade"]
    tagged_met = DATA / f"ladder_metrics_nx{EXTENDED}_weno5_cfl0.8.json"
    cfl08 = json.loads(tagged_met.read_text()) if tagged_met.exists() else {}
    s: Dict = {
        "anchor": "XVAL-FVM-M18737-ROUND2", "round": 10,
        "jobs": jobs or {},
        "provenance": e0a, "reproducibility": e0b,
        "control": c, "promotion": adj["promotion"],
        "reference": ref, "regrade": g,
        "cfl08_reference": {k: cfl08.get(k) for k in ("steps", "seconds_solver", "cfl")},
        "predicted_seconds_cfl04": (cfl08.get("seconds_solver", 0) * 2
                                    if cfl08.get("seconds_solver") else None),
        "cfl08_seconds_per_step": ((cfl08["seconds_solver"] / cfl08["steps"])
                                   if cfl08.get("seconds_solver") and cfl08.get("steps") else None),
        "verdicts": {"E0a": e0a.get("E0a"), "E0b": e0b.get("E0b"), "E1": c.get("E1"),
                     "E2": ref["E2"].get("verdict"), "E3": ref["E3"].get("verdict"),
                     "E4": g["E4"].get("verdict"), "E5": g["E5"].get("verdict")},
    }
    s["figures"] = figures(s)
    if write:
        (HERE / "summary_r10.json").write_text(json.dumps(s, indent=1, default=float) + "\n")
        (HERE / "R10-REF25600-FROZEN-CLOSURE.md").write_text(render(s))
    return s


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--jobs", default='{"rung": "cj4143", "control": "cj4141"}')
    args = ap.parse_args(argv)
    s = build(jobs=json.loads(args.jobs))
    print(json.dumps({"verdicts": s["verdicts"],
                      "eps_ref_25600": s["reference"]["extended_uncertainty"].get("u_ref_percent"),
                      "finest": s["regrade"].get("finest"),
                      "provenance": {k: s["provenance"].get(k) for k in ("cfl", "status", "steps", "seconds_solver")},
                      "repro": {k: s["reproducibility"].get(k) for k in ("rel_delta", "bitwise_equal")}},
                     indent=1, default=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())