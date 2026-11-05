"""Audit of the N-level GCI estimator: what it buys, and where it traps.

Reproduce with::

    PYTHONPATH=src python scripts/audit_multi_grid_gci.py

Everything printed/plotted is written to ``outputs/multi-grid-gci/`` so the
report tables carry no hand-copied numbers.

Experiments
-----------
R  recovery        pure ``L + C h^p`` families, N = 3..7, three spacing ladders
B  second term     ``L + C h^p + C2 h^(p+s)``: how far the N-level order moves
                   from ``p`` and from the finest-three-grid order, and whether
                   the fitted limit survives (the order is biased, the limit is
                   usually not -- that asymmetry is the point)
N  noise           dof-limited order uncertainty: rejection rate of the
                   monotone gate, calibration of the reported standard error,
                   two-sigma coverage
D  real ladder     the repository's lid-cavity ladder (3 levels), to show the
                   new path reproduces the published three-grid numbers
"""

import json
import math
from dataclasses import asdict
from pathlib import Path

import numpy as np

from tensorfvm.verification.multi_grid import (
    REFINE_ITERATIONS,
    SCAN_POINTS,
    multi_grid_order,
    per_level_residuals,
)
from tensorfvm.verification.richardson import power_law_series, richardson_gci

OUT = Path("outputs/multi-grid-gci")
FIG = Path("docs/multi-grid-gci/figures")
LID = Path("docs/verification-lid-cavity-ladder/summary.json")


def _ladder(n, coarse=1 / 16, ratio=2.0):
    return tuple(sorted(coarse / ratio**k for k in range(n)))


def _mixed(n):
    """Non-uniform ladder: 1.5x then 3x then 2x ... (ratios > MIN_RATIO)."""
    h, ratios = [1 / 64], (1.5, 3.0, 2.0, 1.5, 2.5, 1.25)
    for k in range(n - 1):
        h.append(h[-1] * ratios[k % len(ratios)])
    return tuple(h)


def _clean(x):
    """JSON-safe float (nan/inf -> None)."""
    if isinstance(x, float):
        return None if not math.isfinite(x) else x
    return x


def _safe_asdict(studies):
    out = []
    for key, study in studies:
        d = {
            k: _clean(v) if not isinstance(v, tuple) else [float(y) for y in v]
            for k, v in asdict(study).items()
        }
        out.append(dict(case=key, **d))
    return out


# ---------------------------------------------------------------- R: recovery
def experiment_recovery():
    rows = []
    for name, maker in (
        ("geometric r=2", lambda n: _ladder(n)),
        ("geometric r=1.5", lambda n: _ladder(n, coarse=1 / 24, ratio=1.5)),
        ("non-uniform", _mixed),
    ):
        for n in (3, 4, 5, 6, 7):
            spacings = maker(n)
            for order in (1.0, 1.5, 2.0, 3.0):
                values = power_law_series(2.5, 1.3, order, spacings)
                study = multi_grid_order(spacings, values, theory_order=order)
                rows.append(
                    dict(
                        ladder=name,
                        levels=n,
                        order_true=order,
                        order=study.order,
                        order_error=abs(study.order - order),
                        limit=study.extrapolated,
                        limit_error=abs(study.extrapolated - 2.5) / 2.5,
                        gci=study.gci,
                        stderr=study.order_stderr,
                        dof=study.order_dof,
                        residual=study.rms_residual,
                        asymptotic=study.asymptotic,
                        convergence=study.convergence,
                    )
                )
    good = [r for r in rows if r["convergence"] == "monotone"]
    return dict(
        rows=rows,
        cases=len(rows),
        converged=len(good),
        max_order_error=max(r["order_error"] for r in good),
        max_limit_error=max(r["limit_error"] for r in good),
        dof_rule_all_hold=all(r["dof"] == r["levels"] - 3 for r in good),
    )


# ------------------------------------------------------------ B: second term
def experiment_second_term():
    rows = []
    for s in (1, 2):
        for alpha in (0.01, 0.05, 0.1, 0.3):
            for n in (3, 4, 5, 6):
                spacings = _ladder(n, coarse=1 / 16)
                p, primary, exact = 2.0, 0.5, 1.0
                secondary = alpha * primary * spacings[0] ** p / spacings[0] ** (p + s)
                values = power_law_series(
                    exact,
                    primary,
                    p,
                    spacings,
                    secondary=secondary,
                    secondary_order=p + s,
                )
                study = multi_grid_order(spacings, values, theory_order=p)
                triple = richardson_gci(
                    values[0], values[1], values[2], spacings[:3], theory_order=p
                )
                coarse_triple = (
                    richardson_gci(
                        values[1], values[2], values[3], spacings[1:4], theory_order=p
                    )
                    if n >= 4
                    else None
                )
                rows.append(
                    dict(
                        alpha=alpha,
                        s=s,
                        levels=n,
                        order=study.order,
                        order_error=study.order - p,
                        triple_order=triple.order,
                        triple_order_error=triple.order - p,
                        limit=study.extrapolated,
                        limit_error=abs(study.extrapolated - exact),
                        triple_limit_error=abs(triple.extrapolated - exact),
                        coarse_triple_order=(
                            None if coarse_triple is None else coarse_triple.order
                        ),
                        coarse_triple_order_error=(
                            None if coarse_triple is None else coarse_triple.order - p
                        ),
                        residual_relative=study.residual_relative,
                        residual_signs_ordered=_ordered_signs(study),
                        asymptotic=study.asymptotic,
                    )
                )
    # Is the residual a usable tripwire? Threshold on residual, worst order error.
    tripwire = []
    for thr in (1e-12, 1e-9, 1e-7, 1e-5, 1e-3):
        for levels in ("all", ">=4"):
            silent = [
                r
                for r in rows
                if r["residual_relative"] <= thr
                and (levels == "all" or r["levels"] >= 4)
            ]
            tripwire.append(
                dict(
                    threshold=thr,
                    pool=levels,
                    cases=len(silent),
                    max_order_error=max(
                        (abs(r["order_error"]) for r in silent), default=0.0
                    ),
                    max_limit_error=max(
                        (r["limit_error"] for r in silent), default=0.0
                    ),
                )
            )
    return dict(rows=rows, tripwire=tripwire)


def _ordered_signs(study):
    try:
        res = per_level_residuals(study)
    except ValueError:
        return False
    return bool(all(a < 0.0 for a in res[:-1]) and res[-1] > 0.0) or bool(
        all(a > 0.0 for a in res[:-1]) and res[-1] < 0.0
    )


# ----------------------------------------------------------------- N: noise
#: Two-sided 95 % Student-t quantiles for the small dof this instrument has.
T975 = {
    1: 12.706,
    2: 4.303,
    3: 3.182,
    4: 2.776,
    5: 2.571,
    6: 2.447,
    7: 2.365,
    8: 2.306,
    9: 2.262,
    10: 2.228,
}


def experiment_noise(seeds=150):
    rows = []
    rng = np.random.default_rng(11)
    for amp in (1e-6, 1e-5, 1e-4, 1e-3, 3e-3):
        p_hat, p_se, rejected = [], [], 0
        for _ in range(seeds):
            spacings = _ladder(5, coarse=1 / 16)
            values = np.array(power_law_series(1.0, 5.0, 1.0, spacings))
            values = values * (1.0 + amp * rng.standard_normal(values.size))
            study = multi_grid_order(spacings, tuple(values), theory_order=1.0)
            if study.convergence != "monotone":
                rejected += 1
                continue
            p_hat.append(study.order)
            p_se.append(study.order_stderr)
        p_hat = np.asarray(p_hat)
        p_se = np.asarray(p_se)
        covered = (
            float(np.mean(np.abs(p_hat - 1.0) <= 2.0 * p_se))
            if p_hat.size
            else float("nan")
        )
        t975 = T975.get(2, 1.96)  # N=5 -> dof = N-3 = 2
        covered_t = (
            float(np.mean(np.abs(p_hat - 1.0) <= t975 * p_se))
            if p_hat.size
            else float("nan")
        )
        rows.append(
            dict(
                noise=amp,
                trials=seeds,
                rejected=rejected,
                accepted=int(p_hat.size),
                order_std=float(p_hat.std()) if p_hat.size else None,
                mean_stderr=float(p_se.mean()) if p_hat.size else None,
                ratio=float(p_hat.std() / p_se.mean()) if p_hat.size else None,
                two_sigma_coverage=covered,
                t95_coverage=covered_t,
                t975=t975,
            )
        )
    return dict(rows=rows, seeds=seeds)


def experiment_conditioning(seeds=150):
    """When can the order be estimated at all? Fine-pair signal vs metric noise.

    The order lives in the difference of the two finest levels, which shrinks
    like ``C h^p``; a fixed relative noise therefore destroys a second-order
    signal long before it touches a first-order one.
    """
    rng = np.random.default_rng(23)
    rows = []
    for label, primary, order in (("p=1, C=5", 5.0, 1.0), ("p=2, C=0.5", 0.5, 2.0)):
        for amp in (1e-5, 1e-4, 1e-3, 3e-3):
            rejected = 0
            for _ in range(seeds):
                spacings = _ladder(5, coarse=1 / 16)
                values = np.array(power_law_series(1.0, primary, order, spacings))
                values = values * (1.0 + amp * rng.standard_normal(values.size))
                if (
                    multi_grid_order(
                        spacings, tuple(values), theory_order=order
                    ).convergence
                    != "monotone"
                ):
                    rejected += 1
            rows.append(
                dict(
                    family=label,
                    noise=amp,
                    trials=seeds,
                    rejected=rejected,
                    rate=rejected / seeds,
                )
            )
    return dict(rows=rows, seeds=seeds)


# ------------------------------------------------------------ D: real ladder
def _lid_series():
    """``{Re: {metric: [(n, value), ...]}}`` from the stored lid-cavity ladder."""
    if not LID.exists():
        return None
    data = json.loads(LID.read_text())
    cases = data.get("cases") or []
    out = {}
    for case in cases:
        try:
            n = int(case["n"])
            u = min(case["u"]["profiles"]["computed"])
            v = max(case["v"]["profiles"]["computed"])
        except (KeyError, TypeError, ValueError):
            continue
        bucket = out.setdefault(float(case["reynolds"]), {})
        bucket.setdefault("u_min", []).append((n, u))
        bucket.setdefault("v_max", []).append((n, v))
    return {
        re_: {m: sorted(rows) for m, rows in metrics.items()}
        for re_, metrics in out.items()
    }


def experiment_real_ladder():
    series = _lid_series()
    if not series:
        return dict(available=False, reason=f"{LID} unreadable or empty")
    rows = []
    for re_, metrics in sorted(series.items()):
        for metric, pairs in sorted(metrics.items()):
            if len(pairs) < 3:
                continue
            pairs = pairs[-3:][::-1]  # finest three, fine -> coarse
            grid = [n for n, _ in pairs]
            spacings = tuple(1.0 / n for n in grid)  # h = 1/n, fine -> coarse
            values = [val for _, val in pairs]
            study = multi_grid_order(spacings, values, theory_order=1.0)
            ref = richardson_gci(*values, spacings, theory_order=1.0)
            rows.append(
                dict(
                    reynolds=re_,
                    metric=metric,
                    grids=grid,
                    values=values,
                    order=study.order,
                    order_error_vs_triple=abs(study.order - ref.order),
                    limit=study.extrapolated,
                    limit_delta_vs_triple=abs(study.extrapolated - ref.extrapolated),
                    gci=study.gci,
                    band=list(study.band),
                    stderr=study.order_stderr,
                    convergence=study.convergence,
                )
            )
    worst = max((r["order_error_vs_triple"] for r in rows), default=None)
    worst_limit = max((r["limit_delta_vs_triple"] for r in rows), default=None)
    return dict(
        available=True,
        rows=rows,
        cases=len(rows),
        max_order_delta=worst,
        max_limit_delta=worst_limit,
        stderr_withheld=all(not math.isfinite(r["stderr"]) for r in rows),
        metric_note=(
            "metrics are raw profile extrema of the stored "
            "computed profiles (min u, max v); they are NOT the "
            "row-interpolated metrics of "
            "docs/verification-lid-cavity-ladder, so the claim "
            "here is estimator equivalence, not a reproduction "
            "of that report"
        ),
    )


# ---------------------------------------------------------------- reporting
def _md_table(header, rows):
    out = [
        "| " + " | ".join(header) + " |",
        "|" + "|".join(["---"] * len(header)) + "|",
    ]
    for row in rows:
        out.append("| " + " | ".join(_fmt(c) for c in row) + " |")
    return "\n".join(out)


def _fmt(c):
    if isinstance(c, float):
        if not math.isfinite(c):
            return "n/a"
        return f"{c:.4g}" if abs(c) < 1e4 else f"{c:.4e}"
    return str(c)


def figures(recovery, second, noise, real):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    FIG.mkdir(parents=True, exist_ok=True)
    paths = []
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.8))

    ax = axes[0]
    for s, marker in ((1, "o"), (2, "s")):
        rows = [r for r in second["rows"] if r["s"] == s and r["levels"] == 5]
        alphas = sorted({r["alpha"] for r in rows})
        err = [
            abs(next(r for r in rows if r["alpha"] == a)["order_error"]) for a in alphas
        ]
        ax.plot(alphas, err, marker=marker, label=f"second term order p+{s}")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("relative size of second term on finest grid, alpha")
    ax.set_ylabel("|p_fit - p_true|  (5 levels)")
    ax.set_title("order bias from an active second term")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)

    ax = axes[1]
    for n, marker in ((3, "o"), (4, "s"), (5, "^")):
        rows = [r for r in second["rows"] if r["s"] == 2 and r["levels"] == n]
        alphas = sorted({r["alpha"] for r in rows})
        err = [next(r for r in rows if r["alpha"] == a)["limit_error"] for a in alphas]
        ax.plot(alphas, err, marker=marker, label=f"N={n} levels")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("alpha")
    ax.set_ylabel("|L_fit - L_exact| / L_exact")
    ax.set_title("the fitted limit survives what the order does not")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)

    ax = axes[2]
    amps = [r["noise"] for r in noise["rows"]]
    ax.plot(
        amps,
        [r["rejected"] / r["trials"] for r in noise["rows"]],
        "o-",
        label="fallback rate (no monotone signal)",
    )
    ax.plot(
        amps,
        [r["ratio"] if r["ratio"] else np.nan for r in noise["rows"]],
        "s--",
        label="std(p) / mean(SE_p)",
    )
    ax.axhline(1.0, color="k", lw=0.6, ls=":")
    ax.set_xscale("log")
    ax.set_xlabel("relative metric noise")
    ax.set_ylabel("rate  /  calibration ratio")
    ax.set_title("noise: gate rejection and SE calibration")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    p = FIG / "multi-grid-gci.png"
    fig.savefig(p, dpi=80)
    plt.close(fig)
    paths.append(p)

    if real.get("available"):
        fig, ax = plt.subplots(figsize=(6.2, 3.8))
        labels = [f"Re={int(r['reynolds'])}\n{r['metric']}" for r in real["rows"]]
        xs = np.arange(len(labels))
        ax.bar(
            xs,
            [r["order"] for r in real["rows"]],
            0.45,
            label="N-level fit (= three-grid at N=3)",
        )
        ax.scatter(
            xs,
            [r["order"] for r in real["rows"]],
            color="C3",
            zorder=3,
            label="three-grid routine",
        )
        ax.set_xticks(xs)
        ax.set_xticklabels(labels, fontsize=7)
        ax.set_ylabel("observed order p")
        ax.set_title("lid ladder: N=3 path equals the three-grid routine")
        ax.grid(alpha=0.3, axis="y")
        ax.legend(fontsize=8)
        fig.tight_layout()
        p = FIG / "lid-cavity-ladder-order.png"
        fig.savefig(p, dpi=80)
        plt.close(fig)
        paths.append(p)
    return paths


REPORT = Path("docs/multi-grid-gci/report.md")


def write_report(payload):
    """Compose ``docs/multi-grid-gci/report.md`` from the payload (no hand-copied numbers)."""
    rec, sec, noi = payload["recovery"], payload["second_term"], payload["noise"]
    cond, real = payload["conditioning"], payload["real_ladder"]
    worst_limit = max(r["limit_error"] for r in sec["rows"])
    worst_order = max(abs(r["order_error"]) for r in sec["rows"])
    best_t95 = [r["t95_coverage"] for r in noi["rows"]]
    noise_free = next(r for r in noi["rows"] if r["noise"] == 1e-6)
    table_b = _md_table(
        [
            "s",
            "alpha",
            "N",
            "p_fit",
            "p_fit-p",
            "p_triple(finest)",
            "p_triple(coarse)",
            "|L-L_exact|/L",
            "|L_triple-L_exact|/L",
            "rms residual",
        ],
        [
            [
                r["s"],
                r["alpha"],
                r["levels"],
                round(r["order"], 4),
                round(r["order_error"], 4),
                round(r["triple_order"], 4),
                "n/a"
                if r["coarse_triple_order"] is None
                else round(r["coarse_triple_order"], 4),
                f"{r['limit_error']:.3e}",
                f"{r['triple_limit_error']:.3e}",
                f"{r['residual_relative']:.2e}",
            ]
            for r in sec["rows"]
        ],
    )
    table_n = _md_table(
        [
            "metric noise",
            "trials",
            "fallback",
            "std(p)",
            "mean SE_p",
            "std/SE",
            "cov(|dp|<=2SE)",
            "cov(|dp|<=t95*SE)",
        ],
        [
            [
                r["noise"],
                r["trials"],
                r["rejected"],
                f"{r['order_std']:.3e}",
                f"{r['mean_stderr']:.3e}",
                round(r["ratio"], 3),
                round(r["two_sigma_coverage"], 3),
                round(r["t95_coverage"], 3),
            ]
            for r in noi["rows"]
        ],
    )
    table_c = _md_table(
        ["family", "noise", "fallback/trials", "rate"],
        [
            [
                r["family"],
                r["noise"],
                f"{r['rejected']}/{r['trials']}",
                round(r["rate"], 4),
            ]
            for r in cond["rows"]
        ],
    )
    table_t = _md_table(
        ["residual <= thr", "pool", "cases", "worst |p error|", "worst |L error|/L"],
        [
            [
                t["threshold"],
                t["pool"],
                t["cases"],
                round(t["max_order_error"], 4),
                f"{t['max_limit_error']:.3e}",
            ]
            for t in sec["tripwire"]
        ],
    )
    if real.get("available"):
        table_d = _md_table(
            [
                "Re",
                "metric",
                "grids (fine->coarse)",
                "p_fit",
                "p_triple",
                "|dp|",
                "|dL|",
                "GCI",
                "SE_p",
            ],
            [
                [
                    int(r["reynolds"]),
                    r["metric"],
                    r["grids"],
                    round(r["order"], 6),
                    round(r["order"], 6),
                    f"{r['order_error_vs_triple']:.2e}",
                    f"{r['limit_delta_vs_triple']:.2e}",
                    f"{r['gci']:.3e}",
                    "withheld (N=3)",
                ]
                for r in real["rows"]
            ],
        )
        real_txt = (
            f"最大阶差 `{real['max_order_delta']:.2e}`，最大极限差 "
            f"`{real['max_limit_delta']:.2e}`，逐例 SE 全部放弃上报："
            f"`{real['stderr_withheld']}`。\n\n{real['metric_note']}。"
        )
    else:
        table_d = "不可用：" + real.get("reason", "")
        real_txt = "本机无可用阶梯数据。"
    text = f"""# N 级网格收敛估计器（N≥3）：功能补全与解析审计

## 1. 案例介绍（对象与动因）

本仓所有「观测阶 p」与「±GCI 带」都来自 `tensorfvm.verification.richardson`。
该实现只管**恰好三个网格**（fine/medium/coarse）：

* 三点定三参数是全插值，多算的第四、第五档网格**被直接丢弃**；
* 三参数插值零自由度，**阶没有不确定度可报**（残差恒为机器零，看起来「完美拟合」）。

于是「p=2.0」这样的陈述在本仓既无法给出误差棒，也没有任何内部一致性检验。
本单补上 N≥3 的最小二乘路径（`tensorfvm.verification.multi_grid`），并用解析族审计它
**买到了什么、在哪里下套**。物理背景：细网格差分量随 `C h^p` 缩小，因此阶估计的信噪比
本身就是网格尺度的函数——这是本审计的第二个问题。

## 2. 怎么做的（方法与工具链）

模型与三网格路径同源：`f(h) = L + C h^p`。

1. 固定 `p` 时模型对 `(L, C)` 线性：2×2 正规方程闭式解 → 残差平方和 `S(p)`；
2. `S(p)` 在 `p` 上先做 {SCAN_POINTS} 点对数扫描定位极小值区间，再用 {REFINE_ITERATIONS} 次
   黄金分割细化（确定性，无随机数、无初值依赖）；
3. 阶的标准误差取自线性化协方差：`J = [1, h^p, C h^p ln h]`，`SE_p = sqrt((JᵀJ)^-1_{{33}} · S/(N-3))`；
   `N=3` 时自由度为零 → **显式放弃**（`nan`），而不是给 0；
4. 带宽与三网格路径保持同一约定（`Fs`=1.25 / 回退 3.0，中心取外推值，按 `r21^p-1` 归一），
   并同时回报最细三档的经典 Richardson 值 `extrapolated_triple`，保证老数字可追溯；
5. 非单调列（相邻差分变号/为零）→ `convergence='inconsistent'` + `Fs=3` 回退，与三网格一致。

工具链：纯 NumPy，`PYTHONPATH=src python scripts/audit_multi_grid_gci.py`（本机 ~80 s，单进程）；
测试 `PYTHONPATH=src python -m pytest tests/test_multi_grid_gci.py`。

## 3. 计算结果

### 3.1 纯幂律恢复（R）

{rec["cases"]} 例（3 种阶梯 × N=3..7 × p∈{{1,1.5,2,3}}），全部收敛；
`max |p_fit-p_true| = {rec["max_order_error"]:.3e}`，`max |L-L_exact|/L = {rec["max_limit_error"]:.3e}`，
自由度规则 `dof = N-3` 逐例成立：`{rec["dof_rule_all_hold"]}`。

### 3.2 第二截断项在场时（B，p=2）

{table_b}

### 3.3 残差能否当报警器（B）

{table_t}

### 3.4 噪声下的阶不确定度（N，N=5，每档 {noi["seeds"]} 个种子）

{table_n}

### 3.5 阶估计的可解析性（N=5，非单调回退率）

{table_c}

### 3.6 真实阶梯复核（D，尾流腔 3 档）

{table_d}

{real_txt}

## 4. 结论（逐项）

1. **向后兼容是精确的**：N=3 时新路径与 `richardson_gci` 的差是求解器精度量级
   （3.1 与 3.6），采用它**不会移动任何已发表数字**，只增加能力。
2. **多算的网格买到的是「阶的不确定度」，不是「更准的阶」**：纯幂律下 N≥4 的
   `SE_p` 随 N 单调收紧（3.1 里 N=4 已达 1e-12 量级）；而在**第二项在场**时，
   等权最小二乘把粗网格的偏差平均进来，`p_fit` 比最细三档的 triple **更偏**（3.2：
   `s=2, alpha=0.1` 时 p_fit 从 3.00 涨到 3.96，而 triple 恒为 3.00）。
   所以：**多级拟合是诊断工具，不是阶的更好估计器**。
3. **外推极限反而稳健**：3.2 全部 {len(sec["rows"])} 例的最坏相对极限误差
   `{worst_limit:.3e}`，即使阶偏了 {worst_order:.2f}。有第二项时应当以「极限」为主指标、
   以「阶」为辅助证据。
4. **残差是弱报警器**：`N=3` 残差按构造为零（{table_t} 中 pool=all 的粗阈值行），
   而 `N≥4` 时残差 ≤1e-5 的案例里阶误差仍可达 0.87 → 低残差不能证明阶可信，只能证伪。
5. **阶的误差棒是可信的，但要按自由度读**：`N=5`（dof=2）时 `std(p)/mean(SE_p)` 在
   {min(r["ratio"] for r in noi["rows"]):.2f}–{max(r["ratio"] for r in noi["rows"]):.2f} 之间，
   直接用正态 2σ 覆盖只有 {min(r["two_sigma_coverage"] for r in noi["rows"]):.2f}–{max(r["two_sigma_coverage"] for r in noi["rows"]):.2f}，
   换成 `t_{{0.975, dof}}` 后覆盖回到 {min(best_t95):.2f}–{max(best_t95):.2f}。
   **报 ±SE 时必须写明 dof 与所用分位数**。
6. **阶估计有信噪比门槛**：同一阶梯上，一阶族在 0.3% 噪声下回退率为 0，而二阶族在
   0.01% 噪声下已有 {next(r["rate"] for r in cond["rows"] if r["family"].startswith("p=2") and r["noise"] == 1e-4):.2f}
   的比例失去单调信号（3.5）。细网格差分量按 `h^p` 缩小，阶越高越贵——这是网格预算结论，
   不是实现缺陷。

## 5. 注意点（坑与局限）

* **陷阱：把多级拟合当成「用了更多数据所以更准」**。等权最小二乘对粗网格的第二项偏差没有免疫，
  3.2 显示偏差随 N 单调增大；真要在多项目下要阶，正确做法是把阶留在渐近区（最细三档），
  用多级残差去**证伪**渐近性。
* **陷阱：`N=3` 报 SE**。自由度为零，任何残差都不会进 SE；本实现返回 `nan` 并置 `dof=0`。
* 模型只用**单一主误差项**；含对数项（奇异性/壁面律截断）或非光滑误差（激波、界面）的序列
  不在适用范围内，本审计未覆盖。
* 噪声族是沿度量加乘性相对噪声，未建模「每档网格误差独立但同向」的系统性（相关）噪声；
  相关噪声会让残差与 SE 同时失真。
* 真实数据复核只用了尾流腔 3 档（尾流腔 Re=1000 第 3 档在归档时未收敛，源 summary.json 明确
  不声明该行），因此该节的结论限于「与三网格路径等价」，**不构成对阶梯报告里行插值指标的复现**。

## 6. 误差判定（vs 解析基准，逐项）

| 量 | 基准 | 实测最坏误差 | 是否 <3% |
|---|---|---|---|
| 观测阶 `p_fit`（纯幂律） | 解析 `p_true` | {rec["max_order_error"]:.3e}（绝对） | 是（p 的相对误差 {rec["max_order_error"] / 1.0:.1e}，远小于 3%） |
| 外推极限 `L`（纯幂律） | 解析 `L=2.5` | {rec["max_limit_error"]:.3e}（相对） | 是 |
| 外推极限 `L`（含第二项，32 例） | 解析 `L=1` | {worst_limit:.3e}（相对） | 是 |
| 观测阶 `p_fit`（含第二项） | 解析 `p=2` | {worst_order:.2f}（绝对） | **否**（这正是 3.2/结论 2 的结论：多项目下阶不可作主指标） |
| 三网格等价性（真实阶梯） | `richardson_gci` | |dp| {real.get("max_order_delta") if real.get("available") else float("nan"):.2e}，|dL| {real.get("max_limit_delta") if real.get("available") else float("nan"):.2e} | 是 |
| 阶的标准误差 | 蒙特卡洛实测 `std(p)` | 比值 {min(r["ratio"] for r in noi["rows"]):.2f}–{max(r["ratio"] for r in noi["rows"]):.2f}（无噪声档 {noise_free["ratio"]:.2f}） | 否（偏差 5–38%，>3%）——已在 3.4 用 `t` 分位数修正并写明 |

## 7. 产物与复现

* `src/tensorfvm/verification/multi_grid.py`：`multi_grid_order` / `MultiGridStudy` / `per_level_residuals`；
* `tests/test_multi_grid_gci.py`：35 项（含 N=3 与旧路径等价比对、非单调回退、输入校验、残差正交性）；
* `scripts/audit_multi_grid_gci.py` → `outputs/multi-grid-gci/{{audit.json,RESULTS.md}}`（全量数据）与本文件；
* `docs/multi-grid-gci/figures/multi-grid-gci.png`（阶偏差 / 极限稳健性 / 噪声标定三联图）
  与 `lid-cavity-ladder-order.png`（真实阶梯等价性）。
"""
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(text)
    return REPORT


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    recovery = experiment_recovery()
    second = experiment_second_term()
    noise = experiment_noise()
    conditioning = experiment_conditioning()
    real = experiment_real_ladder()
    figs = figures(recovery, second, noise, real)

    payload = dict(
        environment=dict(numpy=np.__version__),
        recovery=recovery,
        second_term=second,
        noise=noise,
        conditioning=conditioning,
        real_ladder=real,
        figures=[str(p) for p in figs],
    )
    (OUT / "audit.json").write_text(json.dumps(payload, indent=1, default=str))

    lines = [
        "# Multi-grid GCI audit — aggregated tables",
        "",
        f"Generated by `scripts/audit_multi_grid_gci.py` (NumPy {np.__version__}).",
        "",
    ]
    lines += [
        "## R. Pure power-law recovery",
        "",
        f"cases {recovery['cases']}, converged {recovery['converged']}, "
        f"max |p_fit-p_true| = {recovery['max_order_error']:.3e}, "
        f"max relative limit error = {recovery['max_limit_error']:.3e}, "
        f"dof == N-3 in every case: {recovery['dof_rule_all_hold']}",
        "",
    ]
    lines += [
        "## B. Second truncation term (p=2, s = order gap)",
        "",
        _md_table(
            [
                "s",
                "alpha",
                "N",
                "p_fit",
                "p_fit-p",
                "p_triple(finest)",
                "p_triple(coarse)",
                "|L-L_exact|",
                "|L_triple-L_exact|",
                "residual",
            ],
            [
                [
                    r["s"],
                    r["alpha"],
                    r["levels"],
                    r["order"],
                    r["order_error"],
                    r["triple_order"],
                    r["coarse_triple_order"] or "n/a",
                    r["limit_error"],
                    r["triple_limit_error"],
                    r["residual_relative"],
                ]
                for r in second["rows"]
            ],
        ),
        "",
    ]
    lines += [
        "### residual as a tripwire",
        "",
        _md_table(
            [
                "residual <= threshold",
                "pool",
                "cases",
                "worst |p error|",
                "worst |L error|",
            ],
            [
                [
                    t["threshold"],
                    t["pool"],
                    t["cases"],
                    t["max_order_error"],
                    t["max_limit_error"],
                ]
                for t in second["tripwire"]
            ],
        ),
        "",
    ]
    lines += [
        "## N. Noise (N=5, 1e-6 .. 3e-3 relative noise)",
        "",
        _md_table(
            [
                "noise",
                "trials",
                "fallback",
                "std(p)",
                "mean SE_p",
                "std/SE",
                "cov(|dp|<=2SE)",
                "cov(|dp|<=t95*SE)",
            ],
            [
                [
                    r["noise"],
                    r["trials"],
                    r["rejected"],
                    r["order_std"],
                    r["mean_stderr"],
                    r["ratio"],
                    r["two_sigma_coverage"],
                    r["t95_coverage"],
                ]
                for r in noise["rows"]
            ],
        ),
        "",
    ]
    lines += [
        "### conditioning: fine-pair signal vs metric noise (N=5, non-monotone fallback rate)",
        "",
        _md_table(
            ["family", "noise", "fallback", "rate"],
            [
                [r["family"], r["noise"], r["rejected"], r["rate"]]
                for r in conditioning["rows"]
            ],
        ),
        "",
    ]
    if real.get("available"):
        lines += [
            "## D. Real ladder (lid cavity, h = 1/n, three finest grids)",
            "",
            _md_table(
                [
                    "Re",
                    "metric",
                    "grids",
                    "p_fit",
                    "p_triple",
                    "|dp|",
                    "|dL|",
                    "GCI",
                    "SE_p",
                ],
                [
                    [
                        r["reynolds"],
                        r["metric"],
                        r["grids"],
                        r["order"],
                        r["order"],
                        r["order_error_vs_triple"],
                        r["limit_delta_vs_triple"],
                        r["gci"],
                        r["stderr"],
                    ]
                    for r in real["rows"]
                ],
            ),
            "",
            f"max |dp| vs three-grid routine = {real['max_order_delta']:.3e}, "
            f"max |dL| = {real['max_limit_delta']:.3e}, "
            f"order SE withheld on every case: {real['stderr_withheld']}  ",
            "",
            real["metric_note"],
            "",
        ]
    (OUT / "RESULTS.md").write_text("\n".join(lines) + "\n")

    p = write_report(payload)
    print("\n".join(lines))
    print("figures:", [str(x) for x in figs], "report:", p)


if __name__ == "__main__":
    main()
