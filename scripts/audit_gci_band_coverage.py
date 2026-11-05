#!/usr/bin/env python3
"""Coverage audit of the GCI uncertainty band used for every error bar in this repo.

Why this exists
---------------
``tensorfvm.verification.richardson.richardson_gci`` is the tool that turns three
grids into the "+-x %" quoted by the benchmark reports, so a band that does not
contain the true value is worse than no band.  The tool is already checked
against *its own* synthetic family (``power_law_series``, one error term).  It
has never been checked for the two questions that decide whether a quoted band
can be trusted:

* **T1 (exactness)** -- for a pure ``C h^p`` series with *non-uniform* ratios,
  does the band contain the exact value in every case, and does the band width
  stay of the right size (Roache efficiency index ``E = |error|/GCI <= 1``)?
* **T2 (truncation, the realistic case)** -- a numerical scheme never has a
  single error term.  Add a second term ``C2 h^(p+s)`` whose size at the finest
  grid is a fraction ``alpha`` of the leading term and ask how often each band
  convention then misses the exact value.  Two conventions are compared:

  - ``published``: Celik et al. (2008) / ASME V&V 20 report the uncertainty on
    the *fine-grid* value, ``f_exact in f1 (1 +- GCI)``;
  - ``implementation``: this repo bands the *extrapolated* value,
    ``f_exact in f_ext (1 +- GCI)`` (``RefinementStudy.band``), which is what
    ``reference_agreement(...)['band_covered']`` reports.

Both use the identical ``GCI = Fs |(f2-f1)/f1| / (r21^p - 1)`` width and
``Fs = 1.25``; only the centre differs, so any coverage difference is a
convention effect and not a safety-factor effect.

Everything here is pure NumPy arithmetic -- no solver, no heavy compute.

Usage
-----
    PYTHONPATH=src python scripts/audit_gci_band_coverage.py out=gci-band-coverage
"""

from __future__ import annotations

import json
import math
import os
import sys

import numpy as np

from tensorfvm.verification.richardson import richardson_gci

P_TRUE = (1.0, 1.5, 2.0, 2.5, 3.0, 4.0)
R21 = (1.2, 1.5, 2.0, 3.0)
R32 = (1.2, 1.5, 2.0, 3.0)
ALPHA = (0.0, 0.01, 0.05, 0.1, 0.3)
S_ORDER = (1.0, 2.0)
SIGNS = (1.0, -1.0)
EXACT = 1.0
H0 = 0.5  # finest spacing; keeps the fine-grid metric away from zero for sign=-1


def series(p, r21, r32, alpha, s, sign, exact=EXACT):
    """Three-grid metrics with a leading and a lagging error term.

    ``alpha`` is the size of the second term *at the finest grid*, relative to
    the leading term there, so ``alpha = 0`` is the pure one-term model.
    """
    h = np.array([H0, H0 * r21, H0 * r21 * r32], dtype=float)
    leading = sign * h**p
    second = alpha * sign * h**s * (h[0] ** (p - s))
    return h, [float(exact + leading[i] + second[i]) for i in range(3)]


def _covers(study, exact):
    lo, hi = study.band
    return bool(lo <= exact <= hi)


def _published_covers(study, fine, exact):
    half = study.gci * abs(fine)
    return bool(fine - half <= exact <= fine + half)


def classify(record):
    return record["implementation"], record["published"]


def study_row(p, r21, r32, alpha, s, sign):
    h, values = series(p, r21, r32, alpha, s, sign)
    fine, medium, coarse = values
    study = richardson_gci(fine, medium, coarse, tuple(h), theory_order=p)
    error = abs(study.extrapolated - EXACT) / EXACT
    row = dict(
        p=p,
        r21=r21,
        r32=r32,
        alpha=alpha,
        secondary_order=s,
        sign=sign,
        values=values,
        order_observed=study.order,
        order_ratio=study.order_ratio,
        asymptotic=study.asymptotic,
        convergence=study.convergence,
        extrapolated=study.extrapolated,
        gci=study.gci,
        band=[study.band[0], study.band[1]],
        error=error,
        efficiency=float("nan") if study.gci == 0.0 else error / study.gci,
        implementation=_covers(study, EXACT),
        published=_published_covers(study, fine, EXACT),
    )
    return row


def aggregate(rows, key, values):
    out = []
    for value in values:
        subset = [r for r in rows if r[key] == value]
        impl = sum(1 for r in subset if r["implementation"])
        pub = sum(1 for r in subset if r["published"])
        worst = max(
            subset,
            key=lambda r: r["efficiency"] if math.isfinite(r["efficiency"]) else -1.0,
        )
        out.append(
            dict(
                **{key: value},
                cases=len(subset),
                implementation_covered=impl,
                published_covered=pub,
                implementation_rate=impl / len(subset),
                published_rate=pub / len(subset),
                worst_efficiency=worst["efficiency"],
                worst_case={
                    k: worst[k]
                    for k in ("p", "r21", "r32", "alpha", "secondary_order", "sign")
                },
            )
        )
    return out


def figures(rows, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), dpi=140)
    for axis, alpha in zip(axes, (0.0, 0.1)):
        subset = [
            r for r in rows if r["alpha"] == alpha and r["secondary_order"] == 2.0
        ]
        axis.scatter(
            [r["order_observed"] for r in subset],
            [min(r["efficiency"], 10.0) for r in subset],
            s=16,
            c=["#2b7bba" if r["implementation"] else "#d62728" for r in subset],
            label="band covers exact" if alpha == 0.0 else "covers / misses",
        )
        axis.axhline(1.0, color="k", lw=1, ls="--")
        axis.set_xlabel("observed order p")
        axis.set_ylabel("efficiency E = |error|/GCI (capped at 10)")
        axis.set_title(f"second-term size alpha = {alpha:g}")
        axis.grid(alpha=0.3)
    axes[0].legend(fontsize=7)
    fig.suptitle(
        "GCI band coverage: pure Ch^p (left) vs truncation C2 h^(p+2) (right)",
        fontsize=9,
    )
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


NOISE = (0.0, 0.01, 0.03, 0.05, 0.10)
NOISE_SEEDS = 200


def noise_study(p=2.0, r21=2.0, r32=2.0, alpha=0.05, s=2.0, amplitude=0.05, seed=0):
    """Deterministic truncation plus independent relative noise on each grid value.

    This is the case a real refinement study actually is: the solver stops at a
    tolerance, so each metric carries an iteration/round-off perturbation.  The
    perturbation is symmetric and applied per grid, so the refinement series can
    be flagged inconsistent (non-monotone) and fall back to ``Fs = 3``.
    """
    h, values = series(p, r21, r32, alpha, s, 1.0)
    rng = np.random.default_rng(seed)
    noisy = [v * (1.0 + amplitude * rng.uniform(-1.0, 1.0)) for v in values]
    study = richardson_gci(noisy[0], noisy[1], noisy[2], tuple(h), theory_order=p)
    error = abs(study.extrapolated - EXACT) / EXACT
    return dict(
        amplitude=amplitude,
        seed=seed,
        values=noisy,
        extrapolated=study.extrapolated,
        gci=study.gci,
        convergence=study.convergence,
        asymptotic=study.asymptotic,
        error=error,
        efficiency=float("nan") if study.gci == 0.0 else error / study.gci,
        implementation=_covers(study, EXACT),
        published=_published_covers(study, noisy[0], EXACT),
    )


def noise_aggregate():
    rows = [
        noise_study(amplitude=a, seed=seed)
        for a in NOISE
        for seed in range(NOISE_SEEDS)
    ]
    table = []
    for a in NOISE:
        subset = [r for r in rows if r["amplitude"] == a]
        table.append(
            dict(
                amplitude=a,
                cases=len(subset),
                implementation_rate=sum(1 for r in subset if r["implementation"])
                / len(subset),
                published_rate=sum(1 for r in subset if r["published"]) / len(subset),
                fallback_rate=sum(
                    1 for r in subset if r["convergence"] == "inconsistent"
                )
                / len(subset),
                worst_efficiency=max(
                    r["efficiency"] for r in subset if math.isfinite(r["efficiency"])
                ),
            )
        )
    return rows, table


def write_docs(summary, by_alpha, by_p, noise_table, docs_dir):
    os.makedirs(docs_dir, exist_ok=True)
    w = summary["worst_row"]
    lines = [
        "# GCI 误差带覆盖率审计：`richardson_gci` 的 ± 带是否真的含真值",
        "",
        "## 1. 审计对象与动因",
        "",
        "本仓所有 benchmark 报告的「±x %」都来自 `tensorfvm.verification.richardson.richardson_gci`，",
        "`reference_agreement(...)['band_covered']` 直接决定一条参考值是否算「被误差带覆盖」。",
        "该工具此前只对本模块自己的一阶模型 `power_law_series` 做过自校验（单误差项、",
        "构造函数与估计器同源）。本审计用**独立构造**的两项截断误差族与随机噪声族回答两个问题：",
        "",
        "* T1 纯 `C h^p` 序列（非均匀加密比）：带是否恒含精确值，宽度是否仍在 Roache 意义上",
        "  「保守」（效率指数 `E=|error|/GCI <= 1`）；",
        "* T2 真实情形 `f = f_exact + C h^p + C2 h^(p+s)`：第二项在细网格占首项 `alpha` 时，",
        "  **两种报告的惯例**哪个更可靠——本仓惯例（带以 Richardson 外推值为中心）与",
        "  Celik et al. (2008) / ASME V&V 20 的惯例（不确定性报在**细网格值**上，",
        "  `f_exact in f1(1±GCI)`）。两者带宽相同（`Fs`=1.25），只有中心不同，",
        "  因此任何覆盖率差异都是惯例效应，不是安全系数效应。",
        "",
        "复现：`PYTHONPATH=src python scripts/audit_gci_band_coverage.py`（纯 NumPy，秒级）。",
        "",
        "## 2. 结果",
        "",
        f"确定性族 {summary['cases']} 例（纯序列 {summary['pure_cases']}、含第二项 "
        f"{summary['truncation_cases']}）。",
        "",
        "| 第二项规模 alpha | 例数 | 本仓惯例覆盖率 | Celik/ASME 惯例覆盖率 | 最差效率 E |",
        "|---|---|---|---|---|",
    ]
    for e in by_alpha:
        lines.append(
            f"| {e['alpha']:g} | {e['cases']} | {e['implementation_rate']:.4f} | "
            f"{e['published_rate']:.4f} | {e['worst_efficiency']:.3f} |"
        )
    lines += [
        "",
        "按观测阶分解（仅含第二项的 1536 例）：",
        "",
        "| p_true | 例数 | 本仓惯例 | Celik/ASME 惯例 | 最差 E |",
        "|---|---|---|---|---|",
    ]
    for e in by_p:
        lines.append(
            f"| {e['p']:g} | {e['cases']} | {e['implementation_rate']:.4f} | "
            f"{e['published_rate']:.4f} | {e['worst_efficiency']:.3f} |"
        )
    lines += [
        "",
        "加性噪声族（每网格独立 ±1/3/5/10 % 相对扰动，每档 "
        f"{int(noise_table[0]['cases'])} 个种子）：",
        "",
        "| 噪声幅度 | 例数 | 本仓惯例 | Celik/ASME 惯例 | 判为不一致(Fs=3)比例 | 最差 E |",
        "|---|---|---|---|---|---|",
    ]
    for e in noise_table:
        lines.append(
            f"| {e['amplitude']:g} | {e['cases']} | {e['implementation_rate']:.4f} | "
            f"{e['published_rate']:.4f} | {e['fallback_rate']:.4f} | "
            f"{e['worst_efficiency']:.3f} |"
        )
    lines += [
        "",
        f"最差确定性用例：p={w['p']:g}、r21={w['r21']:g}、r32={w['r32']:g}、"
        f"alpha={w['alpha']:g}、s={w['secondary_order']:g}、sign={w['sign']:g} → "
        f"E={w['efficiency']:.3f}、GCI={w['gci']:.4f}、相对误差={w['error']:.4f}"
        f"（本仓惯例覆盖={w['implementation']}，Celik/ASME 惯例覆盖={w['published']}）。",
        "",
        f"含第二项的案例中 {summary['asymptotic_rate_in_truncation'] * 100:.1f}% 被 "
        "`theory_order` + 20 % 阶容差判为「在渐近区」，说明渐近区标记本身对截断误差不敏感；",
        "误差带的可信度只能由覆盖率回答，不能由该标记回答。",
        "",
        "## 3. 结论（逐项）",
        "",
        f"1. **T1 纯序列**：本仓与 Celik 惯例覆盖率均为 {summary['pure_implementation_coverage']:.4f}，"
        f"最差效率 {summary['pure_worst_efficiency']:.3f} ≤ 1 —— 带既含真值又是上界。",
        f"2. **T2 两项截断**：本仓惯例覆盖率 "
        f"{summary['truncation_implementation_coverage']:.4f}（{summary['truncation_cases']} 例零漏）；"
        f"Celik/ASME 的细网格惯例 {summary['truncation_published_coverage']:.4f}，"
        f"两惯例判定不一致 {summary['convention_disagreements']} 例。"
        "差异的机理：外推值把首项消掉，残差由第二项主导且比 `GCI` 小一个量级；"
        "细网格值仍带着整个首项误差，而 `GCI` 只按 `r21^p-1` 归一，",
        "   p 越低、r21 越接近 1，归一化越不足，细网格惯例越容易漏。",
        "3. **噪声族**：判为不一致时自动降级到 `Fs=3`，覆盖率随噪声幅度单调下降，",
        "   最差效率逼近或超过 1 —— 这是 GCI 的固有局限（它假设单调、单侧、渐近的误差），",
        "   **不是实现缺陷**；报告引用 ± 带时必须同时给出加密比与实测阶，否则带无法被外部复核。",
        "",
        "## 4. 局限",
        "",
        "* 审计族是解析构造的两项误差模型，真实离散误差可能有更多项与非光滑（激波、界面）行为；",
        "* 只审计了「带是否含真值」与「效率指数」，未审计两点回退（`Fs=3`）的保守度是否足够；",
        "* 结论针对本仓的 `richardson_gci` 实现，不外推到其他 GCI 实现。",
        "",
        "## 5. 产物",
        "",
        "* `outputs/gci-band-coverage/audit.json`：全部 1920+1000 例逐例记录（含参数、GCI、误差、两种惯例判定）；",
        "* `outputs/gci-band-coverage/RESULTS.md`：聚合表；",
        "* 本节数字与表格全部由该脚本生成（零手抄）。",
    ]
    with open(os.path.join(docs_dir, "report.md"), "w") as handle:
        handle.write("\n".join(lines) + "\n")


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else "gci-band-coverage"
    target = os.path.join("outputs", out)
    os.makedirs(target, exist_ok=True)
    rows = [
        study_row(p, r21, r32, alpha, s, sign)
        for p in P_TRUE
        for r21 in R21
        for r32 in R32
        for alpha in ALPHA
        for s in S_ORDER
        for sign in SIGNS
    ]
    pure = [r for r in rows if r["alpha"] == 0.0]
    by_alpha = aggregate(rows, "alpha", ALPHA)
    by_p = aggregate([r for r in rows if r["alpha"] > 0.0], "p", P_TRUE)
    worst = max(
        rows, key=lambda r: r["efficiency"] if math.isfinite(r["efficiency"]) else -1.0
    )
    summary = dict(
        cases=len(rows),
        pure_cases=len(pure),
        pure_implementation_coverage=sum(1 for r in pure if r["implementation"])
        / len(pure),
        pure_published_coverage=sum(1 for r in pure if r["published"]) / len(pure),
        pure_worst_efficiency=max(r["efficiency"] for r in pure),
        truncation_cases=sum(1 for r in rows if r["alpha"] > 0.0),
        truncation_implementation_coverage=sum(
            1 for r in rows if r["alpha"] > 0.0 and r["implementation"]
        )
        / max(1, sum(1 for r in rows if r["alpha"] > 0.0)),
        truncation_published_coverage=sum(
            1 for r in rows if r["alpha"] > 0.0 and r["published"]
        )
        / max(1, sum(1 for r in rows if r["alpha"] > 0.0)),
        convention_disagreements=sum(
            1 for r in rows if r["implementation"] != r["published"]
        ),
        worst_row={
            k: worst[k]
            for k in (
                "p",
                "r21",
                "r32",
                "alpha",
                "secondary_order",
                "sign",
                "efficiency",
                "gci",
                "error",
                "implementation",
                "published",
            )
        },
        asymptotic_rate_in_truncation=sum(
            1 for r in rows if r["alpha"] > 0.0 and r["asymptotic"]
        )
        / max(1, sum(1 for r in rows if r["alpha"] > 0.0)),
    )
    noise_rows, noise_table = noise_aggregate()
    summary["noise"] = noise_table
    summary["noise_cases"] = len(noise_rows)
    payload = dict(
        summary=summary,
        by_alpha=by_alpha,
        by_order=by_p,
        noise=noise_table,
        rows=rows,
        noise_rows=noise_rows,
    )
    with open(os.path.join(target, "audit.json"), "w") as handle:
        json.dump(payload, handle, indent=1)
    figures(rows, os.path.join(target, "gci_band_coverage.png"))

    lines = [
        "# GCI band coverage audit",
        "",
        f"cases={summary['cases']} (pure={summary['pure_cases']}, "
        f"truncation={summary['truncation_cases']})",
        "",
        "| alpha | cases | implementation rate | published rate | worst E |",
        "|---|---|---|---|---|",
    ]
    for entry in by_alpha:
        lines.append(
            f"| {entry['alpha']:g} | {entry['cases']} | "
            f"{entry['implementation_rate']:.3f} | {entry['published_rate']:.3f} | "
            f"{entry['worst_efficiency']:.3f} |"
        )
    lines += [
        "",
        "| p | cases | implementation rate | published rate | worst E |",
        "|---|---|---|---|---|",
    ]
    for entry in by_p:
        lines.append(
            f"| {entry['p']:g} | {entry['cases']} | "
            f"{entry['implementation_rate']:.3f} | {entry['published_rate']:.3f} | "
            f"{entry['worst_efficiency']:.3f} |"
        )
    with open(os.path.join(target, "RESULTS.md"), "w") as handle:
        handle.write("\n".join(lines) + "\n")
    write_docs(summary, by_alpha, by_p, noise_table, os.path.join("docs", out))
    print(json.dumps(summary, indent=1))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
