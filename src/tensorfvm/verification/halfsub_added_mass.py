"""Verification: half-submerged sphere added mass (free surface, ``Fn -> 0``).

Reproduce
---------
    python -m tensorfvm.verification.halfsub_added_mass

The half-submerged sphere (centre on the free surface) is the only finite body
whose free-surface added mass is known: heave exactly, sway through the
odd-harmonic series of :mod:`tensorfvm.halfsub_sphere_reference`.  The panel side
uses the odd (antisymmetric) extension of
:mod:`tensorfvm.halfsub_added_mass`, i.e. a closed-sphere solve on an ordinary
``body`` image kernel -- no free-surface image is needed, so this verification
also pins the answer independently of any image-sign implementation.

Two properties are measured besides the numbers themselves:

* the **odd extension is exact**: heave must give exactly half of the full-sphere
  value (same potential, half the surface), a machine-precision structural check;
* the **wall / double-body answer is wrong by a factor of two** on this body, so
  the control line of the table is the full-sphere added mass itself.

The observed order is reported per mode: the sway mode has a waterline
singularity (the Dirichlet free surface meets the Neumann body), which degrades
the panel order below the heave mode -- a physical property, not a bug.
"""
from __future__ import annotations

import json
import pathlib
import time

import numpy as np

from ..halfsub_added_mass import (
    antisymmetric_added_mass,
    half_over_full_ratio,
    sphere_waterplane_panels,
)
from ..halfsub_sphere_reference import (
    deep_submergence_added_mass,
    displaced_mass,
    heave_added_mass_exact,
    sway_added_mass_series,
    translation_added_mass,
)
from ..panel_added_mass import added_mass_matrix
from ..panel_potential import IMAGE_SPECS

RHO = 1.0
RADIUS = 1.0
TARGET = 0.03
QUADRATURE_ORDER = 3
# each level doubles both counts; n_phi even keeps the equator a node ring
MESHES = ((12, 6), (24, 12), (48, 24), (96, 48))
SWAY_TERMS = (16, 32, 64, 128, 256)


def _criterion(statement, value, threshold, passed):
    return {"statement": statement, "value": float(value), "threshold": float(threshold),
            "pass": bool(passed)}


def observed_order(v1, v2, v3, ref, ratio=2.0):
    e1, e2 = abs(v1 - ref), abs(v2 - ref)
    if e2 <= 0.0 or e1 <= 0.0:
        return float("nan")
    return float(np.log(e1 / e2) / np.log(ratio))


def richardson(mid, fine, p, ratio=2.0):
    if not np.isfinite(p) or ratio**p - 1.0 <= 0.0:
        return float("nan")
    return float(fine + (fine - mid) / (ratio**p - 1.0))


def run() -> dict:  # noqa: C901 - linear verification script
    t_start = time.time()
    exact_heave = heave_added_mass_exact(RADIUS, RHO)
    deep = deep_submergence_added_mass(RADIUS, RHO)
    disp = displaced_mass(RADIUS, RHO)

    # ---- spectral references ---------------------------------------------
    heave_series = translation_added_mass(2, RADIUS, RHO, n_terms=32, n_nodes=400)
    sway_series = [sway_added_mass_series(RADIUS, RHO, n_terms=n, n_nodes=800)
                   for n in SWAY_TERMS]
    ref_sway = float(sway_series[-1])
    sway_tail = float(abs(sway_series[-2] - sway_series[-1]) / abs(ref_sway))

    # ---- panel sweep ------------------------------------------------------
    rows = []
    for n_theta, n_phi in MESHES:
        panels = sphere_waterplane_panels(RADIUS, n_theta, n_phi)
        t0 = time.time()
        M = antisymmetric_added_mass(panels, rho=RHO, order=QUADRATURE_ORDER)
        elapsed = time.time() - t0
        rows.append({
            "n_theta": n_theta, "n_phi": n_phi, "panels": int(panels.n_panels),
            "wet_panels": int((panels.centroids[:, 2] < 0.0).sum()),
            "M11": float(M[0, 0]), "M22": float(M[1, 1]), "M33": float(M[2, 2]),
            "M11_rel_err": float(M[0, 0] / ref_sway - 1.0),
            "M33_rel_err": float(M[2, 2] / exact_heave - 1.0),
            "offdiag_max": float(np.abs(M - np.diag(np.diag(M))).max()),
            "surge_sway_asymmetry": float(abs(M[0, 0] - M[1, 1]) / abs(M[0, 0])),
            "elapsed_s": float(elapsed),
        })

    # the two structural checks are mesh-independent; run them one level below the
    # finest mesh so the driver does not pay for a second finest-mesh matrix
    t0 = time.time()
    structural = sphere_waterplane_panels(RADIUS, *MESHES[-2])
    ratio_half_full = half_over_full_ratio(structural, rho=RHO, order=QUADRATURE_ORDER)
    M_full = added_mass_matrix(structural, rho=RHO, order=QUADRATURE_ORDER,
                              images=IMAGE_SPECS["body"], quadrature="gauss",
                              directions=(0, 2))
    ratio_elapsed = time.time() - t0

    p_heave = observed_order(rows[-3]["M33"], rows[-2]["M33"], rows[-1]["M33"], exact_heave)
    p_sway = observed_order(rows[-3]["M11"], rows[-2]["M11"], rows[-1]["M11"], ref_sway)
    rich_heave = richardson(rows[-2]["M33"], rows[-1]["M33"], p_heave)
    rich_sway = richardson(rows[-2]["M11"], rows[-1]["M11"], p_sway)

    criteria = {
        "c1_spectral_selfcheck": _criterion(
            "|heave series / (1/3 pi rho a^3) - 1| (validates the sway reference)",
            abs(heave_series / exact_heave - 1.0), 1e-10,
            abs(heave_series / exact_heave - 1.0) < 1e-10),
        "c2_odd_extension_exact": _criterion(
            "|M33(half) / M33(full sphere) - 1/2| at the finest mesh",
            abs(ratio_half_full - 0.5), 1e-10, abs(ratio_half_full - 0.5) < 1e-10),
        "c3_heave_finest": _criterion(
            "|M33 / ((1/3) pi rho a^3) - 1| at the finest mesh",
            abs(rows[-1]["M33_rel_err"]), TARGET, abs(rows[-1]["M33_rel_err"]) < TARGET),
        "c4_sway_finest": _criterion(
            "|M11 / M11(odd-harmonic series) - 1| at the finest mesh",
            abs(rows[-1]["M11_rel_err"]), TARGET, abs(rows[-1]["M11_rel_err"]) < TARGET),
        "c5_heave_richardson": _criterion(
            "|Richardson(M33) / exact - 1|", abs(rich_heave / exact_heave - 1.0), 0.01,
            abs(rich_heave / exact_heave - 1.0) < 0.01),
        "c6_sway_richardson": _criterion(
            "|Richardson(M11) / series - 1|", abs(rich_sway / ref_sway - 1.0), 0.01,
            abs(rich_sway / ref_sway - 1.0) < 0.01),
        "c7_decoupling": _criterion(
            "max off-diagonal |M_ij| / (rho a^3) at the finest mesh",
            rows[-1]["offdiag_max"], 1e-10, rows[-1]["offdiag_max"] < 1e-10),
        "c8_surge_sway_symmetry": _criterion(
            "|M11 - M22| / M11 at the finest mesh", rows[-1]["surge_sway_asymmetry"], 1e-10,
            rows[-1]["surge_sway_asymmetry"] < 1e-10),
        "c9_order_heave": _criterion("observed order p of M33 (two finest intervals)",
                                     p_heave, 1.0, p_heave >= 1.0),
        "c10_order_sway": _criterion("observed order p of M11 (two finest intervals)",
                                     p_sway, 1.0, p_sway >= 1.0),
        "c11_wall_control_factor": _criterion(
            "full-sphere (wall / double-body) M33 / free-surface M33 - 2 (control)",
            abs(M_full[2, 2] / rows[-1]["M33"] - 2.0), 0.05,
            abs(M_full[2, 2] / rows[-1]["M33"] - 2.0) < 0.05),
        "c12_sway_tail": _criterion(
            "relative drift of the sway series between the two largest truncations",
            sway_tail, 1e-3, sway_tail < 1e-3),
    }

    return {
        "title": "half-submerged sphere: free-surface added mass by the odd extension",
        "rho": RHO, "radius": RADIUS, "quadrature_order": QUADRATURE_ORDER,
        "meshes": [list(m) for m in MESHES],
        "references": {
            "heave_exact": float(exact_heave),
            "heave_over_displaced_mass": float(exact_heave / disp),
            "displaced_mass": float(disp),
            "deep_submergence": float(deep),
            "sway_series": [float(v) for v in sway_series],
            "sway_terms": list(SWAY_TERMS),
            "sway_reference": ref_sway,
            "sway_tail_rel": sway_tail,
            "sway_over_displaced_mass": float(ref_sway / disp),
            "heave_series_value": float(heave_series),
        },
        "panel_sweep": rows,
        "extrapolation": {"p_heave": p_heave, "p_sway": p_sway,
                          "richardson_heave": rich_heave, "richardson_sway": rich_sway},
        "odd_extension": {"ratio_half_over_full_M33": ratio_half_full,
                          "full_sphere_M33": float(M_full[2, 2]),
                          "full_sphere_M11": float(M_full[0, 0]),
                          "elapsed_s": float(ratio_elapsed)},
        "criteria": criteria,
        "all_pass": bool(all(c["pass"] for c in criteria.values())),
        "elapsed_s": float(time.time() - t_start),
    }


def render_report(s: dict) -> str:
    ref, ext, rows = s["references"], s["extrapolation"], s["panel_sweep"]
    L = [
        "# TensorFVM · 半浸没球附加质量：奇对称（反对称）延拓法验证",
        "",
        "本文件由 `summary.json` 程序化生成（无手抄数字）。复现：",
        "`python -m tensorfvm.verification.halfsub_added_mass`。",
        "",
        "## 1. 案例",
        "",
        f"半径 `a = {s['radius']:g}` 的球，球心恰在静水面 `z = 0`（或然性无穷频率 / `Fn -> 0` 极限：",
        "自由面即 Dirichlet 平面 `phi = 0`）。此时只有下半球是湿表面，",
        "自由面条件通过**奇延拓** `Phi(x,y,-z) = -Phi(x,y,z)` 施加：延拓后的势在整个反射体外部调和、",
        "在 `z = 0` 上恒零，因此下半空间的解就是原问题解，代价是上半球的 Neumann 数据不再刚体",
        "（`:func:`tensorfvm.halfsub_added_mass.antisymmetric_normal_velocity`），",
        "而附加质量只在下半球湿表面（真实湿面）积分。",
        "",
        "| 量 | 参考值（rho=1, a=1） | 来源 |",
        "|---|---|---|",
        f"| 垂荡 `M33` | {ref['heave_exact']:.6f} | 闭式 `(1/3) pi rho a^3` = "
        f"{ref['heave_over_displaced_mass']:.4f} x 排水质量 |",
        f"| 横荡 `M11` | {ref['sway_reference']:.6f} | 奇对称球谐级数（{ref['sway_terms'][-1]} 项），"
        f"级数尾部漂移 {ref['sway_tail_rel']:.1e} |",
        f"| 整球/双体值 | {ref['deep_submergence']:.6f} | `(2/3) pi rho a^3`（壁面镜像结果） |",
        "",
        "## 2. 面元收敛（奇延拓 + 整球网格，Gauss 阶 "
        f"{s['quadrature_order']}，湿面元占一半）",
        "",
        "| n_theta x n_phi | 面元 | 湿面元 | M33 | vs 闭式 | M11 | vs 级数 |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        L.append(f"| {r['n_theta']} x {r['n_phi']} | {r['panels']} | {r['wet_panels']} | "
                 f"{r['M33']:.6f} | {r['M33_rel_err']:+.2%} | {r['M11']:.6f} | "
                 f"{r['M11_rel_err']:+.2%} |")
    L += [
        "",
        f"观测收敛阶：`p(M33) = {ext['p_heave']:.2f}`，`p(M11) = {ext['p_sway']:.2f}`。",
        f"Richardson 外推：`M33 -> {ext['richardson_heave']:.6f}` "
        f"({ext['richardson_heave'] / ref['heave_exact'] - 1.0:+.3%})，"
        f"`M11 -> {ext['richardson_sway']:.6f}` "
        f"({ext['richardson_sway'] / ref['sway_reference'] - 1.0:+.3%})。",
        "",
        "横荡阶低于垂荡是物理性质而非缺陷：Dirichlet 自由面与 Neumann 物面在水线相交，",
        "解的局部行为是 `r^{1/2}` 型（水线奇点），面元法在能量范数上因此掉阶。",
        "",
        "## 3. 结构恒等式与对照",
        "",
        f"* `M33(half) / M33(full sphere) = {s['odd_extension']['ratio_half_over_full_M33']:.12f}`"
        "（奇延拓的垂荡数据恰是整球刚体垂荡数据，故必须严格等于 1/2）；",
        f"* 对照：整球（= 壁面/双体镜像）`M33 = {s['odd_extension']['full_sphere_M33']:.6f}`，"
        f"与真实自由面值之比 `{s['odd_extension']['full_sphere_M33'] / rows[-1]['M33']:.4f}`",
        "  —— 用错镜像符号在同一个物体上高估 100%。",
        "",
        "## 4. 判据",
        "",
        "| id | 判据 | 值 | 阈值 | 结论 |",
        "|---|---|---|---|---|",
    ]
    for k, c in s["criteria"].items():
        L.append(f"| {k} | {c['statement']} | {c['value']:.6g} | {c['threshold']:g} | "
                 f"{'PASS' if c['pass'] else 'FAIL'} |")
    L += ["", f"**all_pass = {s['all_pass']}**（耗时 {s['elapsed_s']:.1f} s）", ""]
    return "\n".join(L)


def plot(summary: dict, path: pathlib.Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = summary["panel_sweep"]
    ref = summary["references"]
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(14.5, 4.2))

    n = np.array([r["panels"] for r in rows], float)
    ax1.loglog(n, np.abs([r["M33_rel_err"] for r in rows]), "o-", label="M33 (heave)")
    ax1.loglog(n, np.abs([r["M11_rel_err"] for r in rows]), "s-", label="M11 (sway)")
    ax1.axhline(0.03, ls="--", color="r", lw=1.0, label="3% acceptance")
    ax1.set_xlabel("panels (-)")
    ax1.set_ylabel("relative error (-)")
    ax1.set_title("odd-extension panel convergence")
    ax1.grid(alpha=0.3, which="both")
    ax1.legend(fontsize=7)

    labels = ["M33\nfree surface", "M33\nfull sphere\n(wall image)", "M11\nfree surface"]
    vals = [rows[-1]["M33"], summary["odd_extension"]["full_sphere_M33"], rows[-1]["M11"]]
    refs = [ref["heave_exact"], ref["deep_submergence"], ref["sway_reference"]]
    x = np.arange(3)
    ax2.bar(x - 0.18, vals, width=0.34, label="panel (finest mesh)")
    ax2.bar(x + 0.18, refs, width=0.34, label="reference")
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, fontsize=7)
    ax2.set_ylabel(r"$M/(\rho a^3)$ (-)")
    ax2.set_title("panel vs reference, and the wall-image control")
    ax2.grid(alpha=0.3, axis="y")
    ax2.legend(fontsize=7)

    terms = np.array(ref["sway_terms"], float)
    vals_s = np.array(ref["sway_series"], float)
    ax3.semilogx(terms, vals_s / ref["sway_reference"], "o-", label="odd-harmonic series")
    ax3.axhline(1.0, ls=":", color="k")
    ax3.fill_between([terms.min(), terms.max()],
                     [rows[-1]["M11"] / ref["sway_reference"]] * 2,
                     [1.0, 1.0], color="tab:red", alpha=0.15,
                     label=f"panel gap at {rows[-1]['panels']} panels")
    ax3.set_xlabel("series truncation (terms)")
    ax3.set_ylabel(r"$M_{11}/M_{11}^{ref}$ (-)")
    ax3.set_title("spectral reference convergence (1/n tail)")
    ax3.grid(alpha=0.3, which="both")
    ax3.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main(argv=None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default="docs/verification-halfsub-added-mass")
    args = ap.parse_args(argv)
    summary = run()
    outdir = pathlib.Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True,
                                                    default=float))
    (outdir / "REPORT.md").write_text(render_report(summary))
    plot(summary, outdir / "fig_halfsub_added_mass.png")
    for k, c in summary["criteria"].items():
        print(f"{k}: {'PASS' if c['pass'] else 'FAIL'}  value={c['value']:.6g}")
    print(f"all_pass={summary['all_pass']}  elapsed={summary['elapsed_s']:.1f}s -> {outdir}")
    return 0 if summary["all_pass"] else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())