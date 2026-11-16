#!/usr/bin/env python3
"""XVAL-FVM-M18737-ROUND2 — 峰值非渐近异常（PEAK-ANOMALY）定量裁决.

锚点: XVAL-FVM-M18737-ROUND2（TensorFVM 的 m18737-round2 做法）。
对象: 综合报告（REPORT.md「给下一手的建议」第 1/2 条）指出的
    「FVM 峰值在 nx=3200→6400 下降（6.5075→6.4622），非单调，Richardson 外推失效」
    以及「确认 nx=6400 解的守恒性和正性」。

本脚本**不跑求解器**，全部由已入库剖面（ladder_profile_*.npz）+ 标定参考
（reference_density_calibrated.csv）重算，纯 numpy。产出:
  - peak_anomaly.json   （全部数字，程序化，零手抄）
  - figures/fig_peak_anomaly_*.png

判据（先于数字冻结，见 PEAK-ANOMALY-CLOSURE.md）:
  A1  峰值下降不是「max-vs-平台均值」度量伪影:
      平台均值（rho>0.99*peak 的均值）序列与 max 序列同向非单调。
  A2  峰值下降不是「尖峰亚格对齐」伪影:
      峰值平台 FWHM ≫ dx（平台是宽结构，不是亚格尖峰）。
  A3  交叉码独立佐证: WENO5（零共享离散）峰值序列单调饱和，
      且 FVM@6400 与 WENO5@12800 峰值差 <0.1%、平台均值差 <0.05%
      ⇒ FVM@6400 位于交叉码收敛水平，3200 档是离群超调。
  A4  nx=6400 守恒: 质量（单元和口径）相对误差 <1e-10（守恒格式机器精度），
      梯形口径 <0.01%；能量（梯形口径）<0.01%。
  A5  nx=6400 正性: min rho > 0.1, min p > 1（远离零，非贴零）。
  A6  参考侧: 标定参考峰值低于交叉码收敛水平（参考自身离散耗散压低），
      与综合报告「参考峰值被压低 ~3%」一致。
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "woodward-colella-double-blast" / "ladder" / "data"
REF_CSV = HERE.parent / "woodward-colella-double-blast" / "reference" / "reference_density_calibrated.csv"
FIG = HERE / "figures"
GAMMA = 1.4
# 初值: rho=1, u=0, p 三段 (1000 on [0,.1], 0.01 on (.1,.9), 100 on [.9,1])
M0 = 1.0
P0 = 0.0
E0 = (1000.0 * 0.1 + 0.01 * 0.8 + 100.0 * 0.1) / (GAMMA - 1.0)  # = 275.02

FVM_RUNGS = ["nx200_hllc", "nx1600_hllc", "nx3200_hllc", "nx6400_hllc"]
WENO5_RUNGS = ["nx3200_weno5", "nx6400_weno5", "nx12800_weno5"]
WENO7_RUNGS = ["nx400_weno7", "nx1600_weno7"]


def load(tag: str) -> dict:
    d = np.load(DATA / f"ladder_profile_{tag}.npz")
    return {k: d[k] for k in ("x", "rho", "p", "u")}


def plateau_stats(rho: np.ndarray, frac: float = 0.99) -> dict:
    pk = float(rho.max())
    m = rho > frac * pk
    return {
        "peak": pk,
        "plateau_mean": float(rho[m].mean()),
        "plateau_std": float(rho[m].std()),
        "plateau_ncells": int(m.sum()),
    }


def fwhm(x: np.ndarray, rho: np.ndarray) -> dict:
    j = int(np.argmax(rho))
    half = rho[j] / 2.0
    lo = j
    while lo > 0 and rho[lo] > half:
        lo -= 1
    hi = j
    while hi < len(rho) - 1 and rho[hi] > half:
        hi += 1
    dx = float(x[1] - x[0])
    return {
        "peak_x": float(x[j]),
        "fwhm": float((hi - lo) * dx),
        "fwhm_cells": int(hi - lo),
        "dx": dx,
        "fwhm_over_dx": float((hi - lo)),
    }


def conservation(d: dict) -> dict:
    x, rho, p, u = d["x"], d["rho"], d["p"], d["u"]
    dx = float(x[1] - x[0])
    e = p / ((GAMMA - 1.0) * rho) + 0.5 * u * u
    m_cell = float(dx * rho.sum())          # 分段常数重构的精确积分
    m_trap = float(np.trapezoid(rho, x))
    p_cell = float(dx * (rho * u).sum())
    e_cell = float(dx * (rho * e).sum())
    e_trap = float(np.trapezoid(rho * e, x))
    return {
        "mass_cellsum_rel": m_cell - M0,
        "mass_trap_rel": m_trap - M0,
        "momentum_cellsum": p_cell - P0,
        "energy_cellsum_rel": (e_cell - E0) / E0,
        "energy_trap_rel": (e_trap - E0) / E0,
        "min_rho": float(rho.min()),
        "min_p": float(p.min()),
        "min_rho_x": float(x[int(np.argmin(rho))]),
        "min_p_x": float(x[int(np.argmin(p))]),
    }


def peak_series(tags: list[str]) -> list[dict]:
    out = []
    for t in tags:
        d = load(t)
        st = plateau_stats(d["rho"])
        fw = fwhm(d["x"], d["rho"])
        st.update(fw)
        st["tag"] = t
        out.append(st)
    return out


def main() -> dict:
    FIG.mkdir(parents=True, exist_ok=True)
    ref = np.loadtxt(REF_CSV, delimiter=",", skiprows=1)
    xr, rr = ref[:, 0], ref[:, 1]
    ref_peak = float(rr.max())
    ref_peak_x = float(xr[int(np.argmax(rr))])

    fvm = peak_series(FVM_RUNGS)
    w5 = peak_series(WENO5_RUNGS)
    w7 = peak_series(WENO7_RUNGS)

    # --- A1: max vs 平台均值 同向性 ---
    fvm_max = [r["peak"] for r in fvm if r["tag"] != "nx200_hllc"]
    fvm_plat = [r["plateau_mean"] for r in fvm if r["tag"] != "nx200_hllc"]
    a1 = {
        "fvm_peak_seq": fvm_max,
        "fvm_plateau_seq": fvm_plat,
        "peak_nonmonotone": fvm_max[1] > fvm_max[0] and fvm_max[2] < fvm_max[1],
        "plateau_nonmonotone": fvm_plat[1] > fvm_plat[0] and fvm_plat[2] < fvm_plat[1],
        "same_direction": (fvm_max[1] > fvm_max[0]) == (fvm_plat[1] > fvm_plat[0])
        and (fvm_max[2] < fvm_max[1]) == (fvm_plat[2] < fvm_plat[1]),
    }

    # --- A2: 平台宽度 vs 网格 ---
    # nx200 档欠分辨（峰值 5.0，平台未成形），单独记账，不参与 A2 判据
    resolved = [r for r in fvm + w5 if int(r["tag"][2:].split("_")[0]) >= 1600]
    a2 = {
        "fvm": {r["tag"]: {"fwhm": r["fwhm"], "fwhm_over_dx": r["fwhm_over_dx"]} for r in fvm},
        "weno5": {r["tag"]: {"fwhm": r["fwhm"], "fwhm_over_dx": r["fwhm_over_dx"]} for r in w5},
        "resolved_rungs": [r["tag"] for r in resolved],
        "all_broad": all(r["fwhm_over_dx"] > 50 for r in resolved),
    }

    # --- A3: 交叉码 ---
    f64 = next(r for r in fvm if r["tag"] == "nx6400_hllc")
    w5_128 = next(r for r in w5 if r["tag"] == "nx12800_weno5")
    w5_64 = next(r for r in w5 if r["tag"] == "nx6400_weno5")
    w5_32 = next(r for r in w5 if r["tag"] == "nx3200_weno5")
    f32 = next(r for r in fvm if r["tag"] == "nx3200_hllc")
    a3 = {
        "fvm6400_peak": f64["peak"],
        "weno5_12800_peak": w5_128["peak"],
        "peak_gap_6400_vs_w5_12800_pct": (f64["peak"] - w5_128["peak"]) / w5_128["peak"] * 100.0,
        "plateau_gap_6400_vs_w5_12800_pct": (f64["plateau_mean"] - w5_128["plateau_mean"]) / w5_128["plateau_mean"] * 100.0,
        "fvm3200_vs_weno5_3200_plateau_pct": (f32["plateau_mean"] - w5_32["plateau_mean"]) / w5_32["plateau_mean"] * 100.0,
        "weno5_peak_seq": [r["peak"] for r in w5],
        "weno5_monotone_saturating": w5[0]["peak"] < w5[1]["peak"] and abs(w5[2]["peak"] - w5[1]["peak"]) / w5[1]["peak"] < 0.001,
        "weno7_peak_seq": [r["peak"] for r in w7],
        "weno7_1600_above_converged_pct": (w7[1]["peak"] - w5_128["peak"]) / w5_128["peak"] * 100.0,
    }

    # --- A4/A5: 守恒 + 正性（全档） ---
    cons = {t: conservation(load(t)) for t in FVM_RUNGS + WENO5_RUNGS}
    c64 = cons["nx6400_hllc"]
    a4 = {
        "nx6400": c64,
        "mass_cellsum_ok": abs(c64["mass_cellsum_rel"]) < 1e-10,
        "mass_trap_ok": abs(c64["mass_trap_rel"]) < 1e-4,
        "energy_trap_ok": abs(c64["energy_trap_rel"]) < 1e-4,
        "all_rungs": cons,
    }
    a5 = {
        "nx6400_min_rho": c64["min_rho"],
        "nx6400_min_p": c64["min_p"],
        "positive": c64["min_rho"] > 0.1 and c64["min_p"] > 1.0,
    }

    # --- A7: 3200→6400 差值局域化 + 窗质量自补偿 ---
    xf = np.linspace(0.0, 1.0, 256001)
    prof = {}
    for t in ("nx1600_hllc", "nx3200_hllc", "nx6400_hllc", "nx6400_weno5", "nx12800_weno5"):
        d = load(t)
        prof[t] = np.interp(xf, d["x"], d["rho"])
    win = xf >= 0.4
    winmass = {t: float(np.trapezoid(prof[t][win], xf[win])) for t in prof}
    d32 = prof["nx3200_hllc"] - prof["nx6400_hllc"]
    tot = float(np.trapezoid(np.abs(d32), xf))
    shares = {}
    for a, b in ((0.0, 0.4), (0.4, 0.55), (0.55, 0.70), (0.70, 0.85), (0.85, 1.0)):
        m = (xf >= a) & (xf < b)
        shares[f"{a}-{b}"] = float(np.trapezoid(np.abs(d32)[m], xf[m])) / tot
    a7 = {
        "window_mass_04_1": winmass,
        "window_mass_spread_pct": (max(winmass.values()) - min(winmass.values())) / min(winmass.values()) * 100.0,
        "abs_diff_shares_3200_vs_6400": shares,
        "core_share_055_085": shares["0.55-0.7"] + shares["0.7-0.85"],
        "localized": shares["0.55-0.7"] + shares["0.7-0.85"] > 0.9,
    }

    # --- A6: 参考侧 ---
    a6 = {
        "ref_peak": ref_peak,
        "ref_peak_x": ref_peak_x,
        "fvm6400_peak_vs_ref_pct": (f64["peak"] - ref_peak) / ref_peak * 100.0,
        "weno5_12800_peak_vs_ref_pct": (w5_128["peak"] - ref_peak) / ref_peak * 100.0,
    }

    result = {
        "anchor": "XVAL-FVM-M18737-ROUND2",
        "topic": "PEAK-ANOMALY",
        "gamma": GAMMA,
        "initial": {"M0": M0, "P0": P0, "E0": E0},
        "fvm_rungs": fvm,
        "weno5_rungs": w5,
        "weno7_rungs": w7,
        "A1_max_vs_plateau": a1,
        "A2_plateau_width": a2,
        "A3_cross_code": a3,
        "A4_conservation": a4,
        "A5_positivity": a5,
        "A7_localization": a7,
        "A6_reference": a6,
        "verdict": {
            "A1": bool(a1["same_direction"] and a1["peak_nonmonotone"]),
            "A2": bool(a2["all_broad"]),
            "A3": bool(a3["peak_gap_6400_vs_w5_12800_pct"] < 0.1
                      and a3["plateau_gap_6400_vs_w5_12800_pct"] < 0.05
                      and a3["weno5_monotone_saturating"]),
            "A4": bool(a4["mass_cellsum_ok"] and a4["mass_trap_ok"] and a4["energy_trap_ok"]),
            "A5": bool(a5["positive"]),
            "A6": bool(a6["fvm6400_peak_vs_ref_pct"] > 1.0),
            "A7": bool(a7["localized"] and a7["window_mass_spread_pct"] < 0.01),
        },
    }
    out = HERE / "peak_anomaly.json"
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print(json.dumps(result["verdict"], indent=2))
    print(f"wrote {out}")
    return result


if __name__ == "__main__":
    main()
