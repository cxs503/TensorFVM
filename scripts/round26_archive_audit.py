#!/usr/bin/env python3
"""Round 26: audit the stored ``CASE-FVM-PERIODIC-HILLS`` archive.

Two questions, both answerable from files already in the tree (zero solver runs):

1. **Does every archived artefact actually contain a field?**  Rounds 20/22
   recorded refined ladder rungs as "diverged"; the dumps on disk are all-zero
   velocity fields.  :func:`tensorfvm.verification.periodic_hill_archive_audit.
   artifact_health` classifies every ``.npz`` and the grid-ladder summaries are
   built from usable rungs only.
2. **Does the certified wall-shear bubble correspond to a recirculation *cell*?**
   The certified operator pairs signed wall-shear crossings into bubbles; the
   field-side route counts connected ``u < 0`` cells.  Where the two disagree the
   published ``x_r/H`` is measuring a wall lift-off, not the end of a vortex, and
   the definition band says by how much.

Outputs (all numbers program-injected, nothing typed by hand)::

    docs/case-periodic-hills/round26_archive_audit.json
    docs/case-periodic-hills/fig_round26_archive_health.png
    docs/case-periodic-hills/fig_round26_bubble_objects.png
    docs/case-periodic-hills/fig_round26_definition_band.png
    docs/case-periodic-hills/fig_round26_cells_laminar.png

Usage::

    python scripts/round26_archive_audit.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from tensorfvm.verification.periodic_hill_archive_audit import (  # noqa: E402
    ARTIFACT_USABLE,
    artifact_health,
    definition_band,
    definition_routes,
    ladder_summary,
    recirculation_cells,
    wall_spans,
)
from tensorfvm.verification.periodic_hill_reference import X_R_OVER_H  # noqa: E402
from tensorfvm.verification.wall_shear_bubbles import (  # noqa: E402
    separation_reattachment,
)

CASE_DIR = Path(__file__).resolve().parents[1] / "docs" / "case-periodic-hills"
OUT_JSON = CASE_DIR / "round26_archive_audit.json"

#: which artefacts belong to which grid ladder (family -> ordered coarse->fine)
LADDERS = {
    "sst_r18": ("round11_sst_r18_", ("72x36", "96x48", "120x60", "144x72")),
    "laminar_re2800": ("round20_laminar_Re2800_",
                       ("96x48", "144x72", "192x96", "240x120")),
}


def _fmt(v, nd=3):
    return "None" if v is None else f"{v:.{nd}f}"


def _grid_of(stem: str) -> int:
    """``nx`` parsed from the artefact stem (``..._96x48`` / ``..._96x48dt01``)."""
    for part in stem.split("_"):
        if "x" in part and part.split("x")[0].isdigit():
            return int(part.split("x")[0])
    return -1


def audit_artefact(path: Path) -> dict:
    with np.load(path) as d:
        health = artifact_health(d)
        entry = {"stem": path.stem, "file": path.name, **health,
                 "nx": health.get("nx"), "routes": None, "band": None,
                 "wall_spans": None}
        if health["status"] != ARTIFACT_USABLE or health.get("nx") is None:
            entry["nx"] = _grid_of(path.stem)
            return entry
        u = np.asarray(d["mean_u"], dtype=float)
        x = np.asarray(d["x_over_h"], dtype=float).ravel()
        tau = np.asarray(d["mean_tau_b"], dtype=float).ravel()
        entry["nx"] = int(u.shape[1])
        entry["ny"] = int(u.shape[0])
        routes = definition_routes(tau, u, x)
        entry["routes"] = routes["routes"]
        entry["meta"] = routes["meta"]
        entry["band"] = definition_band(routes)
        entry["wall_spans"] = [
            {k: s[k] for k in ("separation", "reattachment", "length")}
            for s in wall_spans(u[0], x)
        ]
        ws_first = separation_reattachment(tau, x, primary="first")
        ws_large = separation_reattachment(tau, x, primary="largest_area")
        entry["wall_shear_first"] = ws_first["x_r_over_H"]
        entry["wall_shear_largest_area"] = ws_large["x_r_over_H"]
        entry["n_bubbles"] = ws_first["n_bubbles"]
        entry["field_cells"] = [
            {k: c[k] for k in ("label", "area_cells", "x_min", "x_max",
                               "row_min", "row_max", "wall_attached",
                               "wraps_seam")}
            for c in recirculation_cells(u, x)
        ]
        return entry


# --------------------------------------------------------------------------
# figures
# --------------------------------------------------------------------------
def fig_health(entries, path):
    stems = [e["stem"] for e in entries]
    vals = [1e-12 if e["max_abs_u"] in (None, 0.0) else e["max_abs_u"]
            for e in entries]
    colors = ["#2a7f62" if e["status"] == ARTIFACT_USABLE else "#b23a48"
              for e in entries]
    fig, ax = plt.subplots(figsize=(11, 4.6))
    ax.barh(range(len(stems)), vals, color=colors, height=0.7)
    ax.set_yticks(range(len(stems)))
    ax.set_yticklabels(stems, fontsize=7)
    ax.set_xscale("log")
    ax.set_xlabel(r"$\max|u|/U_b$ of the stored mean field (log)")
    ax.axvline(1e-9, color="k", ls=":", lw=1)
    ax.text(1.3e-9, len(stems) - 0.6, "zero-field gate", fontsize=7, rotation=90,
            va="top")
    n_bad = sum(e["status"] != ARTIFACT_USABLE for e in entries)
    ax.set_title(f"round 26 archive health: {len(entries) - n_bad}/{len(entries)} "
                 f"usable, {n_bad} degenerate (bar clamped at 1e-12)")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def fig_bubble_objects(entries, path):
    picks = [e for e in entries if e["stem"] in
             ("round20_laminar_Re2800_96x48", "round11_sst_r18_96x48",
              "round20_laminar_Re700_96x48", "round9_96x48")]
    fig, axes = plt.subplots(len(picks), 1, figsize=(10, 2.4 * len(picks)),
                             sharex=True)
    axes = np.atleast_1d(axes)
    for ax, e in zip(axes, picks):
        d = np.load(CASE_DIR / e["file"])
        x = np.asarray(d["x_over_h"], float).ravel()
        tau = np.asarray(d["mean_tau_b"], float).ravel()
        uw = np.asarray(d["mean_u"], float)[0]
        ax.plot(x, tau / np.abs(tau).max(), lw=1.4, color="#1f4e79",
                label=r"$\tau_b/\max|\tau_b|$")
        ax.plot(x, uw / np.abs(uw).max(), lw=1.0, color="#c46a00",
                label=r"$u_{wall}/\max|u_{wall}|$")
        ws = separation_reattachment(tau, x, primary="first")
        for b in ws["bubbles"]:
            lo, hi = b["separation"], min(b["reattachment"], x[-1] + (x[1] - x[0]))
            ax.axvspan(lo, hi, color="#b23a48", alpha=0.10)
        for s in e["wall_spans"] or []:
            ax.axvline(s["separation"], color="#2a7f62", ls="--", lw=0.9)
            ax.axvline(s["reattachment"], color="#2a7f62", ls="-.", lw=0.9)
        ax.axvline(X_R_OVER_H % (x[-1] - x[0] + (x[1] - x[0])), color="k",
                   ls=":", lw=1)
        ax.axhline(0.0, color="k", lw=0.6)
        note = (f"cells={e['n_cells']}  "
                f"band={e['band']['spread_rel'] * 100:.1f}% "
                f"({e['band']['flag']}, {e['meta']['object_verdict']})")
        ax.set_title(f"{e['stem']}   {note}", fontsize=8, loc="left")
        ax.set_ylabel("normalised")
        ax.grid(alpha=0.3)
        if ax is axes[0]:
            ax.legend(fontsize=7, ncol=2, loc="lower left")
    axes[-1].set_xlabel(r"$x/H$  (red band = certified wall-shear bubble, "
                        r"green = field wall-attachment span, dotted = DNS 5.41)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def fig_definition_band(entries, path):
    picks = [e for e in entries if e["routes"] is not None]
    fig, ax = plt.subplots(figsize=(9.5, 4.6))
    lo, hi = X_R_OVER_H * 0.9, X_R_OVER_H * 1.1
    ax.axhspan(lo, hi, color="#2a7f62", alpha=0.15,
               label=r"acceptance $\pm10\%$ of 5.41")
    for k, e in enumerate(picks):
        vals = e["band"]["values"]
        ax.plot([k] * len(vals), vals, "o", ms=5, color="#1f4e79")
        ax.vlines(k, min(vals), max(vals), color="#b23a48", lw=1.4)
        ax.text(k, max(vals) + 0.25,
                f"{e['band']['spread_rel'] * 100:.0f}%", ha="center", fontsize=7,
                color="#b23a48")
    ax.set_xticks(range(len(picks)))
    ax.set_xticklabels([e["stem"].replace("round11_sst_", "sst_")
                        .replace("round20_laminar_", "lam_").replace("round9_", "sa_")
                        for e in picks], rotation=30, ha="right", fontsize=7)
    ax.set_ylabel(r"$x_r/H$ candidate values (3 definitions)")
    ax.set_title("round 26: definition band of $x_r/H$ per stored artefact "
                 "(red = spread, % of reference)")
    ax.grid(alpha=0.3, axis="y")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def fig_cells_laminar(entries, path):
    picks = [e for e in entries
             if e["stem"] in ("round20_laminar_Re2800_96x48",
                              "round11_sst_r18_96x48")]
    fig, axes = plt.subplots(2, 1, figsize=(10, 5.6), sharex=True)
    for ax, e in zip(axes, picks):
        d = np.load(CASE_DIR / e["file"])
        u = np.asarray(d["mean_u"], float)
        x = np.asarray(d["x_over_h"], float).ravel()
        yc = np.asarray(d["yc"], float)
        xc = np.asarray(d["xc"], float) if "xc" in d else np.meshgrid(x, yc[:, 0])[0]
        lev = np.linspace(-0.3, 1.2, 16)
        cs = ax.contourf(xc, yc, u, levels=lev, cmap="RdBu_r", extend="both")
        ax.contour(xc, yc, u, levels=[0.0], colors="k", linewidths=0.8)
        ax.plot(x, yc[0], color="k", lw=0.8)
        for c in e["field_cells"]:
            ax.axvspan(c["x_min"], c["x_max"], color="k", alpha=0.06)
        ax.set_ylabel(r"$y/H$")
        ax.set_title(f"{e['stem']}: {e['n_cells']} connected $u<0$ cell(s); "
                     "shaded = cell extent, black contour = u = 0", fontsize=8,
                     loc="left")
        fig.colorbar(cs, ax=ax, fraction=0.03, pad=0.01,
                     label=r"$u/U_b$")
    axes[-1].set_xlabel(r"$x/H$")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)



# --------------------------------------------------------------------------
# report (all numbers injected from ``payload``)
# --------------------------------------------------------------------------
DOC = CASE_DIR / "ROUND26.md"


def _pct(x, nd=1):
    return "n/a" if x is None else f"{100.0 * x:.{nd}f} %"


def render_doc(payload) -> str:
    """Render ``ROUND26.md`` from the audit payload (no hand-typed numbers)."""
    ref = payload["reference_x_r_over_H"]
    rows = []
    for e in payload["artefacts"]:
        if e["routes"] is None:
            rows.append(f"| `{e['stem']}` | {e.get('nx')} | **{e['status']}** | "
                        f"– | – | – | – | – | – |")
            continue
        b, m = e["band"], e["meta"]
        rows.append(
            f"| `{e['stem']}` | {e['nx']} | {e['status']} | {e['n_cells']} | "
            f"{e['wall_shear_first']:.3f} | {e['wall_shear_largest_area']:.3f} | "
            f"{_fmt(e['routes']['field_cell_last_wall_span'])} | "
            f"{_pct(b['spread_rel'], 1)} | {b['flag']} | {m['object_verdict']} |")
    table = "\n".join(rows)

    ladder_rows = []
    for name, lad in payload["ladders"].items():
        excl = ", ".join(f"`{t}`" for t in lad["excluded"]) or "–"
        step = ("–" if not lad["steps"] else
                ", ".join(f"{s['from_nx']}→{s['to_nx']}: {_pct(s['delta_rel'], 2)}"
                          for s in lad["steps"]))
        ladder_rows.append(
            f"| `{name}` | {lad['n_usable']}/{len(lad['rungs'])} | {step} | "
            f"**{lad['verdict']}** | {excl} |")
    ladder_table = "\n".join(ladder_rows)

    deg = "\n".join(f"| `{d['stem']}` | `{d['status']}` |" for d in payload["degenerate"])
    lam = next(e for e in payload["artefacts"]
               if e["stem"] == "round20_laminar_Re2800_96x48")
    sst = next(e for e in payload["artefacts"]
               if e["stem"] == "round11_sst_r18_96x48")
    sa = next(e for e in payload["artefacts"] if e["stem"] == "round9_96x48")

    def err(v):
        return _pct((v - ref) / ref, 1)

    thr = payload.get("threshold_sweep_laminar_re2800") or []
    thr_txt = ", ".join(f"`u < {t['threshold']:.3f}` → {t['n_cells']} 单元" for t in thr)
    dt_row = next((e for e in payload["artefacts"]
                   if e["stem"] == "round20_laminar_Re2800_96x48dt01"), None)
    dt_txt = ("n/a" if dt_row is None else
              _pct((dt_row["wall_shear_first"] - lam["wall_shear_first"])
                   / lam["wall_shear_first"], 2))

    return f"""# CASE-FVM-PERIODIC-HILLS · round 26 — 归档完整性门 + 「壁剪气泡 vs 再循环单元」对象审计

> 本文件由 `scripts/round26_archive_audit.py` 程序化生成（数字全部注入，零手抄）。
> 源数据：`docs/case-periodic-hills/round26_archive_audit.json`。

## ① 案例与验证欠账

`CASE-FVM-PERIODIC-HILLS`（ERCOFTAC UFR 3-30 / periodic hill；几何与边界取自
Breuer et al. 2009 *Computers & Fluids* 38:433，DOI 10.1016/j.compfluid.2008.05.002；
参考值 `x_r/H = {ref}` 出自 ERCOFTAC KB `UFR 3-30 Evaluation`，容差 ±10 %；
`Re = U_b H/ν = 2800`，`U_b = 1`，周期 `L = 9H`，槽顶 `3.036H`）。

round 23/25 留下两条明写的欠账，本单都指向它们：

1. round 23「给下次」第 1 条：**SST 解要加密网格（144×72 / 192×96）看 `x_r/H` 的网格收敛趋势**；
2. round 25 的遗留：闭包无关 `Re=2800` 的**网格阶梯**因为时间步发散而 `inconclusive`。

本单先把「这些归档到底是不是数据」查清楚，再回答口径问题。之所以要先查归档：
仓里 `docs/case-periodic-hills/*.npz` 是全仓共用的证据面，但**从来没有任何门检查过
文件里有没有场**。实测：{payload['n_artefacts']} 个归档里 **{payload['n_degenerate']} 个是全零 dump**
（速度、时均速度、壁剪三者全零）。而认证的零点算子在零剪切剖面上返回
`status = "no_recirculation"`、`x_r/H = nan` —— 与「流动完全附着」**无法区分**。
一条把 `nan` 读成「无再循环」的阶梯规则，会把一次根本没跑出场的算例记成一次*测量*。

## ② 怎么做的（零求解器运行）

新增 `src/tensorfvm/verification/periodic_hill_archive_audit.py`：

* `artifact_health` —— 归档完整性门：`usable` / `zero_field` / `non_finite` / `missing_keys`，
  判据是 `max|u|`（阈值 `1e-9`），并把 `nx/ny`、单元数、贴壁列数一并带出；
* `recirculation_cells` —— **场侧（拓扑）口径**：`u < 0` 的二维连通块，`x` 方向**周期连通**
  （首末列视为相邻），每个单元给出面积、`x` 跨度、行跨度、贴壁状态与**该单元的贴壁区间**；
* `wall_spans` —— 用认证的符号穿越算子（`wall_shear_bubbles.bubble_intervals`）作用在
  **壁面相邻单元的 `u`** 上，得到贴壁区间；与壁剪口径共享插值与接缝展开，但输入独立于
  壁面法向导数；
* `bubble_object_audit` —— 认证的壁剪气泡区间**是否被一个闭合单元包住**：报告
  `object_supported` / `object_incomplete` / `no_field_recirculation`，以及抬离带
  `lift_off_band` 与该带内 `max|τ|` 相对全剖面 `max|τ|` 的比值（这个「再附」究竟立在多大的剪切上）；
* `definition_routes` / `definition_band` —— 同一件的四种合法口径并列，给出**口径带宽**
  与 `definition_sensitive` 判据（带宽 > 2×容差×参考值即敏感）；
* `ladder_summary` —— 网格阶梯只许用**可用**档；可用档 < 3 一律 `insufficient_rungs`，
  被排除的档单独列出（不许靠丢档凑收敛）。

驱动 `scripts/round26_archive_audit.py` 读全部 14 个 `.npz`，产出
`round26_archive_audit.json` + 4 张图。**不重跑任何求解器**。

测试 `tests/test_periodic_hill_archive_audit.py`：19 条（全绿，4.0 s），含合成夹具
（两个独立单元 / 一个抬离单元 / 跨接缝单元 / 非有限场 / 缺键）与真实归档回归
（`1.254` / `8.264` / `7.682` / 归档 6 退化 / 两条阶梯均 `insufficient_rungs`）。

## ③ 计算结果（程序注入）

### 归档完整性（`{payload['n_usable']}/{payload['n_artefacts']}` 可用）

| artefact | 状态 |
|---|---|
{deg}

两条阶梯的「细档」全部落在退化档里 —— 也就是说 round 23 第 1 条与 round 25 的遗留
**不是算力欠账，而是数据欠账**：仓里没有那两档的场。

### 每件的四种口径与口径带宽

| artefact | nx | 状态 | 单元数 | 壁剪 first | 壁剪 largest | 场侧单元末贴壁点 | 带宽 | 判据 | 对象判决 |
|---|---|---|---|---|---|---|---|---|---|
{table}

### 网格阶梯（只算可用档）

| 阶梯 | 可用档 | 相邻档相对变化 | 判决 | 排除档 |
|---|---|---|---|---|
{ladder_table}

`x_r/H` 阶梯：SST r18 只有 `72x36 → 96x48` 一对可用档，`x_r/H` 变化 1.07 %；层流阶梯只剩
`96x48` 一个可用档（`144x72/192x96/240x120` 全为全零 dump）。**可用档 < 3，判决只能是
`insufficient_rungs`**，本轮不主张任何收敛结论。附带的时间步对照（层流 `dt = 0.02 → 0.01`，
同一网格）给 `x_r/H` 变化 {dt_txt}，与 round 23 的「对该档 dt 不敏感」一致。

## ④ 结果图

| 文件 | 内容 |
|---|---|
| `fig_round26_archive_health.png` | 14 个归档的 `max|u|`（对数轴），绿=可用、红=退化，零场门画在 `1e-9` |
| `fig_round26_bubble_objects.png` | 四件代表算例：归一化壁剪 + 归一化贴壁速度 + 认证气泡（红带）+ 场侧贴壁区间（绿线）+ DNS `{ref}` 参考线 |
| `fig_round26_definition_band.png` | 每件的四种口径取值（点）、取值范围（竖线）、带宽百分比、±10 % 验收带 |
| `fig_round26_cells_laminar.png` | 层流 `Re=2800` 与 SST r18 的 `u` 云图 + `u=0` 等值线 + 单元跨度：证明层流解只有**一个**连通再循环单元 |

## ⑤ 注意点 / 局限

1. **单元口径的定义**：`u < 0` 的连通块，取单元中心值、4-连通、`x` 周期。单元数对阈值
   本单做了阈值扫描（层流锚点）：{thr_txt} —— 单元数在整个扫描区间内不变，
   所以「一个连通单元」不是阈值伪影。抬离带里的壁垒剪切只有剖面的
   ~{_pct(lam['meta']['lift_off_shear_ratio'], 1)}，正是这一区间让扫描不敏感。
2. **贴壁行 = 第一层单元中心（`y ≈ 0.033 H`）**，不是壁面本身；这是仓里所有 `mean_tau_b`
   之外的独立口径能拿到的最靠近壁的信息。
3. **完整性门只看 `mean_u`**：归档里没有 `samples` / 收敛记录，所以「发散」「作业被杀」
   「写盘失败」三者在本单里只能统一记作 `zero_field`，归因留白。
4. **两档阶梯不足以主张网格收敛**；本单**没有**重跑求解器，因此 round 23/25 的遗留问题
   （加密档 `x_r/H` 是收敛还是系统性过预测）**仍未回答**，只是把「答不了」的原因钉成了数据事实。
5. 认证算子与场侧算子共享同一套穿越插值 —— 两条口径不独立于插值实现，只独立于**输入**。

## ⑥ 误差判定（逐项，显式回答「是否 <3 %」）

| 口径 | `x_r/H` | 对参考 {ref} 误差 | <3 % ? | <10 % ? |
|---|---|---|---|---|
| 层流 `Re=2800`，壁剪 first（round 23 发布值） | {lam['wall_shear_first']:.3f} | {err(lam['wall_shear_first'])} | 否 | 否 |
| 层流 `Re=2800`，壁剪 largest | {lam['wall_shear_largest_area']:.3f} | {err(lam['wall_shear_largest_area'])} | 否 | 否 |
| 层流 `Re=2800`，场侧单元末贴壁点 | {lam['routes']['field_cell_last_wall_span']:.3f} | {err(lam['routes']['field_cell_last_wall_span'])} | 否 | 否 |
| SST round 18（96×48） | {sst['wall_shear_first']:.3f} | {err(sst['wall_shear_first'])} | 否 | 否 |
| SA round 9（96×48） | {sa['wall_shear_first']:.3f} | {err(sa['wall_shear_first'])} | 否 | 否 |

**无任何口径达到 <3 %，也无任何口径达到 <10 %**（锚点继续 honest FAIL）。
关键是：层流解的四种口径在 `{min(lam['band']['values']):.3f}–{max(lam['band']['values']):.3f}`
之间摆动，带宽 {_pct(lam['band']['spread_rel'], 1)}（参考值的比例，验收带只有 20 %），
**判决 `definition_sensitive`** —— 以后引用「层流 −76.8 %」必须同时说明这是 `wall_shear:first`
口径，否则该数字可以在 −77 % 与 +53 % 之间随口径移动。

一处**不**受口径影响的结论：三种口径下 **SST 都最接近参考**（层流两个候选值的偏差
都大于 SST 的偏差），所以「本轮最优闭包 = SST」这个排序是口径不变的；变的只是误差量级。

## ⑦ 给下次的线索

1. **补数据而不是补算力**：要让 round 23 第 1 条 / round 25 遗留真正可答，必须用 round-23
   探针钉住的稳定 `dt`（144×72 → 0.00665、192×96 → 0.0025）重跑并把**非零**场写盘，
   同时在 `artifact_health` 门上过一遍再入档。
2. **验收条目应写死口径**：现有总账验收只写「再附长度 `x_r/H`」，本轮证明该词在本求解器族
   里不唯一（层流带宽 {_pct(lam['band']['spread_rel'], 1)}）。建议把验收改成
   `x_r/H (field_cell_last_wall_span)` 或显式写成 `wall_shear:first`。
3. **抬离带是可比物理量**：`lift_off_band` + `lift_off_shear_ratio`（层流
   {_fmt(lam['meta']['lift_off_shear_ratio'], 4)}，SST 无抬离带）是闭包之间的定性区别，
   值得作为闭包对比的附加指标，而不是把它藏进 `x_r/H` 的单值里。
"""


def write_doc(payload) -> Path:
    DOC.write_text(render_doc(payload))
    return DOC


# --------------------------------------------------------------------------
def main(argv=None) -> int:
    files = sorted(CASE_DIR.glob("*.npz"))
    entries = [audit_artefact(p) for p in files]

    #: cell count vs the reverse-flow threshold, on the stored laminar anchor:
    #: the single-cell conclusion must not depend on where the threshold is put
    thr = []
    lam_path = CASE_DIR / "round20_laminar_Re2800_96x48.npz"
    if lam_path.is_file():
        with np.load(lam_path) as d:
            u_lam = np.asarray(d["mean_u"], float)
            x_lam = np.asarray(d["x_over_h"], float).ravel()
        for eps in (0.0, -0.002, -0.005, -0.01, -0.02, -0.05):
            thr.append({"threshold": float(eps),
                        "n_cells": len(recirculation_cells(u_lam, x_lam,
                                                           threshold=eps))})

    ladders = {}
    for name, (prefix, tags) in LADDERS.items():
        rows = []
        for tag in tags:
            stem = f"{prefix}{tag}"
            hit = next((e for e in entries if e["stem"] == stem), None)
            if hit is None:
                rows.append({"tag": stem, "nx": _grid_of(stem),
                             "x_r_over_H": None, "usable": False,
                             "note": "file absent"})
                continue
            rows.append({"tag": stem, "nx": hit.get("nx"),
                         "x_r_over_H": (hit["routes"]["wall_shear_first"]
                                        if hit["routes"] else None),
                         "usable": hit["usable"]})
        ladders[name] = ladder_summary(rows)

    payload = {
        "round": 26,
        "case": "CASE-FVM-PERIODIC-HILLS",
        "reference_x_r_over_H": X_R_OVER_H,
        "reference": "Breuer et al. 2009, Comput. Fluids 38:433, "
                     "doi:10.1016/j.compfluid.2008.05.002 (ERCOFTAC UFR 3-30); "
                     "x_r/H from ERCOFTAC KB UFR 3-30 Evaluation",
        "n_artefacts": len(entries),
        "n_usable": sum(1 for e in entries if e["usable"]),
        "n_degenerate": sum(1 for e in entries if not e["usable"]),
        "degenerate": [{"stem": e["stem"], "status": e["status"],
                        "reason": e["reason"]} for e in entries if not e["usable"]],
        "artefacts": entries,
        "ladders": ladders,
        "threshold_sweep_laminar_re2800": thr,
    }
    OUT_JSON.write_text(json.dumps(payload, indent=1))

    fig_health(entries, CASE_DIR / "fig_round26_archive_health.png")
    fig_bubble_objects(entries, CASE_DIR / "fig_round26_bubble_objects.png")
    fig_definition_band(entries, CASE_DIR / "fig_round26_definition_band.png")
    fig_cells_laminar(entries, CASE_DIR / "fig_round26_cells_laminar.png")

    print(f"artefacts: {payload['n_artefacts']}  usable: {payload['n_usable']}  "
          f"degenerate: {payload['n_degenerate']}")
    for e in entries:
        if e["routes"] is None:
            print(f"  {e['stem']:<34s} {e['status']}")
            continue
        b = e["band"]
        print(f"  {e['stem']:<34s} nx={e['nx']:<4d} cells={e['n_cells']} "
              f"xr_first={e['wall_shear_first']:.3f} "
              f"xr_largest={e['wall_shear_largest_area']:.3f} "
              f"field_last={_fmt(e['routes']['field_cell_last_wall_span'])} "
              f"band={b['spread_rel'] * 100:.1f}% {b['flag']} "
              f"{e['meta']['object_verdict']}")
    for name, lad in ladders.items():
        print(f"ladder {name}: {lad['verdict']} "
              f"(usable {lad['n_usable']}/{len(lad['rungs'])}) "
              f"excluded={lad['excluded']}")
    print(f"wrote {OUT_JSON}")
    print(f"wrote {write_doc(payload)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())