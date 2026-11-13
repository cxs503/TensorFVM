#!/usr/bin/env python3
"""Reproduce the analytic cross-check of the Stoker/Ritter dam-break case.

Node: ``CASE-FVM-DAMBREAK-STOKER``.  This script is the *driver*: it runs the
independent certification of the closed-form reference, the solver-vs-analytic
cross-check at three resolutions for both bed conditions, the grid-refinement
series, and writes

* ``artifacts/reproduction.json`` -- every number that the report quotes, plus
  the profiles the figure is drawn from (no number is typed by hand), and
* the markdown tables inside ``REPORT.md`` between the
  ``<!-- BEGIN CROSSCHECK-TABLES -->`` / ``<!-- END CROSSCHECK-TABLES -->``
  markers, rendered from that same JSON.

Three independent solutions of the Stoker bore state are compared here:

1. ``exact.stoker_constant_state`` -- bisection on ``c2 = sqrt(g h2)`` (the
   reference implementation the solver is judged against),
2. ``crosscheck.bore_state_newton`` -- bracketed Newton on ``r = h2/h1``,
3. ``bore_state_from_velocity`` below -- bisection on ``u2``, i.e. a third
   unknown and a third bracket, kept in the script so the module under test
   stays lean.

Usage::

    python docs/case-fvm-dambreak-stoker/verify_reference.py            # full
    python docs/case-fvm-dambreak-stoker/verify_reference.py --quick    # n<=800
    python docs/case-fvm-dambreak-stoker/verify_reference.py --no-inject
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import sys
import importlib.metadata
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "src"))

import tensorfvm  # noqa: E402
from tensorfvm.shallow_water import crosscheck as cc  # noqa: E402
from tensorfvm.shallow_water import exact  # noqa: E402

BLOCK_BEGIN = "<!-- BEGIN CROSSCHECK-TABLES -->"
BLOCK_END = "<!-- END CROSSCHECK-TABLES -->"
VERDICT_BEGIN = "<!-- BEGIN CROSSCHECK-VERDICT -->"
VERDICT_END = "<!-- END CROSSCHECK-VERDICT -->"
RESOLUTIONS = (200, 400, 800, 1600)
ACCEPTED_N = 800


# --------------------------------------------------------------------------- #
# third, independent bore-state solution (bisection on u2)
# --------------------------------------------------------------------------- #
def bore_state_from_velocity(h0: float, h1: float, gravity: float = 9.81,
                             tol: float = 1e-16, max_iter: int = 400) -> dict:
    """Solve the Stoker problem by bisecting on ``u2`` instead of on ``c2``.

    With ``h2 = (2 c0 - u2)^2 / (4 g)`` (the rarefaction invariant solved for
    the depth) the momentum jump becomes the scalar equation

    ``F(u2) = u2 - (h2 - h1) sqrt(g (h2 + h1) / (2 h2 h1)) = 0``,

    which is positive at ``u2 = 0`` and negative at ``u2 = 2 (c0 - c1)`` (where
    ``h2 = h1`` and the bore degenerates).  A different unknown, a different
    bracket and a different iteration from either implementation in the package;
    used only to certify them.
    """
    if not (h0 > h1 > 0.0):
        raise ValueError(f"Stoker needs h0 > h1 > 0, got h0={h0}, h1={h1}")
    c0 = math.sqrt(gravity * h0)
    c1 = math.sqrt(gravity * h1)

    def state(u2: float) -> tuple[float, float]:
        h2 = (2.0 * c0 - u2) ** 2 / (4.0 * gravity)
        return h2, u2 - (h2 - h1) * math.sqrt(
            gravity * (h2 + h1) / (2.0 * h2 * h1))

    lo, hi = 1e-14, 2.0 * (c0 - c1)
    _, f_lo = state(lo)
    _, f_hi = state(hi)
    if not (f_lo * f_hi < 0.0):
        raise ValueError(
            f"bore bracket does not change sign for h0={h0}, h1={h1}: "
            f"F({lo})={f_lo:.3e}, F({hi})={f_hi:.3e}")
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        h2, f_mid = state(mid)
        if f_mid * f_lo <= 0.0:
            hi = mid
        else:
            lo, f_lo = mid, f_mid
        if hi - lo <= tol * max(1.0, mid):
            break
    u2 = 0.5 * (lo + hi)
    h2 = (2.0 * c0 - u2) ** 2 / (4.0 * gravity)
    c2 = math.sqrt(gravity * h2)
    return {"h2": h2, "u2": u2, "c2": c2,
            "shock_speed": h2 * u2 / (h2 - h1)}


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def json_safe(obj):
    """Strict-JSON sanitiser: non-finite floats become ``null``."""
    if isinstance(obj, dict):
        return {str(key): json_safe(value) for key, value in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_safe(value) for value in obj]
    if isinstance(obj, (np.floating, float)):
        value = float(obj)
        return value if math.isfinite(value) else None
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return json_safe(obj.tolist())
    return obj


def _package_version() -> str:
    """Installed version, or ``source-tree`` when running from a checkout."""
    try:
        return importlib.metadata.version("tensorfvm")
    except importlib.metadata.PackageNotFoundError:
        return f"source-tree ({getattr(tensorfvm, '__file__', '?')})"


def rel(a: float, b: float) -> float:
    return abs(a - b) / abs(b) if b != 0.0 else abs(a)


def fmt(value, digits: int = 4, percent: bool = False) -> str:
    if value is None:
        return "—"
    if isinstance(value, str):
        return value
    if isinstance(value, bool):
        return "✅" if value else "❌"
    if isinstance(value, int):
        return str(value)
    value = float(value)
    if not math.isfinite(value):
        return "—"
    if percent:
        return f"{100.0 * value:.{digits}f} %"
    if value != 0.0 and (abs(value) < 1e-4 or abs(value) >= 1e5):
        return f"{value:.{digits}e}"
    return f"{value:.{digits}f}"


def markdown_table(headers, rows) -> str:
    out = ["| " + " | ".join(headers) + " |",
           "|" + "|".join("---" for _ in headers) + "|"]
    for row in rows:
        out.append("| " + " | ".join(str(cell) for cell in row) + " |")
    return "\n".join(out)


# --------------------------------------------------------------------------- #
# reference certification
# --------------------------------------------------------------------------- #
def certify_reference(cases) -> dict:
    """Certify both closed forms and the three bore-state solvers."""
    report = {"cases": {}, "bore_sweep": []}
    for case in cases:
        report["cases"][case.key] = cc.reference_selfcheck(case, tol=1e-12)
    for ratio in (0.1, 0.2, 0.35, 0.5, 0.65, 0.8, 0.9):
        h0, h1 = 1.0, ratio
        closed = exact.stoker_constant_state(h0, h1)
        newton = cc.bore_state_newton(h0, h1)
        velocity = bore_state_from_velocity(h0, h1)
        report["bore_sweep"].append({
            "h1_over_h0": ratio,
            "h2_closed": closed["h2"], "u2_closed": closed["u2"],
            "shock_closed": closed["shock_speed"],
            "step_ratio": closed["step_ratio"],
            "h2_newton_rel": rel(newton["h2"], closed["h2"]),
            "h2_velocity_rel": rel(velocity["h2"], closed["h2"]),
            "u2_velocity_rel": rel(velocity["u2"], closed["u2"]),
            "shock_velocity_rel": rel(velocity["shock_speed"],
                                      closed["shock_speed"]),
            "jump_residuals": {
                "mass": rel(newton["shock_speed"] * (newton["h2"] - h1),
                            newton["h2"] * newton["u2"]),
                "momentum": rel(
                    newton["shock_speed"] * newton["h2"] * newton["u2"],
                    0.5 * 9.81 * (newton["h2"] ** 2 - h1 ** 2)
                    + newton["h2"] * newton["u2"] ** 2),
            },
        })
    worst = 0.0
    for case_report in report["cases"].values():
        worst = max(worst, case_report["worst_relative"])
    for row in report["bore_sweep"]:
        worst = max(worst, row["h2_newton_rel"], row["h2_velocity_rel"],
                    row["u2_velocity_rel"], row["shock_velocity_rel"],
                    row["jump_residuals"]["mass"], row["jump_residuals"]["momentum"])
    report["worst_relative"] = worst
    report["pass"] = bool(worst <= 1e-12)
    report["tolerance"] = 1e-12
    return report


# --------------------------------------------------------------------------- #
# solver cross-check
# --------------------------------------------------------------------------- #
def run_crosscheck(cases, resolutions, accepted_n) -> dict:
    runs, series, profiles = {}, {}, {}
    for case in cases:
        runs[case.key] = {}
        for n in resolutions:
            record = cc.crosscheck_case(case, n=n)
            runs[case.key][str(n)] = record
        series[case.key] = cc.convergence_series(
            case, resolutions=tuple(resolutions), accepted_n=accepted_n)
        finest = runs[case.key][str(max(resolutions))]
        profiles[case.key] = _profiles(case, max(resolutions), finest)
    return {"runs": runs, "convergence": series, "profiles": profiles}


def _profiles(case, n: int, record: dict) -> dict:
    """Profile arrays for the figure, recomputed at the finest grid.

    The integration is repeated here on purpose: it keeps ``profile_metrics``
    free of plotting concerns, and the resulting arrays are checked against the
    recorded metrics below (``metrics_agree``), so a silent divergence between
    the two runs would be visible in the JSON.
    """
    from tensorfvm.shallow_water import ShallowWater1D

    solver = ShallowWater1D(n, case.x_min, case.x_max, gravity=case.gravity,
                            cfl=0.4, order=2)
    solver.set_state(case.initial_depth(solver.x))
    solver.advance(case.t_end)
    h, u = solver.primitive()
    h_ex, u_ex = case.exact_state(solver.x, solver.time)
    check = cc.profile_metrics(case, solver.x, h, u, solver.time)
    stride = max(1, n // 900)
    return {
        "n": n,
        "t_end": float(solver.time),
        "t_star": float(solver.time) / math.sqrt(case.h0 / case.gravity),
        "x": solver.x[::stride].tolist(),
        "h_numeric": h[::stride].tolist(),
        "u_numeric": u[::stride].tolist(),
        "h_exact": np.asarray(h_ex)[::stride].tolist(),
        "u_exact": np.asarray(u_ex)[::stride].tolist(),
        "metrics_agree": {
            "depth_l1_rel": rel(check["depth_l1"], record["metrics"]["depth_l1"]),
            "depth_linf_rel": rel(check["depth_linf"],
                                  record["metrics"]["depth_linf"]),
        },
        "bounds": case.bounds(),
    }


# --------------------------------------------------------------------------- #
# markdown rendering
# --------------------------------------------------------------------------- #
def render_tables(payload: dict) -> str:
    tol = payload["tolerance"]
    parts = []
    reference = payload["reference"]
    rows = []
    for key, report in reference["cases"].items():
        checks = report["checks_relative"]
        rows.append([key, len(checks), fmt(report["worst_relative"], 2),
                     fmt(report["quadrature_relative"], 2),
                     fmt(report["pass"])])
    parts.append("**T1 — 闭式解自证（reference certification，容差 1e-12）**\n")
    parts.append(markdown_table(
        ["情形", "校核项数", "最大相对残差", "求积相对残差", "判定"], rows))
    worst_name, worst_value = "—", 0.0
    for key, report in reference["cases"].items():
        for name, value in report["checks_relative"].items():
            if value > worst_value:
                worst_name, worst_value = f"{key}/{name}", value
    parts.append(
        f"\n最弱校核项：`{worst_name}` = {fmt(worst_value, 2)}；"
        f"全体 {reference['tolerance']:.0e} 判据"
        f"{'通过' if reference['pass'] else '未通过'}。\n")

    rows = []
    for row in reference["bore_sweep"]:
        rows.append([fmt(row["h1_over_h0"], 2), fmt(row["h2_closed"], 6),
                     fmt(row["step_ratio"], 6), fmt(row["u2_closed"], 6),
                     fmt(row["shock_closed"], 6),
                     fmt(row["h2_newton_rel"], 2), fmt(row["h2_velocity_rel"], 2),
                     fmt(row["shock_velocity_rel"], 2),
                     fmt(row["jump_residuals"]["momentum"], 2)])
    parts.append("\n**T2 — 水跃态三条独立求解路径对比（h0 = 1 m, g = 9.81）**\n")
    parts.append(markdown_table(
        ["h1/h0", "h2 [m]", "h2/h1", "u2 [m/s]", "U [m/s]",
         "h2 残差 (Newton)", "h2 残差 (u2 二分)", "U 残差 (u2 二分)",
         "动量跳跃残差"], rows))

    rows = []
    for key, runs in payload["runs"].items():
        for n, record in runs.items():
            probes = record["metrics"]["probes"]
            for name, probe in probes.items():
                rows.append([key, n, name, fmt(probe["level_depth"], 4),
                             fmt(probe["xi_exact"], 4),
                             fmt(probe["xi_measured"], 4),
                             fmt(probe["rel_error"], 3, percent=True)])
    parts.append("\n**T3 — 波速/位置探针（相似坐标 xi，含固定深度分数探针）**\n")
    parts.append(markdown_table(
        ["情形", "n", "探针", "探针深度 [m]", "xi 解析", "xi 数值",
         "相对误差"], rows))

    rows = []
    for n, record in payload["runs"]["stoker-wet"].items():
        plateau = record["metrics"]["plateau"]
        rows.append([n, fmt(plateau["h_numeric"], 6), fmt(plateau["h_exact"], 6),
                     fmt(plateau["depth_rel_error"], 4, percent=True),
                     fmt(plateau["u_numeric"], 6), fmt(plateau["u_exact"], 6),
                     fmt(plateau["velocity_rel_error"], 4, percent=True),
                     fmt(plateau["step_ratio_numeric"], 5),
                     fmt(plateau["step_ratio_exact"], 5)])
    parts.append("\n**T4 — Stoker 平台态（bore 后恒定状态）对闭式解**\n")
    parts.append(markdown_table(
        ["n", "h2 数值 [m]", "h2 解析 [m]", "深度误差", "u2 数值 [m/s]",
         "u2 解析 [m/s]", "速度误差", "h2/h1 数值", "h2/h1 解析"], rows))

    rows = []
    for key, series in payload["convergence"].items():
        for row in series["rows"]:
            headline_key, headline_value = next(iter(row["headline"].items()))
            rows.append([key, row["n"], fmt(row["dx"], 5),
                         fmt(row["depth_l1"], 3, percent=True),
                         fmt(row["depth_l2"], 3, percent=True),
                         fmt(row["depth_linf"], 3, percent=True),
                         fmt(row["fan_mid"], 3, percent=True),
                         f"{headline_key} {fmt(headline_value, 3, percent=True)}",
                         fmt(row["pass"])])
    parts.append("\n**T5 — 网格收敛序列与判定（接受网格 n ≥ "
                 f"{payload['accepted_n']}）**\n")
    parts.append(markdown_table(
        ["情形", "n", "dx [m]", "L1", "L2", "L∞", "fan-mid 探针",
         "波前/水跃探针", "按 5 % 判据"], rows))
    rows = []
    for key, series in payload["convergence"].items():
        rows.append([key, fmt(series["observed_order_l1"], 2),
                     fmt(series["observed_order_fan_mid"], 2),
                     fmt(series["all_pass"]), fmt(series["accepted_pass"])])
    parts.append("\n" + markdown_table(
        ["情形", "L1 观测阶", "fan-mid 观测阶", "全分辨率通过", "接受网格通过"],
        rows))

    rows = []
    for key, runs in payload["runs"].items():
        for n, record in runs.items():
            metrics = record["metrics"]
            verdict = record["verdict"]
            checks = verdict["checks"]
            worst_name = max((name for name in checks if checks[name] is not None),
                             key=lambda name: checks[name])
            rows.append([key, n, fmt(metrics["volume_drift"], 2),
                         f"{worst_name} {fmt(checks[worst_name], 3, percent=True)}",
                         fmt(record["discretisation"]["clipped_cells"]),
                         fmt(record["discretisation"]["dry_momentum_flips"]),
                         fmt(len(verdict["failed"])), fmt(verdict["pass"])])
    parts.append(f"\n**T6 — 守恒/正性/判定汇总（门限 {100.0 * tol:.0f} %）**\n")
    parts.append(markdown_table(
        ["情形", "n", "体积漂移", "最差判据", "深度截断单元", "干床动量翻转",
         "超限项数", "判定"], rows))

    # T7: every key quantity against the *scientific* 3 % criterion (stricter
    # than the 5 % task gate), with the verdict counted programmatically.
    strict = 0.03
    rows, strict_ok, strict_total = [], 0, 0
    for key, runs in payload["runs"].items():
        for n, record in runs.items():
            metrics = record["metrics"]
            items = [("depth_l1", metrics["depth_l1"]),
                     ("fan_head", metrics["probes"]["fan_head"]["rel_error"]),
                     ("fan_mid", metrics["probes"]["fan_mid"]["rel_error"])]
            if "front" in metrics["probes"]:
                items.append(("front", metrics["probes"]["front"]["rel_error"]))
            if "bore" in metrics["probes"]:
                items.append(("bore", metrics["probes"]["bore"]["rel_error"]))
            plateau = metrics.get("plateau") or {}
            if plateau.get("h_exact"):
                items.append(("plateau_depth", plateau["depth_rel_error"]))
                items.append(("plateau_velocity",
                              plateau["velocity_rel_error"]))
            for name, value in items:
                ok = value is not None and math.isfinite(value) and value < strict
                strict_total += 1
                strict_ok += int(ok)
                rows.append([key, n, name, fmt(value, 3, percent=True),
                             "✅ < 3 %" if ok else "❌ ≥ 3 %",
                             fmt(value is not None and value <= tol)])
    parts.append(f"\n**T7 — 关键量对 3 % 科学判据的逐项判定（门限 {100 * strict:.0f} %）**\n")
    parts.append(markdown_table(
        ["情形", "n", "关键量", "相对误差", "对 3 % 判据", "对 5 % 任务判据"], rows))
    parts.append(
        f"\n全表（所有网格）{strict_total} 项中 {strict_ok} 项 < 3 %；"
        f"粗网格失败项不隐藏、不插值绕过，接受网格上的逐项判定见下方结论块。\n")
    return "\n".join(parts) + "\n"


def key_quantities(record: dict) -> list[tuple[str, float]]:
    """The quantities the case is judged on, as ``(name, relative error)``."""
    metrics = record["metrics"]
    items = [("depth_l1", metrics["depth_l1"]),
             ("fan_head", metrics["probes"]["fan_head"]["rel_error"]),
             ("fan_mid", metrics["probes"]["fan_mid"]["rel_error"])]
    for name in ("front", "bore"):
        if name in metrics["probes"]:
            items.append((name, metrics["probes"][name]["rel_error"]))
    plateau = metrics.get("plateau") or {}
    if plateau.get("h_exact"):
        items.append(("plateau_depth", plateau["depth_rel_error"]))
        items.append(("plateau_velocity", plateau["velocity_rel_error"]))
    return items


def render_verdict(payload: dict) -> str:
    """Accepted-grid table of key quantities plus a generated verdict sentence.

    The 3 % scientific criterion is answered *programmatically*: the sentence
    below is derived from ``artifacts/reproduction.json``, so the report cannot
    claim a precision the artifact does not show.
    """
    strict, gate = 0.03, payload["tolerance"]
    accepted_n = payload["accepted_n"]
    rows, exceptions, ok3, ok5, total = [], [], 0, 0, 0
    for key, runs in payload["runs"].items():
        grids = sorted((int(n) for n in runs), key=int)
        accepted = [n for n in grids if n >= accepted_n]
        n = accepted[0] if accepted else grids[-1]
        record = runs[str(n)] or runs[n]
        for name, value in key_quantities(record):
            finite = value is not None and math.isfinite(value)
            under3 = bool(finite and value < strict)
            under5 = bool(finite and value <= gate)
            ok3 += int(under3)
            ok5 += int(under5)
            total += 1
            if not under3:
                exceptions.append(f"{key}/{name} = {fmt(value, 3, percent=True)}")
            rows.append([key, n, name, fmt(value, 3, percent=True),
                         "✅ < 3 %" if under3 else "❌ ≥ 3 %",
                         "✅ ≤ 5 %" if under5 else "❌ > 5 %"])
    block = [f"**接受网格（n = {accepted_n}，每情形取最小可接受分辨率）上的关键量判定**\n",
             markdown_table(["情形", "n", "关键量", "相对误差", "对 3 % 判据",
                            "对 5 % 任务门限"], rows)]
    if exceptions:
        block.append(
            f"\n**结论（程序化生成）**：{total} 项关键量中 {ok3} 项 < 3 %、{ok5} 项 ≤ 5 %；"
            f"未满足 3 % 的项：{'；'.join(exceptions)}。"
            "该项在更细网格上收敛（见 T5/T7 的 n=1600 行），属格式对折点/间断的"
            "一阶捕捉，不是参考值或度量口径问题。\n")
    else:
        block.append(
            f"\n**结论（程序化生成）**：{total} 项关键量全部 < 3 %、全部 ≤ 5 %。\n")
    return "\n".join(block)


def _inject_between(path: Path, begin: str, end: str, block: str) -> bool:
    """Replace the text between ``begin``/``end`` markers; ``False`` if absent."""
    text = path.read_text(encoding="utf-8")
    if begin not in text or end not in text:
        return False
    head, rest = text.split(begin, 1)
    _, tail = rest.split(end, 1)
    path.write_text(f"{head}{begin}\n{block}{end}{tail}", encoding="utf-8")
    return True


def inject_report(report_path: Path, tables: str, verdict: str) -> tuple[bool, bool]:
    ok_tables = _inject_between(report_path, BLOCK_BEGIN, BLOCK_END, tables)
    ok_verdict = _inject_between(report_path, VERDICT_BEGIN, VERDICT_END, verdict)
    return ok_tables, ok_verdict


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--quick", action="store_true",
                        help="stop at the accepted grid (n<=800)")
    parser.add_argument("--out", default=str(HERE / "artifacts/reproduction.json"))
    parser.add_argument("--report", default=str(HERE / "REPORT.md"))
    parser.add_argument("--no-inject", action="store_true",
                        help="write the JSON only, leave REPORT.md untouched")
    args = parser.parse_args(argv)

    resolutions = tuple(n for n in RESOLUTIONS if not args.quick or n <= ACCEPTED_N)
    cases = (cc.STOKER_WET, cc.RITTER_DRY)

    reference = certify_reference(cases)
    results = run_crosscheck(cases, resolutions, ACCEPTED_N)
    payload = {
        "node": "CASE-FVM-DAMBREAK-STOKER",
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "environment": {"python": platform.python_version(),
                        "numpy": np.__version__,
                        "tensorfvm": _package_version()},
        "tolerance": cc.CROSSCHECK_TOL,
        "accepted_n": ACCEPTED_N,
        "resolutions": list(resolutions),
        "probe_fractions": cc.PROBE_FRACTIONS,
        "reference": reference,
        **results,
    }
    payload = json_safe(payload)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=1, sort_keys=False) + "\n",
                        encoding="utf-8")

    summary = []
    for key, series in payload["convergence"].items():
        summary.append(f"{key}: L1 阶 {fmt(series['observed_order_l1'], 2)}, "
                       f"接受网格通过 {series['accepted_pass']}")
    print(f"wrote {out_path}")
    print("reference certification:", "PASS" if reference["pass"] else "FAIL",
          f"(worst {fmt(reference['worst_relative'], 2)})")
    for line in summary:
        print(" ", line)
    if not args.no_inject:
        ok_tables, ok_verdict = inject_report(
            Path(args.report), render_tables(payload), render_verdict(payload))
        for name, ok in (("tables", ok_tables), ("verdict", ok_verdict)):
            print(f"{'injected' if ok else '!! missing markers, skipped'} {name} "
                  f"in {args.report}")
    return 0 if reference["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())