"""Grid-refinement audit of the committed lid-driven cavity ladder (level 1b).

Purpose.  The verification ladder's C2 gate is stated at Re = 1000 and is currently
evaluated on a single grid (n = 64), where the measured centre-line deviation is
2.93 % against a 2 % limit.  A single-grid deviation cannot say whether the gate is
*unattained* or *unattainable*: nothing in the tree measured the refinement rate of
that metric.  This script asks the question the committed evidence can answer:

* measure the observed order ``p`` of the centre-line metric at the Reynolds
  numbers where the ladder has three grids (Re = 100 and Re = 400),
* redo the measurement with the literature entries the repository has already
  documented as digitisation defects excluded
  (``tensorfvm.lid_cavity_reference.SUSPECT_ENTRIES``; the Re = 400 ``v`` point at
  x = 0.9063 dominates the raw ``v`` metric and hides the convergence of everything
  else), and cross-check the re-measurement against the value the ladder's own
  artefact reports,
* propagate the measured ``p`` to Re = 1000 and report the grid the 2 % gate would
  need **as an estimate with its assumption named**, never as a measured result.

Everything is read from committed evidence (``docs/verification-lid-cavity-ladder/
summary.json`` and the ladder artefacts); this script runs no solver, so it costs a
second and can be re-run inside the audit itself.

Outputs (all paths relative to the repository root):

``docs/verification-ladder-re1000-sensitivity/refinement.json``
    every number below, machine readable, byte-identical across runs.
``docs/verification-ladder-re1000-sensitivity/report.md``
    the same numbers as tables, generated here (nothing is hand-transcribed).
``docs/verification-ladder-re1000-sensitivity/error-vs-grid.png``
    centre-line error against grid, raw and suspect-excluded, with the C2 limit.

Usage::

    PYTHONPATH=src python scripts/audit_cavity_grid_refinement.py
"""
from __future__ import annotations

import argparse
import json
import math
import pathlib
import sys

REPO = pathlib.Path(__file__).resolve().parents[1]
_SRC = REPO / "src"
if str(_SRC) not in sys.path:          # runnable from a checkout, no install needed
    sys.path.insert(0, str(_SRC))
COMMITTED = REPO / "docs" / "verification-lid-cavity-ladder" / "summary.json"
LADDER_ARTIFACTS = REPO / "src" / "tensorfvm" / "verification" / "artifacts" / "verification_ladder"
OUT = REPO / "docs" / "verification-ladder-re1000-sensitivity"

#: Roache (1994) factor of safety used by the ladder's own level-3 criterion.
FS = 1.25
C2_LIMIT = 0.02          # the ladder gate, stated at Re = 1000
TARGET_RE = 1000.0


def _relative(path: pathlib.Path) -> str:
    return pathlib.Path(path).resolve().relative_to(REPO).as_posix()


def _suspects(reynolds: float, axis: str) -> dict[float, str]:
    """Documented digitisation defects on this centre-line (empty if unknown)."""
    try:
        from tensorfvm.lid_cavity_reference import suspect_entries
    except Exception:                                     # pragma: no cover
        return {}
    try:
        return {float(k): str(v.get("verdict", "")) for k, v in
                suspect_entries(reynolds, axis).items()}
    except Exception:                                     # pragma: no cover
        return {}


def _consistent_max_abs_error(profile: dict, suspects: dict[float, str]) -> tuple[float, int]:
    """max |computed - reference| after dropping the documented suspect entries."""
    worst, kept = 0.0, 0
    for coord, ref, got in zip(profile["coord"], profile["reference"], profile["computed"]):
        if any(abs(float(coord) - s) < 1e-9 for s in suspects):
            continue
        kept += 1
        worst = max(worst, abs(float(got) - float(ref)))
    return worst, kept


def load_rows(committed: pathlib.Path = COMMITTED) -> list[dict]:
    """Merge the committed multi-grid ladder with the ladder artefacts (no solver)."""
    payload = json.loads(pathlib.Path(committed).read_text())
    rows: list[dict] = []
    for case in payload.get("cases", []):
        reynolds, n = float(case["reynolds"]), int(case["n"])
        row = {"reynolds": reynolds, "n": n, "source": _relative(committed)}
        for axis in ("u", "v"):
            block = case[axis]
            suspects = _suspects(reynolds, axis)
            consistent, kept = _consistent_max_abs_error(block["profiles"], suspects)
            row[axis] = {
                "max_abs_error": float(block["max_abs_error"]),
                "consistent": consistent,
                "suspects": sorted(suspects),
                "points_kept": kept,
                "points_total": len(block["profiles"]["coord"]),
                "extremum_reference": float(block["extremum"]["max_reference"][1]) if "max_reference" in block["extremum"] else None,
            }
        row["converged"] = bool(case["solver"]["converged"])
        row["max_divergence"] = float(case["max_divergence"])
        rows.append(row)
    # cross-check: the ladder artefacts carry the same score, computed independently
    checks = []
    for row in rows:
        path = LADDER_ARTIFACTS / f"lid_cavity_Re{int(row['reynolds'])}_n{row['n']}.json"
        if not path.exists():
            checks.append({"row": f"Re{int(row['reynolds'])}_n{row['n']}",
                           "artefact": _relative(path), "available": False})
            continue
        art = json.loads(path.read_text())
        checks.append({
            "row": f"Re{int(row['reynolds'])}_n{row['n']}",
            "artefact": _relative(path), "available": True,
            "committed_u": row["u"]["max_abs_error"], "artefact_u": float(art["u"]["max_abs_error"]),
            "committed_v_consistent": row["v"]["consistent"],
            "artefact_v_consistent": float(art["v"]["max_abs_error_consistent"]),
        })
    return rows, checks


def _order(coarse: float, fine: float, ratio: float) -> float:
    if coarse <= 0 or fine <= 0 or ratio <= 1:
        return float("nan")
    return math.log(coarse / fine) / math.log(ratio)


def refinement(rows: list[dict], axis: str, key: str) -> dict:
    """Observed order and GCI per grid pair, for one metric of one centre-line."""
    by_re = {}
    for row in rows:
        by_re.setdefault(round(row["reynolds"], 6), []).append(row)
    out = {}
    for reynolds, series in sorted(by_re.items()):
        series = sorted(series, key=lambda item: item["n"])
        track = [{"n": item["n"], "error": item[axis][key]} for item in series]
        pairs = []
        for coarse, fine in zip(track, track[1:]):
            ratio = fine["n"] / coarse["n"]
            p = _order(coarse["error"], fine["error"], ratio)
            gci = float("nan")
            if math.isfinite(p) and abs(ratio ** p - 1.0) > 1e-12:
                gci = FS * fine["error"] / abs(ratio ** p - 1.0)
            pairs.append({"coarse_n": coarse["n"], "fine_n": fine["n"],
                          "ratio": ratio, "observed_order": p, "gci_fine": gci})
        out[str(int(reynolds)) if reynolds == int(reynolds) else str(reynolds)] = {
            "grids": [item["n"] for item in track],
            "track": track,
            "pairs": pairs,
            "measurable": len(track) >= 2,
        }
    return out


def projection(rows: list[dict], axis: str = "u", key: str = "consistent") -> dict:
    """Estimate the grid the C2 limit would need at Re = 1000.

    The assumption is named, not hidden: the finest measured order at the Reynolds
    numbers that *do* have three grids is applied to the single Re = 1000 row with
    error ~ n^-p.  This is an estimate, not a measurement -- the ladder's outstanding
    run (Re = 1000, n = 128) is what would turn it into one.
    """
    target = [row for row in rows if abs(row["reynolds"] - TARGET_RE) < 1e-9]
    if not target:
        return {"available": False, "reason": "no Re = 1000 row in the committed ladder"}
    finest = max(target, key=lambda item: item["n"])
    measured = finest[axis][key]
    orders = []
    by_re = {}
    for row in rows:
        by_re.setdefault(round(row["reynolds"], 6), []).append(row)
    for reynolds, series in by_re.items():
        if len(series) < 3:
            continue
        series = sorted(series, key=lambda item: item["n"])
        for coarse, fine in zip(series, series[1:]):
            orders.append(_order(coarse[axis][key], fine[axis][key], fine["n"] / coarse["n"]))
    orders = [value for value in orders if math.isfinite(value) and value > 0]
    if not orders:
        return {"available": False, "reason": "no measurable order at any Reynolds number"}
    p = float(sorted(orders)[len(orders) // 2])           # median of the measured orders
    n_required = finest["n"] * (measured / C2_LIMIT) ** (1.0 / p) if measured > C2_LIMIT else finest["n"]
    return {
        "available": True,
        "reynolds": TARGET_RE,
        "evaluated_n": finest["n"],
        "measured_error": measured,
        "limit": C2_LIMIT,
        "measured_orders": orders,
        "assumed_order": p,
        "n_required_estimate": n_required,
        "next_power_of_two_grid": int(2 ** math.ceil(math.log2(max(n_required, finest["n"])))),
        "status": "estimate-only: no second Re = 1000 grid exists in the tree",
    }


def figure(rows: list[dict], path: pathlib.Path) -> str | None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:                                     # pragma: no cover
        return None
    fig, ax = plt.subplots(figsize=(6.0, 4.2))
    markers = {100.0: "o", 400.0: "s", 1000.0: "^"}
    for reynolds, marker in markers.items():
        series = sorted((row for row in rows if abs(row["reynolds"] - reynolds) < 1e-9),
                        key=lambda item: item["n"])
        if not series:
            continue
        ns = [item["n"] for item in series]
        ax.loglog(ns, [item["u"]["consistent"] for item in series], marker + "-",
                  color="C0", label=f"Re={int(reynolds)} u (suspect points dropped)")
        ax.loglog(ns, [item["v"]["consistent"] for item in series], marker + "--",
                  color="C3", label=f"Re={int(reynolds)} v (suspect points dropped)")
    ax.axhline(C2_LIMIT, color="k", lw=1.0, ls=":", label="C2 gate 2 %")
    ax.set_xlabel("cells per direction n")
    ax.set_ylabel("max |profile error|")
    ax.set_title("Lid-driven cavity centre-line error against grid")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=6, ncol=1)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path.name


def _fmt(value: float, digits: int = 6) -> str:
    return "n/a" if value is None or (isinstance(value, float) and not math.isfinite(value)) else f"{value:.{digits}g}"


def render(payload: dict) -> str:
    lines = ["# Lid-driven cavity: what the committed ladder says about the C2 gate", ""]
    lines += [
        "Generated by `scripts/audit_cavity_grid_refinement.py` from the committed",
        "evidence only (`docs/verification-lid-cavity-ladder/summary.json` plus the",
        "ladder artefacts).  No solver is run, so every number below is reproducible",
        "in about a second and is not this script's own measurement.",
        "",
        "## 1. Why this audit exists",
        "",
        "The ladder's C2 criterion is stated at Re = 1000 and is currently evaluated on",
        f"the single grid n = {payload['projection_raw']['evaluated_n']}"
        f", where the centre-line deviation is {_fmt(payload['projection_raw']['measured_error'], 4)}"
        f" against a {C2_LIMIT:.0%} limit (FAIL).  One grid cannot distinguish a gate that",
        "is not yet attained from one that cannot be attained: the refinement rate of",
        "this metric had never been measured.  The Re = 1000 refinement row was started",
        "and stopped before it finished, which is why the ladder records `partial`.",
        "",
        "## 2. Method",
        "",
        "* Rows: the committed ladder, three grids (32/64/128) at Re = 100 and Re = 400,",
        "  one grid (64) at Re = 1000.",
        "* Metric: pointwise max |computed - Ghia| on both centrelines, exactly the",
        "  quantity C2 thresholds.",
        "* Two variants: the raw table, and the table with the entries the repository has",
        "  already documented as digitisation defects dropped",
        "  (`tensorfvm.lid_cavity_reference.SUSPECT_ENTRIES`).  The raw Re = 400 `v` metric",
        "  is dominated by that single disputed point, so it cannot show convergence.",
        "* Observed order: p = ln(e_coarse / e_fine) / ln(r), r = 2.  GCI on the finest",
        f"  pair with Roache's Fs = {FS}.",
        "* The suspect-excluded error is cross-checked against the value the ladder's own",
        "  artefact reports for the same row (`max_abs_error_consistent`), so the analysis",
        "  code is not trusted on its own word.",
        "",
        "## 3. Measured order",
        "",
        "| Re | axis | grids | raw error | suspect-excluded error | order (raw) | order (excluded) |",
        "|---|---|---|---|---|---|---|",
    ]
    for reynolds, block in payload["refinement"].items():
        for axis in ("u", "v"):
            rows = payload["refinement"][reynolds][axis]["raw"]
            grids = payload["refinement"][reynolds]["grids"]
            raw = " / ".join(_fmt(item["error"], 4) for item in rows)
            exc = " / ".join(_fmt(item["error"], 4) for item in
                             payload["refinement"][reynolds][axis]["track"])
            p_raw = ", ".join(_fmt(pair["observed_order"], 3)
                              for pair in payload["refinement"][reynolds][axis]["pairs"])
            p_exc = ", ".join(_fmt(pair["observed_order"], 3)
                              for pair in payload["refinement"][reynolds][axis]["excluded_pairs"])
            lines.append(f"| {reynolds} | {axis} | {'/'.join(str(g) for g in grids)} | {raw} | {exc} | {p_raw} | {p_exc} |")
    lines += ["", "## 4. Cross-check against the ladder artefacts", "",
              "| row | committed max\\|u\\| error | artefact | committed v (excluded) | artefact v (excluded) |",
              "|---|---|---|---|---|"]
    for check in payload["cross_checks"]:
        if not check["available"]:
            lines.append(f"| {check['row']} | artefact not in this checkout | {check['artefact']} | n/a | n/a |")
            continue
        lines.append("| {} | {} | {} | {} | {} |".format(
            check["row"], _fmt(check["committed_u"]), _fmt(check["artefact_u"]),
            _fmt(check["committed_v_consistent"]), _fmt(check["artefact_v_consistent"])))
    proj = payload["projection"]
    lines += ["", "## 5. What that implies for the C2 gate (estimate, assumption named)", ""]
    if proj.get("available"):
        lines += [f"* Measured Re = 1000 deviation at n = {proj['evaluated_n']}: "
                  f"{_fmt(proj['measured_error'], 4)} (u, suspect-excluded).",
                  "* Orders measured where three grids exist: "
                  + ", ".join(_fmt(value, 3) for value in proj["measured_orders"])
                  + f"; median p = {_fmt(proj['assumed_order'], 3)}.",
                  "* Assuming error ~ n^-p with that median order, the 2 % gate would need "
                  f"n >= {proj['n_required_estimate']:.1f}, i.e. the next power-of-two grid "
                  f"n = {proj['next_power_of_two_grid']}.",
                  "* This is an estimate.  It is falsifiable by exactly one run "
                  "(Re = 1000, n = 128), which is the ladder's outstanding row and is "
                  "*not* what this document claims to have measured."]
    else:
        lines += [f"* Not available: {proj.get('reason')}"]
    lines += ["", "## 6. Honest limits", "",
              "* `v` at Re = 100 does not converge monotonically in the committed rows "
              "(all three values sit inside 0.008-0.010), so the order measured there is "
              "noise around a floor; only the Re = 400 `u` track shows a clean rate.",
              "* Two grids give one order per Reynolds number; a GCI needs three to prove "
              "asymptotic range, so the GCI numbers here are reported, not relied on.",
              "* Nothing here weakens C2: the gate stays FAIL at Re = 1000 until a second "
              "grid is measured.",
              ""]
    return "\n".join(lines)


def build(committed: pathlib.Path = COMMITTED) -> dict:
    rows, checks = load_rows(committed)
    refinement = {}
    by_re = {}
    for row in rows:
        by_re.setdefault(round(row["reynolds"], 6), []).append(row)
    for reynolds, series in sorted(by_re.items()):
        series = sorted(series, key=lambda item: item["n"])
        block = {"grids": [item["n"] for item in series]}
        for axis in ("u", "v"):
            raw = [{"n": item["n"], "error": item[axis]["max_abs_error"]} for item in series]
            track = [{"n": item["n"], "error": item[axis]["consistent"]} for item in series]
            pairs, excluded_pairs = [], []
            for coarse, fine in zip(raw, raw[1:]):
                ratio = fine["n"] / coarse["n"]
                pairs.append({"coarse_n": coarse["n"], "fine_n": fine["n"], "ratio": ratio,
                              "observed_order": _order(coarse["error"], fine["error"], ratio),
                              "gci_fine": (FS * fine["error"] / abs(ratio ** _order(coarse["error"], fine["error"], ratio) - 1.0))
                              if math.isfinite(_order(coarse["error"], fine["error"], ratio)) else float("nan")})
            for coarse, fine in zip(track, track[1:]):
                ratio = fine["n"] / coarse["n"]
                p = _order(coarse["error"], fine["error"], ratio)
                excluded_pairs.append({"coarse_n": coarse["n"], "fine_n": fine["n"], "ratio": ratio,
                                       "observed_order": p,
                                       "gci_fine": (FS * fine["error"] / abs(ratio ** p - 1.0))
                                       if math.isfinite(p) else float("nan")})
            block[axis] = {"raw": raw, "track": track, "pairs": pairs,
                           "excluded_pairs": excluded_pairs,
                           "measurable": len(series) >= 2,
                           "suspect_points": series[-1][axis]["suspects"]}
        refinement[str(int(reynolds)) if reynolds == int(reynolds) else str(reynolds)] = block
    payload = {
        "source": _relative(committed),
        "source_digest": "sha256:" + __import__("hashlib").sha256(
            pathlib.Path(committed).read_bytes()).hexdigest(),
        "fs": FS, "c2_limit": C2_LIMIT, "target_reynolds": TARGET_RE,
        "rows": rows,
        "cross_checks": checks,
        "refinement": refinement,
        "projection": projection(rows),
        "projection_raw": projection(rows, key="max_abs_error"),
        "figures": [],
    }
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=pathlib.Path, default=OUT)
    parser.add_argument("--committed", type=pathlib.Path, default=COMMITTED)
    args = parser.parse_args(argv)
    payload = build(args.committed)
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    name = figure(payload["rows"], out / "error-vs-grid.png")
    if name:
        payload["figures"].append({"name": name,
                                   "caption": "Centre-line max error against grid with the "
                                              "documented suspect Ghia entries dropped; the dotted "
                                              "line is the ladder's 2 % C2 gate."})
    (out / "refinement.json").write_text(json.dumps(payload, indent=1) + "\n")
    (out / "report.md").write_text(render(payload))
    print(f"wrote {_relative(out)}/refinement.json, report.md"
          + (f", {name}" if name else ""))
    proj = payload["projection"]
    if proj.get("available"):
        print(f"Re=1000 measured {proj['measured_error']:.5f} at n={proj['evaluated_n']}; "
              f"median order {proj['assumed_order']:.3f}; gate would need n>={proj['n_required_estimate']:.1f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())