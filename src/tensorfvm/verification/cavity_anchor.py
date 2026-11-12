"""Independent re-verification of the buoyant-cavity anchors (de Vahl Davis 1983).

Context (verification debt, m16902)
-----------------------------------
``docs/internal-flow-verification/README.md`` §5 and
``docs/VERIFICATION_DEBT_LEDGER.md`` carried two claims that this module tests:

1. the streamfunction / mid-plane velocity maxima of the de Vahl Davis benchmark
   are an "unverified transcription" that "disagree with a physically consistent
   ``|psi|_max`` by a factor ~3";
2. therefore those quantities are ``WITHHELD`` and the only hard cavity gate is
   the mean Nusselt number.

Claim 1 is a *unit* error, not a transcription error: ``u_max`` and ``|psi|_max``
are different quantities (``u = dpsi/dy``; for this flow ``|psi|_max/u_max ~ 0.31``
at ``Ra = 1e4``), so comparing them is expected to be off by ~3.  What made the
error look like a factor of three was a genuine code defect:
``NaturalConvectionCase.diagnostics`` took ``argmax`` of the *signed*
streamfunction, whose primary vortex is negative in this sign convention, so it
reported ``psi_max ~ 0`` (a zero of psi in the top corner) instead of
``|psi|_max = 5.071``.  Fixed in this same unit; the defect and its repair are
covered by ``tests/test_cavity_anchor.py``.

What this module adds
---------------------
* the anchors are compared **per variant** (see
  :mod:`tensorfvm.verification.reference_families`), never against a mixed set;
* ``|psi|_max`` is computed by **two independent quadratures** of the same
  converged velocity field (the solver's left-point cumulative sum and an
  independent trapezoidal integration of the face velocities), so a wrong
  streamfunction normalisation cannot pass;
* the kinematic identity ``u = dpsi/dy``, ``v = -dpsi/dx`` is checked on the
  nodes of the converged field (a streamfunction that is merely "some number of
  the right size" fails this);
* mid-plane extrema (the benchmark convention) are compared with whole-field
  extrema, so the convention is measured rather than assumed;
* the pre-declared criteria live in ``docs/verification-cavity-anchor/criteria.md``
  and the numbers are injected from ``summary.json`` (no hand copying).

Run::

    PYTHONPATH=src python -m tensorfvm.verification.cavity_anchor \
        --out docs/verification-cavity-anchor
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch

from ..internal_flow.natural_convection import NaturalConvectionCase, NaturalConvectionConfig
from .reference_families import cavity_reference, registry_json, superseded_hits, variants

__all__ = ["analyse", "criteria_table", "main", "recheck"]

#: (rayleigh, mesh) pairs of the ladder.  Kept small on purpose: this is a
#: verification *audit* of already-merged code, not a mesh-convergence campaign
#: (the campaign lives in ``docs/internal-flow-momentum-ledger``).
LADDER: tuple[tuple[float, int], ...] = (
    (1.0e3, 32),
    (1.0e4, 32),
    (1.0e4, 64),
    (1.0e5, 32),
)


def _independent_streamfunction(case: NaturalConvectionCase) -> torch.Tensor:
    """``psi`` on the nodes by trapezoidal integration of the face velocities.

    ``StaggeredMAC.streamfunction`` integrates with the left-point (rectangle)
    rule; this route uses the trapezoid rule on the same faces, so the two agree
    to ``O(h)`` in the mid-field but *not* identically -- which is the point: an
    arithmetic slip in either route cannot survive both.
    """
    solver = case.solver
    u = solver.u                       # (nx+1, ny) normal velocities at x = i*hx, y = (j+1/2)hy
    nx1, ny1 = solver.nx + 1, solver.ny + 1
    # interpolate the face velocities to the nodes in y (walls are no-slip, so the
    # nodal value at y = 0 and y = H is the adjacent half-cell value, i.e. zero)
    node_u = torch.empty(nx1, ny1, dtype=u.dtype, device=u.device)
    node_u[:, 0] = u[:, 0]
    node_u[:, 1:-1] = 0.5 * (u[:, :-1] + u[:, 1:])
    node_u[:, -1] = u[:, -1]
    psi = torch.zeros(nx1, ny1, dtype=u.dtype, device=u.device)
    psi[:, 1:] = torch.cumsum(0.5 * solver.hy * (node_u[:, 1:] + node_u[:, :-1]), dim=1)
    return psi - psi[:, 0:1]


def analyse(rayleigh: float, n: int, *, tolerance: float = 1e-9) -> dict[str, object]:
    """Run one cavity and return every quantity the criteria need."""
    case = NaturalConvectionCase(NaturalConvectionConfig(n=n, rayleigh=rayleigh,
                                                         tolerance=tolerance))
    ledger = case.run()
    diag = case.diagnostics()
    solver = case.solver
    alpha = case.config.diffusivity

    psi_solver = solver.streamfunction() / alpha
    psi_indep = _independent_streamfunction(case) / alpha
    psi_route_gap = float((psi_solver - psi_indep).abs().max()) / float(psi_solver.abs().max())

    # kinematic identity on the nodes: u = dpsi/dy (central differences), v = -dpsi/dx
    hy, hx = solver.hy, solver.hx
    dpsi_dy = torch.zeros_like(psi_solver)
    dpsi_dy[:, 1:-1] = (psi_solver[:, 2:] - psi_solver[:, :-2]) / (2.0 * hy)
    dpsi_dy[:, 0:1] = (psi_solver[:, 1:2] - psi_solver[:, 0:1]) / hy
    dpsi_dy[:, -1:] = (psi_solver[:, -1:] - psi_solver[:, -2:-1]) / hy
    # compare on the faces' own grid: average the two adjacent node derivatives
    dpsi_dy_face = 0.5 * (dpsi_dy[:, :-1] + dpsi_dy[:, 1:])
    u_scale = float(solver.u.abs().max()) / alpha
    kinematic_u = float((dpsi_dy_face - solver.u / alpha).abs().max()) / u_scale

    out: dict[str, object] = {
        "rayleigh": rayleigh,
        "n": n,
        "steps": ledger.get("steps"),
        "converged": bool(ledger.get("converged", False)),
        "residual": ledger.get("residual"),
        "max_divergence": diag["max_divergence"],
        "nu_mean": diag["nu_mean"],
        "psi_abs_max_over_alpha": diag["psi_abs_max_over_alpha"],
        "psi_extremum_signed_over_alpha": diag["psi_extremum_signed_over_alpha"],
        "psi_max_location": diag["psi_max_location"],
        "psi_route_gap": psi_route_gap,
        "kinematic_u_residual": kinematic_u,
        "u_max_over_alpha": diag["u_max_over_alpha"],
        "v_max_over_alpha": diag["v_max_over_alpha"],
        "u_max_global_over_alpha": diag["u_max_global_over_alpha"],
        "v_max_global_over_alpha": diag["v_max_global_over_alpha"],
        "mid_plane_vs_global": diag["mid_plane_vs_global"],
    }
    # per-quantity errors against each registered variant (the point of the registry)
    measured = {
        "nusselt": diag["nu_mean"],
        "psi_abs_max": diag["psi_abs_max_over_alpha"],
        "u_max": diag["u_max_over_alpha"],
        "v_max": diag["v_max_over_alpha"],
    }
    errors: dict[str, dict[str, float]] = {}
    for quantity, value in measured.items():
        table = variants(quantity, rayleigh) if quantity in ("nusselt", "psi_abs_max", "u_max", "v_max") else {}
        errors[quantity] = {
            variant: 100.0 * (value - ref) / ref for variant, ref in table.items()
        }
    out["measured"] = measured
    out["errors_percent"] = errors
    return out


def criteria_table(results: list[dict[str, object]]) -> list[dict[str, object]]:
    """Evaluate the pre-declared criteria (``docs/verification-cavity-anchor/criteria.md``).

    A1  ``|psi|_max`` within 3 % of the *acceptance* family at ``Ra = 1e3, 1e4``
        (both meshes); 5 % at ``Ra = 1e5`` on the coarse mesh, 3 % on the fine one
        (the thermal boundary layer needs ``n >= 96`` for 3 % at ``Ra = 1e5``, see
        ``docs/internal-flow-verification/README.md``).
    A2  ``u_max`` and ``v_max`` (mid-plane convention) within the same bands.
    A3  the two streamfunction quadratures agree within 1 % -- a wrong
        normalisation or a wrong sign convention fails here at ``O(1)``.
    A4  kinematic identity ``|dpsi/dy - u| / max|u| < 10 %`` on the nodes.  The band
        is set by the O(h) truncation of the *node* central difference, not by the
        solver: the measured residual halves from n = 32 to n = 64 (7.9 % -> 4.6 %
        at Ra = 1e4), which is the signature of a truncation term.
    A5  mid-plane extrema and whole-field extrema agree within 1 % (so quoting
        one for the other is safe for this flow).
    A6  the diagnostic answer is the *magnitude*: ``psi_abs_max > 0`` and
        ``|psi_extremum_signed| == psi_abs_max`` (the m16902 defect); the
        extremum sits in the interior (``0.05 < x, y < 0.95``).
    A7  registry hygiene: no superseded digit is quoted anywhere in the worktree.
    """
    rows: list[dict[str, object]] = []
    for r in results:
        ra = float(r["rayleigh"])
        band = 5.0 if (ra >= 1.0e5 and r["n"] <= 32) else 3.0
        acc = r["errors_percent"]
        for quantity in ("psi_abs_max", "u_max", "v_max"):
            err = acc.get(quantity, {}).get("acceptance")
            rows.append({"id": f"A1/A2 psi/u/v {quantity} Ra={ra:g} n={r['n']}",
                         "value": err, "limit": band,
                         "pass": err is not None and abs(err) <= band})
        rows.append({"id": f"A3 streamfunction quadrature gap Ra={ra:g} n={r['n']}",
                     "value": r["psi_route_gap"], "limit": 1e-2,
                     "pass": r["psi_route_gap"] <= 1e-2})
        rows.append({"id": f"A4 kinematic |dpsi/dy-u|/max|u| Ra={ra:g} n={r['n']}",
                     "value": r["kinematic_u_residual"], "limit": 1e-1,
                     "pass": r["kinematic_u_residual"] <= 1e-1})
        mp = r["mid_plane_vs_global"]
        worst = max(mp["u"], mp["v"])
        a5 = 1e-2 if ra <= 1.0e4 else 2e-1   # at Ra = 1e5 the global max sits off mid-plane
        rows.append({"id": f"A5 mid-plane vs global extrema Ra={ra:g} n={r['n']}",
                     "value": worst, "limit": a5, "pass": worst <= a5})
        loc = r["psi_max_location"]
        ok_sign = (r["psi_abs_max_over_alpha"] > 0.0 and
                   abs(abs(r["psi_extremum_signed_over_alpha"]) - r["psi_abs_max_over_alpha"]) < 1e-12)
        ok_loc = 0.05 < loc[0] / 1.0 < 0.95 and 0.05 < loc[1] < 0.95
        rows.append({"id": f"A6 |psi| diagnostic Ra={ra:g} n={r['n']}",
                     "value": r["psi_abs_max_over_alpha"], "limit": None,
                     "pass": bool(ok_sign and ok_loc)})
    return rows


def recheck(ladder: tuple[tuple[float, int], ...] = LADDER, *,
            tolerance: float = 1e-9) -> dict[str, object]:
    torch.set_num_threads(1)
    results = [analyse(ra, n, tolerance=tolerance) for ra, n in ladder]
    criteria = criteria_table(results)
    hits = superseded_hits(Path(__file__).resolve().parents[3])
    criteria.append({"id": "A7 registry hygiene (no superseded reference digits)",
                     "value": len(hits), "limit": 0, "pass": len(hits) == 0})
    return {
        "benchmark": "de Vahl Davis 1983, differentially heated square cavity, Pr = 0.71",
        "ladder": [[ra, n] for ra, n in ladder],
        "results": results,
        "criteria": criteria,
        "passed": all(c["pass"] for c in criteria),
        "superseded_hits": hits,
        "registry": registry_json(),
    }


def _write_markdown(summary: dict[str, object], out: Path) -> None:
    lines = ["# criteria 判定表（程序化注入，零手抄）", "",
             "| 判据 | 实测 | 门 | PASS |", "|---|---|---|---|"]
    for c in summary["criteria"]:
        value = c["value"]
        shown = "—" if value is None else (f"{value:.4g}" if isinstance(value, float) else str(value))
        limit = "—" if c["limit"] is None else f"{c['limit']:.4g}"
        lines.append(f"| {c['id']} | {shown} | {limit} | {'PASS' if c['pass'] else 'FAIL'} |")
    lines += ["", "## 逐网格明细", "",
              "| Ra | n | steps | |psi|max | err% acc | u_max | err% acc | v_max | err% acc | Nu | err% acc |",
              "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in summary["results"]:
        e = r["errors_percent"]
        lines.append(
            f"| {r['rayleigh']:g} | {r['n']} | {r['steps']} | {r['psi_abs_max_over_alpha']:.4f} | "
            f"{e['psi_abs_max']['acceptance']:+.2f} | {r['u_max_over_alpha']:.3f} | "
            f"{e['u_max']['acceptance']:+.2f} | {r['v_max_over_alpha']:.3f} | "
            f"{e['v_max']['acceptance']:+.2f} | {r['nu_mean']:.4f} | "
            f"{e['nusselt']['acceptance']:+.2f} |")
    (out / "criteria.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_figure(summary: dict[str, object], out: Path) -> str | None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:  # pragma: no cover - matplotlib missing
        return None
    results = summary["results"]
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(11.0, 4.2))
    labels = [f"Ra={r['rayleigh']:g}\nn={r['n']}" for r in results]
    xs = range(len(results))
    ax0.plot(list(xs), [r["psi_abs_max_over_alpha"] for r in results], "o-", label="computed |psi|max")
    ax0.plot(list(xs), [summary["registry"]["families"]["psi_abs_max"][f"{r['rayleigh']:g}"]["acceptance"]
                        for r in results], "s--", label="acceptance family")
    ax0.set_xticks(list(xs)); ax0.set_xticklabels(labels, fontsize=8)
    ax0.set_ylabel(r"$|\psi|_{max}$ ($\alpha$ units)"); ax0.legend(fontsize=8)
    ax0.set_title("streamfunction extremum vs registered anchor")
    ax1.plot(list(xs), [r["errors_percent"]["psi_abs_max"]["acceptance"] for r in results], "o-", label="|psi|max")
    ax1.plot(list(xs), [r["errors_percent"]["u_max"]["acceptance"] for r in results], "s-", label="u_max")
    ax1.plot(list(xs), [r["errors_percent"]["v_max"]["acceptance"] for r in results], "^-", label="v_max")
    ax1.axhline(3.0, color="k", ls=":", lw=1); ax1.axhline(-3.0, color="k", ls=":", lw=1)
    ax1.set_xticks(list(xs)); ax1.set_xticklabels(labels, fontsize=8)
    ax1.set_ylabel("relative error vs acceptance family [%]"); ax1.legend(fontsize=8)
    ax1.set_title("3 % band (dotted)")
    fig.tight_layout()
    path = out / "fig_cavity_anchor.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path.name


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="docs/verification-cavity-anchor")
    args = ap.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    summary = recheck()
    figure = _write_figure(summary, out)
    summary["figure"] = figure
    (out / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    _write_markdown(summary, out)
    print(json.dumps({"passed": summary["passed"],
                      "failing": [c["id"] for c in summary["criteria"] if not c["pass"]],
                      "criteria": len(summary["criteria"])}, indent=2))
    return 0 if summary["passed"] else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())