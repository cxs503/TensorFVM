"""Grid-refinement study driver for the immersed-solid projection arms.

Fixed ratio ``r = 2`` refined meshes (``H/40, H/80, H/160``), two projection arms
(``masked`` / ``legacy``) and a discrete mass ledger.  All numbers written here
come from a single programmatic path: the driver builds the mask, runs the
solver, extracts the wall-shear zero crossing and writes JSON.  Nothing is typed
in by hand.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

import torch

from .geometry import (CartesianGrid, backward_facing_step, expansion_ratio_realised,
                       symmetric_expansion)
from .solver import ImmersedConfig, ImmersedFlow

# Armaly et al. (1983) backward-facing step at Re_Dh = 800, 2-D limit.
# Same provenance as ``tensorfvm.inc_bench.reference.backward_facing_step``:
# Kim & Moin (1985) / Erturk (2008) reproduce L_r/S = 6.0 +/- 0.2.
REFERENCES = {
    "bfs": {
        "reattachment_over_step": {
            "value": 6.0, "band": 0.2, "grade": "REAL",
            "source": "2-D limit of Armaly et al. (1983) Re_Dh = 800 "
                      "(Kim & Moin 1985; Erturk 2008): L_r/S = 6.0 +/- 0.2",
        },
        "expansion_ratio": {"value": 2.0, "band": 0.0, "grade": "REAL",
                            "source": "geometry definition (ER = H_out / H_in)"},
    },
    "expansion": {
        "loss_coefficient": {
            "value": 0.25, "band": 0.0, "grade": "REAL",
            "source": "Borda-Carnot K = (1 - A_in/A_out)^2 = 0.25 for area ratio 2 "
                      "(inviscid/turbulent jet limit; used as an order-of-magnitude "
                      "reference only, laminar viscous flow adds excess loss)",
        },
        "reattachment": {"value": float("nan"), "band": float("nan"),
                         "grade": "WITHHELD",
                         "source": "no citable 2-D laminar symmetric-expansion "
                                   "reattachment correlation available offline; the "
                                   "criterion used instead is cross-arm agreement"},
    },
}

# Huggett-style dimensionless numbers are not used here: the cases are already
# non-dimensional (H = 1, U_mean = 1).


def build_case(case: str, cells_per_h: int) -> CartesianGrid:
    if case == "bfs":
        return backward_facing_step(cells_per_h)
    if case == "expansion":
        return symmetric_expansion(cells_per_h)
    raise ValueError(f"unknown case {case!r}")


def inlet_bounds(case: str, grid: CartesianGrid) -> tuple[float, float]:
    if case == "bfs":
        # inlet channel sits above the step: S <= y <= S + H_in
        return (1.0, 2.0)
    # symmetric expansion: the upstream channel is centred
    return (0.5 * (grid.height - 1.0), 0.5 * (grid.height + 1.0))


def wall_shear(grid: CartesianGrid, flow: ImmersedFlow, wall: str) -> torch.Tensor:
    return flow.wall_shear(wall)  # type: ignore[arg-type]


def reattachment(grid: CartesianGrid, tau: torch.Tensor, x_start: float = 0.0,
                 reverse: bool = False) -> dict[str, object]:
    """Zero crossings of the wall shear downstream of ``x_start``.

    ``reverse`` flips the sign convention for the upper wall (whose outward normal
    points the other way, so a recirculation bubble has the opposite sign).
    """
    x = grid.x_faces
    t = -tau if reverse else tau
    mask = x > x_start
    xs = x[mask]
    ts = t[mask]
    crossings: list[float] = []
    for k in range(len(ts) - 1):
        a, b = float(ts[k]), float(ts[k + 1])
        if a == 0.0:
            continue
        if a * b < 0.0:
            xa, xb = float(xs[k]), float(xs[k + 1])
            crossings.append(xa + (xb - xa) * (-a) / (b - a))
    # The reattachment point of interest is the end of the *longest* reversed-flow
    # interval (the main separation bubble).  A second, shorter reversed interval
    # can appear at a zero-gradient outflow -- that is an outflow artefact, not the
    # bubble, and taking the first/last sign change would report it by mistake.
    primary = float("nan")
    best_span = 0.0
    k = 0
    n = len(ts)
    while k < n:
        if float(ts[k]) >= 0.0:
            k += 1
            continue
        start = k
        while k < n and float(ts[k]) < 0.0:
            k += 1
        stop = k - 1
        span = float(xs[stop] - xs[max(start - 1, 0)])
        if span > best_span and stop + 1 < n:
            best_span = span
            xa, xb = float(xs[stop]), float(xs[stop + 1])
            ta, tb = float(ts[stop]), float(ts[stop + 1])
            primary = xa + (xb - xa) * (-ta) / (tb - ta)
    return {
        "crossings": crossings,
        "primary_reattachment": primary,
        "bubble_span": best_span,
        "tau_min": float(ts.min()) if ts.numel() else float("nan"),
        "tau_max": float(ts.max()) if ts.numel() else float("nan"),
    }


def run_case(case: str, cells_per_h: int, projection: str, t_end: float, *,
             reynolds_dh: float = 800.0, kappa: float = 1.0e4,
             max_steps: int = 4_000_000, probe_every: float | None = None,
             legacy_relaxation: bool = True, out_dir: str | None = None,
             threads: int = 8) -> dict[str, object]:
    """Run one (case, mesh, arm) and return a JSON-serialisable result record."""
    torch.set_num_threads(threads)
    grid = build_case(case, cells_per_h)
    # Re = U_mean * D_h / nu with D_h = 2 H_in = 2 for both cases
    nu = 2.0 / reynolds_dh
    cfg = ImmersedConfig(
        grid=grid, viscosity=nu, projection=projection, kappa=kappa,
        inlet_bounds=inlet_bounds(case, grid), reference_length=1.0,
        t_end=t_end, max_steps=max_steps, legacy_relaxation=legacy_relaxation,
        name=f"{case}-H{cells_per_h}-{projection}",
    )
    t0 = time.perf_counter()
    flow = ImmersedFlow(cfg)
    t_build = time.perf_counter() - t0
    t0 = time.perf_counter()
    ledger = flow.run(t_end=t_end, max_steps=max_steps)
    wall = time.perf_counter() - t0
    tau_bottom = wall_shear(grid, flow, "bottom")
    tau_top = wall_shear(grid, flow, "top")
    bottom = reattachment(grid, tau_bottom, x_start=0.0)
    top = reattachment(grid, tau_top, x_start=0.0, reverse=True)
    uc = 0.5 * (flow.u[:, :-1] + flow.u[:, 1:])
    vc = 0.5 * (flow.v[:-1, :] + flow.v[1:, :])
    fluid = grid.fluid
    result: dict[str, object] = {
        "case": case,
        "cells_per_h": cells_per_h,
        "arm": projection,
        "reynolds_dh": reynolds_dh,
        "kappa": kappa,
        "t_end": t_end,
        "grid": grid.summary(),
        "expansion_ratio_realised": expansion_ratio_realised(grid),
        "dt_final": flow.timestep(),
        "steps": ledger.steps,
        "time": ledger.time,
        "build_seconds": t_build,
        "wall_seconds": wall,
        "step_seconds": wall / max(ledger.steps, 1),
        "ledger": {
            "div_max_fluid": ledger.div_max_fluid,
            "div_max_solid": ledger.div_max_solid,
            "solid_flux_abs_max": ledger.solid_flux_abs_max,
            "solid_flux_last": ledger.solid_flux_last,
            "drift": ledger.drift,
            "drift_relative": ledger.drift_relative,
            "throughput": ledger.throughput,
            "mass_imbalance_max": ledger.mass_imbalance_max,
            "solid_speed_max": ledger.solid_speed_max,
            "history": ledger.history,
        },
        "metrics": {
            "reattachment_bottom": bottom,
            "reattachment_top": top,
            "u_max": float((uc * fluid).max()),
            "v_abs_max": float((vc * fluid).abs().max()),
            "pressure_drop": float((flow.p[:, 0] * fluid[:, 0]).mean()
                                   - (flow.p[:, -1] * fluid[:, -1]).mean()),
        },
        "reference": REFERENCES[case],
        "generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    dominant = bottom["primary_reattachment"]
    if case == "expansion":
        dominant = bottom["primary_reattachment"]
    result["reattachment_over_h"] = dominant
    if out_dir is not None:
        path = Path(out_dir)
        path.mkdir(parents=True, exist_ok=True)
        tag = f"{case}_H{cells_per_h}_{projection}_Re{reynolds_dh:g}_t{t_end:g}"
        (path / f"{tag}.json").write_text(json.dumps(result, indent=1, default=float))
        torch.save({"u": flow.u, "v": flow.v, "p": flow.p,
                    "x_faces": grid.x_faces, "y_faces": grid.y_faces,
                    "solid": grid.solid}, path / f"{tag}_fields.pt")
    return result


# ---------------------------------------------------------------- convergence
def richardson(fine: float, medium: float, coarse: float, ratio: float = 2.0
               ) -> dict[str, float]:
    """Observed order and GCI for a fixed-refinement-ratio triple."""
    eps21 = medium - fine
    eps32 = coarse - medium
    out = {"value_fine": fine, "value_medium": medium, "value_coarse": coarse}
    if eps21 == 0.0 or eps32 == 0.0 or (eps32 / eps21) <= 0.0:
        out.update({"observed_order": float("nan"), "gci_fine_pct": float("nan"),
                    "extrapolated": float("nan")})
        return out
    r = ratio
    p = abs(math.log(abs(eps32 / eps21)) / math.log(r))
    extrap = fine + (fine - medium) / (r ** p - 1.0)
    gci = 1.25 * abs((fine - medium) / fine) / (r ** p - 1.0) * 100.0
    out.update({"observed_order": p, "extrapolated": extrap, "gci_fine_pct": gci,
                "ratio_eps32_eps21": eps32 / eps21})
    return out


def summarise(results: list[dict[str, object]]) -> dict[str, object]:
    """Group the per-run records by (case, arm) and compute the grid study."""
    groups: dict[tuple[str, str], list[dict[str, object]]] = {}
    for rec in results:
        groups.setdefault((str(rec["case"]), str(rec["arm"])), []).append(rec)
    summary: dict[str, object] = {"groups": {}, "references": REFERENCES}
    for (case, arm), recs in groups.items():
        recs.sort(key=lambda r: int(r["cells_per_h"]))
        values = [float(r["reattachment_over_h"]) for r in recs
                  if r.get("reattachment_over_h") is not None]
        entry: dict[str, object] = {
            "case": case, "arm": arm,
            "meshes": [int(r["cells_per_h"]) for r in recs],
            "reattachment_over_h": values,
            "drift": [float(r["ledger"]["drift"]) for r in recs],  # type: ignore[index]
            "div_max_fluid": [float(r["ledger"]["div_max_fluid"]) for r in recs],  # type: ignore[index]
            "div_max_solid": [float(r["ledger"]["div_max_solid"]) for r in recs],  # type: ignore[index]
            "solid_speed_max": [float(r["ledger"]["solid_speed_max"]) for r in recs],  # type: ignore[index]
            "steps": [int(r["steps"]) for r in recs],
            "time": [float(r["time"]) for r in recs],
            "step_seconds": [float(r["step_seconds"]) for r in recs],
        }
        if len(values) >= 3:
            entry["richardson"] = richardson(values[0], values[1], values[2])
        summary["groups"][f"{case}|{arm}"] = entry
    return summary


def main(argv: list[str] | None = None) -> int:
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", default="bfs", choices=["bfs", "expansion"])
    parser.add_argument("--cells-per-h", type=int, default=40)
    parser.add_argument("--arm", default="masked", choices=["masked", "legacy"])
    parser.add_argument("--t-end", type=float, default=40.0)
    parser.add_argument("--reynolds-dh", type=float, default=800.0)
    parser.add_argument("--kappa", type=float, default=1.0e4)
    parser.add_argument("--max-steps", type=int, default=4_000_000)
    parser.add_argument("--no-legacy-relaxation", action="store_true")
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--json-out", default=None)
    args = parser.parse_args(argv)
    rec = run_case(args.case, args.cells_per_h, args.arm, args.t_end,
                   reynolds_dh=args.reynolds_dh, kappa=args.kappa,
                   max_steps=args.max_steps,
                   legacy_relaxation=not args.no_legacy_relaxation,
                   out_dir=args.out_dir, threads=args.threads)
    text = json.dumps(rec, indent=1, default=float)
    if args.json_out:
        Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json_out).write_text(text)
    print(text)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
