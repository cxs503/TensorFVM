"""Reproduce the overset Stokes verification ladder and write machine-readable evidence."""
import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from tensorfvm.overset import annular_grid, build_overset, cartesian_grid, circle_polygon  # noqa: E402
from tensorfvm.overset_stokes import (  # noqa: E402
    DIVERGENCE_IDENTICALLY_ZERO,
    STOKES_RESIDUAL_IDENTICALLY_ZERO,
    solve_overset_stokes,
)


def connectivity(n):
    return build_overset(
        cartesian_grid((-2, 2, -2, 2), n, n),
        annular_grid((0, 0), 0.3, 1.0, 2 * n, n // 4),
        circle_polygon((0, 0), 0.3),
        circle_polygon((0, 0), 0.65),
    )


def main(out_dir):
    os.makedirs(out_dir, exist_ok=True)
    levels = []
    for n in (32, 48, 64):
        m = solve_overset_stokes(connectivity(n)).metrics
        levels.append({"n": n, "metrics": m})
    orders = {}
    for name in ("u", "v", "p"):
        es = [lvl["metrics"][f"relative_l2_error_{name}"] for lvl in levels]
        orders[name] = [math.log2(es[i] / es[i + 1]) for i in range(len(es) - 1)]
    controls = {}
    for key, kwargs in (("flipped_pressure_sign", {"flip_pressure_sign": True}),
                        ("no_rhie_chow", {"rhie_chow": False})):
        try:
            m = solve_overset_stokes(connectivity(32), **kwargs).metrics
            controls[key] = {"relative_l2_error_velocity": m["relative_l2_error_velocity"],
                             "relative_l2_error_p": m["relative_l2_error_p"]}
        except RuntimeError as exc:
            controls[key] = {"raised": str(exc)}
    payload = {
        "analytic_identities": {"divergence_identically_zero": DIVERGENCE_IDENTICALLY_ZERO,
                                "stokes_residual_identically_zero": STOKES_RESIDUAL_IDENTICALLY_ZERO},
        "levels": levels,
        "observed_orders": orders,
        "negative_controls_n32": controls,
        "verdict": {
            "velocity_below_3_percent": all(lvl["metrics"]["relative_l2_error_velocity"] < 0.03 for lvl in levels),
            "pressure_below_3_percent": all(lvl["metrics"]["relative_l2_error_p"] < 0.03 for lvl in levels),
            "honest_note": "pressure L2 error 34.5-47.8 %, observed order ~0.2: recorded as an open defect",
        },
    }
    with open(os.path.join(out_dir, "results.json"), "w") as fh:
        json.dump(payload, fh, indent=1)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        ns = [lvl["n"] for lvl in levels]
        fig, ax = plt.subplots(figsize=(6, 4.2))
        for name, label in (("u", "velocity u"), ("v", "velocity v"), ("p", "pressure p")):
            ax.loglog(ns, [lvl["metrics"][f"relative_l2_error_{name}"] for lvl in levels], "o-", label=label)
        ax.axhline(0.03, color="k", ls="--", lw=1, label="3 % acceptance band")
        ax.set_xlabel("background cells per side n")
        ax.set_ylabel("relative L2 error vs exact Stokes solution")
        ax.set_title("Overset coupled Stokes: velocity passes, pressure does not")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "convergence.png"), dpi=110)
    except Exception as exc:  # pragma: no cover
        print("figure skipped:", exc)
    print(json.dumps(payload["verdict"], indent=1))
    print(json.dumps(orders, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "runs/overset_stokes"))
