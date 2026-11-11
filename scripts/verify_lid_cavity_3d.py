"""Verification of the 3-D lid-driven cavity solver against the 2-D Ghia benchmark.

``tensorfvm.lid_cavity_3d`` extrudes the verified 2-D staggered discretisation to
a prism ``n_x x n_y x n_z`` with a **periodic** span.  With a span-invariant
state (``w = 0``) the discrete operators must reduce to the 2-D ones exactly, so
the published Ghia et al. (1982) centreline tables are a legitimate accuracy
anchor for the 3-D code.

This script produces four blocks, all machine-checked by the tests in
``tests/test_native_lid_cavity_3d.py``:

``reduction``
    three-dimensional state vs the 2-D solver after the same fixed number of
    explicit steps -- the difference is a round-off statement, not an error bar.
``ghia``
    span-averaged centreline errors against the Ghia tables at Re = 100 / 400 on
    a ``32 x 32 x 4`` prism.
``spanwise``
    a span-invariant lid keeps ``w`` at exactly zero (discrete symmetry); a lid
    modulated in the span drives a genuinely three-dimensional state while the
    projection keeps ``max|div u|`` at round-off, and the spanwise response
    scales linearly with the modulation amplitude.

Usage::

    python scripts/verify_lid_cavity_3d.py            # full sweep, writes JSON
    python scripts/verify_lid_cavity_3d.py --quick    # Re = 100 only
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
import sys

sys.path.insert(0, str(ROOT / "src"))

from tensorfvm.lid_cavity import LidCavityConfig, LidCavitySolver  # noqa: E402
from tensorfvm.lid_cavity_3d import (  # noqa: E402
    LidCavity3DConfig,
    LidCavity3DSolver,
    score_against_ghia_3d,
)

OUT = ROOT / "src/tensorfvm/verification/artifacts/lid-cavity-3d"


def reduction(n: int = 16, n_z: int = 3, steps: int = 30, dt: float = 0.02) -> dict:
    """Span-invariant prism vs the 2-D solver: the difference must be round-off."""
    two_d = LidCavitySolver(LidCavityConfig(n=n, reynolds=100.0, time_step=dt,
                                            tolerance=1e-30, max_steps=4))
    three_d = LidCavity3DSolver(LidCavity3DConfig(n_x=n, n_z=n_z, reynolds=100.0,
                                                  time_step=dt, tolerance=1e-30,
                                                  max_steps=4))
    for _ in range(steps):
        two_d.step()
        three_d.step()
    return {
        "n": n, "n_z": n_z, "steps": steps, "dt": dt,
        "max_abs_delta_u": float((three_d.u[:, :, 0] - two_d.u).abs().max()),
        "max_abs_delta_v": float((three_d.v[:, :, 0] - two_d.v).abs().max()),
        "max_abs_delta_p": float((three_d.p[:, :, 0] - two_d.p).abs().max()),
        "spanwise_max": three_d.spanwise_uniformity(),
    }


def ghia(n_x: int, reynolds: float, n_z: int) -> dict:
    """Converged prism scored against the Ghia tables (span-averaged centrelines)."""
    started = time.time()
    solver = LidCavity3DSolver(LidCavity3DConfig(
        n_x=n_x, n_z=n_z, reynolds=reynolds, tolerance=1e-7,
        max_steps=20_000, check_every=250))
    info = solver.run()
    scored = score_against_ghia_3d(solver)
    y, u_prof = solver.centreline_u()
    x, v_prof = solver.centreline_v()
    return {
        "reynolds": reynolds, "n_x": n_x, "n_z": n_z,
        "converged": bool(info["converged"]), "steps": int(info["steps"]),
        "residual": float(info["residual"]),
        "max_divergence": float(info["max_divergence"]),
        "spanwise_max": float(info["spanwise_max"]),
        "wall_seconds": time.time() - started,
        "u": {k: v for k, v in scored["u"].items() if k != "profiles"},
        "v": {k: v for k, v in scored["v"].items() if k != "profiles"},
        "profiles": {
            "y": y.tolist(), "u_computed": u_prof.tolist(),
            "x": x.tolist(), "v_computed": v_prof.tolist(),
            "u_reference": [p[0:2] for p in
                            __import__("tensorfvm.lid_cavity_reference", fromlist=["table"]).table(reynolds, "u")],
            "v_reference": [p[0:2] for p in
                            __import__("tensorfvm.lid_cavity_reference", fromlist=["table"]).table(reynolds, "v")],
        },
    }


def spanwise(n_x: int = 16, n_z: int = 4, steps: int = 100, dt: float = 0.002) -> dict:
    """Symmetry preservation, genuinely-3-D response, and its linearity."""
    quiet = LidCavity3DSolver(LidCavity3DConfig(n_x=n_x, n_z=n_z, reynolds=100.0,
                                                time_step=dt, max_steps=4))
    for _ in range(steps):
        quiet.step()

    def modulated(amplitude: float) -> dict:
        solver = LidCavity3DSolver(LidCavity3DConfig(
            n_x=n_x, n_z=n_z, reynolds=100.0, time_step=dt, max_steps=4,
            lid_spanwise_amplitude=amplitude))
        for _ in range(steps):
            solver.step()
        return {"amplitude": amplitude,
                "spanwise_max": solver.spanwise_uniformity(),
                "max_divergence": float(solver.divergence().abs().max())}

    full = modulated(0.02)
    half = modulated(0.01)
    return {
        "n_x": n_x, "n_z": n_z, "steps": steps, "dt": dt,
        "span_invariant": {
            "spanwise_max": quiet.spanwise_uniformity(),
            "max_divergence": float(quiet.divergence().abs().max()),
        },
        "modulated_full": full,
        "modulated_half": half,
        "linearity_ratio": (full["spanwise_max"] / half["spanwise_max"]
                            if half["spanwise_max"] else float("nan")),
    }


def main() -> dict:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="Re = 100 only")
    parser.add_argument("--out", default=str(OUT))
    args = parser.parse_args()

    payload: dict = {"reduction": reduction(), "spanwise": spanwise()}
    reynolds_list = [100.0] if args.quick else [100.0, 400.0]
    payload["ghia"] = [ghia(32, re, 4) for re in reynolds_list]

    for block in payload["ghia"]:
        print(f"Re={block['reynolds']:.0f} n={block['n_x']}^2 x {block['n_z']}: "
              f"converged={block['converged']} steps={block['steps']} "
              f"div={block['max_divergence']:.2e} w={block['spanwise_max']:.2e}")
        print(f"   u max {block['u']['max_abs_error']:.4f} rms {block['u']['rms_error']:.4f}"
              f" | v max {block['v']['max_abs_error']:.4f} rms {block['v']['rms_error']:.4f}"
              f" | {block['wall_seconds']:.1f}s")
    red = payload["reduction"]
    print(f"reduction: Δu={red['max_abs_delta_u']:.2e} Δv={red['max_abs_delta_v']:.2e} "
          f"Δp={red['max_abs_delta_p']:.2e} w={red['spanwise_max']:.2e}")
    sp = payload["spanwise"]
    print(f"spanwise: quiet w={sp['span_invariant']['spanwise_max']:.2e} "
          f"div={sp['span_invariant']['max_divergence']:.2e} | "
          f"modulated w={sp['modulated_full']['spanwise_max']:.3e} "
          f"div={sp['modulated_full']['max_divergence']:.2e} "
          f"| linearity ratio {sp['linearity_ratio']:.3f}")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "verification.json").write_text(json.dumps(payload, indent=2))
    print(f"artifact: {out / 'verification.json'}")
    return payload


if __name__ == "__main__":
    main()