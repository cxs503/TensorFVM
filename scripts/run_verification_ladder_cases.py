"""Produce the numerical artefacts consumed by :mod:`tensorfvm.verification.ladder`.

The ladder itself is an *auditor*: it reads measurements and applies thresholds.
This script is the part that produces the measurements with production solvers,
so the two can be re-run independently (a report must never be its own evidence).

Levels and the artefacts they write (all under
``src/tensorfvm/verification/artifacts/verification_ladder/``):

``cavity``
    ``lid_cavity_Re{100,400,1000}_n{32,64,128}.json`` -- the production
    :class:`tensorfvm.lid_cavity.LidCavitySolver` marched to a steady state and
    scored against the Ghia, Ghia & Shin (1982) tables that ship in
    :mod:`tensorfvm.lid_cavity_reference`.
``tgv``
    ``tgv_conservation.json`` -- inviscid and viscous Taylor--Green runs of the
    production periodic MAC solver (:mod:`tensorfvm.verification.periodic_tgv`).
``all``
    both of the above (the default).

Level 1a (1-D analytic), Level 3 (GCI), Level 4 (panel / Thwaites) and the two
WITHHELD levels need no artefact: they are recomputed inside the module, which
is what makes them cheap enough to run on every audit.

Usage::

    PYTHONPATH=src python scripts/run_verification_ladder_cases.py --level all
    PYTHONPATH=src python scripts/run_verification_ladder_cases.py --level cavity --reynolds 1000
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

_SRC = pathlib.Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:                 # runnable from a checkout
    sys.path.insert(0, str(_SRC))

from tensorfvm.verification import ladder      # noqa: E402

#: Reynolds number -> grid sequence.  The pattern matches :data:`ladder.CAVITY_GRIDS`
#: and adds the coarse Re = 1000 grid so the refinement can be inspected.
CAVITY_SEQUENCE = {100: (32, 64), 400: (64, 128), 1000: (64, 128)}


def run_cavity(reynolds: float, cells, *, max_steps: int,
               out: pathlib.Path) -> list[pathlib.Path]:
    """March the lid-driven cavity to steady state and store the Ghia score."""
    from tensorfvm.lid_cavity import (LidCavityConfig, LidCavitySolver,
                                      score_against_ghia)

    written: list[pathlib.Path] = []
    for n in cells:
        started = time.perf_counter()
        solver = LidCavitySolver(LidCavityConfig(n=int(n), reynolds=float(reynolds),
                                                max_steps=int(max_steps)))
        result = solver.run()
        score = score_against_ghia(solver)
        payload = {
            "reynolds": float(reynolds), "n": int(n),
            "steps": result["steps"], "converged": bool(result["converged"]),
            "residual": float(result["residual"]),
            "max_divergence": float(result["max_divergence"]),
            "wall_seconds": time.perf_counter() - started,
            "solver": "tensorfvm.lid_cavity.LidCavitySolver",
            "reference": "Ghia, Ghia & Shin (1982), J. Comput. Phys. 48, 387",
            "u": score["u"], "v": score["v"],
            "vortex_centre": score["vortex_centre"],
        }
        path = out / f"lid_cavity_Re{int(reynolds)}_n{int(n)}.json"
        path.write_text(json.dumps(payload, indent=1) + "\n")
        written.append(path)
        print(f"  {path.name}: converged={payload['converged']} "
              f"steps={payload['steps']} u_err={score['u']['max_abs_error']:.4f} "
              f"v_err={score['v']['max_abs_error']:.4f} "
              f"({payload['wall_seconds']:.0f} s)", flush=True)
    return written


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--level", choices=("cavity", "tgv", "all"), default="all")
    parser.add_argument("--reynolds", type=float, nargs="+",
                        default=[100.0, 400.0, 1000.0])
    parser.add_argument("--cells", type=int, nargs="+", default=None,
                        help="override the grid sequence for every Reynolds number")
    parser.add_argument("--max-steps", type=int, default=900_000)
    parser.add_argument("--tgv-resolutions", type=int, nargs="+", default=[16, 32])
    parser.add_argument("--out", type=pathlib.Path, default=ladder.ARTIFACTS)
    args = parser.parse_args(argv)
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if args.level in ("cavity", "all"):
        for reynolds in args.reynolds:
            cells = tuple(args.cells) if args.cells else CAVITY_SEQUENCE.get(
                int(reynolds), (64,))
            print(f"lid-driven cavity Re = {reynolds:g}, grids {cells}", flush=True)
            run_cavity(reynolds, cells, max_steps=args.max_steps, out=out)
    if args.level in ("tgv", "all"):
        print("Taylor-Green conservation audit", flush=True)
        payload = ladder.tgv_conservation(resolutions=args.tgv_resolutions,
                                         refresh=True, cache=out / "tgv_conservation.json")
        for key, record in payload["records"].items():
            print(f"  {key}: inviscid drift {record['inviscid_energy_drift']:.3e}, "
                  f"viscous integral error "
                  f"{record['viscous_integral_energy_error']:.3e}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())