"""Drive the sudden-expansion momentum-closure study and write its artifacts.

Four stages, each a self-contained JSON ledger in ``--out``:

``sweep``    the eight-point area-ratio sweep ``A1/A2 = 0.1 ... 0.9`` -- Borda-Carnot
             anchor, every closure term, the measured loss and both residuals;
``exit``     the same case closed with the three downstream closures
             (``outlet`` / ``developed`` / ``convective``) -- outlet sensitivity;
``mesh``     grid-refinement study at one area ratio (the discretisation part of
             the deviation from Borda-Carnot);
``reynolds`` Reynolds-number trend of the closure terms at one area ratio.

Every number that the report quotes is written here; nothing is transcribed by
hand.  Runs are independent, so ``--jobs`` fans them out over processes (one
OpenMP thread each) instead of threading inside torch.

Usage (from a checkout root, ~50 min at ``--jobs 8``)::

    python scripts/run_expansion_momentum_closure.py --stage all --jobs 8

The mesh for each area ratio is chosen so that the inlet opening is an *integral*
number of cell rows (``cells_per_downstream_height = m * AR``): a partially open
inlet row conserves flux but not momentum, which shows up as a spurious inlet
term in the closure (see ``k_inlet`` in the ledger and the report).
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

from tensorfvm.internal_flow.expansion_closure import (  # noqa: E402
    EXIT_CLOSURES,
    closure_terms,
)
from tensorfvm.internal_flow.sudden_expansion import (  # noqa: E402
    SuddenExpansionCase,
    SuddenExpansionConfig,
)

#: ``A1 / A2`` sampling; ``AR = A2 / A1`` and the mesh are derived from it below.
AREA_RATIOS = (0.1, 0.2, 0.25, 0.4, 0.5, 0.75, 0.8, 0.9)
#: cells per downstream height of the sweep.  Chosen so that *both* step edges land
#: on cell faces (``n * A1/A2`` and ``n * (1 - A1/A2) / 2`` integers): only then is
#: the inlet momentum-flux coefficient exactly one and the spurious inlet term of
#: the closure absent.  ``0.25 -> 20`` is kept as the fractional-row negative control.
SWEEP_CELLS = {0.1: 40, 0.2: 20, 0.25: 24, 0.4: 20, 0.5: 20, 0.75: 24, 0.8: 20, 0.9: 20}
#: refinement ladder at ``A1/A2 = 0.25``: 8, 16 and 32 keep the opening integral,
#: 20 and 24 do not (``beta_in = 0.9`` and ``0.875``) and quantify that defect.
MESH_CELLS = (8, 16, 20, 24, 32)
REYNOLDS_NUMBERS = (50.0, 100.0, 200.0, 500.0)
REFERENCE_REYNOLDS = 200.0
LENGTH = 15.0
TOLERANCE = 1e-7


def inlet_rows(ratio: float, cells: int) -> float:
    """Opening height in cell rows on a ``cells``-per-height mesh."""
    return ratio * cells


def integral_inlet(ratio: float, cells: int) -> bool:
    """True when both step edges land on cell faces (``beta_in == 1``)."""
    opening = ratio * cells
    half_step = (1.0 - ratio) * cells / 2.0
    return abs(opening - round(opening)) < 1e-9 and abs(half_step - round(half_step)) < 1e-9


def mesh_for(ratio: float) -> int:
    """Cells per downstream height of the published sweep at this area ratio."""
    return SWEEP_CELLS[ratio]


def _run(job: dict) -> dict:
    torch.set_num_threads(1)
    config = SuddenExpansionConfig(
        area_ratio=1.0 / job["area_fraction"],
        reynolds=job["reynolds"],
        cells_per_downstream_height=job["cells"],
        length_in_downstream_heights=job["length"],
        right=job["right"],
        tolerance=job["tolerance"],
        max_steps=job["max_steps"],
    )
    started = time.time()
    case = SuddenExpansionCase(config)
    ledger = case.run()
    terms = closure_terms(case)
    payload = terms.as_dict()
    if job.get("legacy"):
        legacy = case.diagnostics()
        payload["legacy_k_extrapolated"] = legacy["k_extrapolated"]
        payload["legacy_k_raw"] = legacy["k_raw"]
        payload["legacy_k_corrected"] = legacy["k_corrected"]
        payload["legacy_cp_recovery_raw"] = legacy["cp_recovery_raw"]
        payload["legacy_cp_recovery_momentum"] = legacy["cp_recovery_momentum_balance"]
    payload.update(
        stage=job["stage"],
        area_fraction=job["area_fraction"],
        reynolds=job["reynolds"],
        cells=job["cells"],
        length=job["length"],
        right=job["right"],
        inlet_rows=inlet_rows(job["area_fraction"], job["cells"]),
        inlet_rows_integral=integral_inlet(job["area_fraction"], job["cells"]),
        steps=ledger["steps"],
        converged=ledger["converged"],
        steady_residual=ledger["residual"],
        run_seconds=time.time() - started,
        reattachment_bottom=case.reattachment("bottom"),
        reattachment_top=case.reattachment("top"),
        target_wall_units=None,
    )
    return payload


def jobs_for(stage: str) -> list[dict]:
    base = dict(reynolds=REFERENCE_REYNOLDS, length=LENGTH, right="outlet",
                tolerance=TOLERANCE, max_steps=250_000, stage=stage)
    if stage == "sweep":
        return [dict(base, area_fraction=ratio, cells=mesh_for(ratio)) for ratio in AREA_RATIOS]
    if stage == "exit":
        # two domain lengths: at 15D the outflow plane still truncates a developing
        # profile, at 30D the three closures agree because there is nothing left to
        # impose -- the pair separates "outlet sensitivity" from "the outflow
        # station is not downstream of the flow".
        return [dict(base, area_fraction=0.25, cells=24, right=name, length=length)
                for length in (15.0, 30.0) for name in EXIT_CLOSURES]
    if stage == "mesh":
        return [dict(base, area_fraction=0.25, cells=n, length=12.0) for n in MESH_CELLS]
    if stage == "crosscheck":
        # one case, two conventions: the closure's friction-free step-local loss and
        # the published linear-extrapolation loss of ``SuddenExpansionCase.diagnostics``
        return [dict(base, area_fraction=0.25, cells=24, legacy=True)]
    if stage == "reynolds":
        return [dict(base, area_fraction=0.25, cells=24, reynolds=re)
                for re in REYNOLDS_NUMBERS]
    raise ValueError(f"unknown stage {stage!r}")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", default="all",
                        choices=("sweep", "exit", "mesh", "reynolds", "crosscheck", "all"))
    parser.add_argument("--jobs", type=int, default=max(1, min(8, os.cpu_count() or 1)))
    parser.add_argument("--out", default=str(REPO / "docs" / "expansion-momentum-closure" / "artifacts"))
    args = parser.parse_args(argv)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    stages = (("sweep", "exit", "mesh", "reynolds", "crosscheck")
              if args.stage == "all" else (args.stage,))
    summary = {"stages": {}, "reynolds_reference": REFERENCE_REYNOLDS,
               "length_in_downstream_heights": LENGTH, "tolerance": TOLERANCE,
               "area_fractions": list(AREA_RATIOS)}
    for stage in stages:
        jobs = jobs_for(stage)
        with mp.Pool(min(args.jobs, len(jobs))) as pool:
            results = pool.map(_run, jobs)
        results.sort(key=lambda row: (row["area_fraction"], row["reynolds"], row["cells"],
                                      row["length"], row["right"]))
        path = out / f"{stage}.json"
        path.write_text(json.dumps(results, indent=1) + "\n")
        summary["stages"][stage] = str(path.relative_to(REPO))
        for row in results:
            print(f"{stage:8s} A1/A2={row['area_fraction']:.3f} AR={row['area_ratio']:7.4f} "
                  f"n={row['cells']:3d} Re={row['reynolds']:7.1f} {row['right']:11s} "
                  f"K_BC={row['k_borda_carnot']:.4f} K={row['k_measure']:.4f} "
                  f"close={row['error_of_closure'] * 100:+.3f}% "
                  f"BC_err={row['error_vs_borda_carnot'] * 100:+.2f}% "
                  f"steps={row['steps']} conv={row['converged']} {row['run_seconds']:.0f}s",
                  flush=True)
    (out / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())