"""GCI (Grid Convergence Index) analysis for CASE-WOODWARD-COLELLA.

Uses the ASME V&V-20 Richardson extrapolation framework to quantify
discretisation uncertainty on the three-grid ladder (nx=200, 400, 800).

The Woodward-Colella double blast wave is a shock-dominated problem where
the formal order of accuracy (p=2 for MUSCL-Hancock) is not achieved in
the asymptotic range.  The GCI analysis therefore serves two purposes:

1. Quantify the discretisation uncertainty band for each metric.
2. Diagnose whether the solution is in the asymptotic range (it is not,
   due to the strong shocks, but the GCI band still provides a conservative
   uncertainty estimate).

All numbers are computed from the committed ladder artifacts; no solver
runs are performed here.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict

import numpy as np

from tensorfvm.verification.richardson import richardson_gci

# ----------------------------------------------------------------------
# Committed ladder data (from plateau_convergence.json and metrics JSONs)
# ----------------------------------------------------------------------
LADDER_DIR = Path(__file__).resolve().parent / "data"

# Peak density values (from metrics JSONs)
PEAK_DENSITY = {
    200: 5.00678626370563,
    400: 5.649813015225584,
    800: 6.189783163404389,
}

# L1 relative errors (as fractions, from metrics JSONs)
L1_ERROR = {
    200: 0.14825794539824458,
    400: 0.07711573517196672,
    800: 0.029569679382539165,
}

# Window mass integral values (from plateau_convergence.json)
WINDOW_MASS = {
    200: 0.9338469667575857,
    400: 0.9347607573176968,
    800: 0.9351565123446093,
}

# Reference values
REFERENCE = {
    "peak_density": 6.2960765,
    "window_mass": 0.92215100156725,
}

SPACINGS = (1 / 800, 1 / 400, 1 / 200)  # fine, medium, coarse
THEORY_ORDER = 2.0  # MUSCL-Hancock formal order


@dataclass
class GCIResult:
    """GCI analysis result for one metric."""
    metric: str
    fine_value: float
    medium_value: float
    coarse_value: float
    observed_order: float
    asymptotic: bool
    extrapolated: float
    gci_percent: float
    gci_band_low: float
    gci_band_high: float
    convergence: str
    ref_value: float | None
    ref_rel_error_percent: float | None


def analyse_gci() -> Dict[str, GCIResult]:
    """Run GCI analysis on all committed metrics."""
    results = {}

    # Peak density
    r = richardson_gci(
        PEAK_DENSITY[800], PEAK_DENSITY[400], PEAK_DENSITY[200],
        SPACINGS, theory_order=THEORY_ORDER
    )
    results["peak_density"] = GCIResult(
        metric="peak_density",
        fine_value=PEAK_DENSITY[800],
        medium_value=PEAK_DENSITY[400],
        coarse_value=PEAK_DENSITY[200],
        observed_order=r.order,
        asymptotic=r.asymptotic,
        extrapolated=r.extrapolated,
        gci_percent=r.gci * 100,
        gci_band_low=r.band[0],
        gci_band_high=r.band[1],
        convergence=r.convergence,
        ref_value=REFERENCE["peak_density"],
        ref_rel_error_percent=abs(PEAK_DENSITY[800] - REFERENCE["peak_density"])
        / REFERENCE["peak_density"] * 100,
    )

    # L1 error (note: this is an error metric, so the "value" is the error itself)
    r = richardson_gci(
        L1_ERROR[800], L1_ERROR[400], L1_ERROR[200],
        SPACINGS, theory_order=THEORY_ORDER
    )
    results["l1_error"] = GCIResult(
        metric="l1_error",
        fine_value=L1_ERROR[800],
        medium_value=L1_ERROR[400],
        coarse_value=L1_ERROR[200],
        observed_order=r.order,
        asymptotic=r.asymptotic,
        extrapolated=r.extrapolated,
        gci_percent=r.gci * 100,
        gci_band_low=r.band[0],
        gci_band_high=r.band[1],
        convergence=r.convergence,
        ref_value=None,
        ref_rel_error_percent=None,
    )

    # Window mass
    r = richardson_gci(
        WINDOW_MASS[800], WINDOW_MASS[400], WINDOW_MASS[200],
        SPACINGS, theory_order=THEORY_ORDER
    )
    results["window_mass"] = GCIResult(
        metric="window_mass",
        fine_value=WINDOW_MASS[800],
        medium_value=WINDOW_MASS[400],
        coarse_value=WINDOW_MASS[200],
        observed_order=r.order,
        asymptotic=r.asymptotic,
        extrapolated=r.extrapolated,
        gci_percent=r.gci * 100,
        gci_band_low=r.band[0],
        gci_band_high=r.band[1],
        convergence=r.convergence,
        ref_value=REFERENCE["window_mass"],
        ref_rel_error_percent=abs(WINDOW_MASS[800] - REFERENCE["window_mass"])
        / REFERENCE["window_mass"] * 100,
    )

    return results


def summary_table(results: Dict[str, GCIResult]) -> str:
    """Format a markdown summary table."""
    lines = [
        "# GCI Analysis Summary (CASE-WOODWARD-COLELLA)",
        "",
        "Three-grid Richardson extrapolation (nx=200, 400, 800) with ASME V&V-20 GCI.",
        f"Theory order: p={THEORY_ORDER} (MUSCL-Hancock).",
        "",
        "| Metric | Fine value | Observed order | Asymptotic? | GCI % | Extrapolated | Ref rel error % |",
        "|---|---|---|---|---|---|---|",
    ]
    for key in ("peak_density", "l1_error", "window_mass"):
        r = results[key]
        ref_err = f"{r.ref_rel_error_percent:.2f}" if r.ref_rel_error_percent is not None else "—"
        lines.append(
            f"| {r.metric} | {r.fine_value:.6f} | {r.observed_order:.3f} | "
            f"{'Yes' if r.asymptotic else 'No'} | {r.gci_percent:.2f} | "
            f"{r.extrapolated:.6f} | {ref_err} |"
        )
    lines.append("")
    lines.append("## Interpretation")
    lines.append("")
    lines.append("- **peak_density**: Not in asymptotic range (observed order 0.25 << 2.0).")
    lines.append("  The strong shock at the collision peak limits the effective order.")
    lines.append("  GCI band is wide (57%), reflecting the shock-dominated nature.")
    lines.append("- **l1_error**: Not in asymptotic range (observed order 0.58 << 2.0).")
    lines.append("  The L1 norm is dominated by the shock regions where the formal order breaks down.")
    lines.append("- **window_mass**: Not in asymptotic range (observed order 1.21 < 2.0),")
    lines.append("  but GCI is very small (0.04%), indicating the integral quantity is already")
    lines.append("  converged to within the reference solution's own discretisation error.")
    lines.append("")
    return "\n".join(lines)


def main():
    results = analyse_gci()
    print(summary_table(results))

    # Save JSON
    out = {k: asdict(v) for k, v in results.items()}
    out_path = LADDER_DIR / "gci_analysis.json"
    out_path.write_text(json.dumps(out, indent=2) + "\n")
    print(f"\nGCI analysis saved to {out_path}")


if __name__ == "__main__":
    main()
