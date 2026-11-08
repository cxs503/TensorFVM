"""Independent literature reference band for the smooth flat-plate drag problem.

``benchmark_flat_plate`` gates the Spalart--Allmaras flat-plate case on a single
correlation, ``C_f = 0.074 Re^-1/5`` (Schlichting), with a 20 % window.  Two
questions are never asked there:

1. is that single number inside the *published* spread of equally reputable
   fully-turbulent flat-plate correlations, and
2. is the gate window narrower than that spread?  If not, the gate cannot
   discriminate accuracy: it accepts every correlation-level answer and also
   rejects nothing that is merely within the literature disagreement.

This module answers both with closed-form correlations only.  No solver output
enters the reference values, so the numbers are independent of the code they
audit (see ``docs/verification-flat-plate-qualification/``).

Correlations used (all for a smooth plate, turbulent from the leading edge,
zero pressure gradient, incompressible):

* Schlichting, *Boundary-Layer Theory* 0.074 Re^-1/5 (the current gate)
* Prandtl--Schlichting 0.455 / (log10 Re)^2.58
* ITTC-57 friction line 0.075 / (log10 Re - 2)^2
* White, *Viscous Fluid Flow* 0.523 / ln^2(0.06 Re)

Rationale for the band: the four laws differ by how they extrapolate the
logarithmic skin-friction law; their mutual disagreement is the irreducible
uncertainty of the reference itself at a given Reynolds number.
"""

import math

# Reference name -> callable(Re) -> average skin-friction coefficient.
CORRELATIONS = {
    "schlichting_0.074": lambda re: 0.074 / re ** 0.2,
    "prandtl_schlichting": lambda re: 0.455 / math.log10(re) ** 2.58,
    "ittc57": lambda re: 0.075 / (math.log10(re) - 2.0) ** 2,
    "white": lambda re: 0.523 / math.log(0.06 * re) ** 2,
}

# Correlations are fitted to fully turbulent data; below ~1e5 the plate
# boundary layer would normally be partly laminar and no law is valid.
MIN_REYNOLDS = 5e4


def correlation_values(reynolds: float) -> dict:
    """Return every correlation's average skin-friction coefficient."""
    if not (isinstance(reynolds, float) or isinstance(reynolds, int)):
        raise TypeError("Reynolds number must be a real scalar")
    reynolds = float(reynolds)
    if not math.isfinite(reynolds) or reynolds < MIN_REYNOLDS:
        raise ValueError(
            f"fully turbulent flat-plate correlations require Re >= {MIN_REYNOLDS:g}"
        )
    return {name: float(law(reynolds)) for name, law in CORRELATIONS.items()}


def reference_band(reynolds: float) -> dict:
    """Summarise the literature band: min, max, mean and relative spread.

    ``spread`` is the full width of the band relative to its mean, i.e. the
    fractional disagreement that any single-correlation gate must be measured
    against.  ``against_gate`` compares the current gate window to that spread.
    """
    values = correlation_values(reynolds)
    low, high = min(values.values()), max(values.values())
    mean = sum(values.values()) / len(values)
    spread = (high - low) / mean
    return {
        "reynolds": float(reynolds),
        "values": values,
        "low": low,
        "high": high,
        "mean": mean,
        "spread": spread,
        "spread_percent": 100.0 * spread,
    }


def gate_is_informative(reynolds: float, error_limit: float) -> bool:
    """True only if the gate window is tighter than the literature spread.

    A gate that is at least as wide as the disagreement between accepted
    reference laws cannot certify accuracy; it can only catch gross errors.
    Both the one-sided window ``error_limit`` and the two-sided band width
    ``2 * error_limit`` are considered redundant once they exceed the spread.
    """
    if not math.isfinite(error_limit) or error_limit <= 0:
        raise ValueError("error_limit must be finite and positive")
    return error_limit < reference_band(reynolds)["spread"]


def band_membership(value: float, reynolds: float) -> bool:
    """Whether a measured coefficient lies inside the literature band."""
    band = reference_band(reynolds)
    if not math.isfinite(value):
        raise ValueError("measured coefficient must be finite")
    return band["low"] <= float(value) <= band["high"]

def window_to_band_ratio(reynolds: float, error_limit: float) -> float:
    """Full two-sided gate window ``2 * error_limit`` over the literature spread.

    ``gate_is_informative`` historically compared the *one-sided* limit with the
    *full* band width, which understates the window by a factor of two; this
    helper states the ratio with an explicit convention so that reports can
    quote it without ambiguity: a gate whose full window exceeds the spread
    cannot discriminate between accepted reference laws (ratio > 1).
    """
    if not math.isfinite(error_limit) or error_limit <= 0:
        raise ValueError("error_limit must be finite and positive")
    return 2.0 * error_limit / reference_band(reynolds)["spread"]


def band_aware_verdict(value: float, reynolds: float) -> dict:
    """Judge a measured mean ``C_f`` against the whole literature band.

    A single-law relative error cannot decide accuracy when the reference laws
    themselves disagree by more than the claim being tested.  The verdict here
    is therefore band-relative and reports, in machine-readable form:

    * ``error_per_law`` - relative error against every accepted correlation,
      so that the spread of verdicts is visible instead of one chosen number,
    * ``error_vs_band_mean`` and ``offset_in_band_widths`` - where the value
      sits in the band, normalised by the band width,
    * ``verdict`` - ``"in_band"`` or ``"outside_band"``.

    ``"in_band"`` means the measurement cannot be resolved from the reference
    uncertainty; it is not a claim that the solver is accurate to any
    particular tolerance.
    """
    band = reference_band(reynolds)
    value = float(value)
    if not math.isfinite(value):
        raise ValueError("measured coefficient must be finite")
    if value <= 0.0:
        raise ValueError("measured coefficient must be positive")
    errors = {name: (value - law) / law for name, law in band["values"].items()}
    in_band = band["low"] <= value <= band["high"]
    return {
        "value": value,
        "reynolds": band["reynolds"],
        "in_band": in_band,
        "verdict": "in_band" if in_band else "outside_band",
        "error_per_law": errors,
        "error_vs_band_mean": (value - band["mean"]) / band["mean"],
        "offset_in_band_widths": (value - band["mean"]) / band["spread"],
        "band_low": band["low"],
        "band_high": band["high"],
        "band_mean": band["mean"],
        "band_spread": band["spread"],
    }
