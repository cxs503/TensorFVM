"""Primary reference data for the Re = 700 sister case of CASE-FVM-PERIODIC-HILLS.

Round 32 (selfstart, TensorFVM).  The anchor's acceptance names two metrics --
reattachment length ``x_r/H`` and the mean velocity profile at ``y/H = 0.5`` --
against Breuer et al. 2009 (doi:10.1016/j.compfluid.2008.05.002).  Rounds 20--31
could only grade the first metric at the 2-D-admissible Reynolds number
``Re = 700`` (laminar, so a 2-D mean is a defensible model of the flow), because
the repository carried no Re = 700 profile data: the round-10 retrieval had
fetched only the Re = 2800 set (``ercoftac_ufr330``).

This module closes that gap with the **primary** data: the ten DNS profile
files for *case 1 = Re = 700, code LESOCC* distributed by the reference's own
first author on the ERCOFTAC Knowledge Base Wiki (``UFR 3-30 Test Case``,
section "Curvilinear grids", "Case 1  DNS (LESOCC) Re = 700"), fetched
verbatim in this round into ``tests/data/periodic_hill/ercoftac_ufr330_re700``
(see the ``PROVENANCE.md`` there).  The same ten streamwise stations as the
Re = 2800 set, ``x/H = 0.05, 0.5, 1, 2, 3, 4, 5, 6, 7, 8`` in file order, same
seven columns ``y/h, U/U_b, V/U_b, uu/U_b^2, vv/U_b^2, uv/U_b^2, k/U_b^2``.

The reattachment reference is *not* re-tabulated here: it is re-exported from
:mod:`tensorfvm.verification.periodic_hill_lower_re` (``KB_REATTACHMENT_BY_RE``),
the single source for the five KB values, so the two modules cannot drift.

Geometry note (measured from the files themselves): at ``x/H = 0.05, 0.5, 1``
the profile starts at ``y/H >= 0.45`` (inside the hill body) and at
``x/H = 8`` it starts at ``y/H = 0.453`` (the hill surface rising again), so
``y/H = 0.5`` is in the fluid only at ``x/H = 2 .. 7`` -- the same exclusion
the Re = 2800 module makes via :func:`station_covers_y05`.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np

from .periodic_hill_lower_re import (
    KB_SOURCE,
    reattachment_reference,
)

__all__ = [
    "DATA_DIR", "STATIONS", "X_R_OVER_H", "X_R_TOLERANCE", "REFERENCE",
    "load_profile", "profile_at", "y05_table", "station_covers_y05",
    "relative_band_ok",
]

DATA_DIR = (Path(__file__).resolve().parents[3]
            / "tests" / "data" / "periodic_hill" / "ercoftac_ufr330_re700")

#: streamwise stations of the 10 distributed profiles, in file order
STATIONS = (0.05, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0)

#: mean reattachment length at Re = 700 (ERCOFTAC KB, UFR 3-30 Evaluation)
X_R_OVER_H = float(reattachment_reference(700.0)["x_r_over_H"])
#: the ledger's acceptance tolerance
X_R_TOLERANCE = 0.10

REFERENCE = {
    "case": "ERCOFTAC UFR 3-30, periodic hills, Re_b = 700 (DNS, LESOCC)",
    "x_r_over_H": X_R_OVER_H,
    "station_metric": "time-mean u/U_b at y/H = 0.5",
    "source": (KB_SOURCE + "; profile files: 'UFR 3-30 Test Case', case 1 "
                   "(DNS, LESOCC, Re = 700), author Michael Breuer, HSU Hamburg"),
    "data_dir": str(DATA_DIR),
    "provenance": "primary data files published by the reference's first author",
}


def _station_path(station: float) -> Path:
    """File for one streamwise station (exact match against :data:`STATIONS`)."""
    index = int(np.argmin(np.abs(np.array(STATIONS) - float(station))))
    if abs(STATIONS[index] - float(station)) > 1.0e-12:
        raise KeyError(f"station x/H = {station!r} is not one of {STATIONS}")
    return DATA_DIR / f"UFR3-30_C_700_data_MB-{index + 1:03d}.dat"


@lru_cache(maxsize=None)
def load_profile(station: float) -> dict:
    """Load one distributed DNS profile (all seven columns, ascending ``y``)."""
    path = _station_path(station)
    if not path.is_file():
        raise FileNotFoundError(
            f"reference profile missing: {path} (see {DATA_DIR}/PROVENANCE.md)")
    raw = np.loadtxt(path, comments="#")
    if raw.ndim != 2 or raw.shape[1] != 7:
        raise ValueError(f"unexpected column count in {path.name}: {raw.shape}")
    order = np.argsort(raw[:, 0])
    raw = raw[order]
    if np.any(np.diff(raw[:, 0]) <= 0.0):
        raise ValueError(f"non-monotone y column in {path.name}")
    keys = ("y_over_h", "u_over_ub", "v_over_ub", "uu_over_ub2",
            "vv_over_ub2", "uv_over_ub2", "k_over_ub2")
    out = {key: raw[:, i].copy() for i, key in enumerate(keys)}
    out["x_over_h"] = float(station)
    out["source"] = path.name
    return out


def profile_at(x_over_h: float, y_over_h: float = 0.5,
               field: str = "u_over_ub") -> float:
    """Interpolate the reference at one (x/H, y/H) point (linear in ``y``)."""
    prof = load_profile(x_over_h)
    y = prof["y_over_h"]
    values = prof[field]
    y0, y1 = float(y[0]), float(y[-1])
    if not (y0 <= y_over_h <= y1):
        raise ValueError(f"y/H = {y_over_h} outside [{y0}, {y1}] for x/H = {x_over_h}")
    return float(np.interp(y_over_h, y, values))


def station_covers_y05(station: float) -> bool:
    """Is the ``y/H = 0.5`` line inside the fluid at this station?"""
    prof = load_profile(station)
    return bool(prof["y_over_h"][0] <= 0.5 <= prof["y_over_h"][-1])


def y05_table(stations=None) -> dict:
    """``{x/H: u/U_b(y/H = 0.5)}`` for the stations where that line is in fluid."""
    stations = STATIONS if stations is None else stations
    return {float(x): profile_at(x, 0.5) for x in stations
            if station_covers_y05(x)}


def relative_band_ok(computed: float, x_over_h: float, y_over_h: float = 0.5,
                    tolerance: float = X_R_TOLERANCE) -> dict:
    """Round-6 error convention (relative where |ref| >= 0.1, absolute below)."""
    ref = profile_at(x_over_h, y_over_h)
    computed = float(computed)
    if not np.isfinite(computed):
        return {"reference": ref, "computed": computed, "band": "undefined",
                "error": float("inf"), "pass": False}
    if abs(ref) >= 0.1:
        error = abs(computed - ref) / abs(ref)
        band = "relative"
    else:
        error = abs(computed - ref)
        band = "absolute"
    return {"reference": ref, "computed": computed, "band": band,
            "error": float(error), "pass": bool(error <= tolerance)}
