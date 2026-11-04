"""Ghia, Ghia & Shin (1982) lid-driven-cavity benchmark tables.

Reference
---------
U. Ghia, K.N. Ghia, C.T. Shin, "High-Re solutions for incompressible flow using
the Navier-Stokes equations and a multigrid method", *Journal of Computational
Physics* 48 (1982) 387-411.  Tables I (u along the vertical centreline x = 0.5)
and II (v along the horizontal centreline y = 0.5).

The tabulated values are reproduced here as plain tuples so that the numerical
experiment in :mod:`tensorfvm.lid_cavity` can be scored against the published
benchmark without carrying a data file.  Coordinates are normalised by the
cavity side length ``L`` and velocities by the lid speed ``U``; the cavity is
``[0, L] x [0, L]`` with the moving wall at ``y = L``.

Cross-checks used when transcribing the tables (they must all agree, otherwise
the transcription is wrong and the verification is meaningless):

* Re = 100:  u_min = -0.21090 at y = 0.4531,  v_max = +0.17527 at x = 0.2344
* Re = 400:  u_min = -0.32726 at y = 0.2813,  v_max = +0.30203 at x = 0.2266
* Re = 1000: u_min = -0.38289 at y = 0.1719,  v_max = +0.37095 at x = 0.1563

Primary vortex centres reported in the same paper (used as a coarse, independent
check on the solution field rather than only on the two centrelines):

* Re = 100:  (0.6172, 0.7344)
* Re = 400:  (0.5547, 0.6055)
* Re = 1000: (0.5313, 0.5625)
"""

from __future__ import annotations

import math
from typing import Mapping

#: u(y) on the vertical centreline x = 0.5, (y/L, u/U)
U_CENTRELINE: Mapping[float, tuple[tuple[float, float], ...]] = {
    100.0: (
        (1.0000, 1.00000),
        (0.9766, 0.84123),
        (0.9688, 0.78871),
        (0.9609, 0.73722),
        (0.9531, 0.68717),
        (0.8516, 0.23151),
        (0.7344, 0.00332),
        (0.6172, -0.13641),
        (0.5000, -0.20581),
        (0.4531, -0.21090),
        (0.2813, -0.15662),
        (0.1719, -0.10150),
        (0.1016, -0.06434),
        (0.0703, -0.04775),
        (0.0625, -0.04192),
        (0.0547, -0.03717),
        (0.0000, 0.00000),
    ),
    400.0: (
        (1.0000, 1.00000),
        (0.9766, 0.75837),
        (0.9688, 0.68439),
        (0.9609, 0.61756),
        (0.9531, 0.55892),
        (0.8516, 0.29093),
        (0.7344, 0.16256),
        (0.6172, 0.02135),
        (0.5000, -0.11477),
        (0.4531, -0.17119),
        (0.2813, -0.32726),
        (0.1719, -0.24299),
        (0.1016, -0.14612),
        (0.0703, -0.10338),
        (0.0625, -0.09266),
        (0.0547, -0.08186),
        (0.0000, 0.00000),
    ),
    1000.0: (
        (1.0000, 1.00000),
        (0.9766, 0.65928),
        (0.9688, 0.57492),
        (0.9609, 0.51117),
        (0.9531, 0.46604),
        (0.8516, 0.33304),
        (0.7344, 0.18719),
        (0.6172, 0.05702),
        (0.5000, -0.06080),
        (0.4531, -0.10648),
        (0.2813, -0.27805),
        (0.1719, -0.38289),
        (0.1016, -0.29730),
        (0.0703, -0.22220),
        (0.0625, -0.20196),
        (0.0547, -0.18109),
        (0.0000, 0.00000),
    ),
}

#: v(x) on the horizontal centreline y = 0.5, (x/L, v/U)
V_CENTRELINE: Mapping[float, tuple[tuple[float, float], ...]] = {
    100.0: (
        (1.0000, 0.00000),
        (0.9688, -0.05906),
        (0.9609, -0.07391),
        (0.9531, -0.08864),
        (0.9453, -0.10313),
        (0.9063, -0.16914),
        (0.8594, -0.22445),
        (0.8047, -0.24533),
        (0.5000, 0.05454),
        (0.2344, 0.17527),
        (0.2266, 0.17507),
        (0.1563, 0.16077),
        (0.0938, 0.12317),
        (0.0781, 0.10890),
        (0.0703, 0.10091),
        (0.0625, 0.09233),
        (0.0000, 0.00000),
    ),
    400.0: (
        (1.0000, 0.00000),
        (0.9688, -0.12146),
        (0.9609, -0.15663),
        (0.9531, -0.19254),
        (0.9453, -0.22847),
        (0.9063, -0.23827),
        (0.8594, -0.44993),
        (0.8047, -0.38598),
        (0.5000, 0.05186),
        (0.2344, 0.30174),
        (0.2266, 0.30203),
        (0.1563, 0.28124),
        (0.0938, 0.22965),
        (0.0781, 0.20920),
        (0.0703, 0.19713),
        (0.0625, 0.18360),
        (0.0000, 0.00000),
    ),
    1000.0: (
        (1.0000, 0.00000),
        (0.9688, -0.21388),
        (0.9609, -0.27669),
        (0.9531, -0.33714),
        (0.9453, -0.39188),
        (0.9063, -0.51550),
        (0.8594, -0.42665),
        (0.8047, -0.31966),
        (0.5000, 0.02526),
        (0.2344, 0.32235),
        (0.2266, 0.33075),
        (0.1563, 0.37095),
        (0.0938, 0.32627),
        (0.0781, 0.30353),
        (0.0703, 0.29012),
        (0.0625, 0.27485),
        (0.0000, 0.00000),
    ),
}

#: Primary-vortex centre from Ghia et al. (1982), normalised coordinates.
VORTEX_CENTRE: Mapping[float, tuple[float, float]] = {
    100.0: (0.6172, 0.7344),
    400.0: (0.5547, 0.6055),
    1000.0: (0.5313, 0.5625),
}

#: Provenance of the transcribed tables, and what :func:`self_check` does *not* cover.
#:
#: The tables below are re-keys of circulating digital copies of the published
#: centreline data, **not** a fresh transcription from the printed paper.  The
#: self-check validates the published extrema, the coordinate ordering and
#: finiteness only, so a single garbled interior entry passes it silently - see
#: :data:`SUSPECT_ENTRIES` for the one such entry found so far.
TABLE_PROVENANCE: Mapping[str, str] = {
    "source": "U. Ghia, K. N. Ghia, C. T. Shin, J. Comput. Phys. 48 (1982) 387-411, "
              "Table I (u, x/L = 0.5) and Table II (v, y/L = 0.5)",
    "transcription": "re-keyed from a digital copy of the published tables; the printed "
                     "article is not held in this repository, so no entry has been "
                     "checked against the typeset original",
    "self_check_coverage": "extrema + coordinate ordering + finiteness only; interior "
                           "entries are otherwise unverified",
    "consistency_audit": "scripts/audit_ghia_reference_consistency.py -> "
                         "docs/verification-ghia-reference-consistency/results.json",
}

#: Table entries that the three-grid consistency audit rejects as suspect
#: digitisation errors, keyed ``(reynolds, 'u'|'v')`` then by coordinate.
#:
#: Scoring keeps the value *as digitised* (so every committed evidence bundle stays
#: reproducible) but excludes the entry from the consistent-point statistics and
#: reports it explicitly, instead of letting one garbled number decide a pass/fail.
SUSPECT_ENTRIES: Mapping[tuple[float, str], Mapping[float, Mapping[str, object]]] = {
    (400.0, "v"): {
        0.9063: {
            "as_digitized": -0.23827,
            "finest_grid_value": -0.38758,
            "richardson_limit": -0.38989,
            "resolution_uncertainty": 0.00694,
            "discrepancy": 0.14931,
            "ratio_prev_over_finest": 0.95,
            "neighbourhood_delta_max": 0.00431,
            "verdict": (
                "reference defect: the entry sits outside the monotone wall-region "
                "profile of its own neighbours (-0.22847 at x=0.9453, -0.44993 at "
                "x=0.8594), and the residual does not shrink under refinement "
                "(0.115/0.142/0.149 for n=32/64/128, ratio 0.95 = 21.5x the sample's "
                "own resolution uncertainty), while the other 16 published v-points sit "
                "inside 0.0045 and the u-centreline converges at ~2nd order. The three "
                "discrete profiles converge to one smooth limit (-0.3528/-0.3806/-0.3876, "
                "increment ratio 4.0) through the point."
            ),
            "evidence": "docs/verification-ghia-reference-consistency/results.json",
            "action": (
                "excluded from consistent-point statistics; the value must be re-keyed "
                "from the printed Table II before it is scored again"
            ),
        }
    },
}

SUPPORTED_REYNOLDS = tuple(sorted(U_CENTRELINE))


def _finite(*values: float) -> bool:
    return all(isinstance(v, (int, float)) and math.isfinite(v) for v in values)


def table(reynolds: float, which: str) -> tuple[tuple[float, float], ...]:
    """Return the published table for ``reynolds``; ``which`` is ``u`` or ``v``."""
    re = float(reynolds)
    if re not in U_CENTRELINE:
        raise KeyError(f"no Ghia table for Re={reynolds}; available {SUPPORTED_REYNOLDS}")
    if which not in ("u", "v"):
        raise ValueError("which must be 'u' or 'v'")
    return (U_CENTRELINE if which == "u" else V_CENTRELINE)[re]



def suspect_entries(reynolds: float, which: str) -> dict[float, Mapping[str, object]]:
    """Return the audit-rejected entries for one table (may be empty)."""
    return dict(SUSPECT_ENTRIES.get((float(reynolds), which), {}))


def consistent_points(reynolds: float, which: str,
                      *, include_suspect: bool = False) -> tuple[tuple[float, float], ...]:
    """Return the table with audit-rejected entries removed.

    ``include_suspect=True`` restores the raw digitised table, which is what the
    committed evidence bundles and :func:`table` always report.
    """
    pts = table(reynolds, which)
    if include_suspect:
        return tuple(pts)
    bad = suspect_entries(reynolds, which)
    return tuple(p for p in pts if p[0] not in bad)

def extrema(reynolds: float, which: str) -> tuple[tuple[float, float], tuple[float, float]]:
    """Return (argmin, argmax) points of the published centreline table."""
    pts = table(reynolds, which)
    lo = min(pts, key=lambda p: p[1])
    hi = max(pts, key=lambda p: p[1])
    return lo, hi


def self_check() -> dict[str, object]:
    """Validate the transcribed tables against their published extrema."""
    report: dict[str, object] = {"ok": True, "cases": {}}
    expected = {
        100.0: ((-0.21090, 0.4531), (0.17527, 0.2344)),
        400.0: ((-0.32726, 0.2813), (0.30203, 0.2266)),
        1000.0: ((-0.38289, 0.1719), (0.37095, 0.1563)),
    }
    for re, ((u_min, y_min), (v_max, x_max)) in expected.items():
        ulo, _ = extrema(re, "u")
        _, vhi = extrema(re, "v")
        ok = (abs(ulo[1] - u_min) < 1e-9 and abs(ulo[0] - y_min) < 1e-9
              and abs(vhi[1] - v_max) < 1e-9 and abs(vhi[0] - x_max) < 1e-9)
        report["cases"][re] = {"ok": ok, "u_min": ulo, "v_max": vhi}
        report["ok"] = bool(report["ok"]) and ok
    # monotone coordinate ordering, unit walls
    for re in SUPPORTED_REYNOLDS:
        for which in ("u", "v"):
            pts = table(re, which)
            coords = [p[0] for p in pts]
            if coords != sorted(coords, reverse=True):
                report["ok"] = False
                report.setdefault("ordering_errors", []).append((re, which))
            if not all(_finite(*p) for p in pts):
                report["ok"] = False
                report.setdefault("nonfinite", []).append((re, which))
    return report