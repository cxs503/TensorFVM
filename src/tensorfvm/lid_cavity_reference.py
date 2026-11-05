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

#: Bibliographic target, quoted verbatim in every artefact that scores against it.
REFERENCE = ("U. Ghia, K.N. Ghia, C.T. Shin, J. Comput. Phys. 48 (1982) 387-411, "
             "Tables I (u at x = 0.5) and II (v at y = 0.5)")

#: Reference entries the solver cannot reproduce at *any* resolution.
#:
#: A dispute is *not* a licence to edit the transcription: the published number
#: stays exactly as printed and any gate that ignores it must say so out loud.
#: Each record carries the evidence that produced it, so the claim can be
#: re-checked (and overturned) without reading the prose report.
DISPUTES: dict[tuple[float, str, float], dict[str, object]] = {
    (400.0, "v", 0.9063): {
        "published": -0.23827,
        "computed_n": {32: -0.35284, 64: -0.38064, 128: -0.38758},
        "observed_order": 2.00,
        "richardson_limit": -0.38989,
        "limit_relative_error": 0.637,
        "metric_interpolation_error_bound_n128": 1.0e-3,
        "neighbour_deltas_n128": {0.9453: 0.0016, 0.8594: 0.0021, 0.8047: 0.0028},
        "evidence": ("scripts/audit_ghia_re400_v_attribution.py over the committed "
                     "ladder bundle docs/verification-lid-cavity-ladder/summary.json; "
                     "clean second-order convergence to a limit 64% away from the "
                     "tabulated value while the three neighbouring entries are inside "
                     "0.3%, and the interpolation error of the metric path is two "
                     "orders of magnitude smaller than the gap"),
        "adjudication": ("needs the printed Table II or an independent DNS; offline "
                         "copies of the paper are not available in this checkout"),
    },
}


#: Corrections that were *proposed* for a printed reference entry and *rejected*.
#:
#: A proposal is recorded verbatim so the reasoning stays auditable even though it
#: is not in force.  Branch ``agent/m3240`` (mission m3240) proposed replacing the
#: printed ``V_CENTRELINE[400.0]`` value at ``x = 0.9063`` with ``-0.38760``, calling
#: ``-0.23827`` a transcription error.  The proposal is refused: ``-0.38760`` is the
#: solver's *own* extrapolated limit (Richardson limit ``-0.38989``, see
#: :data:`DISPUTES`, agreement 0.6 %), so scoring against it would be circular -- the
#: reference would be fitted to the solver it exists to judge, and the Re = 400
#: wall-adjacent field defect would leave the gate without being fixed.  It is also
#: unsourced: the printed Table II has never been consulted in this checkout.  The
#: published number therefore stays exactly as printed and the defect stays visible
#: as a strict xfail; see ``docs/GHIA_RE400_V_ATTRIBUTION.md`` and ``revisit_if``.
REJECTED_CORRECTIONS: dict[tuple[float, str, float], dict[str, object]] = {
    (400.0, "v", 0.9063): {
        "published": -0.23827,
        "proposed": -0.38760,
        "proposed_by": "branch agent/m3240 (mission m3240)",
        "proposal_rationale": (
            "the printed value was called a transcription error because it breaks "
            "the boundary-layer profile between x = 0.9453 (-0.22847) and "
            "x = 0.8594 (-0.44993), and because the solver's Re=400 v-centreline "
            "error grew under refinement (0.1146/0.1424/0.1493 at n = 32/64/128)"),
        "rejection_reasons": (
            "circular: the replacement reproduces the solver's own Richardson limit "
            "(-0.38989) to 0.6 %, so it cannot separate solver error from reference "
            "error; unsourced: no printed Table II or independent DNS was consulted; "
            "lossy: adopting it hides the wall-adjacent field defect that the strict "
            "xfail exists to keep visible"),
        "disposition": "rejected",
        "revisit_if": (
            "the printed Table II is checked against a physical copy of Ghia et al. "
            "(1982), or an independent Re=400 DNS puts v(0.5, 0.9063) far from "
            "-0.23827"),
    },
}


def smoothness_audit(threshold: float = 10.0) -> dict:
    """Flag tabulated entries whose adjacent secant slopes jump by ``threshold``.

    A published centreline profile is a smooth curve sampled on a non-uniform
    grid: between two neighbouring intervals the secant slope may change by tens
    of percent, but a change of an order of magnitude across one interval means
    the sampled curve has a corner there, i.e. at least one of the three entries
    is inconsistent with the other two.  The statistic is deliberately local
    (both neighbours must lie within ``0.06 L``) so it cannot be triggered by the
    genuine large-scale curvature of a high-Reynolds-number profile.
    """
    flagged = []
    entries = 0
    for reynolds in SUPPORTED_REYNOLDS:
        for which in ("u", "v"):
            pts = sorted(table(reynolds, which))
            for index in range(1, len(pts) - 1):
                left, here, right = pts[index - 1], pts[index], pts[index + 1]
                if (here[0] - left[0]) > 0.06 or (right[0] - here[0]) > 0.06:
                    continue
                entries += 1
                s_left = (here[1] - left[1]) / (here[0] - left[0])
                s_right = (right[1] - here[1]) / (right[0] - here[0])
                scale = max(abs(s_left), abs(s_right), 1e-12)
                ratio = scale / max(min(abs(s_left), abs(s_right)), 1e-12)
                if ratio >= threshold:
                    flagged.append({
                        "key": f"{reynolds:g}/{which}/{here[0]:.4f}",
                        "reynolds": reynolds, "which": which, "x": here[0],
                        "value": here[1], "slope_ratio": ratio,
                        "slope_left": s_left, "slope_right": s_right,
                    })
    registered = {f"{k[0]:g}/{k[1]}/{k[2]:.4f}" for k in DISPUTES}
    unexplained = [item for item in flagged
                   if not any(item["reynolds"] == k[0] and item["which"] == k[1]
                              and abs(item["x"] - k[2]) <= 0.06 for k in DISPUTES)]
    return {"entries": entries, "threshold": threshold, "flagged": flagged,
            "registered": sorted(registered),
            "registered_flagged": sorted(registered & {item["key"] for item in flagged}),
            "unexplained": [item["key"] for item in unexplained],
            "ok": bool(flagged) and not unexplained
                  and registered <= {item["key"] for item in flagged}}
