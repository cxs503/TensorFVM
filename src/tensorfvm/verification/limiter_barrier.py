"""TVD-limiter accuracy barrier at smooth extrema, resolved by error norm.

Why this module exists
----------------------
:mod:`tensorfvm.compressible.reconstruct` advertises its MUSCL reconstruction as
"second order in space and time for smooth flow and TVD across shocks".  That
sentence is true under one norm and false under another, and the merged
three-grid study (:mod:`tensorfvm.verification.cellcentre_loglog`) could not see
the difference because it only ever fitted the **L1** cell-average error.  This
module measures the same merged solver on the same exact solution but resolves
the observed order by norm, so the claim can be checked instead of inherited.

The physics/maths under test is the classical monotonicity barrier
(van Leer 1979, *J. Comput. Phys.* 32:101; LeVeque 2002, *Finite Volume Methods
for Hyperbolic Problems*, §6.13; Suresh & Huynh 1997, *J. Comput. Phys.*
136:83 for the accuracy-preserving remedy): a TVD slope limiter is forced to
suppress the reconstruction where the data has a local extremum, so the
reconstruction degenerates there and the pointwise (L-infinity) convergence rate
of a formally second-order scheme drops below two.  The L1 norm averages the
degraded few cells over O(1/h) well-resolved cells and hides the loss.

What is measured (no number in this file is typed from a table)
--------------------------------------------------------------
``control``      the unlimited centred slope (``limiter="none"``) must recover
                 order 2 in *both* norms -- otherwise the deficit is a solver
                 bug, not a limiter property.
``masking``      for every TVD limiter (``minmod``, ``vanleer``, ``mc``,
                 ``superbee``) the L1 observed order stays inside the 15 % band
                 around 2 while the L-infinity order leaves it downwards.
``mechanism``    the limiter's own arithmetic is inspected on the converged
                 field: the set of cells whose data carries a local extremum has
                 a **grid-independent count** (two, one crest and one trough per
                 period), and there ``minmod``/``vanleer``/``superbee`` return a
                 slope that is exactly zero while ``mc`` provably never
                 amplifies the centred slope.  The defective set therefore does
                 not shrink under refinement -- which is *why* L-infinity cannot
                 converge at order two, rather than a fitted observation.
``price``        the amplification of the L-infinity error relative to the
                 unlimited scheme at the finest grid, and the growth of that
                 amplification under refinement (the error-rate deficit).

Everything is deterministic (fixed grids, fixed CFL) and runs the *merged*
solver through :func:`tensorfvm.verification.cellcentre_loglog.run_case`, so this
module adds measurements, not a second solver path.  The limiter arithmetic is
mirrored in NumPy only for inspection; :func:`mirror_error` proves that mirror
agrees with :func:`tensorfvm.compressible.reconstruct.limited_slope` to round-off
before any number derived from it is used.
"""
from __future__ import annotations

import json
import math
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np

from .cellcentre_loglog import THRESHOLD_ACCEPTABLE
from .cellcentre_loglog import adjacent_slopes, loglog_slope, spacings
from .cellcentre_loglog import run_case, three_grid_readout

#: Unlimited centred slope: the second-order baseline, not a TVD limiter.
UNLIMITED = "none"
#: TVD limiters shipped by ``tensorfvm.compressible.reconstruct.LIMITERS``.
TVD_LIMITERS: Tuple[str, ...] = ("mc", "vanleer", "superbee", "minmod")
#: Everything measured here, control first.
PROBES: Tuple[str, ...] = (UNLIMITED,) + TVD_LIMITERS

#: Error norms the observed order is fitted on.  ``l1`` is what the merged
#: study fitted; ``linf`` is what exposes the barrier.
NORMS: Tuple[str, ...] = ("l1", "linf")

#: Owner precision standard: a relative deviation above 15 % is not acceptable.
THRESHOLD = THRESHOLD_ACCEPTABLE
#: An order deficit below this is numerically indistinguishable from round-off.
MIN_NORM_GAP = 0.30
#: A clip factor below this means the limiter removed over half of the slope.
HEAVY_CLIP = 0.5
#: ``|s_cen|`` below this fraction of the field slope scale is treated as "the
#: centred slope itself is zero" and gets no clip factor.
CLIP_SCALE_FLOOR = 1e-6
#: Amplification of the L-infinity error that counts as a measured price.
MIN_AMPLIFICATION = 5.0
#: Growth of that amplification from the coarsest to the finest grid.
MIN_AMPLIFICATION_GROWTH = 1.5


# ----------------------------------------------------------------------
# limiter arithmetic, mirrored for inspection
# ----------------------------------------------------------------------
def _minmod(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.where(a * b > 0.0, np.where(np.abs(a) < np.abs(b), a, b), 0.0)


def centred_slope(q: np.ndarray) -> np.ndarray:
    """Unlimited centred slope ``(q_{i+1} - q_{i-1})/2`` of a periodic field."""
    return 0.5 * (np.roll(q, -1) - np.roll(q, 1))


def slopes(q: np.ndarray, limiter: str = UNLIMITED) -> np.ndarray:
    """Slope of a periodic cell-centred field, exactly as the solver computes it.

    Mirrors :func:`tensorfvm.compressible.reconstruct.limited_slope` on the
    interior of a periodically padded field: ``a = q_{i+1} - q_i``,
    ``b = q_i - q_{i-1}``.  :func:`mirror_error` binds the two implementations.
    """
    q = np.asarray(q, dtype=float)
    if q.ndim != 1:
        raise ValueError("a 1-D periodic field is required")
    a = np.roll(q, -1) - q
    b = q - np.roll(q, 1)
    centred = 0.5 * (a + b)
    if limiter == "none":
        return centred
    if limiter == "minmod":
        return _minmod(a, b)
    if limiter == "vanleer":
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where(a * b > 0.0, 2.0 * a * b / (a + b), 0.0)
        return np.where(a * b > 0.0, ratio, 0.0)
    if limiter == "mc":
        lim = np.minimum(np.minimum(2.0 * np.abs(a), 2.0 * np.abs(b)),
                         np.abs(centred))
        return np.sign(centred) * lim
    if limiter == "superbee":
        s1 = _minmod(2.0 * a, b)
        s2 = _minmod(a, 2.0 * b)
        return np.where(np.abs(s1) >= np.abs(s2), s1, s2)
    raise ValueError(f"unknown limiter {limiter!r}; choose from {sorted(PROBES)}")


def mirror_error(seed: int = 20261114, size: int = 97,
                 limiters: Sequence[str] = PROBES) -> Dict[str, float]:
    """Max relative disagreement between :func:`slopes` and the solver's torch path.

    A smooth field, a monotone field, a field with exact extrema and a random
    field are compared, so the mirror is exercised on all four branches
    (``a*b > 0``, ``a*b < 0``, ``a*b == 0``, ties in ``superbee``).
    """
    import torch

    from ..compressible.reconstruct import limited_slope

    rng = np.random.default_rng(seed)
    x = (np.arange(size) + 0.5) / size
    fields = {
        "smooth": 1.0 + 0.2 * np.sin(2.0 * np.pi * x),
        "monotone": np.linspace(0.0, 1.0, size),
        "extremum": np.cos(2.0 * np.pi * x),
        "random": rng.standard_normal(size),
    }
    out: Dict[str, float] = {}
    for limiter in limiters:
        worst = 0.0
        for name, field in fields.items():
            # two periodic ghost cells on each side, as the solver pads
            padded = torch.tensor(
                np.concatenate([field[-2:], field, field[:2]]),
                dtype=torch.float64).reshape(-1, 1)
            ref = limited_slope(padded, 0, limiter=limiter).numpy()[2:-2, 0]
            got = slopes(field, limiter)
            scale = max(float(np.max(np.abs(ref))), np.finfo(float).tiny)
            worst = max(worst, float(np.max(np.abs(ref - got))) / scale)
        out[limiter] = worst
    return out


# ----------------------------------------------------------------------
# per-cell diagnostics on a converged field
# ----------------------------------------------------------------------
def extremum_cells(q: np.ndarray) -> np.ndarray:
    """Cells where the TVD constraint applies (``a*b <= 0``, both non-zero).

    ``a = q_{i+1} - q_i`` and ``b = q_i - q_{i-1}``; the limiter suppresses the
    slope unless ``a*b > 0``, so a vanishing product counts too -- this is the
    cell that sits *on* a smooth extremum, and it is the common case on a
    symmetric grid (``a*b`` is then ``-0.0``, hence the ``<=``).  A locally flat
    cell (``a == b == 0``) is excluded: the centred slope is zero there anyway.

    The count is a property of the *solution*, not of the grid: a density wave of
    one period per unit length has exactly two such cells whatever ``nx`` is.
    """
    q = np.asarray(q, dtype=float)
    a = np.roll(q, -1) - q
    b = q - np.roll(q, 1)
    flat = (a == 0.0) & (b == 0.0)
    return ((a * b) <= 0.0) & ~flat


def clipping(q: np.ndarray, limiter: str) -> Dict:
    """How much of the centred slope the limiter keeps, cell by cell."""
    centred = slopes(q, "none")
    limited = slopes(q, limiter)
    scale = float(np.max(np.abs(centred)))
    if scale <= 0.0:
        raise ValueError("field has no slope scale")
    floor = CLIP_SCALE_FLOOR * scale
    usable = np.abs(centred) > floor
    with np.errstate(divide="ignore", invalid="ignore"):
        factor = np.where(usable, np.abs(limited) / np.abs(centred), np.nan)
    extrema = extremum_cells(q)
    return {
        "limiter": limiter,
        "slope_scale": scale,
        "centred": centred,
        "limited": limited,
        "clip_factor": factor,
        "extremum_mask": extrema,
        "extremum_cells": int(extrema.sum()),
        "extremum_slope_zero": bool(np.all(limited[extrema] == 0.0)),
        "extremum_slope_max_ratio": (
            float(np.max(np.abs(limited[extrema]) / np.abs(centred[extrema])))
            if extrema.any() else 0.0),
        #: True for the diffusive limiters; ``superbee`` is compressive and may
        #: legitimately return a slope larger than the centred one.
        "never_amplifies": bool(np.all(np.abs(limited) <= np.abs(centred)
                                       + 1e-12 * scale)),
        "heavy_clip_cells": int(np.sum(usable & (factor < HEAVY_CLIP))),
        "min_clip_factor": (float(np.nanmin(factor)) if np.any(usable) else float("nan")),
    }


# ----------------------------------------------------------------------
# ladder of solver runs
# ----------------------------------------------------------------------
def run_field(nx: int, limiter: str, *, order: int = 2, cfl: float = 0.4,
              t_end: float = 0.2, phase: float = 0.0) -> Dict:
    """One solver run plus the per-cell limiter diagnostics on its final field."""
    row = run_case(int(nx), order=int(order), cfl=float(cfl), t_end=float(t_end),
                   limiter=str(limiter), phase=float(phase))
    clip = clipping(row["rho"], limiter)
    out = {k: v for k, v in row.items()
           if k not in ("rho", "rho_exact_average", "rho_exact_point", "x")}
    out["clip"] = {k: v for k, v in clip.items()
                   if k not in ("centred", "limited", "clip_factor",
                               "extremum_mask")}
    return out


def ladder(grids: Iterable[int], limiter: str, **kwargs) -> List[Dict]:
    """Run one limiter ladder, coarse -> fine as ``grids`` is given."""
    return [run_field(int(n), limiter, **kwargs) for n in grids]


def _readouts(grids: Sequence[int], rows: Sequence[Dict], metric: str,
              norm: str, theory: float) -> Dict:
    """Three-grid readout on the first three grids, plus the long-ladder slope.

    :func:`cellcentre_loglog.three_grid_readout` is deliberately restricted to
    exactly three grids (that is where the Richardson solve is defined); a longer
    ladder is summarised by the least-squares log-log slope and the adjacent
    slopes only, which is what the "not a three-grid artifact" check compares.
    """
    series = [float(row[metric][norm]) for row in rows]
    three = list(grids[:3])[: len(series)]
    read = three_grid_readout(three, series[:3], theory_order=theory)
    read["metric"] = metric
    read["norm"] = norm
    read["loglog_rel_dev_from_theory"] = abs(read["p_loglog"] - theory) / theory
    read["loglog_within_15"] = bool(read["loglog_rel_dev_from_theory"] <= THRESHOLD)
    if len(series) > 3:
        h_all = spacings(grids)
        read["long_ladder"] = {
            "grids": [int(n) for n in grids],
            "errors": series,
            "p_loglog": loglog_slope(h_all, series),
            "p_adjacent": adjacent_slopes(h_all, series),
        }
    return read


def study(grids: Sequence[int], *, probes: Sequence[str] = PROBES,
          theory: float = 2.0, **kwargs) -> Dict:
    """Full norm-resolved barrier study: one ladder per probe, both norms fitted.

    ``theory`` is the order the reconstruction declares (2).  The first three
    grids of ``grids`` are the three-grid ladder the merged study used; a longer
    ladder is used for the "not a three-grid artifact" check when available.
    """
    grids = [int(n) for n in grids]
    if len(grids) < 3:
        raise ValueError("at least three grids are needed for an observed order")
    out: Dict = {
        "grids": grids,
        "probes": list(probes),
        "theory_order": float(theory),
        "threshold": THRESHOLD,
        "min_norm_gap": MIN_NORM_GAP,
        "rows": {},
        "readouts": {},
        "mirror_max_rel_error": mirror_error(),
    }
    for limiter in probes:
        rows = ladder(grids, limiter, **kwargs)
        out["rows"][limiter] = [
            {k: v for k, v in row.items() if k not in ("point", "sampling_bias")}
            for row in rows]
        out["readouts"][limiter] = {}
        for norm in NORMS:
            out["readouts"][limiter][norm] = _readouts(
                grids, rows, "cell_average", norm, theory)
    return out


# ----------------------------------------------------------------------
# declared criteria
# ----------------------------------------------------------------------
def _orders(study_out: Dict, limiter: str) -> Dict[str, float]:
    return {norm: float(study_out["readouts"][limiter][norm]["p_loglog"])
            for norm in NORMS}


def amplification(study_out: Dict, limiter: str, norm: str = "linf") -> List[float]:
    """Error of ``limiter`` over the unlimited error, one entry per grid."""
    base = [float(row["cell_average"][norm])
            for row in study_out["rows"][UNLIMITED]]
    got = [float(row["cell_average"][norm]) for row in study_out["rows"][limiter]]
    return [g / b for g, b in zip(got, base)]


def criteria(study_out: Dict) -> List[Dict]:
    """The predeclared criteria, evaluated on a finished study (machine verdicts)."""
    theory = float(study_out["theory_order"])
    rows: Dict[str, List[Dict]] = study_out["rows"]
    out: List[Dict] = []

    def add(name: str, claim: str, observed, limit: str, passed: bool) -> None:
        out.append({"name": name, "claim": claim, "observed": observed,
                    "threshold": limit, "passed": bool(passed)})

    # B1 -- the inspection mirror is the solver's own limiter arithmetic
    mirror = study_out["mirror_max_rel_error"]
    worst = max(float(v) for v in mirror.values())
    add("B1", "the NumPy limiter mirror reproduces the solver's torch path",
        worst, "<= 1e-12 relative, all five probes", worst <= 1e-12)

    # B2 -- unlimited control: order 2 in both norms
    ctrl = _orders(study_out, UNLIMITED)
    ctrl_dev = {norm: abs(ctrl[norm] - theory) / theory for norm in NORMS}
    add("B2", "unlimited centred slope recovers order 2 in L1 and in L-inf",
        {k: round(v, 4) for k, v in ctrl.items()},
        "each |p-2|/2 <= 15 %",
        all(v <= THRESHOLD for v in ctrl_dev.values()))

    # B3 -- L1 masking: every TVD limiter still reads order 2 in L1
    l1 = {lim: _orders(study_out, lim)["l1"] for lim in TVD_LIMITERS}
    l1_dev = {lim: abs(p - theory) / theory for lim, p in l1.items()}
    add("B3", "every TVD limiter stays inside the 15 % order band on the L1 norm",
        {k: round(v, 4) for k, v in l1.items()},
        "each |p-2|/2 <= 15 %",
        all(v <= THRESHOLD for v in l1_dev.values()))

    # B4 -- L-inf barrier: every TVD limiter leaves the band downwards
    linf = {lim: _orders(study_out, lim)["linf"] for lim in TVD_LIMITERS}
    linf_def = {lim: (theory - p) / theory for lim, p in linf.items()}
    add("B4", "every TVD limiter's L-inf observed order is below 2 by > 15 %",
        {k: round(v, 4) for k, v in linf.items()},
        "each (2-p)/2 >= 15 %",
        all(v >= THRESHOLD for v in linf_def.values()))

    # B5 -- the norm is what decides the verdict
    gaps = {lim: l1[lim] - linf[lim] for lim in TVD_LIMITERS}
    add("B5", "the L1-L-inf order gap of every TVD limiter exceeds 0.3",
        {k: round(v, 4) for k, v in gaps.items()},
        f"each gap >= {MIN_NORM_GAP}",
        all(v >= MIN_NORM_GAP for v in gaps.values()))

    # B6 -- the defective set does not shrink under refinement
    counts = {lim: [int(row["clip"]["extremum_cells"]) for row in rows[lim]]
              for lim in TVD_LIMITERS}
    add("B6", "the cell set where the TVD constraint bites has a grid-independent "
              "count (2, or 4 when an extremum falls on a cell boundary)",
        counts, "constant over the ladder and <= 4 for every probe",
        all(len(set(series)) == 1 and series[0] <= 4
            for series in counts.values()))

    # B7 -- the mechanism, read off the limiter's own arithmetic
    zero_at_extrema = {lim: [bool(row["clip"]["extremum_slope_zero"])
                             for row in rows[lim]] for lim in TVD_LIMITERS}
    no_amplify = {lim: [bool(row["clip"]["never_amplifies"])
                        for row in rows[lim]] for lim in TVD_LIMITERS}
    structural = ("minmod/vanleer/superbee return an exactly zero slope at every "
                  "extremum cell; minmod/vanleer/mc never amplify the centred slope "
                  "(superbee is compressive and may, so it is exempt)")
    add("B7", structural,
        {"extrema_zero": {k: all(v) for k, v in zero_at_extrema.items()},
         "never_amplifies": {k: all(v) for k, v in no_amplify.items()}},
        "all true (mc is checked by no-amplification instead of exact zero; "
        "superbee is exempt from no-amplification)",
        all(all(no_amplify[lim]) for lim in ("minmod", "vanleer", "mc"))
        and all(all(zero_at_extrema[lim]) for lim in ("minmod", "vanleer", "superbee")))

    # B8 / B9 -- the price, and the fact that it grows
    amp_finest = {lim: amplification(study_out, lim)[-1] for lim in TVD_LIMITERS}
    amp_growth = {}
    for lim in TVD_LIMITERS:
        series = amplification(study_out, lim)
        amp_growth[lim] = series[-1] / series[0] if series[0] > 0 else float("inf")
    add("B8", "the L-inf error of every TVD limiter is amplified > 5x at the finest grid",
        {k: round(v, 3) for k, v in amp_finest.items()},
        f"each amplification >= {MIN_AMPLIFICATION}",
        all(v >= MIN_AMPLIFICATION for v in amp_finest.values()))
    add("B9", "the amplification grows under refinement (the error rates diverge)",
        {k: round(v, 3) for k, v in amp_growth.items()},
        f"finest/coarsest >= {MIN_AMPLIFICATION_GROWTH}",
        all(v >= MIN_AMPLIFICATION_GROWTH for v in amp_growth.values()))

    # B10 -- not a three-grid artifact
    grids = list(study_out["grids"])
    if len(grids) >= 4:
        wide = {}
        for lim in TVD_LIMITERS:
            long_read = study_out["readouts"][lim]["linf"].get("long_ladder")
            p_wide = float(long_read["p_loglog"])
            p_three = linf[lim]
            wide[lim] = {
                "p_long_ladder": p_wide,
                "p_three_grid": p_three,
                "rel_diff": abs(p_wide - p_three) / p_three,
                "adjacent": long_read["p_adjacent"],
            }
        add("B10", "the deficit survives a longer ladder (not a three-grid artifact)",
            {k: {kk: (round(vv, 4) if isinstance(vv, float) else
                      [round(a, 4) for a in vv])
                 for kk, vv in v.items()} for k, v in wide.items()},
            "long-ladder L-inf order within 15 % of the three-grid value",
            all(v["rel_diff"] <= THRESHOLD for v in wide.values()))
    else:
        add("B10", "the deficit survives a longer ladder (not a three-grid artifact)",
            "not_run: fewer than four grids in this study", "long ladder", False)

    # B11 -- the idealised "exactly first order" hypothesis is NOT confirmed
    deficit_to_one = {lim: abs(linf[lim] - 1.0) / 1.0 for lim in TVD_LIMITERS}
    not_one = all(v > THRESHOLD for v in deficit_to_one.values())
    add("B11", "the idealisation 'a TVD limiter is exactly first order in L-inf "
               "at a smooth extremum on accessible grids' is not confirmed",
        {k: round(v, 4) for k, v in deficit_to_one.items()},
        "declared NOT_CONFIRMED when every |p-1|/1 > 15 %",
        not_one)

    # B12 -- no false promotion of the declared order
    labels = _labels(study_out)
    add("B12", "no TVD probe is reported as uniformly second order",
        {k: v["label"] for k, v in labels.items()},
        "no TVD probe labelled 'second order in both norms'",
        all(labels[lim]["label"] != "second-order-in-both-norms"
            for lim in TVD_LIMITERS))

    out.append({"name": "B13", "claim": "the module docstring of the reconstruction "
                "states the norm dependence of its order claim",
                "observed": declared_claim_audit()["documented"],
                "threshold": "the words 'L1' and 'L-inf' both appear in the "
                             "reconstruct docstring next to the order claim",
                "passed": bool(declared_claim_audit()["documented"])})
    return out


def declared_claim_audit() -> Dict:
    """Check the order claim in :mod:`tensorfvm.compressible.reconstruct`.

    The module used to state an unconditional "second order ... for smooth
    flow".  The audit requires the docstring to name both norms, so the claim
    cannot drift back into an unqualified one.
    """
    from ..compressible import reconstruct

    doc = reconstruct.__doc__ or ""
    documented = ("L1" in doc) and ("L-inf" in doc)
    return {
        "module": "tensorfvm.compressible.reconstruct",
        "mentions_L1": "L1" in doc,
        "mentions_Linf": "L-inf" in doc,
        "documented": bool(documented),
    }


# ----------------------------------------------------------------------
# verdict
# ----------------------------------------------------------------------
def _labels(study_out: Dict) -> Dict[str, Dict]:
    """Per-probe label, derived only from the measured orders (no criteria call)."""
    theory = float(study_out["theory_order"])
    per_probe: Dict[str, Dict] = {}
    for limiter in study_out["probes"]:
        orders = _orders(study_out, limiter)
        l1_ok = abs(orders["l1"] - theory) / theory <= THRESHOLD
        linf_ok = abs(orders["linf"] - theory) / theory <= THRESHOLD
        if l1_ok and linf_ok:
            label = "second-order-in-both-norms"
            note = "declared order recovered under both norms"
        elif l1_ok and not linf_ok:
            label = "second-order-in-L1-only"
            note = ("the documented order holds in the volume-averaged norm and "
                    "fails pointwise: the TVD constraint bites at smooth extrema")
        elif linf_ok and not l1_ok:
            label = "second-order-in-Linf-only"
            note = "inconsistent: the pointwise norm is the stricter one"
        else:
            label = "below-declared-order-in-both-norms"
            note = "the declared order is not delivered under either norm"
        per_probe[limiter] = {"label": label, "note": note,
                              "p_l1": orders["l1"], "p_linf": orders["linf"],
                              "amplification_finest": (
                                  amplification(study_out, limiter)[-1]
                                  if limiter != UNLIMITED else 1.0)}
    return per_probe


def verdict(study_out: Dict) -> Dict:
    """Per-probe label plus the criterion tally (criteria call ``_labels``)."""
    per_probe = _labels(study_out)
    checks = criteria(study_out)
    passed = [c for c in checks if c["passed"]]
    return {
        "per_probe": per_probe,
        "criteria_passed": len(passed),
        "criteria_total": len(checks),
        "all_passed": len(passed) == len(checks),
        "headline": ("the declared second order of the MUSCL/TVD reconstruction "
                     "is a norm-dependent claim: it holds in L1 and fails in "
                     "L-inf for every TVD limiter"),
    }


# ----------------------------------------------------------------------
# figure + entry point
# ----------------------------------------------------------------------
def figure(study_out: Dict, path: str) -> str:
    """Three panels: log-log error, observed order per norm, limiter clipping."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    grids = study_out["grids"]
    h = np.asarray(spacings(grids), dtype=float)
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.6))
    colours = {lim: c for lim, c in zip(
        PROBES, ("#444444", "#1f77b4", "#2ca02c", "#d62728", "#9467bd"))}

    ax = axes[0]
    for limiter in study_out["probes"]:
        rows = study_out["rows"][limiter]
        for norm, style in (("l1", "--"), ("linf", "-")):
            e = [float(row["cell_average"][norm]) for row in rows]
            ax.loglog(h, e, style, color=colours[limiter], marker="o", ms=3.5,
                      label=f"{limiter} {norm}")
    ref = np.asarray([float(study_out["rows"][UNLIMITED][0]["cell_average"]["linf"])])
    ax.loglog(h, ref * (h / h[0]) ** 2, ":", color="k", lw=1.2, label="slope 2")
    ax.set_xlabel("h = 1/nx")
    ax.set_ylabel("L1 (dashed) / L-inf (solid) cell-average error")
    ax.set_title("entropy wave, MUSCL-Hancock + HLLC")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=6.5, ncol=2)

    ax = axes[1]
    xs = np.arange(len(study_out["probes"]))
    w = 0.38
    l1 = [_orders(study_out, lim)["l1"] for lim in study_out["probes"]]
    linf = [_orders(study_out, lim)["linf"] for lim in study_out["probes"]]
    ax.bar(xs - w / 2, l1, w, label="L1 order", color="#1f77b4")
    ax.bar(xs + w / 2, linf, w, label="L-inf order", color="#d62728")
    ax.axhline(study_out["theory_order"], color="k", ls="--", lw=1.2,
               label="declared order 2")
    ax.axhspan(study_out["theory_order"] * (1 - THRESHOLD),
               study_out["theory_order"] * (1 + THRESHOLD), color="k", alpha=0.08,
               label="15 % band")
    ax.set_xticks(xs)
    ax.set_xticklabels(study_out["probes"], rotation=20)
    ax.set_ylabel("observed order")
    ax.set_title("the verdict depends on the norm")
    ax.legend(fontsize=7)
    ax.grid(True, axis="y", alpha=0.3)

    ax = axes[2]
    for limiter in TVD_LIMITERS:
        rows = study_out["rows"][limiter]
        ax.plot(grids, [row["clip"]["heavy_clip_cells"] for row in rows],
                marker="o", color=colours[limiter], label=limiter)
        ax.plot(grids, [row["clip"]["extremum_cells"] for row in rows],
                ls=":", marker=".", color=colours[limiter])
    ax.set_xlabel("nx")
    ax.set_ylabel("cells with clip factor < 0.5  (dotted: extremum cells)")
    ax.set_title("the defective set does not shrink with h")
    ax.set_xscale("log", base=2)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=7)

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def run(grids: Sequence[int] = (40, 80, 160, 320, 640),
        out_dir: str = "outputs/limiter-barrier") -> Dict:
    """End-to-end study: measure, judge, write ``summary.json`` and the figure."""
    import os

    study_out = study(grids)
    checks = criteria(study_out)
    verdict_out = verdict(study_out)
    os.makedirs(out_dir, exist_ok=True)
    fig_path = os.path.join(out_dir, "fig_limiter_barrier.png")
    figure(study_out, fig_path)
    summary = {
        "grids": study_out["grids"],
        "probes": study_out["probes"],
        "theory_order": study_out["theory_order"],
        "threshold": study_out["threshold"],
        "mirror_max_rel_error": study_out["mirror_max_rel_error"],
        "orders": {lim: _orders(study_out, lim) for lim in study_out["probes"]},
        "readouts": study_out["readouts"],
        "amplification": {lim: amplification(study_out, lim)
                          for lim in TVD_LIMITERS},
        "clip": {lim: [row["clip"] for row in study_out["rows"][lim]]
                 for lim in study_out["probes"]},
        "errors": {lim: [{k: row["cell_average"][k] for k in NORMS}
                         for row in study_out["rows"][lim]]
                   for lim in study_out["probes"]},
        "criteria": checks,
        "verdict": verdict_out,
        "declared_claim_audit": declared_claim_audit(),
        "figure": fig_path,
    }
    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=1, sort_keys=True)
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    import sys

    args = list(sys.argv[1:] if argv is None else argv)
    grids = tuple(int(a) for a in args) if args else (40, 80, 160, 320, 640)
    summary = run(grids)
    verdict_out = summary["verdict"]
    print(f"criteria: {verdict_out['criteria_passed']}/{verdict_out['criteria_total']} PASS")
    for lim, order in summary["orders"].items():
        print(f"  {lim:9s} p_L1={order['l1']:.4f} p_Linf={order['linf']:.4f}")
    for check in summary["criteria"]:
        flag = "PASS" if check["passed"] else "FAIL"
        print(f"  {check['name']:4s} {flag}  {check['claim']}")
    return 0 if verdict_out["all_passed"] else 1


if __name__ == "__main__":  # pragma: no cover - thin CLI
    raise SystemExit(main())


__all__ = [
    "UNLIMITED", "TVD_LIMITERS", "PROBES", "NORMS", "THRESHOLD", "MIN_NORM_GAP",
    "HEAVY_CLIP", "MIN_AMPLIFICATION", "MIN_AMPLIFICATION_GROWTH",
    "centred_slope", "slopes", "mirror_error", "extremum_cells", "clipping",
    "run_field", "ladder", "study", "criteria", "declared_claim_audit",
    "amplification", "verdict", "figure", "run", "main",
]