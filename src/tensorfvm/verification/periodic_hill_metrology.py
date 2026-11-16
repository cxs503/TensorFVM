"""Metrology audit of CASE-FVM-PERIODIC-HILLS: whose Reynolds number is it?

Why this exists
---------------
Thirty previous rounds graded this case against ``x_R/h = 5.41`` and the
``u/U_b`` profiles for ``Re = U_b H / nu = 2800``.  Every one of them took the
*definition* of ``U_b`` for granted: the solver imposes

    u_b = 1/2.035 * <u>(y) integrated over y/H in [1, 3.035]   (x-averaged)

and the reference data are normalised by whatever ``U_b`` the reference's
authors used.  Nothing in the tree checked that these two functionals agree.
If they do not, the computed case is not at Re = 2800 and part of every
published error is a units artefact rather than physics.

The reference data answer the question on their own.  For a periodic flow the
streamwise flux per unit width

    Q = (1/L) * integral over the fluid of u dA

is the same through *every* cross-section (continuity + periodicity), so the
files' fixed normalisation ``U/U_b`` makes ``Q/U_b`` a **measurable constant**:
integrating the published profiles in ``y`` gives ``Q/U_b`` at all ten stations.
This module measures it and compares it with the two candidate functionals:

``crest_band``
    the solver's current functional, ``<u>(y)`` x-averaged from the crest level
    ``y = H`` upward.  It drops the flux in the band between the local wall and
    the crest (the valley), so it is ``(Q - delta)/2.035`` where
    ``delta = mean_x integral_{h(x)}^{H} u dy`` -- a *solution dependent* number.
``flux``
    ``Q / (top - H)``, the reference's functional under the reading that
    ``Q/U_b = top - h_crest``; this one is fixed by geometry alone.

Both are evaluated here for any field, so the audit is quantitative:

* on the reference's own DNS profiles, ``Q/U_b = 2.0313 +- 0.0002`` at all ten
  stations -- the published data are internally consistent (this is the first
  time they are checked at all; before this module they were only transcribed)
  and ``2.0313`` is within 0.1 % of ``top - h_crest = 2.0334``;
* the solver's functional differs from that by ``1 - (Q-delta)/Q``, i.e. by the
  valley deficit.  Measured on the reference profiles (delta/Q = 6.3 %) the
  mislabel would be +6.9 %; measured on the archived ``Re = 2800`` laminar run
  (which has a large recirculation and therefore a small forward valley flux)
  it is only +0.9 %.

So the audit has two outcomes at once, and both are reported honestly: a real
*definition* defect (the imposed "Re" is solution dependent, contradicting the
config docstring's "grid-independent input" claim), and a *bounded*
consequence -- on the archived runs it is about 1 %, which cannot explain the
+52 % reattachment error.  The example pool entry must state the functional it
means, because two codes can disagree by 7 % in Re without disagreeing about
any physics.

Geometry is audited the same way: every reference profile starts at the wall it
was sampled from, so ``min(y/H)`` per station is the wall height at that
station.  Comparing it with :func:`tensorfvm.periodic_hill.hill_profile` checks
the case geometry against the reference data and simultaneously tests the
canonical station assignment (a mirrored assignment puts ``x/H = 1`` and
``x/H = 8`` at different wall heights; the files give 0.4536 vs 0.4527).
"""

from __future__ import annotations

import math

import numpy as np

from tensorfvm.verification import periodic_hill_reference as ref

#: Hill crest height in metres for the acceptance configuration.
HILL_HEIGHT = 1.0

#: Channel top of the acceptance configuration (``PeriodicHillConfig``).
CHANNEL_TOP = 3.036

#: Band the solver integrates over: ``[H, H + 2.035 H]``.
BAND_LO = HILL_HEIGHT
BAND_HI = HILL_HEIGHT + 2.035 * HILL_HEIGHT

#: Stations whose reference profile reaches down to the wall (no hill body).
WALL_STATIONS = tuple(float(x) for x in ref.STATIONS)


# ---------------------------------------------------------------- reference side
def profile_flux(profile: dict) -> float:
    """``Q/U_b`` implied by one published profile: ``integral (U/U_b) dy``.

    Trapezoidal in ``y/H`` over exactly the samples the file carries (the first
    sample sits one grid point above the wall, the last at ``y/H = 3.0334``);
    the truncation is what makes the measured constant 2.0313 instead of
    ``top - h_crest = 2.0334`` and is reported, not corrected away.
    """
    y = np.asarray(profile["y_over_h"], dtype=float)
    u = np.asarray(profile["u_over_ub"], dtype=float)
    if y.ndim != 1 or y.size < 2 or y.shape != u.shape:
        raise ValueError("profile must carry matching 1-D y and u")
    if not np.all(np.isfinite(y)) or not np.all(np.isfinite(u)):
        raise ValueError("profile contains non-finite samples")
    order = np.argsort(y)  # the published files are not consistently ordered
    return float(np.trapezoid(u[order], y[order]))


def reference_flux_invariant(stations=None) -> dict:
    """``Q/U_b`` at every station, plus the spread that shows it is a constant."""
    stations = WALL_STATIONS if stations is None else tuple(float(s) for s in stations)
    rows = []
    for station in stations:
        prof = ref.load_profile(station)
        rows.append({
            "x_over_h": station,
            "wall_y_over_h": float(np.min(prof["y_over_h"])),
            "top_y_over_h": float(np.max(prof["y_over_h"])),
            "q_over_ub": profile_flux(prof),
            "source": prof["source"],
        })
    q = np.array([row["q_over_ub"] for row in rows], dtype=float)
    mean = float(q.mean()) if q.size else float("nan")
    spread = float((q.max() - q.min()) / mean) if q.size and mean else float("nan")
    #: The constant must be the same at every section for a periodic flow; 0.1 %
    #: is a generous bound on trapezoid truncation, not a physical tolerance.
    tolerance = 1.0e-3
    return {
        "rows": rows,
        "mean_q_over_ub": mean,
        "min_q_over_ub": float(q.min()) if q.size else None,
        "max_q_over_ub": float(q.max()) if q.size else None,
        "relative_spread": spread,
        "tolerance": tolerance,
        "periodic_consistent": bool(math.isfinite(spread) and spread <= tolerance),
        "top_minus_crest": CHANNEL_TOP - HILL_HEIGHT,
        "mean_minus_top_minus_crest": mean - (CHANNEL_TOP - HILL_HEIGHT),
    }


def reference_deficit(stations=None, crest_level: float = HILL_HEIGHT) -> dict:
    """Valley deficit implied by the reference profiles, i.e. its own bias band.

    For each station the band ``[crest_level, top]`` integral of ``U/U_b`` is
    compared with ``Q/U_b`` of the same station; the mean shortfall is what the
    solver's functional would throw away if it were fed the *reference* field.
    This is the largest mislabel the definition can produce for this case: it
    needs a field with the reference's (turbulent, wall-jet) valley flux.
    """
    stations = WALL_STATIONS if stations is None else tuple(float(s) for s in stations)
    rows = []
    for station in stations:
        prof = ref.load_profile(station)
        y = np.asarray(prof["y_over_h"], dtype=float)
        u = np.asarray(prof["u_over_ub"], dtype=float)
        q = profile_flux(prof)
        mask = y >= crest_level
        band = float(np.trapezoid(u[mask], y[mask])) if np.count_nonzero(mask) > 1 else 0.0
        rows.append({"x_over_h": station, "q_over_ub": q,
                     "crest_band_over_ub": band,
                     "deficit_over_flux": (q - band) / q if q else float("nan")})
    deficits = np.array([row["deficit_over_flux"] for row in rows], dtype=float)
    mean_deficit = float(deficits.mean())
    return {
        "rows": rows,
        "mean_deficit_over_flux": mean_deficit,
        "min_deficit_over_flux": float(deficits.min()),
        "max_deficit_over_flux": float(deficits.max()),
        "implied_relative_reynolds_bias": reynolds_bias_from_deficit(mean_deficit),
        "note": ("the band integral is only defined at stations whose profile "
                 "reaches the crest level; the crest station's own profile has "
                 "no wall below it, so its deficit is zero by construction"),
    }


def reference_wall_surface(config=None) -> dict:
    """Wall height per station from the profiles vs the code's hill contour.

    ``min(y/H)`` of each file is the first sample above the wall of that
    station, so it over-estimates the wall by one cell; the residual therefore
    has a known sign and a magnitude bounded by the near-wall grid spacing
    (measured below as the offset implied by the crest station, where the wall
    height is exactly ``H``).
    """
    from tensorfvm.periodic_hill import hill_profile  # local: keeps import cost low

    stations = WALL_STATIONS
    crest_offset = None
    rows = []
    for station in stations:
        prof = ref.load_profile(station)
        sampled = float(np.min(prof["y_over_h"]))
        code = float(np.asarray(hill_profile(np.array([station]))[0])[0])
        if crest_offset is None and abs(station - 0.05) < 1.0e-12:
            crest_offset = sampled - HILL_HEIGHT
        rows.append({
            "x_over_h": station,
            "sampled_wall_y_over_h": sampled,
            "code_hill_y_over_h": code,
            "residual": sampled - code,
            "source": prof["source"],
        })
    offset = 0.0 if crest_offset is None else float(crest_offset)
    for row in rows:
        row["residual_minus_offset"] = row["residual"] - offset
    worst = max(abs(row["residual_minus_offset"]) for row in rows)
    #: 1 % of H: the hill contour is a cubic-fit published to four decimals, so
    #: anything larger means the code is not solving the reference's geometry.
    tolerance = 0.01 * HILL_HEIGHT
    return {
        "rows": rows,
        "crest_sample_offset": offset,
        "worst_abs_residual": float(worst),
        "tolerance": float(tolerance),
        "geometry_matches_reference": bool(worst <= tolerance),
    }


def mirrored_assignment_residual() -> dict:
    """Negative control: grade the files with ``x -> L - x`` (mirrored hill).

    Wall heights are *not* mirror symmetric about ``L/2 = 4.5`` in the
    reference's own sampling (``h(1) = 0.4536`` vs ``h(8) = 0.4527`` differ only
    by the near-wall offset), so the discriminating stations are the crest ones:
    a mirror maps ``x/H = 0.05`` (wall at the crest, ``y/H = 1.0``) onto
    ``x/H = 8.95`` (in the valley), which the code's contour rejects outright.
    """
    from tensorfvm.periodic_hill import hill_profile

    period = float(ref.STATIONS[-1]) + 1.0  # x/H = 8 is the last canonical station
    rows = []
    for station in WALL_STATIONS:
        mirrored = period - station
        prof = ref.load_profile(station)
        sampled = float(np.min(prof["y_over_h"]))
        code = float(np.asarray(hill_profile(np.array([mirrored]))[0])[0])
        rows.append({"x_over_h": station, "mirrored_x_over_h": mirrored,
                     "sampled_wall_y_over_h": sampled, "mirrored_code_y_over_h": code,
                     "residual": sampled - code})
    worst = max(abs(row["residual"]) for row in rows)
    direct = reference_wall_surface()
    return {
        "rows": rows,
        "worst_abs_residual_mirrored": float(worst),
        "worst_abs_residual_direct": direct["worst_abs_residual"],
        "direct_assignment_preferred": bool(
            direct["worst_abs_residual"] < worst),
    }


# -------------------------------------------------------------------- code side
def _mesh(config):
    from tensorfvm.periodic_hill import PeriodicHillMesh

    return PeriodicHillMesh(config)


def functionals(field, config=None) -> dict:
    """Both bulk-velocity functionals of one field, and the Re mislabel they imply.

    ``field`` is the cell-centred velocity normalised by the solver's ``U_b``
    (i.e. ``mean_u`` in the archived ``.npz``).  The mesh is rebuilt from
    ``config`` so the audit cannot silently use a different geometry than the
    run did; ``mesh.xc``/``mesh.yc`` can be compared with the archive.

    The ``crest_band`` weights reproduce
    :meth:`tensorfvm.periodic_hill.PeriodicHillSolver._bulk_weights` to
    round-off (a test pins the two to ``1e-12`` relative), so ``u_b_code`` *is*
    the functional the solver imposes, not a lookalike.
    """
    if config is None:
        from tensorfvm.periodic_hill import PeriodicHillConfig

        config = PeriodicHillConfig()
    mesh = _mesh(config)
    u = np.asarray(field, dtype=float)
    if u.shape != mesh.volume.shape:
        raise ValueError(
            f"field shape {u.shape} does not match mesh {mesh.volume.shape}")
    if not np.all(np.isfinite(u)):
        raise ValueError("field contains non-finite velocities")
    volume = mesh.volume
    cell_height = mesh.volume / mesh.dx
    bottom = mesh.yc - 0.5 * cell_height
    top = mesh.yc + 0.5 * cell_height
    overlap = np.clip(np.minimum(top, BAND_HI) - np.maximum(bottom, BAND_LO), 0.0, None)
    weight = overlap / cell_height
    weight_sum = float((weight * volume).sum())
    band_height = BAND_HI - BAND_LO
    band_integral = float((weight * u * volume).sum() / weight_sum) * band_height
    u_b_code = band_integral / band_height
    q = float((u * volume).sum() / config.length_m)
    delta = q - band_integral
    top_minus_crest = config.channel_top_m - config.height_m
    u_b_reference = q / top_minus_crest
    ratio = u_b_code / u_b_reference if u_b_reference else float("nan")
    reynolds = float(config.reynolds)
    return {
        "weights": weight,
        "u_b_code": u_b_code,
        "u_b_reference": u_b_reference,
        "flux_per_width": q,
        "band_integral": band_integral,
        "band_height": band_height,
        "valley_deficit": delta,
        "valley_deficit_over_flux": delta / q if q else float("nan"),
        "functional_ratio": ratio,
        "reynolds_labelled": reynolds,
        "reynolds_in_reference_convention":
            reynolds * u_b_reference / u_b_code if u_b_code else float("nan"),
        "relative_reynolds_bias":
            (u_b_reference / u_b_code - 1.0) if u_b_code else float("nan"),
        "top_minus_crest": top_minus_crest,
    }


def field_profile(field, x_over_h: float, config=None) -> tuple:
    """Sample ``mean_u`` on the vertical line ``x = x_over_h * H``.

    Mirrors :meth:`tensorfvm.periodic_hill.PeriodicHillSolver.profile_at`
    (linear interpolation in ``x`` between the two neighbouring columns,
    ``y`` taken on the same columns) without building a solver; returns
    ``(y/H, u/U_b)`` with ``U_b`` the configured bulk velocity.
    """
    if config is None:
        from tensorfvm.periodic_hill import PeriodicHillConfig

        config = PeriodicHillConfig()
    mesh = _mesh(config)
    u = np.asarray(field, dtype=float)
    if u.shape != mesh.volume.shape:
        raise ValueError(
            f"field shape {u.shape} does not match mesh {mesh.volume.shape}")
    x = float(x_over_h) * config.height_m
    xv = mesh.xc[0]
    nx = mesh.xc.shape[1]
    i = int(np.clip(np.searchsorted(xv, x) - 1, 0, nx - 1))
    x0, x1 = xv[i], xv[(i + 1) % nx]
    t = 0.0 if x1 == x0 else (x - x0) / (x1 - x0)
    column = (1.0 - t) * u[:, i] + t * u[:, (i + 1) % nx]
    y = ((1.0 - t) * mesh.yc[:, i] + t * mesh.yc[:, (i + 1) % nx]) / config.height_m
    return y, column / config.bulk_velocity


def station_assignment_score(field, config=None, y_over_h: float = 0.5,
                             period: float = 9.0) -> dict:
    """Which station map does the reference's sampling correspond to?

    The wall heights cannot answer this: the hill contour is symmetric about
    ``L/2``, so grading the files with ``x -> L - x`` gives the same residual as
    the direct map (:func:`mirrored_assignment_residual` shows this as a
    negative result).  The *flow* is not symmetric, so the assignment is scored
    here on the acceptance line ``y/H = 0.5``: the computed values are compared
    with the reference under the direct map and under the mirrored one, and the
    better agreement identifies the map.
    """
    stations = [float(x) for x in ref.STATIONS if ref.station_covers_y05(x)]
    direct, mirrored = [], []
    for station in stations:
        target = ref.profile_at(station, y_over_h)
        y_d, u_d = field_profile(field, station, config)
        y_m, u_m = field_profile(field, period - station, config)
        direct.append((target, float(np.interp(y_over_h, y_d, u_d))))
        mirrored.append((target, float(np.interp(y_over_h, y_m, u_m))))

    def _score(pairs):
        ref_v = np.array([p[0] for p in pairs], dtype=float)
        got_v = np.array([p[1] for p in pairs], dtype=float)
        rms = float(np.sqrt(np.mean((got_v - ref_v) ** 2)))
        if ref_v.size > 1 and np.std(ref_v) > 0 and np.std(got_v) > 0:
            corr = float(np.corrcoef(ref_v, got_v)[0, 1])
        else:
            corr = float("nan")
        return {"rms": rms, "correlation": corr, "values": got_v.tolist(),
                "reference": ref_v.tolist()}

    d, m = _score(direct), _score(mirrored)
    return {
        "stations": stations,
        "direct": d,
        "mirrored": m,
        "direct_wins": bool(d["rms"] < m["rms"]),
        "note": ("a field that is itself mirror symmetric cannot discriminate; "
                 "the archived laminar field is not (its recirculation sits "
                 "downstream of the crest)"),
    }


def reynolds_bias_from_deficit(deficit_over_flux: float,
                               band_height: float = BAND_HI - BAND_LO,
                               top_minus_crest: float = CHANNEL_TOP - HILL_HEIGHT
                               ) -> float:
    """Closed form of the mislabel: ``bias = band/cap / (1 - delta/Q) - 1``.

    With ``u_b_code = (Q - delta)/band`` and ``u_b_ref = Q/cap``::

        Re_ref / Re_code - 1 = band/cap / (1 - delta/Q) - 1

    ``band`` and ``cap`` are passed in because the solver's band
    (``[H, H + 2.035 H]``) and its channel top (``3.036 H``) differ by ``1e-3 H``
    -- itself a (harmless) inconsistency that the formula exposes rather than
    hides.  Kept as a function so the discrete code path can be checked against
    an independent formula, and so a reference-derived deficit converts without
    building a mesh.
    """
    d = float(deficit_over_flux)
    if not d < 1.0:
        raise ValueError("deficit/flux must be < 1 (the band must carry flux)")
    if band_height <= 0.0 or top_minus_crest <= 0.0:
        raise ValueError("band height and channel top must be positive")
    return (float(band_height) / float(top_minus_crest)) / (1.0 - d) - 1.0


def audited_reynolds(bias: float, labelled: float) -> float:
    """``Re`` in the reference's convention given the labelled value and bias.

    The solver fixes ``u_b_code = 1`` with ``nu = 1/Re_labelled``; the same field
    carries ``u_b_ref = u_b_code (1 + bias)``, hence ``Re_ref = Re_labelled
    (1 + bias)``.  (Metrology does not change the viscosity, only the label.)
    """
    if not math.isfinite(bias) or bias <= -1.0:
        raise ValueError("bias must be finite and > -1")
    return float(labelled) * (1.0 + float(bias))


def verdict(table) -> dict:
    """Fold a table of audited runs into one statement.

    ``table`` rows need ``tag``, ``reynolds_labelled``,
    ``relative_reynolds_bias`` and (optionally) ``x_r_over_H`` and
    ``x_r_error``.  The verdict answers the question the audit exists for: can
    the definition band explain the published reattachment error?
    """
    rows = [row for row in table if row.get("bulk") is not None]
    if not rows:
        return {"status": "no_usable_fields", "n_rows": 0}
    biases = [abs(row["bulk"]["relative_reynolds_bias"]) for row in rows]
    worst = max(biases)
    worst_row = rows[int(np.argmax(biases))]
    errors = [abs(row["x_r_error"]) for row in rows
              if row.get("x_r_error") is not None and math.isfinite(row["x_r_error"])]
    largest_xr_error = max(errors) if errors else None
    return {
        "n_rows": len(rows),
        "worst_abs_relative_reynolds_bias": float(worst),
        "worst_row": worst_row.get("tag"),
        "largest_x_r_relative_error": largest_xr_error,
        "bias_is_solution_dependent": bool(
            max(biases) - min(biases) > 1.0e-4),
        "bias_can_explain_x_r_error": bool(
            largest_xr_error is not None and worst >= largest_xr_error),
        "status": ("metrology_defect_confirmed_bounded"
                   if worst < 0.05 else "metrology_defect_large"),
    }