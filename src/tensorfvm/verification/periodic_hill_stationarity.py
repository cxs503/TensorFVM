"""Round 22: stationarity of the *published* periodic-hill metric.

Anchor: ``CASE-FVM-PERIODIC-HILLS`` (TensorFVM), acceptance ``x_r/H`` vs
Breuer et al. (2009) within 10 %.

Why this module exists (round 22 of ``CASE-FVM-PERIODIC-HILLS``)
----------------------------------------------------------------
Every published ``x_r/H`` in this repository is the **crossing of the time-mean
wall shear** (rounds 17--20, ``x_r/H`` = 8.078 @ Re=700, 8.264 @ Re=2800 on the
closure-free branch).  Round 21 found that the artefact's ``stationarity`` block
carries **two** different stationarity statements that do not agree:

* ``stationarity["x_r_over_H"]`` -- a :func:`stationarity_report` on the
  **per-sample instantaneous** reattachment ``series["x_r_over_H"]``, which is
  produced by the *legacy* round-9 operator.  On the closure-free profiles that
  operator takes a wrong/shallow zero (round 21: the last-frame snapshot's
  backflow endpoint is 5.766 while the ensemble mean is 3.993 -- they cannot
  both be true), so this report's ``ok``/``spread`` **does not** describe the
  published metric.  It reports ``ok=False`` (spread 0.164) for Re=2800.
* ``stationarity["x_r_over_H_by_block"]`` -- the certified round-17 operator
  applied to **per-block mean wall shear**, which *is* the published metric's
  own stationarity.  It gives ``[8.30, 8.24, 8.25, 8.26]`` (spread 0.76 %) for
  Re=2800 -- stationary within the 10 % acceptance band.

The two disagree by construction (different operator, different quantity), and
round 21's lead #1 is exactly this: **make the stationarity of the published
metric explicit and primary**, and **persist the per-block mean shear profiles**
so the block-level ``x_r/H`` can be re-scored by any reader instead of being
trusted as a derived number.

What is here
------------
* :func:`block_stationarity` -- re-score ``x_r/H`` per block from **saved**
  per-block mean wall-shear profiles (``block_tau_b`` in the ``.npz``) with the
  certified round-17 operator, and report block-to-block dispersion as the
  *stationarity of the published metric*.  This is the independent re-scoring
  path round 21 asked for; it never trusts the JSON's derived
  ``x_r_over_H_by_block``.
* :func:`published_metric_stationarity` -- the same dispersion from the JSON's
  ``x_r_over_H_by_block`` (for artefacts produced before round 22, which have no
  saved per-block profiles), so old artefacts can still be scored.
* :func:`reconcile_definitions` -- the round-21 lead #3 reconciliation: the
  **mean-field** definition (crossing of time-mean wall shear) is corroborated
  by the independent ``mean_u < 0`` backflow extent (a different quantity on a
  different data path), while the **ensemble** definition (mean of instantaneous
  reattachment) is not -- so the 50 % gap between the two is explained as a
  legacy-operator artefact, not as two equally valid measurements.

The verdicts here are *measurement-integrity* verdicts (is the published number
stationary? is it corroborated?), not physics-accuracy verdicts: they say nothing
about whether ``x_r/H`` is within 10 % of Breuer 2009 (that is
:mod:`tensorfvm.verification.periodic_hill_lower_re`'s job, and it is FAIL).
"""

from __future__ import annotations

import math

import numpy as np

from .periodic_hill_field_metrics import (
    DEFINITION_ENSEMBLE,
    DEFINITION_MEAN_FIELD,
    backflow_extent,
    definition_spread,
)
from .wall_shear_bubbles import separation_reattachment

__all__ = [
    "block_stationarity",
    "published_metric_stationarity",
    "reconcile_definitions",
    "STATIONARY_TOLERANCE",
]

#: The case's own acceptance tolerance (10 %), used as the stationarity
#: threshold: a block-to-block spread of the published metric inside this band
#: means the time-mean is a stable, single number rather than a moving target.
STATIONARY_TOLERANCE = 0.10

#: Admissibility window of the certified operator (round 5, kept as a reporting
#: filter -- a measured value outside it is reported, not discarded).
_WINDOW = (1.9, 7.1)


def _finite(value) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _block_dispersion(block_xr) -> dict:
    """Block-to-block dispersion of a list of per-block ``x_r/H`` values.

    ``None``/``nan`` entries (empty or non-finite blocks) are excluded; if fewer
    than two blocks are finite the dispersion is ``None`` (not ``inf`` -- an
    unmeasurable spread is reported as such, never as a pass).
    """
    finite = [v for v in block_xr if v is not None and _finite(v)]
    if len(finite) < 2:
        return {"n_finite_blocks": len(finite), "n_blocks": len(block_xr),
                "spread_rel": None, "spread_abs": None, "mean": None,
                "min": None, "max": None, "stationary": None}
    mean = float(np.mean(finite))
    spread_abs = float(max(finite) - min(finite))
    spread_rel = spread_abs / abs(mean) if mean else None
    return {
        "n_finite_blocks": len(finite), "n_blocks": len(block_xr),
        "spread_rel": spread_rel, "spread_abs": spread_abs, "mean": mean,
        "min": float(min(finite)), "max": float(max(finite)),
        "stationary": bool(spread_rel is not None and spread_rel <= STATIONARY_TOLERANCE),
    }


def block_stationarity(block_tau_b, x_over_h, *, window=_WINDOW) -> dict:
    """Re-score ``x_r/H`` per block from **saved** per-block mean wall shear.

    ``block_tau_b`` is the ``(n_blocks, nx)`` array persisted by
    :func:`tensorfvm.verification.periodic_hill_round11.run_time_averaged_case`
    (round 22); ``x_over_h`` is the streamwise abscissa in units of ``H``.  Each
    block's mean shear is scored with the certified round-17 operator, so the
    returned ``x_r_over_H_by_block`` is an *independent* re-scoring of the
    published metric -- it never reads the JSON's derived field.

    Returns a dict with ``x_r_over_H_by_block`` (per-block re-scored values,
    ``None`` for a ``nan``/empty block), ``dispersion`` (the
    :func:`_block_dispersion` summary) and ``status`` (``"stationary"`` when the
    block spread is inside the 10 % band, ``"not_stationary"`` otherwise,
    ``"insufficient_blocks"`` when fewer than two blocks are finite).
    """
    b = np.atleast_2d(np.asarray(block_tau_b, dtype=float))
    x = np.asarray(x_over_h, dtype=float)
    if b.shape[1] != x.shape[0]:
        raise ValueError("block_tau_b and x_over_h must share the streamwise axis")
    block_xr = []
    for k in range(b.shape[0]):
        row = b[k]
        if not np.all(np.isfinite(row)):
            block_xr.append(None)
            continue
        block_xr.append(separation_reattachment(row, x, window=window)["x_r_over_H"])
    disp = _block_dispersion(block_xr)
    status = ("stationary" if disp["stationary"] is True
              else "not_stationary" if disp["stationary"] is False
              else "insufficient_blocks")
    return {"x_r_over_H_by_block": block_xr, "dispersion": disp, "status": status}


def published_metric_stationarity(payload: dict) -> dict:
    """Block-to-block dispersion of the published metric from a JSON artefact.

    Reads ``payload["stationarity"]["x_r_over_H_by_block"]`` (the certified
    operator's per-block values, present in every round-11/20/21 artefact) and
    reports the dispersion.  This is the *fallback* path for artefacts produced
    before round 22 (no saved per-block shear); :func:`block_stationarity` is the
    preferred independent re-scoring path when ``block_tau_b`` is available.
    """
    stat = payload.get("stationarity") or {}
    block_xr = stat.get("x_r_over_H_by_block")
    if block_xr is None:
        return {"x_r_over_H_by_block": None, "dispersion": None,
                "status": "not_recorded"}
    disp = _block_dispersion(block_xr)
    status = ("stationary" if disp["stationary"] is True
              else "not_stationary" if disp["stationary"] is False
              else "insufficient_blocks")
    return {"x_r_over_H_by_block": block_xr, "dispersion": disp, "status": status}


def reconcile_definitions(payload: dict, *, mean_u=None, x_over_h=None,
                          mean_tau_b=None) -> dict:
    """Reconcile the two ``x_r/H`` definitions on one artefact.

    Round 21 lead #3: the mean-field and ensemble definitions differ by ~50 % on
    the closure-free branch.  This function decides which one is the *measured*
    recirculation length by checking each against an **independent** data path --
    the ``mean_u < 0`` backflow extent on the wall-adjacent row (a different
    quantity, no differentiation, no metric terms).

    * ``mean_field_corroborated`` -- the mean-field ``x_r/H`` (crossing of
      time-mean wall shear) agrees with the backflow extent within 3 % (the
      platform's paper-level tolerance): the two independent paths measure the
      same recirculation, so the mean-field is the published metric.
    * ``ensemble_corroborated`` -- the ensemble mean of instantaneous
      reattachment agrees with the backflow extent within 3 %: only then is the
      ensemble a defensible second definition.
    * ``definitions`` -- the :func:`definition_spread` flag (``sensitive`` /
      ``insensitive`` / ``not_measured``).

    ``mean_u``/``x_over_h``/``mean_tau_b`` are read from the artefact's ``.npz``
    when supplied; otherwise the JSON's ``time_averaged``/``horizontal_profile``
    fields are used (with the caveat that the JSON's ``x_r_over_H`` is the
    derived number, not an independent re-scoring).
    """
    stat = payload.get("stationarity") or {}
    block_xr = stat.get("x_r_over_H_by_block")
    mean_field = None
    if mean_tau_b is not None and x_over_h is not None:
        mean_field = separation_reattachment(
            np.asarray(mean_tau_b, dtype=float), np.asarray(x_over_h, dtype=float),
            window=_WINDOW)["x_r_over_H"]
    if mean_field is None or not _finite(mean_field):
        mean_field = (payload.get("time_averaged") or {}).get("x_r_over_H")

    ensemble = None
    series = (payload.get("series") or {}).get("x_r_over_H")
    if series:
        vals = [v for v in series if v is not None and _finite(v)]
        if vals:
            ensemble = float(np.median(vals))

    backflow = None
    if mean_u is not None and x_over_h is not None:
        backflow = backflow_extent(np.asarray(mean_u, dtype=float)[0],
                                   np.asarray(x_over_h, dtype=float))["x_end"]

    def _corroborated(value, ref, tol=0.03):
        if value is None or ref is None or not _finite(value) or not _finite(ref):
            return None
        return bool(abs(float(value) - float(ref)) / abs(float(ref)) <= tol)

    return {
        "mean_field_x_r_over_H": mean_field,
        "ensemble_x_r_over_H": ensemble,
        "backflow_extent_row0": backflow,
        "mean_field_corroborated": _corroborated(mean_field, backflow),
        "ensemble_corroborated": _corroborated(ensemble, backflow),
        "definitions": definition_spread(mean_field, ensemble),
        "mean_field_definition": DEFINITION_MEAN_FIELD,
        "ensemble_definition": DEFINITION_ENSEMBLE,
    }
