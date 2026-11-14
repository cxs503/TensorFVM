"""Martin & Moyce (1952) surge-front anchor in the *paper's own* convention.

Why a second anchor module
--------------------------
:mod:`tensorfvm.vof_phase0.anchors` consumed the digitised Martin & Moyce
surge-front curve on a rescaled axis,

.. math:: t_{\rm old} = t\\sqrt{g/a}, \\qquad x_{\\rm old} = (x_f - a)/a ,

and the dam-break gates built on it therefore read the experiment at the
**wrong time** (a factor ``sqrt(2)`` too late) and with the **initial position
subtracted out** (an ``a``-sized ordinate offset).  Both deviate from the
convention used by the primary source and by the independently maintained
reference implementations:

* Martin & Moyce (1952) plot, and every modern reproduction of their Fig. 3
  uses :math:`T = t\\sqrt{2g/a}`, :math:`X = x_f/a` with :math:`X(0) = 1`
  (``a`` = initial column *width*, column ``a x 2a``);
* the official post-processing script of the open-source Lethe examples
  (``examples/multiphysics/dam-break/dam-break-2d.py``) applies verbatim
  ``time_list = [x * ((2 * g / L1) ** 0.5) for x in time_list]`` and
  ``x_list = [x / L1 for x in x_list]``;
* the same defect was found and retracted independently in the sibling
  TensorLBM benchmark (``benchmarks/pending/dam_break_sc/result.json`` records
  the retraction of the ``T = t sqrt(g/a)`` axis as "a factor sqrt(2) too
  small" and freezes the checkpoints ``X(T) = 1.326 / 2.360 / 3.670`` at
  ``T = 1 / 2 / 2.96``).

Consequences are quantified by :func:`legacy_offset_audit`: the old axis is not
a harmless relabelling, it makes the *reference* 40 % faster than the
experiment at the first gate time (see ``docs/CASE-FVM-DAMBREAK-WET``).

Units
-----
``a`` is the column width in metres, ``g`` the gravitational acceleration in
m/s^2, ``t`` the physical time in seconds, ``x_f`` the absolute surge-front
position measured from the upstream wall in metres.

Data provenance
---------------
``MM_T`` / ``MM_X`` below are the 14-point digitisation of Martin & Moyce
(1952), Phil. Trans. R. Soc. A **244**, 312, Fig. 3 (``h0/a = 2``, smooth bed,
toe of the surge on the floor), read on the (T, X) axes above.  The values are
byte-for-byte the table registered in the sibling repository as
``reference_digitisation_lethe`` (Lethe ``dam-break-2d`` example data, which the
TensorLBM audit cross-checks against the Koshizuka-Oka / Trixi-family
digitisation).  They are experimental readings: a few percent of digitisation
noise is intrinsic and no claim below 1 % should be read as physics.

The last tabulated abscissa is ``T = 2.96``; :func:`mm_front` interpolates
linearly inside that support and **refuses to extrapolate** (it raises), so a
gate can never grade against invented data.
"""

from __future__ import annotations

import math

import numpy as np

__all__ = [
    "MM_CHECKPOINTS_T",
    "MM_T",
    "MM_X",
    "REFERENCE_CHECKPOINTS",
    "legacy_offset_audit",
    "mm_checkpoint_table",
    "mm_front",
    "mm_front_from_physical",
    "mm_time",
    "mm_time_to_physical",
    "mm_x",
    "physical_time",
]

#: Dimensionless time ``T = t sqrt(2 g / a)`` of the digitised front curve.
MM_T = np.array(
    [0.00, 0.41, 0.84, 1.19, 1.43, 1.63, 1.82, 1.97, 2.20, 2.32, 2.50, 2.64,
     2.82, 2.96]
)

#: Front position ``X = x_f / a`` (``X(0) = 1``) digitised from Martin & Moyce
#: (1952) Fig. 3, aspect ratio ``h0/a = 2``, smooth bed.
MM_X = np.array(
    [1.00, 1.11, 1.23, 1.44, 1.67, 1.89, 2.11, 2.33, 2.56, 2.78, 3.00, 3.22,
     3.44, 3.67]
)

#: Predeclared comparison points used by the ``CASE-FVM-DAMBREAK-WET`` gate.
#: Chosen *inside* the digitised support (``T <= 2.96``) and on the tabulated
#: abscissae themselves where possible, so the gate never relies on
#: extrapolation.
MM_CHECKPOINTS_T = (1.00, 1.50, 2.00, 2.50, 2.96)

#: Values obtained by linear interpolation of ``(MM_T, MM_X)``; pinned by
#: ``tests/test_dambreak_mm_anchor.py`` and, for ``T = 1 / 2 / 2.96``, identical
#: to the checkpoints frozen in the sibling TensorLBM benchmark.
REFERENCE_CHECKPOINTS = {
    1.00: 1.326,
    1.50: 1.747, 
    2.00: 2.36,
    2.50: 3.00,
    2.96: 3.67,
}


def mm_time(t: float, a: float, gravity: float = 9.81) -> float:
    """Martin & Moyce dimensionless time ``T = t sqrt(2 g / a)``."""
    return float(t) * math.sqrt(2.0 * gravity / float(a))


def mm_x(x_front: float, a: float) -> float:
    """Martin & Moyce front ordinate ``X = x_f / a`` (``X(0) = 1``)."""
    return float(x_front) / float(a)


def physical_time(t_star: float, a: float, gravity: float = 9.81) -> float:
    """Inverse of :func:`mm_time`."""
    return float(t_star) / math.sqrt(2.0 * gravity / float(a))


def mm_front(t_star: float) -> float:
    """Digitised experimental front ``X(T)``.

    Raises
    ------
    ValueError
        Outside the digitised support ``[0, MM_T[-1]]``: extrapolating an
        experimental curve would manufacture evidence.
    """
    t = float(t_star)
    if t < -1e-12 or t > MM_T[-1] + 1e-12:
        raise ValueError(
            f"T={t:.6g} is outside the digitised Martin & Moyce support "
            f"[0, {MM_T[-1]:.4g}]; extrapolation is not allowed")
    return float(np.interp(t, MM_T, MM_X))


def mm_front_from_physical(t: float, a: float, gravity: float = 9.81) -> float:
    """Experimental front ``x_f`` [m] at physical time ``t`` [s]."""
    return mm_front(mm_time(t, a, gravity)) * float(a)


def mm_time_to_physical(t_star: float, a: float, gravity: float = 9.81) -> float:
    """Alias of :func:`physical_time` (kept for symmetry with :func:`mm_time`)."""
    return physical_time(t_star, a, gravity)


def mm_checkpoint_table(a: float, gravity: float = 9.81) -> list[dict]:
    """The predeclared comparison points, in dimensionless and physical units."""
    rows = []
    for t_star in MM_CHECKPOINTS_T:
        rows.append(
            {
                "t_star": float(t_star),
                "t": physical_time(t_star, a, gravity),
                "x_star": mm_front(t_star),
                "x_front": mm_front(t_star) * float(a),
            }
        )
    return rows


def legacy_offset_audit(a: float, gravity: float = 9.81) -> dict:
    """Quantify the defect of the legacy (:mod:`tensorfvm.vof_phase0.anchors`) axis.

    The legacy anchor is a table of ``(t_old, x_old)``.  Two interpretations are
    tested against the primary-source curve ``(T, X)``:

    ``relabel``
        the legacy numbers *are* ``(T, X)`` -- i.e. only the code labels were
        wrong.  Max deviation inside ``T <= 2.96`` measures the digitisation
        noise between the two readings of Fig. 3.
    ``as_coded``
        the legacy numbers mean what the code consumed, ``(t sqrt(g/a),
        (x_f - a)/a)``.  The equivalent ``X`` at the *same physical time* is
        ``1 + x_old`` evaluated at ``T = sqrt(2) t_old``, i.e. the reference is
        read ``sqrt(2)`` too late.

    The audit also reports the mean front acceleration implied by the first
    legacy segment, ``2 x_f / t^2`` in units of ``g``: a gravity-driven collapse
    cannot exceed ``g H0 / W = 2 g`` at the toe, so a value far above that is a
    physical impossibility flag on the anchor.
    """
    from .anchors import MARTIN_MOYCE_FRONT_T, MARTIN_MOYCE_FRONT_X

    rows = []
    for t_old, x_old in zip(MARTIN_MOYCE_FRONT_T, MARTIN_MOYCE_FRONT_X):
        if t_old > MM_T[-1] + 1e-12:
            break
        ref_relabel = mm_front(t_old)
        t_coded = math.sqrt(2.0) * t_old
        ref_as_coded = mm_front(t_coded) if t_coded <= MM_T[-1] + 1e-12 else float("nan")
        rows.append(
            {
                "t_old": float(t_old),
                "x_old": float(x_old),
                "X_legacy_relabel": float(x_old),
                "X_primary_at_T_legacy": ref_relabel,
                "relabel_error": (float(x_old) - ref_relabel) / ref_relabel,
                "X_as_coded": 1.0 + float(x_old),
                "X_primary_at_sqrt2_T": ref_as_coded,
                "as_coded_error": (
                    (1.0 + float(x_old) - ref_as_coded) / ref_as_coded
                    if np.isfinite(ref_as_coded) else float("nan")),
            }
        )
    # First non-trivial legacy segment: (t_old, x_old) = (0.5, 0.75).
    t_seg = 0.5
    travel = 0.75 * float(a)
    accel_g = 2.0 * travel / (t_seg / math.sqrt(gravity / a)) ** 2 / gravity
    return {
        "column_width": float(a),
        "gravity": float(gravity),
        "rows": rows,
        # t_old = 0 is excluded: there the legacy table's 0 collides with the
        # primary ordinate X(0) = 1, i.e. the row measures the very offset the
        # relabel hypothesis is about, not digitisation noise.
        "max_abs_relabel_error": max(
            abs(r["relabel_error"]) for r in rows if r["t_old"] > 0.0),
        "max_abs_as_coded_error": max(
            abs(r["as_coded_error"]) for r in rows
            if np.isfinite(r["as_coded_error"])),
        "legacy_first_segment_acceleration_g": accel_g,
        "physical_acceleration_bound_g": 2.0,
    }