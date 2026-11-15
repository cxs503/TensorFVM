"""Certified separation / reattachment measurement on a periodic wall shear.

Why this module exists (round 17 of ``CASE-FVM-PERIODIC-HILLS``)
----------------------------------------------------------------
The case anchor is the reattachment length ``x_r/H`` of the periodic hill
(Breuer et al. 2009, ``x_r/H = 5.41`` at ``Re = 2800``).  Rounds 5--16 measured
it with :func:`tensorfvm.verification.periodic_hill_round9.zero_crossings_periodic`,
which has two properties that silently corrupt the verdict:

1. **Only rising (negative -> positive) crossings are detected.**  Downstream of
   the crest the wall shear ``tau`` is negative (reversed flow), so a rising
   crossing *is* a reattachment.  The attached -> reversed (falling) crossing,
   i.e. the actual *separation* point, is never detected: the returned
   ``separation`` is the first *rising* crossing below ``lower = 1.9``, which is
   the true separation only when no other rising crossing exists there.  On the
   stored round-9/11/12 profiles this happens to be the case (``x_s/H`` =
   0.40--0.64), but nothing in the operator enforces it.
2. **A hard admissibility window ``[1.9, 7.1]`` turns a measured value into
   NaN.**  ``tau`` on the stored mean profiles has crossings at 7.651 (round 9,
   SA) and 7.682 (round 16, SST patankar).  Both are outside ``7.1``, so
   ``reattachment`` came back ``nan`` and the artefacts recorded
   ``x_r_over_H = null`` -- reported as UNDEFINED / WITHHELD, i.e. as *no
   measurement*, while a number had in fact been measured.  ROUND16.md then
   suggested a grid-refinement ladder to find out "whether UNDEFINED disappears
   with refinement", a question that only existed because of the window.

This module replaces (for new work) that measurement with an operator that

* detects **signed** crossings -- ``-1`` falling (separation), ``+1`` rising
  (reattachment) -- with sub-cell linear interpolation;
* pairs them into **recirculation bubbles**, unwrapping the periodic seam so a
  bubble that starts at ``x/H = 8.2`` and ends at ``0.3`` has length ``1.1``
  instead of a negative or clipped one;
* keeps the admissibility window as a *reporting filter with an explicit
  status* (``"out_of_window"``) that still carries the measured number, so
  "outside the window" can never again be confused with "not measured";
* selects the **primary** bubble by reversed-shear area (the main bubble), and
  reports every bubble so secondary corner vortices are visible rather than
  silently becoming "the" reattachment.

It also provides :func:`tanh_shear_profile`, a synthetic profile with
*prescribed* zeros, so the operator itself can be certified against a known
truth instead of only against the solver it is used to judge.

Conventions
-----------
``x_over_h`` is strictly increasing inside one period, counted from the hill
crest (``x/H = 0``).  ``tau`` is the kinematic wall shear ``tau/rho`` at the
same face centres; ``tau < 0`` means reversed flow.  Separation is a falling
crossing (``+`` -> ``-``), reattachment a rising one (``-`` -> ``+``).
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "STATUS_MEASURED",
    "STATUS_OUT_OF_WINDOW",
    "STATUS_NO_RECIRCULATION",
    "STATUS_ALL_REVERSED",
    "STATUS_NOT_FINITE",
    "signed_crossings",
    "bubble_intervals",
    "separation_reattachment",
    "tanh_shear_profile",
    "period_of",
]

STATUS_MEASURED = "measured"
STATUS_OUT_OF_WINDOW = "out_of_window"
STATUS_NO_RECIRCULATION = "no_recirculation"
STATUS_ALL_REVERSED = "all_reversed"
STATUS_NOT_FINITE = "not_finite"


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def period_of(x_over_h) -> float:
    """Period of a strictly increasing, uniformly spaced periodic abscissa."""
    x = np.asarray(x_over_h, dtype=float)
    if x.size < 3:
        raise ValueError("need at least 3 samples to infer a period")
    steps = np.diff(x)
    if np.any(steps <= 0.0):
        raise ValueError("x_over_h must be strictly increasing inside one period")
    if not np.allclose(steps, steps[0], rtol=1e-9, atol=1e-12):
        raise ValueError("x_over_h must be uniformly spaced (period inference)")
    return float(x[-1] - x[0] + steps[0])


def _validate(tau, x_over_h):
    tau = np.asarray(tau, dtype=float)
    x = np.asarray(x_over_h, dtype=float)
    if tau.ndim != 1 or x.ndim != 1 or tau.shape != x.shape:
        raise ValueError("tau and x_over_h must be 1-D arrays of equal length")
    if tau.size < 3:
        raise ValueError("need at least 3 wall-shear samples")
    return tau, x


def signed_crossings(tau, x_over_h, period=None):
    """All zero crossings of ``tau`` with their direction.

    Returns a list of dicts ``{"x_over_H", "direction", "index"}`` sorted by
    ``x_over_H`` inside one period, where ``direction = -1`` is a falling
    crossing (attached -> reversed, i.e. **separation**) and ``direction = +1``
    a rising one (**reattachment**).

    Exact zeros are attributed deterministically: ``tau[i] == 0`` counts as
    *not negative*, so ``(-, 0)`` is rising at ``x[i+1]`` and ``(0, -)`` is
    falling at ``x[i]``.  A plateau of exact zeros therefore yields one crossing
    per sign change, never a duplicated pair.
    """
    tau, x = _validate(tau, x_over_h)
    period = period_of(x) if period is None else float(period)
    out = []
    n = tau.size
    for i in range(n):
        a = float(tau[i])
        b = float(tau[(i + 1) % n])
        falling = (a >= 0.0) and (b < 0.0)
        rising = (a < 0.0) and (b >= 0.0)
        if not (falling or rising):
            continue
        dx = (x[(i + 1) % n] - x[i]) % period
        t = 0.0 if b == a else -a / (b - a)
        out.append({
            "x_over_H": float(x[i] + t * dx),
            "direction": -1 if falling else 1,
            "index": int(i),
        })
    out.sort(key=lambda c: c["x_over_H"])
    return out


def _reversed_area(x, tau, lo, hi, period):
    """``int clip(-tau, 0, None) dx`` between unwrapped positions ``lo`` < ``hi``.

    Three periodic copies are stacked so both bounds are interior to the
    abscissa even when the bubble crosses the seam; the integral is exact for
    the piecewise-linear interpolant of ``tau``.
    """
    if not hi > lo:
        raise ValueError("need lo < hi for a bubble interval")
    x3 = np.concatenate([x - period, x, x + period])
    g = np.clip(-np.concatenate([tau, tau, tau]), 0.0, None)
    cum = np.concatenate([[0.0], np.cumsum(0.5 * (g[1:] + g[:-1]) * np.diff(x3))])
    if lo < x3[0] or hi > x3[-1]:
        raise ValueError("interval outside three stacked periods")
    return float(np.interp(hi, x3, cum) - np.interp(lo, x3, cum))


def bubble_intervals(tau, x_over_h, period=None):
    """Recirculation bubbles of a periodic wall-shear profile.

    A bubble starts at a falling (separation) crossing and ends at the next
    rising (reattachment) crossing.  ``reattachment`` is **unwrapped**: it may
    exceed the period for a bubble that crosses the periodic seam, and
    ``length = reattachment - separation`` is then still positive.
    ``reversed_area`` is the integral of ``|tau|`` over reversed flow inside the
    bubble and is used to identify the primary bubble.

    Returns ``(bubbles, status)`` where ``status`` is one of
    ``"measured"``, ``"no_recirculation"`` (no reversed flow anywhere) or
    ``"all_reversed"`` (reversed flow over the whole period, no crossing).
    """
    tau, x = _validate(tau, x_over_h)
    period = period_of(x) if period is None else float(period)
    cross = signed_crossings(tau, x, period)
    if not cross:
        if np.all(tau < 0.0):
            return [], STATUS_ALL_REVERSED
        return [], STATUS_NO_RECIRCULATION

    starts = [k for k, c in enumerate(cross) if c["direction"] == -1]
    if not starts:
        # tau < 0 at x[0] and only rising crossings: bubble wraps the seam
        # from the last rising crossing to the first one.
        if not (tau[0] < 0.0 or tau[-1] < 0.0):
            return [], STATUS_NO_RECIRCULATION
        raise ValueError(
            "inconsistent crossing set: reversed flow without a separation "
            "crossing inside the period")
    # rotate so the walk starts at the first separation crossing
    k0 = starts[0]
    walked = cross[k0:] + cross[:k0]
    bubbles = []
    for j in range(0, len(walked), 2):
        fall = walked[j]
        if fall["direction"] != -1:
            break
        if j + 1 >= len(walked):
            break
        rise = walked[j + 1]
        if rise["direction"] != 1:
            break
        lo = fall["x_over_H"]
        hi = rise["x_over_H"]
        if hi <= lo:
            hi += period
        bubbles.append({
            "separation": float(lo),
            "reattachment": float(hi),
            "length": float(hi - lo),
            "reversed_area": _reversed_area(x, tau, lo, hi, period),
            "spans_seam": bool(hi > period),
        })
    if not bubbles:
        return [], STATUS_NO_RECIRCULATION
    return bubbles, STATUS_MEASURED


def separation_reattachment(tau, x_over_h, window=None, period=None,
                            primary="largest_area"):
    """Separation and reattachment of the **primary** bubble.

    Parameters
    ----------
    tau, x_over_h : array_like
        Wall shear and abscissa in units of the hill height, one period.
    window : (lower, upper), optional
        Admissibility window of the case (round 5 used ``(1.9, 7.1)``).  It is a
        *reporting filter*, not a detector: when the primary reattachment falls
        outside it, ``status`` becomes ``"out_of_window"`` and
        ``out_of_window`` is ``True``, but ``reattachment``/``x_r_over_H`` still
        carry the measured number.
    primary : {"largest_area", "first"}
        Which bubble is "the" bubble of the case.  ``"largest_area"`` (default)
        picks the largest reversed-shear area -- the main recirculation bubble,
        which is the quantity ``x_r/H`` refers to; ``"first"`` picks the earliest
        separation after the crest.

    Returns
    -------
    dict with ``status``, ``finite``, ``x_s_over_H``, ``x_r_over_H`` (aliases
    ``separation``/``reattachment``), ``out_of_window``, ``window``, ``bubbles``
    (all of them, with lengths and reversed areas), ``n_bubbles``,
    ``n_crossings`` and ``primary_index``.  A measured value is never replaced
    by ``nan`` because of the window.
    """
    tau, x = _validate(tau, x_over_h)
    ok = bool(np.all(np.isfinite(tau)))
    base = {
        "status": STATUS_MEASURED if ok else STATUS_NOT_FINITE,
        "finite": ok,
        "x_s_over_H": float("nan"),
        "x_r_over_H": float("nan"),
        "separation": float("nan"),
        "reattachment": float("nan"),
        "out_of_window": False,
        "window": None if window is None else [float(window[0]), float(window[1])],
        "bubbles": [],
        "n_bubbles": 0,
        "n_crossings": 0,
        "crossings": [],
        "sign_changes": 0,
        "primary_index": None,
    }
    if not ok:
        return base

    period = period_of(x) if period is None else float(period)
    cross = signed_crossings(tau, x, period)
    # drop-in schema of the legacy operator: ``crossings`` / ``sign_changes``
    # keep the round-9 meaning (rising crossings only) so stored readers do not
    # silently reinterpret the fields; ``all_crossings`` is the signed set.
    base["crossings"] = [c["x_over_H"] for c in cross if c["direction"] == 1]
    base["sign_changes"] = len(base["crossings"])
    base["n_crossings"] = len(cross)
    base["all_crossings"] = [{"x_over_H": c["x_over_H"],
                              "direction": c["direction"]} for c in cross]
    bubbles, status = bubble_intervals(tau, x, period)
    if status != STATUS_MEASURED:
        base["status"] = status
        return base

    if primary == "largest_area":
        idx = max(range(len(bubbles)), key=lambda k: bubbles[k]["reversed_area"])
    elif primary == "first":
        idx = 0
    else:
        raise ValueError("primary must be 'largest_area' or 'first'")

    main = bubbles[idx]
    base.update({
        "status": STATUS_MEASURED,
        "x_s_over_H": main["separation"],
        "x_r_over_H": main["reattachment"],
        "separation": main["separation"],
        "reattachment": main["reattachment"],
        "bubbles": bubbles,
        "n_bubbles": len(bubbles),
        "primary_index": int(idx),
    })
    if window is not None:
        lo, hi = float(window[0]), float(window[1])
        if not (lo <= main["reattachment"] <= hi):
            base["status"] = STATUS_OUT_OF_WINDOW
            base["out_of_window"] = True
    return base


def tanh_shear_profile(x_over_h, zeros, period=9.0, sharpness=6.0, sign=1.0):
    """Synthetic wall shear with *prescribed* zero crossings (certification).

    ``tau(x) = prod_k tanh(sharpness * sin(pi (x - z_k) / period))``, so the sign
    flips exactly at every ``z_k`` (``sin`` crosses zero there for any
    ``sharpness``) and the profile is **periodic by construction**: no periodic
    image of ``x - z_k`` is needed, which matters because taking the nearest
    image of the distance injects a discontinuous jump of ``2 tanh(sharpness *
    period/2)`` at the antipode of every zero -- a spurious sign change that a
    round-17 self-check caught (it put fake crossings at ``z_k + period/2``).

    Parameters
    ----------
    x_over_h : array_like
        Abscissa inside one period.
    zeros : sequence of float
        Prescribed crossing positions inside ``[0, period)``, in increasing
        order.  ``zeros[0]`` is a *separation* (falling) crossing, i.e. the
        profile is attached at ``x = 0``-ish only when the first zero is a
        falling one; use :func:`signed_crossings` to check.
    sharpness : float
        Larger is more step-like; the crossing position is exact for any value.
    sign : float
        Global sign of the profile (``+1`` attached at ``x = 0`` when the first
        zero is falling).  ``-1`` reverses that, which is how a bubble spanning
        the periodic seam is built: ``zeros=[0.3, 8.2], sign=-1`` is reversed
        flow over ``8.2 -> 9.3`` (i.e. across the seam) and attached in
        between.

    Returns
    -------
    (tau, x) arrays of the same length as ``x_over_h``.
    """
    x = np.asarray(x_over_h, dtype=float)
    z = np.asarray(zeros, dtype=float)
    if z.size == 0:
        raise ValueError("need at least one prescribed zero")
    tau = np.ones_like(x)
    for zk in z:
        tau = tau * np.tanh(sharpness * np.sin(np.pi * (x - zk) / period))
    return float(sign) * tau, x