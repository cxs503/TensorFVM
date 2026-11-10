"""Interface diagnostics for the multi-material surface.

Two different measures of "how wide is the mixed region" are implemented because
they answer different questions and a report that conflates them is worthless:

``interface_position``
    For every transverse column, the ``x`` where the material-2 mass fraction
    crosses ``0.5``.  For a single-mode perturbation this yields the bubble and
    spike fronts directly, hence the amplitude ``a(t) = (x_spike - x_bubble)/2``
    that linear Richtmyer theory predicts.
``mixing_width``
    The width of the ``1 % ... 99 %`` band of the *transversely averaged* mass
    fraction.  This is the quantity mixing-layer models (Brouillette 2002 review,
    alpha-model ``W = 2 alpha A+ du t``) are written for.

``pressure_oscillation``
    Max deviation of the interface pressure from the exact Riemann value in a
    band around a material contact: the metric that decides whether the
    ghost-fluid treatment is doing anything.
"""

from __future__ import annotations

from typing import Dict, Sequence

import torch

__all__ = [
    "interface_position",
    "amplitude_from_columns",
    "mixing_width",
    "transverse_mean",
    "pressure_oscillation",
    "alpha_model_width",
    "linear_width",
    "growth_rate_fit",
    "first_crossing_time",
]


def interface_position(Y: torch.Tensor, xc: torch.Tensor, level: float = 0.5) -> torch.Tensor:
    """``x`` where a column of mass fraction crosses ``level`` (linear interp).

    ``Y`` has shape ``(nx, ny)``.  Columns with no crossing return ``nan``.
    """
    above = Y >= level
    out = torch.full((Y.shape[1],), float("nan"), dtype=torch.float64)
    for j in range(Y.shape[1]):
        col = above[:, j]
        if not bool(col.any()) or bool(col.all()):
            continue
        i = int(torch.nonzero(col)[0])
        y0, y1 = float(Y[i - 1, j]), float(Y[i, j])
        denom = y1 - y0
        t = 0.0 if abs(denom) < 1e-300 else (level - y0) / denom
        out[j] = float(xc[i - 1]) + t * (float(xc[i]) - float(xc[i - 1]))
    return out


def amplitude_from_columns(pos: torch.Tensor) -> Dict[str, float]:
    """Bubble/spike fronts and amplitude from a set of column positions."""
    finite = pos[torch.isfinite(pos)]
    if finite.numel() == 0:
        return {"spike": float("nan"), "bubble": float("nan"),
                "amplitude": float("nan"), "mean": float("nan"),
                "count": 0}
    spike = float(finite.max())
    bubble = float(finite.min())
    return {
        "spike": spike,
        "bubble": bubble,
        "amplitude": 0.5 * (spike - bubble),
        "mean": float(finite.mean()),
        "count": int(finite.numel()),
    }


def transverse_mean(field: torch.Tensor) -> torch.Tensor:
    """Average a ``(nx, ny)`` field over the transverse direction."""
    return field.mean(dim=1)


def mixing_width(mean_profile: torch.Tensor, xc: torch.Tensor,
                 low: float = 0.01, high: float = 0.99) -> Dict[str, float]:
    """Width of the ``low..high`` band of a transversely averaged profile."""
    q = mean_profile
    qmin, qmax = float(q.min()), float(q.max())
    if qmax - qmin < 1e-12:
        return {"width": 0.0, "x_low": float("nan"), "x_high": float("nan")}
    lo = qmin + low * (qmax - qmin)
    hi = qmin + high * (qmax - qmin)

    def cross(level: float) -> float:
        s = q - level
        sign_change = torch.nonzero(s[:-1] * s[1:] <= 0).flatten()
        if sign_change.numel() == 0:
            return float("nan")
        i = int(sign_change[0])
        d = float(s[i + 1] - s[i])
        t = 0.0 if abs(d) < 1e-300 else float(-s[i] / d)
        return float(xc[i]) + t * float(xc[i + 1] - xc[i])

    x_lo, x_hi = cross(lo), cross(hi)
    return {
        "width": abs(x_hi - x_lo) if x_lo == x_lo and x_hi == x_hi else float("nan"),
        "x_low": x_lo,
        "x_high": x_hi,
    }


def alpha_model_width(t: float, a0_post: float, atwood: float, delta_u: float,
                      alpha: float = 0.06, w0: float | None = None) -> float:
    """Brouillette-type alpha model for the post-shock mixing width.

    ``W(t) = W0 + 2 alpha A+ du t`` reproduces the observed *linear-in-time*
    growth of a shock-driven mixing zone once it is self-similar; the
    coefficient ``alpha`` is the calibrated "growth constant" of that model.  The
    linear (impulsive) value ``W0 = 4 a0+`` is used as the initial width.
    """
    if w0 is None:
        w0 = 2.0 * a0_post
    return w0 + 2.0 * alpha * atwood * delta_u * t


def linear_width(t: float, a0_post: float, growth_rate: float,
                 w0: float | None = None) -> float:
    """Linear-regime width ``W = 2 a0+ + 2 (da/dt) t`` (spike minus bubble).

    ``W = spike - bubble = 2a`` for a single-mode perturbation, so the initial
    width is ``2 a0+`` and the growth rate is twice the amplitude growth rate.
    """
    if w0 is None:
        w0 = 2.0 * a0_post
    return w0 + 2.0 * growth_rate * t


def growth_rate_fit(times: Sequence[float], amplitudes: Sequence[float],
                    t_from: float, t_to: float) -> Dict[str, float]:
    """Least-squares slope of ``a(t)`` inside ``[t_from, t_to]``."""
    ts = [t for t, a in zip(times, amplitudes)
          if t_from <= t <= t_to and a == a]  # skips nan
    as_ = [a for t, a in zip(times, amplitudes)
           if t_from <= t <= t_to and a == a]
    if len(ts) < 3:
        return {"slope": float("nan"), "intercept": float("nan"), "n": len(ts)}
    n = len(ts)
    mt = sum(ts) / n
    ma = sum(as_) / n
    num = sum((t - mt) * (a - ma) for t, a in zip(ts, as_))
    den = sum((t - mt) ** 2 for t in ts)
    slope = num / den
    return {"slope": slope, "intercept": ma - slope * mt, "n": n}


def first_crossing_time(times: Sequence[float], values: Sequence[float],
                        threshold: float) -> float:
    """First time a monotone-ish series exceeds ``threshold`` (linear interp)."""
    prev_t = prev_v = None
    for t, v in zip(times, values):
        if v != v:
            continue
        if v >= threshold:
            if prev_v is None or prev_v >= threshold:
                return t
            span = v - prev_v
            frac = 0.0 if abs(span) < 1e-300 else (threshold - prev_v) / span
            return prev_t + frac * (t - prev_t)
        prev_t, prev_v = t, v
    return float("nan")


def pressure_oscillation(p: torch.Tensor, xc: torch.Tensor, x_contact: float,
                         reference_pressure: float, half_width: float) -> Dict[str, float]:
    """Spurious pressure wiggle in a band around a material contact.

    Returns the peak-to-peak and the rms deviation of the pressure in
    ``|x - x_contact| < half_width`` relative to the exact interface pressure.
    """
    mask = (xc - x_contact).abs() < half_width
    band = p[mask]
    if band.numel() == 0:
        return {"pp": float("nan"), "rms": float("nan"), "n": 0}
    dev = band - reference_pressure
    return {
        "pp": float(dev.max() - dev.min()),
        "rms": float(torch.sqrt((dev * dev).mean())),
        "n": int(band.numel()),
    }