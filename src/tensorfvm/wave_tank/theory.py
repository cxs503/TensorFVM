"""Wave theory used by the time-domain numerical wave tank (NWT).

Everything here is closed-form or a small scalar root solve, so it plays the
role of the "target" the CFD tank is measured against:

* :func:`dispersion_k` -- finite-depth linear dispersion ``w^2 = g k tanh(k h)``;
* :func:`airy_kinematics` -- first-order surface elevation + orbital velocity,
  the wave that the relaxation zone imposes;
* :func:`stokes2_elevation` -- second-order (Stokes II) bound harmonic added to
  the elevation, i.e. the "Stokes-II target wave" of the campaign node;
* :func:`harmonic_fit` -- least-squares first-harmonic amplitude/phase of a time
  series (used by every wave probe);
* :func:`three_probe_wavenumber` -- ``k`` recovered from the spatial phase of
  three probes, to be compared with :func:`dispersion_k`;
* :func:`goda_two_point` -- incident/reflected separation from two gauges
  (Goda & Suzuki 1976), giving the reflection coefficient ``C_R``;
* :func:`added_resistance_energy` -- period-mean horizontal force, the
  Gerritsma-Meijers (1981) energy/momentum route to ``R_aw``;
* :func:`rao_from_series` -- RAO from a measured first harmonic.

Conventions
-----------
``x`` runs along the tank, ``y`` is vertical *upwards* with the still water level
at ``y = 0``.  Wave phase is ``theta = k x - w t``, so a probe at fixed ``x``
sees ``eta ~ a cos(k x - w t)``.
"""
from __future__ import annotations

import math

import numpy as np

__all__ = [
    "GRAVITY",
    "dispersion_k",
    "phase_speed",
    "airy_kinematics",
    "stokes2_elevation",
    "harmonic_fit",
    "three_probe_wavenumber",
    "goda_two_point",
    "added_resistance_energy",
    "rao_from_series",
    "compute_added_resistance_gerritsma_meijers",
]

GRAVITY = 9.80665


def dispersion_k(omega: float, depth: float, g: float = GRAVITY,
                 tol: float = 1e-14, max_iter: int = 200) -> float:
    """Finite-depth wavenumber from ``w^2 = g k tanh(k h)``.

    Bracketed Newton: the root is bracketed in ``[1e-12, k_hi]`` where ``k_hi``
    is grown until ``g k tanh(k h)`` exceeds ``w^2``, so the iteration cannot
    overshoot the way a deep-water start does in the shallow limit
    (``w -> 0``, where the deep-water guess is off by orders of magnitude).
    """
    if omega <= 0.0:
        raise ValueError("omega must be positive")
    if depth <= 0.0:
        raise ValueError("depth must be positive")

    def f(k):
        return g * k * math.tanh(k * depth) - omega * omega

    lo = 1.0e-12
    hi = max(omega * omega / g, 1.0 / depth)
    while f(hi) < 0.0:
        hi *= 2.0
        if hi > 1.0e12:
            raise RuntimeError("dispersion bracket failed")
    k = omega / math.sqrt(g * depth) if omega * omega * depth / g < 0.1 else omega * omega / g
    k = min(max(k, lo), hi)
    for _ in range(max_iter):
        fk = f(k)
        if fk > 0.0:
            hi = k
        else:
            lo = k
        th = math.tanh(k * depth)
        df = g * th + g * k * depth * (1.0 - th * th)
        step = fk / df if df > 0 else 0.0
        k_new = k - step
        if not (lo < k_new < hi):
            k_new = 0.5 * (lo + hi)
        if abs(k_new - k) <= tol * max(k, 1e-12):
            k = k_new
            break
        k = k_new
    return k


def phase_speed(omega: float, depth: float, g: float = GRAVITY) -> float:
    """Phase speed ``c = w / k`` (m/s)."""
    return omega / dispersion_k(omega, depth, g)


def airy_kinematics(x, y, t: float, amp: float, omega: float, k: float,
                    depth: float, g: float = GRAVITY, phase0: float = 0.0):
    """First-order (Airy) wave: elevation and orbital velocity.

    ``x`` is horizontal (m, measured from the generation origin), ``y`` is
    elevation *above the still water level* (m, can be negative).  Returns
    ``(eta, u, v)``.  The depth factor uses ``cosh(k(y+h))`` so the solution is
    valid for finite depth and reduces to ``exp(k y)`` as ``k h -> inf``.
    """
    th = k * np.asarray(x, dtype=float) - omega * t + phase0
    kh = k * depth
    sh = math.sinh(kh)
    eta = amp * np.cos(th)
    u = amp * omega * np.cosh(k * (np.asarray(y, dtype=float) + depth)) / sh * np.cos(th)
    v = amp * omega * np.sinh(k * (np.asarray(y, dtype=float) + depth)) / sh * np.sin(th)
    return eta, u, v


def stokes2_elevation(x, t: float, amp: float, omega: float, k: float,
                      depth: float, order2: bool = True, phase0: float = 0.0):
    """Stokes-II free-surface elevation (second-order bound harmonic).

    ``eta = a cos(theta) + (k a^2/4) (3 - s^2)/s^3 cos(2 theta)`` with
    ``s = tanh(k h)``; the deep-water limit ``s -> 1`` recovers the textbook
    ``(k a^2 / 2) cos 2 theta``.  With ``order2=False`` the Airy elevation is
    returned (used for the first-harmonic-only control case).
    """
    th = k * np.asarray(x, dtype=float) - omega * t + phase0
    s = math.tanh(k * depth)
    eta = amp * np.cos(th)
    if order2:
        eta = eta + (k * amp * amp / 4.0) * (3.0 - s * s) / s ** 3 * np.cos(2.0 * th)
    return eta


def harmonic_fit(t, signal, omega: float):
    """Least-squares first harmonic of ``signal(t)`` at frequency ``omega``.

    Returns ``(amp, phase, mean, residual)`` with
    ``signal ~ mean + Re[ A exp(-i omega t) ]``, ``amp = |A|``,
    ``phase = arg(A)`` (rad).  Unlike a raw FFT this handles a non-integer
    number of periods; ``residual`` is the relative rms of the unfitted part,
    a direct measure of how periodic/steady the series is.
    """
    t = np.asarray(t, dtype=float)
    s = np.asarray(signal, dtype=float)
    c = np.cos(omega * t)
    sn = np.sin(omega * t)
    A = np.column_stack([np.ones_like(t), c, sn])
    coef, *_ = np.linalg.lstsq(A, s, rcond=None)
    mean, ac, as_ = coef
    amp = math.hypot(ac, as_)
    phase = math.atan2(as_, ac)
    resid = s - A @ coef
    denom = np.sqrt(np.mean((s - s.mean()) ** 2))
    return float(amp), float(phase), float(mean), float(
        np.sqrt(np.mean(resid ** 2)) / denom if denom > 0 else 0.0)


def three_probe_wavenumber(x_probe, amp, phase):
    """Wavenumber from three probe phases (dispersion-relation check).

    Each probe ``i`` measures ``A_i exp(i k x_i)`` (waves travelling +x), so
    ``phase_i = k x_i + const``; a straight-line fit over the three probes gives
    ``k``.  The spacings must satisfy ``|k dx| < pi`` for the phase to be
    unambiguous; the caller is responsible for choosing them.
    """
    x = np.asarray(x_probe, dtype=float)
    ph = np.unwrap(np.asarray(phase, dtype=float))
    A = np.column_stack([x, np.ones_like(x)])
    coef, *_ = np.linalg.lstsq(A, ph, rcond=None)
    return float(coef[0]), float(coef[1])


def goda_two_point(x1: float, x2: float, h1: complex, h2: complex, k: float):
    """Goda two-point incident/reflected separation.

    With ``eta = A_I exp(i(k x - w t)) + A_R exp(-i(k x - w t))`` the two
    complex gauge amplitudes satisfy a 2x2 linear system; solving it gives
    ``(A_I, A_R)`` exactly for a two-component wave field.  ``k`` must be the
    *incident* wavenumber (from the dispersion relation) and the spacing
    ``k (x2 - x1)`` should avoid ``n pi`` where the system is singular.
    """
    kdx = k * (x2 - x1)
    if abs(math.sin(kdx)) < 0.05:
        raise ValueError(f"gauge spacing too close to singular: k dx = {kdx:.4f}")
    M = np.array([[complex(np.exp(1j * k * x1)), complex(np.exp(-1j * k * x1))],
                  [complex(np.exp(1j * k * x2)), complex(np.exp(-1j * k * x2))]])
    aI, aR = np.linalg.solve(M, np.array([h1, h2], dtype=complex))
    return complex(aI), complex(aR)


def added_resistance_energy(force_series, t, period: float, skip_periods: float = 0.0):
    """Period-mean horizontal force = added resistance (Gerritsma-Meijers).

    ``R_aw = -(1/T) int_T F_x dt`` over an integer number of periods after the
    transient.  ``force_series`` may be the body pressure integral (near field)
    or the closed control-surface momentum flux; both are the same statement of
    momentum balance and the campaign node reports both.

    Returns ``(raw_aw, se)``: the mean force and the standard error of that mean
    over the retained periods (a direct measure of how steady the drift force
    is -- a drift force computed from one period is not evidence).
    """
    t = np.asarray(t, dtype=float)
    f = np.asarray(force_series, dtype=float)
    keep = t >= skip_periods * period
    t, f = t[keep], f[keep]
    n_per = max(1, int(round((t[-1] - t[0]) / period)))
    t_end = t[0] + n_per * period
    sel = t <= t_end
    t, f = t[sel], f[sel]
    raw = -float(np.trapezoid(f, t) / (t[-1] - t[0]))
    per_means = []
    for i in range(n_per):
        s = (t >= t[0] + i * period) & (t <= t[0] + (i + 1) * period)
        if s.sum() > 2:
            per_means.append(-float(np.trapezoid(f[s], t[s]) / (t[s][-1] - t[s][0])))
    se = float(np.std(per_means, ddof=1) / math.sqrt(len(per_means))) if len(per_means) > 2 else float("nan")
    return raw, se


def rao_from_series(t, signal, omega: float, wave_amp: float, scale: float = 1.0):
    """Motion RAO from a time series: ``RAO = amp(signal) / (scale * wave_amp)``.

    ``scale = 1`` for a translation (heave) and ``scale = k`` for a rotation
    (pitch, so the RAO is normalised by the wave slope ``k a``).
    """
    amp, phase, mean, resid = harmonic_fit(t, signal, omega)
    return amp / (scale * wave_amp), phase, mean, resid


def compute_added_resistance_gerritsma_meijers(
    wave_amplitude: float,
    omega: float,
    k: float,
    h: float,
    rho: float,
    g: float,
    B: float,
    L: float,
    Cb: float,
    speed: float = 0.0,
) -> float:
    """Gerritsma-Meijers energy method for added resistance in waves.

    Semi-empirical formula based on wave energy flux and ship breadth.
    Reference: Gerritsma & Meijers (1962).

    Parameters
    ----------
    wave_amplitude : float
        Wave amplitude [m]
    omega : float
        Wave frequency [rad/s]
    k : float
        Wavenumber [1/m]
    h : float
        Water depth [m]
    rho : float
        Water density [kg/m^3]
    g : float
        Gravitational acceleration [m/s^2]
    B : float
        Ship breadth [m]
    L : float
        Ship length [m]
    Cb : float
        Block coefficient [-]
    speed : float, optional
        Ship speed [m/s], default 0

    Returns
    -------
    float
        Added resistance [N]
    """
    E = 0.5 * rho * g * wave_amplitude ** 2

    kh = k * h
    cg = 0.5 * (omega / k) * (1 + 2 * kh / math.sinh(2 * kh))

    P = E * cg * B * Cb

    c = omega / k
    delta_R = P / (c + 1e-12)

    if speed > 0:
        Fn = speed / math.sqrt(g * L)
        delta_R *= (1 + 2 * Fn)

    return delta_R
