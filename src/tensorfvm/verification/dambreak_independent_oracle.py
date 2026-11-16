"""Third-party (independent) oracle and instruments for the Ritter / Stoker dam break.

Why this module exists
----------------------
Every number published by the ``CASE-FVM-DAMBREAK-STOKER`` case pool is graded
against :mod:`tensorfvm.shallow_water.exact`.  That module is at the same time
the *source* of the acceptance values and the *reference implementation*, and
its self-check ``check_stoker_solution`` re-derives the bore jump conditions
with the same formulas it is checking.  A systematic sign/coefficient error in
the closed form would therefore pass both sides of the gate.  This module
supplies

1. an oracle written independently from shallow-water theory -- self-similar
   rarefaction fan + Rankine-Hugoniot bore, the middle-state depth found by
   bisection, no shared code with ``shallow_water.exact``;
2. measurement instruments that do **not** reuse the pool's estimators: depth
   L1, a half-jump shock locator, measured plateau levels, a fixed-level front
   locator, and a prominence-resolved extrema counter;
3. calibration of those instruments (injected shift / plateau scaling / smooth
   wiggle) so that a verdict produced here can be trusted: an instrument that
   cannot see a 5 % shift cannot certify a 5 % gate.

Consumption rule: :func:`grade_case` drives the production solver
(:class:`tensorfvm.shallow_water.swe1d.ShallowWater1D`, the code under test)
and grades it with the independent oracle.  Nothing here imports from
:mod:`tensorfvm.shallow_water.exact` -- :func:`oracle_crosscheck` imports it
lazily *only* to compare the two oracles against each other.

References: Ritter (1892) Z. Ver. Deutsch. Ing. 36:947; Stoker (1957) *Water
Waves*, ch. 10; Toro (2009) ch. 6 for the shallow-water Riemann structure.
"""

from __future__ import annotations

import math

import numpy as np

__all__ = [
    "RITTER_REFERENCE",
    "STOKER_REFERENCE",
    "STANDARD_CASES",
    "ritter_reference",
    "stoker_constants_independent",
    "stoker_reference",
    "rankine_hugoniot_residuals",
    "rarefaction_invariant",
    "oracle_crosscheck",
    "run_profile",
    "depth_l1_relative",
    "half_jump_shock_position",
    "plateau_levels",
    "overshoot_metrics",
    "local_extrema",
    "level_crossing",
    "detection_power_shift",
    "detection_power_step",
    "wiggle_detection_threshold",
    "grade_case",
]

#: reference strings kept next to the numbers they own
RITTER_REFERENCE = "Ritter (1892) Z. Ver. Deutsch. Ing. 36:947-954"
STOKER_REFERENCE = "Stoker (1957) Water Waves, ch. 10 (bore on a wet bed)"

#: The two cases of the pool, as plain data (no solver, no oracle import).
STANDARD_CASES = {
    "ritter-dry": {
        "kind": "dry", "h0": 1.0, "h1": 0.0, "gravity": 9.81,
        "x_dam": 0.0, "x_min": -4.0, "x_max": 6.0, "t_star": 2.0,
        "reference": RITTER_REFERENCE,
    },
    "stoker-wet": {
        "kind": "wet", "h0": 1.0, "h1": 0.5, "gravity": 9.81,
        "x_dam": 0.0, "x_min": -4.0, "x_max": 6.0, "t_star": 2.0,
        "reference": STOKER_REFERENCE,
    },
}


# --------------------------------------------------------------------------- #
# the independent oracle
# --------------------------------------------------------------------------- #
def _bisect(func, lo: float, hi: float, iterations: int = 200) -> float:
    """Plain bisection -- no scipy dependency, machine-precision bracket."""
    f_lo = func(lo)
    f_hi = func(hi)
    if f_lo == 0.0:
        return lo
    if f_hi == 0.0:
        return hi
    if f_lo * f_hi > 0.0:
        raise ValueError(f"root not bracketed: f({lo})={f_lo}, f({hi})={f_hi}")
    for _ in range(iterations):
        mid = 0.5 * (lo + hi)
        f_mid = func(mid)
        if f_mid == 0.0:
            return mid
        if f_lo * f_mid <= 0.0:
            hi, f_hi = mid, f_mid
        else:
            lo, f_lo = mid, f_mid
    return 0.5 * (lo + hi)


def ritter_reference(x, t: float, h0: float, gravity: float = 9.81,
                     x_dam: float = 0.0):
    """Exact Ritter (1892) dry-bed solution ``(h, u)`` at ``(x, t)``.

    Derived here from the self-similar ansatz ``xi = (x - x_dam)/t``: the
    fan carries the invariant ``u + 2 c = 2 c0`` and the parabola
    ``h = (2 c0 - xi)^2 / (9 g)``; ahead of the front (``xi > 2 c0``) the bed
    is dry, behind the tail (``xi < -c0``) the fluid is at rest at ``h0``.
    """
    x = np.asarray(x, dtype=float)
    if t <= 0.0:
        raise ValueError("t must be positive for the self-similar solution")
    c0 = math.sqrt(gravity * h0)
    xi = (x - x_dam) / t
    h = np.zeros_like(xi)
    u = np.zeros_like(xi)
    in_fan = (xi >= -c0) & (xi <= 2.0 * c0)
    h[in_fan] = (2.0 * c0 - xi[in_fan]) ** 2 / (9.0 * gravity)
    u[in_fan] = (2.0 / 3.0) * (xi[in_fan] + c0)
    left = xi < -c0
    h[left] = h0
    return h, u


def stoker_constants_independent(h0: float, h1: float, gravity: float = 9.81) -> dict:
    """Closed-form Stoker (wet-bed) states from the jump conditions.

    The middle state ``(h2, u2)`` is where the left rarefaction
    (``u2 = 2 (c0 - c2)``, Riemann invariant) meets the bore.  Eliminating the
    bore speed ``s`` between the Rankine-Hugoniot mass and momentum jumps
    (right state at rest, ``h1``) gives the single scalar equation

    .. math::
        4 (\\sqrt{h_0} - \\sqrt{h_2})^2
        = \\frac{(h_2 - h_1)^2 (h_2 + h_1)}{2 h_2 h_1},

    which is monotone in ``h2`` on ``(h1, h0)`` and solved here by bisection.
    """
    if not (h0 > h1 > 0.0):
        raise ValueError(f"Stoker wet bed needs h0 > h1 > 0, got {h0}, {h1}")
    c0 = math.sqrt(gravity * h0)
    c1 = math.sqrt(gravity * h1)

    def residual(h2: float) -> float:
        rarefaction = 4.0 * (math.sqrt(h0) - math.sqrt(h2)) ** 2
        bore = (h2 - h1) ** 2 * (h2 + h1) / (2.0 * h2 * h1)
        return rarefaction - bore

    h2 = _bisect(residual, h1, h0)
    c2 = math.sqrt(gravity * h2)
    u2 = 2.0 * (c0 - c2)
    shock_speed = h2 * u2 / (h2 - h1)          # mass jump with u1 = 0
    return {
        "h0": float(h0), "h1": float(h1), "g": float(gravity),
        "c0": c0, "c1": c1, "c2": c2,
        "h2": h2, "u2": u2,
        "shock_speed": shock_speed,
        "tail_speed": u2 - c2,                 # xi where the fan meets h2
        "head_speed": -c0,                     # xi of the fan head
        "step_ratio": h2 / h1,
        "residual": residual(h2),
    }


def stoker_reference(x, t: float, h0: float, h1: float, gravity: float = 9.81,
                     x_dam: float = 0.0):
    """Exact Stoker solution ``(h, u)``: fan | middle state | bore | right state."""
    x = np.asarray(x, dtype=float)
    if t <= 0.0:
        raise ValueError("t must be positive for the self-similar solution")
    sol = stoker_constants_independent(h0, h1, gravity)
    c0 = sol["c0"]
    xi = (x - x_dam) / t
    h = np.full_like(xi, h1)
    u = np.zeros_like(xi)
    in_fan = (xi >= -c0) & (xi <= sol["tail_speed"])
    h[in_fan] = (2.0 * c0 - xi[in_fan]) ** 2 / (9.0 * gravity)
    u[in_fan] = (2.0 / 3.0) * (xi[in_fan] + c0)
    constant = (xi > sol["tail_speed"]) & (xi < sol["shock_speed"])
    h[constant] = sol["h2"]
    u[constant] = sol["u2"]
    left = xi < -c0
    h[left] = h0
    return h, u


def rankine_hugoniot_residuals(h0: float, h1: float, gravity: float = 9.81) -> dict:
    """Mass and momentum jump residuals at the bore (relative, right state at rest).

    These are computed from the *independent* middle state, so they certify the
    closed form rather than assuming it.
    """
    sol = stoker_constants_independent(h0, h1, gravity)
    h2, u2, s = sol["h2"], sol["u2"], sol["shock_speed"]
    mass = s * (h2 - h1) - h2 * u2
    momentum = (s * h2 * u2
                - (h2 * u2 ** 2 + 0.5 * gravity * h2 ** 2 - 0.5 * gravity * h1 ** 2))
    mass_scale = abs(s * (h2 - h1)) + abs(h2 * u2)
    momentum_scale = (abs(s * h2 * u2) + abs(h2 * u2 ** 2)
                      + abs(0.5 * gravity * (h2 ** 2 - h1 ** 2)))
    return {
        "mass_residual": mass,
        "mass_relative": abs(mass) / mass_scale,
        "momentum_residual": momentum,
        "momentum_relative": abs(momentum) / momentum_scale,
        "rarefaction_invariant_defect": rarefaction_invariant(h0, h2, u2, gravity),
        "middle_state_scalar_residual": abs(sol["residual"]),
    }


def rarefaction_invariant(h0: float, h2: float, u2: float,
                          gravity: float = 9.81) -> float:
    """Defect of ``u + 2 sqrt(g h) = 2 sqrt(g h0)`` between the two fan edges."""
    return abs(u2 + 2.0 * math.sqrt(gravity * h2) - 2.0 * math.sqrt(gravity * h0))


def oracle_crosscheck(t_star: float = 2.0, points: int = 4001) -> dict:
    """Compare this module's oracle against ``tensorfvm.shallow_water.exact``.

    Deliberately a lazy import: the point of the exercise is that the shipped
    pool oracle and this one were built independently, so they must agree to
    round-off on the values the pool publishes (the middle state, the bore
    speed, and the whole ``t* = 2`` profile of both cases).
    """
    from tensorfvm.shallow_water import exact as shipped

    out = {"t_star": float(t_star), "cases": {}}
    for key, case in STANDARD_CASES.items():
        h0, h1, g = case["h0"], case["h1"], case["gravity"]
        t_end = t_star * math.sqrt(h0 / g)
        x = np.linspace(case["x_min"], case["x_max"], points)
        if case["kind"] == "dry":
            h_mine, u_mine = ritter_reference(x, t_end, h0, g, case["x_dam"])
            h_ship, u_ship = shipped.ritter_state(x, t_end, h0, g, case["x_dam"])
            constants = {}
        else:
            h_mine, u_mine = stoker_reference(x, t_end, h0, h1, g, case["x_dam"])
            h_ship, u_ship = shipped.stoker_state(x, t_end, h0, h1, g, case["x_dam"])
            mine = stoker_constants_independent(h0, h1, g)
            theirs = shipped.stoker_constant_state(h0, h1, g)
            constants = {
                name: {
                    "mine": float(mine[name]),
                    "shipped": float(theirs[name]),
                    "relative": (abs(mine[name] - theirs[name])
                                 / abs(theirs[name])),
                }
                for name in ("h2", "u2", "shock_speed", "tail_speed",
                             "step_ratio")
            }
        scale_h = float(np.max(np.abs(h_ship))) or 1.0
        scale_u = float(np.max(np.abs(u_ship))) or 1.0
        out["cases"][key] = {
            "constants": constants,
            "max_depth_diff_scaled": float(np.max(np.abs(h_mine - h_ship)) / scale_h),
            "max_velocity_diff_scaled": float(np.max(np.abs(u_mine - u_ship)) / scale_u),
        }
    out["oracle_agrees"] = all(
        entry["max_depth_diff_scaled"] <= 1e-12
        and entry["max_velocity_diff_scaled"] <= 1e-12
        and all(c["relative"] <= 1e-12 for c in entry["constants"].values())
        for entry in out["cases"].values()
    )
    return out


# --------------------------------------------------------------------------- #
# driving the production solver (the code under test)
# --------------------------------------------------------------------------- #
def run_profile(key: str, n: int = 1600, cfl: float = 0.4, order: int = 2,
                t_star: float | None = None) -> dict:
    """Integrate one pool case with ``ShallowWater1D`` and return the state.

    The solver is the object of the verification; only the *reference* and the
    *instruments* are independent.  ``t_star`` follows the pool convention
    ``t = t* sqrt(h0/g)``.
    """
    from tensorfvm.shallow_water.swe1d import ShallowWater1D

    case = STANDARD_CASES[key]
    h0, h1, g = case["h0"], case["h1"], case["gravity"]
    t_end_star = case["t_star"] if t_star is None else float(t_star)
    t_end = t_end_star * math.sqrt(h0 / g)
    solver = ShallowWater1D(n, case["x_min"], case["x_max"], gravity=g,
                            cfl=cfl, order=order)
    upstream = h0 * (case["x_dam"] - case["x_min"])
    downstream = h1 * (case["x_max"] - case["x_dam"])
    solver.set_state(np.where(solver.x < case["x_dam"], h0, h1))
    volume0 = solver.total_volume()
    steps = solver.advance(t_end)
    return {
        "key": key, "n": int(n), "cfl": float(cfl), "order": int(order),
        "t_star": t_end_star, "t_end": float(t_end), "time": float(solver.time),
        "x": solver.x.copy(), "h": solver.h.copy(), "hu": solver.hu.copy(),
        "dx": float(solver.dx), "steps": int(steps),
        "volume0": float(volume0),
        "volume": float(solver.total_volume()),
        "clipped_cells": int(getattr(solver, "clipped_cells", 0)),
        "dry_flips": int(getattr(solver, "dry_flips", 0)),
        "initial_mass_analytic": float(upstream + downstream),
    }


# --------------------------------------------------------------------------- #
# instruments (none of them reuses the pool's estimators)
# --------------------------------------------------------------------------- #
def depth_l1_relative(h_num, h_ref, dx: float) -> float:
    """``sum |h_num - h_ref| dx / sum h_ref dx`` -- a plain L1 distance."""
    h_num = np.asarray(h_num, dtype=float)
    h_ref = np.asarray(h_ref, dtype=float)
    numerator = float(np.sum(np.abs(h_num - h_ref)) * dx)
    denominator = float(np.sum(np.abs(h_ref)) * dx)
    return numerator / denominator if denominator > 0.0 else float("nan")


def _median(values) -> float:
    values = np.asarray(values, dtype=float)
    return float(np.median(values)) if values.size else float("nan")


def plateau_levels(x, h, x_dam: float, h_up: float,
                   tail_fraction: float = 0.01,
                   quiet_fraction: float = 0.25) -> dict:
    """Measured downstream depth ``h1`` and middle-state depth ``h2`` (wet bed).

    The two states are *measured*, not read from the exact solution:

    * ``h1`` -- median over the far downstream quarter of the domain, which the
      bore has not reached at ``t* = 2``;
    * ``h2`` -- median of the depth over the first fifth of the downstream
      half-domain (the rarefaction fan and the bore both lie outside it at
      ``t* = 2``, so it holds the constant state only);
    * the bore position -- the half-jump crossing found by scanning from the
      right (:func:`half_jump_shock_position`);
    * the plateau window -- the inner half of the interval between the bore and
      the fan tail, the tail being the last cell left of the bore whose depth
      exceeds ``h2 + tail_fraction (h_up - h2)`` (or the dam if there is none).

    The function therefore needs the plateaus to dominate their half-planes,
    which is the case at the pool's grading time ``t* = 2``; it raises rather
    than silently returning nonsense if the bore is not identifiable.
    """
    x = np.asarray(x, dtype=float)
    h = np.asarray(h, dtype=float)
    span = x[-1] - x[0]
    quiet = x > (x[-1] - quiet_fraction * span)
    h1 = _median(h[quiet])
    if not np.isfinite(h1) or h1 <= 0.0:
        raise ValueError("no wet downstream state found")

    downstream = (x > x_dam) & (x < x_dam + 0.2 * span)
    h2_rough = _median(h[downstream])
    if not np.isfinite(h2_rough) or h2_rough <= 1.02 * h1:
        raise ValueError(
            f"bore not identifiable downstream: h2~{h2_rough}, h1~{h1}")
    level = 0.5 * (h2_rough + h1)
    x_shock = half_jump_shock_position(x, h, level=level, x_dam=x_dam)
    if not np.isfinite(x_shock):
        raise ValueError("bore crossing not found")

    # fan tail: rightmost cell left of the bore above the middle state
    margin = tail_fraction * max(h_up - h2_rough, 0.0)
    left = np.nonzero((x < x_shock) & (h > h2_rough + margin))[0]
    x_tail = float(x[left[-1]]) if left.size else float(x_dam)
    lo = x_tail + 0.25 * (x_shock - x_tail)
    hi = x_tail + 0.75 * (x_shock - x_tail)
    window = (x >= lo) & (x <= hi)
    h2 = _median(h[window]) if np.any(window) else float("nan")
    return {
        "h1": h1, "h2": h2, "step_ratio": h2 / h1 if h1 > 0.0 else float("nan"),
        "x_shock": x_shock, "x_fan_tail": x_tail,
        "window": [float(lo), float(hi)], "window_cells": int(np.sum(window)),
        "plateau_window_mask": window,
    }


def half_jump_shock_position(x, h, level: float, x_dam: float = 0.0) -> float:
    """Rightmost crossing of ``h`` through ``level`` (linear interpolation).

    For a bore the depth jumps from the middle state to the still-water depth;
    the mid-level crossing is the standard second-order bore locator and it
    requires no knowledge of the exact shock speed.
    """
    x = np.asarray(x, dtype=float)
    h = np.asarray(h, dtype=float)
    above = h >= level
    idx = np.nonzero(above & (x > x_dam - 0.5 * (x[1] - x[0])))[0]
    if idx.size == 0:
        return float("nan")
    i = int(idx[-1])
    if i + 1 >= x.size:
        return float(x[i])
    h_a, h_b = h[i], h[i + 1]
    if h_a == h_b:
        return float(x[i])
    frac = (level - h_a) / (h_b - h_a)
    return float(x[i] + frac * (x[i + 1] - x[i]))


def level_crossing(x, h, level: float, from_right: bool = True) -> float:
    """Location where ``h`` last crosses ``level`` (linear interpolation)."""
    x = np.asarray(x, dtype=float)
    h = np.asarray(h, dtype=float)
    above = h >= level
    if from_right:
        idx = np.nonzero(above)[0]
        if idx.size == 0:
            return float("nan")
        i = int(idx[-1])
        if i + 1 >= x.size:
            return float(x[i])
        h_a, h_b = h[i], h[i + 1]
        frac = 0.0 if h_a == h_b else (level - h_a) / (h_b - h_a)
        return float(x[i] + frac * (x[i + 1] - x[i]))
    idx = np.nonzero(above)[0]
    if idx.size == 0:
        return float("nan")
    i = int(idx[0])
    if i == 0:
        return float(x[i])
    h_a, h_b = h[i - 1], h[i]
    frac = 0.0 if h_a == h_b else (level - h_a) / (h_b - h_a)
    return float(x[i - 1] + frac * (x[i] - x[i - 1]))


def local_extrema(h, prominence: float) -> dict:
    """Interior local maxima/minima whose prominence exceeds ``prominence``.

    ``prominence`` of an extremum at ``i`` is the smaller of the two adjacent
    depth differences, so a single-cell spike of amplitude ``a`` has prominence
    ``a`` and a smooth wiggle must exceed ``prominence`` per cell to be counted.
    """
    h = np.asarray(h, dtype=float)
    if h.size < 3:
        return {"maxima": 0, "minima": 0, "total": 0, "max_prominence": 0.0}
    left = h[1:-1] - h[:-2]
    right = h[1:-1] - h[2:]
    maxima = np.nonzero((left > 0.0) & (right > 0.0))[0]
    minima = np.nonzero((left < 0.0) & (right < 0.0))[0]
    prom_max = np.minimum(left[maxima], right[maxima]) if maxima.size else np.zeros(0)
    prom_min = np.minimum(-left[minima], -right[minima]) if minima.size else np.zeros(0)
    counted_max = int(np.sum(prom_max > prominence))
    counted_min = int(np.sum(prom_min > prominence))
    largest = max(
        float(np.max(prom_max)) if prom_max.size else 0.0,
        float(np.max(prom_min)) if prom_min.size else 0.0,
    )
    return {"maxima": counted_max, "minima": counted_min,
            "total": counted_max + counted_min, "max_prominence": largest}


def overshoot_metrics(h, mask, h_plateau: float, h_up: float) -> dict:
    """Gibbs-type overshoot of the plateaus (upstream and middle state).

    Reported, not gated: the pool's monotonicity gate is a prominence count,
    which is blind to anything below its threshold.  Here the *amplitude* of
    the residual overshoot inside the constant-state window is measured
    directly, so "no spurious oscillation" can be read quantitatively.
    """
    h = np.asarray(h, dtype=float)
    mask = np.asarray(mask, dtype=bool)
    values = h[mask]
    if values.size == 0:
        return {"h_max": float("nan"), "overshoot": float("nan"),
                "overshoot_fraction": float("nan")}
    h_max = float(np.max(values))
    over = h_max - h_plateau
    return {
        "h_max": h_max,
        "h_min": float(np.min(values)),
        "overshoot": over,
        "overshoot_fraction": over / h_up if h_up > 0.0 else float("nan"),
    }


# --------------------------------------------------------------------------- #
# instrument calibration (detection power)
# --------------------------------------------------------------------------- #
def detection_power_shift(profile: dict, shifts) -> list[dict]:
    """Inject a known bore shift and check the half-jump locator recovers it.

    The profile is re-sampled as ``h(x - shift)`` (a rigid shift of the whole
    wave), the locator is re-run, and the recovered displacement is compared
    with the injected one.  A locator that cannot recover a 5 % shift within
    20 % of it has no business certifying a 5 % gate.
    """
    x = np.asarray(profile["x"], dtype=float)
    h = np.asarray(profile["h"], dtype=float)
    case = STANDARD_CASES[profile["key"]]
    h0, h1, g = case["h0"], case["h1"], case["gravity"]
    sol = stoker_constants_independent(h0, h1, g)
    x_shock_exact = case["x_dam"] + sol["shock_speed"] * profile["time"]
    levels = plateau_levels(x, h, case["x_dam"], h0)
    level = 0.5 * (levels["h2"] + levels["h1"])
    rows = []
    for shift in shifts:
        h_shift = np.interp(x - shift, x, h)
        x_measured = half_jump_shock_position(x, h_shift, level=level,
                                              x_dam=case["x_dam"])
        recovered = x_measured - levels["x_shock"]
        rows.append({
            "injected_shift": float(shift),
            "injected_fraction_of_travel": float(shift / (x_shock_exact - case["x_dam"])),
            "recovered_shift": float(recovered),
            "recovery_relative_error": (abs(recovered - shift) / abs(shift)
                                        if shift else 0.0),
        })
    return rows


def detection_power_step(profile: dict, scales) -> list[dict]:
    """Perturb only the middle state and check the step detector recovers it.

    A *uniform* rescaling of the depth field cannot move ``h2/h1`` (the ratio is
    scale invariant), so the meaningful calibration is a multiplicative
    perturbation of the constant-state window alone: the measured step must move
    by ``(1 + eps)^2`` for a plateau scaled by ``1 + eps`` (both the plateau
    median and the shock locator shift), and the recovered factor is compared
    with the injected one.  If this instrument were insensitive, the published
    step error would be noise rather than a measurement.
    """
    x = np.asarray(profile["x"], dtype=float)
    h = np.asarray(profile["h"], dtype=float)
    case = STANDARD_CASES[profile["key"]]
    h0 = case["h0"]
    base = plateau_levels(x, h, case["x_dam"], h0)
    window = base["plateau_window_mask"]
    rows = []
    for scale in scales:
        h_pert = h.copy()
        h_pert[window] *= scale
        perturbed = plateau_levels(x, h_pert, case["x_dam"], h0)
        recovered_h2 = perturbed["h2"] / base["h2"]
        rows.append({
            "applied_scale": float(scale),
            "step_ratio_base": base["step_ratio"],
            "step_ratio_perturbed": perturbed["step_ratio"],
            "recovered_h2_scale": recovered_h2,
            "recovered_step_factor": perturbed["step_ratio"] / base["step_ratio"],
            "recovery_relative_error": abs(recovered_h2 - scale) / abs(scale),
        })
    return rows


def wiggle_detection_threshold(profile: dict, cells=(2, 4, 8),
                               prominences=(1e-2, 1e-4, 1e-6, 1e-8)) -> list[dict]:
    """Smallest cosine wiggle amplitude each prominence threshold can see.

    A smooth wiggle of ``k`` cells and amplitude ``a`` inserted into the middle
    state produces a per-cell prominence of roughly ``a * (1 - cos(pi/k))``,
    hence the detection floor grows with ``k``.  This quantifies the blind spot
    of a prominence-based "no spurious oscillation" gate.
    """
    x = np.asarray(profile["x"], dtype=float)
    h = np.asarray(profile["h"], dtype=float)
    case = STANDARD_CASES[profile["key"]]
    h0 = case["h0"]
    levels = plateau_levels(x, h, case["x_dam"], h0)
    window = levels["plateau_window_mask"]
    if not np.any(window):
        raise ValueError("empty plateau window")
    idx = np.nonzero(window)[0]
    centre = int(idx[len(idx) // 2])
    rows = []
    for k in cells:
        half = k // 2
        offsets = np.arange(-half, k - half)
        shape = 0.5 * (1.0 - np.cos(2.0 * np.pi * (offsets + 0.5) / k))
        for prominence in prominences:
            # the baseline profile may already carry extrema at this
            # prominence; the threshold is the amplitude that trips the counter
            # *in addition* to that baseline
            base_total = local_extrema(h, prominence)["total"]
            lo, hi = 0.0, 1.0e4 * prominence
            h_test = h.copy()
            for _ in range(80):
                amp = 0.5 * (lo + hi)
                h_test[:] = h
                cells_idx = centre + offsets
                valid = (cells_idx >= 0) & (cells_idx < h.size)
                h_test[cells_idx[valid]] += amp * shape[valid]
                total = local_extrema(h_test, prominence)["total"]
                if total > base_total:
                    hi = amp
                else:
                    lo = amp
            rows.append({
                "cells": int(k), "prominence": float(prominence),
                "baseline_extrema": int(base_total),
                "amplitude_threshold": float(hi),
                "amplitude_over_h0": float(hi / h0),
                "per_cell_prominence_ratio": float(hi * (1.0 - math.cos(math.pi / k))
                                                   / max(prominence, 1e-300)),
            })
    return rows


# --------------------------------------------------------------------------- #
# grading
# --------------------------------------------------------------------------- #
def grade_case(key: str, n: int = 1600, cfl: float = 0.4, order: int = 2,
               tolerance: float = 0.05, t_star: float | None = None) -> dict:
    """Grade one pool case with the independent oracle and instruments.

    Graded quantities (all relative, all ``< tolerance`` required):

    * ``depth_l1`` -- whole-profile L1 distance;
    * ``shock_position`` (wet) / ``front_level`` (dry) -- a *position* metric;
    * ``step_ratio`` (wet) -- measured plateau step ``h2/h1``;
    * ``monotone`` -- interior extrema above ``1e-4 h0`` must be zero.

    Reported but not gated: the overshoot amplitude inside the middle state, the
    stricter extrema counts, and the solver's positivity counters.
    """
    case = STANDARD_CASES[key]
    h0, h1, g = case["h0"], case["h1"], case["gravity"]
    profile = run_profile(key, n=n, cfl=cfl, order=order, t_star=t_star)
    x, h, t = profile["x"], profile["h"], profile["time"]

    if case["kind"] == "dry":
        h_ref, u_ref = ritter_reference(x, t, h0, g, case["x_dam"])
    else:
        h_ref, u_ref = stoker_reference(x, t, h0, h1, g, case["x_dam"])

    result = {
        "case": key, "reference": case["reference"], "n": int(n),
        "cfl": float(cfl), "order": int(order), "t_star": profile["t_star"],
        "t_end": profile["t_end"], "dt_final_steps": profile["steps"],
        "dx": profile["dx"], "tolerance": float(tolerance),
        "depth_l1": depth_l1_relative(h, h_ref, profile["dx"]),
        "momentum_l1": depth_l1_relative(h * profile["hu"] / np.maximum(h, 1e-300),
                                         h_ref * u_ref, profile["dx"]),
        "volume_drift": abs(profile["volume"] - profile["volume0"])
        / max(profile["volume0"], 1e-300),
        "clipped_cells": profile["clipped_cells"],
        "dry_flips": profile["dry_flips"],
        "max_depth": float(np.max(h)),
        "max_abs_velocity": float(np.max(np.abs(profile["hu"] / np.maximum(h, 1e-300)))),
        "initial_mass_analytic": profile["initial_mass_analytic"],
        "volume0": profile["volume0"],
    }

    if case["kind"] == "dry":
        c0 = math.sqrt(g * h0)
        x_front = case["x_dam"] + 2.0 * c0 * t
        levels = {}
        for frac in (0.02, 0.05, 0.1, 0.2):
            level = frac * h0
            xi_level = 2.0 * c0 - 3.0 * math.sqrt(g * level)
            x_ref = case["x_dam"] + xi_level * t
            x_meas = level_crossing(x, h, level)
            levels[f"level_{frac:g}"] = {
                "level": level, "x_measured": x_meas, "x_exact": x_ref,
                "relative_error": abs(x_meas - x_ref) / abs(x_ref - case["x_dam"]),
            }
        wetted = np.nonzero(h > 0.0)[0]
        x_wetted = float(x[wetted[-1]]) if wetted.size else float("nan")
        result["front_level_positions"] = levels
        result["front_level_05_error"] = levels["level_0.05"]["relative_error"]
        result["front_wetted_edge"] = {
            "x_measured": x_wetted, "x_exact": x_front,
            "relative_error": abs(x_wetted - x_front) / (x_front - case["x_dam"]),
        }
        result["front_exact"] = x_front
        result["extrema_1e-4"] = local_extrema(h, 1e-4 * h0)["total"]
        result["extrema_1e-6"] = local_extrema(h, 1e-6 * h0)["total"]
        result["extrema_1e-8"] = local_extrema(h, 1e-8 * h0)["total"]
        result["monotone"] = bool(result["extrema_1e-4"] == 0)
        result["graded"] = {
            "depth_l1": result["depth_l1"],
            "front_level_05": result["front_level_05_error"],
            "monotone": 0.0 if result["monotone"] else 1.0,
        }
    else:
        sol = stoker_constants_independent(h0, h1, g)
        x_shock_exact = case["x_dam"] + sol["shock_speed"] * t
        levels = plateau_levels(x, h, case["x_dam"], h0)
        step_error = abs(levels["step_ratio"] - sol["step_ratio"]) / sol["step_ratio"]
        over = overshoot_metrics(h, levels["plateau_window_mask"], sol["h2"], h0)
        result.update({
            "stoker_independent": {k: float(sol[k]) for k in
                                   ("h2", "u2", "shock_speed", "tail_speed",
                                    "step_ratio")},
            "measured_h1": levels["h1"], "measured_h2": levels["h2"],
            "measured_step_ratio": levels["step_ratio"],
            "step_ratio_error": step_error,
            "shock_position": {
                "x_measured": levels["x_shock"], "x_exact": x_shock_exact,
                "relative_error": abs(levels["x_shock"] - x_shock_exact)
                / (x_shock_exact - case["x_dam"]),
            },
            "plateau_overshoot": over,
            "extrema_1e-4": local_extrema(h, 1e-4 * h0)["total"],
            "extrema_1e-6": local_extrema(h, 1e-6 * h0)["total"],
            "extrema_1e-8": local_extrema(h, 1e-8 * h0)["total"],
            "plateau_window": levels["window"],
        })
        result["monotone"] = bool(result["extrema_1e-4"] == 0)
        result["graded"] = {
            "depth_l1": result["depth_l1"],
            "shock_position": result["shock_position"]["relative_error"],
            "step_ratio": step_error,
            "monotone": 0.0 if result["monotone"] else 1.0,
        }

    result["pass"] = bool(all(value <= tolerance
                             for value in result["graded"].values()))
    result["worst_graded"] = max(result["graded"].values())
    result["profile"] = {"x": x.tolist(), "h": h.tolist(), "h_ref": h_ref.tolist()}
    return result


def negative_control(key: str, n: int = 800, tolerance: float = 0.05) -> dict:
    """Feed the graders a deliberately wrong profile -- they must reject it.

    The wrong profile is built from the exact solution of a *different* case
    (dry bed read as a wet one, or vice versa) so every instrument sees a
    genuinely wrong answer, not a round-off.
    """
    case = STANDARD_CASES[key]
    h0, h1, g = case["h0"], case["h1"], case["gravity"]
    profile = run_profile(key, n=n)
    x, t = profile["x"], profile["time"]
    if case["kind"] == "wet":
        wrong, _ = stoker_reference(x, t, h0, 0.75 * h1, g, case["x_dam"])
    else:
        wrong, _ = ritter_reference(x, t, 0.75 * h0, g, case["x_dam"])
    l1 = depth_l1_relative(profile["h"], wrong, profile["dx"])
    return {"case": key, "depth_l1_vs_wrong_reference": l1,
            "rejected": bool(l1 > tolerance)}