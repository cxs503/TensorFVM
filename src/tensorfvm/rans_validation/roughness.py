"""Equivalent sand-grain roughness for the equilibrium channel closure.

Why this module exists
----------------------
:mod:`.channel` solves the equilibrium half-channel with a *hydraulically
smooth* wall: ``wall_values`` imposes the published smooth-wall ``omega`` (or
epsilon) value and nothing else.  Real surfaces -- a ship hull after plating,
coating and fouling -- are rough, and the standard way to carry that into a
RANS closure is the *equivalent sand-grain* height ``k_s`` of Nikuradse (1933),
i.e. the height of a uniform sand roughness that produces the same friction.

This module adds that missing degree of freedom in the smallest form that is
still physically checkable:

* the **wall value of the transported variable** is capped by Wilcox's (1988)
  rough-wall condition

      omega_w = (u_tau^2 / nu) * S_R,   S_R = (50 / k_s^+)^2   k_s^+ < 25
                                        S_R = 100 / k_s^+      k_s^+ >= 25

  which is continuous at ``k_s^+ = 25`` (both branches give ``S_R = 4``) and
  asymptotes to stronger roughness (smaller ``omega_w``) as ``k_s`` grows;
* the **analytic reference** is Nikuradse's fully rough log law
  ``u^+ = (1/kappa) ln(y^+/k_s^+) + 8.5``, i.e. a velocity shift

      du^+ = (B_rough - B_smooth) - (1/kappa) ln(k_s^+)                     (1)

  relative to the smooth law ``u^+ = (1/kappa) ln y^+ + 5.2``.  Equation (1) is
  a *differential* anchor: the offset ``B_rough - B_smooth = 3.3`` carries the
  classical scatter of the sand-grain data (published values run 3.0-3.5), but
  the slope ``d du^+ / d ln k_s^+ = -1/kappa`` does not, so the slope is the
  sharp test and the offset is graded with the data scatter as its tolerance.

Design notes and limits
-----------------------
* Only the ``low_re`` treatments (``komega``, ``sst``) accept roughness.  The
  standard k-epsilon model here is paired with an *equilibrium wall function*
  whose near-wall velocity comes from Spalding's smooth law and whose
  ``epsilon`` wall value is a smooth-wall relation; a rough-wall version needs
  the rough wall function in *both* places and a matching unresolved mass flux,
  which this module does not implement -- the request raises instead of
  silently returning a smooth answer.
* The cap is ``omega_w = min(omega_w_smooth, omega_w_rough)``: the smooth value
  is the upper bound, so a vanishing roughness recovers the smooth wall exactly
  and a large one can only make the wall rougher.  Using ``max`` instead would
  make roughness unreachable (``omega_w_rough`` *decreases* with ``k_s``).
* ``S_R`` was calibrated by Wilcox against the *k-omega* model.  Applying the
  same condition to SST-2003 is an extrapolation of the condition, not a
  calibrated result; the verification driver grades SST separately and says so.
* The condition is a wall-value boundary condition, so it does not resolve the
  roughness sublayer: ``k_s`` must be a roughness height, not a geometry.  A
  mesh whose first cell sits below ``k_s^+`` is *not* resolving the roughness,
  it is only carrying it -- which is the intended use.

Everything here is nondimensional in the channel normalisation of
:mod:`.channel`: half height ``H = 1``, ``u_tau = 1``, ``nu = 1/Re_tau``, so
``k_s^+ = k_s Re_tau`` and ``u_tau^2/nu = Re_tau``.
"""
from __future__ import annotations

import numpy as np

from .models import (WALL_LAW_KAPPA, WALL_TREATMENT, constants_for, implied_kappa,
                     wall_values)

#: von Karman constant used by the repository's wall laws.
KAPPA = WALL_LAW_KAPPA
#: Smooth-wall log-law intercept (Nikuradse 1933 fit of the sand-rough data).
B_SMOOTH = 5.2
#: Fully rough log-law intercept, ``u^+ = (1/kappa) ln(y^+/k_s^+) + B_ROUGH``.
B_ROUGH = 8.5
#: Published spread of the fully rough intercept across the sand-grain data.
B_ROUGH_SPREAD = 0.5
#: Branches of Wilcox's ``S_R``; they meet at ``k_s^+ = 25`` with ``S_R = 4``.
S_R_SWITCH = 25.0
#: ``k_s^+`` above which a smooth answer is a *failure*, not a limit: the
#: transition range of the sand-grain data ends at about 5 wall units, so any
#: larger roughness whose wall value is still the smooth one has a first cell
#: too coarse for the condition to act (see ``rough_wall_value``).
K_S_PLUS_TRANSITION = 5.0

#: Models whose wall treatment admits an equivalent sand-grain height here.
ROUGHNESS_MODELS = tuple(name for name, treatment in WALL_TREATMENT.items() if treatment == 'low_re')


def nikuradse_shift(k_s_plus, *, kappa=KAPPA, b_smooth=B_SMOOTH, b_rough=B_ROUGH):
    """Velocity shift ``u^+_rough - u^+_smooth`` of Nikuradse's rough law (1)."""
    k_s_plus = np.asarray(k_s_plus, dtype=float)
    if not np.isfinite(k_s_plus).all() or np.any(k_s_plus <= 0):
        raise ValueError('k_s^+ must be positive and finite')
    return (b_rough - b_smooth) - np.log(k_s_plus) / kappa


def nikuradse_u_plus(y_plus, k_s_plus, *, kappa=KAPPA, b_rough=B_ROUGH):
    """Fully rough log law ``u^+ = (1/kappa) ln(y^+/k_s^+) + b_rough``."""
    y_plus = np.asarray(y_plus, dtype=float)
    k_s_plus = np.asarray(k_s_plus, dtype=float)
    if not np.isfinite(y_plus).all() or np.any(y_plus <= 0):
        raise ValueError('y^+ must be positive and finite')
    if not np.isfinite(k_s_plus).all() or np.any(k_s_plus <= 0):
        raise ValueError('k_s^+ must be positive and finite')
    return np.log(y_plus / k_s_plus) / kappa + b_rough


def wilcox_s_r(k_s_plus):
    """Wilcox (1988) roughness function ``S_R(k_s^+)``, continuous at 25."""
    k_s_plus = np.asarray(k_s_plus, dtype=float)
    if not np.isfinite(k_s_plus).all() or np.any(k_s_plus <= 0):
        raise ValueError('k_s^+ must be positive and finite')
    return np.where(k_s_plus < S_R_SWITCH, (50.0 / k_s_plus) ** 2, 100.0 / k_s_plus)


def rough_wall_value(model, *, k_s_plus, smooth_value, u_tau_over_nu):
    """Wall value of the transported variable with roughness applied.

    ``u_tau_over_nu`` is ``u_tau/nu`` in the caller's units, so that
    ``omega_w_rough = u_tau_over_nu * u_tau * S_R``; in the channel
    normalisation of :mod:`.channel` it equals ``Re_tau``.
    Returns ``(value, info)`` where ``info`` records which branch was used so a
    result can never claim roughness it did not apply.
    """
    if model not in ROUGHNESS_MODELS:
        raise NotImplementedError(
            'equivalent sand-grain roughness is not implemented for the %r wall '
            'treatment; only the low-Re treatments %s carry it here'
            % (WALL_TREATMENT.get(model, model), ', '.join(ROUGHNESS_MODELS)))
    k_s_plus = float(k_s_plus)
    if not np.isfinite(k_s_plus) or k_s_plus <= 0:
        raise ValueError('k_s^+ must be positive and finite')
    u_tau_over_nu = float(u_tau_over_nu)
    if not np.isfinite(u_tau_over_nu) or u_tau_over_nu <= 0:
        raise ValueError('u_tau/nu must be positive and finite')
    s_r = float(wilcox_s_r(k_s_plus))
    rough = u_tau_over_nu * s_r          # u_tau^2/nu * S_R with u_tau = 1
    if rough >= float(smooth_value) and k_s_plus >= K_S_PLUS_TRANSITION:
        # The smooth cap wins for a roughness that is already past the
        # transitional range.  That happens when the first cell is coarse
        # (the smooth omega grows like 1/d^2), and returning the smooth value
        # would silently delete the roughness the caller asked for.
        raise ValueError(
            'k_s^+ = %.3g is past the transitional range but the smooth wall '
            'value (%.3g) is still below the rough one (%.3g): the first cell '
            'is too far from the wall for the sand-grain condition to act. '
            'Refine the near-wall mesh (y_1^+ <= 1) or drop k_s'
            % (k_s_plus, float(smooth_value), rough))
    value = min(float(smooth_value), rough)
    return value, dict(k_s_plus=k_s_plus, s_r=s_r, omega_w_rough=rough,
                       omega_w_smooth=float(smooth_value), value=float(value),
                       branch='rough' if rough < float(smooth_value) else 'smooth')


def crossover_k_s_plus(model, *, re_tau, first_cell_y_plus):
    """``k_s^+`` at which the rough branch takes over from the smooth value.

    Solved by bisection on ``S_R(k_s^+) = omega_w_smooth / (u_tau/nu)``; the
    left-hand side decreases monotonically, so the crossing is unique.  Reported
    by the verification driver: a case whose ``k_s^+`` sits below it is a
    smooth-wall case whatever ``k_s`` was requested.  The crossing depends on
    the grid through ``omega_w_smooth ~ nu/d^2``, so it is a property of a
    (model, grid) pair -- not of the model alone.
    """
    c = constants_for(model)
    re_tau = float(re_tau)
    smooth_wall_distance = float(first_cell_y_plus) / re_tau
    _, smooth = wall_values(model, smooth_wall_distance, 1.0 / re_tau, c)
    need = float(smooth) / re_tau
    if need > float(wilcox_s_r(np.array(1e-6))):
        return 0.0
    low, high = 1e-6, 1e6
    for _ in range(200):
        mid = np.sqrt(low * high)
        if float(wilcox_s_r(np.array(mid))) > need:
            low = mid
        else:
            high = mid
    return float(np.sqrt(low * high))


def fit_rough_log_law(y_plus, u_plus, *, low, high, k_s_plus):
    """Least-squares ``(kappa_fit, B_fit)`` of ``u^+`` against ``ln(y^+/k_s^+)``.

    Returns ``None`` when the band holds fewer than three points.  ``window``
    records the band so a fitted intercept is never quoted out of context.
    """
    y_plus = np.asarray(y_plus, dtype=float)
    u_plus = np.asarray(u_plus, dtype=float)
    mask = (y_plus >= float(low)) & (y_plus <= float(high))
    if np.count_nonzero(mask) < 3:
        return None
    x = np.log(y_plus[mask] / float(k_s_plus))
    slope, intercept = np.polyfit(x, u_plus[mask], 1)
    residual = u_plus[mask] - (slope * x + intercept)
    predicted = slope * x + intercept
    return dict(
        kappa_fit=float(1.0 / slope) if slope != 0 else float('inf'),
        b_fit=float(intercept),
        slope=float(slope),
        points=int(np.count_nonzero(mask)),
        low=float(low),
        high=float(high),
        rms_u_plus=float(np.sqrt(np.mean(residual ** 2))),
        max_relative=float(np.max(np.abs(residual / predicted))),
    )


def model_kappa(model, constants=None):
    """Log-layer slope implied by the model's own constant set."""
    return implied_kappa(model, constants)
