"""Post-processing metrics for the internal-flow ledger study.

All functions are pure and take tensors, so a stored field can be re-graded
without re-running the solver.
"""

from __future__ import annotations

import math

import torch


def zero_crossing(x: torch.Tensor, values: torch.Tensor, start: int = 0) -> float:
    """First sign change of ``values`` at or after ``start``, linearly interpolated."""
    for i in range(max(start, 0), len(values) - 1):
        a, b = float(values[i]), float(values[i + 1])
        if a == 0.0:
            return float(x[i])
        if a * b < 0.0:
            xa, xb = float(x[i]), float(x[i + 1])
            return xa + (xb - xa) * abs(a) / (abs(a) + abs(b))
    return float("nan")


def wall_tangential_velocity(u: torch.Tensor, wall_row: int = 0) -> torch.Tensor:
    """Tangential velocity of the face row adjacent to a horizontal wall."""
    return u[0, :] if wall_row == 0 else u[-1, :]


def wall_shear(u: torch.Tensor, dy: float, nu: float, rho: float = 1.0,
               wall_row: int = 0) -> torch.Tensor:
    """Wall shear stress ``tau_w = rho nu du/dn`` with a half-cell wall distance.

    Positive ``tau_w`` means the wall drags the fluid in the +x direction (the
    bottom wall of a +x flow), which is exactly the sign convention used by the
    momentum ledger.
    """
    ut = wall_tangential_velocity(u, wall_row)
    tau = 2.0 * rho * nu * ut / dy
    return tau if wall_row == 0 else -tau


def reattachment_length(u: torch.Tensor, fluid: torch.Tensor, dx: float, dy: float,
                        nu: float, *, start_face: int, scale: float = 1.0,
                        rho: float = 1.0) -> dict[str, float]:
    """Reattachment of the primary recirculation on the bottom wall.

    Two routes are evaluated so that post-processing is self-checked:

    ``shear``  first sign change of the wall shear ``tau_w(x)`` downstream of
               ``start_face`` (the physically standard definition),
    ``velocity`` first sign change of the wall-adjacent tangential velocity
               (identical in the continuum, different interpolation weights in
               the discrete field).
    """
    tau = wall_shear(u, dy, nu, rho, wall_row=0)
    ut = wall_tangential_velocity(u, wall_row=0)
    nx = int(fluid.shape[1])
    x = torch.arange(nx + 1, dtype=torch.float64) * dx
    rows = fluid[0, :]
    idx = torch.arange(start_face, nx + 1)
    idx = idx[idx < nx]
    keep = [int(i) for i in idx if bool(rows[min(int(i), nx - 1)])]
    if not keep:
        return {"shear": float("nan"), "velocity": float("nan"), "x": [], "tau": [],
                "min_tau": float("nan")}
    sel = torch.tensor(keep, dtype=torch.long)
    xs = x[sel]
    ts = tau[sel]
    us = ut[sel]
    neg_t = torch.nonzero(ts < 0.0).flatten()
    neg_u = torch.nonzero(us < 0.0).flatten()
    r_t = zero_crossing(xs, ts, start=int(neg_t[-1])) if neg_t.numel() else float("nan")
    r_u = zero_crossing(xs, us, start=int(neg_u[-1])) if neg_u.numel() else float("nan")
    return {"shear": r_t / scale, "velocity": r_u / scale,
            "x": (xs / scale).tolist(), "tau": ts.tolist(),
            "min_tau": float(ts.min()) if ts.numel() else float("nan")}


def wall_nusselt(T: torch.Tensor, dx: float, hot_value: float, cold_value: float,
                 delta_t: float = 1.0, length: float = 1.0) -> dict[str, float]:
    """Wall Nusselt numbers of a differentially heated cavity.

    ``order2`` uses the three-point one-sided derivative
    ``Nu = (8 T_w - 9 T_0 + T_1) / (3 h dT) * L`` which is second-order accurate on
    the wall (the cell centres are at ``h/2`` and ``3h/2``); ``order1`` is the
    half-cell gradient ``2 (T_w - T_0) / (h dT) * L`` used by the earlier
    :mod:`tensorfvm.inc_bench` surface and kept here for a like-for-like
    comparison.
    """
    h = dx
    hot2 = (8.0 * hot_value - 9.0 * T[:, 0] + T[:, 1]) / (3.0 * h * delta_t) * length
    # mirrored one-sided derivative: dT/dx at the cold wall is +, so the mirror of
    # the hot-wall expression carries a minus sign
    cold2 = -(8.0 * cold_value - 9.0 * T[:, -1] + T[:, -2]) / (3.0 * h * delta_t) * length
    hot1 = 2.0 * (hot_value - T[:, 0]) / (h * delta_t) * length
    cold1 = 2.0 * (T[:, -1] - cold_value) / (h * delta_t) * length
    return {
        "nusselt_hot": float(hot2.mean()), "nusselt_cold": float(cold2.mean()),
        "nusselt_mean": 0.5 * float(hot2.mean() + cold2.mean()),
        "nusselt_hot_order1": float(hot1.mean()),
        "nusselt_cold_order1": float(cold1.mean()),
        "energy_balance_rel": abs(float(hot2.mean()) - float(cold2.mean()))
                              / max(abs(float(hot2.mean())), 1e-30),
        "profile_hot": hot2.tolist(),
    }


def streamfunction(u: torch.Tensor, v: torch.Tensor, dy: float,
                   fluid: torch.Tensor) -> torch.Tensor:
    """``psi = int_0^y u dy`` for a closed box (bottom wall is a streamline)."""
    uc = 0.5 * (u[:, :-1] + u[:, 1:])
    psi = torch.zeros_like(uc)
    running = torch.zeros_like(uc[0])
    for i in range(uc.shape[0]):
        running = running + uc[i] * dy
        psi[i] = running
    return torch.where(fluid, psi, torch.zeros_like(psi))


def richardson(coarse: float, medium: float, fine: float, ratio: float = 2.0
               ) -> dict[str, float]:
    """Observed order and Richardson extrapolate from three grids.

    ``p = log(|f_c - f_m| / |f_m - f_f|) / log(r)``; the extrapolate is
    ``f_exact = f_f + (f_f - f_m) / (r^p - 1)``.  Non-monotone or zero differences
    return ``nan`` instead of a fabricated order.
    """
    d1 = abs(coarse - medium)
    d2 = abs(medium - fine)
    out = {"difference_coarse_medium": d1, "difference_medium_fine": d2,
           "ratio": ratio, "observed_order": float("nan"),
           "extrapolated": float("nan"), "fine_vs_extrapolated": float("nan")}
    if d1 <= 0.0 or d2 <= 0.0 or not (math.isfinite(d1) and math.isfinite(d2)):
        return out
    p = math.log(d1 / d2) / math.log(ratio)
    out["observed_order"] = p
    rp = ratio ** p
    if not math.isfinite(rp) or abs(rp - 1.0) < 1e-12:
        return out
    exact = fine + (fine - medium) / (rp - 1.0)
    out["extrapolated"] = exact
    out["fine_vs_extrapolated"] = abs(fine - exact) / max(abs(exact), 1e-30)
    return out