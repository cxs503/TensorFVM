"""Lax (1954) shock tube: case builder, independent exact solution, verification harness.

Why this case
-------------
The Sod problem (``cases.sod``) starts from a gas at rest and produces a strong
right shock (``p*/p_R = 3``).  The Lax problem exercises a different combination
that matters for a density-based solver: a **prescribed inflow on the left**
(``u_L = 0.698``), a strong left rarefaction, a **strong right shock**
(``p*/p_R = 4.32``, ``rho*_R/rho_R = 2.61``) and a contact that separates two
star densities differing by a factor ``3.78``.  Shock, contact and a wide fan are
all active at once, so a HLLC/MUSCL implementation can pass Sod and still smear
the contact or clip the post-shock plateau here -- worth an independent anchor.

Reference for the initial data
------------------------------
Lax, P. D. (1954), *Weak solutions of nonlinear hyperbolic equations and their
numerical computation*, Comm. Pure Appl. Math. **7**, 159--193 -- original data
set.  The same data set is reproduced as the second standard one-dimensional
test problem in Toro (2009), *Riemann Solvers and Numerical Methods for Fluid
Dynamics*, 3rd ed., Wiley, ch. 4 (test-problem data + the exact-solution
construction of Sec. 4.2, which is what ``riemann.ExactRiemann`` implements).

Non-dimensionalisation: ``rho_L = 0.445``, ``p_L = 3.528``; lengths in ``[0, 1]``
with the diaphragm at ``x0 = 0.5``; ``t_end = 0.16`` (Toro's stopping time for
this test), so the four wave features sit at

==========================  ======================================
feature                     exact position ``x = x0 + S t``
==========================  ======================================
left rarefaction head       ``x0 + (u_L - c_L) t``
left rarefaction tail       ``x0 + (u* - c*_L) t``
contact discontinuity       ``x0 + u* t``
right shock                 ``x0 + S_R t``
==========================  ======================================

Zero hand-copy discipline
-------------------------
Every number in the report is produced by this module at run time.  There are
*two* independent exact solutions:

``exact_riemann``
    ``riemann.ExactRiemann`` (Toro's pressure-function iteration, in-repo).
``independent_star_state`` / ``independent_sample``
    a from-scratch re-derivation written here for the sole purpose of checking
    the first one: bisection on the ideal-gas pressure function, Rankine--
    Hugoniot for the shocks, isentropic relations for the fan.  It shares no
    code with ``riemann.py``.

The agreement of the two is the evidence that the reference values are the
exact solution of the stated initial-value problem and not an artefact of one
implementation.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch

from .eos import StiffenedGasMixture
from .riemann import ExactRiemann
from .solver import EulerSolver

__all__ = [
    "LaxConfig",
    "LAX_GAMMA",
    "independent_star_state",
    "independent_sample",
    "exact_riemann",
    "exact_waves",
    "lax",
    "profile_metrics",
    "run_case",
    "sweep",
    "THRESHOLDS",
]

#: specific-heat ratio of the ideal gas filling both chambers
LAX_GAMMA = 1.4

#: tolerance of the acceptance criteria of this benchmark (relative, <2 %)
THRESHOLDS = {
    "plateau_rel_error": 0.02,
    "wave_position_rel_error": 0.02,
    "l1_rel_error": 0.02,
    "independent_exact_rel_error": 1.0e-9,
    "conservation_drift": 5.0e-3,
    #: the grid the percentage criteria are graded on (``leg_lax``: full vs
    #: quick sweep).  They are *not* a convergence-order test: a first-order
    #: scheme brings ``l1[rho]`` under 2 % at nx ~ 1.6e3, so "< 2 %" is only
    #: meaningful together with this grid -- see ``docs/audit-CASE-LAX`` (F1).
    "grading_nx": {"full": 800, "quick": 400},
}


@dataclass(frozen=True)
class LaxConfig:
    """Initial data of the Lax problem (Lax 1954; Toro 2009 ch. 4, test 2)."""

    rho_l: float = 0.445
    u_l: float = 0.698
    p_l: float = 3.528
    rho_r: float = 0.500
    u_r: float = 0.0
    p_r: float = 0.571
    gamma: float = LAX_GAMMA
    x0: float = 0.5
    t_end: float = 0.16

    def as_dict(self) -> Dict[str, float]:
        return {
            "rho_l": self.rho_l, "u_l": self.u_l, "p_l": self.p_l,
            "rho_r": self.rho_r, "u_r": self.u_r, "p_r": self.p_r,
            "gamma": self.gamma, "x0": self.x0, "t_end": self.t_end,
        }


# ----------------------------------------------------------------------
# case builder
# ----------------------------------------------------------------------
def lax(nx: int = 400, *, flux_mode: str = "hllc", order: int = 2,
        limiter: str = "mc", t_end: float | None = None, cfl: float = 0.4,
        cfg: LaxConfig | None = None):
    """Lax shock tube on ``[0, 1]``; returns ``(solver, meta)`` with the exact solution."""
    cfg = LaxConfig() if cfg is None else cfg
    t_end = cfg.t_end if t_end is None else t_end
    eos = StiffenedGasMixture([cfg.gamma, cfg.gamma], names=["air", "air"])
    solver = EulerSolver(
        nx=nx, ny=1, lx=1.0, ly=1.0, eos=eos,
        bc=("outflow", "outflow", "periodic", "periodic"),
        flux_mode=flux_mode, order=order, limiter=limiter, cfl=cfl,
    )
    xc = (torch.arange(nx, dtype=torch.float64) + 0.5) / nx
    left = xc < cfg.x0
    solver.set_primitive({
        "rho": torch.where(left, torch.full_like(xc, cfg.rho_l), torch.full_like(xc, cfg.rho_r)),
        "u": torch.where(left, torch.full_like(xc, cfg.u_l), torch.full_like(xc, cfg.u_r)),
        "v": torch.zeros_like(xc),
        "p": torch.where(left, torch.full_like(xc, cfg.p_l), torch.full_like(xc, cfg.p_r)),
        "Y": torch.zeros_like(xc),
    })
    exact = exact_riemann(cfg)
    meta = {"case": "lax", "nx": nx, "t_end": t_end, "xc": xc, "x0": cfg.x0,
            "exact": exact, "config": cfg}
    return solver, meta


# ----------------------------------------------------------------------
# exact solution (in-repo Toro iteration)
# ----------------------------------------------------------------------
def exact_riemann(cfg: LaxConfig | None = None) -> ExactRiemann:
    """Solve the Lax Riemann problem with the in-repo Toro exact solver."""
    cfg = LaxConfig() if cfg is None else cfg
    exact = ExactRiemann(cfg.rho_l, cfg.u_l, cfg.p_l, cfg.gamma,
                         cfg.rho_r, cfg.u_r, cfg.p_r, cfg.gamma)
    exact.solve()
    return exact


def exact_waves(cfg: LaxConfig | None = None, exact: ExactRiemann | None = None) -> Dict[str, float]:
    """Positions of the four wave features at ``t_end`` plus the star state."""
    cfg = LaxConfig() if cfg is None else cfg
    exact = exact_riemann(cfg) if exact is None else exact
    p_star = float(exact.pstar)
    u_star = float(exact.ustar)
    rho_sl = float(exact.density_star_l())
    rho_sr = float(exact.density_star_r())
    c_sl = math.sqrt(cfg.gamma * p_star / rho_sl)
    s_head = float(exact.wave_speed_l())      # u_L - c_L (rarefaction family)
    s_tail = u_star - c_sl                    # u* - c*_L
    s_contact = u_star
    s_shock = float(exact.wave_speed_r())     # S_R (shock family)
    t = cfg.t_end
    return {
        "p_star": p_star, "u_star": u_star,
        "rho_star_l": rho_sl, "rho_star_r": rho_sr,
        "c_star_l": c_sl,
        "speed_head": s_head, "speed_tail": s_tail,
        "speed_contact": s_contact, "speed_shock": s_shock,
        "x_head": cfg.x0 + s_head * t,
        "x_tail": cfg.x0 + s_tail * t,
        "x_contact": cfg.x0 + s_contact * t,
        "x_shock": cfg.x0 + s_shock * t,
        "left_wave": "rarefaction" if p_star < cfg.p_l else "shock",
        "right_wave": "shock" if p_star > cfg.p_r else "rarefaction",
    }


# ----------------------------------------------------------------------
# exact solution, independent re-derivation (no shared code with riemann.py)
# ----------------------------------------------------------------------
def _f_k(p: float, rho: float, u: float, p_k: float, gamma: float) -> float:
    """Toro eq. (4.6): pressure function of one side, ideal gas."""
    c_k = math.sqrt(gamma * p_k / rho)
    if p > p_k:                                   # shock branch
        a_k = 2.0 / ((gamma + 1.0) * rho)
        b_k = (gamma - 1.0) / (gamma + 1.0) * p_k
        return (p - p_k) * math.sqrt(a_k / (p + b_k))
    ratio = p / p_k                               # rarefaction branch
    return 2.0 * c_k / (gamma - 1.0) * (ratio ** ((gamma - 1.0) / (2.0 * gamma)) - 1.0)


def independent_star_state(cfg: LaxConfig | None = None, tol: float = 1e-14) -> Dict[str, float]:
    """``(p*, u*, rho*_L, rho*_R)`` by plain bisection on the pressure function.

    Deliberately naive (bisection, not Newton; the pressure function is written
    out here rather than imported) so that agreement with
    :func:`riemann.ExactRiemann` is a genuine cross-check.
    """
    cfg = LaxConfig() if cfg is None else cfg
    g = cfg.gamma

    def total(p):
        # Toro eq. (4.5): f(p) = f_L(p) + f_R(p) + (u_R - u_L) = 0
        return (_f_k(p, cfg.rho_l, cfg.u_l, cfg.p_l, g)
                + _f_k(p, cfg.rho_r, cfg.u_r, cfg.p_r, g)
                + (cfg.u_r - cfg.u_l))

    lo, hi = 1e-8 * cfg.p_r, 1e3 * max(cfg.p_l, cfg.p_r)
    if total(lo) > 0.0 or total(hi) < 0.0:       # pragma: no cover - guarded data
        raise RuntimeError("pressure function does not bracket a root")
    for _ in range(300):
        mid = 0.5 * (lo + hi)
        if total(mid) < 0.0:
            lo = mid
        else:
            hi = mid
        if hi - lo < tol * max(1.0, hi):
            break
    p_star = 0.5 * (lo + hi)
    f_l = _f_k(p_star, cfg.rho_l, cfg.u_l, cfg.p_l, g)
    f_r = _f_k(p_star, cfg.rho_r, cfg.u_r, cfg.p_r, g)
    u_star = 0.5 * (cfg.u_l + cfg.u_r) + 0.5 * (f_r - f_l)
    return {
        "p_star": p_star, "u_star": u_star,
        "rho_star_l": _rho_star(p_star, cfg.rho_l, cfg.p_l, g),
        "rho_star_r": _rho_star(p_star, cfg.rho_r, cfg.p_r, g),
    }


def _rho_star(p_star: float, rho_k: float, p_k: float, gamma: float) -> float:
    """Star density of one side: Hugoniot for a shock, isentrope for a fan."""
    if p_star > p_k:
        ratio = p_star / p_k
        return rho_k * (((gamma + 1.0) * ratio + (gamma - 1.0))
                        / ((gamma - 1.0) * ratio + (gamma + 1.0)))
    return rho_k * (p_star / p_k) ** (1.0 / gamma)


def independent_sample(xi: torch.Tensor, cfg: LaxConfig | None = None) -> Tuple[torch.Tensor, ...]:
    """Exact ``(rho, u, p)`` at similarity coordinate ``xi = (x - x0)/t``.

    Independent construction: shocks from the Rankine--Hugoniot relations, the
    fan from the Riemann invariant ``u + 2c/(gamma-1)``, no call into
    ``riemann.py``.
    """
    cfg = LaxConfig() if cfg is None else cfg
    g = cfg.gamma
    star = independent_star_state(cfg)
    p_star, u_star = star["p_star"], star["u_star"]

    def one_side(xi_, rho_k, u_k, p_k, rho_sk, right: bool):
        c_k = math.sqrt(g * p_k / rho_k)
        const = lambda v: torch.full_like(xi_, float(v))
        if p_star > p_k:                          # ---- shock: R-H jump at S_K
            s = u_k + (1.0 if right else -1.0) * c_k * math.sqrt(
                (g + 1.0) / (2.0 * g) * p_star / p_k + (g - 1.0) / (2.0 * g))
            outside = xi_ > s if right else xi_ < s
            return (torch.where(outside, const(rho_k), const(rho_sk)),
                    torch.where(outside, const(u_k), const(u_star)),
                    torch.where(outside, const(p_k), const(p_star)))
        # ---- rarefaction: isentropic fan between head and tail
        c_sk = math.sqrt(g * p_star / rho_sk)
        head = u_k + (1.0 if right else -1.0) * c_k
        tail = u_star + (1.0 if right else -1.0) * c_sk
        lo, hi = min(head, tail), max(head, tail)
        inside = (xi_ > lo) & (xi_ < hi)
        if right:                                 # u - 2c/(g-1) is the invariant
            inv = u_k - 2.0 * c_k / (g - 1.0)
            c_in = torch.clamp((xi_ - inv) * (g - 1.0) / (g + 1.0), min=1e-30)
            u_in = xi_ - c_in
        else:                                     # u + 2c/(g-1) is the invariant
            inv = u_k + 2.0 * c_k / (g - 1.0)
            c_in = torch.clamp((inv - xi_) * (g - 1.0) / (g + 1.0), min=1e-30)
            u_in = xi_ + c_in
        rho_in = rho_k * (c_in / c_k) ** (2.0 / (g - 1.0))
        p_in = p_k * (c_in / c_k) ** (2.0 * g / (g - 1.0))
        outside = xi_ < lo if not right else xi_ > hi
        rho = torch.where(outside, const(rho_k), torch.where(inside, rho_in, const(rho_sk)))
        u = torch.where(outside, const(u_k), torch.where(inside, u_in, const(u_star)))
        p = torch.where(outside, const(p_k), torch.where(inside, p_in, const(p_star)))
        return rho, u, p

    rho_l, u_l_, p_l_ = one_side(xi, cfg.rho_l, cfg.u_l, cfg.p_l, star["rho_star_l"], right=False)
    rho_r, u_r_, p_r_ = one_side(xi, cfg.rho_r, cfg.u_r, cfg.p_r, star["rho_star_r"], right=True)
    left = xi <= u_star
    return (torch.where(left, rho_l, rho_r),
            torch.where(left, u_l_, u_r_),
            torch.where(left, p_l_, p_r_))


def independent_check(cfg: LaxConfig | None = None, n_xi: int = 4001) -> Dict[str, float]:
    """Compare the in-repo exact solver with the independent re-derivation."""
    cfg = LaxConfig() if cfg is None else cfg
    exact = exact_riemann(cfg)
    ref = exact_waves(cfg, exact)
    ind = independent_star_state(cfg)
    xi = torch.linspace(-6.0, 6.0, n_xi, dtype=torch.float64)
    rho_e, u_e, p_e = exact.sample(xi)
    rho_i, u_i, p_i = independent_sample(xi, cfg)
    scale = max(abs(ref["p_star"]), abs(ref["u_star"]), cfg.rho_l, cfg.p_l)
    return {
        "p_star": ind["p_star"], "u_star": ind["u_star"],
        "rho_star_l": ind["rho_star_l"], "rho_star_r": ind["rho_star_r"],
        "rel_dp_star": abs(ind["p_star"] - ref["p_star"]) / ref["p_star"],
        "rel_du_star": abs(ind["u_star"] - ref["u_star"]) / max(abs(ref["u_star"]), 1e-30),
        "rel_d_rho_star_l": abs(ind["rho_star_l"] - ref["rho_star_l"]) / ref["rho_star_l"],
        "rel_d_rho_star_r": abs(ind["rho_star_r"] - ref["rho_star_r"]) / ref["rho_star_r"],
        "max_abs_profile_diff": float(max(
            (rho_e - rho_i).abs().max(), (u_e - u_i).abs().max(), (p_e - p_i).abs().max())),
        "max_abs_profile_diff_scaled": float(max(
            (rho_e - rho_i).abs().max(), (u_e - u_i).abs().max(), (p_e - p_i).abs().max()) / scale),
        "n_xi": n_xi,
    }


def rankine_hugoniot_residual(cfg: LaxConfig | None = None) -> Dict[str, float]:
    """Residual of mass/momentum/energy Rankine--Hugoniot for the right shock.

    A first-principles self-consistency check of the reference state that does
    not depend on either solver: with ``S_R`` the shock speed and states
    ``(rho_R, u_R, p_R) -> (rho*_R, u*, p*)``, the three jumps must balance.
    """
    cfg = LaxConfig() if cfg is None else cfg
    w = exact_waves(cfg)
    g = cfg.gamma
    s = w["speed_shock"]
    # Rankine--Hugoniot in the form S (U* - U_R) = F* - F_R, component by component
    mass = s * (w["rho_star_r"] - cfg.rho_r) - (w["rho_star_r"] * w["u_star"] - cfg.rho_r * cfg.u_r)
    mom = (s * (w["rho_star_r"] * w["u_star"] - cfg.rho_r * cfg.u_r)
           - ((w["p_star"] + w["rho_star_r"] * w["u_star"] ** 2)
              - (cfg.p_r + cfg.rho_r * cfg.u_r ** 2)))
    e_r = cfg.p_r / (g - 1.0) + 0.5 * cfg.rho_r * cfg.u_r ** 2
    e_s = w["p_star"] / (g - 1.0) + 0.5 * w["rho_star_r"] * w["u_star"] ** 2
    f_r = cfg.u_r * (e_r + cfg.p_r)
    f_s = w["u_star"] * (e_s + w["p_star"])
    energy = s * (e_s - e_r) - (f_s - f_r)
    # scales keep the ratios bounded when u_R = 0 (f_R then vanishes identically)
    ref_energy = abs(s * e_s) + abs(f_s)
    return {
        "rel_residual_mass": abs(mass) / (abs(s * w["rho_star_r"]) + 1e-30),
        "rel_residual_momentum": abs(mom) / (abs(s * w["rho_star_r"] * w["u_star"])
                                             + abs(w["p_star"]) + 1e-30),
        "rel_residual_energy": abs(energy) / (ref_energy + 1e-30),
    }


# ----------------------------------------------------------------------
# measurements
# ----------------------------------------------------------------------
def _crossing(x: torch.Tensor, f: torch.Tensor, level: float,
              x_lo: float, x_hi: float, rising: bool = True) -> float:
    """Sub-cell position where ``f`` crosses ``level`` inside ``[x_lo, x_hi]``."""
    mask = (x >= x_lo) & (x <= x_hi)
    xs, fs = x[mask], f[mask]
    if xs.numel() < 2:
        return float("nan")
    diff = fs - level
    for i in range(xs.numel() - 1):
        a, b = float(diff[i]), float(diff[i + 1])
        if rising and a <= 0.0 < b:
            t = -a / (b - a)
            return float(xs[i] + t * (xs[i + 1] - xs[i]))
        if (not rising) and a >= 0.0 > b:
            t = a / (a - b)
            return float(xs[i] + t * (xs[i + 1] - xs[i]))
    return float("nan")


def _window_mean(x: torch.Tensor, f: torch.Tensor, lo: float, hi: float) -> float:
    mask = (x >= lo) & (x <= hi)
    if int(mask.sum()) == 0:
        return float("nan")
    return float(f[mask].mean())


def _rel(got: float, ref: float) -> float:
    if got != got or ref != ref:
        return float("nan")
    return abs(got - ref) / abs(ref) if abs(ref) > 1e-30 else abs(got - ref)


def measure_positions(x: torch.Tensor, rho: torch.Tensor, p: torch.Tensor,
                      w: Dict[str, float], cfg: LaxConfig) -> Dict[str, float]:
    """Locate the three measurable wave features of the Lax solution.

    ``x_fan``      mid-rarefaction point: rho half-way between rho_L and rho*_L
    ``x_contact``  contact discontinuity: rho half-way between rho*_L and rho*_R
    ``x_shock``    shock: p half-way between p* and p_R
    """
    return {
        "x_fan": _crossing(x, rho, 0.5 * (cfg.rho_l + w["rho_star_l"]),
                           0.0, w["x_tail"], rising=False),
        "x_contact": _crossing(x, rho, 0.5 * (w["rho_star_l"] + w["rho_star_r"]),
                               w["x_tail"], w["x_shock"], rising=True),
        "x_shock": _crossing(x, p, 0.5 * (cfg.p_r + w["p_star"]),
                             w["x_contact"], 1.0, rising=False),
    }


def profile_metrics(meta: Dict, prim: Dict[str, torch.Tensor],
                    cfg: LaxConfig | None = None) -> Dict[str, object]:
    """Compare one numerical run against the exact solution and the star state."""
    cfg = meta["config"]
    t = meta["t_end"]
    x = meta["xc"]
    rho_n = prim["rho"][:, 0].double().cpu()
    u_n = prim["u"][:, 0].double().cpu()
    p_n = prim["p"][:, 0].double().cpu()
    xi = (x - meta["x0"]) / t
    rho_e, u_e, p_e = meta["exact"].sample(xi)
    rho_e, u_e, p_e = rho_e.cpu(), u_e.cpu(), p_e.cpu()

    def l1(a, b):
        return float((a - b).abs().mean() / b.abs().mean())

    def linf(a, b):
        return float((a - b).abs().max() / b.abs().max())

    w = exact_waves(cfg, meta["exact"])
    # plateau windows: stay 15 % of each sub-region away from the jumps
    wl = w["x_contact"] - w["x_tail"]
    wr = w["x_shock"] - w["x_contact"]
    lo_l, hi_l = w["x_tail"] + 0.15 * wl, w["x_contact"] - 0.15 * wl
    lo_r, hi_r = w["x_contact"] + 0.15 * wr, w["x_shock"] - 0.15 * wr

    plateau = {
        "p_star": _window_mean(x, p_n, lo_l, hi_r),
        "u_star": _window_mean(x, u_n, lo_l, hi_r),
        "rho_star_l": _window_mean(x, rho_n, lo_l, hi_l),
        "rho_star_r": _window_mean(x, rho_n, lo_r, hi_r),
    }
    plateau_ref = {
        "p_star": w["p_star"], "u_star": w["u_star"],
        "rho_star_l": w["rho_star_l"], "rho_star_r": w["rho_star_r"],
    }
    plateau_rel = {k: _rel(plateau[k], plateau_ref[k]) for k in plateau}

    # Wave features are located with ONE estimator applied to both fields (the
    # numerical one and the exact one sampled on a fine grid), so the estimator
    # bias cancels in the comparison.  ``x_fan`` is the mid-fan point: the fan is
    # a smooth rarefaction, so "where rho is half-way between rho_L and rho*_L"
    # is the only resolution-independent definition of its location.
    positions = measure_positions(x, rho_n, p_n, w, cfg)
    x_ex = torch.linspace(0.0, 1.0, 20001, dtype=torch.float64)
    rho_x, _, p_x = meta["exact"].sample((x_ex - meta["x0"]) / t)
    positions_ref = measure_positions(x_ex, rho_x.cpu(), p_x.cpu(), w, cfg)
    positions_rel = {k: _rel(positions[k], positions_ref[k]) for k in positions}

    return {
        "l1_rel": {"rho": l1(rho_n, rho_e), "u": l1(u_n, u_e), "p": l1(p_n, p_e)},
        "linf_rel": {"rho": linf(rho_n, rho_e), "u": linf(u_n, u_e), "p": linf(p_n, p_e)},
        "plateau": plateau,
        "plateau_exact": plateau_ref,
        "plateau_rel_error": plateau_rel,
        "position": {k: (None if v != v else v) for k, v in positions.items()},
        "position_exact": positions_ref,
        "position_rel_error": {k: (None if v != v else v) for k, v in positions_rel.items()},
        "windows": {"left": [lo_l, hi_l], "right": [lo_r, hi_r]},
    }


# ----------------------------------------------------------------------
# runs
# ----------------------------------------------------------------------
def run_case(nx: int, *, order: int = 2, limiter: str = "mc", cfl: float = 0.4,
             flux_mode: str = "hllc", cfg: LaxConfig | None = None,
             keep_profile: bool = False) -> Dict[str, object]:
    """Run one Lax case and return its measurements against the exact solution."""
    t0 = time.time()
    solver, meta = lax(nx, flux_mode=flux_mode, order=order, limiter=limiter,
                       cfl=cfl, cfg=cfg)
    initial = solver.integrals()
    solver.run(meta["t_end"])
    prim = solver.primitive()
    metrics = profile_metrics(meta, prim, cfg)
    drift = solver.budget_drift(initial)
    row = {
        "nx": nx, "order": order, "limiter": limiter, "cfl": cfl, "flux_mode": flux_mode,
        "steps": solver.nsteps, "seconds": round(time.time() - t0, 2),
        "positivity_fixes": solver.positivity_fixes,
        "drift": drift,
        **{k: metrics[k] for k in ("l1_rel", "linf_rel", "plateau", "plateau_exact",
                                   "plateau_rel_error", "position", "position_exact",
                                   "position_rel_error")},
    }
    if keep_profile:
        xi = (meta["xc"] - meta["x0"]) / meta["t_end"]
        rho_e, u_e, p_e = meta["exact"].sample(xi)
        row["profile"] = {
            "x": meta["xc"].tolist(),
            "rho": prim["rho"][:, 0].tolist(), "rho_exact": rho_e.tolist(),
            "u": prim["u"][:, 0].tolist(), "u_exact": u_e.tolist(),
            "p": prim["p"][:, 0].tolist(), "p_exact": p_e.tolist(),
        }
    return row


def sweep(sizes: Tuple[int, ...] = (200, 400, 800, 1600), *, cfg: LaxConfig | None = None,
          **kwargs) -> Dict[str, object]:
    """Grid sweep with the production scheme plus the observed convergence order."""
    rows = []
    for nx in sizes:
        rows.append(run_case(nx, keep_profile=(nx == max(sizes)), cfg=cfg, **kwargs))
    orders = {}
    for key in ("rho", "u", "p"):
        obs = []
        for a, b in zip(rows[:-1], rows[1:]):
            ea, eb = a["l1_rel"][key], b["l1_rel"][key]
            if ea > 0.0 and eb > 0.0:
                obs.append(math.log(ea / eb) / math.log(b["nx"] / a["nx"]))
            else:
                obs.append(float("nan"))
        orders[key] = obs
    return {"rows": rows, "observed_order": orders}


# ----------------------------------------------------------------------
# acceptance
# ----------------------------------------------------------------------
def verdicts(rows: List[Dict[str, object]], *, cfg: LaxConfig | None = None) -> Dict[str, object]:
    """Apply the predeclared <2 % criteria to the finest run of a sweep."""
    cfg = LaxConfig() if cfg is None else cfg
    fine = max(rows, key=lambda r: r["nx"])
    checks: Dict[str, object] = {}
    for k, v in fine["plateau_rel_error"].items():
        ok = bool(v == v and v <= THRESHOLDS["plateau_rel_error"])
        checks[f"plateau[{k}]"] = {"value": v, "ok": ok}
    for k, v in fine["position_rel_error"].items():
        # None == the feature could not be located in the field (measurement failure
        ok = bool(v is not None and v == v and v <= THRESHOLDS["wave_position_rel_error"])
        checks[f"position[{k}]"] = {"value": v, "ok": ok}
    for k, v in fine["l1_rel"].items():
        ok = bool(v == v and v <= THRESHOLDS["l1_rel_error"])
        checks[f"l1[{k}]"] = {"value": v, "ok": ok}
    drift = fine["drift"]
    checks["mass_drift"] = {"value": abs(drift["mass"]),
                            "ok": bool(abs(drift["mass"]) <= THRESHOLDS["conservation_drift"])}
    checks["no_positivity_fix"] = {"value": fine["positivity_fixes"],
                                   "ok": bool(fine["positivity_fixes"] == 0)}
    return {"nx": fine["nx"], "checks": checks,
            "all_ok": all(c["ok"] for c in checks.values())}