"""Static bubble: Young-Laplace pressure jump and spurious currents.

Case statement
--------------
A circular bubble of radius ``R`` sits in a quiescent, immiscible liquid with
surface tension ``sigma``.  The exact solution of the coupled problem is
``u = 0`` everywhere and a pressure jump ``p_in - p_out = sigma/R`` across the
interface (Young-Laplace, 2D circular interface).  The discrete solution cannot
hold that balance exactly: the residual of the discrete capillary force,
``f_sigma = sigma kappa grad(alpha)``, is a body force with a non-zero curl, so
a quasi-steady *spurious* (parasitic) current appears.  Its magnitude is the
accepted quality metric for the surface-tension/free-surface coupling.

Normalisation (criterion of this node)
-------------------------------------
The current is non-dimensionalised with the capillary velocity scale

.. math::  u_sigma = \\sqrt{\\sigma/(\\rho_l R)}

and the criterion is ``max|u| / u_sigma <= 0.01`` (literature band 1e-3 .. 1e-2
for balanced CSF formulations).  The absolute current is *not* used as a
criterion because it is Mach-like: it scales with ``sigma`` and ``R`` and so an
absolute threshold silently changes meaning between fluids.

Why the balanced-CSF control matters
-----------------------------------
The package drives the same case twice: with ``csf_balance="balanced"``
(face-sampled force using the same two-point ``alpha`` difference as the
pressure gradient, Francois et al. 2006) and with ``csf_balance="legacy"``
(cell-centred force average).  The legacy variant is known to leave a
divergence/curl residual (see ``docs/two-phase-csf-balance``); running it
through the *same* metric is what proves the metric discriminates rather than
just reporting a small number.

References
----------
* Brackbill, Kothe & Zemach, J. Comput. Phys. 100 (1992) 335 (CSF).
* Francois et al., J. Comput. Phys. 213 (2006) 141 (balanced CSF).
* Popinet, J. Comput. Phys. 228 (2009) 5838 (height-function curvature).
* Abadie, Aubin & Legendre, Comput. Fluids 114 (2015) 99 (spurious currents
  of VOF/level-set surface-tension models).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch

from ..two_phase_solver import GridConfig, TwoPhaseConfig, TwoPhaseSolver

__all__ = [
    "StaticBubbleConfig",
    "bubble_alpha",
    "capillary_velocity",
    "laplace_pressure_jump",
    "laplace_number",
    "run_static_bubble",
    "spurious_current",
]


@dataclass
class StaticBubbleConfig:
    """Discretisation and physics of one static-bubble run."""

    cells: int = 64
    #: Bubble radius (domain is the unit square, so ``R`` is also R/L).
    radius: float = 0.2
    domain: float = 1.0
    #: Interfacial tension and the two densities/viscosities.
    sigma: float = 0.07
    rho_liquid: float = 1000.0
    rho_gas: float = 1.0
    mu_liquid: float = 1.0e-2
    mu_gas: float = 1.0e-2
    gravity: float = 0.0
    #: The run length is ``damping_times`` viscous times at the grid scale,
    #: ``tau_mu = rho_l h^2 / mu_l``: that is the time the (force-balanced)
    #: spurious current needs to saturate.  ``max_steps`` caps the cost, and
    #: ``dt_capillary_fraction`` sets ``dt`` as a fraction of the capillary
    #: time step ``sqrt(rho_l h^3/(2 pi sigma))``.  Both limits were measured,
    #: not guessed: ``dt_capillary_fraction >= 0.2`` is unstable for the
    #: explicit CSF predictor (see ``docs/P447W2-N01``, section 5).
    damping_times: float = 3.0
    max_steps: int = 4000
    dt_capillary_fraction: float = 0.02
    #: Fixed step override (used by the tests to keep the runtime short).
    steps: int | None = None
    #: Discretisation choices (the point of the sweep is to compare them).
    #: ``"balanced"`` / ``"legacy"`` surface-tension sampling, ``"height_function"``
    #: / ``"divergence"`` curvature, and the alpha-smoothing pass count.
    csf_balance: str = "balanced"
    csf_curvature: str = "height_function"
    csf_smoothing: int = 3
    #: Interface profile: ``"ramp"`` (one-cell linear ramp, sharpest) or
    #: ``"tanh"`` (diffuse over ~2 cells, the usual CFD initialisation).
    profile: str = "ramp"
    #: Sample history every ``track_every`` steps.
    track_every: int = 10

    def __post_init__(self) -> None:
        if self.cells < 16:
            raise ValueError("need at least 16 cells per direction")
        if self.radius <= 0.0 or self.radius > 0.45 * self.domain:
            raise ValueError("radius must be positive and fit the domain")
        if self.sigma <= 0.0:
            raise ValueError("sigma must be positive for a static bubble")
        if self.csf_balance not in ("balanced", "legacy"):
            raise ValueError(f"unknown csf_balance {self.csf_balance!r}")
        if self.csf_curvature not in ("height_function", "divergence"):
            raise ValueError(f"unknown csf_curvature {self.csf_curvature!r}")

    @property
    def dx(self) -> float:
        return self.domain / self.cells

    @property
    def laplace_number(self) -> float:
        """``La = rho_l sigma R / mu_l^2`` (the capillary/inertia-viscous ratio)."""
        return self.rho_liquid * self.sigma * self.radius / self.mu_liquid ** 2

    @property
    def tau_mu(self) -> float:
        """Viscous time at the grid scale, ``rho_l h^2/mu_l``."""
        return self.rho_liquid * self.dx ** 2 / self.mu_liquid

    @property
    def bond_number(self) -> float:
        """``Bo = (rho_l - rho_g) g R^2 / sigma`` (zero at zero gravity)."""
        return (self.rho_liquid - self.rho_gas) * self.gravity * self.radius ** 2 / self.sigma


def capillary_velocity(sigma: float, rho_liquid: float, radius: float) -> float:
    """``u_sigma = sqrt(sigma/(rho_l R))`` -- the scale the current is measured in."""
    return math.sqrt(sigma / (rho_liquid * radius))


def laplace_number(cfg: StaticBubbleConfig) -> float:
    return cfg.laplace_number


def bubble_alpha(cells: int, radius: float, dx: float, profile: str = "ramp") -> tuple:
    """Colour field of a centred bubble, plus the radius of every cell centre."""
    x = (torch.arange(cells, dtype=torch.float64) + 0.5) * dx
    xx, yy = torch.meshgrid(x, x, indexing="xy")
    rr = torch.sqrt((xx - 0.5) ** 2 + (yy - 0.5) ** 2)
    if profile == "ramp":
        alpha = torch.clamp((radius - rr) / dx + 0.5, 0.0, 1.0)
    elif profile == "tanh":
        alpha = 0.5 * (1.0 - torch.tanh((rr - radius) / dx))
    else:
        raise ValueError(f"unknown profile {profile!r}")
    return alpha, rr


def _make_solver(cfg: StaticBubbleConfig) -> tuple:
    tp = TwoPhaseConfig(
        enabled=True,
        rho_liquid=cfg.rho_liquid,
        rho_gas=cfg.rho_gas,
        mu_liquid=cfg.mu_liquid,
        mu_gas=cfg.mu_gas,
        gravity=cfg.gravity,
        sigma=cfg.sigma,
        surface_tension=True,
        # A sealed box is an all-Neumann pressure problem; the atmospheric
        # datum on the top face supplies the pressure level and, at zero
        # gravity, has no dynamical effect on a centred bubble.
        top_atmosphere=True,
        bottom_wall=True,
        pressure_form="reduced",
        warm_start_pressure=True,
        csf_balance=cfg.csf_balance,
        csf_curvature=cfg.csf_curvature,
        csf_smoothing=cfg.csf_smoothing,
    )
    solver = TwoPhaseSolver(
        GridConfig(nx=cfg.cells, ny=cfg.cells, length=cfg.domain, height=cfg.domain),
        tp,
    )
    alpha, rr = bubble_alpha(cfg.cells, cfg.radius, solver.dx, cfg.profile)
    solver.set_alpha(alpha)
    return solver, rr


def spurious_current(solver: TwoPhaseSolver) -> dict:
    """``max|u|`` on the MAC faces, and its normalisation by ``u_sigma``."""
    u_max = float(solver.u.abs().max())
    v_max = float(solver.v.abs().max())
    return {"max_u": max(u_max, v_max), "max_u_component": u_max, "max_v_component": v_max}


def laplace_pressure_jump(solver: TwoPhaseSolver, rr: torch.Tensor, radius: float) -> dict:
    """Measured ``p_in - p_out`` against ``sigma/R``.

    ``p_in`` is the mean over ``r < R/2`` (bulk liquid, away from the interface
    band) and ``p_out`` the mean over ``r > 1.6 R`` (undisturbed gas).  With
    ``pressure_form="reduced"`` and zero gravity ``p`` is the physical pressure.
    """
    sigma_r = solver.tp.sigma / radius
    p_in = float(solver.p[rr < 0.5 * radius].mean())
    p_out = float(solver.p[rr > 1.6 * radius].mean())
    jump = p_in - p_out
    return {
        "sigma_over_R": sigma_r,
        "p_inside": p_in,
        "p_outside": p_out,
        "pressure_jump": jump,
        "jump_error_relative": jump / sigma_r - 1.0,
    }


def run_static_bubble(cfg: StaticBubbleConfig | None = None, **overrides) -> dict:
    """Run one static bubble to a quasi-steady state and measure the current.

    Returns a JSON-ready dict: the normalised spurious current (the criterion
    quantity), its time history (so a drifting/diverging run is visible), the
    Laplace pressure jump, the volume drift and the configuration fingerprint.
    """
    cfg = cfg or StaticBubbleConfig()
    if overrides:
        cfg = StaticBubbleConfig(**{**cfg.__dict__, **overrides})
    solver, rr = _make_solver(cfg)

    dt_cap = math.sqrt(cfg.rho_liquid * solver.dx ** 3 / (2.0 * math.pi * cfg.sigma))
    dt = cfg.dt_capillary_fraction * dt_cap
    u_sigma = capillary_velocity(cfg.sigma, cfg.rho_liquid, cfg.radius)
    if cfg.steps is not None:
        n_steps = cfg.steps
    else:
        n_steps = int(min(cfg.max_steps, math.ceil(cfg.damping_times * cfg.tau_mu / dt)))
    n_steps = max(n_steps, 10)

    m0 = float(solver.alpha.sum())
    history = []
    diverged = False
    last_ratio = float("nan")
    for step in range(n_steps):
        try:
            solver.step(dt)
        except RuntimeError as exc:  # "non-finite iterate" from the solver
            diverged = True
            divergence_message = str(exc)
            break
        if step % cfg.track_every == 0 or step == n_steps - 1:
            cur = spurious_current(solver)
            last_ratio = cur["max_u"] / u_sigma
            history.append(
                {
                    "step": step + 1,
                    "time": solver.time,
                    "max_u": cur["max_u"],
                    "ratio": cur["max_u"] / u_sigma,
                }
            )

    cur = spurious_current(solver)
    if not math.isfinite(cur["max_u"]) or not math.isfinite(float(solver.alpha.max())):
        diverged = True
    jump = laplace_pressure_jump(solver, rr, cfg.radius)
    # Plateau test: with a saturated current the last quarter of the history is
    # flat; a still-growing run is reported as such instead of being quoted as
    # a steady state.
    quarter = max(2, len(history) // 4)
    tail = [h["ratio"] for h in history[-quarter:]]
    plateau_ratio = (tail[-1] / max(tail[0], 1e-300)) if len(tail) > 1 else float("nan")
    saturated = bool(len(tail) > 1 and abs(plateau_ratio - 1.0) < 0.05 and not diverged)
    return {
        "case": "static_bubble",
        "cells": cfg.cells,
        "dx": solver.dx,
        "radius": cfg.radius,
        "sigma": cfg.sigma,
        "rho_liquid": cfg.rho_liquid,
        "rho_gas": cfg.rho_gas,
        "mu_liquid": cfg.mu_liquid,
        "mu_gas": cfg.mu_gas,
        "gravity": cfg.gravity,
        "laplace_number": cfg.laplace_number,
        "bond_number": cfg.bond_number,
        "csf_balance": cfg.csf_balance,
        "csf_curvature": cfg.csf_curvature,
        "csf_smoothing": cfg.csf_smoothing,
        "profile": cfg.profile,
        "dt": dt,
        "dt_capillary": dt_cap,
        "steps": n_steps,
        "damping_times": cfg.damping_times,
        "tau_mu": cfg.tau_mu,
        "saturated": saturated,
        "diverged": diverged,
        "divergence_time": solver.time if diverged else None,
        "last_finite_ratio": last_ratio,
        "completed_steps": len(history) and history[-1]["step"],
        "time_end": solver.time,
        "capillary_velocity": u_sigma,
        "spurious_max_u": cur["max_u"],
        "spurious_ratio": cur["max_u"] / u_sigma,
        "history_step": [h["step"] for h in history],
        "history_ratio": [h["ratio"] for h in history],
        "plateau_ratio": plateau_ratio,
        **{k: v for k, v in jump.items()},
        "volume_drift": abs(float(solver.alpha.sum()) - m0) / m0,
        "max_alpha": float(solver.alpha.max()),
        "min_alpha": float(solver.alpha.min()),
        "_solver": solver,
    }