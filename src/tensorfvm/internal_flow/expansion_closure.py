"""Momentum closure of the planar sudden-expansion pressure loss.

Why a closure
-------------
The Borda-Carnot value ``K_BC = (1 - 1/AR)**2`` (``AR = A2 / A1``) is the
sudden-enlargement momentum balance under three assumptions that a real
separated flow does not satisfy:

1. the velocity is uniform at both stations, so the momentum-flux coefficient
   ``beta = <u^2> / <u>^2`` is one at the outlet;
2. the pressure acting on the annular step face equals the upstream pressure
   ``p1``, i.e. the dead water behind the step carries no base suction;
3. wall friction and the normal viscous stress on the outflow plane are
   negligible.

Relaxing all three while keeping the momentum balance exact gives a *closure*
rather than a coefficient: every departure from Borda-Carnot is one named,
measurable term of the control volume ``[x in (0, L), y in (0, H)]``

    K = (1 - 1/AR)^2                     Borda-Carnot anchor
        + 2 (1 - beta_in) / AR           inlet momentum-profile term
        + 2 (beta_out - 1) / AR^2        outlet momentum-profile term
        + zeta_step (1 - 1/AR)           step-face base-pressure term
        + f                              wall-friction term
        - t_out                          outflow normal-viscous-stress term

with ``K = 1 - 1/AR^2 - Cp``, ``Cp = (p_out - p_in) / (0.5 rho U1^2)``,
``beta = <u^2> / <u>^2`` (``beta_in = 1`` for a uniform inlet; it is *not* one on
a mesh whose inlet rows are partially open, see :func:`closure_terms`),
``zeta_step = (p_in - p_step) / (0.5 rho U1^2)``, ``f = F_wall / (0.5 rho U1^2 A2)``
and ``t_out = T_xx / (0.5 rho U1^2 A2)``.  Derivation, sign conventions and the
discrete residual check are in :func:`closure_terms`; the algebraic identity is
pinned by :func:`closure_loss` and by ``tests/test_expansion_momentum_closure.py``.

Two normalisations are reported, because the literature mixes them:

* ``k_measure`` -- ``1 - 1/AR^2 - Cp`` with **area**-averaged static pressure.
  This is the quantity the Borda-Carnot expression refers to and the one the
  closure identity above reproduces.
* ``k_flux`` -- the mass-flux-weighted total-pressure difference used by
  :meth:`tensorfvm.internal_flow.sudden_expansion.SuddenExpansionCase.diagnostics`.
  It differs from ``k_measure`` by ``-(beta_out - 1) / AR^2``: the flux average
  of ``0.5 rho u^2`` is ``0.5 rho beta u_bar^2``, so a peaked outlet profile
  lowers the flux-averaged total pressure while raising the momentum flux.  A
  report that quotes one number and compares it with the other mixes a
  *measurement convention* into a *physics* difference.

Regime
------
The terms are measured on the solution of whatever velocity/pressure field is
handed in (laminar in-tree MAC solver, or any RANS field dumped to the same
grid); they are not tied to the turbulence model.  What a RANS closure changes
is the *value* of ``beta_out`` and ``zeta_step``, not the identity.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import torch

from .mac import StaggeredMAC
from .reference import borda_carnot_loss
from .sudden_expansion import (
    SuddenExpansionCase,
    SuddenExpansionConfig,
    inlet_profile,
)

#: Right-hand closures compared by :func:`exit_closure_study`.
EXIT_CLOSURES = ("outlet", "developed", "convective")


@dataclass(frozen=True)
class ClosureTerms:
    """Every term of the expansion momentum closure, dimensionless.

    Lengths are normalised by the downstream height ``H`` and stresses by the
    upstream dynamic pressure ``0.5 rho U1^2``; ``k_*`` are loss coefficients
    referred to ``U1``.
    """

    area_ratio: float
    beta_in: float
    beta_out: float
    cp_recovery: float
    cp_step: float
    wall_friction: float
    outlet_viscous: float
    k_borda_carnot: float
    k_inlet: float
    k_profile: float
    k_base: float
    k_closed: float
    k_measure: float
    k_flux: float
    closure_residual: float
    discrete_residual: float

    @property
    def k_friction(self) -> float:
        """Net friction contribution of the closure (wall drag minus outflow stress)."""
        return self.wall_friction - self.outlet_viscous

    @property
    def k_step(self) -> float:
        """Step-local loss: Borda-Carnot plus the step-face base-pressure term.

        This is the part of the measured loss that the step itself injects; it is
        free of the momentum-profile and wall-friction terms, and therefore of the
        station at which the outlet is closed.  A loss coefficient that moves when
        the outflow closure changes is a property of the outflow station, not of
        the expansion, and this decomposition says by how much.
        """
        return self.k_borda_carnot + self.k_base

    @property
    def error_vs_borda_carnot(self) -> float:
        """Relative error of the raw Borda-Carnot value against the measurement."""
        if self.k_measure == 0.0:
            return float("nan")
        return (self.k_borda_carnot - self.k_measure) / self.k_measure

    @property
    def error_vs_borda_carnot_step(self) -> float:
        """Relative error of the anchor against the *friction-free* step-local loss.

        The Borda-Carnot value refers to the loss of the step, not to the loss of
        the downstream duct: comparing it with ``k_measure`` charges the anchor for
        15 heights of wall shear, which at ``A1/A2 = 0.9`` is 97 % of the measured
        number.  ``k_step`` removes that term and is the fair counterpart.
        """
        if self.k_step == 0.0:
            return float("nan")
        return (self.k_borda_carnot - self.k_step) / self.k_step

    @property
    def error_of_closure(self) -> float:
        """Relative error of the closure assembly against the measurement."""
        if self.k_measure == 0.0:
            return float("nan")
        return (self.k_closed - self.k_measure) / self.k_measure

    def as_dict(self) -> dict[str, float]:
        payload = asdict(self)
        payload["k_friction"] = self.k_friction
        payload["k_step"] = self.k_step
        payload["error_vs_borda_carnot"] = self.error_vs_borda_carnot
        payload["error_vs_borda_carnot_step"] = self.error_vs_borda_carnot_step
        payload["error_of_closure"] = self.error_of_closure
        return payload


def closure_loss(area_ratio: float, beta_in: float = 1.0, beta_out: float = 1.0,
                 cp_step: float = 0.0, wall_friction: float = 0.0,
                 outlet_viscous: float = 0.0) -> float:
    """Assemble the closure, recovering Borda-Carnot exactly in the ideal limit.

    ``closure_loss(AR) == borda_carnot_loss(AR)`` to round-off; each of the five
    non-ideal terms is additive and independently measurable, which is what makes
    a term-by-term attribution of a measured loss possible.
    """
    if not math.isfinite(area_ratio) or area_ratio <= 1.0:
        raise ValueError("a sudden expansion requires area_ratio > 1")
    return (borda_carnot_loss(area_ratio)
            + 2.0 * (1.0 - beta_in) / area_ratio
            + 2.0 * (beta_out - 1.0) / area_ratio ** 2
            + cp_step * (1.0 - 1.0 / area_ratio)
            + wall_friction
            - outlet_viscous)


def _inlet_fractions(config: SuddenExpansionConfig) -> torch.Tensor:
    """Open fraction of every inlet row (1 in the opening, 0 on the step faces)."""
    profile = torch.tensor(inlet_profile(config), dtype=torch.float64)
    return torch.clamp(profile / config.inlet_velocity, 0.0, 1.0)


def closure_terms(case: SuddenExpansionCase) -> ClosureTerms:
    """Measure every term of the momentum closure on a converged solution.

    The control volume is the whole channel.  Integrals are the discrete sums
    that match the finite-volume operator (``hy`` per row, ``hy``/``hx`` per
    column), so the pressure difference telescopes exactly and the closure
    residual isolates the boundary fluxes that the continuum expression
    approximates: the van Leer convective flux at the outflow column and the
    one-sided wall shear.
    """
    solver: StaggeredMAC = case.solver
    config = case.config
    rho = solver.rho
    hy, hx = solver.hy, solver.hx
    reference = config.inlet_velocity
    dynamic = 0.5 * rho * reference ** 2
    area_ratio = config.area_ratio
    height = config.downstream_height

    fractions = _inlet_fractions(config)
    blocked = 1.0 - fractions
    area_in = float(fractions.sum()) * hy
    area_step = float(blocked.sum()) * hy
    area_out = height

    u_in = solver.u[0, :]
    flux_in = float(u_in.sum()) * hy
    mean_in = flux_in / area_in
    beta_in = float((u_in ** 2).sum()) * hy / (mean_in ** 2 * area_in)
    p_in = float((solver.p[0, :] * fractions).sum()) * hy / area_in
    p_step = float((solver.p[0, :] * blocked).sum()) * hy / area_step

    u_out = solver.u[solver.nx, :]
    flux_out = float(u_out.sum()) * hy
    mean_out = flux_out / area_out
    beta_out = float((u_out ** 2).sum()) * hy / (mean_out ** 2 * area_out)
    p_out = float(solver.p[solver.nx - 1, :].mean())

    # wall drag, two walls, same sign convention as ``wall_friction_loss``
    drag = 0.0
    for row in (0, solver.ny - 1):
        tau = rho * solver.nu * 2.0 * solver.u[:, row] / hy
        drag += float(torch.trapz(tau, dx=hx))
    wall_friction = drag / (dynamic * area_out)

    # normal viscous stress on the outflow plane, one-sided gradient
    tau_out = float((2.0 * rho * solver.nu
                     * (solver.u[solver.nx, :] - solver.u[solver.nx - 1, :]) / hx).sum()) * hy
    outlet_viscous = tau_out / (dynamic * area_out)

    cp_recovery = (p_out - p_in) / dynamic
    cp_step = (p_in - p_step) / dynamic

    k_measure = 1.0 - 1.0 / area_ratio ** 2 - cp_recovery
    total_in = solver.flux_weighted_total_pressure(0)
    total_out = solver.flux_weighted_total_pressure(solver.nx)
    k_flux = (total_in - total_out) / dynamic
    k_inlet = 2.0 * (1.0 - beta_in) / area_ratio
    k_closed = closure_loss(area_ratio, beta_in, beta_out, cp_step, wall_friction,
                            outlet_viscous)

    # continuum control-volume balance: momentum out - in + pressure + drag - tau
    balance = (rho * (float((u_out ** 2).sum()) - float((u_in ** 2).sum())) * hy
               + p_out * area_out - p_in * area_in - p_step * area_step
               + drag - tau_out)
    closure_residual = balance / (dynamic * area_out)

    # discrete balance: the solver's own momentum operator summed over the
    # control volume must telescope to the same boundary fluxes
    residual_u = solver._u_tendency()[1:solver.nx, :] \
        - (solver.p[1:, :] - solver.p[:-1, :]) / hx
    discrete_residual = float(residual_u.sum()) * hx * hy / (dynamic * area_out)

    return ClosureTerms(
        area_ratio=area_ratio,
        beta_in=beta_in,
        beta_out=beta_out,
        cp_recovery=cp_recovery,
        cp_step=cp_step,
        wall_friction=wall_friction,
        outlet_viscous=outlet_viscous,
        k_borda_carnot=borda_carnot_loss(area_ratio),
        k_inlet=k_inlet,
        k_profile=2.0 * (beta_out - 1.0) / area_ratio ** 2,
        k_base=cp_step * (1.0 - 1.0 / area_ratio),
        k_closed=k_closed,
        k_measure=k_measure,
        k_flux=k_flux,
        closure_residual=closure_residual,
        discrete_residual=discrete_residual,
    )


def run_case(area_ratio: float, reynolds: float, cells: int, length: float = 15.0,
             right: str = "outlet", tolerance: float = 1e-7,
             max_steps: int = 400_000) -> tuple[ClosureTerms, dict[str, object]]:
    """Solve one expansion case and return its closure terms and solver ledger."""
    config = SuddenExpansionConfig(
        area_ratio=area_ratio,
        reynolds=reynolds,
        cells_per_downstream_height=cells,
        length_in_downstream_heights=length,
        tolerance=tolerance,
        max_steps=max_steps,
        right=right,
    )
    case = SuddenExpansionCase(config)
    ledger = case.run()
    terms = closure_terms(case)
    return terms, {"ledger": ledger, "case": case}


def area_ratio_sweep(area_ratios, reynolds: float, cells, length: float = 15.0,
                     right: str = "outlet", tolerance: float = 1e-7) -> list[ClosureTerms]:
    """Closure decomposition over a list of area ratio ``A2 / A1`` values.

    ``cells`` is either one integer for every area ratio or one value per area
    ratio (the inlet height shrinks with ``AR``, so a mesh that resolves the jet
    at ``AR = 1.25`` does not resolve it at ``AR = 10``).
    """
    ratios = list(area_ratios)
    cells_list = [cells] * len(ratios) if isinstance(cells, int) else list(cells)
    if len(cells_list) != len(ratios):
        raise ValueError("cells must be one value or one value per area ratio")
    out = []
    for ratio, cell in zip(ratios, cells_list):
        terms, _ = run_case(ratio, reynolds, cell, length=length, right=right,
                            tolerance=tolerance)
        out.append(terms)
    return out


def exit_closure_study(area_ratio: float, reynolds: float, cells: int,
                       length: float = 15.0, closures=EXIT_CLOSURES,
                       tolerance: float = 1e-7) -> dict[str, ClosureTerms]:
    """Repeat one case with each downstream boundary closure.

    The differences are the *outlet sensitivity* of the loss coefficient: an
    expansion loss that moves when the outflow treatment changes is not yet a
    property of the expansion.
    """
    return {name: run_case(area_ratio, reynolds, cells, length=length, right=name,
                           tolerance=tolerance)[0]
            for name in closures}


def sensitivity(study: dict[str, ClosureTerms]) -> dict[str, float]:
    """Max relative spread of ``k_measure`` across the compared exit closures."""
    values = {name: terms.k_measure for name, terms in study.items()}
    if len(values) < 2:
        return {"spread": 0.0, "reference": next(iter(values.values()))}
    reference = values["outlet"]
    spread = max(abs(value - reference) for value in values.values())
    return {"spread": spread / abs(reference) if reference else float("nan"),
            "reference": reference, "values": values}