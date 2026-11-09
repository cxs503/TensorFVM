"""Control-volume momentum ledger: the integral closure chain.

The steady incompressible x-momentum equation integrated over a full-height
control volume ``[x0, x1] x [0, H]`` of a channel (no-slip walls at
``y = 0`` and ``y = H``) is the *ledger*

    [int rho u^2 dy]_{x0}^{x1}
    - int p(x1, y) dy + int p(x0, y) dy          (pressure, open + solid part)
    + int_{x0}^{x1} tau_w,top dx - int_{x0}^{x1} tau_w,bot dx   =  0,

where the upstream pressure integral runs over the whole plane: the open part
carries the inlet pressure and the solid part (the step face of a sudden
expansion or of a backward-facing step) carries the step-face pressure, which
the discrete solution provides directly.  The ``uv`` wall terms vanish by
no-slip.  Every entry of the ledger is evaluated from the *discrete* fields of
a converged :class:`~tensorfvm.internal_flow.mac.StaggeredMAC` solution, so
the closure residual

    |LHS| / (rho U_ref^2 A_open)

measures how well the discrete solution satisfies the exact integral balance.
A converged solution closes the ledger to the level of the time-marching
tolerance; the campaign gate is 1 %.

The same routine serves the verification cases:

* sudden expansion -- ``x0 = 0`` (partly open inlet), ``x1 = L`` (outlet);
* backward-facing step -- ``x0 = 0`` (partly open inlet), ``x1 = L``;
* fully open channel -- any two stations, no solid part.

The wall shear uses the same mirror-ghost gradient that the solver uses for
the wall friction, so the ledger and the reported ``C_f`` share one
discretisation.
"""

from __future__ import annotations

import math

import torch

from .mac import StaggeredMAC


def _wall_shear(solver: StaggeredMAC, wall: str) -> torch.Tensor:
    """Wall shear ``tau_w = rho nu du/dy`` along the bottom or top wall.

    ``du/dy`` is the mirror-ghost one-sided difference, identical to the one
    used by :meth:`StaggeredMAC.bottom_wall_friction`; positive at the bottom
    wall for forward flow, negative at the top wall.
    """
    row = 0 if wall == "bottom" else solver.ny - 1
    sign = 1.0 if wall == "bottom" else -1.0
    du_dy = sign * 2.0 * solver.u[:, row] / solver.hy
    return solver.rho * solver.nu * du_dy


def _open_mask(solver: StaggeredMAC, i: int) -> torch.Tensor:
    """Boolean mask of the open rows of face column ``i``.

    Only the upstream plane of an inlet case is partly solid; every other
    column is fully open.
    """
    if i == 0 and solver.config.left == "inlet" and solver.config.inlet_profile is not None:
        return torch.tensor([v > 0.0 for v in solver.config.inlet_profile],
                            dtype=torch.bool, device=solver.device)
    return torch.ones(solver.ny, dtype=torch.bool, device=solver.device)


def channel_momentum_ledger(solver: StaggeredMAC,
                           x0: float = 0.0, x1: float | None = None,
                           reference_velocity: float | None = None) -> dict[str, float]:
    """Integral x-momentum ledger of the control volume ``[x0, x1] x [0, H]``.

    Parameters
    ----------
    solver:
        Converged MAC solution.
    x0, x1:
        Streamwise extent of the control volume in metres; defaults to the
        whole domain.  Both planes must coincide with face columns.
    reference_velocity:
        Velocity that normalises the closure residual (defaults to the
        solver's ``reference_velocity``).

    Returns
    -------
    dict
        Every ledger entry (momentum fluxes, pressure forces, wall friction)
        in SI units, the closure residual ``|LHS| / (rho U_ref^2 A_open)``,
        and the individual terms so a report can show the chain.
    """
    cfg = solver.config
    rho = solver.rho
    H = cfg.height
    u_ref = cfg.reference_velocity if reference_velocity is None else float(reference_velocity)
    # The outlet face column is set by the boundary condition (zero-gradient
    # plus the global flux correction), not by the momentum equation, so the
    # default control volume ends at the last *interior* face column where the
    # momentum equation is actually solved.
    x1 = (solver.nx - 1) * solver.hx if x1 is None else float(x1)
    x0 = float(x0)
    if not (0.0 <= x0 < x1 <= cfg.length + 1e-12):
        raise ValueError("the control volume must lie inside the domain")

    i0 = int(round(x0 / solver.hx))
    i1 = int(round(x1 / solver.hx))
    if i1 > solver.nx - 1:
        raise ValueError("the outlet plane must be an interior face column")
    if abs(i0 * solver.hx - x0) > 1e-9 or abs(i1 * solver.hx - x1) > 1e-9:
        raise ValueError("the control-volume planes must coincide with face columns")

    mask_in = _open_mask(solver, i0)
    mask_out = _open_mask(solver, i1)

    def column_momentum_flux(i: int, mask: torch.Tensor) -> float:
        u = solver.u[i, :]
        return float(rho * (u ** 2)[mask].sum() * solver.hy)

    def column_pressure(i: int, mask: torch.Tensor) -> tuple[float, float]:
        """``(int p dy over the open rows, int p dy over the solid rows)``."""
        p = solver.p[i - 1 if i > 0 else 0, :]
        return (float(p[mask].sum() * solver.hy),
                float(p[~mask].sum() * solver.hy))

    mom_in = column_momentum_flux(i0, mask_in)
    mom_out = column_momentum_flux(i1, mask_out)
    p_open_in, p_solid_in = column_pressure(i0, mask_in)
    p_open_out, _ = column_pressure(i1, mask_out)

    tau_bot = _wall_shear(solver, "bottom")
    tau_top = _wall_shear(solver, "top")
    friction = float(torch.trapz(tau_top[i0:i1 + 1], dx=solver.hx)
                   - torch.trapz(tau_bot[i0:i1 + 1], dx=solver.hx))

    lhs = ((mom_out - mom_in)
           + (-p_open_out + p_open_in + p_solid_in)
           + friction)
    a_open = float(mask_in.sum()) * solver.hy
    scale = rho * u_ref ** 2 * a_open
    return {
        "x0_m": x0, "x1_m": x1,
        "momentum_flux_in": mom_in,
        "momentum_flux_out": mom_out,
        "pressure_force_open_in": p_open_in,
        "pressure_force_solid_in": p_solid_in,
        "pressure_force_out": p_open_out,
        "wall_friction": friction,
        "closure_lhs": lhs,
        "closure_residual": abs(lhs) / scale,
        "normalisation": f"rho U_ref^2 A_open = {scale:.6g}",
    }


def momentum_ledger_percent(ledger: dict[str, float]) -> float:
    """The closure residual of :func:`channel_momentum_ledger` in percent."""
    return 100.0 * float(ledger["closure_residual"])
