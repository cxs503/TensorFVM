"""Isentropic vortex: the exact solution the compressible solver is missing.

The compressible surface of :mod:`tensorfvm` was verified against *discontinuous*
exact solutions only (Sod shock tube, two-gas contact, Richtmyer--Meshkov,
double blast).  Those pin the shock/contact *positions* and the star states, so
they prove that the Riemann solver is right, but they cannot measure the
truncation error of the smooth flow the scheme is actually second order for: a
convergence study needs an exact solution that is smooth everywhere, so that the
error norm decreases with the grid at the design rate instead of at the rate of
the smeared discontinuity.

The standard smooth 2-D Euler verification is the isentropic vortex (Shu 1998,
*SIAM Rev.* 40, 743; the same case is used by every high-order code).  With
``r**2 = X**2 + Y**2`` measured from the (advecting) core,

.. math::

   \\delta u = -\\frac{\\beta}{2\\pi} Y e^{(1-r^2)/2}, \\qquad
   \\delta v = +\\frac{\\beta}{2\\pi} X e^{(1-r^2)/2},

   T = 1 - \\frac{(\\gamma-1)\\beta^2}{8\\gamma\\pi^2} e^{1-r^2}, \\qquad
   \\rho = T^{1/(\\gamma-1)}, \\qquad p = \\rho T = \\rho^{\\gamma},

the last identity because the vortex is *isentropic* (``p/rho**gamma = 1``): the
pressure work of the swirl is exactly balanced by the temperature drop, which is
why the field has no acoustic signature and therefore no boundary-layer-in-time
error to pollute the order measurement.  With a uniform free stream ``(U, V)``
the whole field is an exact solution of the Euler equations that simply
translates at ``(U, V)``, so on a *periodic* box the exact solution at any time
is the initial field evaluated at the shifted centre -- no interpolation, no
characteristic decomposition, no numerical reference to trust.

What this module makes executable
---------------------------------
``vortex_primitive``
    The exact solution as a function of ``(x, y, t)``, with the periodic image
    of the core chosen by minimum-image wrapping.  On the default 10 x 10 box the
    nearest *neglected* image sits ``r = 5`` from the wall cells, i.e. amplitude
    ``e^{(1-25)/2} = 6.1e-6`` -- three to four orders below the N=32 truncation
    error and dt-independent, but *not* double precision; the envelope is pinned
    by ``tests/test_compressible_isentropic_vortex.py::
    test_single_image_representation_is_bounded_by_the_nearest_image`` and
    ``envelope.json:periodic_image``.
``set_vortex`` / ``initial_state``
    Install the ``t``-snapshot as the solver state.
``error_norms``
    ``L1``/``L2``/``Linf`` between a numerical and the exact field, normalised by
    the exact field's own norm so the numbers are scale free.
``observed_order``
    ``log(e_coarse/e_fine)/log(h_coarse/h_fine)`` for a grid triple, i.e. the
    Richardson estimate of the scheme's order, with no assumption about the
    error constant.
``mass_and_momentum``
    The two invariants a flux-form scheme on a periodic box must conserve to
    round-off.  They are the *passivity* counterpart of the order test: an order
    test can be satisfied by a scheme that leaks mass, a conservation test by one
    that is only first order, so a credible verification needs both.

The verification harness lives in ``scripts/verify_compressible_vortex_order.py``
and the gate in ``tests/test_compressible_isentropic_vortex.py``.
"""

from __future__ import annotations

import math
from typing import Dict, Iterable, List, Sequence

import torch

from .eos import StiffenedGasMixture
from .solver import EulerSolver

__all__ = [
    "GAMMA_AIR",
    "VORTEX_DEFAULTS",
    "single_gas_eos",
    "cell_centres",
    "vortex_primitive",
    "initial_state",
    "set_vortex",
    "run_vortex",
    "error_norms",
    "observed_order",
    "mass_and_momentum",
    "convergence_study",
]

GAMMA_AIR = 1.4

#: Default case: unit free stream along x, box 10x10 (>> 10 core radii), the
#: vortex core in the box centre, swirl strength ``beta = 5`` (the Shu value that
#: keeps the minimum temperature ``T = 0.8825`` comfortably positive).
VORTEX_DEFAULTS = {
    "beta": 5.0,
    "u_inf": 1.0,
    "v_inf": 0.0,
    "gamma": GAMMA_AIR,
    "t_end": 1.0,
    "lx": 10.0,
    "ly": 10.0,
}


def single_gas_eos(gamma: float = GAMMA_AIR) -> StiffenedGasMixture:
    """Ideal-gas EOS on the solver's mandatory two-material interface.

    The Euler solver requires ``nmat == 2``; declaring both materials identical
    and setting ``Y = 0`` makes the mixture EOS an exact single ideal gas
    (``gamma(Y=0) = gamma``, ``p_inf = 0``), so the vortex case exercises the
    gas dynamics and not the interface machinery.
    """
    return StiffenedGasMixture([gamma, gamma], names=["air", "air"])


def cell_centres(nx: int, ny: int, lx: float, ly: float):
    """Cell-centre coordinates shaped ``(nx, ny)`` to match the solver state."""
    dx, dy = lx / nx, ly / ny
    x = (torch.arange(nx, dtype=torch.float64) + 0.5) * dx
    y = (torch.arange(ny, dtype=torch.float64) + 0.5) * dy
    return x.view(-1, 1).expand(nx, ny), y.view(1, -1).expand(nx, ny)


def _minimum_image(x: torch.Tensor, lx: float) -> torch.Tensor:
    """Wrap a displacement into ``[-lx/2, lx/2)`` (nearest periodic image)."""
    return torch.remainder(x + 0.5 * lx, lx) - 0.5 * lx


def vortex_primitive(
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    t: float = 0.0,
    lx: float = VORTEX_DEFAULTS["lx"],
    ly: float = VORTEX_DEFAULTS["ly"],
    x0: float | None = None,
    y0: float | None = None,
    u_inf: float = VORTEX_DEFAULTS["u_inf"],
    v_inf: float = VORTEX_DEFAULTS["v_inf"],
    beta: float = VORTEX_DEFAULTS["beta"],
    gamma: float = VORTEX_DEFAULTS["gamma"],
) -> Dict[str, torch.Tensor]:
    """Exact isentropic-vortex state at time ``t`` on the periodic box.

    Returns ``rho, u, v, p, Y`` broadcast to ``x``'s shape.  ``x0, y0`` default
    to the box centre; the core position at time ``t`` is ``(x0 + u_inf t,
    y0 + v_inf t)`` taken modulo the box, so the field is periodic in ``t`` with
    period ``lx/u_inf``.
    """
    x0 = 0.5 * lx if x0 is None else x0
    y0 = 0.5 * ly if y0 is None else y0
    dx = _minimum_image(x - (x0 + u_inf * t), lx)
    dy = _minimum_image(y - (y0 + v_inf * t), ly)
    r2 = dx * dx + dy * dy
    g = torch.exp(0.5 * (1.0 - r2))
    swirl = beta / (2.0 * math.pi)
    # temperature drop that balances the swirl pressure work exactly
    t_base = 1.0 - (gamma - 1.0) * beta * beta / (8.0 * gamma * math.pi ** 2) * torch.exp(1.0 - r2)
    rho = t_base ** (1.0 / (gamma - 1.0))
    p = rho ** gamma
    return {
        "rho": rho,
        "u": u_inf - swirl * dy * g,
        "v": v_inf + swirl * dx * g,
        "p": p,
        "Y": torch.zeros_like(rho),
    }


def initial_state(
    nx: int,
    ny: int,
    *,
    lx: float = VORTEX_DEFAULTS["lx"],
    ly: float = VORTEX_DEFAULTS["ly"],
    **kwargs,
) -> Dict[str, torch.Tensor]:
    """Exact solution sampled at the cell centres of an ``nx x ny`` grid."""
    x, y = cell_centres(nx, ny, lx, ly)
    return vortex_primitive(x, y, lx=lx, ly=ly, **kwargs)


def set_vortex(solver: EulerSolver, *, t: float = 0.0, **kwargs) -> None:
    """Initialise ``solver`` with the exact vortex field at time ``t``."""
    kw = dict(lx=solver.lx, ly=solver.ly)
    kw.update(kwargs)
    state = initial_state(solver.nx, solver.ny, t=t, **kw)
    solver.set_primitive(state)


def run_vortex(
    nx: int,
    ny: int | None = None,
    *,
    order: int = 2,
    t_end: float = VORTEX_DEFAULTS["t_end"],
    cfl: float = 0.3,
    limiter: str = "mc",
    flux_mode: str = "hllc",
    lx: float = VORTEX_DEFAULTS["lx"],
    ly: float = VORTEX_DEFAULTS["ly"],
    beta: float = VORTEX_DEFAULTS["beta"],
    u_inf: float = VORTEX_DEFAULTS["u_inf"],
    v_inf: float = VORTEX_DEFAULTS["v_inf"],
    gamma: float = VORTEX_DEFAULTS["gamma"],
    x0: float | None = None,
    y0: float | None = None,
):
    """Advect the vortex to ``t_end`` on a fully periodic box.

    Returns ``(solver, exact)`` where ``exact`` is the analytic field at
    ``t_end`` on the same grid.  A uniform free stream plus periodic boundaries
    means the only error sources are the reconstruction, the HLLC flux and the
    time integrator -- no inflow/outflow closure, no wall model.
    """
    ny = nx if ny is None else ny
    solver = EulerSolver(
        nx=nx,
        ny=ny,
        lx=lx,
        ly=ly,
        eos=single_gas_eos(gamma),
        bc=("periodic", "periodic", "periodic", "periodic"),
        flux_mode=flux_mode,
        order=order,
        limiter=limiter,
        cfl=cfl,
    )
    kw = dict(
        lx=lx, ly=ly, beta=beta, u_inf=u_inf, v_inf=v_inf, gamma=gamma,
        x0=x0, y0=y0,
    )
    set_vortex(solver, t=0.0, **kw)
    t = 0.0
    while t < t_end - 1e-15:
        dt = min(solver.stable_dt(), t_end - t)
        solver.step(dt)
        t += dt
    exact = initial_state(nx, ny, t=t_end, **kw)
    return solver, exact


def error_norms(numeric: torch.Tensor, exact: torch.Tensor) -> Dict[str, float]:
    """``L1`` (mean absolute), ``L2`` and ``Linf``, each scaled by the exact norm.

    Scaling by the exact norm makes the errors comparable across variables whose
    magnitudes differ by an order of magnitude (``p ~ 1``, ``delta u ~ 0.6``) and
    keeps the reported order invariant to that choice.
    """
    diff = numeric - exact
    l1 = float(torch.mean(torch.abs(diff)))
    l2 = float(torch.sqrt(torch.mean(diff * diff)))
    linf = float(torch.max(torch.abs(diff)))
    scale_l1 = float(torch.mean(torch.abs(exact))) or 1.0
    scale_l2 = float(torch.sqrt(torch.mean(exact * exact))) or 1.0
    scale_inf = float(torch.max(torch.abs(exact))) or 1.0
    return {
        "L1": l1 / scale_l1,
        "L2": l2 / scale_l2,
        "Linf": linf / scale_inf,
        "L1_abs": l1,
        "L2_abs": l2,
        "Linf_abs": linf,
    }


def observed_order(coarse: float, fine: float, ratio: float = 2.0) -> float:
    """Richardson estimate ``log(e_c/e_f)/log(ratio)``; ``nan`` if degenerate."""
    if coarse <= 0.0 or fine <= 0.0:
        return float("nan")
    return math.log(coarse / fine) / math.log(ratio)


def mass_and_momentum(prim: Dict[str, torch.Tensor], hx: float, hy: float) -> Dict[str, float]:
    """Discrete mass and x-momentum integrals (the periodic-box invariants)."""
    cell = hx * hy
    return {
        "mass": float(torch.sum(prim["rho"])) * cell,
        "momentum_x": float(torch.sum(prim["rho"] * prim["u"])) * cell,
        "momentum_y": float(torch.sum(prim["rho"] * prim["v"])) * cell,
    }


def convergence_study(
    grids: Sequence[int] = (32, 64, 128),
    *,
    order: int = 2,
    variables: Iterable[str] = ("rho", "u", "v", "p"),
    **kwargs,
) -> Dict[str, object]:
    """Run one grid family and report errors, orders and the conservation drift.

    ``kwargs`` are forwarded to :func:`run_vortex` (``t_end``, ``cfl``, ``beta``,
    ``limiter``, ``flux_mode``, ...).  Every number in the returned dict is
    computed here, so a report quoting it can be regenerated byte for byte.
    """
    grids = list(grids)
    rows: List[Dict[str, float]] = []
    for n in grids:
        solver, exact = run_vortex(n, order=order, **kwargs)
        num = solver.primitive()
        dx, dy = solver.dx, solver.dy
        init = initial_state(
            n,
            n,
            lx=solver.lx,
            ly=solver.ly,
            t=0.0,
            beta=kwargs.get("beta", VORTEX_DEFAULTS["beta"]),
            u_inf=kwargs.get("u_inf", VORTEX_DEFAULTS["u_inf"]),
            v_inf=kwargs.get("v_inf", VORTEX_DEFAULTS["v_inf"]),
            gamma=kwargs.get("gamma", VORTEX_DEFAULTS["gamma"]),
            x0=kwargs.get("x0"),
            y0=kwargs.get("y0"),
        )
        row: Dict[str, float] = {"n": n, "h": dx, "steps": solver.nsteps}
        for name in variables:
            row.update(
                {f"{name}_{k}": v for k, v in error_norms(num[name], exact[name]).items()}
            )
        inv0 = mass_and_momentum(init, dx, dy)
        inv1 = mass_and_momentum(num, dx, dy)
        row["mass_drift"] = (inv1["mass"] - inv0["mass"]) / inv0["mass"]
        row["momentum_x_drift"] = (
            inv1["momentum_x"] - inv0["momentum_x"]
        ) / abs(inv0["momentum_x"])
        row["min_rho"] = float(torch.min(num["rho"]))
        row["min_p"] = float(torch.min(num["p"]))
        rows.append(row)

    orders: Dict[str, List[float]] = {}
    for name in variables:
        seq = [row[f"{name}_L1"] for row in rows]
        orders[name] = [
            observed_order(seq[i], seq[i + 1]) for i in range(len(seq) - 1)
        ]
    return {
        "case": "isentropic_vortex",
        "order": order,
        "settings": {k: v for k, v in kwargs.items()},
        "grids": grids,
        "rows": rows,
        "observed_orders": orders,
    }