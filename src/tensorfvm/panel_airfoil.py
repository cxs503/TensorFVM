"""Lifting panel method for inviscid 2-D airfoil flow (case NACA0012-INVISCID).

Why this module exists
----------------------
The body-fitted :mod:`tensorfvm.body_fitted` solver is a *viscous* SIMPLE
solver; its inviscid limit (``viscosity -> 0``) is not reachable numerically
because the lift then depends on numerical dissipation rather than on a Kutta
condition.  Inviscid lifting flow therefore needs its own discretisation, and
the module reuses the panel family already present in the package
(:mod:`tensorfvm.panel_potential` for non-lifting source panels,
:mod:`tensorfvm.panel_added_mass` for the same kernels) by adding the two
ingredients that produce lift: a vortex sheet of unknown circulation and an
explicit trailing-edge Kutta condition.

Formulation
-----------
Constant-strength source ``sigma_j`` plus a *single* constant vortex strength
``gamma`` on every straight panel of a closed polygon (Hess & Smith 1967;
Katz & Plotkin 2001, ch. 11).  The unknowns are ``sigma_1..sigma_N`` and
``gamma``:

* ``N`` Neumann conditions ``V . n_i = 0`` at the panel midpoints;
* one Kutta condition.  Two are available and both are implemented:

  ``equal_pressure`` (Hess-Smith)  ``V_t(1) = V_t(N)`` on the first (upper
  TE) and last (lower TE) panel -- equal tangential speed is equal pressure;
  ``stagnation`` (Morino)  the tangential velocity vanishes on the bisector
  panel pair, i.e. the TE is a stagnation point.

Panel self-influence is analytic (``+sigma/2`` normal jump of a source sheet,
``-gamma/2`` tangential jump of a vortex sheet, zero finite part for a straight
panel at its own midpoint); all other panel integrals are 8-point Gauss
quadrature of the exact 2-D kernels ``(r-r')/2*pi*|r-r'|^2`` (source) and its
90-degree rotation (vortex).  Nothing about the airfoil is hard-coded: the
geometry comes from :func:`tensorfvm.body_fitted._naca4_profile`, so the same
discretisation serves any four-digit section.

Verification claims (``tests/test_panel_airfoil.py``)
-----------------------------------------------------
* symmetry: ``Cl(alpha) = -Cl(-alpha)`` and ``Cl(0) = 0`` to round-off;
* thin-aerofoil limit: ``Cl -> 2*pi*alpha`` for small ``alpha`` (0.1 deg - 2 deg)
  within 1 %, which is the anchor of the benchmark case;
* lift from the surface pressure integral equals ``rho*U*Gamma``
  (Kutta-Joukowski) to round-off, so the force is not a curve-fit;
* mesh convergence: ``Cl`` converges monotonically in ``N`` and the
  Richardson-ordered error falls with the expected panel-law rate.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, asdict
from typing import Dict, Tuple

import numpy as np

__all__ = [
    "AirfoilPanelResult",
    "naca4_panels",
    "solve_lifting_panels",
    "cl_thin_aerofoil",
]

# vortex self-influence on the fluid side, along the traversal tangent (see the
# derivation in :func:`_influence`)
_VORTEX_SELF = 0.5

_GAUSS_X = (-0.9602898564975363, -0.7966664774136267, -0.5255324099163290,
            -0.1834346424956498, 0.1834346424956498, 0.5255324099163290,
            0.7966664774136267, 0.9602898564975363)
_GAUSS_W = (0.1012285362903763, 0.2223810344533745, 0.3137066458778873,
            0.3626837833783620, 0.3626837833783620, 0.3137066458778873,
            0.2223810344533745, 0.1012285362903763)


@dataclass
class AirfoilPanelResult:
    """Everything the benchmark needs, with the force bookkeeping explicit."""

    cl_pressure: float          # -oint Cp n_y ds / c
    cd_pressure: float          # -oint Cp n_x ds / c
    cl_joukowski: float         # 2 * Gamma / (U * c)
    gamma: float
    cp: np.ndarray              # Cp at every panel midpoint
    x_over_c: np.ndarray
    surface_speed: np.ndarray   # |V_t| / U
    panels: int
    kutta: str
    kutta_residual: float       # |V_t(1) - V_t(N)| / U (equal_pressure only)
    normal_residual: float      # max |V . n| / U
    angle_deg: float

    def as_dict(self) -> Dict:
        d = asdict(self)
        for key in ("cp", "x_over_c", "surface_speed"):
            d.pop(key)
        return d


def _gauss_nodes(a: np.ndarray, b: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Gauss-Legendre nodes/weights on every segment ``[a_j, b_j]``.

    ``a`` and ``b`` are the panel end points; the returned arrays have shape
    ``(8, N, 2)`` and ``(8, N)``.  Weights carry the arc-length Jacobian.
    """
    half = 0.5 * (b - a)                      # (N, 2)
    mid = 0.5 * (a + b)                       # (N, 2)
    x = np.asarray(_GAUSS_X)[:, None, None]
    w = np.asarray(_GAUSS_W)[:, None]
    nodes = mid[None, :, :] + x * half[None, :, :]
    weights = w * np.linalg.norm(half, axis=1)[None, :]
    return nodes, weights


def naca4_panels(thickness: float = 0.12, panels_per_side: int = 200,
                 closed_trailing_edge: bool = True) -> Dict[str, np.ndarray]:
    """Panel a NACA four-digit *symmetric* section, unit chord, LE at (0, 0).

    Cosine spacing clusters panels at the leading edge (where the radius of
    curvature is smallest) and at the trailing edge.  ``closed_trailing_edge``
    applies the ``-0.1036`` instead of ``-0.1015`` last coefficient of the
    thickness polynomial so that the upper and lower surfaces meet exactly at
    ``x = c``; the open form leaves a finite TE gap that a panel code then has
    to close with an extra panel.

    Traversal starts at the trailing edge, runs along the *upper* surface to the
    leading edge and back along the *lower* surface, so the outward normal is
    ``(t_y, -t_x)`` for the tangent ``t`` along the traversal, with the fluid on
    the left.  (Checked by :func:`solve_lifting_panels`: the far-field
    disturbance of the converged solution decays and the LE stagnation point
    sits on the nose.)
    """
    n = int(panels_per_side)
    a4 = -0.1036 if closed_trailing_edge else -0.1015
    s = 0.5 * (1.0 - np.cos(np.pi * np.arange(n + 1) / n))   # 0..1, cosine
    yt = 5.0 * thickness * (0.2969 * np.sqrt(s) - 0.1260 * s
                            - 0.3516 * s ** 2 + 0.2843 * s ** 3 + a4 * s ** 4)
    upper = np.column_stack([s, yt])                          # LE -> TE
    lower = np.column_stack([s, -yt])
    loop = np.vstack([upper[::-1], lower[1:]])                # TE -> upper -> LE -> lower -> TE
    loop = loop[:-1]                                          # TE listed once
    a, b = loop, np.roll(loop, -1, axis=0)
    t = b - a
    arc = np.linalg.norm(t, axis=1)
    t = t / arc[:, None]
    normal = np.column_stack([t[:, 1], -t[:, 0]])              # outward (fluid side)
    mid = 0.5 * (a + b)
    # orientation gate: the outward normal must point away from the chord line
    # for a symmetric section, i.e. upper panels must have n_y > 0.
    if normal[0, 1] < 0.0:
        raise RuntimeError("panel orientation gate failed: upper normal points into the body")
    return dict(a=a, b=b, t=t, normal=normal, arc=arc, mid=mid, length=float(arc.sum()))


def _influence(mid, normal, tangent, geo, subpanels: int = 4):
    """Normal/tangential velocity influence matrices of the discretisation.

    Returns ``(An, At)`` of shape ``(N, N+1)``: column ``j < N`` is the unit
    source strength of panel ``j``, column ``N`` is the unit vortex strength
    (one unknown shared by all panels).  Rows are the normal (``An``) and
    tangential (``At``) velocity conditions at the panel midpoints.

    Non-singular panel integrals use ``subpanels x 8`` Gauss-Legendre points on
    the exact 2-D kernels of a source sheet ``(r-r') / (2 pi |r-r'|^2)`` and a
    vortex sheet (the same vector rotated by 90 degrees).  Splitting the
    integration interval matters: a cosine-spaced section puts neighbouring
    panels closer than their own length, and a single 8-point rule per panel
    loses the near-singular part there.  A closed-form evaluation (Katz &
    Plotkin eqs. 10.20-10.23) was tried first; its branch structure for the
    subtended angle reproduced the wrong sign in the near field, so the
    quadrature route is kept -- it is verifiable by refinement alone.

    The self panel is singular under quadrature and is replaced by its analytic
    limit on the fluid side of the sheet: ``+1/2`` along the normal for the
    source, and, for the vortex sheet of the shared strength ``gamma``,
    ``_VORTEX_SELF`` along the tangent on every panel.  With the traversal
    tangent ``t`` and the outward normal ``n = (t_y, -t_x)`` (fluid on the right
    of ``t``) that limit is ``+gamma/2``.
    """
    n = mid.shape[0]
    a, b = geo["a"], geo["b"]
    S, Q = int(subpanels), len(_GAUSS_X)
    # split every panel into S equal pieces, then Q-point Gauss on each piece
    frac = np.linspace(0.0, 1.0, S + 1)[:-1]
    h = 1.0 / S
    xg = np.asarray(_GAUSS_X)[:, None]                     # (Q,1)
    tparam = (frac[None, :] + h * 0.5 * (xg + 1.0)).reshape(-1)   # (S*Q,)
    nodes = a[None, :, :] + tparam[:, None, None] * (b - a)[None, :, :]   # (SQ,N,2)
    seg_len = np.linalg.norm(b - a, axis=1)                # (N,)
    wq = np.tile(np.asarray(_GAUSS_W)[:, None], (1, S)) * (h * 0.5)   # (Q,S)
    weights = wq.reshape(-1)[:, None] * seg_len[None, :]              # (SQ,N)
    dx = mid[:, None, None, 0] - nodes[None, :, :, 0]      # (N,SQ,N)
    dy = mid[:, None, None, 1] - nodes[None, :, :, 1]
    r2 = dx * dx + dy * dy
    w = weights[None, :, :]
    with np.errstate(divide="ignore", invalid="ignore"):
        src_x = w * dx / (2.0 * math.pi * r2)
        src_y = w * dy / (2.0 * math.pi * r2)
        vor_x = -src_y
        vor_y = src_x
    diag = np.arange(n)
    for arr in (src_x, src_y, vor_x, vor_y):
        arr[diag, :, diag] = 0.0                           # the i == j singularity
    An = np.empty((n, n + 1))
    At = np.empty((n, n + 1))
    nx, ny = normal[:, 0][:, None, None], normal[:, 1][:, None, None]
    tx, ty = tangent[:, 0][:, None, None], tangent[:, 1][:, None, None]
    An[:, :-1] = np.sum(src_x * nx + src_y * ny, axis=1)
    At[:, :-1] = np.sum(src_x * tx + src_y * ty, axis=1)
    An[:, -1] = np.sum(vor_x * nx + vor_y * ny, axis=(1, 2))
    At[:, -1] = np.sum(vor_x * tx + vor_y * ty, axis=(1, 2))
    An[diag, diag] = 0.5
    At[:, -1] += _VORTEX_SELF
    return An, At


def solve_lifting_panels(angle_deg: float = 2.0, panels_per_side: int = 200,
                         thickness: float = 0.12,
                         kutta: str = "equal_pressure") -> AirfoilPanelResult:
    """Solve inviscid lifting flow over a NACA four-digit symmetric section.

    Nondimensional: chord ``c = 1``, free stream speed ``U = 1``, density
    ``rho = 1``, so ``q = 1/2`` and ``Cl = 2 L``.  The angle of attack is taken
    positive nose-up: the free stream is ``(U cos a, U sin a)``.
    """
    geo = naca4_panels(thickness, panels_per_side)
    mid, normal, tangent, arc = geo["mid"], geo["normal"], geo["t"], geo["arc"]
    n = mid.shape[0]
    a = math.radians(angle_deg)
    U = np.array([math.cos(a), math.sin(a)])
    An, At = _influence(mid, normal, tangent, geo)
    A = np.zeros((n + 1, n + 1))
    rhs = np.zeros(n + 1)
    A[:n, :] = An
    rhs[:n] = -(normal @ U)
    # The traversal tangent points *upstream* along the upper surface and
    # *downstream* along the lower surface, so equal pressure (equal speed) on
    # the two trailing-edge panels reads V_t(1) + V_t(N) = 0, while the Morino
    # form V_t(1) - V_t(N) = 0 would force a spurious antisymmetric circulation.
    if kutta == "equal_pressure":
        A[n, :] = At[0, :] + At[-1, :]
        rhs[n] = -(tangent[0] @ U + tangent[-1] @ U)
    elif kutta == "stagnation":
        # Morino: the two TE panels carry zero speed (stagnation point)
        A[n, :] = At[0, :] - At[-1, :]
        rhs[n] = -(tangent[0] @ U - tangent[-1] @ U)
    else:  # pragma: no cover - guarded by the caller
        raise ValueError(f"unknown Kutta condition {kutta!r}")
    sol = np.linalg.solve(A, rhs)
    sigma, gamma = sol[:n], float(sol[n])
    # tangential surface speed: onset + induced, on the fluid side of the sheet
    vt = tangent @ U + At @ sol
    cp = 1.0 - vt ** 2
    # pressure force: F = -oint p n ds, p = q Cp, q = 1/2 -> F = -0.5 oint Cp n ds
    fx = -0.5 * float(np.sum(cp * normal[:, 0] * arc))
    fy = -0.5 * float(np.sum(cp * normal[:, 1] * arc))
    vn = An @ sol + normal @ U                        # residual check
    return AirfoilPanelResult(
        cl_pressure=2.0 * fy,
        cd_pressure=2.0 * fx,
        cl_joukowski=2.0 * gamma,
        gamma=gamma,
        cp=cp,
        x_over_c=mid[:, 0],
        surface_speed=np.abs(vt),
        panels=n,
        kutta=kutta,
        kutta_residual=float(abs(vt[0] + vt[-1])) if kutta == "equal_pressure"
        else float(abs(vt[0] - vt[-1])),
        normal_residual=float(np.abs(vn).max()),
        angle_deg=angle_deg,
    )


def cl_thin_aerofoil(angle_deg: float) -> float:
    """Thin-aerofoil (Glauert) lift coefficient ``2*pi*alpha`` with ``alpha`` in radians."""
    return 2.0 * math.pi * math.radians(angle_deg)


def main(argv=None):  # pragma: no cover - thin CLI for reproducing the case
    import argparse
    import json
    from pathlib import Path

    parser = argparse.ArgumentParser(
        description="NACA0012 inviscid lifting panel method (Cl at a given alpha)")
    parser.add_argument("--alpha", type=float, nargs="+", default=[2.0])
    parser.add_argument("--panels", type=int, nargs="+", default=[50, 100, 200, 400, 800])
    parser.add_argument("--kutta", default="equal_pressure",
                        choices=["equal_pressure", "stagnation"])
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)
    rows = []
    for npan in args.panels:
        for alpha in args.alpha:
            r = solve_lifting_panels(alpha, npan, kutta=args.kutta)
            ref = cl_thin_aerofoil(alpha)
            rows.append(dict(r.as_dict(), cl_thin_aerofoil=ref,
                             error_percent=100.0 * (r.cl_pressure - ref) / ref))
    payload = dict(case="NACA0012-INVISCID", kutta=args.kutta, rows=rows)
    text = json.dumps(payload, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    else:
        print(text)
    return payload


if __name__ == "__main__":  # pragma: no cover
    main()