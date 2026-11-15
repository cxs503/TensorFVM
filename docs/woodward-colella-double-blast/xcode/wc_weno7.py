"""WENO7 reconstruction probe for the Woodward-Colella two-blast-wave case.

Why this file exists
-------------------
``docs/XVAL-FVM-M18737-ROUND2/CERTIFIED-REF-CLOSURE.md`` closed the round-2
certified-<1 % question with an *honest FAIL*: the TensorFVM MUSCL/HLLC
nx=3200 solution agrees with the independent WENO5@12800 reference to
L1 = 1.776 % (> 1 %) and to 0.758 % in peak density.  It also asserted, but
did not measure, that the residual is "dragged by the contact discontinuity /
plateau region" and that "uniform refinement gains are already flat between
6400 and 12800".

The open question that bounds the *next* round is an **attribution** question:
of the 1.776 %, how much is still *reference-side* (the WENO5 reference's own
reconstruction error at contacts / in the rarefied plateau) and how much is
*graded-side* (the second-order MUSCL/HLLC discretisation)?  A finer WENO5
grid cannot answer it: the same family converges to the same (possibly biased)
limit.  A **higher-order** member of the independent family can.

This module is that probe: Jiang & Shu (1996) 7th-order WENO reconstruction
(4 sub-stencils of 4 points, linear weights 1/35, 12/35, 18/35, 4/35) with
SSP-RK3 and the same Rusanov/Lax-Friedrichs interface flux as
``wc_crosscode.py``, on the same cell centres.

Independence caveat (stated up front, not buried)
-------------------------------------------------
WENO7 shares the time integrator (SSP-RK3) and the interface flux (Rusanov)
with the WENO5 reference, so it is **not** an independent *code*.  It is an
independent *reconstruction order* probe: it isolates the spatial
reconstruction error of the reference family, which is exactly the quantity the
round-2 attribution argument needs.  Cross-code independence is still provided
by ``wc_crosscode.py`` (finite difference / Rusanov / WENO5) versus TensorFVM
(finite volume / HLLC / MUSCL-Hancock).

Where the coefficients come from
--------------------------------
No smoothness-indicator constant is typed from memory.  For every sub-stencil

* the face-value coefficients are ``c = [1, 1/2, 1/4, 1/8] @ V^{-1}`` (the
  unique cubic through the four cell values, evaluated at ``xi = 1/2``);
* the smoothness indicator is the Jiang-Shu functional
  ``beta = sum_l dx^{2l-1} int (d^l p/dxi^l)^2 dxi`` evaluated **exactly** as a
  quadratic form ``beta = v^T Q v`` with ``Q = V^{-T} (sum_l D_l^T C D_l) V^{-1}``
  and ``C`` the analytic Gram matrix of monomials on ``[-1/2, 1/2]``.

``tests/test_wc_ref_attribution.py`` checks that these coefficients reproduce
an exact degree-6 polynomial (the condition for optimal 7th order) and that
the reconstruction actually attains high observed order on a smooth field.

Usage::

    PYTHONPATH=src python docs/woodward-colella-double-blast/xcode/wc_weno7.py \
        --order-study
    PYTHONPATH=src python docs/woodward-colella-double-blast/xcode/wc_weno7.py \
        --nx 3200 --out docs/woodward-colella-double-blast/ladder/data
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Dict, Sequence, Tuple

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:  # reuse the case datum / flux conventions
    sys.path.insert(0, str(HERE))

import wc_crosscode as xc  # noqa: E402

GAMMA = xc.GAMMA
WC_T_END = xc.WC_T_END

#: 7th-order WENO: four 4-point sub-stencils around the cell of face ``i+1/2``
NSUB = 4
STENCIL = 7
SUB_OFFSETS = ((-3, -2, -1, 0), (-2, -1, 0, 1), (-1, 0, 1, 2), (0, 1, 2, 3))
EPS_JS = 1e-6


def _vander(offsets: Sequence[int]) -> np.ndarray:
    """Standard Vandermonde: row ``j`` holds the powers of ``offsets[j]``.

    With ``a`` the monomial coefficients of the interpolating polynomial,
    ``v_j = p(offsets[j]) = sum_m a_m offsets[j]^m``, i.e. ``v = Vm a`` and
    ``a = Vm^{-1} v``.
    """
    n = len(offsets)
    return np.array([[float(s) ** m for m in range(n)] for s in offsets], float)


def _poly_face_coeffs(offsets: Sequence[int]) -> np.ndarray:
    """*Point-value* face coefficients (exact for the polynomial through the
    cell-centre values) -- kept for the record / for the reconstruction note.

    ``p(1/2) = e^T a = e^T Vm^{-1} v`` with ``e = [1, 1/2, 1/4, 1/8]``, so
    ``c = Vm^{-T} e``.
    """
    n = len(offsets)
    e = np.array([0.5 ** m for m in range(n)], float)
    return np.linalg.inv(_vander(offsets)).T @ e


def _fv_face_coeffs(offsets: Sequence[int]) -> np.ndarray:
    """*Cell-average* (finite-volume) face coefficients -- what the solvers need.

    ``offsets`` are consecutive integer cell indices; the face sits at
    ``xi = +1/2`` (units of ``dx``, measured from the centre of cell 0).  The
    primitive ``W(xi) = int v dxi`` is known exactly at the ``n+1`` cell faces
    from the averages, ``W_{j+1} - W_j = vbar_j``; interpolating ``W`` by the
    degree-``n`` polynomial through those knots and differentiating gives
    ``v(1/2) = P'(1/2) = sum_j c_j vbar_j``.  For the three-cell stencil
    ``(-2,-1,0)`` this reproduces the textbook ``(1/3, -7/6, 11/6)``.
    """
    n = len(offsets)
    knots = np.array([offsets[0] - 0.5 + j for j in range(n + 1)], float)
    Vk = _vander(knots)                      # (n+1, n+1) powers of the knots
    e_d = np.array([m * 0.5 ** (m - 1) if m >= 1 else 0.0 for m in range(n + 1)])
    # dW_knot[j] = sum_{m<j} vbar_m   ->  K (n+1, n)
    K = np.array([[1.0 if m < j else 0.0 for m in range(n)] for j in range(n + 1)])
    c = K.T @ (np.linalg.inv(Vk).T @ e_d)
    return c


def _beta_matrix(offsets: Sequence[int]) -> np.ndarray:
    """Exact Jiang-Shu smoothness quadratic form for one sub-stencil."""
    n = len(offsets)
    Vinv = np.linalg.inv(_vander(offsets))
    M = np.zeros((n, n))
    for order in range(1, n):
        D = np.zeros((n, n))
        for m in range(order, n):
            # b = D a  (b_p = coefficient of xi^p in p', p = m - order)
            D[m - order, m] = math.factorial(m) / math.factorial(m - order)
        C = np.zeros((n, n))
        for p in range(n):
            for q in range(n):
                C[p, q] = 0.0 if (p + q) % 2 else 1.0 / (p + q + 1.0)
        M += D.T @ C @ D
    # v = Vm a  =>  a = Vinv v  =>  beta = a^T M a = v^T Vinv^T M Vinv v
    return Vinv.T @ M @ Vinv


#: cell-average (finite-volume) coefficients -- the solvers below are FV updates
FACE_COEFFS = np.array([_fv_face_coeffs(o) for o in SUB_OFFSETS])     # (4, 4)
BETA_MATS = np.array([_beta_matrix(o) for o in SUB_OFFSETS])          # (4, 4, 4)


def _optimal_weights() -> np.ndarray:
    """Linear weights solved -- not typed -- so they reproduce the 7-point face
    interpolation exactly.

    ``sum_k d_k p_k(1/2)`` must equal the degree-6 interpolation of the seven
    cells at ``xi = 1/2``.  That is a 7x4 linear system in ``d``; solving it
    gives ``(1/64, 21/64, 35/64, 7/64)`` with a ~1e-15 residual.  (Applying the
    same construction to the WENO5 stencils reproduces the audited
    ``(0.1, 0.6, 0.3)`` -- see the test file.)
    """
    offsets = np.arange(-3, 4)
    exact = _fv_face_coeffs(offsets)                   # exact 7-cell FV coefficients
    A = np.zeros((STENCIL, NSUB))
    for k in range(NSUB):
        A[k:k + 4, k] = FACE_COEFFS[k]
    d, *_ = np.linalg.lstsq(A, exact, rcond=None)
    return d


LIN_WEIGHTS = _optimal_weights()


def optimal_face_coeffs() -> np.ndarray:
    """Linear combination of the sub-stencils with the optimal weights.

    For a smooth field this must equal the 7-point face interpolation
    coefficient vector (exact for polynomials of degree <= 6) -- the check that
    the linear weights above are the right ones.
    """
    combined = np.zeros(STENCIL)
    for k in range(NSUB):
        combined[k:k + 4] += LIN_WEIGHTS[k] * FACE_COEFFS[k]
    return combined


def weno7_left(v: np.ndarray) -> np.ndarray:
    """Left-biased WENO7 state at every interior face of ``v``.

    ``out[k]`` is the state on the left of the face between ``v[k+3]`` and
    ``v[k+4]``, built from the seven cells ``v[k:k+7]``.
    """
    m = v.size - (STENCIL - 1)
    if m <= 0:
        raise ValueError("field too short for WENO7")
    vals = np.stack([v[j:j + m] for j in range(STENCIL)])      # (7, m)
    p = np.empty((NSUB, m))
    beta = np.empty((NSUB, m))
    for k in range(NSUB):
        sub = vals[k:k + 4]
        p[k] = FACE_COEFFS[k] @ sub
        beta[k] = np.einsum("ij,im,jm->m", BETA_MATS[k], sub, sub)
    alpha = LIN_WEIGHTS[:, None] / (EPS_JS + beta) ** 2
    w = alpha / alpha.sum(axis=0)
    return np.einsum("km,km->m", w, p)


def reconstruction_order(sizes: Sequence[int] = (40, 80, 160, 320, 640),
                         slope_frac: float = 0.5) -> Dict:
    """Observed order of the WENO7 (and WENO5) reconstruction alone.

    The field is ``f(x) = sin(2 pi x)`` on a periodic unit interval.  WENO-JS
    loses accuracy at first-order critical points, so the error is measured
    only on faces where ``|f'| >= slope_frac * max|f'|`` (away from extrema).
    No time integration is involved: this isolates the reconstruction.
    """
    fx = lambda x: np.sin(2.0 * np.pi * x)
    dfx = lambda x: 2.0 * np.pi * np.cos(2.0 * np.pi * x)
    rows = []
    for nx in sizes:
        x = (np.arange(nx) + 0.5) / nx
        f = fx(x)
        # padded so that face values are available for the whole periodic patch
        g = np.concatenate([f[-6:], f, f[:6]])
        left7 = weno7_left(g)
        left5 = xc._weno_left(g)  # noqa: SLF001 - reuse the audited WENO5 routine
        xf = (np.arange(nx) + 1.0) / nx                      # face positions
        exact = fx(xf)
        # face k of the padded array corresponds to face k-3 of the patch
        it7 = left7[3:nx + 3]
        it5 = left5[4:nx + 4]
        mask = np.abs(dfx(xf)) >= slope_frac * np.abs(dfx(xf)).max()
        e7 = float(np.mean(np.abs(it7[mask] - exact[mask])))
        e5 = float(np.mean(np.abs(it5[mask] - exact[mask])))
        rows.append({"nx": nx, "h": 1.0 / nx, "err_weno7": e7, "err_weno5": e5,
                     "faces_used": int(mask.sum())})

    def _order(key: str) -> float:
        e = [r[key] for r in rows]
        n = [r["nx"] for r in rows]
        return float(-(math.log(e[-1]) - math.log(e[0])) / (math.log(n[-1]) - math.log(n[0])))
    return {"rows": rows,
            "observed_order_weno7": _order("err_weno7"),
            "observed_order_weno5": _order("err_weno5"),
            "sizes": list(sizes), "slope_frac": slope_frac}


# ----------------------------------------------------------------------
# 1-D Euler with WENO7 + SSP-RK3 + Rusanov (same conventions as WENO5 file)
# ----------------------------------------------------------------------
def _pad7(U: np.ndarray) -> np.ndarray:
    """Four reflective ghost cells on each side."""
    g = np.concatenate([U[:, :4][:, ::-1], U, U[:, -4:][:, ::-1]], axis=1)
    g[1, :4] = -U[1, :4][::-1]
    g[1, -4:] = -U[1, -4:][::-1]
    return g


def _face_states7(g: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    n = g.shape[1]
    nx = n - 8
    left = np.empty((3, nx + 1))
    right = np.empty((3, nx + 1))
    for k in range(3):
        ll = weno7_left(g[k])
        rr = weno7_left(g[k][::-1])[::-1]
        left[k] = ll[:nx + 1]
        right[k] = rr[1:nx + 2]
    return left, right


def _rhs7(U: np.ndarray, dx: float, counters: Dict[str, int]) -> np.ndarray:
    nx = U.shape[1]
    UL, UR = _face_states7(_pad7(U))
    rl, ul, pl = xc._prim(UL)  # noqa: SLF001
    rr, ur, pr = xc._prim(UR)  # noqa: SLF001
    bad = (rl <= 0) | (pl <= 0) | (rr <= 0) | (pr <= 0)
    if bad.any():  # same positivity guard as the WENO5 reference
        counters["positivity_fallbacks"] += int(bad.sum())
        g = _pad7(U)
        for k in range(3):
            UL[k, bad] = g[k, 3:-4][bad]
            UR[k, bad] = g[k, 4:-3][bad]
        rl, ul, pl = xc._prim(UL)  # noqa: SLF001
        rr, ur, pr = xc._prim(UR)  # noqa: SLF001
    cl = np.sqrt(GAMMA * pl / rl)
    cr = np.sqrt(GAMMA * pr / rr)
    a = np.maximum(np.abs(ul) + cl, np.abs(ur) + cr)
    Fhat = 0.5 * (xc._flux(UL) + xc._flux(UR)) - 0.5 * a * (UR - UL)  # noqa: SLF001
    return -(Fhat[:, 1:] - Fhat[:, :-1]) / dx


def weno7_solve(nx: int, t_end: float = WC_T_END, cfl: float = 0.4,
                datum: str = "blast") -> Dict:
    """Run the WENO7 reference; ``datum='sod'`` for Sod's shock tube."""
    x = (np.arange(nx, dtype=np.float64) + 0.5) / nx
    if datum == "blast":
        rho = np.full(nx, xc.WC_RHO)
        p = np.where(x < xc.WC_X1, xc.WC_P_LEFT,
                     np.where(x < xc.WC_X2, xc.WC_P_MIDDLE, xc.WC_P_RIGHT))
    elif datum == "sod":
        left = x < 0.5
        rho = np.where(left, 1.0, 0.125)
        p = np.where(left, 1.0, 0.1)
    else:
        raise ValueError(datum)
    U = xc._cons(rho, np.zeros(nx), p)  # noqa: SLF001
    dx = 1.0 / nx
    counters = {"positivity_fallbacks": 0}
    mass0 = float(U[0].sum() * dx)
    t = 0.0
    steps = 0
    t0 = time.time()
    while t < t_end:
        _, uu, pp = xc._prim(U)  # noqa: SLF001
        c = np.sqrt(GAMMA * pp / U[0])
        dt = cfl * dx / float(np.max(np.abs(uu) + c))
        if t + dt > t_end:
            dt = t_end - t
        k1 = _rhs7(U, dx, counters)
        U1 = U + dt * k1
        k2 = _rhs7(U1, dx, counters)
        U2 = 0.75 * U + 0.25 * (U1 + dt * k2)
        k3 = _rhs7(U2, dx, counters)
        U = (1.0 / 3.0) * U + (2.0 / 3.0) * (U2 + dt * k3)
        t += dt
        steps += 1
    rho, uu, pp = xc._prim(U)  # noqa: SLF001
    return {
        "method": "fd-weno7-ssprk3-rusanov",
        "datum": datum, "nx": nx, "t_end": t_end, "cfl": cfl,
        "x": x.tolist(), "rho": rho.tolist(), "u": uu.tolist(), "p": pp.tolist(),
        "steps": steps, "seconds": round(time.time() - t0, 2),
        "positivity_fallbacks": counters["positivity_fallbacks"],
        "mass": float(U[0].sum() * dx),
        "mass_drift": abs(float(U[0].sum() * dx) - mass0),
        "rho_max": float(rho.max()), "x_at_rho_max": float(x[int(np.argmax(rho))]),
        "p_max": float(pp.max()),
    }


# ----------------------------------------------------------------------
# convergence-order probe (smooth entropy wave, periodic) for both schemes
# ----------------------------------------------------------------------
def _periodic_rhs(U: np.ndarray, dx: float, recon, ghosts: int) -> np.ndarray:
    """Semi-discrete Euler operator with periodic boundaries.

    ``recon`` is the left-state reconstructor (WENO5 or WENO7); ``ghosts`` its
    ghost-cell count.  Used only by the verification probe: the blast case is a
    closed tube, so it must use the reflective padding of the solver itself.
    """
    nx = U.shape[1]
    g = np.concatenate([U[:, -ghosts:], U, U[:, :ghosts]], axis=1)
    UL = np.empty((3, nx + 1))
    UR = np.empty((3, nx + 1))
    for k in range(3):
        ll = recon(g[k])
        rr = recon(g[k][::-1])[::-1]
        UL[k] = ll[:nx + 1]
        UR[k] = rr[1:nx + 2]
    rl, ul, pl = xc._prim(UL)   # noqa: SLF001
    rr_, ur, pr = xc._prim(UR)  # noqa: SLF001
    a = np.maximum(np.abs(ul) + np.sqrt(GAMMA * pl / rl),
                   np.abs(ur) + np.sqrt(GAMMA * pr / rr_))
    Fhat = 0.5 * (xc._flux(UL) + xc._flux(UR)) - 0.5 * a * (UR - UL)  # noqa: SLF001
    return -(Fhat[:, 1:] - Fhat[:, :-1]) / dx


def _entropy_wave(rhs, nx: int, amp: float, t_end: float, cfl_base: float,
                  cfl_exp: float, nx_ref: int = 80) -> float:
    """L1 error of the density after advecting ``rho = 1 + amp sin(2 pi x)``.

    ``u = 1``, ``p = 1`` is an exact entropy wave of the Euler equations, so the
    exact solution at ``t`` is ``rho = 1 + amp sin(2 pi (x - t))``.  The time
    step is shrunk as ``dx**cfl_exp`` so that the 3rd-order SSP-RK3 error falls
    below the spatial error and the *spatial* order is what is measured.
    """
    x = (np.arange(nx) + 0.5) / nx
    U = xc._cons(1.0 + amp * np.sin(2.0 * np.pi * x), np.ones(nx), np.ones(nx))  # noqa: SLF001
    dx = 1.0 / nx
    cfl = cfl_base * (dx * nx_ref) ** cfl_exp
    t = 0.0
    while t < t_end:
        _, uu, pp = xc._prim(U)  # noqa: SLF001
        dt = cfl * dx / float(np.max(np.abs(uu) + np.sqrt(GAMMA * pp / U[0])))
        if t + dt > t_end:
            dt = t_end - t
        k1 = rhs(U, dx)
        U1 = U + dt * k1
        k2 = rhs(U1, dx)
        U2 = 0.75 * U + 0.25 * (U1 + dt * k2)
        k3 = rhs(U2, dx)
        U = (1.0 / 3.0) * U + (2.0 / 3.0) * (U2 + dt * k3)
        t += dt
    rho, _, _ = xc._prim(U)  # noqa: SLF001
    exact = 1.0 + amp * np.sin(2.0 * np.pi * (x - t_end))
    return float(np.mean(np.abs(rho - exact)))


def convergence_order(sizes: Sequence[int] = (80, 160, 320, 640, 1280),
                      amp: float = 0.2, t_end: float = 0.1,
                      cfl_base: float = 0.4, cfl_exp: float = 1.2) -> Dict:
    """Observed order of the audited WENO5 scheme and of this WENO7 scheme.

    Both are finite-volume updates (cell averages in, ``-dF/dx`` out), so the
    probe feeds cell averages and compares cell averages.  A probe that fed
    *point* values instead would show only 2nd order for any FV scheme -- this
    is the trap documented in ``REF-ATTRIBUTION.md``.
    """
    rhs5 = lambda U, dx: _periodic_rhs(U, dx, xc._weno_left, 3)   # noqa: SLF001
    rhs7 = lambda U, dx: _periodic_rhs(U, dx, weno7_left, 4)
    out = {"sizes": list(sizes), "cfl_exp": cfl_exp, "amp": amp, "t_end": t_end}
    for name, rhs in (("weno5_audited", rhs5), ("weno7", rhs7)):
        errs = [_entropy_wave(rhs, n, amp, t_end, cfl_base, cfl_exp) for n in sizes]
        orders = [float(-math.log(errs[i] / errs[i - 1]) /
                        math.log(sizes[i] / sizes[i - 1])) for i in range(1, len(sizes))]
        out[name] = {"errors": errs, "orders": orders,
                     "order_finest": orders[-1], "order_overall": orders[-2]}
    return out


def store(run: Dict, outdir: Path) -> Dict:
    """Persist a rung in the ladder-data convention (profile npz + metrics json)."""
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    tag = f"nx{run['nx']}_{'weno7' if run['method'].startswith('fd-weno7') else 'other'}"
    np.savez(outdir / f"ladder_profile_{tag}.npz",
             x=np.asarray(run["x"]), rho=np.asarray(run["rho"]),
             p=np.asarray(run["p"]), u=np.asarray(run["u"]))
    metrics = {k: v for k, v in run.items() if k not in ("x", "rho", "u", "p")}
    (outdir / f"ladder_metrics_{tag}.json").write_text(json.dumps(metrics, indent=2) + "\n")
    return {"tag": tag, "profile": str(outdir / f"ladder_profile_{tag}.npz"),
            "metrics": str(outdir / f"ladder_metrics_{tag}.json")}


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--nx", type=int, default=None)
    ap.add_argument("--cfl", type=float, default=0.4)
    ap.add_argument("--datum", default="blast")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--order-study", action="store_true")
    ap.add_argument("--json", type=Path, default=None)
    args = ap.parse_args()

    report: Dict = {}
    if args.order_study:
        report["reconstruction_order"] = reconstruction_order()
    if args.nx:
        run = weno7_solve(args.nx, cfl=args.cfl, datum=args.datum)
        report["run"] = {k: v for k, v in run.items() if k not in ("x", "rho", "u", "p")}
        if args.out:
            report["stored"] = store(run, args.out)
    print(json.dumps(report, indent=2))
    if args.json:
        args.json.write_text(json.dumps(report, indent=2) + "\n")