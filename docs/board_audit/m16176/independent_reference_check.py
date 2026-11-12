"""Board-audit independent probe of the Ethier--Steinman closed form (m16077).

This script deliberately does NOT import tensorfvm: the audit question is whether
the *reference* used to judge the solver is itself an exact unforced
Navier--Stokes solution.  The formula is re-implemented here straight from the
text printed in the mission report / RESULTS.md, with plain central finite
differences and its own sampling.

Checks per parameter set:
  1. max |div u|                    -> exact solution must be divergence-free
  2. max |lap u + a^2 u|            -> exact Laplacian eigenfunction
  3. max |curl R| at nu = d^2/a^2   -> unforced momentum balance (pressure = gradient,
                                       so the residual must be curl-free)
  4. max |curl R| at a wrong nu     -> negative control; must be O(1), which is what
                                       makes check 3 sharp rather than trivially small

Step sizes: gradient h=1e-5 (truncation ~1e-10), curl of R h=1e-3 (truncation
~1e-5, i.e. the reported floor below; both far under the O(1) negative control).

Usage: python3 independent_reference_check.py [--json out.json]
"""

from __future__ import annotations

import argparse
import json

import numpy as np

CYCLIC = ((0, 1, 2), (1, 2, 0), (2, 0, 1))
H_GRAD = 1e-5
H_CURL = 1e-3


def velocity(points, a, d, t):
    """u_i = -a [ e^{a x_i} sin(a(x_j+x_k)) + e^{a x_k} cos(a(x_i+x_j)) ] e^{-d^2 t}."""
    pts = np.atleast_2d(np.asarray(points, dtype=float))
    decay = np.exp(-d * d * t)
    u = np.zeros((pts.shape[0], 3))
    for i, j, k in CYCLIC:
        xi, xj, xk = pts[:, i], pts[:, j], pts[:, k]
        u[:, i] = -a * (
            np.exp(a * xi) * np.sin(a * (xj + xk))
            + np.exp(a * xk) * np.cos(a * (xi + xj))
        ) * decay
    return u


def _d1(f, points, axis, h):
    p = points.copy()
    p[:, axis] += h
    m = points.copy()
    m[:, axis] -= h
    return (f(p) - f(m)) / (2.0 * h)


def _d2(f, points, axis, h=2e-4):
    p = points.copy()
    p[:, axis] += h
    m = points.copy()
    m[:, axis] -= h
    return (f(p) - 2.0 * f(points) + f(m)) / (h * h)


def gradient(points, a, d, t):
    g = np.zeros((points.shape[0], 3, 3))  # g[n, comp, axis]
    for comp in range(3):
        f = lambda q, c=comp: velocity(q, a, d, t)[:, c]
        for axis in range(3):
            g[:, comp, axis] = _d1(f, points, axis, H_GRAD)
    return g


def residual(points, a, d, t, nu_used):
    """R = du/dt + (u.grad)u - nu lap u, using the eigenfunction identity lap u = -a^2 u
    (check 2) so that R is smooth enough for a tight curl stencil."""
    u = velocity(points, a, d, t)
    g = gradient(points, a, d, t)
    conv = np.einsum("nc,nca->na", u, g)
    lap = -a * a * u
    return -d * d * u + conv - nu_used * lap


def curl_max(points, a, d, t, nu_used):
    R = lambda p: residual(p, a, d, t, nu_used)
    c = np.zeros((points.shape[0], 3))
    c[:, 0] = _d1(lambda p: R(p)[:, 2], points, 1, H_CURL) - _d1(
        lambda p: R(p)[:, 1], points, 2, H_CURL
    )
    c[:, 1] = _d1(lambda p: R(p)[:, 0], points, 2, H_CURL) - _d1(
        lambda p: R(p)[:, 2], points, 0, H_CURL
    )
    c[:, 2] = _d1(lambda p: R(p)[:, 1], points, 0, H_CURL) - _d1(
        lambda p: R(p)[:, 0], points, 1, H_CURL
    )
    return float(np.max(np.abs(c)))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default=None)
    ap.add_argument("--time", type=float, default=0.25)
    ap.add_argument("--samples", type=int, default=12)
    args = ap.parse_args(argv)

    rng = np.random.default_rng(20261112)
    pts = rng.uniform(-0.85, 0.85, size=(args.samples, 3))  # interior of [-1,1]^3

    cases = [
        ("re1", 1.0, 1.0, 1.0),
        ("re100", 1.0, 0.1, 0.01),
        ("re100_resolved", 0.2, 0.02, 0.01),
    ]
    out = {"time_s": args.time, "samples": args.samples, "h_grad": H_GRAD,
           "h_curl": H_CURL, "cases": {}}

    for tag, a, d, nu in cases:
        u = velocity(pts, a, d, args.time)
        g = gradient(pts, a, d, args.time)
        div = g[:, 0, 0] + g[:, 1, 1] + g[:, 2, 2]
        lap = np.zeros((pts.shape[0], 3))
        for comp in range(3):
            f = lambda q, c=comp: velocity(q, a, d, args.time)[:, c]
            for axis in range(3):
                lap[:, comp] += _d2(f, pts, axis)
        rec = {
            "a": a, "d": d, "nu": nu,
            "nu_relation_d2_over_a2": d * d / (a * a),
            "div_u_max": float(np.max(np.abs(div))),
            "laplacian_eigen_max": float(np.max(np.abs(lap + a * a * u))),
            "curl_residual_at_case_nu": curl_max(pts, a, d, args.time, nu),
            "curl_residual_at_wrong_nu": curl_max(pts, a, d, args.time, 1.0),
            "u_linf": float(np.max(np.abs(u))),
        }
        out["cases"][tag] = rec
        print(
            f"{tag:16s} a={a:<5g} d={d:<5g} nu={nu:<6g} "
            f"div={rec['div_u_max']:.2e} lap+a2u={rec['laplacian_eigen_max']:.2e} "
            f"curlR(nu={nu:g})={rec['curl_residual_at_case_nu']:.2e} "
            f"curlR(nu=1)={rec['curl_residual_at_wrong_nu']:.2e}"
        )

    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(out, fh, indent=2)
        print(f"wrote {args.json}")
    return out


if __name__ == "__main__":
    main()