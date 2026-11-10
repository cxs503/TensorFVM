"""Chunked driver for the k-omega SST RANS adjoint verification.

The verification is split into blocks (``laminar``, ``sst_nut``, ``grid``,
``coupled``) so that each one can be run and cached separately -- the full
sequence is a few minutes of Newton solves and the blocks are independent.

    PYTHONPATH=src python scripts/sst_adjoint_study.py --block laminar
    PYTHONPATH=src python scripts/sst_adjoint_study.py --merge

Writes ``docs/sst-rans-adjoint/blocks/<block>.json`` and, after ``--merge``,
``docs/sst-rans-adjoint/result.json``.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from tensorfvm.adjoint_shape.sst_study import (
    _mk,
    _rel_err,
    case_for,
    fd_gradient,
    make_theta,
    sst_profile_nut,
)

OUT = Path("docs/sst-rans-adjoint")


def block_laminar(nx, ny, re, eps_list, theta):
    case = case_for(nx, ny, re, "laminar")
    U, info = case.solve(theta, tol=1e-12, maxit=40)
    J0 = float(case.objective(U, case._theta_t(theta)))
    g_adj, ainfo = case.adjoint_gradient(theta, U=U, info=info)
    fds = fd_gradient(case, theta, eps_list, U0=U)
    e1, e2 = float(eps_list[0]), float(eps_list[1])
    g_rich = (4.0 * fds[e2] - fds[e1]) / 3.0
    return {
        "J": J0,
        "residual": info["residual"],
        "relative_residual": info["relative_residual"],
        "newton_iterations": info["iterations"],
        "adjoint_residual": ainfo["adjoint_residual"],
        "gradient_adjoint": g_adj.tolist(),
        "gradient_fd_1e-4": fds[e1].tolist(),
        "gradient_fd_5e-5": fds[e2].tolist(),
        "gradient_richardson": g_rich.tolist(),
        "rel_err_vs_fd_1e-4": _rel_err(g_adj, fds[e1]).tolist(),
        "rel_err_vs_richardson": _rel_err(g_adj, g_rich).tolist(),
        "max_rel_err_vs_richardson": float(_rel_err(g_adj, g_rich).max()),
        "mean_rel_err_vs_richardson": float(_rel_err(g_adj, g_rich).mean()),
        "fd_two_step_change": float(np.max(np.abs(fds[e1] - fds[e2]))),
        "converged": bool(info["relative_residual"] < 1e-10),
    }


def block_sst_nut(nx, ny, re, eps_list, theta):
    e1, e2 = float(eps_list[0]), float(eps_list[1])
    case = case_for(nx, ny, re, "sst_nut")
    U, info = case.solve(theta, tol=1e-12, maxit=80)
    J = float(case.objective(U, case._theta_t(theta)))
    g_full, ai_full = case.adjoint_gradient(theta, U=U, info=info, mode="full")
    g_froz, ai_froz = case.adjoint_gradient(theta, U=U, info=info, mode="frozen_nut")
    g_fd_full = fd_gradient(case, theta, eps_list, U0=U, mode="full")
    gr_full = (4.0 * g_fd_full[e2] - g_fd_full[e1]) / 3.0
    err_full = _rel_err(g_full, gr_full)
    bias = _rel_err(g_froz, g_full)
    bias_fd = _rel_err(g_froz, gr_full)

    # frozen *operator*: nu_t held as a constant field -> adjoint must be exact
    nut_fixed = case.nut_fn(case._theta_t(theta), case).detach()
    casef = _mk(dict(nx=nx, ny=ny, re=re, u_mean=1.0, radius0=0.25, center=(2.0, 0.9)),
                turbulent=False, nut_fixed=nut_fixed)
    Uf, infof = casef.solve(theta, tol=1e-12, maxit=60)
    g_fix, ai_fix = casef.adjoint_gradient(theta, U=Uf, info=infof)
    fd_fix = fd_gradient(casef, theta, eps_list, U0=Uf)
    gr_fix = (4.0 * fd_fix[e2] - fd_fix[e1]) / 3.0
    err_fix = _rel_err(g_fix, gr_fix)
    return {
        "J": J,
        "J_frozen_operator": float(casef.objective(Uf, casef._theta_t(theta))),
        "residual": info["residual"],
        "relative_residual": info["relative_residual"],
        "newton_iterations": info["iterations"],
        "adjoint_residual_full": ai_full["adjoint_residual"],
        "adjoint_residual_frozen": ai_froz["adjoint_residual"],
        "gradient_full_differential": g_full.tolist(),
        "gradient_frozen": g_froz.tolist(),
        "gradient_fd_full_richardson": gr_full.tolist(),
        "rel_err_full_vs_fd": err_full.tolist(),
        "max_rel_err_full_vs_fd": float(err_full.max()),
        "mean_rel_err_full_vs_fd": float(err_full.mean()),
        "frozen_bias_rel": bias.tolist(),
        "frozen_bias_max_rel": float(bias.max()),
        "frozen_bias_mean_rel": float(bias.mean()),
        "frozen_bias_from_fd_rel": bias_fd.tolist(),
        "frozen_bias_from_fd_max_rel": float(bias_fd.max()),
        "frozen_operator": {
            "adjoint_residual": float(ai_fix["adjoint_residual"]),
            "rel_err_vs_fd": err_fix.tolist(),
            "max_rel_err_vs_fd": float(err_fix.max()),
        },
        "nut_max_over_nu": float(case.nut_fn(case._theta_t(theta), case).max().item() / case.nu),
        "rel_err_frozen_vs_fd_frozen": bias_fd.tolist(),
        "max_rel_err_frozen_vs_fd": float(bias_fd.max()),
        "grad_frozen_fd_two_step_change": float(np.max(np.abs(g_fd_full[e1] - g_fd_full[e2]))),
    }


def block_grid(re, theta, grids, kind="sst_nut"):
    rows = []
    for (gx, gy) in grids:
        c = case_for(gx, gy, re, kind)
        U, info = c.solve(theta, tol=1e-12, maxit=80)
        J = float(c.objective(U, c._theta_t(theta)))
        g, ag = c.adjoint_gradient(theta, U=U, info=info, mode="full")
        rows.append({
            "nx": gx, "ny": gy, "h": c.grid.dx, "J": J,
            "grad_norm": float(np.linalg.norm(g)), "gradient": g.tolist(),
            "adjoint_residual": ag["adjoint_residual"],
            "relative_residual": info["relative_residual"],
        })
        print(f"  grid {gx}x{gy}: J={J:.8f} |g|={np.linalg.norm(g):.6e}", flush=True)

    def order(vals, hs):
        vals = np.abs(np.asarray(vals, float))
        return float(np.log(vals[0] / vals[-1]) / np.log(hs[0] / hs[-1]))

    hs = [r["h"] for r in rows]
    Js = [r["J"] for r in rows]
    gs = [r["grad_norm"] for r in rows]
    pJ = order(Js, hs)
    pG = order(gs, hs)
    return {
        "rows": rows,
        "observed_order_CD": pJ,
        "observed_order_grad_norm": pG,
        "CD_richardson_fine": float(Js[-2] + (Js[-2] - Js[-1]) / (2.0**pJ - 1.0)),
        "grad_richardson_fine": float(gs[-2] + (gs[-2] - gs[-1]) / (2.0**pG - 1.0)),
        "rel_diff_CD_fine_vs_mid": float(abs(Js[-1] - Js[-2]) / abs(Js[-2])),
        "rel_diff_grad_fine_vs_mid": float(abs(gs[-1] - gs[-2]) / gs[-2]),
    }


def block_coupled(nx, ny, theta, tries=((1.0e3, "newton", 10), (1.0e4, "segregated", 8))):
    out = []
    for re_try, tag, maxit in tries:
        c = _mk(dict(nx=nx, ny=ny, re=re_try, u_mean=1.0, radius0=0.25, center=(2.0, 0.9)),
                turbulent=True)
        U0 = c.initial_guess(k0=4e-3, w0=4.0)
        if tag == "newton":
            U, it = c.solve(theta, U0=U0, tol=1e-11, maxit=maxit)
        else:
            U, it = c.solve_segregated(theta, U0=U0, tol=1e-10, maxit=maxit)
        F = c.unpack(U)
        out.append({
            "re": re_try, "solver": tag,
            "converged": bool(it["converged"]),
            "relative_residual": float(it["relative_residual"]),
            "residual": float(it["residual"]),
            "history": [float(h) for h in it["history"]],
            "k_max": float(F["k"].max()),
            "omega_min": float(F["w"].min()),
            "omega_max": float(F["w"].max()),
            "CD": float(c.objective(U, c._theta_t(theta))),
        })
        print(f"  coupled Re={re_try:g} {tag}: rel={it['relative_residual']:.2e}", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--block", default="")
    ap.add_argument("--merge", action="store_true")
    ap.add_argument("--nx", type=int, default=20)
    ap.add_argument("--ny", type=int, default=10)
    ap.add_argument("--re", type=float, default=1.0e3)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--grids", default="10x6,20x10,40x20")
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "blocks").mkdir(exist_ok=True)
    theta = make_theta(args.seed)
    eps_list = (1e-4, 5e-5)
    t0 = time.time()

    if args.block == "laminar":
        res = block_laminar(args.nx, args.ny, args.re, eps_list, theta)
        print(f"laminar max rel err {res['max_rel_err_vs_richardson']:.2e}", flush=True)
    elif args.block == "sst_nut":
        res = block_sst_nut(args.nx, args.ny, args.re, eps_list, theta)
        print(f"sst_nut full {res['max_rel_err_full_vs_fd']:.2e} "
              f"bias {res['frozen_bias_max_rel']:.2%}", flush=True)
    elif args.block == "grid":
        grids = [tuple(int(v) for v in g.split("x")) for g in args.grids.split(",")]
        res = block_grid(args.re, theta, grids)
    elif args.block == "coupled":
        res = block_coupled(args.nx, args.ny, theta)
    elif args.merge:
        res = None
    else:
        raise SystemExit("pick --block {laminar,sst_nut,grid,coupled} or --merge")

    if res is not None:
        (OUT / "blocks" / f"{args.block}.json").write_text(json.dumps(res, indent=1))
        print(f"wrote blocks/{args.block}.json in {time.time() - t0:.0f}s")
        return

    blocks = {}
    for name in ("laminar", "sst_nut", "grid", "coupled"):
        p = OUT / "blocks" / f"{name}.json"
        if p.exists():
            blocks[name] = json.loads(p.read_text())
    lam, sst, grid = blocks["laminar"], blocks["sst_nut"], blocks["grid"]
    coupled = blocks.get("coupled", [])
    crit = {
        "C1_adjoint_vs_FD_rel_lt_1e-3_sst_nut": bool(sst["max_rel_err_full_vs_fd"] < 1e-3),
        "C1_max_rel_err_sst_nut": sst["max_rel_err_full_vs_fd"],
        "C1_adjoint_vs_FD_rel_lt_1e-3_laminar": bool(lam["max_rel_err_vs_richardson"] < 1e-3),
        "C1_max_rel_err_laminar": lam["max_rel_err_vs_richardson"],
        "C1_frozen_operator_rel_lt_1e-3": bool(sst["frozen_operator"]["max_rel_err_vs_fd"] < 1e-3),
        "C1_frozen_operator_max_rel_err": sst["frozen_operator"]["max_rel_err_vs_fd"],
        "C2_adjoint_residual_lt_1e-10_sst_nut": bool(sst["adjoint_residual_full"] < 1e-10),
        "C2_adjoint_residual_sst_nut": sst["adjoint_residual_full"],
        "C2_adjoint_residual_laminar": lam["adjoint_residual"],
        "C3_frozen_bias_lt_5pct": bool(sst["frozen_bias_max_rel"] < 0.05),
        "C3_frozen_bias_max_rel": sst["frozen_bias_max_rel"],
        "C3_scope": "SST-closure nu_t(theta) on a prescribed profile; transported SST not converged",
        "C4_order_ge_1.5_CD": bool(grid["observed_order_CD"] >= 1.5),
        "C4_order_CD": grid["observed_order_CD"],
        "C4_order_ge_1.5_grad": bool(grid["observed_order_grad_norm"] >= 1.5),
        "C4_order_grad": grid["observed_order_grad_norm"],
        "coupled_sst_converged": bool(any(r.get("converged") for r in coupled)),
    }
    out = {
        "config": {
            "nx": args.nx, "ny": args.ny, "re": args.re, "radius0": 0.25,
            "center": [2.0, 0.9], "eta": 1e-3, "eps_fd": list(eps_list),
            "theta_base": theta.tolist(),
        },
        "blocks": {"laminar": lam, "sst_nut": sst, "grid": grid, "coupled_sst": coupled},
        "criteria": crit,
    }
    (OUT / "result.json").write_text(json.dumps(out, indent=1))
    print("merged ->", OUT / "result.json")
    print(json.dumps(crit, indent=1))


if __name__ == "__main__":
    main()