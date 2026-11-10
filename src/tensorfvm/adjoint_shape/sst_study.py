"""k-omega SST RANS adjoint study: gradient verification, grid convergence, frozen bias.

Runs (all with the exact coloured discrete adjoint of
:mod:`tensorfvm.adjoint_shape.sst_rans`):

1. ``laminar``   -- ``nu_t = 0``: adjoint ``dC_D/dtheta`` vs central differences
   of the re-converged solve (two step sizes + Richardson), 8 Fourier parameters.
2. ``sst_nut``   -- the k-omega SST closure evaluated on a *prescribed* (k, omega)
   profile anchored to the body surface, so ``nu_t = nu_t(theta)``; the **full
   differential** adjoint (which carries ``d nu_t / d theta``) is compared with
   the same central differences, and with the **frozen-turbulence** adjoint.
3. ``grid``      -- ``C_D`` and ``dC_D/dtheta`` on three grids, observed order.
4. ``coupled``   -- honest record of the fully coupled SST transport solve
   (Newton and segregated), which does *not* reach a converged steady state.

Writes ``docs/sst-rans-adjoint/result.json`` and the report figures.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch

from .sst_rans import A1, BETA_STAR, SSTChannel, _smax

OUT = Path("docs/sst-rans-adjoint")


def make_theta(seed: int = 7, scale: float = 0.05):
    rng = np.random.default_rng(seed)
    return scale * rng.standard_normal(8)


def sst_profile_nut(k_ref=2.0e-2, omega_ref=40.0, strain_ref=2.0, length=0.25):
    """``nu_t(theta)`` from the SST formula on a prescribed (k, omega) profile.

    ``k`` peaks one mixing length off the body and decays away from it, ``omega``
    decays like ``1/d``; the shape enters through the distance field
    ``d(theta) = min(y_wall, |g(theta)|)`` of the Fourier body, so
    ``d nu_t / d theta != 0`` -- exactly the term the frozen-turbulence
    approximation drops.
    """
    def nut_fn(theta, case: SSTChannel):
        g = case.grid
        _, _, _, gc = case.geometry(theta)
        y = case._coords["c"][1]
        y_wall = torch.minimum(y, g.height - y)
        d = torch.clamp(torch.minimum(y_wall, torch.abs(gc)), min=1e-6)
        k = k_ref * torch.exp(-d / (0.5 * length)) * (1.0 - torch.exp(-d / (0.05 * length)))
        w = omega_ref / (1.0 + d / (0.5 * length))
        S = torch.full_like(d, strain_ref)
        arg2 = _smax(
            2.0 * torch.sqrt(k) / (BETA_STAR * w * d),
            500.0 * case.nu / (d * d * w),
        )
        F2 = torch.tanh(arg2**2)
        return A1 * k / _smax(A1 * w, F2 * S)

    return nut_fn


def _rel_err(a, b):
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    return np.abs(a - b) / np.maximum(np.abs(b), 1e-12)


def case_for(nx, ny, re, kind, radius0=0.25, center=(2.0, 0.9)):
    grid_kw = dict(nx=nx, ny=ny, re=re, u_mean=1.0, radius0=radius0, center=center)
    if kind == "laminar":
        return SSTChannel(**{"grid": None}, **{}) if False else _mk(grid_kw, turbulent=False)
    if kind == "sst_nut":
        return _mk(grid_kw, turbulent=False, nut_fn=sst_profile_nut())
    raise ValueError(kind)


def _mk(grid_kw, **kw):
    from .sst_rans import make_case

    return make_case(**grid_kw, **kw)


def fd_gradient(case, theta, eps_list, U0=None, mode="full"):
    """Central differences per parameter at several steps; returns ``{eps: g}``."""
    out = {}
    for eps in eps_list:
        g = np.zeros(8)
        for q in range(8):
            e = np.zeros(8)
            e[q] = eps
            Up, _ = case.solve(theta + e, U0=U0, mode=mode)
            Um, _ = case.solve(theta - e, U0=U0, mode=mode)
            Jp = float(case.objective(Up, case._theta_t(theta + e)))
            Jm = float(case.objective(Um, case._theta_t(theta - e)))
            g[q] = (Jp - Jm) / (2.0 * eps)
        out[float(eps)] = g
    return out


def run_study(nx=40, ny=20, re=1.0e3, eps_list=(1e-4, 5e-5), grid_list=None, seed=7):
    t00 = time.time()
    theta = make_theta(seed)
    res = {
        "config": {
            "nx": nx,
            "ny": ny,
            "re": re,
            "radius0": 0.25,
            "center": [2.0, 0.9],
            "eta": 1e-3,
            "eps_fd": list(eps_list),
            "theta_base": theta.tolist(),
        },
        "blocks": {},
    }

    # ---------------- 1. laminar reference -------------------------------
    case = case_for(nx, ny, re, "laminar")
    U, info = case.solve(theta, tol=1e-12, maxit=40)
    J0 = float(case.objective(U, case._theta_t(theta)))
    g_adj, ainfo = case.adjoint_gradient(theta, U=U, info=info)
    lam_res = ainfo["adjoint_residual"]
    fds = fd_gradient(case, theta, eps_list, U0=U)
    e1, e2 = float(eps_list[0]), float(eps_list[1])
    g_rich = (4.0 * fds[e2] - fds[e1]) / 3.0
    err_lad = _rel_err(g_adj, fds[e1])
    err_rich = _rel_err(g_adj, g_rich)
    res["blocks"]["laminar"] = {
        "J": J0,
        "residual": info["residual"],
        "relative_residual": info["relative_residual"],
        "newton_iterations": info["iterations"],
        "adjoint_residual": lam_res,
        "gradient_adjoint": g_adj.tolist(),
        "gradient_fd_1e-4": fds[e1].tolist(),
        "gradient_fd_5e-5": fds[e2].tolist(),
        "gradient_richardson": g_rich.tolist(),
        "rel_err_vs_fd_1e-4": err_lad.tolist(),
        "rel_err_vs_richardson": err_rich.tolist(),
        "max_rel_err_vs_richardson": float(err_rich.max()),
        "mean_rel_err_vs_richardson": float(err_rich.mean()),
        "converged": bool(info["converged"] or info["relative_residual"] < 1e-10),
    }
    print(
        f"[laminar] J={J0:.6f} |R|rel={info['relative_residual']:.2e} "
        f"adj_res={lam_res:.2e} max rel err vs Richardson={err_rich.max():.2e}",
        flush=True,
    )

    # ---------------- 2. SST-derived nu_t(theta) -------------------------
    casev = case_for(nx, ny, re, "sst_nut")
    Uv, infov = casev.solve(theta, tol=1e-12, maxit=60, U0=None)
    Jv = float(casev.objective(Uv, casev._theta_t(theta)))
    g_full, ainfov = casev.adjoint_gradient(theta, U=Uv, info=infov, mode="full")
    g_froz, afroz = casev.adjoint_gradient(theta, U=Uv, info=infov, mode="frozen_nut")
    g_fd_frozen = fd_gradient(casev, theta, eps_list, U0=Uv, mode="frozen_nut")
    g_fd_full = fd_gradient(casev, theta, eps_list, U0=Uv, mode="full")
    gr_full = (4.0 * g_fd_full[e2] - g_fd_full[e1]) / 3.0
    gr_froz = (4.0 * g_fd_frozen[e2] - g_fd_frozen[e1]) / 3.0
    err_full = _rel_err(g_full, gr_full)
    err_froz = _rel_err(g_froz, gr_froz)
    bias = _rel_err(g_froz, g_full)
    res["blocks"]["sst_nut"] = {
        "J": Jv,
        "residual": infov["residual"],
        "relative_residual": infov["relative_residual"],
        "newton_iterations": infov["iterations"],
        "adjoint_residual_full": ainfov["adjoint_residual"],
        "adjoint_residual_frozen": afroz["adjoint_residual"],
        "gradient_full_differential": g_full.tolist(),
        "gradient_frozen": g_froz.tolist(),
        "gradient_fd_full_richardson": gr_full.tolist(),
        "gradient_fd_frozen_richardson": gr_froz.tolist(),
        "rel_err_full_vs_fd": err_full.tolist(),
        "rel_err_frozen_vs_fd_frozen": err_froz.tolist(),
        "max_rel_err_full_vs_fd": float(err_full.max()),
        "max_rel_err_frozen_vs_fd": float(err_froz.max()),
        "frozen_bias_rel": bias.tolist(),
        "frozen_bias_max_rel": float(bias.max()),
        "frozen_bias_mean_rel": float(bias.mean()),
        "nut_max_over_nu": float(
            (casev.nut_fn(theta, casev)).max().item() / casev.nu
        ),
        "grad_frozen_fd_two_step_change": float(
            np.abs(g_fd_frozen[e1] - g_fd_frozen[e2]).max()
        ),
    }
    print(
        f"[sst_nut] J={Jv:.6f} full max rel err={err_full.max():.2e} "
        f"frozen max rel err={err_froz.max():.2e} frozen bias max={bias.max():.2%}",
        flush=True,
    )

    # ---------------- 3. grid convergence --------------------------------
    grids = grid_list or [(20, 10), (40, 20), (80, 40)]
    rows = []
    for (gx, gy) in grids:
        c = case_for(gx, gy, re, "sst_nut")
        Ug, ig = c.solve(theta, tol=1e-12, maxit=60)
        Jg = float(c.objective(Ug, c._theta_t(theta)))
        gg, ag = c.adjoint_gradient(theta, U=Ug, info=ig, mode="full")
        rows.append(
            {
                "nx": gx,
                "ny": gy,
                "h": c.grid.dx,
                "J": Jg,
                "grad_norm": float(np.linalg.norm(gg)),
                "gradient": gg.tolist(),
                "adjoint_residual": ag["adjoint_residual"],
                "relative_residual": ig["relative_residual"],
            }
        )
        print(f"[grid {gx}x{gy}] J={Jg:.8f} |g|={np.linalg.norm(gg):.6e}", flush=True)

    def _order(vals, hs):
        vals = np.abs(np.asarray(vals, float))
        return float(np.log(vals[0] / vals[-1]) / np.log(hs[0] / hs[-1]))

    hs = [r["h"] for r in rows]
    Js = [r["J"] for r in rows]
    gs = [r["grad_norm"] for r in rows]
    # Richardson estimate of the exact value from the two finest grids
    pJ = _order(Js[-3:], hs[-3:])
    pG = _order(gs[-3:], hs[-3:])
    res["blocks"]["grid"] = {
        "rows": rows,
        "observed_order_CD": pJ,
        "observed_order_grad_norm": pG,
        "CD_richardson_fine": float(
            Js[-2] + (Js[-2] - Js[-1]) / (2.0**pJ - 1.0)
        ),
        "grad_richardson_fine": float(
            gs[-2] + (gs[-2] - gs[-1]) / (2.0**pG - 1.0)
        ),
        "rel_diff_CD_fine_vs_mid": float(abs(Js[-1] - Js[-2]) / abs(Js[-2])),
        "rel_diff_grad_fine_vs_mid": float(abs(gs[-1] - gs[-2]) / gs[-2]),
    }
    print(f"[grid] order CD={pJ:.2f} grad={pG:.2f}", flush=True)

    # ---------------- 4. coupled SST transport solve (honest record) ------
    coupled = []
    for re_try, nturb in ((1.0e3, True), (1.0e4, True)):
        for tag, solver in (("newton", "solve"), ("segregated", "solve_segregated")):
            c = _mk(
                dict(nx=nx, ny=ny, re=re_try, u_mean=1.0, radius0=0.25, center=(2.0, 0.9)),
                turbulent=True,
            )
            U0 = c.initial_guess(k0=4e-3, w0=4.0)
            try:
                if solver == "solve":
                    Ut, it = c.solve(theta, U0=U0, tol=1e-11, maxit=20)
                else:
                    Ut, it = c.solve_segregated(theta, U0=U0, tol=1e-10, maxit=15)
                F = c.unpack(Ut)
                coupled.append(
                    {
                        "re": re_try,
                        "solver": tag,
                        "converged": bool(it["converged"]),
                        "relative_residual": float(it["relative_residual"]),
                        "residual": float(it["residual"]),
                        "history": [float(h) for h in it["history"]],
                        "k_max": float(F["k"].max()),
                        "omega_min": float(F["w"].min()),
                        "omega_max": float(F["w"].max()),
                        "CD": float(c.objective(Ut, c._theta_t(theta))),
                    }
                )
                print(
                    f"[coupled Re={re_try:g} {tag}] conv={it['converged']} "
                    f"rel={it['relative_residual']:.2e}",
                    flush=True,
                )
            except Exception as exc:  # pragma: no cover - recorded, not raised
                coupled.append({"re": re_try, "solver": tag, "error": str(exc)})
    res["blocks"]["coupled_sst"] = coupled

    # ---------------- criteria -------------------------------------------
    lam = res["blocks"]["laminar"]
    sst = res["blocks"]["sst_nut"]
    crit = {
        "C1_adjoint_vs_FD_rel_lt_1e-3_sst_nut": bool(sst["max_rel_err_full_vs_fd"] < 1e-3),
        "C1_max_rel_err_sst_nut": sst["max_rel_err_full_vs_fd"],
        "C1_adjoint_vs_FD_rel_lt_1e-3_laminar": bool(lam["max_rel_err_vs_richardson"] < 1e-3),
        "C1_max_rel_err_laminar": lam["max_rel_err_vs_richardson"],
        "C2_adjoint_residual_lt_1e-10_sst_nut": bool(sst["adjoint_residual_full"] < 1e-10),
        "C2_adjoint_residual_sst_nut": sst["adjoint_residual_full"],
        "C2_adjoint_residual_laminar": lam["adjoint_residual"],
        "C3_frozen_bias_lt_5pct": bool(sst["frozen_bias_max_rel"] < 0.05),
        "C3_frozen_bias_max_rel": sst["frozen_bias_max_rel"],
        "C3_scope": "prescribed SST-profile nu_t(theta); coupled SST transport not converged",
        "C4_order_ge_1.5_CD": bool(pJ >= 1.5),
        "C4_order_CD": pJ,
        "C4_order_ge_1.5_grad": bool(pG >= 1.5),
        "C4_order_grad": pG,
        "coupled_sst_converged": bool(
            any(r.get("converged") for r in coupled if "converged" in r)
        ),
    }
    res["criteria"] = crit
    res["wall_seconds"] = time.time() - t00
    return res


def main(out_dir: Path = OUT):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    res = run_study()
    (out_dir / "result.json").write_text(json.dumps(res, indent=1))
    print("wrote", out_dir / "result.json", "%.0fs" % res["wall_seconds"])
    return res


if __name__ == "__main__":  # pragma: no cover
    main()