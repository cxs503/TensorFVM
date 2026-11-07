#!/usr/bin/env python3
"""Evidence probe for the incompressible FVM kernel and the three baselines.

Writes one JSON per probe into ``runs/incompressible-baselines/``.  The probe is
deliberately cheap: it documents what the kernel reproduces (analytic channel,
multigrid vs direct solve, conduction-limit cavity) and what the three
benchmark cases actually produce at a modest grid/budget, so the verification
status can be audited without re-running the long jobs.
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np



def _load_kernel():
    """Import the kernel without executing ``tensorfvm/__init__.py``.

    The package initialiser imports torch (an otherwise unrelated legacy
    solver).  On the loong64 compute nodes the installed torch wheel cannot load
    its BLAS runtime, and importing the package would then hard-fail even though
    this kernel is pure NumPy.  Loading the two modules under a synthetic package
    keeps the relative imports working and never touches torch.
    """
    import importlib.util
    import types

    src = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
    pkg_name = "_tfvm_incompressible"
    if pkg_name in sys.modules:
        return sys.modules[pkg_name + ".incompressible_baselines"]
    pkg = types.ModuleType(pkg_name)
    pkg.__path__ = [os.path.join(src, "tensorfvm")]
    sys.modules[pkg_name] = pkg
    for mod in ("staggered_simple", "incompressible_baselines"):
        name = f"{pkg_name}.{mod}"
        spec = importlib.util.spec_from_file_location(
            name, os.path.join(src, "tensorfvm", mod + ".py"))
        m = importlib.util.module_from_spec(spec)
        sys.modules[name] = m
        spec.loader.exec_module(m)
    return sys.modules[pkg_name + ".incompressible_baselines"]

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))


OUT = "runs/incompressible-baselines"

_k = _load_kernel()
_simple = sys.modules["_tfvm_incompressible.staggered_simple"]
MaskedPoisson = _simple.MaskedPoisson
StaggeredSimple = _simple.StaggeredSimple
backward_facing_step_metrics = _k.backward_facing_step_metrics
build_backward_facing_step = _k.build_backward_facing_step
build_natural_convection_cavity = _k.build_natural_convection_cavity
build_sudden_expansion = _k.build_sudden_expansion
cavity_nusselt = _k.cavity_nusselt
cavity_streamfunction = _k.cavity_streamfunction
sudden_expansion_metrics = _k.sudden_expansion_metrics



def dump(name, payload):
    os.makedirs(OUT, exist_ok=True)
    payload = dict(payload)
    payload["probe"] = name
    with open(os.path.join(OUT, name + ".json"), "w") as fh:
        json.dump(payload, fh, indent=2, default=float)
    print(name, json.dumps({k: v for k, v in payload.items()
                            if not isinstance(v, (list, dict))}, default=float)[:400])


def probe_channel(levels=(16, 32, 64)):
    """Plane Poiseuille: exact solution u = 4 Umax y(H-y)/H^2 on a full domain."""
    rows = []
    for ny in levels:
        nx = 2 * ny
        H, L, nu, umax = 1.0, 4.0, 0.05, 1.0
        hy, hx = H / ny, L / nx
        y = (np.arange(ny) + 0.5) * hy
        prof = 4.0 * umax * y * (H - y) / H ** 2
        s = StaggeredSimple(nx, ny, hx, hy, nu, inlet_profile=prof, limited=False)
        r = s.run(max_iter=1500, tol=1e-9, check=5, momentum_sweeps=6, p_cycles=4)
        uc = 0.5 * (s.u[:-1] + s.u[1:])
        mid = nx // 2
        g_exact = 8.0 * nu * umax / H ** 2
        dpdx = float(np.mean(s.p[mid + 3] - s.p[mid - 3]) / (6 * hx))
        rows.append(dict(ny=ny, nx=nx, iterations=r["iterations"],
                         continuity=r["continuity"],
                         profile_max_error=float(np.abs(uc[mid] - prof).max() / umax),
                         dpdx=dpdx, dpdx_exact=-g_exact,
                         dpdx_error_rel=float(abs(dpdx + g_exact) / g_exact),
                         flux_error=float(abs(np.sum(s.u[0]) - np.sum(s.u[-1])) / np.sum(s.u[0]))))
    order = None
    if len(rows) >= 2:
        e1, e2 = rows[-2]["profile_max_error"], rows[-1]["profile_max_error"]
        if e2 > 0:
            order = float(np.log(e1 / e2) / np.log(2.0))
    dump("probe_channel_poiseuille", dict(levels=rows, observed_order=order,
                                          note="Re-independent analytic check of the momentum/pressure kernel"))


def probe_multigrid():
    """Masked multigrid vs a dense direct solve on a step-shaped domain."""
    nx, ny = 32, 16
    fluid = np.ones((nx, ny), dtype=bool)
    fluid[:nx // 3, :ny // 2] = False
    mg = MaskedPoisson(fluid)
    rng = np.random.default_rng(0)
    ku = np.ones((nx + 1, ny))
    kv = np.ones((nx, ny + 1))
    rhs = np.where(fluid, rng.normal(size=(nx, ny)), 0.0)
    phi = mg.solve(ku, kv, rhs, cycles=60, tol=1e-12)
    e, w, n, s, diag = mg._operator(fluid, ku, kv)
    res0 = float(np.abs(rhs).max())
    res = float(np.abs(np.where(fluid, rhs - mg._apply(e, w, n, s, diag, phi), 0.0)).max())
    # dense reference on the same operator
    cells = np.argwhere(fluid)
    m = len(cells)
    index = -np.ones((nx, ny), dtype=int)
    index[fluid] = np.arange(m)
    A = np.zeros((m, m))
    A[np.arange(m), np.arange(m)] = -diag[fluid]
    for arr, di, dj in ((e, 1, 0), (w, -1, 0), (n, 0, 1), (s, 0, -1)):
        ii = cells[:, 0] + di
        jj = cells[:, 1] + dj
        ok = (ii >= 0) & (ii < nx) & (jj >= 0) & (jj < ny)
        nb = np.full(m, -1, dtype=int)
        nb[ok] = index[ii[ok], jj[ok]]
        has = nb >= 0
        A[np.arange(m)[has], nb[has]] += arr[fluid][has]
    pin = index[nx - 1, ny - 1]
    A[pin, :] = 0.0
    A[:, pin] = 0.0
    A[pin, pin] = 1.0
    bm = rhs[fluid].copy()
    bm[pin] = 0.0
    ref = np.zeros((nx, ny))
    ref[fluid] = np.linalg.solve(A, bm)
    dump("probe_multigrid", dict(
        levels=[list(map(int, l.shape)) for l in mg.levels],
        residual_reduction=float(res / res0),
        error_vs_direct=float(np.abs(phi - ref)[fluid].max() / max(np.abs(ref[fluid]).max(), 1e-30)),
        note="masked V-cycle is a correction-level solver; see VERIFICATION.md limitations"))


def probe_conduction_cavity():
    s = build_natural_convection_cavity(16, rayleigh=1e4)
    h = s.hx
    xs = (np.arange(16) + 0.5) * h
    s.T[:] = (1.0 - xs)[:, None]
    for _ in range(3):
        s._solve_energy()
    nu_hot, nu_cold = cavity_nusselt(s)
    dump("probe_conduction_limit", dict(
        nusselt_hot=nu_hot, nusselt_cold=nu_cold,
        exact=1.0, error_rel=float(abs(nu_hot - 1.0)),
        t_profile_max_dev=float(np.abs(s.T - (1.0 - xs)[:, None]).max())))


def probe_bfs(n=20, iters=600):
    s = build_backward_facing_step(n)
    s.limited = False
    s.u[:] = s.inlet_values[None, :]
    s.apply_bc()
    t0 = time.time()
    hist = []
    for k in range(iters // 100):
        st = s.iterate(100, momentum_sweeps=4, p_cycles=2)
        m = backward_facing_step_metrics(s)
        hist.append(dict(iteration=s.iterations, continuity=st["continuity"],
                         reattachment=m["reattachment_over_H"],
                         min_u=m["min_u_in_recirculation"]))
    dump("probe_backward_facing_step", dict(
        n_over_H=n, re_H=s.meta["re_H"], nu=s.nu, wall_seconds=time.time() - t0,
        history=hist, reference=6.10, contract_target=6.0,
        status="FAIL: reattachment length collapses towards zero (see history)"))


def probe_expansion(n=10, iters=600):
    s = build_sudden_expansion(n)
    s.limited = False
    hist = []
    for k in range(iters // 100):
        st = s.iterate(100, momentum_sweeps=4, p_cycles=2)
        m = sudden_expansion_metrics(s)
        hist.append(dict(iteration=s.iterations, continuity=st["continuity"],
                         k_total=m["k_total_head"], k_static=m["k_static_recovery"],
                         reattachment=m["reattachment_over_D"]))
    dump("probe_sudden_expansion", dict(
        n_over_D=n, re_D=s.meta["re_D"], history=hist,
        k_theory=0.5625, status="see history: K not converged at this budget"))


def probe_cavity(n=24, rayleigh=1e4, iters=800):
    s = build_natural_convection_cavity(n, rayleigh=rayleigh)
    s.limited = False
    hist = []
    for k in range(iters // 100):
        st = s.iterate(100, momentum_sweeps=5, p_cycles=3)
        nu_hot, _ = cavity_nusselt(s)
        psi = cavity_streamfunction(s)
        hist.append(dict(iteration=s.iterations, continuity=st["continuity"],
                         nusselt=nu_hot, psi_abs_max=float(np.abs(psi).max()),
                         psi_over_alpha=float(np.abs(psi).max() / (s.nu / s.prandtl))))
    dump("probe_natural_convection", dict(
        n=n, rayleigh=rayleigh, history=hist, nusselt_reference=2.243,
        status="see history: Nu decays towards the conduction limit"))


if __name__ == "__main__":
    np.seterr(all="ignore")
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    if which in ("all", "channel"):
        probe_channel()
    if which in ("all", "mg"):
        probe_multigrid()
    if which in ("all", "conduction"):
        probe_conduction_cavity()
    if which in ("all", "bfs"):
        probe_bfs()
    if which in ("all", "expansion"):
        probe_expansion()
    if which in ("all", "cavity"):
        probe_cavity()
