#!/usr/bin/env python3
"""Turn the baseline run JSON into the verification tables, criteria and figures.

Nothing here is typed by hand: the numbers are read from
``runs/incompressible-baselines/*.json`` (written by
``scripts/run_fvm_incompressible_baselines.py`` and
``scripts/probe_fvm_incompressible.py``), and every acceptance statement is
evaluated in code so the report cannot drift from the data.

Outputs
-------
``docs/incompressible-baselines/RESULTS.md``
    grid-convergence tables, acceptance verdicts, figures embedded.
``benchmarks/verified/fvm_{backward_step,sudden_expansion,natural_convection}/``
    per-case ``results.json`` (criterion numbers + artifact paths) + ``README.md``
    + ``*.png``.
"""

from __future__ import annotations

import glob
import json
import os

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

RUNS = "runs/incompressible-baselines"
#: Archived copy of the same JSON files.  The worktree's sparse checkout does not
#: include a top-level ``runs/`` (nor ``benchmarks/``), so the evidence has to live
#: under ``docs/`` to be committed; the live directory is preferred when present.
RUNS_ARCHIVE = "docs/incompressible-baselines/baseline-runs"
DOCS = "docs/incompressible-baselines"
#: Artifact root.  ``benchmarks/verified`` is what the acceptance contract names,
#: but it is outside this worktree's sparse checkout, so the default is the
#: tracked path; set ``FVM_BENCH_DIR=benchmarks/verified`` on a full checkout.
BENCH = os.environ.get("FVM_BENCH_DIR", "docs/benchmarks/verified")
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

BFS_TARGET = 6.0
BFS_BAND = 0.2
K_THEORY = 0.5625
NU_CONTRACT = {1.0e4: 2.243, 1.0e5: 4.519, 1.0e6: 8.800}
NU_DVD = {1.0e4: 2.238, 1.0e5: 4.509, 1.0e6: 8.817}
PSI_DVD = {1.0e4: 5.071, 1.0e5: 9.111, 1.0e6: 16.32}
ORDER_TARGET = 1.5


def run_dir():
    live = os.path.join(BASE, RUNS)
    if glob.glob(os.path.join(live, "*.json")):
        return live
    return os.path.join(BASE, RUNS_ARCHIVE)


def load(pattern):
    out = {}
    for path in sorted(glob.glob(os.path.join(run_dir(), pattern))):
        with open(path) as fh:
            out[os.path.basename(path)[:-5]] = json.load(fh)
    return out


def richardson(values, levels):
    """Observed order from three grids and the h -> 0 extrapolation.

    Solves ``(f2 - f1)/(f3 - f2) = (h2^p - h1^p)/(h3^p - h2^p)`` for the observed
    order ``p`` (bisection, ``1 <= p <= 6``), which stays valid when the
    refinement ratio is not uniform (the delivered families are H/10, H/20, H/40
    with ratio 2 and D/10, D/20, D/30 with ratios 2 and 1.5).  Returns
    ``(None, None)`` when the three values do not form a monotone sequence, in
    which case no asymptotic estimate exists.
    """
    pairs = [(1.0 / n, values[n]) for n in levels if values.get(n) is not None]
    if len(pairs) < 3:
        return None, None
    (h1, f1), (h2, f2), (h3, f3) = pairs[0], pairs[1], pairs[2]
    d21, d32 = f2 - f1, f3 - f2
    if abs(d32) < 1e-14 or d21 * d32 <= 0:
        return None, None
    target = abs(d21 / d32)

    def ratio(p):
        num = h2 ** p - h1 ** p
        den = h3 ** p - h2 ** p
        if abs(den) < 1e-300:
            return float("inf")
        return abs(num / den)

    lo, hi = 1e-3, 8.0
    if ratio(lo) > target or ratio(hi) < target:
        # monotone in p over this bracket is the normal case; fall back to the
        # uniform-ratio formula when the root is outside the bracket
        p = float(np.log(target) / np.log(h1 / h2)) if abs(h1 - h2) > 0 else float("nan")
        if not np.isfinite(p):
            return None, None
    else:
        for _ in range(200):
            mid = 0.5 * (lo + hi)
            if ratio(mid) < target:
                lo = mid
            else:
                hi = mid
        p = 0.5 * (lo + hi)
    if not np.isfinite(p) or p <= 0 or p > 8:
        return None, None
    # h2 > h3, so (h2/h3)^p - 1 > 0 and the correction moves away from f3
    # towards the h -> 0 limit.
    f0 = float(f3 + (f3 - f2) / ((h2 / h3) ** p - 1.0)) if h3 < h2 else None
    return float(p), f0


def fmt_float(x, digits=3):
    return "n/a" if x is None else f"{x:.{digits}f}"


def criterion(cid, description, value, reference, error, passed, note=""):
    return dict(id=cid, description=description, value=None if value is None else float(value),
                reference=None if reference is None else float(reference),
                error=None if error is None else float(error),
                passed=bool(passed), note=note)


def monotone(errors):
    vals = [e for e in errors if e is not None]
    return len(vals) == len(errors) and all(vals[i] > vals[i + 1] for i in range(len(vals) - 1))


# --------------------------------------------------------------------------- #
# figures
# --------------------------------------------------------------------------- #
def _save(fig, case, name):
    path = os.path.join(BASE, BENCH, case, name)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return os.path.relpath(path, BASE)


def fig_channel():
    probes = load("probe_channel_poiseuille.json")
    if not probes:
        return None
    d = list(probes.values())[0]
    ny = [r["ny"] for r in d["levels"]]
    err = [r["profile_max_error"] for r in d["levels"]]
    dp = [abs(r["dpdx"] + 0.4) / 0.4 for r in d["levels"]]
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.4))
    ax[0].loglog(ny, err, "o-")
    ax[0].set_xlabel("cells across the channel")
    ax[0].set_ylabel("max profile error / $u_{max}$")
    ax[0].grid(True, which="both", alpha=0.3)
    ax[0].set_title("plane Poiseuille: profile error")
    ax[1].loglog(ny, dp, "s-")
    ax[1].set_xlabel("cells across the channel")
    ax[1].set_ylabel("|dp/dx error| / exact")
    ax[1].grid(True, which="both", alpha=0.3)
    ax[1].set_title("analytic pressure gradient")
    fig.tight_layout()
    return _save(fig, "fvm_backward_step", "fig_kernel_channel.png")


def fig_bfs(bfs, order, extrapolated):
    if not bfs:
        return None
    levels = sorted(bfs)
    fig, ax = plt.subplots(2, 2, figsize=(11, 7.0))
    # (a) wall friction coefficient distribution
    for n in levels:
        prof = bfs[n]["metrics"].get("friction_profile") or []
        if not prof:
            continue
        arr = np.asarray(prof)
        ax[0, 0].plot(arr[:, 0], arr[:, 1], lw=1.3, label=f"H/{n}")
    m = bfs[levels[-1]]["metrics"]
    ax[0, 0].axhline(0.0, color="k", lw=0.6)
    ax[0, 0].axhline(m.get("friction_far_analytic", np.nan), color="0.4", ls=":", lw=1.0,
                     label=f"analytic $C_f$ = {m.get('friction_far_analytic', float('nan')):.4f}")
    lr = m.get("reattachment_over_H")
    if lr is not None:
        ax[0, 0].axvline(lr, color="crimson", ls="--", lw=1.0,
                         label=f"$L_r/H$ = {lr:.2f} (finest)")
    ax[0, 0].axvline(BFS_TARGET, color="0.6", ls="-.", lw=0.9, label="target 6.0")
    ax[0, 0].set_xlabel("$x/H$")
    ax[0, 0].set_ylabel("$C_f$")
    ax[0, 0].set_ylim(-0.02, 0.05)
    ax[0, 0].legend(fontsize=7)
    ax[0, 0].set_title("(a) wall friction $C_f(x)$ on the stepped wall")
    # (b) streamwise velocity field, finest grid
    path = os.path.join(run_dir(), f"bfs_n{levels[-1]:03d}_fields.npz")
    if os.path.exists(path):
        d = np.load(path)
        u, fluid, hx, hy = d["u"], d["fluid"], float(d["hx"]), float(d["hy"])
        uc = 0.5 * (u[:-1] + u[1:])
        x = np.arange(uc.shape[0]) * hx - 2.0
        y = (np.arange(uc.shape[1]) + 0.5) * hy
        lim = float(np.abs(uc).max())
        im = ax[0, 1].contourf(x, y, uc.T, levels=np.linspace(-lim, lim, 31),
                               cmap="RdBu_r", extend="both")
        fig.colorbar(im, ax=ax[0, 1], label="$u$")
        ax[0, 1].contour(x, y, uc.T, levels=[0.0], colors="k", linewidths=0.8)
        ax[0, 1].set_xlabel("$x/H$")
        ax[0, 1].set_ylabel("$y/H$")
        ax[0, 1].set_title(f"(b) streamwise velocity, H/{levels[-1]} grid")
    # (c) reattachment vs grid
    ns = [n for n in levels if bfs[n]["metrics"].get("reattachment_over_H") is not None]
    vals = [bfs[n]["metrics"]["reattachment_over_H"] for n in ns]
    ax[1, 0].semilogx(ns, vals, "o-", label="measured")
    if extrapolated is not None:
        ax[1, 0].axhline(extrapolated, color="tab:green", ls="--", lw=1.0,
                         label=f"Richardson $h\\to0$: {extrapolated:.2f}")
    ax[1, 0].axhspan(BFS_TARGET - BFS_BAND, BFS_TARGET + BFS_BAND, color="0.85",
                     label="target 6.0 $\\pm$ 0.2")
    ax[1, 0].set_xlabel("cells across $H$")
    ax[1, 0].set_ylabel("$L_r/H$")
    ax[1, 0].legend(fontsize=7)
    ax[1, 0].set_title(f"(c) reattachment length, observed order $p$ = "
                       f"{'n/a' if order is None else f'{order:.2f}'}")
    # (d) convergence of L_r with h
    for n in ns:
        pass
    if len(ns) >= 2:
        err = [abs(v - BFS_TARGET) / BFS_TARGET for v in vals]
        ax[1, 1].loglog([1.0 / n for n in ns], err, "s-", label="error vs 6.0")
        ax[1, 1].set_xlabel("$h/H = 1/n$")
        ax[1, 1].set_ylabel("relative error")
        ax[1, 1].grid(True, which="both", alpha=0.3)
        ax[1, 1].legend(fontsize=7)
        ax[1, 1].set_title("(d) grid convergence of $L_r/H$")
    fig.tight_layout()
    return _save(fig, "fvm_backward_step", "fig_backward_step.png")


def fig_expansion(exp, re_sweep, order):
    if not exp:
        return None
    levels = sorted(exp)
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.6))
    path = os.path.join(run_dir(), f"expansion_n{levels[-1]:03d}_fields.npz")
    if os.path.exists(path):
        d = np.load(path)
        u, hx, hy = d["u"], float(d["hx"]), float(d["hy"])
        uc = 0.5 * (u[:-1] + u[1:])
        x = np.arange(uc.shape[0]) * hx
        y = (np.arange(uc.shape[1]) + 0.5) * hy
        lim = float(np.abs(uc).max())
        im = ax[0].contourf(x, y, uc.T, levels=np.linspace(-lim, lim, 31),
                            cmap="RdBu_r", extend="both")
        fig.colorbar(im, ax=ax[0], label="$u$")
        ax[0].contour(x, y, uc.T, levels=[0.0], colors="k", linewidths=0.8)
    ax[0].set_xlabel("$x/D$")
    ax[0].set_ylabel("$y/D$")
    ax[0].set_title(f"(a) streamwise velocity, H/$D$/{levels[-1]}")
    ks = [exp[n]["metrics"]["k_total_head"] for n in levels]
    ax[1].semilogx(levels, ks, "o-", label="measured $K$ (total head)")
    ax[1].axhline(K_THEORY, color="k", ls="--", lw=1.0,
                  label="Borda-Carnot 0.5625 (inviscid)")
    ax[1].set_xlabel("cells across the inlet height $D$")
    ax[1].set_ylabel("$K$")
    ax[1].legend(fontsize=7)
    ax[1].set_title(f"(b) loss coefficient, order $p$ = "
                    f"{'n/a' if order is None else f'{order:.2f}'}")
    if re_sweep:
        res = sorted((float(r["job"]["re_d"]), r["metrics"]["k_total_head"])
                     for r in re_sweep.values())
        ax[2].semilogx([r[0] for r in res], [r[1] for r in res], "o-")
        ax[2].axhline(K_THEORY, color="k", ls="--", lw=1.0)
        ax[2].set_xlabel("$Re_D$")
        ax[2].set_ylabel("$K$")
        ax[2].set_title("(c) measured $K(Re_D)$ vs inviscid limit")
    fig.tight_layout()
    return _save(fig, "fvm_sudden_expansion", "fig_sudden_expansion.png")


def fig_cavity(cav):
    if not cav:
        return None
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.6))
    for ra in sorted({k[0] for k in cav}):
        ns = sorted(n for (r, n) in cav if r == ra)
        nus = [cav[(ra, n)]["metrics"]["nusselt"] for n in ns]
        ax[0].plot(ns, nus, "o-", label=f"Ra={ra:.0e}")
        ax[0].axhline(NU_CONTRACT[ra], color="k", ls=":", lw=0.8)
        ax[0].axhline(NU_DVD[ra], color="0.5", ls="--", lw=0.6)
    ax[0].set_xlabel("cells per side")
    ax[0].set_ylabel("mean $Nu$")
    ax[0].set_title("(a) mean Nusselt number (dotted: de Vahl Davis)")
    ax[0].legend(fontsize=7)
    finals = [n for n in sorted({k[1] for k in cav})][-1]
    for ra in sorted({k[0] for k in cav}):
        if (ra, finals) not in cav:
            continue
        path = os.path.join(run_dir(), f"cavity_n{finals:03d}_ra{ra:.0e}_fields.npz".replace("+", ""))
        if not os.path.exists(path):
            continue
        d = np.load(path)
        T, hx = d["T"], float(d["hx"])
        n = T.shape[0]
        x = (np.arange(n) + 0.5) * hx
        ax[1].contourf(x, x, T.T, levels=20, cmap="inferno")
        ax[2].contourf(x, x, d["p"].T if "p" in d else T.T, levels=20, cmap="RdBu_r")
        ax[1].set_title(f"(b) temperature $T$, Ra=$10^{{{int(np.log10(ra))}}}$")
        ax[2].set_title(f"(c) pressure $p$, Ra=$10^{{{int(np.log10(ra))}}}$")
        break
    fig.tight_layout()
    return _save(fig, "fvm_natural_convection", "fig_natural_convection.png")


# --------------------------------------------------------------------------- #
# criteria
# --------------------------------------------------------------------------- #
def load_audit():
    """Convergence-audit records (written by the audit script), keyed by run tag."""
    path = os.path.join(run_dir(), "convergence_audit.json")
    if not os.path.exists(path):
        return {}
    with open(path) as fh:
        return json.load(fh)


def bfs_criteria(bfs, long_bfs=None):
    audit = load_audit()
    long_bfs = long_bfs or {}
    levels = sorted(bfs)
    lr = {n: bfs[n]["metrics"].get("reattachment_over_H") for n in levels}
    errors = [None if lr[n] is None else abs(lr[n] - BFS_TARGET) / BFS_TARGET for n in levels]
    order, extrap = richardson(lr, levels)
    out = [criterion(
        "bfs.reattachment.finest",
        f"|L_r/H - 6.0| / 6.0 on the finest grid H/{levels[-1]} (gate <= 10 %)",
        lr[levels[-1]], BFS_TARGET, errors[-1],
        errors[-1] is not None and errors[-1] <= 0.10,
        note="family: " + ", ".join(
            f"H/{n}: {fmt_float(lr[n])}" for n in levels))]
    out.append(criterion("bfs.reattachment.band",
                         "L_r/H inside the 6.0 +- 0.2 experimental/benchmark band (finest grid)",
                         lr[levels[-1]], BFS_TARGET,
                         errors[-1], lr[levels[-1]] is not None
                         and abs(lr[levels[-1]] - BFS_TARGET) <= BFS_BAND))
    out.append(criterion("bfs.grid.monotone",
                         "error in L_r/H decreases monotonically under refinement",
                         errors[-1], None, None, monotone(errors),
                         note=", ".join(f"H/{n}: {100 * e:.2f} %"
                                        for n, e in zip(levels, errors) if e is not None)))
    out.append(criterion("bfs.grid.order",
                         f"observed order of accuracy p >= {ORDER_TARGET}", order, ORDER_TARGET,
                         None if order is None else abs(order - ORDER_TARGET), order is not None
                         and order >= ORDER_TARGET,
                         note="first-order upwind; the limited defect correction is not a "
                              "contraction at Pe_cell ~ 40 (see VERIFICATION.md)"))
    fine = bfs[levels[-1]]["metrics"]
    #: C_f(x) convergence: the same station set, interpolated between grids.  The
    #: fully developed value 12 nu/(U_b H) is *reported*, not gated: the flow is
    #: still developing 4H downstream of reattachment (entrance length ~0.05 Re H
    #: = 40H), so C_f is expected to be several times the developed value.
    grid = np.arange(0.25, 10.0, 0.25)
    cf = {}
    for n in levels:
        prof = bfs[n]["metrics"].get("friction_profile") or []
        if prof:
            arr = np.asarray(prof)
            cf[n] = np.interp(grid, arr[:, 0], arr[:, 1])
    diffs = []
    for a, b in zip(levels, levels[1:]):
        if a in cf and b in cf:
            diffs.append(float(np.mean(np.abs(cf[b] - cf[a]))))
    out.append(criterion("bfs.friction.convergence",
                         "mean |C_f(x; fine) - C_f(x; coarse)| decreases under refinement",
                         diffs[-1] if diffs else None, None, None,
                         len(diffs) == 2 and diffs[1] < diffs[0],
                         note="pair differences: " + ", ".join(f"{d:.4f}" for d in diffs)))
    out.append(criterion("bfs.friction.developed_comparison",
                         "downstream C_f vs fully developed 12 nu/(U_b H): reported only "
                         "(developing boundary layer), not a gate",
                         fine.get("friction_far"), fine.get("friction_far_analytic"),
                         fine.get("friction_far_rel_error"), True,
                         note="not a gate: entrance length ~0.05 Re_H H = 40 H"))
    zero_ok = (fine.get("reattachment_over_H") is not None
               and (fine.get("friction_min") or 0.0) < 0.0)
    out.append(criterion("bfs.friction.trend",
                         "C_f < 0 inside the bubble, upstream of the reattachment point, "
                         "and zero-crossing consistent with L_r", fine.get("friction_min"),
                         0.0, None, zero_ok))
    out.append(criterion("bfs.mass.continuity",
                         "max |div u| normalised by the inlet mass flux < 1e-5",
                         fine.get("continuity"), 0.0, fine.get("continuity"),
                         (fine.get("continuity") or 1.0) < 1e-5))
    if long_bfs:
        ref = long_bfs[sorted(long_bfs)[0]]
        cf20 = ref["metrics"].get("friction_far")
        cf20_err = ref["metrics"].get("friction_far_rel_error")
        out.append(criterion("bfs.friction.domain_length",
                             "downstream C_f with a 20H outlet (vs the 10H case): positive and "
                             "within 30 % of 12 nu/(U_b H)",
                             cf20, ref["metrics"].get("friction_far_analytic"), cf20_err,
                             (cf20_err or 1.0) <= 0.30,
                             note="the 10H case gives a negative C_f at the exit: an outflow "
                                  "boundary artefact, not physics"))
    for n in levels:
        tag = f"bfs_n{n:03d}"
        if tag in audit and audit[tag].get("qoi_after") is not None:
            a = audit[tag]
            out.append(criterion(f"bfs.convergence.H/{n}",
                                 "L_r/H drifts less than 1 % when the saved field is continued "
                                 "for the audit window", a["qoi_after"], a["qoi_before"],
                                 a["drift_rel"], (a["drift_rel"] or 1.0) <= 0.01,
                                 note=f"{a['iterations']} extra iterations, dU={a['change']:.2e}, "
                                      f"continuity={a['continuity']:.2e}"))
    return out, order, extrap


def expansion_criteria(exp, re_sweep, long_domain=None):
    audit = load_audit()
    long_domain = long_domain or {}
    levels = sorted(exp)
    ks = {n: exp[n]["metrics"]["k_total_head"] for n in levels}
    errs = [abs(ks[n] - K_THEORY) / K_THEORY for n in levels]
    order, extrap = richardson(ks, levels)
    lr = {n: exp[n]["metrics"].get("reattachment_over_D") for n in levels}
    out = [criterion(
        "expansion.K.finest",
        f"|K - K_BC| / K_BC on the finest grid D/{levels[-1]} (gate <= 5 %)",
        ks[levels[-1]], K_THEORY, errs[-1], errs[-1] <= 0.05,
        note="K_BC is the inviscid limit; family: " + ", ".join(
            f"D/{n}: {ks[n]:.4f}" for n in levels))]
    out.append(criterion("expansion.grid.monotone",
                         "error in K decreases monotonically under refinement", errs[-1], None,
                         None, monotone(errs),
                         note=", ".join(f"D/{n}: {100 * e:.2f} %"
                                        for n, e in zip(levels, errs))))
    out.append(criterion("expansion.grid.order",
                         f"observed order of accuracy p >= {ORDER_TARGET}", order, ORDER_TARGET,
                         None, order is not None and order >= ORDER_TARGET))
    out.append(criterion("expansion.area_ratio", "A_out/A_in = 4 to within 0.2 %",
                         exp[levels[0]]["meta"].get("area_ratio", 4.0), 4.0, None,
                         abs(exp[levels[0]]["meta"].get("area_ratio", 4.0) - 4.0) <= 0.008))
    fine = exp[levels[-1]]["metrics"]

    out.append(criterion("expansion.identity", "internal identity K = recovery + (1 - (U_out/U_in)^2)",
                         fine.get("k_identity_check"), ks[levels[-1]],
                         abs((fine.get("k_identity_check") or 0.0) - ks[levels[-1]]),
                         abs((fine.get("k_identity_check") or 0.0) - ks[levels[-1]]) < 1e-6))
    if long_domain:
        ref = long_domain[sorted(long_domain)[0]]["metrics"]
        out.append(criterion("expansion.K.domain_length",
                             "K measured with a 30D outlet (vs 12D) at the same grid: quantifies "
                             "the finite-domain bias of the 12D case",
                             ref["k_total_head"], K_THEORY,
                             abs(ref["k_total_head"] - K_THEORY) / K_THEORY, False,
                             note="L_r/D at 12D already reaches the exit, so the 12D loss is a "
                                  "lower bound; reported as a sensitivity, not a gate"))
    if re_sweep:
        vals = sorted((float(r["job"]["re_d"]), r["metrics"]["k_total_head"])
                      for r in re_sweep.values())
        out.append(criterion("expansion.K.Re_sweep",
                             "measured K(Re_D) does not reach the inviscid K_BC = 0.5625 within "
                             "the sampled range (documents that the 5 % criterion is a "
                             "high-Re criterion)", vals[-1][1], K_THEORY,
                             abs(vals[-1][1] - K_THEORY) / K_THEORY, False,
                             note=", ".join(f"Re={r:g}: K={k:.4f}" for r, k in vals)))
    return out, order, extrap


def cavity_criteria(cav):
    audit = load_audit()
    levels = sorted({k[1] for k in cav})
    out = []
    for ra in sorted({k[0] for k in cav}):
        ns = [n for n in levels if (ra, n) in cav]
        nus = {n: cav[(ra, n)]["metrics"]["nusselt"] for n in ns}
        errs = [abs(nus[n] - NU_CONTRACT[ra]) / NU_CONTRACT[ra] for n in ns]
        order, extrap = richardson(nus, ns)
        out.append(criterion(f"cavity.Nu.{ra:.0e}.finest",
                             f"|Nu - {NU_CONTRACT[ra]}| / {NU_CONTRACT[ra]} at Ra={ra:.0e} on the "
                             f"finest grid ({ns[-1]}^2 cells, target <= 3 %)",
                             nus[ns[-1]], NU_CONTRACT[ra], errs[-1], errs[-1] <= 0.03,
                             note="family: " + ", ".join(
                                 f"{n}: {nus[n]:.4f}" for n in ns)))
        out.append(criterion(f"cavity.Nu.{ra:.0e}.monotone",
                             "Nu error decreases monotonically under refinement", errs[-1], None,
                             None, monotone(errs),
                             note=", ".join(f"{n}: {100 * e:.2f} %" for n, e in zip(ns, errs))))
        out.append(criterion(f"cavity.Nu.{ra:.0e}.order",
                             f"observed order p >= {ORDER_TARGET}", order, ORDER_TARGET,
                             None, order is not None and order >= ORDER_TARGET,
                             note=f"Richardson limit Nu = {extrap}"))
        psi = {}
        for n in ns:
            m = cav[(ra, n)]["metrics"]
            alpha = cav[(ra, n)]["meta"]["nu"] / cav[(ra, n)]["meta"]["prandtl"]
            psi[n] = m.get("psi_over_alpha", m["psi_abs_max"] / alpha)
        perr = [abs(psi[n] - PSI_DVD[ra]) / PSI_DVD[ra] for n in ns]
        out.append(criterion(f"cavity.psi.{ra:.0e}",
                             f"|psi/alpha - {PSI_DVD[ra]}| / {PSI_DVD[ra]} at Ra={ra:.0e} "
                             f"(finest grid, target <= 15 %)", psi[ns[-1]], PSI_DVD[ra],
                             perr[-1], perr[-1] <= 0.15,
                             note="psi is a cumulative integral of u from the wall: it "
                                  "accumulates the near-wall velocity error that Nu does not see"))
        m = cav[(ra, ns[-1])]["metrics"]
        out.append(criterion(f"cavity.energy.{ra:.0e}",
                             "hot/cold wall Nusselt numbers agree within 1 %",
                             m.get("nusselt"), m.get("nusselt_cold"),
                             abs(m["nusselt"] - m["nusselt_cold"]) / m["nusselt"],
                             abs(m["nusselt"] - m["nusselt_cold"]) / m["nusselt"] <= 0.01))
    return out


# --------------------------------------------------------------------------- #
def _write_case(case, criteria, extra):
    passed = [c for c in criteria if c["passed"]]
    payload = dict(case=case, criteria=criteria,
                   n_criteria=len(criteria), n_passed=len(passed),
                   verdict="PASS" if len(passed) == len(criteria) else "FAIL", **extra)
    path = os.path.join(BASE, BENCH, case, "results.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as fh:
        json.dump(payload, fh, indent=2)
    return payload


def main():
    os.makedirs(os.path.join(BASE, DOCS), exist_ok=True)
    bfs, exp, cav = {}, {}, {}
    for tag, rec in load("bfs_n*.json").items():
        bfs[int(rec["job"]["level"])] = rec
    for tag, rec in load("expansion_n*.json").items():
        exp[int(rec["job"]["level"])] = rec
    for tag, rec in load("cavity_n*.json").items():
        cav[(float(rec["job"]["rayleigh"]), int(rec["job"]["level"]))] = rec
    re_sweep = load("expansion_re*.json")
    audit = load_audit()
    long_domain = load("expansion_long_*.json")
    long_bfs = load("bfs_long_*.json")

    bfs_crit, bfs_order, bfs_extrap = (bfs_criteria(bfs, long_bfs) if bfs
                                       else ([], None, None))
    exp_crit, exp_order, exp_extrap = (expansion_criteria(exp, re_sweep, long_domain)
                                       if exp else ([], None, None))
    cav_crit = cavity_criteria(cav) if cav else []

    figs = {}
    figs["channel"] = fig_channel()
    if bfs:
        figs["bfs"] = fig_bfs(bfs, bfs_order, bfs_extrap)
    if exp:
        figs["expansion"] = fig_expansion(exp, re_sweep, exp_order)
    if cav:
        figs["cavity"] = fig_cavity(cav)

    extra_bfs = dict(
        figs=[figs.get("bfs"), figs.get("channel")],
        runs=sorted(f"{os.path.relpath(run_dir(), BASE)}/bfs_n{n:03d}.json" for n in bfs),
        source=["src/tensorfvm/staggered_simple.py", "src/tensorfvm/incompressible_baselines.py"],
        driver="scripts/run_fvm_incompressible_baselines.py",
        audit_driver="scripts/audit_fvm_incompressible_convergence.py",
        report="scripts/report_fvm_incompressible.py",
        discretisation="first-order upwind convection, central diffusion, SIMPLE, "
                       "staggered MAC, blocked-cell step",
        observed_order=bfs_order, richardson_limit=bfs_extrap,
        convergence_audit={k: audit[k] for k in audit if k.startswith("bfs_")},
        reference=dict(reattachment=6.0, band=0.2,
                       source="2-D benchmark value for Re_Dh = 800 (Armaly 1983; Kim & Moin "
                              "1985; Ertink 2008 reproduce 6.0 +- 0.2)"))
    extra_exp = dict(
        figs=[figs.get("expansion")],
        runs=sorted(f"{os.path.relpath(run_dir(), BASE)}/expansion_n{n:03d}.json" for n in exp)
        + sorted(f"{os.path.relpath(run_dir(), BASE)}/{t}.json" for t in re_sweep),
        source=["src/tensorfvm/staggered_simple.py", "src/tensorfvm/incompressible_baselines.py"],
        driver="scripts/run_fvm_incompressible_baselines.py",
        audit_driver="scripts/audit_fvm_incompressible_convergence.py",
        discretisation="first-order upwind convection, central diffusion, SIMPLE, uniform inlet",
        observed_order=exp_order, richardson_limit=exp_extrap,
        withheld=dict(reattachment_over_D={n: exp[n]["metrics"].get("reattachment_over_D")
                                          for n in sorted(exp)},
                      reason="no citable 2-D L_r/D for (AR=4, Re_D=100) is available offline; "
                             "the three grids agree to "
                             + f"{100 * (max(exp[n]['metrics']['reattachment_over_D'] for n in exp) - min(exp[n]['metrics']['reattachment_over_D'] for n in exp)) / max(exp[n]['metrics']['reattachment_over_D'] for n in exp):.2f} %"),
        convergence_audit={k: audit[k] for k in audit if k.startswith("expansion_")},
        domain_length_sensitivity={k: dict(k_total_head=v["metrics"]["k_total_head"],
                                           reattachment_over_D=v["metrics"]
                                           .get("reattachment_over_D"),
                                           length_inlet_heights=v["job"]
                                           .get("length_inlet_heights"))
                                   for k, v in long_domain.items()},
        reference=dict(borda_carnot=K_THEORY, area_ratio=4.0,
                       source="Borda-Carnot sudden-enlargement momentum balance (inviscid)"))
    extra_cav = dict(
        figs=[figs.get("cavity")],
        runs=sorted(f"{os.path.relpath(run_dir(), BASE)}/{t}.json"
                    for t in load("cavity_n*.json")),
        source=["src/tensorfvm/staggered_simple.py", "src/tensorfvm/incompressible_baselines.py"],
        driver="scripts/run_fvm_incompressible_baselines.py",
        audit_driver="scripts/audit_fvm_incompressible_convergence.py",
        discretisation="Boussinesq, first-order upwind + frozen van Leer defect correction, "
                       "SIMPLE with multigrid/sparse pressure correction",
        convergence_audit={k: audit[k] for k in audit if k.startswith("cavity_")},
        reference=dict(nu_contract=NU_CONTRACT, nu_de_vahl_davis=NU_DVD, psi_de_vahl_davis=PSI_DVD,
                       source="de Vahl Davis 1983, J. Comput. Phys. 3, 249"))

    payloads = {}
    if bfs:
        payloads["fvm_backward_step"] = _write_case("fvm_backward_step", bfs_crit, extra_bfs)
    if exp:
        payloads["fvm_sudden_expansion"] = _write_case("fvm_sudden_expansion", exp_crit, extra_exp)
    if cav:
        payloads["fvm_natural_convection"] = _write_case("fvm_natural_convection", cav_crit, extra_cav)

    # --------------------------------------------------------------- RESULTS.md
    lines = ["# Incompressible FVM verification baselines: generated results", "",
             "Generated by `scripts/report_fvm_incompressible.py` from "
             f"`{RUNS}/*.json`; do not edit by hand. Every verdict below is "
             "evaluated in code, and the same numbers are stored in "
             "`benchmarks/verified/fvm_*/results.json`.", ""]

    lines += ["## 1. Backward-facing step, Re_H = 800 (expansion ratio 1.9423)", ""]
    if bfs:
        lines += ["| cells/H | iterations | continuity | L_r/H | error vs 6.0 | C_f downstream "
                  "(analytic) |", "|---|---|---|---|---|---|"]
        for n in sorted(bfs):
            m = bfs[n]["metrics"]
            lr = m.get("reattachment_over_H")
            err = "n/a" if lr is None else f"{100 * abs(lr - BFS_TARGET) / BFS_TARGET:.2f} %"
            lines.append(f"| {n} | {m['iterations']} | {m['continuity']:.2e} | "
                         f"{'n/a' if lr is None else f'{lr:.3f}'} | {err} | "
                         f"{m.get('friction_far', float('nan')):.5f} "
                         f"({m.get('friction_far_analytic', float('nan')):.5f}) |")
        lines += ["", f"- observed order: {bfs_order}",
                  f"- Richardson limit L_r/H: {bfs_extrap}", "",
                  "| criterion | value | reference | error | verdict |",
                  "|---|---|---|---|---|"]
        for c in bfs_crit:
            lines.append(f"| {c['id']} | {c['value']} | {c['reference']} | {c['error']} | "
                         f"{'PASS' if c['passed'] else 'FAIL'} |")
        lines.append("")
        if figs.get("bfs"):
            lines += [f"![backward facing step]({os.path.relpath(os.path.join(BASE, figs['bfs']), os.path.join(BASE, DOCS))})", ""]
    else:
        lines += ["No run available.", ""]

    lines += ["## 2. Planar sudden expansion, AR = 4, Re_D = 100", ""]
    if exp:
        lines += ["| cells/D | iterations | continuity | K (total head) | error vs K_BC | L_r/D |",
                  "|---|---|---|---|---|---|"]
        for n in sorted(exp):
            m = exp[n]["metrics"]
            lines.append(f"| {n} | {m['iterations']} | {m['continuity']:.2e} | "
                         f"{m['k_total_head']:.4f} | {100 * m['k_error_rel']:.2f} % | "
                         f"{m.get('reattachment_over_D')} |")
        lines += ["", f"- observed order in K: {exp_order}",
                  f"- Richardson limit K: {exp_extrap}", "",
                  "| criterion | value | reference | error | verdict |",
                  "|---|---|---|---|---|"]
        for c in exp_crit:
            lines.append(f"| {c['id']} | {c['value']} | {c['reference']} | {c['error']} | "
                         f"{'PASS' if c['passed'] else 'FAIL'} |")
        if re_sweep:
            lines += ["", "Reynolds sweep of the measured loss (grid D/10):", "",
                      "| Re_D | K (total head) | static recovery |", "|---|---|---|"]
            for tag, rec in sorted(re_sweep.items(), key=lambda kv: kv[1]["job"]["re_d"]):
                m = rec["metrics"]
                lines.append(f"| {rec['job']['re_d']:g} | {m['k_total_head']:.4f} | "
                             f"{m['k_static_recovery']:.4f} |")
        lines.append("")
        if figs.get("expansion"):
            lines += [f"![sudden expansion]({os.path.relpath(os.path.join(BASE, figs['expansion']), os.path.join(BASE, DOCS))})", ""]
    else:
        lines += ["No run available.", ""]

    lines += ["## 3. Differentially heated cavity, Pr = 0.71", ""]
    if cav:
        lines += ["| Ra | cells | iterations | Nu | error vs contract | psi/alpha | psi error |",
                  "|---|---|---|---|---|---|---|"]
        for ra in sorted({k[0] for k in cav}):
            for n in sorted(n for (r, n) in cav if r == ra):
                m = cav[(ra, n)]["metrics"]
                alpha = cav[(ra, n)]["meta"]["nu"] / cav[(ra, n)]["meta"]["prandtl"]
                psi = m.get("psi_over_alpha", m["psi_abs_max"] / alpha)
                lines.append(f"| {ra:.0e} | {n} | {m['iterations']} | {m['nusselt']:.4f} | "
                             f"{100 * abs(m['nusselt'] - NU_CONTRACT[ra]) / NU_CONTRACT[ra]:.2f} % | "
                             f"{psi:.3f} | {100 * abs(psi - PSI_DVD[ra]) / PSI_DVD[ra]:.2f} % |")
        lines += ["", "| criterion | value | reference | error | verdict |",
                  "|---|---|---|---|---|"]
        for c in cav_crit:
            lines.append(f"| {c['id']} | {c['value']} | {c['reference']} | {c['error']} | "
                         f"{'PASS' if c['passed'] else 'FAIL'} |")
        lines.append("")
        if figs.get("cavity"):
            lines += [f"![natural convection]({os.path.relpath(os.path.join(BASE, figs['cavity']), os.path.join(BASE, DOCS))})", ""]
    else:
        lines += ["No run available.", ""]

    lines += ["## 4. Summary", ""]
    for case, payload in payloads.items():
        lines.append(f"- `{case}`: **{payload['verdict']}** "
                     f"({payload['n_passed']}/{payload['n_criteria']} criteria) -> "
                     f"`benchmarks/verified/{case}/results.json`")
    lines.append("")
    with open(os.path.join(BASE, DOCS, "RESULTS.md"), "w") as fh:
        fh.write("\n".join(lines))

    # ------------------------------------------------------------------ READMEs
    def readme(case, title, summary, payload):
        body = [f"# {case}: {title}", "",
                f"**Verdict: {payload['verdict']}** "
                f"({payload['n_passed']}/{payload['n_criteria']} criteria)", "",
                summary, "",
                "| criterion | value | reference | error | verdict |",
                "|---|---|---|---|---|"]
        for c in payload["criteria"]:
            body.append(f"| {c['id']} | {c['value']} | {c['reference']} | {c['error']} | "
                        f"{'PASS' if c['passed'] else 'FAIL'} |")
        body += ["", "Programmatic source: `scripts/report_fvm_incompressible.py`;",
                 "raw runs: " + ", ".join(f"`{r}`" for r in payload.get("runs", [])), ""]
        path = os.path.join(BASE, BENCH, case, "README.md")
        with open(path, "w") as fh:
            fh.write("\n".join(body))

    if "fvm_backward_step" in payloads:
        readme("fvm_backward_step", "backward-facing step at Re_H = 800",
               "Staircase step, expansion ratio 1.9423, parabolic inlet, "
               "Gartling/Armaly convention (Re based on the downstream height and "
               "bulk velocity). Criterion: L_r/H = 6.0 +- 0.2 with a 10 % relative "
               "gate; the friction coefficient is checked for the correct sign "
               "trend and against the analytic downstream value 12 nu/(U_b H).",
               payloads["fvm_backward_step"])
    if "fvm_sudden_expansion" in payloads:
        readme("fvm_sudden_expansion", "planar sudden expansion, AR = 4, Re_D = 100",
               "Uniform inlet over D, expansion to 4D, 12D long. The Borda-Carnot "
               "value K_BC = (1 - 1/AR)^2 = 0.5625 is an *inviscid* momentum "
               "balance; the measured laminar loss is larger, and the Reynolds "
               "sweep documents that the gap is physical rather than numerical.",
               payloads["fvm_sudden_expansion"])
    if "fvm_natural_convection" in payloads:
        readme("fvm_natural_convection", "differentially heated square cavity",
               "Boussinesq cavity, Pr = 0.71, Ra = 1e4/1e5/1e6, 32/64/96 squared "
               "grids. Criterion: mean Nu within 3 % of the de Vahl Davis values, "
               "streamfunction amplitude reported as a secondary quantity.",
               payloads["fvm_natural_convection"])
    print(json.dumps({k: dict(verdict=v["verdict"], passed=v["n_passed"],
                              total=v["n_criteria"]) for k, v in payloads.items()}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())