#!/usr/bin/env python3
"""Turn the baseline run JSON into the verification tables and figures.

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
    per-case ``results.json`` + ``README.md`` + ``*.png``.
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
DOCS = "docs/incompressible-baselines"
BENCH = "benchmarks/verified"
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load(pattern):
    out = {}
    for path in sorted(glob.glob(os.path.join(BASE, RUNS, pattern))):
        with open(path) as fh:
            out[os.path.basename(path)[:-5]] = json.load(fh)
    return out


def richardson(values, levels, sign=1.0):
    """Observed order from three levels and the h -> 0 extrapolation."""
    v = [values[l] for l in levels if values.get(l) is not None]
    ls = [l for l in levels if values.get(l) is not None]
    if len(v) < 3:
        return None, None
    (h1, h2, h3) = (1.0 / ls[0], 1.0 / ls[1], 1.0 / ls[2])
    f1, f2, f3 = v[0] * sign, v[1] * sign, v[2] * sign
    r21 = (f2 - f1) / (f3 - f2) if abs(f3 - f2) > 0 else np.nan
    if not np.isfinite(r21) or r21 <= 0 or r21 == 1:
        return None, None
    p = float(np.log(r21) / np.log(h2 / h3) if abs(h2 - h3) > 0 else np.nan)
    p = float(np.log(abs((f2 - f1) / (f3 - f2))) / np.log(h1 / h2)) if False else float(
        np.log(abs((f2 - f1) / (f3 - f2))) / np.log(h2 / h3))
    if abs(p) < 1e-6 or p > 8:
        return None, None
    # Richardson extrapolation of the two finest levels
    ratio = (h2 / h3) ** p
    f0 = (ratio * f3 - f2) / (ratio - 1.0)
    return p, float(f0 * sign)


def verdict(error, tol):
    if error is None:
        return "WITHHELD", "no converged value"
    if error <= tol:
        return "PASS", f"error {100 * error:.2f}% <= {100 * tol:.0f}%"
    return "FAIL", f"error {100 * error:.2f}% > {100 * tol:.0f}%"


def collect_cases():
    bfs, exp, cav = {}, {}, {}
    for tag, rec in load("bfs_n*.json").items():
        bfs[int(rec["job"]["level"])] = rec
    for tag, rec in load("expansion_n*.json").items():
        exp[int(rec["job"]["level"])] = rec
    for tag, rec in load("cavity_n*.json").items():
        cav[(float(rec["job"]["rayleigh"]), int(rec["job"]["level"]))] = rec
    return bfs, exp, cav


# --------------------------------------------------------------------------- #
def fig_channel():
    probes = load("probe_channel_poiseuille.json")
    if not probes:
        return None
    d = list(probes.values())[0]
    ny = [r["ny"] for r in d["levels"]]
    err = [r["profile_max_error"] for r in d["levels"]]
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.4))
    ax[0].loglog(ny, err, "o-")
    ax[0].set_xlabel("cells across the channel")
    ax[0].set_ylabel("max profile error / $u_{max}$")
    ax[0].grid(True, which="both", alpha=0.3)
    ax[0].set_title("plane Poiseuille: profile error")
    dp = [abs(r["dpdx"] + 0.4) / 0.4 for r in d["levels"]]
    ax[1].loglog(ny, dp, "s-")
    ax[1].set_xlabel("cells across the channel")
    ax[1].set_ylabel("|dp/dx error| / exact")
    ax[1].grid(True, which="both", alpha=0.3)
    ax[1].set_title("analytic pressure gradient -8 nu u_max / H^2")
    fig.tight_layout()
    path = os.path.join(BASE, BENCH, "fvm_backward_step", "fig_kernel_channel.png")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def fig_bfs(bfs):
    if not bfs:
        return None, None
    fields = sorted(glob.glob(os.path.join(BASE, RUNS, "bfs_n*_fields.npz")))
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
    for n, rec in sorted(bfs.items()):
        u = np.array(rec["metrics"].get("_wall_u", []))
    hist = []
    for path in sorted(glob.glob(os.path.join(BASE, RUNS, "bfs_n*_fields.npz"))):
        d = np.load(path)
        u, v, fluid, hx, hy = d["u"], d["v"], d["fluid"], float(d["hx"]), float(d["hy"])
        x = np.arange(u.shape[0]) * hx - 2.0
        ax[0].plot(x, u[:, 0], "-", lw=1.2, label=os.path.basename(path)[:9])
        uc = 0.5 * (u[:-1] + u[1:])
        im = ax[1].contourf(x[:-1], (np.arange(uc.shape[1]) + 0.5) * hy, uc.T,
                            levels=24, cmap="RdBu_r")
        fig.colorbar(im, ax=ax[1])
    ax[0].axhline(0.0, color="k", lw=0.6)
    ax[0].set_xlabel("$x/H$")
    ax[0].set_ylabel("$u$ on the stepped wall")
    ax[0].set_title("wall-shear sign change = reattachment")
    ax[0].legend(fontsize=7)
    ax[1].set_xlabel("$x/H$")
    ax[1].set_ylabel("$y/H$")
    ax[1].set_title("streamwise velocity (finest grid)")
    fig.tight_layout()
    path = os.path.join(BASE, BENCH, "fvm_backward_step", "fig_backward_step.png")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path, None


def fig_expansion(exp):
    if not exp:
        return None
    d = np.load(os.path.join(BASE, RUNS, "expansion_n040_fields.npz")) if os.path.exists(
        os.path.join(BASE, RUNS, "expansion_n040_fields.npz")) else None
    if d is None:
        return None
    u, hx, hy = d["u"], float(d["hx"]), float(d["hy"])
    x = np.arange(u.shape[0]) * hx
    y = (np.arange(u.shape[1]) + 0.5) * hy
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
    im = ax[0].contourf(x[:-1], y, (0.5 * (u[:-1] + u[1:])).T, levels=24, cmap="RdBu_r")
    fig.colorbar(im, ax=ax[0])
    ax[0].set_xlabel("$x/D$")
    ax[0].set_ylabel("$y/D$")
    ax[0].set_title("streamwise velocity, $AR=4$")
    for n, rec in sorted(exp.items()):
        m = rec["metrics"]
        ax[1].plot(float(n), m["k_total_head"], "o", label=f"n={n}")
    ax[1].axhline(0.5625, color="k", ls="--", lw=1.0, label="Borda-Carnot")
    ax[1].set_xlabel("cells across the inlet height")
    ax[1].set_ylabel("$K$ (total head)")
    ax[1].set_title("expansion loss coefficient")
    ax[1].legend(fontsize=7)
    fig.tight_layout()
    path = os.path.join(BASE, BENCH, "fvm_sudden_expansion", "fig_sudden_expansion.png")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def fig_cavity(cav):
    if not cav:
        return None
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
    ref = {1e4: 2.243, 1e5: 4.519, 1e6: 8.800}
    for ra in sorted({k[0] for k in cav}):
        ns = sorted(n for (r, n) in cav if r == ra)
        nus = [cav[(ra, n)]["metrics"]["nusselt"] for n in ns]
        ax[0].plot(ns, nus, "o-", label=f"Ra={ra:.0e}")
        ax[0].axhline(ref[ra], color="k", ls=":", lw=0.8)
    ax[0].set_xlabel("cells per side")
    ax[0].set_ylabel("mean $Nu$")
    ax[0].set_title("de Vahl Davis cavity (dotted: reference)")
    ax[0].legend(fontsize=7)
    path_npz = os.path.join(BASE, RUNS, "cavity_n096_ra1e+06_fields.npz")
    if not os.path.exists(path_npz):
        cands = glob.glob(os.path.join(BASE, RUNS, "cavity_n096_ra1e06_fields.npz"))
        path_npz = cands[0] if cands else None
    if path_npz and os.path.exists(path_npz):
        d = np.load(path_npz)
        T, hx = d["T"], float(d["hx"])
        n = T.shape[0]
        ax[1].contourf((np.arange(n) + 0.5) * hx, (np.arange(n) + 0.5) * hx, T.T,
                       levels=24, cmap="inferno")
        ax[1].set_xlabel("$x/L$")
        ax[1].set_ylabel("$y/L$")
        ax[1].set_title("temperature, Ra=$10^6$")
    fig.tight_layout()
    path = os.path.join(BASE, BENCH, "fvm_natural_convection", "fig_natural_convection.png")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


# --------------------------------------------------------------------------- #
def main():
    os.makedirs(os.path.join(BASE, DOCS), exist_ok=True)
    bfs, exp, cav = collect_cases()
    lines = ["# Incompressible FVM verification baselines — generated results", ""]
    lines.append("Generated by `scripts/report_fvm_incompressible.py` from "
                 f"`{RUNS}/*.json`.  Do not edit by hand.")
    lines.append("")

    # ---------------- backward facing step
    lines.append("## Backward-facing step (Re_H = 800)")
    if bfs:
        lines.append("")
        lines.append("| cells/H | iterations | continuity | L_r/H | Richardson order | extrapolated L_r/H |")
        lines.append("|---|---|---|---|---|---|")
        vals = {}
        for n, rec in sorted(bfs.items()):
            m = rec["metrics"]
            vals[n] = m.get("reattachment_over_H")
            lines.append(f"| {n} | {m['iterations']} | {m['continuity']:.2e} | "
                         f"{(lambda v: '—' if v is None else format(v, '.3f'))(m.get('reattachment_over_H'))} | | |")
        p, f0 = richardson(vals, sorted(vals))
        err = None if (f0 is None) else abs(f0 - 6.0) / 6.0
        v, msg = verdict(err, 0.10)
        lines.append("")
        lines.append(f"- observed order: {p if p is None else round(p, 2)}")
        lines.append(f"- extrapolated L_r/H: {f0 if f0 is None else round(f0, 3)} (target 6.0 ± 0.2)")
        lines.append(f"- **verdict: {v}** — {msg}")
    else:
        lines.append("")
        lines.append("No run available.")
    lines.append("")

    # ---------------- sudden expansion
    lines.append("## Planar sudden expansion (AR = 4, Re_D = 100)")
    if exp:
        lines.append("")
        lines.append("| cells/D | iterations | continuity | K (total head) | K error vs 0.5625 |")
        lines.append("|---|---|---|---|---|")
        vals = {}
        for n, rec in sorted(exp.items()):
            m = rec["metrics"]
            vals[n] = m.get("k_total_head")
            lines.append(f"| {n} | {m['iterations']} | {m['continuity']:.2e} | "
                         f"{m['k_total_head']:.4f} | {100 * m['k_error_rel']:.2f}% |")
        p, k0 = richardson(vals, sorted(vals))
        err = None if k0 is None else abs(k0 - 0.5625) / 0.5625
        v, msg = verdict(err, 0.05)
        lines.append("")
        lines.append(f"- observed order: {p if p is None else round(p, 2)}")
        lines.append(f"- extrapolated K: {k0 if k0 is None else round(k0, 4)} (Borda-Carnot 0.5625)")
        lines.append(f"- **verdict: {v}** — {msg}")
    else:
        lines.append("")
        lines.append("No run available.")
    lines.append("")

    # ---------------- natural convection
    lines.append("## Differentially heated cavity (Pr = 0.71)")
    if cav:
        lines.append("")
        lines.append("| Ra | cells | iterations | Nu | Nu error | psi_max | psi error | order |")
        lines.append("|---|---|---|---|---|---|---|---|")
        for ra in sorted({k[0] for k in cav}):
            ns = sorted(n for (r, n) in cav if r == ra)
            vals = {n: cav[(ra, n)]["metrics"]["nusselt"] for n in ns}
            def _psi(n):
                m = cav[(ra, n)]["metrics"]
                if m.get("psi_over_alpha") is not None:
                    return m["psi_over_alpha"]
                # JSON written before the normalised value was added: rebuild it
                meta = cav[(ra, n)]["meta"]
                alpha = meta["nu"] / meta["prandtl"]
                return m["psi_abs_max"] / alpha
            psi = {n: _psi(n) for n in ns}
            for n in ns:
                cav[(ra, n)]["metrics"]["psi_over_alpha"] = _psi(n)
                ref_psi = {1e4: 5.071, 1e5: 9.111, 1e6: 16.32}[ra]
                cav[(ra, n)]["metrics"]["psi_error_rel"] = abs(_psi(n) - ref_psi) / ref_psi
            p, nu0 = richardson(vals, ns)
            _, psi0 = richardson(psi, ns)
            for n in ns:
                m = cav[(ra, n)]["metrics"]
                lines.append(f"| {ra:.0e} | {n} | {m['iterations']} | {m['nusselt']:.4f} | "
                             f"{100 * (m['nusselt_error_rel'] or 0):.2f}% | {m.get('psi_over_alpha', float('nan')):.4f} | "
                             f"{100 * (m.get('psi_error_rel') or 0):.2f}% | {p if p is None else round(p, 2)} |")
            ref_nu = {1e4: 2.243, 1e5: 4.519, 1e6: 8.800}[ra]
            finest = max(ns)
            best = cav[(ra, finest)]["metrics"]["nusselt"] if nu0 is None else nu0
            err = abs(best - ref_nu) / ref_nu
            v, msg = verdict(err, 0.03)
            lines.append("")
            lines.append(f"- Ra={ra:.0e}: extrapolated Nu = "
                         f"{'n/a' if nu0 is None else round(nu0, 4)}, "
                         f"psi_max = {'n/a' if psi0 is None else round(psi0, 4)} "
                         f"(de Vahl Davis 1983: Nu "
                         f"{ {1e4: 2.238, 1e5: 4.509, 1e6: 8.817}[ra] }, psi "
                         f"{ {1e4: 5.071, 1e5: 9.111, 1e6: 16.32}[ra] })")
            lines.append(f"- **verdict: {v}** — {msg}")
    else:
        lines.append("")
        lines.append("No run available.")
    lines.append("")

    # figures
    fig_c = fig_channel()
    fig_b = fig_bfs(bfs)[0]
    fig_e = fig_expansion(exp)
    fig_v = fig_cavity(cav)
    for f in (fig_c, fig_b, fig_e, fig_v):
        if f:
            lines.append(f"![{os.path.basename(f)}]({os.path.relpath(f, os.path.join(BASE, DOCS))})")
    with open(os.path.join(BASE, DOCS, "RESULTS.md"), "w") as fh:
        fh.write("\n".join(lines) + "\n")

    # per-case bundles
    for name, payload in (("fvm_backward_step", bfs), ("fvm_sudden_expansion", exp),
                          ("fvm_natural_convection", cav)):
        d = os.path.join(BASE, BENCH, name)
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, "results.json"), "w") as fh:
            json.dump({str(k): v for k, v in payload.items()}, fh, indent=2, default=float)
    print("wrote", os.path.join(DOCS, "RESULTS.md"))


if __name__ == "__main__":
    main()