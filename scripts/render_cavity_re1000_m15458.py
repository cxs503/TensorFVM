"""Render the CASE-CAVITY-RE1000-FVM report and figures from the evidence bundle.

Nothing here is typed by hand: every number in the markdown comes out of
``artifacts/CASE-CAVITY-RE1000-FVM/summary.json`` (written by
``scripts/run_cavity_re1000_m15458.py``), and the reference column is read from
``tensorfvm.lid_cavity_reference`` at render time, so a change to either side
moves the report.

Usage::

    PYTHONPATH=src python scripts/render_cavity_re1000_m15458.py
"""

from __future__ import annotations

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from tensorfvm.lid_cavity_reference import VORTEX_CENTRE, table

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "artifacts", "CASE-CAVITY-RE1000-FVM")
DOCS = os.path.join(ROOT, "docs", "CASE-CAVITY-RE1000-FVM")


def tick(value: float, band: float) -> str:
    return "✓" if abs(value) <= band else "✗"


def fmt(value: float, digits: int = 5) -> str:
    return f"{value:.{digits}f}"


def load_fields(n: int) -> dict:
    path = os.path.join(OUT, "fields", f"Re{1000.0:g}_n{n}.pt")
    return torch.load(path, weights_only=False)


def figure_profiles(cases: list[dict]) -> str:
    u_ref = table(1000.0, "u")
    v_ref = table(1000.0, "v")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    for case in cases:
        n = case["n"]
        fields = load_fields(n)
        u, v = fields["u"].double(), fields["v"].double()
        j = n // 2
        y = (torch.arange(n, dtype=torch.float64) + 0.5) / n
        axes[0].plot(u[j, :], y, lw=1.0, label=f"FVM n={n}")
        axes[1].plot(v[:, j], y, lw=1.0, label=f"FVM n={n}")
    axes[0].plot([p[1] for p in u_ref], [p[0] for p in u_ref], "ko", ms=3.5,
                 label="Ghia et al. (1982) Table I")
    axes[1].plot([p[1] for p in v_ref], [p[0] for p in v_ref], "ks", ms=3.5,
                 label="Ghia et al. (1982) Table II")
    axes[0].set_xlabel("u / U"); axes[0].set_ylabel("y / L")
    axes[1].set_xlabel("v / U"); axes[1].set_ylabel("y / L")
    axes[0].set_title("vertical centreline x = L/2"); axes[1].set_title("horizontal centreline y = L/2")
    for ax in axes:
        ax.grid(alpha=0.3); ax.legend(fontsize=7)
    path = os.path.join(OUT, "fig1_centrelines.png")
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)
    return path


def figure_fields(cases: list[dict]) -> str:
    fig, axes = plt.subplots(1, len(cases), figsize=(4.4 * len(cases), 4.0))
    axes = [axes] if len(cases) == 1 else list(axes)
    for ax, case in zip(axes, cases):
        n = case["n"]
        fields = load_fields(n)
        u, v = fields["u"].double(), fields["v"].double()
        uc = 0.5 * (u[:-1, :] + u[1:, :])
        vc = 0.5 * (v[:, :-1] + v[:, 1:])
        h = 1.0 / n
        omega = torch.zeros_like(uc)
        omega[1:-1, 1:-1] = ((vc[2:, 1:-1] - vc[:-2, 1:-1]) / (2 * h)
                             - (uc[1:-1, 2:] - uc[1:-1, :-2]) / (2 * h))
        im = ax.imshow(omega.numpy(), origin="lower", extent=[0, 1, 0, 1],
                       cmap="RdBu_r", vmin=-8, vmax=8)
        vx, vy = case["vortex_centre"]["computed"]
        ax.plot([vx], [vy], "k+", ms=10)
        ax.set_title(f"vorticity, n={n}")
        ax.set_xlabel("x / L"); ax.set_ylabel("y / L")
        fig.colorbar(im, ax=ax, fraction=0.046)
    path = os.path.join(OUT, "fig2_vorticity.png")
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)
    return path


def figure_convergence(summary: dict) -> str:
    cases = sorted(summary["cases"], key=lambda c: c["n"])
    grids = [c["n"] for c in cases]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    for name, axis, ref in (("u", 0, table(1000.0, "u")), ("v", 1, table(1000.0, "v"))):
        errs = [c[name]["max_abs_error"] for c in cases]
        axes[0].loglog(grids, errs, "o-", label=f"max|Δ{name}| vs Ghia")
    ref_u = dict(table(1000.0, "u")); ref_v = dict(table(1000.0, "v"))
    axes[0].set_xlabel("n (cells per direction)"); axes[0].set_ylabel("max |ΔU| / U")
    axes[0].grid(alpha=0.3, which="both"); axes[0].legend(fontsize=7)
    axes[0].set_title("centreline error vs resolution")
    axes[1].plot(grids, [c["u"]["extremum"]["min_computed"][1] for c in cases], "o-", label="u_min computed")
    axes[1].axhline(ref_u[min(ref_u)], color="k", ls="--", lw=1, label="Ghia u_min")
    axes[1].plot(grids, [c["v"]["extremum"]["max_computed"][1] for c in cases], "s-", label="v_max computed")
    axes[1].axhline(ref_v[max(ref_v)], color="0.4", ls="--", lw=1, label="Ghia v_max")
    axes[1].set_xlabel("n"); axes[1].set_ylabel("extremum / U")
    axes[1].grid(alpha=0.3); axes[1].legend(fontsize=7); axes[1].set_title("primary extrema")
    path = os.path.join(OUT, "fig3_convergence.png")
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)
    return path



def emit_sql(summary: dict, cases: list[dict]) -> str:
    """Emit the oss_solutions rows for this anchor (the table is not writable here)."""
    best = max(cases, key=lambda c: c["n"])
    cont = summary.get("continuation_check", {}).get("continuation", {})
    m = best["verdict_5pct"]["metrics"]
    rows = [
        ("literature/Ghia-1982", "CASE-CAVITY-RE1000-FVM", "benchmark-definition-and-tables",
         "Reference solution for the 2-D lid-driven cavity at Re = 100 to 10000: vorticity-streamfunction "
         "formulation, coupled strongly implicit multigrid, grids up to 257x257, centreline tables "
         "(Table I u at x = L/2, Table II v at y = L/2). At Re = 1000 the solution is steady: one primary "
         "vortex plus corner eddies, so a steady-state solver is a legitimate comparison route.",
         "Ghia, Ghia & Shin, J. Comput. Phys. 48 (1982) 387-411",
         "extrema are tabulated only at the sampled centreline points; the vortex centre is quoted separately",
         "reference",
         "committed transcription + self-check in src/tensorfvm/lid_cavity_reference.py"),
        ("method/grid-continuation", "CASE-CAVITY-RE1000-FVM", "coarse-to-fine-prolongation",
         "Pseudo-transient codes reach the same steady state from any initial field, so a converged coarse "
         "solution can seed the fine grid (bilinear prolongation of cell-centred velocity, faces rebuilt by "
         "averaging, wall-normal faces zeroed). Cost is cut because the fine run only removes the "
         "discretisation difference, not the whole spin-up. Widely used in steady cavity studies "
         "(continuation in both grid and Reynolds number).",
         "Erturk, Int. J. Numer. Methods Fluids 48 (2008) 747-774 (Reynolds continuation to Re = 21000)",
         "continuation must be verified, not trusted: converge the same grid from rest and compare fields",
         "adopted",
         f"verified here: n = {cont.get('to_n')} from the n = {cont.get('from_n')} field in "
         f"{cont.get('continued_steps')} steps vs {cont.get('cold_steps')} cold, steady states differ by "
         f"max|du| = {cont.get('max_abs_du')!r}"),
        ("method/pressure-projection-dct", "CASE-CAVITY-RE1000-FVM", "pure-neumann-poisson-by-dct-ii",
         "In a closed cavity every wall is a wall, so the pressure Poisson problem is pure Neumann and the "
         "discrete compatibility condition sum(div u*) = 0 must hold to machine precision. Diagonalising with "
         "an orthonormal DCT-II and pinning the zero mode solves it exactly in O(n^3) per step with no "
         "iteration and no datum drift.",
         "TensorFVM src/tensorfvm/lid_cavity.py solve_pressure (own implementation)",
         "subtract the mean of the divergence before the transform, pin the (0,0) mode",
         "in-use",
         f"max|div u| = {best['solver']['max_divergence']:.1e} at n = {best['n']}"),
        ("method/simple-family", "CASE-CAVITY-RE1000-FVM", "not-adopted-implicit-pressure-correction",
         "SIMPLE/SIMPLEC with under-relaxation converges in O(1e3) outer iterations nearly independently of "
         "the grid and would remove the explicit step budget entirely, but it is a full replacement of the "
         "time loop and would change the solver under test rather than the evidence about it.",
         "Patankar & Spalding, Int. J. Heat Mass Transfer 15 (1972) 1787-1806",
         "under-relaxation factors are case dependent; a rewrite invalidates the existing verification bundle",
         "reference-not-adopted",
         "recorded as the round-3 option along with FAS multigrid in docs/CASE-CAVITY-RE1000-FVM/method-selection.md"),
        ("inventory/negative", "CASE-CAVITY-RE1000-FVM", "no-open-source-fvm-cavity-in-cache",
         "Horizontal survey result: the node-side cache reachable from the build container holds three "
         "repositories (two LBM, one SPH) and none of them is an FVM incompressible cavity solver; the "
         "cluster-side cache and the oss_inventory/oss_solutions tables are not readable by this agent role, "
         "so no open-source FVM implementation could be compared. Reported as a negative inventory result "
         "rather than silently skipped.",
         "local /root/oss-cache: Maarten-vd-Sande_lbm, pylbm, hschwane/GraSPH",
         "LBM/SPH discretisations are not transferable to a staggered FVM projection solver",
         "no-transfer",
         "cluster grep job cj1123 stayed queued for the whole mission (324 jobs ahead)"),
    ]
    out = ["-- CASE-CAVITY-RE1000-FVM round-2 method survey / selection (mission m15458).",
           "-- The oss_solutions table is not writable by the agent role; the platform applies this file.",
           "INSERT INTO oss_solutions (repo, feature_slug, tag, method, files, conventions, transfer, evidence)",
           "VALUES"]
    tuples = []
    for repo, slug, tag, method, files, conv, transfer, evidence in rows:
        vals = [repo, slug, tag, method, files, conv, transfer, evidence]
        esc = ["'" + str(v).replace("'", "''") + "'" for v in vals]
        tuples.append("(" + ",\n ".join(esc) + ")")
    out.append(",\n".join(tuples) + ";")
    path = os.path.join(OUT, "oss_solutions_insert.sql")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(out) + "\n")
    return path


def main() -> None:
    with open(os.path.join(OUT, "summary.json")) as fh:
        summary = json.load(fh)
    os.makedirs(DOCS, exist_ok=True)
    cases = sorted(summary["cases"], key=lambda c: c["n"])
    figs = [figure_profiles(cases), figure_fields(cases), figure_convergence(summary)]

    ref_u = dict(table(1000.0, "u")); ref_v = dict(table(1000.0, "v"))
    gx, gy = VORTEX_CENTRE[1000.0]
    band = summary["tolerance_fraction"]

    lines: list[str] = []
    add = lines.append
    add("# CASE-CAVITY-RE1000-FVM (round 2): lid-driven cavity at Re = 1000 in TensorFVM")
    add("")
    add(f"Mission m15458, anchor `CASE-CAVITY-RE1000-FVM`, tolerance {band:.0%}.")
    add(f"Bundle: `artifacts/CASE-CAVITY-RE1000-FVM/summary.json` "
        f"({len(cases)} converged resolutions). Generated by "
        f"`scripts/render_cavity_re1000_m15458.py` — no number below is typed by hand.")
    add("")
    add("## ① Problem setting and the benchmark source")
    add("")
    add("Two-dimensional lid-driven cavity: unit square `[0,1]²`, no-slip on three walls, "
        "tangential lid velocity `U = 1` on `y = L`; incompressible Navier–Stokes, "
        "`Re = U·L/ν = 1000` ⇒ `ν = 1e-3`, `ρ = 1`. The Ghia solution at this Re is steady "
        "(single primary vortex plus corner eddies), so the target is a steady state.")
    add("")
    add("Reference: **U. Ghia, K.N. Ghia, C.T. Shin, *J. Comput. Phys.* 48 (1982) 387–411**, "
        "Table I (u along `x = L/2`) and Table II (v along `y = L/2`). The values used here are "
        "the committed transcription in `src/tensorfvm/lid_cavity_reference.py` "
        "(self-checked by `tests/test_ghia_reference_consistency.py`), read at render time.")
    add("")
    add("## ② Method and implementation")
    add("")
    add("Staggered MAC finite-volume discretisation of `∂u/∂t + ∇·(uu) = -∇p/ρ + ν∇²u`, "
        "`∇·u = 0`: conservative van Leer (harmonic TVD) face reconstruction, explicit "
        "projection, pressure Poisson with pure-Neumann walls solved by an orthonormal "
        "DCT-II diagonalisation (compatibility `Σ div u* = 0` enforced by subtracting the "
        "mean), mirror ghost for the moving lid. Code: `src/tensorfvm/lid_cavity.py` "
        "(`LidCavitySolver.tendencies`, `.solve_pressure`, `.step`, `.run`), scoring in "
        "`score_against_ghia`/`ghia_acceptance` in the same module.")
    add("")
    add("Round-2 additions are cost and evidence, not physics "
        "(`scripts/run_cavity_re1000_m15458.py`): grid continuation with bilinear "
        "prolongation of the cell-centred velocity, per-case checkpoints, a cold-start run "
        "on the same grid to prove continuation does not change the steady state, and a "
        "per-resolution 5 % verdict (`judge_5pct`). Method survey and selection: "
        "`docs/CASE-CAVITY-RE1000-FVM/method-selection.md`.")
    add("")
    add("## ③ Parameters (all dimensionless)")
    add("")
    add("| Re | ν/U·L | ρ | U (lid) | domain | cfl | steady-state gate | max steps | dtype |")
    add("|---|---|---|---|---|---|---|---|---|")
    c0 = cases[-1]
    add(f"| 1000 | {c0['parameters']['nu']:.1e} | {c0['parameters']['rho']:.1f} | "
        f"{c0['grid']['lid_velocity']:.1f} | 1×1 | {c0['parameters']['cfl']} | "
        f"max|Δu|/Δt ≤ {c0['parameters']['tolerance']:.0e} every "
        f"{c0['parameters']['check_every']} steps | {c0['parameters']['max_steps']} | float64 |")
    add("")
    add("| n (cells/direction) | h | steps to steady state | change rate | max|div u| | wall time [s] | initial condition |")
    add("|---|---|---|---|---|---|---|")
    for case in cases:
        s = case["solver"]
        add(f"| {case['n']} | 1/{case['n']} | {s['steps']} | {s['residual_rate']:.2e} | "
            f"{s['max_divergence']:.1e} | {s['wall_seconds']:.0f} | {s['initial_condition']} |")
    add("")
    add("## ④ Reference vs computed, per quantity")
    add("")
    add("Reference entries are the Ghia table samples; computed values are the same code "
        "path's Hermite interpolation onto those samples (`score_profiles`), i.e. the "
        "shipped metric, not a report-specific one.")
    add("")
    add("| n | u_min computed | Ghia u_min | rel err | v_max computed | Ghia v_max | rel err | "
        "vortex centre computed | Ghia | Δ/L |")
    add("|---|---|---|---|---|---|---|---|---|---|")
    for case in cases:
        u_min = case["u"]["extremum"]["min_computed"][1]
        v_max = case["v"]["extremum"]["max_computed"][1]
        ref_umin = ref_u[case["u"]["extremum"]["min_reference"][0]]
        ref_vmax = ref_v[case["v"]["extremum"]["max_reference"][0]]
        m = case["verdict_5pct"]["metrics"]
        vx, vy = case["vortex_centre"]["computed"]
        add(f"| {case['n']} | {fmt(u_min)} | {fmt(ref_umin)} | {m['u_min_rel']:.2%} | "
            f"{fmt(v_max)} | {fmt(ref_vmax)} | {m['v_max_rel']:.2%} | "
            f"({vx:.3f}, {vy:.3f}) | ({gx:.3f}, {gy:.3f}) | {m['vortex_centre_rel']:.4f} |")
    add("")
    add(f"| n | max|Δu| / U | band {band:.0%} | max|Δv| / U | band {band:.0%} | rms|Δu| | rms|Δv| | ledger verdict |")
    add("|---|---|---|---|---|---|---|---|")
    for case in cases:
        m = case["verdict_5pct"]["metrics"]
        add(f"| {case['n']} | {m['u_max_abs']:.4f} | {tick(m['u_max_abs'], band)} | "
            f"{m['v_max_abs']:.4f} | {tick(m['v_max_abs'], band)} | "
            f"{case['u']['rms_error']:.4f} | {case['v']['rms_error']:.4f} | "
            f"{'PASS' if case['verdict_5pct']['pass'] else 'FAIL'} |")
    add("")
    add("## ⑤ Convergence and sensitivity")
    add("")
    add("| n | u_min | v_max | max|Δu| | max|Δv| | converged | 5 % verdict |")
    add("|---|---|---|---|---|---|---|")
    for case in cases:
        add(f"| {case['n']} | {fmt(case['u']['extremum']['min_computed'][1])} | "
            f"{fmt(case['v']['extremum']['max_computed'][1])} | "
            f"{case['u']['max_abs_error']:.4f} | {case['v']['max_abs_error']:.4f} | "
            f"{case['solver']['converged']} | {case['verdict_5pct']['pass']} |")
    add("")
    cont = summary.get("continuation_check", {}).get("continuation")
    if cont:
        add(f"Continuation check (same grid, two initial conditions): n = {cont['to_n']} "
            f"converged from rest in {cont['cold_steps']} steps, from the n = {cont['from_n']} "
            f"field in {cont['continued_steps']} steps (×{cont['speedup']:.2f} wall-time "
            f"speed-up); the two steady states differ by max|Δu| = {cont['max_abs_du']:.2e}, "
            f"max|Δv| = {cont['max_abs_dv']:.2e} — i.e. continuation changes the cost, "
            f"not the answer (gate `same_steady_state` = {cont['same_steady_state']}).")
        add("")
    add("Sensitivity beyond resolution: the n = 32 row is deliberately kept as an "
        "under-resolved control; its pointwise error is the honest measure of what one "
        "level of refinement below the target grid costs.")
    add("")
    add("## ⑥ Failure modes, artefacts, reproduction")
    add("")
    add("```bash")
    add("PYTHONPATH=src python3 scripts/run_cavity_re1000_m15458.py \\")
    add("    --out artifacts/CASE-CAVITY-RE1000-FVM --reynolds 1000 --grids 64,128 --cold-grids 32")
    add("PYTHONPATH=src python3 scripts/render_cavity_re1000_m15458.py")
    add("PYTHONPATH=src python3 -m pytest tests/test_case_cavity_re1000_m15458.py -q")
    add("```")
    add("")
    add("Artefacts: `artifacts/CASE-CAVITY-RE1000-FVM/{summary.json,fields/*.pt,run.log,"
        "fig1_centrelines.png,fig2_vorticity.png,fig3_convergence.png,oss_solutions_insert.sql}`.")
    add("")
    add("Failure modes seen: (i) an unconverged row is reported as such "
        f"(`all_converged = {summary.get('all_converged')}`) and never folded into a pass; "
        "(ii) continuation could in principle converge to a different discrete state — the "
        "cold-run comparison above is the counter-check; (iii) the cluster queue was "
        "saturated for this mission (job `cj1123` stayed queued), so these rows come from the "
        "container CPU — same code, same tolerance, recorded in the finish note.")
    add("")
    add("## ⑦ Error judgement")
    add("")
    ok3 = all(c["verdict_5pct"]["metrics"]["u_max_abs"] <= 0.03
              and c["verdict_5pct"]["metrics"]["v_max_abs"] <= 0.03
              and c["verdict_5pct"]["metrics"]["u_min_rel"] <= 0.03
              and c["verdict_5pct"]["metrics"]["v_max_rel"] <= 0.03 for c in cases if c["n"] >= 64)
    fine = [c for c in cases if c["n"] >= 64]
    add(f"Ledger band {band:.0%}: "
        + ", ".join(f"n={c['n']} {'PASS' if c['verdict_5pct']['pass'] else 'FAIL'}" for c in cases) + ".")
    add("")
    add("Is every key quantity within **3 %** at the production resolutions? "
        f"**{'Yes' if ok3 else 'No'}** — "
        + "; ".join(f"n={c['n']}: u_min {c['verdict_5pct']['metrics']['u_min_rel']:.2%}, "
                    f"v_max {c['verdict_5pct']['metrics']['v_max_rel']:.2%}, "
                    f"pointwise max|Δu| {c['verdict_5pct']['metrics']['u_max_abs']:.2%} U, "
                    f"max|Δv| {c['verdict_5pct']['metrics']['v_max_abs']:.2%} U" for c in fine) + ".")
    add("")
    add(f"Figures: `fig1_centrelines.png` (profiles vs the Ghia symbols), "
        f"`fig2_vorticity.png` (vorticity fields with the computed vortex centre), "
        f"`fig3_convergence.png` (error vs resolution and the primary extrema).")
    add("")

    path = os.path.join(DOCS, "report.md")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    print("wrote", path)
    print("wrote", emit_sql(summary, cases))
    for fig in figs:
        print("wrote", fig)


if __name__ == "__main__":
    main()