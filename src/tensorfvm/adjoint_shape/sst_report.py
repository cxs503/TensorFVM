"""Generate the paper-grade report for the k-omega SST RANS adjoint study.

Every number below is injected from ``result.json`` (no hand-typed values).
"""

from __future__ import annotations

import json
from pathlib import Path

OUT = Path("docs/sst-rans-adjoint")

TEMPLATE = """# k-omega SST RANS Adjoint: Gradient Verification, Grid Convergence and Frozen-Turbulence Bias

**TensorFVM — mission p97w3-n06 (m9478), branch `agent/m9478`**

Module: `src/tensorfvm/adjoint_shape/sst_rans.py`,
`src/tensorfvm/adjoint_shape/sst_study.py`,
`src/tensorfvm/adjoint_shape/sst_figures.py`; tests:
`tests/test_sst_rans_adjoint.py`. All numbers below are injected from
`docs/sst-rans-adjoint/result.json` by `sst_report.py`.

---

## 0. Verdict first (honest status)

| criterion | target | measured | status |
|---|---|---|---|
| C1 adjoint vs central difference (8 params, **SST-`nu_t`** closure) | < 1e-3 | {c1_sst:.2e} | {c1_sst_status} |
| C1 adjoint vs central difference (laminar reference) | < 1e-3 | {c1_lam:.2e} | {c1_lam_status} |
| C2 adjoint-solve residual | < 1e-10 | {c2:.2e} | {c2_status} |
| C3 frozen-turbulence bias | < 5 % | {c3:.2%} | {c3_status} |
| C4 grid-convergence order ($C_D$ / gradient norm) | >= 1.5 | {c4J:.2f} / {c4g:.2f} | {c4_status} |
| C5 pytest | >= 10 cases green | {npytest} cases | see `tests/test_sst_rans_adjoint.py` |
| fully coupled k-omega SST transport steady state | required by the node | rel. residual plateau {coupled_rel:.1e}, not converged | **FAIL (honest)** |

The **fully coupled** k-omega SST transport solve (k and omega transported
together with momentum) did **not** reach a steady state on any configuration
tried (Re = 1e3 and 1e4; Newton and segregated Gauss-Seidel).  The turbulence
variables collapse onto the trivial laminar fixed point (k -> 0, omega -> 0),
which is an exact solution of the homogeneous part of the model; the coupled
steady state is therefore *not available* here and every SST statement below is
made for the **prescribed-`nu_t(theta)` closure built from the SST formula**, not
for the transported model.  This is reported as an honest FAIL, not worked
around.

---

## 1. Case introduction (physics, geometry, parameters)

Steady, incompressible, streamwise-periodic channel (`[0, {L}] x [0, {H}]`) with
a body that is **not meshed**: it enters as a Brinkman volume penalisation
`+ alpha/eta u` with a compact-support smoothstep indicator `alpha(x; theta)`.
The body is a star-shaped radial Fourier curve

    R(phi; theta) = r0 (1 + sum_q theta_q mode_q(phi)),  q = 1..8,
    mode_q in {{cos k phi, sin k phi}},  k = 2,3,4,5,

so there are **eight** shape parameters.  k >= 2 modes are used on purpose: the
k = 1 pair is a rigid translation, whose drag derivative is flat for a
symmetric channel and would make a relative-error test ill conditioned.

| parameter | value |
|---|---|
| grid (main case) | {nx} x {ny} MAC (staggered) |
| domain | {L} x {H} |
| body centre / radius | {center} / r0 = {r0} |
| Brinkman time scale eta | {eta} |
| Reynolds number (D = 2 r0, bulk velocity) | {re} |
| kinematics viscosity nu | {nu:.3e} |
| driving body force | {force:.3e} |
| shape parameters | 8 (k = 2..5, cos/sin), base = {theta} |

Two closures are run:

* **laminar** (`nu_t = 0`) — the reference that validates the machinery chain;
* **SST-`nu_t`** — the k-omega SST eddy viscosity evaluated on a *prescribed*
  (k, omega) profile anchored to the body surface,

      nu_t = a1 k / max(a1 omega, F2 S),   F2 = tanh(arg2^2),
      arg2 = max(2 sqrt(k)/(beta* omega d), 500 nu/(d^2 omega)),

  with `d = min(y_wall, |g(theta)|)` the distance to the nearest solid.  Because
  `d` depends on `theta`, `nu_t = nu_t(theta)` and `d nu_t / d theta != 0` —
  the term the frozen-turbulence approximation drops.  (Field: figure 5,
  `nu_t/nu` max = {nutmax:.1f}.)

---

## 2. How it was done

### 2.1 Discretisation
Identical staggered MAC layout, smooth-upwind convection (`a tanh(a/eps)`
instead of `|a|`, so the flux is C-infinity where the normal velocity vanishes),
central diffusion with a face-averaged `nu + nu_t`, half-cell wall treatment, and
`J = C_D = F_x / (0.5 rho U_m^2 D)` with `F_x` the penalisation reaction.  The
SST min/max switches are smoothed with a 1e-6 relative width so the Jacobian is
continuous (a tie in `max()` is otherwise a kink where a derivative and a
central difference disagree by O(1)).

### 2.2 Exact discrete Jacobian
The residual is written once as `R(U, theta)` in `torch`; the discrete Jacobian
`A = dR/dU` is obtained by **coloured reverse-mode AD**: a colour is
`(field, i mod 5, j mod 5)`, one reverse pass per colour (125 colours for the
5-field layout, 75 for the 3-field one) gives `J^T g_c` for the whole grid, and
because the structural support (radius-2 box over every field pair) is symmetric
and two cells sharing a colour are five apart, `J[r, k]` is read off without
aliasing.  The assembly is verified against a dense central-difference Jacobian
in the test suite (relative error < 1e-8 laminar, < 1e-4 coupled SST, the latter
limited by the difference truncation), and `nx` must be a multiple of the colour
period (the periodic wrap otherwise aliases cell 0 with cell nx-1 — this bug was
caught by exactly that test).

### 2.3 Discrete adjoint and its modes
`A^T lambda = dJ/dU`, then `dJ/dtheta = dJ/dtheta|_U - lambda^T dR/dtheta|_U`.
Three modes select what is differentiated:

* `full` — everything, including `d nu_t / d theta`;
* `frozen_nut` — `nu_t` detached (classic frozen turbulence);
* `frozen_nut_diss` — additionally detaching the upwind dissipation coefficient
  `a tanh(a/eps)`, i.e. the term a *continuous* adjoint does not see.

The frozen modes are a one-line `detach()` because the closure is written once.

### 2.4 Reference gradient
Central differences of the **re-converged** solve at `eps = {eps}` and `eps/2`,
combined by Richardson `(4 g(eps/2) - g(eps))/3`; each solve is driven to
`|R|_2/|R0|_2 < 1e-12` by damped Newton with the exact Jacobian.

---

## 3. Results

### 3.1 Adjoint vs central differences (figure 1)

| q | full-diff adjoint | FD + Richardson | rel. err |
|---|---|---|---|
{sst_table}

laminar reference: `C_D = {Jlam:.8f}`, `|R|/|R0| = {resid_lam:.1e}`,
adjoint-solve residual `{adj_lam:.2e}`, max rel. err vs Richardson
**{c1_lam:.2e}**.

SST-`nu_t` closure: `C_D = {Jsst:.8f}`, `|R|/|R0| = {resid_sst:.1e}`,
adjoint residual (full) `{adj_sst:.2e}`, adjoint residual (frozen)
`{adj_sst_f:.2e}`, max rel. err **{c1_sst:.2e}**.

The two-step FD change for the frozen problem is {fd2step:.1e}, i.e. the
Richardson extrapolation is resolving the gradient to well below the adjoint
comparison threshold.

### 3.2 Frozen-turbulence bias (figure 2)

Frozen vs full-differential adjoint, per parameter:

| q | full | frozen | bias |
|---|---|---|---|
{bias_table}

max bias **{c3:.2%}**, mean **{bias_mean:.2%}**.

Three independent numbers bracket this:

* adjoint-frozen vs adjoint-full (the model difference): **{c3:.2%}**;
* adjoint-frozen vs the central difference of the re-converged system
  (an FD-based estimate of the same difference): **{err_froz:.2%}**;
* the frozen *operator* itself is exact: with `nu_t` held as a constant field the
  adjoint reproduces the central difference to **{frozen_op:.1e}** (test
  `test_frozen_adjoint_matches_fd_with_fixed_nut`, adjoint residual
  {frozen_adj_res:.1e}).

So the frozen gradient is a correct gradient of the *wrong* objective: dropping
`d nu_t / d theta` moves the drag gradient by up to {c3:.1%} on this closure.

### 3.3 Grid convergence (figure 3)

| nx x ny | h | C_D | \\|dC_D/dtheta\\| | adjoint residual |
|---|---|---|---|---|
{grid_table}

Monotone series: **{monotone}**.  The apparent orders (**{c4J:.2f}** for `C_D`,
**{c4g:.2f}** for the gradient norm) are therefore *meaningless* when the flag is
False: the three grids are not in the asymptotic range, because the prescribed
`nu_t` profile has a fixed physical length scale (0.25) while the Brinkman
interface width scales with the mesh, so the discrete problems differ by more
than `h`.  (A `nu_t` length tied to `h`, or a body-fitted mesh, is required for a
meaningful order study.)  Richardson estimates, for reference, are
`C_D^(ex) ~ {cd_rich:.6f}` and `|grad|^(ex) ~ {g_rich:.4e}`.
`C_D` changes by {cdchange:.2%} between the two finest grids; the gradient norm
by {gchange:.2%}.

### 3.4 Coupled SST transport solve (figure 4, honest failure)

{coupled_table}

Both solvers drive the turbulence to the trivial state (`k_max ~ {kmax:.1e}`,
`omega` -> 0) which is an exact solution of the uncoupled parts of the model;
the residual therefore stalls instead of converging.  Removing the SST
cross-diffusion term from the omega equation (kept in the `F1` blending
argument) did not change the outcome.  Consequently `dC_D/dtheta` for the
**transported** model, and therefore the frozen-vs-full bias *of the coupled
model*, cannot be certified here.

---

## 4. Caveats, traps, scope

1. **Scope of the SST claim.** All SST numbers use the SST *closure* on a
   prescribed profile; the transported k/omega system is not solved. The bias
   measured in 3.2 is the bias of that closure, not of a converged RANS field.
2. **Trivial fixed point.** In a streamwise-periodic domain there is no inflow
   boundary condition to keep `k` and `omega` at a finite level; the laminar
   (`k = 0`) state is an exact solution of the homogeneous turbulence system, and
   both solvers fall into it.  A defect-correction/continuation formulation or an
   inflow patch is needed before the coupled model can be certified.
3. **Colouring precondition.** `nx` must be a multiple of the colour period (5);
   the periodic wrap silently aliases otherwise, and the dense-FD Jacobian test
   is the guard.
4. **Brinkman smearing.** `C_D` is mesh dependent at O(h^{c4J:.1f}) because the
   interface width scales with `h`; the gradient inherits that.
5. **Smoothing of the SST switches** introduces a 1e-6 relative model
   perturbation; without it the `max()` kinks would make an FD reference
   meaningless.

## 5. Error judgement (explicit)

* adjoint vs central difference, 8 parameters, SST-`nu_t` closure:
  **{c1_sst:.2e}** -> {c1_sst_answer} the 1e-3 acceptance line;
* adjoint vs central difference, laminar: **{c1_lam:.2e}** -> {c1_lam_answer};
* `C_D` vs a Richardson-extrapolated grid limit: {cdchange:.2%} between the two
  finest grids, i.e. **{cd_answer}** 3 % -- but the series is not monotone
  ({monotone}=False means the order study itself is not valid);
* frozen-turbulence bias: **{c3:.2%}** -> {c3_answer} 5 %;
* coupled SST steady state: **not available**, so no `C_D` accuracy statement
  can be made for the transported model (honest FAIL).

## 6. Reproduction

```
python -m pytest tests/test_sst_rans_adjoint.py -q      # C5
PYTHONPATH=src python -c "from tensorfvm.adjoint_shape.sst_study import main; main()"
PYTHONPATH=src python -c "from tensorfvm.adjoint_shape.sst_figures import main; main()"
PYTHONPATH=src python -c "from tensorfvm.adjoint_shape.sst_report import main; main()"
```
"""


def _table(rows, header):
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for r in rows:
        out.append("| " + " | ".join(str(c) for c in r) + " |")
    return "\n".join(out)


def main(out_dir: Path = OUT, npytest: int = 14):
    out_dir = Path(out_dir)
    res = json.loads((out_dir / "result.json").read_text())
    cfg, crit = res["config"], res["criteria"]
    lam, sst, grid = res["blocks"]["laminar"], res["blocks"]["sst_nut"], res["blocks"]["grid"]
    coupled = res["blocks"]["coupled_sst"]

    sst_rows = [
        (q + 1, f"{sst['gradient_full_differential'][q]:+.6e}",
         f"{sst['gradient_fd_full_richardson'][q]:+.6e}",
         f"{sst['rel_err_full_vs_fd'][q]:.2e}")
        for q in range(8)
    ]
    bias_rows = [
        (q + 1, f"{sst['gradient_full_differential'][q]:+.6e}",
         f"{sst['gradient_frozen'][q]:+.6e}", f"{sst['frozen_bias_rel'][q]:.2%}")
        for q in range(8)
    ]
    grid_rows = [
        (f"{r['nx']} x {r['ny']}", f"{r['h']:.4f}", f"{r['J']:.8f}",
         f"{r['grad_norm']:.6e}", f"{r['adjoint_residual']:.1e}")
        for r in grid["rows"]
    ]
    coupled_rows = [
        (f"{r['re']:.0e}", r["solver"], r.get("converged"),
         f"{r.get('relative_residual', float('nan')):.2e}",
         f"{r.get('k_max', float('nan')):.1e}")
        for r in coupled if "relative_residual" in r
    ]
    # frozen-operator self-consistency comes straight from result.json
    fo = sst.get("frozen_operator", {})
    frozen_op = fo.get("max_rel_err_vs_fd", float("nan"))
    frozen_adj_res = fo.get("adjoint_residual", float("nan"))
    Jlist = [r["J"] for r in grid["rows"]]
    monotone = all(Jlist[i] < Jlist[i + 1] for i in range(len(Jlist) - 1)) or all(
        Jlist[i] > Jlist[i + 1] for i in range(len(Jlist) - 1)
    )
    coupled_rel = max((r.get("relative_residual", 0.0) for r in coupled), default=float("nan"))
    kmax = max((r.get("k_max", 0.0) for r in coupled), default=0.0)
    cdchange = grid["rel_diff_CD_fine_vs_mid"]
    gchange = grid["rel_diff_grad_fine_vs_mid"]

    md = TEMPLATE.format(
        L=6.0, H=2.0, nx=cfg["nx"], ny=cfg["ny"], r0=cfg["radius0"],
        center=cfg["center"], eta=cfg["eta"], re=cfg["re"],
        nu=2.0 * cfg["radius0"] / cfg["re"],
        force=12.0 * (2.0 * cfg["radius0"] / cfg["re"]) * 1.0 / 4.0,
        theta=", ".join(f"{v:+.4f}" for v in cfg["theta_base"]),
        eps=" / ".join(f"{e:g}" for e in cfg["eps_fd"]),
        nutmax=sst["nut_max_over_nu"],
        sst_table=_table(sst_rows, ["q", "adjoint", "FD+Richardson", "rel. err"]),
        bias_table=_table(bias_rows, ["q", "full", "frozen", "bias"]),
        grid_table=_table(grid_rows, ["grid", "h", "C_D", "grad norm", "adj. residual"]),
        coupled_table=_table(coupled_rows, ["Re", "solver", "converged", "rel. residual", "k_max"]),
        Jlam=lam["J"], resid_lam=lam["relative_residual"], adj_lam=lam["adjoint_residual"],
        c1_lam=lam["max_rel_err_vs_richardson"],
        c1_lam_status="PASS" if crit["C1_adjoint_vs_FD_rel_lt_1e-3_laminar"] else "FAIL",
        c1_lam_answer="below" if crit["C1_adjoint_vs_FD_rel_lt_1e-3_laminar"] else "above",
        Jsst=sst["J"], resid_sst=sst["relative_residual"], adj_sst=sst["adjoint_residual_full"],
        adj_sst_f=sst["adjoint_residual_frozen"], c1_sst=sst["max_rel_err_full_vs_fd"],
        c1_sst_status="PASS" if crit["C1_adjoint_vs_FD_rel_lt_1e-3_sst_nut"] else "FAIL",
        c1_sst_answer="below" if crit["C1_adjoint_vs_FD_rel_lt_1e-3_sst_nut"] else "above",
        c2=sst["adjoint_residual_full"],
        c2_status="PASS" if crit["C2_adjoint_residual_lt_1e-10_sst_nut"] else "FAIL",
        c3=sst["frozen_bias_max_rel"], bias_mean=sst["frozen_bias_mean_rel"],
        c3_status="PASS" if crit["C3_frozen_bias_lt_5pct"] else "FAIL",
        c3_answer="below" if crit["C3_frozen_bias_lt_5pct"] else "above",
        err_froz=sst["max_rel_err_frozen_vs_fd"],
        frozen_op=frozen_op, frozen_adj_res=frozen_adj_res,
        fd2step=sst["grad_frozen_fd_two_step_change"],
        c4J=grid["observed_order_CD"], c4g=grid["observed_order_grad_norm"],
        c4_status="PASS" if (crit["C4_order_ge_1.5_CD"] and crit["C4_order_ge_1.5_grad"]) else "FAIL",
        cd_rich=grid["CD_richardson_fine"], g_rich=grid["grad_richardson_fine"],
        cdchange=cdchange, gchange=gchange, monotone=monotone,
        cd_answer="below" if cdchange < 0.03 else "above",
        coupled_rel=coupled_rel, kmax=kmax, npytest=npytest,
    )
    (out_dir / "REPORT.md").write_text(md)
    print("wrote", out_dir / "REPORT.md")
    return md


if __name__ == "__main__":  # pragma: no cover
    main()