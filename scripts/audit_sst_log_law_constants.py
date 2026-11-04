"""Audit the SST-2003 constant set against the log-layer calibration identity.

The turbulence constants are never checked against the calibration that fixed
them: the equilibrium logarithmic layer.  A transcription slip in beta, gamma or
sigma_omega moves the model's own log-law slope kappa away from the 0.41 the
wall functions assume, which biases skin friction while every channel/plate
regression test stays green.

Three independent legs, no solver involved:

1. **symbolic** -- substitute the log-layer ansatz into the k-omega equations
   with SymPy and solve the omega residual for kappa^2: a closed form in
   (beta, beta_star, gamma, sigma_omega);
2. **numeric** -- build the log-layer state, feed it to the repository closure
   (``tensorfvm.native_sst3d.sst2003_closure``) and root-find the kappa that
   zeroes the omega budget.  The closure is the object under audit; the root
   finder knows nothing about the closed form;
3. **consequence** -- convert the implied kappa into the log-law offset and the
   skin-friction bias it produces against the wall-function kappa.

Writes ``docs/verification-sst-log-law-constants/`` (audit.json, table.md,
three figures).  Run ``python scripts/audit_sst_log_law_constants.py``.
"""
from __future__ import annotations

import json
import math
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
OUT = os.path.join(ROOT, "docs", "verification-sst-log-law-constants")

import sympy as sp  # noqa: E402

from tensorfvm import native_sst3d as sst  # noqa: E402
from tensorfvm.wall_functions import E as SPALDING_E  # noqa: E402
from tensorfvm.wall_functions import KAPPA as WALL_KAPPA  # noqa: E402

# Families as implemented.  Nothing here re-types a constant: they are read from
# the module, so a silent edit to one of them is what the audit reports.
FAMILIES = {
    "inner_k_omega": dict(beta=sst.BETA1, gamma=sst.GAMMA1, sigma_omega=sst.SIGMA_W1),
    "outer_k_epsilon": dict(beta=sst.BETA2, gamma=sst.GAMMA2, sigma_omega=sst.SIGMA_W2),
}
BETA_STAR = sst.BETA_STAR
A1 = sst.A1
Y_PLUS_MIN = 300.0
NU = 1.0 / Y_PLUS_MIN          # y = 1 <-> y+ = 300 at u_tau = 1


# --------------------------------------------------------------------------- #
# leg 1 -- symbolic identity
# --------------------------------------------------------------------------- #
def symbolic_identity(beta_star=BETA_STAR):
    """Closed form for kappa^2, re-derived from the equilibrium budget."""
    beta, gamma, sigma, kappa, y = sp.symbols(
        "beta gamma sigma_omega kappa y", positive=True)
    u = sp.Symbol("u_tau", positive=True)
    k = u ** 2 / sp.sqrt(beta_star)
    omega = u / (sp.sqrt(beta_star) * kappa * y)
    nu_t = kappa * u * y
    production = nu_t * (u / (kappa * y)) ** 2          # P/rho, per unit mass
    k_residual = sp.simplify(production - beta_star * k * omega)
    diffusion = sp.diff((sigma * nu_t) * sp.diff(omega, y), y)
    omega_residual = sp.simplify(gamma * production / nu_t - beta * omega ** 2 + diffusion)
    # The residual carries kappa only through kappa**2, so solve for the square
    # (with kappa symbolic SymPy cannot decide which root is the physical one).
    q = sp.Symbol("q", positive=True)
    squared_residual = sp.simplify(omega_residual.subs(kappa, sp.sqrt(q)))
    roots = sp.solve(sp.Eq(squared_residual, 0), q)
    if len(roots) != 1:
        raise RuntimeError(f"expected one root in kappa**2, got {roots}")
    kappa_squared = sp.simplify(roots[0])
    identity = (beta / sp.sqrt(beta_star) - gamma * sp.sqrt(beta_star)) / sigma
    return dict(
        k_residual=str(k_residual),
        k_budget_closes_identically=bool(k_residual == 0),
        omega_residual=str(sp.simplify(omega_residual)),
        kappa_squared=str(kappa_squared),
        identity_closes_symbolically=bool(sp.simplify(kappa_squared - identity) == 0),
        kappa_squared_latex=sp.latex(kappa_squared),
    )


def closed_form_kappa(beta, gamma, sigma_omega, beta_star=BETA_STAR):
    value = (beta / math.sqrt(beta_star) - gamma * math.sqrt(beta_star)) / sigma_omega
    return math.sqrt(value), value


# --------------------------------------------------------------------------- #
# leg 2 -- numeric root of the omega budget through the repository closure
# --------------------------------------------------------------------------- #
def log_layer_state(kappa, u_tau=1.0, y_plus_min=Y_PLUS_MIN, points=161, nu=NU):
    y_plus = np.geomspace(y_plus_min, y_plus_min * 30.0, points)
    y = y_plus * nu / u_tau
    k = np.full_like(y, u_tau ** 2 / math.sqrt(BETA_STAR))
    omega = u_tau / (math.sqrt(BETA_STAR) * kappa * y)
    gradient = np.zeros(y.shape + (3, 3))
    gradient[..., 0, 1] = u_tau / (kappa * y)          # du_x/dy only
    grad_k = np.zeros(y.shape + (3,))                  # k uniform along y
    grad_omega = np.zeros(y.shape + (3,))
    grad_omega[..., 1] = -omega / y                    # omega ~ 1/y
    return dict(y=y, y_plus=y_plus, nu=nu, k=k, omega=omega,
                G=gradient, gk=grad_k, gw=grad_omega)


def closure_on(state):
    return sst.sst2003_closure(state["k"], state["omega"], state["y"], state["nu"],
                               state["G"], state["gk"], state["gw"])


def omega_budget(kappa, state=None):
    """Per-mass omega budget of the repository closure on the log-layer state."""
    state = state if state is not None else log_layer_state(kappa)
    c = closure_on(state)
    domega_dy = -state["omega"] / state["y"]
    turbulent = np.gradient(c["sigma_omega"] * c["nu_t"] * domega_dy, state["y"],
                            edge_order=2)
    molecular = np.gradient(state["nu"] * domega_dy, state["y"], edge_order=2)
    residual = (c["omega_production"] - c["omega_destruction"]
                + c["cross_diffusion_source"] + turbulent)
    scale = np.abs(c["omega_production"]) + np.abs(c["omega_destruction"])
    # The identity is the nu -> 0 statement; the molecular part is a 1/y+
    # correction that belongs to the finite-Re state, not to the constant set.
    return dict(residual=residual, scale=scale, closure=c,
                mean_residual=float(np.mean(residual)),
                molecular_relative=float(np.max(np.abs(molecular) / scale)),
                relative=float(np.max(np.abs(residual) / scale)))


def numeric_root(kappa_lo=0.30, kappa_hi=0.55, iterations=200, points=161):
    """Bisection on the self-consistent kappa that zeroes the omega budget."""

    def signed(kappa):
        state = log_layer_state(kappa, points=points)
        return float(np.mean(omega_budget(kappa, state)["residual"]))

    lo, hi = kappa_lo, kappa_hi
    f_lo, f_hi = signed(lo), signed(hi)
    if f_lo * f_hi > 0:
        raise RuntimeError(f"no bracket in [{lo}, {hi}]: {f_lo:.3e}, {f_hi:.3e}")
    for _ in range(iterations):
        mid = 0.5 * (lo + hi)
        f_mid = signed(mid)
        if f_lo * f_mid <= 0:
            hi, f_hi = mid, f_mid
        else:
            lo, f_lo = mid, f_mid
        if hi - lo < 1e-15 * hi:
            break
    return 0.5 * (lo + hi)


# --------------------------------------------------------------------------- #
# leg 3 -- consequences for the wall law and skin friction
# --------------------------------------------------------------------------- #
def wall_law_consequence(kappa_model, kappa_wall=WALL_KAPPA, spalding_e=SPALDING_E):
    """Log-law offset and Cf bias between the model's kappa and the wall's."""
    B_wall = math.log(spalding_e) / kappa_wall
    rows = []
    for y_plus in (30.0, 100.0, 300.0, 1000.0):
        logy = math.log(y_plus)
        u_plus_model = logy / kappa_model + B_wall
        u_plus_wall = logy / kappa_wall + B_wall
        delta = u_plus_model - u_plus_wall
        rows.append(dict(y_plus=y_plus, u_plus_model=u_plus_model,
                         u_plus_wall=u_plus_wall, delta_u_plus=delta,
                         delta_u_plus_percent=100.0 * delta / u_plus_wall,
                         cf_bias_percent=200.0 * delta / u_plus_wall))
    return rows


def limiter_margin(a1=A1, beta_star=BETA_STAR):
    """Branch margin of nu_t = a1 k / max(a1 omega, Omega F2) in the log layer.

    The log layer needs a1 omega >= Omega F2, i.e. a1 >= sqrt(beta_star); below
    that threshold nu_t is biased low by exactly the ratio a1/sqrt(beta_star).
    """
    threshold = math.sqrt(beta_star)
    return dict(a1=a1, sqrt_beta_star=threshold, ratio=a1 / threshold,
                margin_percent=100.0 * (a1 / threshold - 1.0))


def f1_in_log_layer():
    """F1 along the log layer: which constant family actually governs it."""
    state = log_layer_state(WALL_KAPPA)
    c = closure_on(state)
    return dict(y_plus=state["y_plus"].tolist(), F1=c["F1"].tolist(),
                F1_min=float(np.min(c["F1"])), F1_max=float(np.max(c["F1"])))


def nu_t_matches_kappa_u_tau_y():
    """The closure's eddy viscosity must equal kappa u_tau y in the log layer."""
    state = log_layer_state(WALL_KAPPA)
    c = closure_on(state)
    expected = WALL_KAPPA * state["y"]                   # u_tau = 1
    return float(np.max(np.abs(c["nu_t"] / expected - 1.0)))


# --------------------------------------------------------------------------- #
# negative controls -- an audit that cannot fail proves nothing
# --------------------------------------------------------------------------- #
def sensitivity_controls(perturbation=0.01):
    base = dict(beta=sst.BETA1, gamma=sst.GAMMA1, sigma_omega=sst.SIGMA_W1,
                beta_star=BETA_STAR)
    kappa0, _ = closed_form_kappa(**base)
    cases = {
        "sigma_omega1 +1%": dict(sigma_omega=base["sigma_omega"] * (1 + perturbation)),
        "sigma_omega1 -1%": dict(sigma_omega=base["sigma_omega"] * (1 - perturbation)),
        "beta1 +1%": dict(beta=base["beta"] * (1 + perturbation)),
        "gamma1 +1%": dict(gamma=base["gamma"] * (1 + perturbation)),
        "beta_star +1%": dict(beta_star=base["beta_star"] * (1 + perturbation)),
    }
    controls = []
    for name, override in cases.items():
        args = dict(base)
        args.update(override)
        kappa, _ = closed_form_kappa(**args)
        controls.append(dict(perturbation=name, kappa=kappa,
                             relative_change_percent=100.0 * (kappa - kappa0) / kappa0))
    # a kappa that violates the budget must not be silently accepted
    probe = omega_budget(WALL_KAPPA)["relative"]
    return controls, probe


# --------------------------------------------------------------------------- #
# figures
# --------------------------------------------------------------------------- #
def make_figures(kappa_inner, kappa_outer, scan, consequence):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(OUT, exist_ok=True)
    paths = []

    kappas = np.linspace(0.395, 0.425, 61)
    residual = [omega_budget(k)["relative"] for k in kappas]
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.semilogy(kappas, np.maximum(residual, 1e-18), "o-", ms=3,
                label="closure omega budget (repo SST-2003)")
    ax.axvline(kappa_inner, color="C1", ls="--",
               label=rf"inner family $\kappa$={kappa_inner:.5f}")
    ax.axvline(kappa_outer, color="C2", ls=":",
               label=rf"outer family $\kappa$={kappa_outer:.5f}")
    ax.axvline(WALL_KAPPA, color="k", lw=0.8,
               label=rf"wall functions $\kappa$={WALL_KAPPA}")
    ax.set_xlabel(r"assumed $\kappa$ in the log-layer state")
    ax.set_ylabel("max |omega budget residual| / budget")
    ax.set_title("The log layer is only self-consistent at one slope")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=7, loc="upper center")
    path = os.path.join(OUT, "fig1_omega_budget_vs_kappa.png")
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)
    paths.append(path)

    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.plot(scan["sigma_omega"], scan["kappa"], "o-", ms=3)
    ax.axhline(WALL_KAPPA, color="k", lw=0.8, label=rf"$\kappa$={WALL_KAPPA} (wall law)")
    ax.axvline(sst.SIGMA_W1, color="C1", ls="--",
               label=rf"implemented $\sigma_{{\omega 1}}$={sst.SIGMA_W1:g}")
    ax.axvline(scan["sigma_for_exact_kappa"], color="C3", ls=":",
               label=rf"$\sigma$ for exact: {scan['sigma_for_exact_kappa']:.5f}")
    ax.set_xlabel(r"$\sigma_{\omega 1}$ (beta1, gamma1 fixed)")
    ax.set_ylabel(r"implied log-law slope $\kappa$")
    ax.set_title("Which sigma_omega would make kappa exactly 0.41")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7)
    path = os.path.join(OUT, "fig2_sigma_to_kappa.png")
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)
    paths.append(path)

    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    yp = [row["y_plus"] for row in consequence]
    ax.semilogx(yp, [row["delta_u_plus_percent"] for row in consequence], "o-",
                label=r"$\Delta u^+/u^+$ [%]")
    ax.semilogx(yp, [row["cf_bias_percent"] for row in consequence], "s--",
                label=r"implied $c_f$ bias [%]")
    ax.axhspan(-3, 3, color="0.8", alpha=0.4, label=r"$\pm3\%$ acceptance band")
    ax.set_xlabel(r"$y^+$")
    ax.set_ylabel("percent")
    ax.set_title("Consequence of the model-vs-wall kappa mismatch")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=7)
    path = os.path.join(OUT, "fig3_wall_law_consequence.png")
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)
    paths.append(path)
    return [os.path.relpath(p, ROOT) for p in paths]


# --------------------------------------------------------------------------- #
# driver
# --------------------------------------------------------------------------- #
def run():
    symbolic = symbolic_identity()
    kappa_inner, kappa_sq_inner = closed_form_kappa(**FAMILIES["inner_k_omega"])
    kappa_outer, kappa_sq_outer = closed_form_kappa(**FAMILIES["outer_k_epsilon"])
    # Second-order differences on a geometric grid: report the root as a
    # refinement ladder so the gap to the closed form is shown to be truncation.
    ladder = []
    for samples in (81, 161, 321, 641):
        r = numeric_root(points=samples)
        ladder.append(dict(points=samples, root_kappa=r, gap_to_closed_form=r - kappa_inner))
    root = ladder[-1]["root_kappa"]
    budget_root = omega_budget(root)
    res_root = budget_root["relative"]
    res_wall = omega_budget(WALL_KAPPA)["relative"]
    res_inner = omega_budget(kappa_inner)["relative"]

    sigma_for_exact = (sst.BETA1 / math.sqrt(BETA_STAR)
                       - sst.GAMMA1 * math.sqrt(BETA_STAR)) / WALL_KAPPA ** 2
    scan_sigma = np.linspace(0.40, 0.60, 41)
    scan = dict(sigma_omega=scan_sigma.tolist(),
                kappa=[closed_form_kappa(sst.BETA1, sst.GAMMA1, s)[0] for s in scan_sigma],
                sigma_for_exact_kappa=float(sigma_for_exact))

    consequence = wall_law_consequence(kappa_inner)
    controls, probe = sensitivity_controls()
    limiter = limiter_margin()
    f1 = f1_in_log_layer()
    nut_error = nu_t_matches_kappa_u_tau_y()

    spread = 100.0 * (kappa_outer - kappa_inner) / WALL_KAPPA
    result = dict(
        schema="tensorfvm.sst-log-law-constants/1",
        constants=dict(beta_star=BETA_STAR, beta1=sst.BETA1, beta2=sst.BETA2,
                       gamma1=sst.GAMMA1, gamma2=sst.GAMMA2,
                       sigma_omega1=sst.SIGMA_W1, sigma_omega2=sst.SIGMA_W2,
                       a1=A1, wall_kappa=WALL_KAPPA, spalding_e=SPALDING_E,
                       spalding_B=math.log(SPALDING_E) / WALL_KAPPA),
        symbolic=symbolic,
        families={name: dict(**FAMILIES[name], kappa=kappa,
                             kappa_squared=kappa_sq,
                             deviation_from_wall_kappa_percent=100.0 * (kappa - WALL_KAPPA) / WALL_KAPPA)
                  for name, kappa, kappa_sq in (
                      ("inner_k_omega", kappa_inner, kappa_sq_inner),
                      ("outer_k_epsilon", kappa_outer, kappa_sq_outer))},
        numeric=dict(root_kappa=root,
                     closed_form_kappa=kappa_inner,
                     closed_form_minus_root=kappa_inner - root,
                     convergence_ladder=ladder,
                     relative_residual_at_root=res_root,
                     relative_residual_at_closed_form=res_inner,
                     relative_residual_at_wall_kappa=res_wall,
                     molecular_share_at_root=budget_root["molecular_relative"],
                     nu_t_matches_kappa_u_tau_y_max_rel_error=nut_error),
        log_layer=dict(f1_min=f1["F1_min"], f1_max=f1["F1_max"],
                       governing_family="inner_k_omega" if f1["F1_min"] > 0.99 else "blended"),
        family_spread_percent=spread,
        sigma_omega_for_exact_wall_kappa=float(sigma_for_exact),
        limiter=limiter,
        wall_law_consequence=consequence,
        negative_controls=controls,
        residual_at_wall_kappa_if_forced=probe,
        gates=dict(
            symbolic_identity_closes=symbolic["identity_closes_symbolically"]
            and symbolic["k_budget_closes_identically"],
            numeric_root_matches_closed_form=abs(kappa_inner - root) < 1e-3,
            numeric_gap_shrinks_under_refinement=abs(ladder[-1]["gap_to_closed_form"])
            < abs(ladder[0]["gap_to_closed_form"]),
            constants_consistent_within_0p5_percent=abs(100.0 * (kappa_inner - WALL_KAPPA) / WALL_KAPPA) < 0.5,
            family_spread_within_0p5_percent=spread < 0.5,
            eddy_viscosity_recovers_kappa_u_tau_y=nut_error < 1e-12,
            audit_detects_1pct_constant_edit=all(
                abs(c["relative_change_percent"]) > 0.1 for c in controls),
            cf_bias_below_acceptance_band=max(abs(r["cf_bias_percent"]) for r in consequence) < 3.0,
            limiter_has_positive_margin=limiter["margin_percent"] > 0,
        ),
    )
    result["figures"] = make_figures(kappa_inner, kappa_outer, scan, consequence)

    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "audit.json"), "w") as fh:
        json.dump(result, fh, indent=1)

    lines = [
        "# SST-2003 constant algebra: what log-law slope do the constants imply?",
        "",
        "`k` budget of the log-layer ansatz closes identically:",
        f"  {symbolic['k_residual']}",
        "",
        "omega budget solved for $\\kappa^2$:",
        f"  $\\kappa^2 = {symbolic['kappa_squared_latex']}$",
        "",
        "| constant family | beta | gamma | sigma_omega | implied kappa | vs 0.41 |",
        "|---|---|---|---|---|---|",
    ]
    for name in ("inner_k_omega", "outer_k_epsilon"):
        row = result["families"][name]
        lines.append("| %s | %.4f | %.5f | %.3f | %.5f | %+.3f%% |" % (
            name, row["beta"], row["gamma"], row["sigma_omega"], row["kappa"],
            row["deviation_from_wall_kappa_percent"]))
    lines += [
        "",
        "Numeric root through `sst2003_closure` (no closed form used): %.12f" % root,
        "|closed form - root|: %.2e" % (kappa_inner - root),
        "closure eddy viscosity vs kappa*u_tau*y, max rel error: %.2e" % nut_error,
        "the gap is finite-difference truncation: it shrinks with sampling, see ladder",
        "",
        "| samples | numeric root | gap to closed form |",
        "|---|---|---|",
    ] + ["| %d | %.9f | %+.2e |" % (row["points"], row["root_kappa"],
                                    row["gap_to_closed_form"]) for row in ladder] + [
        "",
        "F1 across the log layer: [%.3e, %.3e]" % (f1["F1_min"], f1["F1_max"]),
        "",
        "| y+ | u+ (model kappa) | u+ (wall kappa) | du+ | du+/u+ [%] | cf bias [%] |",
        "|---|---|---|---|---|---|",
    ]
    for row in consequence:
        lines.append("| %g | %.4f | %.4f | %+.5f | %+.4f | %+.4f |" % (
            row["y_plus"], row["u_plus_model"], row["u_plus_wall"], row["delta_u_plus"],
            row["delta_u_plus_percent"], row["cf_bias_percent"]))
    lines += ["", "Negative controls (1% constant edits):", "",
              "| edit | implied kappa | change [%] |", "|---|---|---|"]
    for c in controls:
        lines.append("| %s | %.6f | %+.4f |" % (c["perturbation"], c["kappa"],
                                                c["relative_change_percent"]))
    with open(os.path.join(OUT, "table.md"), "w") as fh:
        fh.write("\n".join(lines) + "\n")
    return result


if __name__ == "__main__":
    data = run()
    print(json.dumps({k: data[k] for k in ("gates", "numeric", "family_spread_percent",
                                           "sigma_omega_for_exact_wall_kappa",
                                           "limiter", "figures")}, indent=1))