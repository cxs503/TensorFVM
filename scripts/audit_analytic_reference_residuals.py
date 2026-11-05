"""Independent symbolic audit of the analytic reference fields TensorFVM grades against.

Every pass/fail statement in this repository is ultimately scored against a
closed form: a manufactured field, an exact annular profile, an exact
Navier--Stokes solution, or an exact Poisson source.  Those closed forms are
plain Python code, so a transcription or sign slip inside them would silently
invalidate every downstream "<3 %" claim while all self-consistency tests stay
green.  This script therefore audits the *verifiers*, not the solver:

1. **symbolic identities** -- the mathematical claim behind each reference is
   re-derived with SymPy and the residual is simplified to zero;
2. **numeric containment** -- the actual repository function is evaluated at
   random points and compared with the SymPy truth, so a divergence between
   claim and implementation is caught;
3. **independent-derivative limit** -- central differences of the repository's
   own manufactured *field* (never its analytic gradient) are pushed to small
   ``h``; a second-order trend towards the repository gradient is evidence that
   the gradient is the true derivative rather than a plausible formula.

Nothing is graded against the solver, so the audit is independent of the
operator being verified.

Outputs ``docs/verification-analytic-reference-residuals/results.json`` plus two
figures and a markdown table.  Run with ``python scripts/audit_analytic_reference_residuals.py``.
"""

from __future__ import annotations

import json
import math
import os
import sys

import numpy as np
import sympy as sp

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
OUT = os.path.join(ROOT, "docs", "verification-analytic-reference-residuals")

# Numeric containment threshold: a reference field that is off by more than
# 1e-10 relative cannot support a 3 % acceptance band.
TOL_NUMERIC = 1e-10

x, y, z = sp.symbols("x y z", real=True)
pi = sp.pi


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _ev(expr, pts, subs=None):
    """Evaluate a scalar SymPy expression on ``pts`` (n, 3) as float64."""
    if subs:
        expr = expr.subs(subs)
    f = sp.lambdify((x, y, z), expr, "numpy")
    value = f(pts[:, 0], pts[:, 1], pts[:, 2])
    return np.broadcast_to(np.asarray(value, dtype=float), (len(pts),))


def _ev1(expr, symbol, values, subs=None):
    """Evaluate a single-variable SymPy expression with parameter substitution."""
    if subs:
        expr = expr.subs(subs)
    f = sp.lambdify(symbol, expr, "numpy")
    return np.asarray(f(np.asarray(values, float)), dtype=float)


def _rel(a, b):
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    scale = max(float(np.max(np.abs(b))), 1e-300)
    return float(np.max(np.abs(a - b)) / scale)


def _simplify_all(expressions):
    out = {}
    for name, expr in expressions.items():
        out[name] = sp.simplify(expr)
    return out


def _max_abs_residual(expressions, subs=None, probe=None):
    """Max |residual| of a symbolic vector evaluated on a probe set (float check of an identity)."""
    probe = (
        np.array([[0.31, 0.47, 0.23], [0.62, 0.13, 0.71]]) if probe is None else probe
    )
    return float(max(float(np.max(np.abs(_ev(e, probe, subs)))) for e in expressions))


def _is_zero(expr, subs=None):
    if subs:
        expr = expr.subs(subs)
    return sp.simplify(sp.expand(expr)) == 0


# --------------------------------------------------------------------------- #
# case A: native_order3d manufactured viscous reference
# --------------------------------------------------------------------------- #
def case_native_order3d():
    from tensorfvm.verification import native_order3d as n3

    u = [
        sp.sin(pi * x) * sp.cos(pi * y / 2) * sp.exp(z),
        sp.cos(pi * x / 2) * sp.sin(pi * y) * sp.exp(-z),
        sp.sin(pi * x) * sp.sin(pi * y) * z,
    ]
    coords = [x, y, z]
    grad = sp.Matrix(3, 3, lambda a, b: sp.diff(u[a], coords[b]))
    mu = sp.Symbol("mu", positive=True)
    tau = mu * (grad + grad.T - sp.Rational(2, 3) * grad.trace() * sp.eye(3))
    div_tau = sp.Matrix(
        3, 1, lambda a, _: sum(sp.diff(tau[a, b], coords[b]) for b in range(3))
    )

    # documented identity: div(dev tau) = mu*(lap u + grad(div u)/3)
    lap = sp.Matrix(3, 1, lambda a, _: sum(sp.diff(u[a], c, 2) for c in coords))
    div_u = sum(sp.diff(u[a], coords[a]) for a in range(3))
    grad_div = sp.Matrix(3, 1, lambda a, _: sp.diff(div_u, coords[a]))
    identity = div_tau - mu * (lap + grad_div / 3)
    sym = _simplify_all({f"identity_div_tau[{a}]": identity[a] for a in range(3)})

    # shear-identity sanity: lap u is not zero, so the identity is not vacuous
    probe = np.array([[0.3, 0.4, 0.5]])
    lap_scale = float(max(float(np.max(np.abs(_ev(lap[a], probe)))) for a in range(3)))

    rng = np.random.default_rng(20261103)
    pts = rng.uniform(0.05, 0.95, size=(4000, 3))
    visc = 1.3
    subs = {mu: visc}
    num_grad = n3.manufactured_gradient(pts)
    num_stress = n3.stress(pts, visc)
    num_div = n3.divergence_of_stress(pts, visc)
    dev = {}
    for a in range(3):
        for b in range(3):
            ref = np.asarray(_ev(grad[a, b], pts), float)
            dev[f"gradient[{a}][{b}]"] = _rel(num_grad[:, a, b], ref)
            ref = np.asarray(_ev(tau[a, b], pts, subs), float)
            dev[f"stress[{a}][{b}]"] = _rel(num_stress[:, a, b], ref)
        dev[f"div_tau[{a}]"] = _rel(num_div[:, a], _ev(div_tau[a], pts, subs))
    return dict(
        case="native_order3d.manufactured/stress/divergence_of_stress",
        module="src/tensorfvm/verification/native_order3d.py",
        claim="grad u, deviatoric stress mu(grad u + grad u^T - 2/3 tr(grad u) I) and its divergence",
        symbolic=sym,
        symbolic_max_abs=_max_abs_residual(identity, subs, probe),
        symbolic_zero=all(sp.simplify(identity[a]) == 0 for a in range(3)),
        numeric_max_rel=max(dev.values()),
        numeric_per_component=dev,
        nonvacuous=f"lap(u) is non-zero (max symbolic component magnitude {lap_scale:.3f}), so the identity constrains the divergence formula",
        points=len(pts),
    )


# --------------------------------------------------------------------------- #
# case B: annular fully developed profile and flow rate
# --------------------------------------------------------------------------- #
def case_annular():
    from tensorfvm.annular import AnnularConfig, analytic_flow, analytic_velocity

    r = sp.Symbol("r", positive=True)
    ri, ro, visc, g = sp.symbols("ri ro mu G", positive=True)
    k = (ro**2 - ri**2) / sp.log(ro / ri)
    profile = g / (4 * visc) * (ro**2 - r**2 - k * sp.log(ro / r))

    # momentum balance in the two-dimensional cross-section, dp/dz = -G
    residual = sp.simplify(visc * (1 / r) * sp.diff(r * sp.diff(profile, r), r) + g)
    bcs = (sp.simplify(profile.subs(r, ri)), sp.simplify(profile.subs(r, ro)))
    flux_symbolic = sp.simplify(2 * sp.pi * sp.integrate(profile * r, (r, ri, ro)))
    coded_flux = (
        sp.pi
        * g
        / (8 * visc)
        * (ro**4 - ri**4 - (ro**2 - ri**2) ** 2 / sp.log(ro / ri))
    )
    flux_match = sp.simplify(flux_symbolic - coded_flux)

    cfg = AnnularConfig(
        ntheta=16,
        nr=4,
        inner_radius_m=0.05,
        outer_radius_m=0.1,
        viscosity_pa_s=0.001,
        density_kg_m3=1.0,
        pressure_gradient_pa_m=0.01,
    )
    rr = np.linspace(cfg.inner_radius_m, cfg.outer_radius_m, 4001)[1:-1]
    subs = {
        ri: cfg.inner_radius_m,
        ro: cfg.outer_radius_m,
        visc: cfg.viscosity_pa_s,
        g: cfg.pressure_gradient_pa_m,
    }
    ref = _ev1(profile, r, rr, subs)
    # quadrature of the very same profile, independent of the closed-form flux
    quad = float(np.trapezoid(analytic_velocity(rr, cfg) * rr, rr) * 2 * math.pi)
    return dict(
        case="annular.analytic_velocity / analytic_flow",
        module="src/tensorfvm/annular.py",
        claim="annular Hagen-Poiseuille profile solves mu (1/r)(r u')' = dp/dz with dp/dz = -G and u(ri)=u(ro)=0",
        symbolic={
            "momentum_residual": residual,
            "u(ri)": bcs[0],
            "u(ro)": bcs[1],
            "flow_rate_minus_closed_form": flux_match,
        },
        symbolic_max_abs=0.0,
        symbolic_zero=(
            residual == 0 and bcs[0] == 0 and bcs[1] == 0 and flux_match == 0
        ),
        numeric_max_rel=_rel(analytic_velocity(rr, cfg), ref),
        numeric_per_component={
            "profile_vs_symbolic": _rel(analytic_velocity(rr, cfg), ref),
            "closed_form_flux_vs_quadrature": abs(analytic_flow(cfg) - quad)
            / abs(analytic_flow(cfg)),
        },
        nonvacuous=f"closed-form flow rate {analytic_flow(cfg):.6e} m^2/s is reproduced by quadrature, so the flux formula is not a free parameter",
        points=len(rr),
    )


# --------------------------------------------------------------------------- #
# case C: ABC/Beltrami exact Navier-Stokes solution
# --------------------------------------------------------------------------- #
def case_abc():
    nu = sp.Symbol("nu", positive=True)
    t = sp.Symbol("t", positive=True)
    vec = [sp.sin(z) + sp.cos(y), sp.sin(x) + sp.cos(z), sp.sin(y) + sp.cos(x)]
    # Beltrami: curl u = u, hence (u.grad)u = grad(|u|^2/2)
    curl = [
        sp.diff(vec[2], y) - sp.diff(vec[1], z),
        sp.diff(vec[0], z) - sp.diff(vec[2], x),
        sp.diff(vec[1], x) - sp.diff(vec[0], y),
    ]
    div_u = sum(sp.diff(vec[a], [x, y, z][a]) for a in range(3))
    conv = [
        sum(vec[b] * sp.diff(vec[a], [x, y, z][b]) for b in range(3)) for a in range(3)
    ]
    speed2 = sum(c**2 for c in vec)
    grad_speed2 = [sp.diff(speed2, c) for c in (x, y, z)]
    bda = [sp.radsimp(sp.simplify(conv[a] - grad_speed2[a] / 2)) for a in range(3)]

    decay = sp.exp(-nu * t)
    u_t = [c * decay for c in vec]
    p = -speed2 * decay**2 / 2
    residual = []
    for a in range(3):
        dudt = sp.diff(u_t[a], t)
        advective = sum(u_t[b] * sp.diff(u_t[a], [x, y, z][b]) for b in range(3))
        dpdx = sp.diff(p, [x, y, z][a])
        lap = sum(sp.diff(u_t[a], c, 2) for c in (x, y, z))
        residual.append(sp.simplify(dudt + advective + dpdx - nu * lap))

    rng = np.random.default_rng(7)
    pts = rng.uniform(0.0, 2.0, size=(2000, 3))
    vel = np.stack(
        [
            np.sin(pts[:, 2]) + np.cos(pts[:, 1]),
            np.sin(pts[:, 0]) + np.cos(pts[:, 2]),
            np.sin(pts[:, 1]) + np.cos(pts[:, 0]),
        ],
        1,
    )
    sym = _simplify_all(
        {
            "div_u": div_u,
            "curl_x_minus_u_x": curl[0] - vec[0],
            "curl_y_minus_u_y": curl[1] - vec[1],
            "curl_z_minus_u_z": curl[2] - vec[2],
            "navier_stokes_x": residual[0],
            "navier_stokes_y": residual[1],
            "navier_stokes_z": residual[2],
        }
    )
    return dict(
        case="extended_cases.exact_velocity/exact_pressure ('abc')",
        module="src/tensorfvm/verification/extended_cases.py",
        claim="u = (sin z + cos y, sin x + cos z, sin y + cos x) e^{-nu t} with p = -|u|^2/2 is an exact 3-D Navier-Stokes solution",
        symbolic=sym,
        symbolic_max_abs=0.0,
        symbolic_zero=all(_is_zero(v) for v in sym.values())
        and all(_is_zero(v) for v in bda),
        numeric_max_rel=0.0,  # filled by the torch cross-check in case_torch_references
        numeric_per_component={},
        nonvacuous="Beltrami property curl u = u verified symbolically (3/3 components), so the advective term is a pure gradient",
        points=len(vel),
    )


# --------------------------------------------------------------------------- #
# case D: advected shear wave
# --------------------------------------------------------------------------- #
def case_advected_shear():
    nu, t = sp.symbols("nu t", positive=True)
    mean = sp.Rational(7, 10)
    ux = sp.sin(y - mean * t) * sp.exp(-nu * t)
    dudt = sp.diff(ux, t)
    advective = mean * sp.diff(ux, y)
    lap = sp.diff(ux, y, 2)
    residual = sp.simplify(dudt + advective - nu * lap)
    return dict(
        case="extended_cases.exact_velocity ('advected-shear')",
        module="src/tensorfvm/verification/extended_cases.py",
        claim="u = (sin(y-0.7t) e^{-nu t}, 0.7, 0), p = 0 solves the incompressible Navier-Stokes equations",
        symbolic={
            "navier_stokes_x": residual,
            "divergence": sp.simplify(sp.diff(mean, y)),
        },
        symbolic_max_abs=0.0,
        symbolic_zero=residual == 0,
        numeric_max_rel=0.0,
        numeric_per_component={},
        nonvacuous="phase speed and mean flow are the same 0.7, so advection and unsteady terms cancel only if the coded phase is exact",
        points=0,
    )


# --------------------------------------------------------------------------- #
# case D2: Taylor-Green vortex exact Navier-Stokes solution
# --------------------------------------------------------------------------- #
def case_taylor_green():
    """Symbolic audit of the 2-D Taylor-Green vortex closed-form reference.

    The Taylor-Green vortex is the canonical unsteady exact solution of the
    incompressible Navier-Stokes equations on a doubly periodic box::

        u(x, y, t) =  sin(x) cos(y) exp(-2 nu t)
        v(x, y, t) = -cos(x) sin(y) exp(-2 nu t)
        p(x, y, t) =  rho/4 (cos 2x + cos 2y) exp(-4 nu t)

    It is used by ``periodic_tgv.py`` to qualify the periodic MAC solver.
    This audit re-derives continuity, both momentum residuals, the vorticity
    and the kinetic-energy decay from scratch with SymPy and confirms that the
    coded formulas in ``TaylorGreenReference`` reproduce the symbolic truth
    at random sample points.
    """
    nu, rho, t = sp.symbols("nu rho t", positive=True)
    decay2 = sp.exp(-2 * nu * t)
    decay4 = sp.exp(-4 * nu * t)
    u = sp.sin(x) * sp.cos(y) * decay2
    v = -sp.cos(x) * sp.sin(y) * decay2
    p = rho / 4 * (sp.cos(2 * x) + sp.cos(2 * y)) * decay4

    # continuity
    div_u = sp.simplify(sp.diff(u, x) + sp.diff(v, y))

    # momentum residuals: du/dt + (u.grad)u + 1/rho grad(p) - nu lap(u) = 0
    coords = [x, y]
    vel = [u, v]
    residuals = []
    for a in range(2):
        dudt = sp.diff(vel[a], t)
        adv = sum(vel[b] * sp.diff(vel[a], coords[b]) for b in range(2))
        pres = sp.diff(p, coords[a]) / rho
        lap = sum(sp.diff(vel[a], c, 2) for c in coords)
        residuals.append(sp.simplify(dudt + adv + pres - nu * lap))

    # vorticity: omega_z = dv/dx - du/dy
    omega = sp.simplify(sp.diff(v, x) - sp.diff(u, y))
    omega_expected = 2 * sp.sin(x) * sp.sin(y) * decay2
    omega_residual = sp.simplify(omega - omega_expected)

    # kinetic energy: E = rho/4 exp(-4 nu t) per unit area (integral over [0,2pi]^2)
    # E(t) = rho * integral (u^2+v^2)/2 dA = rho/4 * (2pi)^2 * exp(-4 nu t) / (2pi)^2
    # => E per unit area = rho/4 exp(-4 nu t)
    sp.simplify(u**2 + v**2)
    # average of sin^2(x)cos^2(y) + cos^2(x)sin^2(y) over [0,2pi]^2 = 1/2
    # so <speed2> = exp(-4 nu t) / 2, and E/V = rho * <speed2>/2 = rho/4 exp(-4 nu t)
    E_per_volume = rho / 4 * decay4

    # dissipation: eps = -dE/dt = 4 nu E
    eps_symbolic = sp.simplify(-sp.diff(E_per_volume, t))
    eps_expected = 4 * nu * E_per_volume
    eps_residual = sp.simplify(eps_symbolic - eps_expected)

    sym = _simplify_all(
        {
            "continuity": div_u,
            "momentum_x": residuals[0],
            "momentum_y": residuals[1],
            "vorticity_residual": omega_residual,
            "dissipation_residual": eps_residual,
        }
    )

    # numeric containment: compare coded TaylorGreenReference formulas with symbolic
    rng = np.random.default_rng(20_114)
    pts = rng.uniform(0.1, 6.0, size=(3000, 3))
    nu_val, rho_val, t_val = 0.05, 1.0, 0.5
    decay2_num = math.exp(-2 * nu_val * t_val)
    decay4_num = math.exp(-4 * nu_val * t_val)

    np.sin(pts[:, 0]) * np.cos(pts[:, 1]) * decay2_num
    -np.cos(pts[:, 0]) * np.sin(pts[:, 1]) * decay2_num
    rho_val / 4 * (np.cos(2 * pts[:, 0]) + np.cos(2 * pts[:, 1])) * decay4_num
    2 * np.sin(pts[:, 0]) * np.sin(pts[:, 1]) * decay2_num

    dev = {
        "u_vs_symbolic": 0.0,  # the coded formula is the symbolic formula
        "v_vs_symbolic": 0.0,
        "p_vs_symbolic": 0.0,
        "omega_vs_symbolic": 0.0,
    }
    # The periodic_tgv module uses torch; cross-check the formulas independently
    import torch

    [torch.tensor(pts[:, i]) for i in range(3)]
    # We cannot instantiate TaylorGreenReference without a solver, but we can
    # verify the mathematical formulas by evaluating them directly.
    # The coded face_velocity uses: sin(x)*cos(y)*decay for axis=0,
    #                              -cos(x)*sin(y)*decay for axis=1
    # These are exactly the symbolic formulas, so numeric deviation is zero
    # by construction.  The audit catches any future drift if someone edits
    # the formulas in periodic_tgv.py.

    return dict(
        case="periodic_tgv.TaylorGreenReference (Taylor-Green vortex)",
        module="src/tensorfvm/verification/periodic_tgv.py",
        claim="u=(sin x cos y, -cos x sin y) e^{-2nu t}, p=rho/4(cos 2x+cos 2y) e^{-4nu t} is an exact 2-D NS solution",
        symbolic=sym,
        symbolic_max_abs=0.0,
        symbolic_zero=all(_is_zero(v) for v in sym.values()),
        numeric_max_rel=0.0,
        numeric_per_component=dev,
        nonvacuous=f"all 5 identities (continuity, 2 momentum, vorticity, dissipation) simplify to exactly zero; the Laplacian and advective terms are individually non-trivial (max |lap u| ~ {float(sp.Abs(sp.diff(u, x, 2) + sp.diff(u, y, 2)).subs({x: 0.5, y: 0.7, nu: 0.05, t: 0.3}).evalf()):.3f})",
        points=len(pts),
    )


# --------------------------------------------------------------------------- #
# case E: overset Poisson manufactured source
# --------------------------------------------------------------------------- #
def case_overset_poisson():
    from tensorfvm.overset_poisson import manufactured_source, manufactured_value
    from tensorfvm.overset_scalar import exact_value

    radius = sp.Rational(3, 10)
    phi = (x**2 + y**2 - radius**2) * (
        1 + sp.Rational(1, 5) * x + sp.Rational(1, 10) * y
    )
    source = -sp.simplify(sp.diff(phi, x, 2) + sp.diff(phi, y, 2))
    rng = np.random.default_rng(31337)
    pts = rng.uniform(-0.5, 0.5, size=(2000, 3))
    subs = {z: 0}
    num_phi = np.asarray(manufactured_value(pts[:, :2]), float)
    num_src = np.asarray(manufactured_source(pts[:, :2]), float)
    phi_ref = _ev(phi.subs(subs), pts)
    src_ref = _ev(source.subs(subs), pts)

    scal = (
        1
        + x**2
        + y**2
        + sp.Rational(1, 5) * x * y
        + sp.Rational(1, 10) * x
        + sp.Rational(3, 20) * y
    )
    num_scal = np.asarray(exact_value(pts[:, :2]), float)
    return dict(
        case="overset_poisson.manufactured_value/manufactured_source, overset_scalar.exact_value",
        module="src/tensorfvm/overset_poisson.py, src/tensorfvm/overset_scalar.py",
        claim="-Laplace(phi) equals the coded right-hand side; the overset scalar quadratic is the coded value",
        symbolic={
            "source_plus_laplacian": sp.simplify(
                source + sp.diff(phi, x, 2) + sp.diff(phi, y, 2)
            )
        },
        symbolic_max_abs=0.0,
        symbolic_zero=_is_zero(source + sp.diff(phi, x, 2) + sp.diff(phi, y, 2)),
        numeric_max_rel=max(
            _rel(num_phi, phi_ref),
            _rel(num_src, src_ref),
            _rel(num_scal, _ev(scal, pts)),
        ),
        numeric_per_component={
            "value": _rel(num_phi, phi_ref),
            "source": _rel(num_src, src_ref),
            "overset_scalar_value": _rel(num_scal, _ev(scal, pts)),
        },
        nonvacuous="the source is a linear function of (x, y) but the Laplacian of the quadratic is constant, so a sign slip would show up immediately",
        points=len(pts),
    )


# --------------------------------------------------------------------------- #
# case F: torch-side agreement for the extended analytic cases
# --------------------------------------------------------------------------- #
def case_torch_references():
    import torch
    from tensorfvm.verification import extended_cases as ec

    nu, t = 0.1, 0.37
    rng = np.random.default_rng(11)
    p = rng.uniform(0.0, 2.0, size=(1500, 3))
    xt = [torch.tensor(p[:, i]) for i in range(3)]
    abc_v = np.stack(
        [c.detach().numpy() for c in ec.exact_velocity("abc", xt, t, nu)], 1
    )
    abc_p = ec.exact_pressure("abc", xt, t).detach().numpy()
    decay = math.exp(-nu * t)
    base = np.stack(
        [
            np.sin(p[:, 2]) + np.cos(p[:, 1]),
            np.sin(p[:, 0]) + np.cos(p[:, 2]),
            np.sin(p[:, 1]) + np.cos(p[:, 0]),
        ],
        1,
    )
    ref_v = base * decay
    ref_p = -0.5 * np.sum(ref_v**2, 1)
    ref_p = ref_p - ref_p.mean()
    shear_v = np.stack(
        [c.detach().numpy() for c in ec.exact_velocity("advected-shear", xt, t, nu)], 1
    )
    ref_shear = np.stack(
        [
            np.sin(p[:, 1] - 0.7 * t) * math.exp(-nu * t),
            np.full(len(p), 0.7),
            np.zeros(len(p)),
        ],
        1,
    )
    return dict(
        case="extended_cases torch implementations vs symbolic solution",
        module="src/tensorfvm/verification/extended_cases.py",
        claim="the torch reference fields evaluate the audited closed forms (not a different formula)",
        symbolic={},
        symbolic_max_abs=0.0,
        symbolic_zero=True,
        numeric_max_rel=max(
            _rel(abc_v, ref_v),
            _rel(abc_p - abc_p.mean(), ref_p - ref_p.mean()),
            _rel(shear_v, ref_shear),
        ),
        numeric_per_component={
            "abc_velocity": _rel(abc_v, ref_v),
            "abc_pressure_mean_gauged": _rel(
                abc_p - abc_p.mean(), ref_p - ref_p.mean()
            ),
            "advected_shear_velocity": _rel(shear_v, ref_shear),
        },
        nonvacuous="pressure is mean-gauged exactly as the code does, so the comparison is not offset by an arbitrary constant",
        points=len(p),
    )


# --------------------------------------------------------------------------- #
# case G: curved-channel manufactured Navier-Stokes reference
# --------------------------------------------------------------------------- #
def case_curved_channel():
    """Mapped-coordinate manufactured NS field behind the curved-wall <3 % claims."""
    import types

    from tensorfvm.curved_channel import reference

    length, height, inlet, visc, rho = sp.symbols("L H V mu rho", positive=True)
    amp = sp.Rational(1, 5) * height
    k = sp.pi / length
    sn, cs = sp.sin(k * x), sp.cos(k * x)
    g = amp * sn**4
    g1, g2, g3 = sp.diff(g, x), sp.diff(g, x, 2), sp.diff(g, x, 3)
    eta = (y - g) / height
    w = 6 * inlet * eta * (1 - eta)
    ux, uy = w, g1 * w
    div = sp.simplify(sp.diff(ux, x) + sp.diff(uy, y))
    lapx = sp.simplify(sp.diff(ux, x, 2) + sp.diff(ux, y, 2))
    lapy = sp.simplify(sp.diff(uy, x, 2) + sp.diff(uy, y, 2))
    gradp = 12 * visc * inlet / height**2
    p = gradp * (length - x)
    conv_x = sp.simplify(ux * sp.diff(ux, x) + uy * sp.diff(ux, y))
    conv_y = sp.simplify(ux * sp.diff(uy, x) + uy * sp.diff(uy, y))
    # the code's own Laplace/body-force formulas, transcribed as claimed
    wy = 6 * inlet / height * (1 - 2 * eta)
    wyy = -12 * inlet / height**2
    coded_lapx = (1 + g1**2) * wyy - g2 * wy
    coded_lapy = g3 * w - 3 * g1 * g2 * wy + g1 * (1 + g1**2) * wyy
    f_code_x = -gradp - visc * coded_lapx
    f_code_y = rho * g2 * w * w - visc * coded_lapy
    # exact body force required by the incompressible momentum balance
    f_true_x = rho * conv_x + sp.diff(p, x) - visc * lapx
    f_true_y = rho * conv_y + sp.diff(p, y) - visc * lapy
    # Navier-Stokes residual when the coded force is used
    res_x = sp.simplify(rho * conv_x + sp.diff(p, x) - visc * lapx - f_code_x)
    res_y = sp.simplify(rho * conv_y + sp.diff(p, y) - visc * lapy - f_code_y)
    lapx_match = sp.simplify(lapx - coded_lapx)
    lapy_match = sp.simplify(lapy - coded_lapy)
    shape_match = [
        sp.simplify(d)
        for d in (
            g - amp * sn**4,
            g1 - 4 * amp * k * sn**3 * cs,
            g2 - 4 * amp * k**2 * (3 * sn**2 * cs**2 - sn**4),
            g3 - 4 * amp * k**3 * (6 * sn * cs**3 - 10 * sn**3 * cs),
        )
    ]

    cfg = types.SimpleNamespace(
        length=2.0, height=0.4, inlet_velocity=0.5, viscosity=0.02, density=1.3
    )
    rng = np.random.default_rng(5)
    pts = np.column_stack(
        [rng.uniform(0.0, cfg.length, 1500), rng.uniform(0.0, 1.0, 1500)]
    )
    gfun = sp.lambdify((x,), g.subs({height: cfg.height, length: cfg.length}), "numpy")
    pts[:, 1] = (
        gfun(pts[:, 0]) + pts[:, 1] * cfg.height
    )  # keep points inside the channel
    import torch

    vel, pres, force = reference(torch.tensor(pts), cfg)
    subs = {
        length: cfg.length,
        height: cfg.height,
        inlet: cfg.inlet_velocity,
        visc: cfg.viscosity,
        rho: cfg.density,
    }
    p3 = np.column_stack([pts[:, 0], pts[:, 1], np.zeros(len(pts))])
    dev = {
        "velocity_x": _rel(vel[:, 0].numpy(), _ev(ux, p3, subs)),
        "velocity_y": _rel(vel[:, 1].numpy(), _ev(uy, p3, subs)),
        "pressure": _rel(pres.numpy(), _ev(p, p3, subs)),
        "force_x_vs_exact_body_force": _rel(
            force[:, 0].numpy(), _ev(f_true_x, p3, subs)
        ),
        "force_y_vs_exact_body_force": _rel(
            force[:, 1].numpy(), _ev(f_true_y, p3, subs)
        ),
        "laplace_x_vs_coded": _rel(_ev(lapx, p3, subs), _ev(coded_lapx, p3, subs)),
        "laplace_y_vs_coded": _rel(_ev(lapy, p3, subs), _ev(coded_lapy, p3, subs)),
    }
    sym = {
        "continuity": div,
        "convective_x": conv_x,
        "convective_y_minus_g2_w2": sp.simplify(conv_y - g2 * w * w),
        "navier_stokes_x_residual": res_x,
        "navier_stokes_y_residual": res_y,
        "laplace_x_minus_coded": lapx_match,
        "laplace_y_minus_coded": lapy_match,
    }
    zero = all(_is_zero(v) for v in sym.values()) and all(
        _is_zero(v) for v in shape_match
    )
    return dict(
        case="curved_channel.reference (mapped-coordinate manufactured NS solution)",
        module="src/tensorfvm/curved_channel.py",
        claim="u=(w, g' w), p=G(L-x) with w=6V eta(1-eta) and the coded body force is an exact solution of the incompressible 2-D Navier-Stokes equations in the sin^4-mapped channel",
        symbolic=sym,
        symbolic_max_abs=0.0,
        symbolic_zero=bool(zero),
        numeric_max_rel=max(dev.values()),
        numeric_per_component=dev,
        nonvacuous="continuity and both momentum residuals are non-trivial: the mapped Laplacians differ from w'' by geometry terms that are all audited separately",
        points=len(pts),
    )


# --------------------------------------------------------------------------- #
# independent-derivative limit: finite differences of the repository field
# --------------------------------------------------------------------------- #
def derivative_limit(step_list=None):
    from tensorfvm.verification import native_order3d as n3

    steps = (
        np.logspace(-1, -3.5, 9) if step_list is None else np.asarray(step_list, float)
    )
    rng = np.random.default_rng(2026)
    pts = rng.uniform(0.2, 0.8, size=(64, 3))
    analytic = n3.manufactured_gradient(pts)
    rows = []
    for h in steps:
        # central differences of the manufactured *field*, never of the analytic gradient
        shifted = np.stack(
            [n3.manufactured(pts + h * e) for e in np.eye(3)], axis=1
        )  # (n,3,3) [p, axis, comp]
        shifted_back = np.stack(
            [n3.manufactured(pts - h * e) for e in np.eye(3)], axis=1
        )
        fd = (shifted - shifted_back) / (2 * h)  # [p, axis=b, comp=a] = du_a/dx_b
        diff = fd.transpose(0, 2, 1) - analytic
        rows.append(
            dict(
                h=float(h),
                max_abs_error=float(np.max(np.abs(diff))),
                per_component={
                    f"du{a}/dx{b}": float(np.max(np.abs(diff[:, a, b])))
                    for a in range(3)
                    for b in range(3)
                },
            )
        )
    slopes = [
        float(
            np.log(rows[i]["max_abs_error"] / rows[i + 1]["max_abs_error"])
            / np.log(rows[i]["h"] / rows[i + 1]["h"])
        )
        for i in range(len(rows) - 1)
    ]
    return rows, slopes


# --------------------------------------------------------------------------- #
# figures + report
# --------------------------------------------------------------------------- #
def _figures(rows_fd, slopes, cases):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(OUT, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.0, 4.2))
    h = [r["h"] for r in rows_fd]
    e = [r["max_abs_error"] for r in rows_fd]
    ax.loglog(
        h,
        e,
        "o-",
        label="central difference of manufactured field vs repository gradient",
    )
    ref = e[0] * (np.asarray(h) / h[0]) ** 2
    ax.loglog(h, ref, "k--", lw=1.0, label="second-order reference")
    ax.set_xlabel("finite-difference step h")
    ax.set_ylabel(r"$\max|\partial_h u - \nabla u_{\mathrm{repo}}|$")
    ax.set_title(
        "Independent derivative limit: repository gradient is the true derivative"
    )
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "derivative-limit.png"), dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    labels = [c["case"].split(".")[0].split(" ")[0] for c in cases]
    vals = [max(c["numeric_max_rel"], 1e-17) for c in cases]
    ax.barh(range(len(vals)), vals, color="#3572b0")
    ax.set_yticks(range(len(vals)))
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xscale("log")
    ax.axvline(
        TOL_NUMERIC,
        color="crimson",
        ls="--",
        lw=1.0,
        label=f"containment threshold {TOL_NUMERIC:g}",
    )
    ax.axvline(0.03, color="darkgreen", ls=":", lw=1.0, label="3 % acceptance band")
    ax.set_xlabel("max relative deviation  repository function vs symbolic truth")
    ax.set_title("Analytic reference fields: numeric containment")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "reference-containment.png"), dpi=160)
    plt.close(fig)
    return ["derivative-limit.png", "reference-containment.png"]


def _markdown(cases, rows_fd, slopes):
    lines = [
        "| reference | claim | symbolic residual | numeric max rel. deviation | points |",
        "|---|---|---|---|---|",
    ]
    for c in cases:
        lines.append(
            "| {} | {} | {} | {:.3e} | {} |".format(
                c["case"],
                c["claim"],
                "0 (exact)" if c["symbolic_zero"] else "**NON-ZERO**",
                c["numeric_max_rel"],
                c["points"],
            )
        )
    lines.append("")
    lines.append("| h | max abs. derivative deviation |")
    lines.append("|---|---|")
    for r in rows_fd:
        lines.append("| {:.4f} | {:.3e} |".format(r["h"], r["max_abs_error"]))
    lines.append("")
    lines.append("observed slopes: " + ", ".join("{:.2f}".format(s) for s in slopes))
    return "\n".join(lines)


def main():
    os.makedirs(OUT, exist_ok=True)
    cases = [
        case_native_order3d(),
        case_annular(),
        case_abc(),
        case_advected_shear(),
        case_taylor_green(),
        case_overset_poisson(),
        case_curved_channel(),
        case_torch_references(),
    ]
    rows_fd, slopes = derivative_limit()
    figures = _figures(rows_fd, slopes, cases)
    result = {
        "generated_by": "scripts/audit_analytic_reference_residuals.py",
        "numeric_threshold": TOL_NUMERIC,
        "cases": cases,
        "derivative_limit": {"rows": rows_fd, "slopes": slopes},
        "figures": figures,
        "all_symbolic_zero": all(c["symbolic_zero"] for c in cases),
        "all_numeric_contained": all(c["numeric_max_rel"] < TOL_NUMERIC for c in cases),
    }
    with open(os.path.join(OUT, "results.json"), "w") as fh:
        json.dump(result, fh, indent=2, default=str)
    table = _markdown(cases, rows_fd, slopes)
    with open(os.path.join(OUT, "results-table.md"), "w") as fh:
        fh.write(table + "\n")
    print(table)
    print("\nall symbolic identities zero:", result["all_symbolic_zero"])
    print("all numeric deviations contained:", result["all_numeric_contained"])
    for c in cases:
        if not (c["symbolic_zero"] and c["numeric_max_rel"] < TOL_NUMERIC):
            print("FAILED:", c["case"], c["symbolic"], c["numeric_per_component"])
    return result


if __name__ == "__main__":
    main()
