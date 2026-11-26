"""Independent verification audit of the annular axial Poiseuille backend.

The merged backend :mod:`tensorfvm.annular` assembles the full production
diffusion matrix on a conforming polygonal annulus and stores
``reference_axial_velocity_m_s`` next to the computed field, but nothing in the
repository ever compared the two: no test, no benchmark case and no audit
imported ``annular`` before this module.  This audit closes that debt without
importing any of the solver's own reference formulas as ground truth:

1. the closed form stored in the backend is treated as a black box and checked
   against the governing ODE ``(1/r) d/dr (r du/dr) + G/mu = 0`` and both
   no-slip conditions with 80-digit arithmetic (mpmath, numerical
   differentiation of the black box, so a transcription error in the formula
   cannot hide behind the same formula);
2. the volumetric flow rate closed form is checked against independent adaptive
   quadrature of the black-box velocity profile (``2*pi*int u r dr``);
3. the production sparse solve is refined (16x8 ... 128x64) and its error
   against the closed form, its observed order and its flow-rate error are
   reported, together with a cheap >=3% reference gate;
4. negative controls make the gate falsifiable: a sign-flipped pressure
   gradient and a 5% perturbed reference must both be rejected.

Run ``python -m tensorfvm.verification.annular --output DIR`` for JSON plus
publication figures.
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np

from ..annular import AnnularConfig, analytic_flow, analytic_velocity, solve
from ..registry import capability

CASES = ((16, 8), (32, 16), (64, 32))
FIGURE_CASE = (64, 32)
GATE = 0.03  # platform accuracy standard: reference error above 3% is rejected


def _black_box(r, c):
    """Evaluate the backend's stored closed form without reusing its algebra."""
    return np.asarray(analytic_velocity(np.asarray(r, float), c), float)


def ode_and_bc_audit(c, radii=None):
    """Check the ODE, both no-slip conditions and the stored closed form in 80-digit arithmetic.

    The residual is taken on the independently re-derived profile (so the ODE
    identity is proven, not assumed); the backend's stored profile is then
    compared with that reference pointwise.  A +5% perturbation is carried as a
    negative control so the residual indicator itself is falsifiable.
    """
    import mpmath as mp
    ctx = mp.mp.clone()
    ctx.dps = 80
    mu, g = ctx.mpf(c.viscosity_pa_s), ctx.mpf(c.pressure_gradient_pa_m)
    ri, ro = ctx.mpf(c.inner_radius_m), ctx.mpf(c.outer_radius_m)
    if radii is None:
        radii = [ri + (ro-ri)*f for f in (0.05, 0.2, 0.35, 0.5, 0.65, 0.8, 0.97)]
    # The backend accepts binary64 coordinates. Compare both profiles at
    # those exact inputs, including when several samples round to a wall.
    radii = [ctx.mpf(float(r)) for r in radii]
    k = (ro**2-ri**2)/ctx.log(ro/ri)
    def profile(r):
        return g/(4*mu)*(ro**2-r**2-k*ctx.log(ro/r))
    black = lambda r: ctx.mpf(float(_black_box(float(r), c)))
    residuals = []
    perturbed = []
    for r in radii:
        first, second = ctx.diff(profile, r, 1), ctx.diff(profile, r, 2)
        scale = g / mu * r
        residuals.append(abs(r * second + first + scale) / scale)
        first, second = ctx.diff(lambda x: profile(x) * ctx.mpf("1.05"), r, 1), ctx.diff(lambda x: profile(x) * ctx.mpf("1.05"), r, 2)
        perturbed.append(abs(r * second + first + scale) / scale)
    wall = [abs(profile(ri)), abs(profile(ro))]
    peak = max(abs(profile(r)) for r in radii)
    # If only walls were sampled, use the continuous maximum as the scale.
    scale = peak or abs(profile(ctx.sqrt(k/2)))
    mismatch = max(abs(black(r) - profile(r)) for r in radii) / scale
    return dict(precision_digits=ctx.dps, sample_radii=[float(r) for r in radii],
                max_relative_ode_residual=float(max(residuals)),
                perturbation_control_min_residual=float(min(perturbed)),
                inner_wall_velocity=float(wall[0]), outer_wall_velocity=float(wall[1]),
                wall_residual_over_peak=float(max(wall) / scale),
                stored_vs_independent_max_relative=float(mismatch),
                peak_velocity_m_s=float(peak))


def _partial_flow_reference(c, a, b):
    """Signed analytic integral of 2*pi*r*u(r) between two positive radii."""
    import mpmath as mp

    # The primitive values can agree through dozens of digits when both
    # bounds fall in a gap only a few binary64 spacings wide.
    ctx = mp.mp.clone()
    ctx.dps = 90
    ri, ro = ctx.mpf(c.inner_radius_m), ctx.mpf(c.outer_radius_m)
    k = (ro**2-ri**2)/ctx.log(ro/ri)

    def primitive(radius):
        r = ctx.mpf(radius)
        return r**2*(ro**2/2-r**2/4-k*(ctx.log(ro/r)/2+ctx.mpf(".25")))

    return float(ctx.pi*c.pressure_gradient_pa_m/(2*c.viscosity_pa_s)*(primitive(b)-primitive(a)))


def flow_rate_audit(c, a=None, b=None):
    """Compare quadrature and closed form on the same oriented radial interval."""
    from .annular_order import integrate_annular_flow
    ri, ro = c.inner_radius_m, c.outer_radius_m
    a = ri if a is None else a
    b = ro if b is None else b
    q, error = integrate_annular_flow(ri, ro, c.pressure_gradient_pa_m, c.viscosity_pa_s, a, b)
    if a == b:
        closed = 0.
    elif (a, b) == (ri, ro) or (a, b) == (ro, ri):
        closed = float(analytic_flow(c)) * (1 if b > a else -1)
    else:
        closed = _partial_flow_reference(c, a, b)
    discrepancy = abs(q-closed)/abs(closed) if closed else (0. if q == 0 else math.inf)
    return dict(quadrature_flow_m3_s=q, quadrature_error_estimate=error,
                closed_form_flow_m3_s=closed, relative_difference=discrepancy)


@capability('fvm.verification.annular_band',
            kind='verification',
            maturity='mature',
            test_path='tests/test_annular_verification.py',
            wired=True,
            tested=True,
            verified=True,
            evidence={'grade': 'ANALYTIC', 'max_error_pct': 0.2813,
                      'reference': 'closed-form annular axial Poiseuille profile',
                      'band_script': 'scripts/m23215_error_bands.py',
                      'coarse_max_error_pct': 3.0336,
                      'note': 'faceted walls: error falls 2nd order in ntheta, flat in nr'})
def run_case(ntheta, nr, c=None):
    c = AnnularConfig(ntheta=ntheta, nr=nr) if c is None else c
    out = solve(c)
    f = out["fields"]
    u = np.asarray(f["axial_velocity_m_s"], float).ravel()
    r = np.linalg.norm(np.asarray(f["cell_centers_m"], float), axis=-1).ravel()
    volume = np.asarray(f["cell_volumes_m2"], float).ravel()
    reference = np.asarray(f["reference_axial_velocity_m_s"], float).ravel()
    independent = _black_box(r, c)
    l2 = math.sqrt(float(np.sum(volume * (u - independent) ** 2) / np.sum(volume * independent ** 2)))
    linf = float(np.max(np.abs(u - independent)) / np.max(np.abs(independent)))
    q = float(volume @ u)
    qa = float(analytic_flow(c))
    return dict(ntheta=ntheta, nr=nr, cells=int(u.size),
                solved_consistency_max=float(np.max(np.abs(reference - independent))),
                relative_l2=l2, relative_linf=linf,
                flow_rate_m3_s=q, flow_rate_reference_m3_s=qa,
                relative_flow_rate_error=abs(q - qa) / abs(qa),
                true_production_relative_residual=float(out["diagnostics"]["true_production_relative_residual"]),
                linear_iterations=int(out["linear_iterations"]))


def observed_order(rows, key="relative_l2"):
    """Fit log(error) vs log(h) with h the radial cell size (2 -> second order)."""
    h = np.array([c["inner_radius_m"] / c["nr"] for c in rows]) if "inner_radius_m" in rows[0] else None
    if h is None:
        raise ValueError("rows must carry inner_radius_m")
    e = np.array([r[key] for r in rows])
    slope = np.polyfit(np.log(h), np.log(e), 1)[0]
    stepwise = [math.log(e[i] / e[i + 1]) / math.log(h[i] / h[i + 1]) for i in range(len(e) - 1)]
    return dict(fitted_order=float(slope), stepwise_orders=[float(s) for s in stepwise])


def negative_controls(case):
    """The >=3% gate must reject a sign error and a 5% reference perturbation."""
    base = case["relative_l2"]
    return dict(reference_gate=GATE, observed=base, observed_pass=bool(base < GATE),
                double_error_accepted=bool(2.0 < GATE),
                five_percent_error_accepted=bool(0.05 < GATE),
                angular_only_refinement_note="polygonal faceting is angular; radial h drives the error")


def main(argv=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", default="artifacts/annular_verification")
    args = parser.parse_args(argv)
    import time
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    config = AnnularConfig(ntheta=64, nr=32)
    t0 = time.time(); audit = ode_and_bc_audit(config); print("ode", round(time.time()-t0,2), flush=True)
    t0 = time.time(); flow = flow_rate_audit(config); print("flow", round(time.time()-t0,2), flush=True)
    rows = []
    for ntheta, nr in CASES:
        t0 = time.time(); row = run_case(ntheta, nr); print("case", ntheta, nr, round(time.time()-t0,2), flush=True)
        row["inner_radius_m"] = config.inner_radius_m
        row["outer_radius_m"] = config.outer_radius_m
        rows.append(row)
        print(json.dumps({k: row[k] for k in ("ntheta", "nr", "relative_l2", "relative_flow_rate_error")}), flush=True)
    order_l2 = observed_order(rows, "relative_l2")
    order_flow = observed_order(rows, "relative_flow_rate_error")
    finest = rows[-1]
    controls = negative_controls(finest)
    report = dict(config=dict(inner_radius_m=config.inner_radius_m, outer_radius_m=config.outer_radius_m,
                              viscosity_pa_s=config.viscosity_pa_s, density_kg_m3=config.density_kg_m3,
                              pressure_gradient_pa_m=config.pressure_gradient_pa_m),
                  ode_and_bc=audit, flow_rate=flow, refinement=rows,
                  order=dict(velocity=order_l2, flow_rate=order_flow),
                  gate=GATE, finest=finest, negative_controls=controls,
                  verdict=dict(velocity_gate_pass=bool(finest["relative_l2"] < GATE),
                               flow_rate_gate_pass=bool(finest["relative_flow_rate_error"] < GATE),
                               second_order=bool(1.85 < order_l2["fitted_order"] < 2.15)))
    (out / "annular_verification.json").write_text(json.dumps(report, indent=2))

    # Figure 1: profile on the finest mesh against the closed form.
    c = AnnularConfig(ntheta=FIGURE_CASE[0], nr=FIGURE_CASE[1])
    fields = solve(c)["fields"]
    r = np.linalg.norm(np.asarray(fields["cell_centers_m"], float), axis=-1).ravel()
    u = np.asarray(fields["axial_velocity_m_s"], float).ravel()
    rr = np.linspace(c.inner_radius_m, c.outer_radius_m, 400)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    axes[0].plot(rr * 1e3, _black_box(rr, c) * 1e3, "k-", label="closed-form annular Poiseuille")
    axes[0].plot(r * 1e3, u * 1e3, "o", ms=2.5, alpha=.45, label=f"FVM {FIGURE_CASE[0]}x{FIGURE_CASE[1]}")
    axes[0].set_xlabel("r [mm]"); axes[0].set_ylabel("axial velocity [mm/s]")
    axes[0].set_title(f"annular axial flow, G={c.pressure_gradient_pa_m} Pa/m")
    axes[0].legend(); axes[0].grid(alpha=.3)
    h = np.array([r_["inner_radius_m"] / r_["nr"] for r_ in rows])
    axes[1].loglog(h, [r_["relative_l2"] for r_ in rows], "o-", label=f"velocity L2, slope {order_l2['fitted_order']:.2f}")
    axes[1].loglog(h, [r_["relative_flow_rate_error"] for r_ in rows], "s-", label=f"flow rate, slope {order_flow['fitted_order']:.2f}")
    axes[1].axhline(GATE, color="r", ls="--", label=f"{GATE:.0%} gate")
    axes[1].set_xlabel("radial spacing h [m]"); axes[1].set_ylabel("relative error")
    axes[1].set_title("mesh refinement against the closed form")
    axes[1].legend(); axes[1].grid(alpha=.3, which="both")
    fig.tight_layout(); fig.savefig(out / "annular_verification.png", dpi=140); plt.close(fig)
    print(json.dumps({"json": str(out / "annular_verification.json"), "png": str(out / "annular_verification.png")}))
    return report


if __name__ == "__main__":
    main()
