"""Turbulence-state validation for CASE-FVM-CHANNEL-KMM180.

While ``kmm1987`` grades the *mean* velocity ``u+(y+)`` and ``kmm180_budget``
audits the DNS budget files, this module closes the loop by comparing the
RANS model's **Reynolds shear stress** ``-u'v'>+ = nu_t+ dU+/dy+`` directly
against the DNS-implied stress from the momentum balance ``1 - y/h - dU+/dy+``.

Three diagnostics are produced per model:

1. **Reynolds stress profile** ``-u'v'>+(y+)``: model vs DNS-implied, with
   per-band RMS in viscous/buffer/log/outer regions.
2. **Eddy viscosity profile** ``nu_t+(y+)``: model vs DNS-implied
   ``nu_t+ = (1 - y/h)/dU+/dy+ - 1``, which is the exact inversion of the
   momentum identity.
3. **Production–dissipation ratio** ``P+/eps+``: in the log layer this should
   approach unity for an equilibrium closure; the deviation quantifies how
   far the model is from local equilibrium.

All quantities are in wall units (``u_tau = 1``, ``nu = 1/Re_tau``).

Not graded: the DNS archive in this repository has no ``k+`` profile (only
budget terms), so the model's ``k+`` is reported but not compared against a
reference.  The asymptotic log-layer value ``k+ = 1/sqrt(beta*) = 3.333`` is
quoted as a sanity check.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.interpolate import PchipInterpolator

from ..rans_validation.channel import solve_channel
from ..rans_validation.models import constants_for
from .kmm180_budget import (
    DNS_PARENT,
    load_budget,
    shear_from_momentum,
    window_stats,
    windows,
)
from .kmm1987 import DNS_ROOT, load_kmm1987_reference

#: Models to audit.
MODELS = ('sst', 'komega', 'kepsilon')
#: Resolutions for the standard matrix.
RESOLUTIONS = (64, 128, 256)


def dns_implied_nut_plus(dns):
    """Eddy viscosity implied by the DNS mean profile.

    From the momentum identity ``(nu + nu_t) dU/dy = 1 - y/h``:
    ``nu_t+ = (1 - y/h) / (dU+/dy+) - 1``.
    """
    y_plus = np.asarray(dns['y_plus'], dtype=float)
    slope = np.asarray(dns['dU_plus_dy_plus'], dtype=float)
    re_tau = float(dns['re_tau'])
    total_stress = 1.0 - y_plus / re_tau
    # Avoid division by zero near the centreline
    with np.errstate(divide='ignore', invalid='ignore'):
        nut_plus = total_stress / slope - 1.0
    nut_plus[~np.isfinite(nut_plus)] = np.nan
    return nut_plus


def model_reynolds_stress(result):
    """Reynolds shear stress ``-u'v'>+ = nu_t+ dU+/dy+`` from the model.

    Returns ``y_plus``, ``stress_plus`` (the model's Reynolds stress), and
    ``slope`` (dU+/dy+).
    """
    y_plus = np.asarray(result['y_plus'], dtype=float)
    u_plus = np.asarray(result['u_plus'], dtype=float)
    nut = np.asarray(result['nut'], dtype=float)
    re_tau = float(result['re_tau'])
    nut_plus = nut * re_tau

    # Compute dU+/dy+ from the model profile
    slope = np.gradient(u_plus, y_plus)

    stress_plus = nut_plus * slope
    return y_plus, stress_plus, slope, nut_plus


def model_production(result):
    """Turbulent kinetic energy production ``P+ = nu_t+ (dU+/dy+)^2``."""
    y_plus, stress_plus, slope, nut_plus = model_reynolds_stress(result)
    production_plus = nut_plus * slope ** 2
    return y_plus, production_plus


def model_dissipation(result):
    """Dissipation rate ``eps+`` from the model state.

    For omega-based models: ``eps+ = beta* k+ omega+``.
    For k-epsilon: ``eps+ = epsilon+`` directly.
    """
    model = result['model']
    c = constants_for(model)
    y_plus = np.asarray(result['y_plus'], dtype=float)
    k_plus = np.asarray(result['k'], dtype=float)  # k+ = k (u_tau = 1)
    re_tau = float(result['re_tau'])

    if model == 'kepsilon':
        # epsilon+ = epsilon / (u_tau^4 / nu) = epsilon * nu / u_tau^4
        # In the solver, epsilon is stored as variable_values
        eps = np.asarray(result['variable_values'], dtype=float)
        eps_plus = eps / re_tau  # epsilon+ = epsilon * nu / u_tau^3 = epsilon / Re_tau
    else:
        # omega-based: eps+ = beta* k+ omega+
        beta_star = c.get('beta_star', 0.09)
        omega = np.asarray(result['variable_values'], dtype=float)
        omega_plus = omega / re_tau  # omega+ = omega * nu / u_tau^2 = omega / Re_tau
        eps_plus = beta_star * k_plus * omega_plus

    return y_plus, eps_plus


def audit_turbulence_state(model, n, *, dns=None, budget=None):
    """Full turbulence-state audit for one (model, n) case.

    Returns a dict with:
    - model, n, converged
    - reynolds_stress: {y_plus, model_stress, dns_stress, window_stats}
    - eddy_viscosity: {y_plus, model_nut, dns_nut, window_stats}
    - production_dissipation: {y_plus, P_plus, eps_plus, ratio, log_layer_mean_ratio}
    - k_plus: {y_plus, k_plus, asymptotic_value, log_layer_mean}
    """
    if dns is None:
        dns = load_kmm1987_reference()
    if budget is None:
        budget = load_budget('0180', kind='uv')

    result = solve_channel(model, n, re_tau=float(dns['re_tau']))
    if not result['converged']:
        return dict(model=model, n=n, converged=False)

    re_tau = float(dns['re_tau'])

    # Reynolds stress
    y_plus, stress_plus, slope, nut_plus = model_reynolds_stress(result)
    dns_stress = shear_from_momentum(dns)

    # Interpolate DNS stress onto model grid
    dns_stress_interp = PchipInterpolator(dns['y_plus'], dns_stress)(y_plus)

    # Eddy viscosity
    dns_nut_plus = dns_implied_nut_plus(dns)
    dns_nut_interp = PchipInterpolator(dns['y_plus'], dns_nut_plus)(y_plus)

    # Production and dissipation
    _, production_plus = model_production(result)
    _, eps_plus = model_dissipation(result)

    # P+/eps+ ratio (only where both are positive)
    with np.errstate(divide='ignore', invalid='ignore'):
        pd_ratio = np.where(
            (production_plus > 0) & (eps_plus > 0),
            production_plus / eps_plus,
            np.nan
        )

    # k+ profile
    k_plus = np.asarray(result['k'], dtype=float)
    c = constants_for(model)
    beta_star = c.get('beta_star', 0.09)
    k_plus_asymptotic = 1.0 / np.sqrt(beta_star)

    # Window statistics
    rs_stats = window_stats(y_plus, stress_plus, dns_stress_interp, re_tau=re_tau)
    nut_stats = window_stats(y_plus, nut_plus, dns_nut_interp, re_tau=re_tau)

    # Log-layer mean P+/eps+ ratio
    log_mask = (y_plus >= 30.0) & (y_plus <= 0.2 * re_tau)
    log_pd_ratio = float(np.nanmean(pd_ratio[log_mask])) if np.any(log_mask) else None

    # Log-layer mean k+
    log_k_plus = float(np.mean(k_plus[log_mask])) if np.any(log_mask) else None

    return dict(
        model=model, n=n, converged=True, re_tau=re_tau,
        reynolds_stress=dict(
            y_plus=y_plus.tolist(),
            model_stress=stress_plus.tolist(),
            dns_stress=dns_stress_interp.tolist(),
            window_stats=rs_stats,
        ),
        eddy_viscosity=dict(
            y_plus=y_plus.tolist(),
            model_nut=nut_plus.tolist(),
            dns_nut=dns_nut_interp.tolist(),
            window_stats=nut_stats,
        ),
        production_dissipation=dict(
            y_plus=y_plus.tolist(),
            P_plus=production_plus.tolist(),
            eps_plus=eps_plus.tolist(),
            ratio=pd_ratio.tolist(),
            log_layer_mean_ratio=log_pd_ratio,
        ),
        k_plus=dict(
            y_plus=y_plus.tolist(),
            k_plus=k_plus.tolist(),
            asymptotic_value=k_plus_asymptotic,
            log_layer_mean=log_k_plus,
        ),
    )


def full_audit(dns=None, budget=None):
    """Run the turbulence-state audit for all models and resolutions."""
    if dns is None:
        dns = load_kmm1987_reference()
    if budget is None:
        budget = load_budget('0180', kind='uv')

    results = []
    for model in MODELS:
        for n in RESOLUTIONS:
            results.append(audit_turbulence_state(model, n, dns=dns, budget=budget))
    return results


def write_artifacts(results, out_dir):
    """Write the turbulence-state artifacts to JSON."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Summary JSON
    summary = dict(
        schema='tensorfvm.kmm180-turbulence-state/1',
        anchor='CASE-FVM-CHANNEL-KMM180',
        results=results,
    )
    (out_dir / 'turbulence_state.json').write_text(json.dumps(summary, indent=2))

    # CSV summary of window statistics
    import csv
    with (out_dir / 'turbulence_state_summary.csv').open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'model', 'n', 'quantity', 'band', 'cells', 'rms_relative', 'max_relative'
        ])
        for r in results:
            if not r.get('converged'):
                continue
            key = f"{r['model']}_n{r['n']}"
            for quantity, stats_key in [
                ('reynolds_stress', 'reynolds_stress'),
                ('eddy_viscosity', 'eddy_viscosity'),
            ]:
                stats = r[stats_key]['window_stats']
                for band, row in stats.items():
                    writer.writerow([
                        r['model'], r['n'], quantity, band,
                        row['cells'], row['rms_relative'], row['max_relative']
                    ])
            # P+/eps+ ratio in log layer
            pd = r['production_dissipation']
            if pd['log_layer_mean_ratio'] is not None:
                writer.writerow([
                    r['model'], r['n'], 'P_over_eps', 'log_layer',
                    '-', pd['log_layer_mean_ratio'], '-'
                ])
            # k+ in log layer
            kp = r['k_plus']
            if kp['log_layer_mean'] is not None:
                writer.writerow([
                    r['model'], r['n'], 'k_plus', 'log_layer',
                    '-', kp['log_layer_mean'], kp['asymptotic_value']
                ])
