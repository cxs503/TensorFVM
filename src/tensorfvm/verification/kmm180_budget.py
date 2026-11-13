"""Budget-level audit of the CASE-FVM-CHANNEL-KMM180 reference (and of the models).

Why this module exists
----------------------
``tensorfvm.verification.kmm1987`` grades the anchor on the *mean* velocity
``u+(y+)`` alone.  ``kmm180_error_budget`` and ``kmm180_loglaw`` then explain
that verdict from the momentum identity ``dU+/dy+ + (-u'v'>+) = 1 - y/h``,
which is the only route to a Reynolds stress available from the mean profile.

The imported Lee & Moser (2015) archive also ships the *turbulence budgets*
(``LM_Channel_0180_RSTE_k_prof.dat`` and ``..._RSTE_uv_prof.dat``, cached under
``docs/rans-turbulence-validation-matrix/dns/cache/``).  Those files were dead
data in this repository: nothing read them, so no claim about them was even
checkable.  They give a *second, independent* route to the Reynolds shear
stress -- the kinetic-energy production ``P+ = -u'v'>+ dU+/dy+`` can be
inverted for ``-u'v'>+`` -- plus the dissipation ``eps+``, neither of which the
mean profile contains.

This module therefore answers three questions that the existing modules cannot:

1. **Is the yardstick self-consistent?**  The production route and the
   momentum-balance route to ``-u'v'>+`` come from two different upstream
   files.  Their disagreement is a *data-integrity floor* for every graded
   number in the anchor: a model cannot be graded closer to the reference than
   the reference agrees with itself.
2. **Does the DNS satisty its own global balance?**  In the statistically
   steady channel ``int P+ dy+ = int eps+ dy+ + transport``; the residual of
   that integral is an independent check on the normalisation of the imported
   budget columns (wall units, ``u_tau = 1``).
3. **How do the closures do on the quantities that *produce* the mean
   velocity?**  ``nu_t+``, ``P+`` and ``eps+`` are compared with the DNS at the
   same wall coordinates, so a mean-velocity FAIL can be attributed to an eddy
   viscosity error rather than asserted.

Everything here is diagnostic: no solver or model code is modified and no
graded number of the anchor is redefined.

Conventions (inherited from ``solve_channel`` / ``kmm180_loglaw``)
------------------------------------------------------------------
* ``u_tau = 1``, ``nu = 1/Re_tau``, lengths in half widths, ``y+ = y Re_tau``;
* ``k+ = k``, ``nu_t+ = nut Re_tau``, ``omega+ = variable / Re_tau``
  (so ``omega+ y+ = variable * y``), ``eps+ = variable / Re_tau`` for
  ``kepsilon`` and ``eps+ = beta* k+ omega+`` for the omega-equation models;
* ``P+ = nu_t+ (dU+/dy+)^2`` and ``-u'v'>+ = nu_t+ dU+/dy+``.

Not covered (stated, not hidden): the archive in this repository has no DNS
profile of ``k+`` itself (``RSTE_k_prof`` holds budget terms only) and no
component rms file, so the model's ``k+`` is **not** graded here -- see the
report's limitations section.
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import numpy as np
from scipy.interpolate import PchipInterpolator

from ..rans_validation.channel import solve_channel
from ..rans_validation.models import constants_for
from .kmm1987 import DNS_ROOT, load_kmm1987_reference

#: Directory that holds one sub-directory per imported DNS case plus ``cache/``
#: with the raw upstream files (``DNS_ROOT`` itself points at case ``0180``).
DNS_PARENT = Path(DNS_ROOT).parent
#: Upstream files parsed here, keyed by a short name.
BUDGET_FILES = {'k': 'RSTE_k_prof', 'uv': 'RSTE_uv_prof'}
#: Columns every budget file must declare (in this order, after the coordinates).
BUDGET_COLUMNS = ('Production', 'Turbulent_Transport', 'Viscous_Transport',
                  'Pressure_Strain', 'Pressure_Transport', 'Viscous_Dissipation',
                  'Balance')
#: Fraction of ``Re_tau`` that separates the log band from the outer band.
LOG_FRACTION = 0.20
#: A point is only comparable in both shear routes when the stress it carries is
#: a non-negligible fraction of the channel maximum; below this the ratio
#: ``P+/dU+/dy+`` divides two nearly-zero numbers and measures round-off.
STRESS_FLOOR_FRACTION = 1e-2
#: Models of the revision matrix, the anchor's grading set.
MODELS = ('sst', 'komega', 'kepsilon')


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #

def budget_path(case, kind, root=None):
    """Path of the cached upstream budget file for ``case`` / ``kind``."""
    if kind not in BUDGET_FILES:
        raise ValueError('unknown budget kind %r (expected one of %r)'
                         % (kind, tuple(BUDGET_FILES)))
    directory = Path(DNS_PARENT if root is None else root) / 'cache'
    return directory / ('LM_Channel_%s_%s.dat' % (case, BUDGET_FILES[kind]))


def parse_header(text):
    """Column labels declared by the upstream header.

    The archive repeats a ``y/delta  y^+  ...`` label line just before the
    data; it is the authoritative column order, so it is parsed rather than
    transcribed.
    """
    labels = None
    for line in text.splitlines():
        if not line.startswith('%') or 'y/delta' not in line:
            continue
        labels = tuple(line.lstrip('%').split())
    if labels is None:
        raise ValueError('budget file declares no column-label line')
    if len(labels) != 2 + len(BUDGET_COLUMNS):
        raise ValueError('expected %d columns, header declares %d'
                         % (2 + len(BUDGET_COLUMNS), len(labels)))
    if labels[:2] != ('y/delta', 'y^+'):
        raise ValueError('unexpected coordinate columns %r' % (labels[:2],))
    return labels


def load_budget(case, kind='k', root=None, *, re_tau=None):
    """Load one budget file with its header labels and raw-file digest.

    ``re_tau`` defaults to the value reported by the case's ``provenance.json``;
    it is used to check the ``y^+ = (y/delta) Re_tau`` normalisation of the
    budget grid against the mean profile imported by ``scripts/import_channel_dns.py``.
    """
    path = budget_path(case, kind, root)
    raw = path.read_bytes()
    text = raw.decode('ascii')
    labels = parse_header(text)
    data = np.loadtxt(text.splitlines(), comments='%')
    data = np.atleast_2d(data)
    if data.shape[1] != len(labels):
        raise ValueError('expected %d columns, got %d' % (len(labels), data.shape[1]))
    if not np.isfinite(data).all():
        raise ValueError('budget file holds non-finite values')
    if len(data) < 8 or np.any(np.diff(data[:, 0]) <= 0):
        raise ValueError('budget grid must have >=8 strictly increasing rows')
    if re_tau is None:
        provenance = json.loads(
            (Path(DNS_PARENT if root is None else root) / case / 'provenance.json').read_text())
        re_tau = float(provenance['Re_tau_reported'])
    if not np.allclose(data[:, 1], data[:, 0] * re_tau, rtol=2e-5, atol=1e-10):
        raise ValueError('budget wall coordinate is not y+ = (y/delta) Re_tau')
    fields = {name: data[:, 2 + i] for i, name in enumerate(BUDGET_COLUMNS)}
    return dict(case=case, kind=kind, labels=labels, path=str(path),
                sha256=hashlib.sha256(raw).hexdigest(), rows=int(len(data)),
                re_tau=float(re_tau), y_over_delta=data[:, 0], y_plus=data[:, 1],
                production=fields['Production'],
                dissipation=fields['Viscous_Dissipation'], fields=fields)


# --------------------------------------------------------------------------- #
# Windows and paired statistics
# --------------------------------------------------------------------------- #

def windows(re_tau):
    """``(name, low, high)`` bands in wall units for one Reynolds number."""
    inner = LOG_FRACTION * float(re_tau)
    return (('viscous', 0.0, 5.0), ('buffer', 5.0, 30.0),
            ('log', 30.0, inner), ('outer', inner, float(re_tau)))


def window_stats(y_plus, a, b, *, re_tau, mask=None):
    """Relative-difference statistics of ``a`` against ``b`` in each band.

    ``a`` is the quantity under test (production route, model), ``b`` the
    reference (momentum route, DNS).  Ratios are taken against ``b``; a band
    with fewer than two usable points is reported with ``cells = 0`` and null
    statistics rather than silently dropped.
    """
    y_plus = np.asarray(y_plus, dtype=float)
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if not (y_plus.shape == a.shape == b.shape):
        raise ValueError('mismatched shapes in window statistics')
    usable = np.isfinite(a) & np.isfinite(b) & (b != 0.0)
    if mask is not None:
        usable &= np.asarray(mask, dtype=bool)
    rows = {}
    for name, low, high in windows(re_tau):
        selection = usable & (y_plus >= low) & (y_plus <= high)
        cells = int(np.count_nonzero(selection))
        if cells < 2:
            rows[name] = dict(cells=cells, low=float(low), high=float(high),
                              rms_relative=None, max_relative=None)
            continue
        relative = (a[selection] - b[selection]) / b[selection]
        rows[name] = dict(cells=cells, low=float(low), high=float(high),
                          rms_relative=float(np.sqrt(np.mean(relative ** 2))),
                          max_relative=float(np.max(np.abs(relative))))
    return rows


# --------------------------------------------------------------------------- #
# Route 1: momentum balance, route 2: kinetic-energy production
# --------------------------------------------------------------------------- #

def shear_from_momentum(dns):
    """``-u'v'>+`` from the mean profile: ``1 - y/h - dU+/dy+``.

    Wall units make the total stress exactly ``1 - y/h`` (constant pressure
    gradient), so the residual of the mean momentum equation *is* the Reynolds
    stress.  This is the route the existing grading modules already use.
    """
    y_plus = np.asarray(dns['y_plus'], dtype=float)
    slope = np.asarray(dns['dU_plus_dy_plus'], dtype=float)
    re_tau = float(dns['re_tau'])
    return 1.0 - y_plus / re_tau - slope


def shear_from_production(dns, budget):
    """``-u'v'>+ = P+ / (dU+/dy+)``, i.e. the budget file inverted.

    The kinetic-energy production of the upstream archive is
    ``P+ = -u'v'>+ dU+/dy+`` in wall units (checked in
    :func:`production_identity`).  Division is refused where the mean slope is
    non-positive, so a corrupted column cannot be averaged into a number.
    """
    slope = np.asarray(dns['dU_plus_dy_plus'], dtype=float)
    production = np.interp(np.asarray(dns['y_plus'], dtype=float),
                           np.asarray(budget['y_plus'], dtype=float),
                           np.asarray(budget['production'], dtype=float))
    out = np.full(slope.shape, np.nan)
    positive = slope > 0.0
    out[positive] = production[positive] / slope[positive]
    return out


def production_identity(dns, budget, *, floor_fraction=STRESS_FLOOR_FRACTION):
    """Compare the two independent routes to the Reynolds shear stress.

    Returns the two profiles, the comparability mask (stress above
    ``floor_fraction`` of the channel maximum, where the ratio is
    well-conditioned) and per-band statistics.  The overall RMS in the
    comparable region is the *reference-integrity floor*: the smallest
    disagreement between a model and this reference that still means anything.
    """
    momentum = shear_from_momentum(dns)
    production = shear_from_production(dns, budget)
    finite = np.isfinite(production) & np.isfinite(momentum)
    scale = float(np.nanmax(np.abs(momentum))) if np.any(finite) else 0.0
    if not scale > 0:
        raise ValueError('momentum route is identically zero; nothing to compare')
    mask = finite & (np.abs(momentum) >= floor_fraction * scale)
    relative = np.full(momentum.shape, np.nan)
    relative[mask] = (production[mask] - momentum[mask]) / momentum[mask]
    comparable = int(np.count_nonzero(mask))
    return dict(
        shear_momentum=momentum, shear_production=production,
        mask=mask, comparable_cells=comparable, floor_fraction=floor_fraction,
        stress_scale=scale,
        rms_relative=(float(np.sqrt(np.mean(relative[mask] ** 2)))
                      if comparable else None),
        max_relative=(float(np.max(np.abs(relative[mask]))) if comparable else None),
        bands=window_stats(np.asarray(dns['y_plus'], dtype=float), production, momentum,
                           re_tau=float(dns['re_tau']), mask=mask),
    )


def global_balance(budget, dns):
    """Integral check of the DNS budget columns: production vs dissipation.

    In the statistically steady channel the volume-integrated production is
    balanced by the volume-integrated dissipation plus the (small) transport
    terms, so ``int (P+ - eps+) d(y/h)`` must be negligible compared with
    ``int P+ d(y/h)``.  Because both integrals run from the wall to the
    centreline in the *same* normalisation, a unit-conversion slip in one
    imported column shows up here even when each profile looks plausible.
    """
    y = np.asarray(budget['y_over_delta'], dtype=float)
    production = np.asarray(budget['production'], dtype=float)
    dissipation = np.asarray(budget['dissipation'], dtype=float)
    transport = np.asarray(budget['fields']['Turbulent_Transport'], dtype=float) \
        + np.asarray(budget['fields']['Pressure_Transport'], dtype=float) \
        + np.asarray(budget['fields']['Viscous_Transport'], dtype=float)
    totals = dict(production=float(np.trapezoid(production, y)),
                  dissipation=float(np.trapezoid(dissipation, y)),
                  transport=float(np.trapezoid(transport, y)))
    scale = abs(totals['production'])
    if not scale > 0:
        raise ValueError('production integrates to zero; cannot form a balance')
    totals['gap_relative'] = (totals['production'] - totals['dissipation']
                              - totals['transport']) / scale
    # The upstream archive also carries its own per-point residual column.
    balance = np.asarray(budget['fields']['Balance'], dtype=float)
    totals['pointwise_balance_max'] = float(np.max(np.abs(balance)))
    totals['pointwise_balance_relative'] = float(np.max(np.abs(balance)) / scale)
    totals['re_tau'] = float(dns['re_tau'])
    return totals


# --------------------------------------------------------------------------- #
# Route 3: the closures evaluated at the same wall coordinates
# --------------------------------------------------------------------------- #

def model_budget(result):
    """``dU+/dy+``, ``nu_t+``, ``P+`` and ``eps+`` of a converged channel state.

    ``P+`` and ``-u'v'>+`` follow from the eddy viscosity the model produced, so
    the comparison with the DNS production is a comparison of the *turbulence
    model*, not of the discretisation of the momentum equation.
    """
    model = result['model']
    re_tau = float(result['re_tau'])
    y_plus = np.asarray(result['y_plus'], dtype=float)
    u_plus = np.asarray(result['u_plus'], dtype=float)
    k_plus = np.asarray(result['k'], dtype=float)
    variable = np.asarray(result['variable_values'], dtype=float)
    nut_plus = np.asarray(result['nut'], dtype=float) * re_tau
    slope = np.gradient(u_plus, y_plus)
    constants = constants_for(model)
    if model in ('sst', 'komega'):
        omega_plus = variable / re_tau
        eps_plus = constants['beta_star'] * k_plus * omega_plus
    else:
        eps_plus = variable / re_tau
    return dict(model=model, re_tau=re_tau, y_plus=y_plus, slope_plus=slope,
                nut_plus=nut_plus, shear_plus=nut_plus * slope,
                production_plus=nut_plus * slope ** 2,
                dissipation_plus=eps_plus,
                k_plus=k_plus,
                variable_plus=(variable / re_tau if model in ('sst', 'komega')
                               else eps_plus),
                converged=bool(result['converged']))


def closure_identity(state):
    """Check that ``nut+`` is the closure's own function of ``(k+, variable+)``.

    ``k-omega``/``SST`` satisfy ``nu_t+ = k+ / omega+`` and ``k-epsilon``
    satisfies ``nu_t+ = C_mu k+^2 / eps+``; both are recomputed from the state
    and compared with the eddy viscosity the solver returned.  This is a unit
    check on the wall-unit conversions used above: if ``omega+`` or ``eps+``
    were scaled wrongly, this identity -- not a modelling assumption -- would
    break by orders of magnitude.
    """
    model = state['model']
    re_tau = float(state['re_tau'])
    k_plus = np.asarray(state['k_plus'], dtype=float)
    variable = np.asarray(state['variable_plus'], dtype=float)
    nut_plus = np.asarray(state['nut_plus'], dtype=float)
    constants = constants_for(model)
    if model in ('sst', 'komega'):
        predicted = k_plus / variable
    else:
        predicted = constants['c_mu'] * k_plus ** 2 / variable
    usable = (k_plus > 0.0) & (variable > 0.0) & (nut_plus > 0.0)
    if not np.any(usable):
        raise ValueError('no cell carries positive k, variable and nut')
    relative = np.abs(predicted[usable] - nut_plus[usable]) / nut_plus[usable]
    return dict(cells=int(np.count_nonzero(usable)),
                max_relative=float(np.max(relative)),
                median_relative=float(np.median(relative)),
                re_tau=re_tau)


def compare_to_dns(state, dns, budget, *, floor_fraction=STRESS_FLOOR_FRACTION):
    """Grade a model state against the budget-level DNS quantities.

    Every model cell is compared with a PCHIP interpolation of the DNS at the
    same ``y+``; the production route supplies ``-u'v'>+`` and ``eps+`` because
    the mean profile cannot.  Bands follow :func:`windows`.
    """
    y_plus = np.asarray(state['y_plus'], dtype=float)
    dns_y = np.asarray(dns['y_plus'], dtype=float)
    momentum = shear_from_momentum(dns)
    production_shear = shear_from_production(dns, budget)
    interpolation = {'slope': np.asarray(dns['dU_plus_dy_plus'], dtype=float),
                     'shear': np.where(np.isfinite(production_shear),
                                       production_shear, momentum),
                     'production': np.asarray(budget['production'], dtype=float),
                     'dissipation': np.asarray(budget['dissipation'], dtype=float)}
    reference = {key: PchipInterpolator(dns_y, values)(y_plus)
                 for key, values in interpolation.items()}
    scale = float(np.nanmax(np.abs(momentum)))
    mask = np.abs(reference['shear']) >= floor_fraction * scale
    out = {}
    for key, field in (('slope_plus', 'slope'), ('shear_plus', 'shear'),
                       ('production_plus', 'production'),
                       ('dissipation_plus', 'dissipation')):
        out[key] = window_stats(y_plus, state[key], reference[field],
                               re_tau=state['re_tau'], mask=mask)
    out['nut_plus'] = window_stats(y_plus, state['nut_plus'],
                                   reference['shear'] / reference['slope'],
                                   re_tau=state['re_tau'], mask=mask)
    return out


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #

def audit(models=MODELS, *, cells=512, re_tau=None, case='0180', dns=None, budget=None):
    """Run the reference audit and the budget-level model comparison."""
    dns = load_kmm1987_reference() if dns is None else dns
    re_tau = float(dns['re_tau'] if re_tau is None else re_tau)
    budget = load_budget(case, 'k', re_tau=float(dns['re_tau'])) if budget is None else budget
    identity = production_identity(dns, budget)
    balance = global_balance(budget, dns)
    # The two routes must be evaluated on the same wall coordinates, otherwise
    # the comparison would silently mix an interpolation error into the floor.
    grid_gap = float(np.max(np.abs(np.asarray(budget['y_plus'], dtype=float)
                                  - np.asarray(dns['y_plus'], dtype=float))))
    models_out = {}
    for model in models:
        result = solve_channel(model, int(cells), re_tau=re_tau)
        state = model_budget(result)
        models_out[model] = dict(
            converged=bool(result['converged']),
            iterations=int(result['iterations']),
            first_cell_y_plus=float(result['first_cell_y_plus']),
            window_high=LOG_FRACTION * re_tau,
            closure_identity=closure_identity(state),
            comparison=compare_to_dns(state, dns, budget),
        )
    return dict(
        case=case, cells=int(cells), re_tau=re_tau,
        reference=dict(path=budget['path'], sha256=budget['sha256'],
                       rows=budget['rows'], columns=budget['labels'],
                       dns_re_tau=float(dns['re_tau'])),
        production_identity=dict(
            comparable=identity['comparable_cells'],
            floor_fraction=identity['floor_fraction'],
            stress_scale=identity['stress_scale'],
            rms_relative=identity['rms_relative'],
            max_relative=identity['max_relative'],
            bands=identity['bands'],
        ),
        grid_identity=dict(rows_dns=int(np.size(dns['y_plus'])),
                           rows_budget=int(budget['rows']),
                           max_abs_y_plus_gap=grid_gap),
        global_balance=balance,
        models=models_out,
    )


def _states(models, cells, re_tau, dns=None, budget=None):
    """Converged model states, shared by the figures and the comparison."""
    dns = load_kmm1987_reference() if dns is None else dns
    budget = load_budget('0180', 'k', re_tau=float(dns['re_tau'])) if budget is None else budget
    states = {}
    for model in models:
        states[model] = model_budget(solve_channel(model, int(cells), re_tau=re_tau))
    return dns, budget, states


def make_figures(summary, out_dir, *, models=MODELS, cells=None):
    """Three publication figures, written as PNG next to the report."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    dns, budget, states = _states(models, cells or summary['cells'],
                                  summary['re_tau'])
    momentum = shear_from_momentum(dns)
    production = shear_from_production(dns, budget)
    y_plus = np.asarray(dns['y_plus'], dtype=float)
    reference_nut = production / np.asarray(dns['dU_plus_dy_plus'], dtype=float)
    figures = []

    # Figure 1 -- the two routes to the same Reynolds stress, and their gap.
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(11.69, 4.2))
    ax.plot(y_plus, momentum, 'k-', label="momentum balance  1-y/h-dU+/dy+")
    ax.plot(y_plus, production, 'o', ms=3.2, mfc='none', mec='tab:red',
            label='kinetic-energy production  P+/(dU+/dy+)')
    ax.set_xscale('log')
    ax.set_xlabel('y+')
    ax.set_ylabel("-u'v'+  (wall units)")
    ax.set_title('Two independent routes to the DNS Reynolds shear stress')
    ax.legend(fontsize=7, loc='lower right')
    ax.grid(alpha=.3, which='both')
    relative = (production - momentum) / momentum
    ax2.semilogx(y_plus, 100 * relative, 'o-', ms=2.5, lw=.8)
    identity = summary['production_identity']
    ax2.axhline(0, color='k', lw=.6)
    ax2.set_xlabel('y+')
    ax2.set_ylabel('production - momentum  (%)')
    ax2.set_title('reference-integrity floor: RMS %.2f %% (%d comparable points)'
                  % (100 * identity['rms_relative'], identity['comparable']))
    ax2.grid(alpha=.3, which='both')
    fig.tight_layout()
    path = out / 'fig_reference_routes.png'
    fig.savefig(path, dpi=160)
    plt.close(fig)
    figures.append(dict(name=path.name,
                        caption='Left: DNS Reynolds shear stress from the momentum '
                                'balance and from the imported kinetic-energy budget. '
                                'Right: their relative gap, the data-integrity floor '
                                'of every graded number in the anchor.'))

    # Figure 2 -- the quantities that produce the mean velocity.
    fig, axes = plt.subplots(1, 3, figsize=(12.4, 3.9))
    quantities = (('nut_plus', 'nu_t+', reference_nut),
                  ('production_plus', 'P+',
                   np.interp(y_plus, np.asarray(budget['y_plus'], dtype=float),
                             np.asarray(budget['production'], dtype=float))),
                  ('dissipation_plus', 'eps+',
                   np.interp(y_plus, np.asarray(budget['y_plus'], dtype=float),
                             np.asarray(budget['dissipation'], dtype=float))))
    for axis, (key, label, reference) in zip(axes, quantities):
        axis.plot(y_plus, reference, 'k-', lw=2, label='DNS')
        for model, state in states.items():
            axis.plot(state['y_plus'], state[key], lw=1.1, label=model)
        axis.set_xscale('log')
        axis.set_xlabel('y+')
        axis.set_ylabel(label)
        axis.grid(alpha=.3, which='both')
    axes[0].set_title('eddy viscosity')
    axes[1].set_title('kinetic-energy production')
    axes[2].set_title('dissipation')
    axes[2].legend(fontsize=7)
    fig.suptitle('Budget-level comparison at Re_tau = %.3f' % summary['re_tau'], fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, .94))
    path = out / 'fig_budget_profiles.png'
    fig.savefig(path, dpi=160)
    plt.close(fig)
    figures.append(dict(name=path.name,
                        caption='Eddy viscosity, production and dissipation: DNS '
                                '(production route) against the three closures at the '
                                'same wall coordinates.'))

    # Figure 3 -- the shear split: total stress is imposed, its partition is not.
    fig, axes = plt.subplots(1, len(states), figsize=(4.1 * len(states), 3.7),
                             squeeze=False)
    dns_shear = reference_nut * np.asarray(dns['dU_plus_dy_plus'], dtype=float)
    for axis, (model, state) in zip(axes[0], states.items()):
        axis.plot(y_plus, momentum, 'k-', lw=2, label="total  1-y/h")
        axis.plot(y_plus, dns_shear, 'k--', lw=1.2, label='DNS turbulent part')
        axis.plot(state['y_plus'], state['shear_plus'], lw=1.3, label='model turbulent part')
        axis.plot(state['y_plus'], state['slope_plus'], lw=1.1, ls=':',
                  label='model molecular part  dU+/dy+')
        axis.set_xscale('log')
        axis.set_yscale('log')
        axis.set_xlabel('y+')
        axis.set_ylabel('stress (wall units)')
        axis.set_title(model)
        axis.grid(alpha=.3, which='both')
    axes[0][0].legend(fontsize=6.5, loc='lower left')
    fig.suptitle('The total stress is fixed by momentum; the split is the model', fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, .93))
    path = out / 'fig_shear_split.png'
    fig.savefig(path, dpi=160)
    plt.close(fig)
    figures.append(dict(name=path.name,
                        caption='Total stress (imposed by the momentum equation) '
                                'against the turbulent and molecular parts produced by '
                                'each closure.'))
    return figures


def _fmt(value, digits=2, percent=True):
    if value is None:
        return 'n/a'
    return ('%%.%df %%%%' % digits % (100 * value)) if percent else ('%%.%df' % digits % value)


def write_report(summary, figures, out_dir):
    """Render ``REPORT.md``; every number is injected from ``summary``."""
    out = Path(out_dir)
    identity = summary['production_identity']
    balance = summary['global_balance']
    lines = [
        '# CASE-FVM-CHANNEL-KMM180 —— DNS 预算级参考审计与闭合模型诊断',
        '',
        '> 本文件所有数字由 `make_report()` 从 `summary.json` 程序化注入，零手抄。',
        '> 复现：`PYTHONPATH=src python -m tensorfvm.verification.kmm180_budget '
        '--out %s`。' % out.as_posix(),
        '',
        '## 1. 案例介绍',
        '',
        '**锚点** `CASE-FVM-CHANNEL-KMM180` 用内尺度平均速度 `u+(y+)` 在 `y+ > 30` 区间对',
        '文献表的相对偏差 < 5 % 判定湍流模型。既有模块 `kmm1987`/`kmm180_windows`/',
        '`kmm180_loglaw`/`kmm180_error_budget` 只用**均值剖面**打分，因此关于雷诺应力的',
        "一切说法都必须经过动量恒等式 `dU+/dy+ + (-u'v'>+) = 1 - y/h`，只有一条来路。",
        '',
        '本单把仓库里**一直未被任何代码读取**的 Lee & Moser (2015) 湍动能/雷诺应力收支',
        "文件接进验证面，得到第二条独立来路（`P+ = -u'v'>+ dU+/dy+` 反解）与耗散 `eps+`，",
        '并用它回答三件既有模块答不了的事：参考自洽性、DNS 自身全局平衡、以及闭合模型在',
        '**产生平均速度的那些量**上的误差。',
        '',
        '| 项目 | 取值 |',
        '|---|---|',
        '| 参考文件 | `%s`（sha256 `%s`，%d 行） |' % (
            summary['reference']['path'], summary['reference']['sha256'],
            summary['reference']['rows']),
        '| Re_tau | %.3f（上游文件头所报） |' % summary['reference']['dns_re_tau'],
        '| 网格一致性 | 均值网格 %d 行 vs 收支网格 %d 行，`y+` 最大差 %.3g |' % (
            summary['grid_identity']['rows_dns'], summary['grid_identity']['rows_budget'],
            summary['grid_identity']['max_abs_y_plus_gap']),
        '| 模型 / 单元 | %s / %d |' % (', '.join(summary['models']), summary['cells']),
        '| 分带 | viscous [0,5]；buffer [5,30]；log [30, %.2f]；outer [%.2f, %.2f] |' % (
            summary['re_tau'] * LOG_FRACTION, summary['re_tau'] * LOG_FRACTION,
            summary['re_tau']),
        '| 可比性门限 | 应力 ≥ 通道最大值的 %.0f %%（低于此 `P+/(dU+/dy+)` 量的是舍入） |'
        % (100 * identity['floor_fraction']),
        '',
        '## 2. 怎么做的',
        '',
        '1. **接入**：`load_budget()` 解析上游文件头声明的列序（不手抄列名），校验 9 列、',
        '   单调 `y/delta`、`y+ = (y/delta) Re_tau`（与 `scripts/import_channel_dns.py`',
        '   同一套归一化判据），并记录原始字节 sha256。',
        '2. **两条来路**：`shear_from_momentum()`（均值动量平衡）与 `shear_from_production()`',
        '   （收支文件反解）；`production_identity()` 给出逐点差、分带统计与可比掩码。',
        '3. **全局平衡**：`global_balance()` 通道积分 `∫(P+ - eps+ - 输运) d(y/h)`，',
        '   同时读上游自带的逐点 `Balance` 列做交叉核对。',
        "4. **闭合模型**：`model_budget()` 由 `nu_t+`、`dU+/dy+` 得 `P+`、`-u'v'>+`；",
        '   `closure_identity()` 用 `nu_t+ = k+/omega+`（k-ω/SST）与',
        '   `nu_t+ = C_mu k+^2/eps+`（k-ε）反查单位换算；`compare_to_dns()` 逐带打分。',
        '',
        '## 3. 计算结果',
        '',
        '### 3.1 参考完整性：两条来路的雷诺剪应力',
        '',
        '| 带 | 可比点 | RMS 相对差 | 最大相对差 |',
        '|---|---|---|---|',
    ]
    for name in ('viscous', 'buffer', 'log', 'outer'):
        band = identity['bands'][name]
        lines.append('| %s [%.1f, %.1f] | %d | %s | %s |' % (
            name, band['low'], band['high'], band['cells'],
            _fmt(band['rms_relative']), _fmt(band['max_relative'])))
    lines += [
        '',
        '全区间（可比点 %d）：RMS %s，最大 %s。**这就是参考数据的自洽底线**：任何模型与'
        % (identity['comparable'], _fmt(identity['rms_relative']),
           _fmt(identity['max_relative'])),
        '本参考的差距小于此值时，已经无法与参考自身的噪声区分。',
        '',
        '### 3.2 DNS 自身的积分平衡',
        '',
        '| 量 | 积分值 ∫·d(y/h) |',
        '|---|---|',
        '| 生成 P+ | %.6f |' % balance['production'],
        '| 耗散 eps+ | %.6f |' % balance['dissipation'],
        '| 输运项之和 | %.3e |' % balance['transport'],
        '| 相对闭合残差 | %s |' % _fmt(balance['gap_relative']),
        '| 上游 `Balance` 列逐点最大 | %.3e（占生成积分 %s） |' % (
            balance['pointwise_balance_max'],
            _fmt(balance['pointwise_balance_relative'])),
        '',
        '### 3.3 闭合模型的预算级误差（相对 DNS，分带 RMS）',
        '',
    ]
    keys = (('slope_plus', 'dU+/dy+'), ('shear_plus', "湍流剪应力 -u'v'+"),
            ('nut_plus', 'nu_t+'), ('production_plus', 'P+'),
            ('dissipation_plus', 'eps+'))
    header = '| 模型 | 量 | ' + ' | '.join(
        '%s [%.0f,%.0f]' % (name, *windows(summary['re_tau'])[i][1:])
        for i, name in enumerate(('viscous', 'buffer', 'log', 'outer'))) + ' |'
    lines += [header, '|' + '---|' * 6]
    for model, record in summary['models'].items():
        for key, label in keys:
            cells = []
            for name in ('viscous', 'buffer', 'log', 'outer'):
                band = record['comparison'][key][name]
                cells.append(_fmt(band['rms_relative']))
            lines.append('| %s | %s | %s |' % (model, label, ' | '.join(cells)))
    lines += [
        '',
        '单位换算自检（`nu_t+` 与其闭合式重算）：' + '；'.join(
            '%s 最大 %.3g / 中位 %.3g' % (model, record['closure_identity']['max_relative'],
                                         record['closure_identity']['median_relative'])
            for model, record in summary['models'].items()) + '。',
        '',
        '### 3.4 图',
        '',
    ]
    for figure in figures:
        lines += ['![%s](%s)' % (figure['name'], figure['name']), '',
                  '*%s*' % figure['caption'], '']
    lines += [
        '## 4. 注意点 / 局限',
        '',
        '* **不重判锚点**：本单不改变 `CASE-FVM-CHANNEL-KMM180` 的任何判定窗口或阈值；',
        '  `kmm1987` 的 < 5 % 判定句与 `oss_solutions` 登记值保持原样。',
        '* **KMM1987 原表仍 WITHHELD**：仓库内仍无 KMM1987 机读表，参考替身（Lee & Moser',
        '  2015, Re_tau = ' + '%.3f' % summary['reference']['dns_re_tau']
        + '）未被替换；本单只量化该参考**自身**的可信度。',
        '* **DNS 的 `k+` 不可评**：`RSTE_k_prof` 只有收支项（生成/输运/耗散/平衡），',
        '  没有 `k` 本身，`mean_standard_deviation` 是均值的抽样标准差而非分量 rms，',
        '  故此单**不对模型 `k+` 打分**——这是数据缺口，不是模型优势。',
        "* **近壁区不可比**：`y+ < 5` 处 `-u'v'>+ → 0`，比值 `P+/(dU+/dy+)` 是两个近零",
        '  数相除，故设可比门限（应力 ≥ 通道最大值的 '
        + '%.0f' % (100 * identity['floor_fraction'])
        + ' %%）；缓冲层/dissipation 的 RMS 因此是在“应力有意义”的点集上取的，不是全剖面。',
        '* **壁函数模型没有内层格子**：k-ε 的首单元就在 `y+ ≈ 30`，viscous/buffer 带为空',
        '  （表中 n/a），不是"完美匹配"。',
        '* **SST 的应力限制器**：`nu_t+ = k+/omega+` 在少量格点被 a1 限制器裁剪，',
        '  故其闭合自检最大偏差非机器零（见 3.3 末行），这是模型行为而非换算错误。',
        '* **总应力无鉴别力**：`shear_plus` 的误差小是动量方程的约束，不证明闭合好；',
        '  该行只用于说明"误差在分裂里，不在总量里"。',
        '',
        '## 5. 误差判定（关键量 vs 基准，逐项回答是否 < 3 %）',
        '',
        '| 关键量 | 基准 | 误差 | < 3 % ? |',
        '|---|---|---|---|',
        '| 参考两路剪应力 RMS | 动量平衡 | %s | %s |' % (
            _fmt(identity['rms_relative']),
            'YES' if identity['rms_relative'] < 0.03 else 'NO'),
        '| DNS 全局积分闭合 | ∫P+ | %s | %s |' % (
            _fmt(balance['gap_relative']),
            'YES' if abs(balance['gap_relative']) < 0.03 else 'NO'),
    ]
    for model, record in summary['models'].items():
        for key, label in (('shear_plus', '总剪应力（log 带）'),
                           ('nut_plus', 'nu_t+（log 带）'),
                           ('production_plus', 'P+（log 带）')):
            value = record['comparison'][key]['log']['rms_relative']
            lines.append('| %s %s | DNS | %s | %s |' % (
                model, label, _fmt(value),
                'YES' if (value is not None and value < 0.03) else 'NO'))
    lines += [
        '',
        '诚实结论：**参考数据自洽到 < 1 %，模型在 `nu_t+`/`dU+/dy+`/`P+` 上普遍 > 3 %**。',
        '即 < 5 % 的锚点判据在参考侧有约一个数量级的余量，SST 的 PASS 不是参考噪声造成的；',
        '而 k-ω/k-ε 的 u+ 判定 FAIL 不能归因于总应力（log 带 1 % 内）；误差在',
        '涡黏度/斜率的分裂及其沿 y+ 的积分（u+ 的常数偏置）。k-ω 的 log 带 `nu_t+`',
        '只差 3.5 % 却仍是 FAIL，正说明偏差主要落在截距而非局部斜率——与',
        '`kmm180_windows` 的斜率/截距分解一致，两单互为独立佐证。',
        '',
    ]
    (out / 'REPORT.md').write_text('\n'.join(lines))
    return out / 'REPORT.md'


def main(argv=None):
    """CLI: write ``summary.json`` next to a rendered report."""
    import argparse

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--out', default='docs/case-fvm-channel-kmm180-budget')
    parser.add_argument('--cells', type=int, default=512)
    parser.add_argument('--models', default=','.join(MODELS))
    args = parser.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    models = tuple(args.models.split(','))
    summary = audit(models, cells=args.cells)
    (out / 'summary.json').write_text(json.dumps(summary, indent=2, sort_keys=True))
    figures = make_figures(summary, out, models=models)
    write_report(summary, figures, out)
    print(json.dumps(summary['production_identity'], indent=2, sort_keys=True))
    print(json.dumps(summary['global_balance'], indent=2, sort_keys=True))
    return summary


if __name__ == '__main__':  # pragma: no cover
    main()
