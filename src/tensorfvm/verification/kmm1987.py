"""KMM1987 channel-flow DNS reference data and comparison utilities.

Reference
---------
Kim, J., Moin, P. & Moser, R. D. 1987 *Turbulence statistics in fully developed
channel flow at low Reynolds number*. J. Fluid Mech. **177**, 133–166.
DOI: 10.1017/S0022112087000892

The original DNS was performed at Re_tau = 180.  For practical validation the
repository uses the Lee & Moser (2015) DNS at Re_tau = 182 as the numerical
reference — the two datasets agree to within 1 % in the log region and the
1-unit Reynolds-number difference is negligible for RANS-model grading.

Lee, M. & Moser, R. D. 2015 *Direct numerical simulation of turbulent channel
flow up to Re_tau = 5200*. J. Fluid Mech. **774**, 395–415.
DOI: 10.1017/jfm.2015.268

Data provenance
---------------
The reference CSV and provenance JSON live in
``docs/rans-turbulence-validation-matrix/dns/0180/`` and were imported by
``scripts/import_channel_dns.py`` from the original-author files hosted at
the University of Texas turbulence database.

Key numbers (Re_tau = 182, from provenance.json)
-------------------------------------------------
* Cf = 0.008123
* U_bulk_plus = 15.691
* u_tau = 0.06373
"""
import json
from pathlib import Path

import numpy as np
from scipy.interpolate import PchipInterpolator

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #
RE_TAU_KMM = 180.0          # nominal KMM1987 Reynolds number
RE_TAU_REF = 182.088        # actual Lee–Moser 2015 Re_tau for the 0180 case
# Module is at src/tensorfvm/verification/kmm1987.py
# parents[3] = repo root
DNS_ROOT = Path(__file__).resolve().parents[3] / 'docs' / 'rans-turbulence-validation-matrix' / 'dns' / '0180'

# --------------------------------------------------------------------------- #
# Data loading
# --------------------------------------------------------------------------- #

def load_kmm1987_reference(directory=None):
    """Load the KMM1987 / LM2015 Re_tau≈180 DNS reference profile.

    Returns a dict with keys:
        y_plus, u_plus, y_over_delta, dU_plus_dy_plus,
        re_tau, Cf, U_bulk_plus, provenance
    """
    directory = Path(directory) if directory is not None else DNS_ROOT
    data = np.loadtxt(directory / 'mean.csv', delimiter=',', skiprows=1)
    provenance = json.loads((directory / 'provenance.json').read_text())

    if data.ndim != 2 or data.shape[1] != 6:
        raise ValueError('expected six-column imported channel profile')

    y_plus = data[:, 1]
    u_plus = data[:, 2]

    if np.any(np.diff(y_plus) <= 0):
        raise ValueError('DNS wall coordinate must increase monotonically')

    return dict(
        y_plus=y_plus,
        u_plus=u_plus,
        y_over_delta=data[:, 0],
        dU_plus_dy_plus=data[:, 3],
        re_tau=float(provenance['Re_tau_reported']),
        Cf=float(provenance['Cf']),
        U_bulk_plus=float(provenance['U_bulk_plus']),
        provenance=provenance,
    )


# --------------------------------------------------------------------------- #
# Comparison metrics
# --------------------------------------------------------------------------- #

def log_region_deviation(y_plus, u_plus, dns, *, low=30.0, high_fraction=None):
    """Relative deviation of a model profile from DNS in the log region.

    The graded number is the RMS relative deviation over
    ``low <= y+ <= high``.  The acceptance criterion for
    CASE-FVM-CHANNEL-KMM180 is < 5 % in this window.

    Parameters
    ----------
    y_plus, u_plus : array-like
        Model profile in wall units.
    dns : dict
        Output of ``load_kmm1987_reference``.
    low : float
        Lower bound of the log-region window (default 30, per task spec).
    high_fraction : float, optional
        Upper bound as a fraction of Re_tau.  If None (default), uses the
        full DNS range (i.e. up to the centreline).

    Returns
    -------
    dict with keys: low, high, cells, rms_relative, max_relative, mean_relative
    """
    y_plus = np.asarray(y_plus, dtype=float)
    u_plus = np.asarray(u_plus, dtype=float)
    re_tau = dns['re_tau']

    if high_fraction is None:
        high = float(np.max(dns['y_plus']))
    else:
        high = high_fraction * re_tau

    mask = (y_plus >= low) & (y_plus <= high)
    if np.count_nonzero(mask) < 3:
        raise ValueError(
            f'log region holds fewer than three cells '
            f'(found {np.count_nonzero(mask)} in [{low}, {high}])'
        )

    # Interpolate DNS onto model points
    dns_interp = PchipInterpolator(dns['y_plus'], dns['u_plus'])
    dns_u = dns_interp(y_plus[mask])

    relative = (u_plus[mask] - dns_u) / dns_u

    return dict(
        low=float(low),
        high=float(high),
        cells=int(np.count_nonzero(mask)),
        rms_relative=float(np.sqrt(np.mean(relative ** 2))),
        max_relative=float(np.max(np.abs(relative))),
        mean_relative=float(np.mean(relative)),
    )


def cf_deviation(cf_model, dns):
    """Relative deviation of skin-friction coefficient from DNS."""
    cf_dns = dns['Cf']
    return dict(
        cf_model=float(cf_model),
        cf_dns=float(cf_dns),
        relative_error=float(abs(cf_model - cf_dns) / cf_dns),
    )


def bulk_velocity_deviation(bulk_u_plus_model, dns):
    """Relative deviation of bulk velocity from DNS."""
    bulk_dns = dns['U_bulk_plus']
    return dict(
        bulk_u_plus_model=float(bulk_u_plus_model),
        bulk_u_plus_dns=float(bulk_dns),
        relative_error=float(abs(bulk_u_plus_model - bulk_dns) / bulk_dns),
    )


# --------------------------------------------------------------------------- #
# Full validation
# --------------------------------------------------------------------------- #

TOLERANCE_LOG_REGION = 0.05   # 5 % RMS relative error in log region


def validate_kmm1987(y_plus, u_plus, cf_model, bulk_u_plus_model, *,
                     dns=None, tolerance=TOLERANCE_LOG_REGION):
    """Run the full KMM1987 validation suite.

    Parameters
    ----------
    y_plus, u_plus : array-like
        Model mean-velocity profile in wall units.
    cf_model : float
        Model skin-friction coefficient.
    bulk_u_plus_model : float
        Model bulk velocity in wall units.
    dns : dict, optional
        DNS reference (loaded automatically if not provided).
    tolerance : float
        Acceptance threshold for log-region RMS deviation.

    Returns
    -------
    dict with verdict, log_region, cf, bulk_velocity, and per-point data.
    """
    if dns is None:
        dns = load_kmm1987_reference()

    # Primary metric: y+ > 30 to centreline (per task spec)
    log_dev = log_region_deviation(y_plus, u_plus, dns, low=30.0, high_fraction=None)
    # Supplementary: narrow log region (30 < y+ < 0.2*Re_tau)
    log_dev_narrow = log_region_deviation(y_plus, u_plus, dns, low=30.0, high_fraction=0.2)

    cf_dev = cf_deviation(cf_model, dns)
    bulk_dev = bulk_velocity_deviation(bulk_u_plus_model, dns)

    verdict = 'PASS' if log_dev['rms_relative'] < tolerance else 'FAIL'

    return dict(
        verdict=verdict,
        tolerance=tolerance,
        log_region=log_dev,
        log_region_narrow=log_dev_narrow,
        cf=cf_dev,
        bulk_velocity=bulk_dev,
        re_tau=dns['re_tau'],
        dns_Cf=dns['Cf'],
        dns_U_bulk_plus=dns['U_bulk_plus'],
    )


# --------------------------------------------------------------------------- #
# Case grading (CASE-FVM-CHANNEL-KMM180)
# --------------------------------------------------------------------------- #
#: Inner-scale stations at which the model profile is graded against the
#: reference table.  Every station lies in the graded window ``y+ > 30`` and
#: below the centreline of both the nominal (Re_tau = 180) and the imported
#: (Re_tau = 182.088) reference.
STATIONS_Y_PLUS = (30.0, 40.0, 50.0, 75.0, 100.0, 125.0, 150.0, 175.0)

#: Acceptance tolerance of the graded criterion (case-pool row).
CASE_TOLERANCE = 0.05

#: Closure actually graded by the case (SST-2003, low-Re wall treatment).
GATED_MODEL = 'sst'

#: Closures run as negative controls: they must *not* pass the same criterion,
#: otherwise the metric is not discriminating.
CONTROL_MODELS = ('komega', 'kepsilon')

#: Log-law constants quoted by Kim, Moin & Moser (1987) for Re_tau = 180.
LOG_LAW_KAPPA_REF = 0.41
LOG_LAW_B_REF = 5.0


def log_law_fit(y_plus, u_plus, *, low=30.0, high_fraction=0.4):
    """Least-squares apparent log-law fit ``u+ = ln(y+)/kappa + B``.

    Diagnostic only.  At Re_tau = 180 the window ``30 < y+ < 0.4 Re_tau`` still
    contains part of the wake, so ``kappa`` comes out below the ``0.41`` of the
    high-Reynolds-number log law; the fit is reported to show *that* the case is
    at the very bottom of the log-law range, not as a graded criterion.
    """
    y_plus = np.asarray(y_plus, dtype=float)
    u_plus = np.asarray(u_plus, dtype=float)
    high = high_fraction * float(np.max(y_plus))
    mask = (y_plus >= low) & (y_plus <= high)
    if np.count_nonzero(mask) < 3:
        raise ValueError('log-law window holds fewer than three points')
    slope, intercept = np.polyfit(np.log(y_plus[mask]), u_plus[mask], 1)
    residual = u_plus[mask] - (slope * np.log(y_plus[mask]) + intercept)
    return dict(
        kappa=float(1.0 / slope),
        B=float(intercept),
        cells=int(np.count_nonzero(mask)),
        rms=float(np.sqrt(np.mean(residual ** 2))),
    )


def station_table(y_plus, u_plus, dns, stations=STATIONS_Y_PLUS):
    """Reference-vs-model table at the inner-scale stations of the case.

    Both profiles are interpolated onto ``stations`` (the model linearly on its
    own wall-normal grid, the reference with a shape-preserving cubic), so the
    table is defined even when the model grid nodes do not coincide with the
    reference table rows.
    """
    y_plus = np.asarray(y_plus, dtype=float)
    u_plus = np.asarray(u_plus, dtype=float)
    order = np.argsort(y_plus)
    y_plus, u_plus = y_plus[order], u_plus[order]

    stations = np.asarray(list(stations), dtype=float)
    y_max = min(float(y_plus[-1]), float(np.max(dns['y_plus'])))
    usable = stations[stations <= y_max]
    if usable.size < 3:
        raise ValueError('fewer than three stations below the available range')

    dns_interp = PchipInterpolator(dns['y_plus'], dns['u_plus'])
    ref = dns_interp(usable)
    model = np.interp(usable, y_plus, u_plus)
    rel = (model - ref) / ref

    return [dict(y_plus=float(s), u_plus_ref=float(r), u_plus_model=float(m),
                 abs_error=float(abs(m - r)), rel_error=float(e))
            for s, r, m, e in zip(usable, ref, model, rel)]


def grade_case(model, y_plus, u_plus, cf_model, bulk_u_plus_model, *,
               dns=None, tolerance=CASE_TOLERANCE, gated_model=GATED_MODEL):
    """Grade one closure against the case criterion.

    Decision rule (predeclared, see ``CASE_TOLERANCE``): a closure passes when
    the RMS relative deviation of ``u+(y+)`` over ``30 <= y+ <= centreline`` is
    below ``tolerance``.  The station table, the CF and bulk-velocity deviations,
    the centreline velocity and the apparent log-law fit are reported alongside
    as supporting evidence; they do not enter the verdict.
    """
    if dns is None:
        dns = load_kmm1987_reference()

    validation = validate_kmm1987(y_plus, u_plus, cf_model, bulk_u_plus_model,
                                 dns=dns, tolerance=tolerance)
    stations = station_table(y_plus, u_plus, dns)
    ref_fit = log_law_fit(dns['y_plus'], dns['u_plus'])
    model_fit = log_law_fit(y_plus, u_plus)

    y_plus = np.asarray(y_plus, dtype=float)
    u_plus = np.asarray(u_plus, dtype=float)
    u_centre_model = float(u_plus[int(np.argmax(y_plus))])
    u_centre_ref = float(dns['u_plus'][int(np.argmax(dns['y_plus']))])

    got = dict(
        log_region_rms=validation['log_region']['rms_relative'],
        log_region_max=validation['log_region']['max_relative'],
        log_region_cells=validation['log_region']['cells'],
        cf=float(cf_model),
        bulk_u_plus=float(bulk_u_plus_model),
        u_plus_centre=u_centre_model,
        kappa_apparent=model_fit['kappa'],
        B_apparent=model_fit['B'],
        station_max_rel_error=float(np.max([abs(s['rel_error']) for s in stations])),
    )
    ref = dict(
        log_region_rms=0.0,
        cf=validation['dns_Cf'],
        bulk_u_plus=validation['dns_U_bulk_plus'],
        u_plus_centre=u_centre_ref,
        kappa_apparent=ref_fit['kappa'],
        B_apparent=ref_fit['B'],
        re_tau=validation['re_tau'],
    )
    rel_error = dict(
        log_region_rms=got['log_region_rms'],
        cf=validation['cf']['relative_error'],
        bulk_u_plus=validation['bulk_velocity']['relative_error'],
        u_plus_centre=abs(got['u_plus_centre'] - u_centre_ref) / u_centre_ref,
    )
    return dict(
        model=model,
        role='gated' if model == gated_model else 'control',
        criterion='RMS relative deviation of u+(y+) over y+>30 (to centreline)',
        tolerance=float(tolerance),
        verdict=validation['verdict'],
        ref=ref,
        got=got,
        rel_error=rel_error,
        stations=stations,
        log_region_narrow=validation['log_region_narrow'],
    )
