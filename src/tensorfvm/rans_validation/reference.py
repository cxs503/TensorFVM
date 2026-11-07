"""Reference wall laws and the log-region deviation metric.

Two independent references are used, both classical and both analytic:

* ``reichardt`` -- Reichardt (1951) composite law, fitted to channel and pipe
  data and used across the whole layer (viscous sublayer to the outer edge):
  ``U+ = ln(1 + kappa y+)/kappa + 7.8 (1 - e^{-y+/11} - (y+/11) e^{-y+/3})``.
  The same expression anchors ``tests/test_wall_law_literature.py``.
* ``log_law`` -- the classic logarithmic law ``U+ = ln(y+)/kappa + B`` with
  kappa = 0.41 and B = 5.2, i.e. the slope the repository wall functions assume.

Neither is a substitute for a primary DNS table: they are correlations, so a
model that matches them inherits their own fitting error (a few percent in the
overlap layer, larger below y+ = 30).  The matrix therefore reports deviation
against *both* references instead of grading against a single curve.
"""
import json
from pathlib import Path

import numpy as np

from ..wall_functions import KAPPA


def reichardt(y_plus, kappa=KAPPA):
    """Reichardt (1951) composite law of the wall."""
    y = np.asarray(y_plus, dtype=float)
    if not np.isfinite(y).all() or np.any(y < 0):
        raise ValueError('non-negative finite y+ required')
    scaled = y / 11
    return np.log1p(kappa * y) / kappa + 7.8 * (1 - np.exp(-scaled) - scaled * np.exp(-y / 3))


def log_law(y_plus, kappa=KAPPA, b=5.2):
    """Classic logarithmic law for y+ > 0."""
    y = np.asarray(y_plus, dtype=float)
    if not np.isfinite(y).all() or np.any(y <= 0):
        raise ValueError('positive finite y+ required')
    return np.log(y) / kappa + b


REFERENCE_LAWS = {'reichardt': reichardt, 'log_law': log_law}


def log_region_deviation(y_plus, u_plus, reference, *, re_tau, low=30., high_fraction=.2):
    """Relative deviation of a profile from ``reference`` in the log region.

    The graded number is the volume-weighted RMS relative deviation over
    ``low <= y+ <= high_fraction*re_tau``; the maximum relative deviation over the
    same window is returned as a supplementary, stricter indicator.
    """
    y_plus = np.asarray(y_plus, dtype=float)
    u_plus = np.asarray(u_plus, dtype=float)
    high = float(high_fraction) * float(re_tau)
    mask = (y_plus >= low) & (y_plus <= high)
    if np.count_nonzero(mask) < 3:
        raise ValueError('log region holds fewer than three cells; refine the mesh or widen the window')
    reference_values = reference(y_plus[mask])
    relative = (u_plus[mask] - reference_values) / reference_values
    weights = np.full(relative.size, 1. / relative.size)
    return dict(low=float(low), high=high, cells=int(np.count_nonzero(mask)),
                rms_relative=float(np.sqrt(np.sum(weights * relative ** 2))),
                max_relative=float(np.max(np.abs(relative))),
                mean_relative=float(np.mean(relative)),
                mean_reference=float(np.sum(weights * reference_values)),
                weights=np.full(relative.size, 1. / relative.size))


def fit_log_law(y_plus, u_plus, *, re_tau, low=30., high_fraction=.2):
    """Least-squares fit ``U+ = ln(y+)/kappa + B`` inside the log region."""
    y_plus = np.asarray(y_plus, dtype=float)
    u_plus = np.asarray(u_plus, dtype=float)
    mask = (y_plus >= low) & (y_plus <= high_fraction * re_tau)
    if np.count_nonzero(mask) < 3:
        raise ValueError('log region holds fewer than three cells')
    design = np.column_stack([np.log(y_plus[mask]), np.ones(np.count_nonzero(mask))])
    (slope, intercept), *_ = np.linalg.lstsq(design, u_plus[mask], rcond=None)
    kappa = 1 / slope
    residual = design @ np.array([slope, intercept]) - u_plus[mask]
    return dict(kappa=float(kappa), intercept=float(intercept),
                rms_residual=float(np.sqrt(np.mean(residual ** 2))), cells=int(np.count_nonzero(mask)))

def log_law_indicator(y_plus, u_plus, *, re_tau, low=30., high_fraction=.3):
    """Local ``Xi = y+ dU+/dy+`` in the log window, whose reciprocal is kappa.

    In a logarithmic layer ``Xi`` is constant and equals ``1/kappa``, so its
    mean over the window measures the *local* slope the model produces, without
    the intercept of a two-parameter fit contaminating the answer.
    """
    y_plus = np.asarray(y_plus, dtype=float)
    u_plus = np.asarray(u_plus, dtype=float)
    mask = (y_plus >= low) & (y_plus <= high_fraction * re_tau)
    if np.count_nonzero(mask) < 4:
        raise ValueError('log region holds fewer than four cells')
    y = y_plus[mask]
    u = u_plus[mask]
    slope = np.gradient(u, y, edge_order=2)
    xi = y * slope
    interior = slice(1, -1)
    return dict(kappa_local=float(1 / np.mean(xi[interior])),
                xi_spread=float(np.max(xi[interior]) - np.min(xi[interior])),
                cells=int(np.count_nonzero(mask)), low=float(y[1]), high=float(y[-2]))

def load_dns_profile(directory):
    """Load an imported original-DNS channel profile (Lee & Moser 2015).

    ``directory`` must be an output of ``scripts/import_channel_dns.py``: a CSV
    with ``y_over_delta, y_plus, U, dU_dy, W, P`` and the provenance JSON that
    records the upstream URL, the raw-file SHA-256 and the documented Cf.  The
    reported ``Re_tau`` is the value in the DNS header, not the nominal case
    name, so the model can be run at exactly the DNS Reynolds number.
    """
    directory = Path(directory)
    data = np.loadtxt(directory / 'mean.csv', delimiter=',', skiprows=1)
    provenance = json.loads((directory / 'provenance.json').read_text(encoding='utf-8'))
    if data.ndim != 2 or data.shape[1] != 6:
        raise ValueError('expected the six-column imported channel profile')
    y_plus, u_plus = data[:, 1], data[:, 2]
    if np.any(np.diff(y_plus) <= 0):
        raise ValueError('DNS wall coordinate must increase monotonically')
    reported = float(provenance['Re_tau_reported'])
    if not np.isclose(y_plus[-1], np.max(y_plus)):
        raise ValueError('inconsistent DNS wall coordinate')
    return dict(y_plus=y_plus, u_plus=u_plus, y_over_delta=data[:, 0],
                dU_plus_dy_plus=data[:, 3], re_tau=reported,
                Cf=float(provenance['Cf']), upstream=provenance['files']['mean']['upstream_url'],
                raw_sha256=provenance['files']['mean']['raw_sha256'],
                case=Path(provenance['files']['mean']['upstream_url']).name)


def dns_deviation(y_plus, u_plus, dns, *, low=30., high_fraction=.2, re_tau=None):
    """Relative deviation of a model profile from a DNS profile in the log region.

    The DNS profile is interpolated (monotone cubic) onto the model cell centres
    that fall inside the window, so no extrapolation and no re-gridding of the
    model output enters the comparison.
    """
    from scipy.interpolate import PchipInterpolator
    re_tau = dns['re_tau'] if re_tau is None else float(re_tau)
    y_plus = np.asarray(y_plus, dtype=float)
    u_plus = np.asarray(u_plus, dtype=float)
    mask = (y_plus >= low) & (y_plus <= high_fraction * re_tau)
    if np.count_nonzero(mask) < 3:
        raise ValueError('log region holds fewer than three cells')
    if y_plus[mask].min() < dns['y_plus'].min() or y_plus[mask].max() > dns['y_plus'].max():
        raise ValueError('log window leaves the DNS sample range')
    reference = PchipInterpolator(dns['y_plus'], dns['u_plus'])(y_plus[mask])
    relative = (u_plus[mask] - reference) / reference
    return dict(low=float(low), high=float(high_fraction * re_tau), cells=int(np.count_nonzero(mask)),
                rms_relative=float(np.sqrt(np.mean(relative ** 2))),
                max_relative=float(np.max(np.abs(relative))),
                mean_relative=float(np.mean(relative)),
                dns_Cf=float(dns['Cf']), dns_re_tau=float(dns['re_tau']))


def dns_reynolds_shear(dns):
    """Turbulent shear stress ``-u'v'/u_tau^2`` implied by the DNS momentum balance.

    For a fully developed channel the mean momentum balance is exact:
    ``dU+/dy+ + (-u'v'/u_tau^2) = 1 - y/delta``.  With the DNS mean profile and
    its gradient both archived, the turbulent shear stress follows without any
    model: ``-u'v'+ = 1 - y/delta - dU+/dy+``.  This is the quantity a RANS
    closure replaces by ``nu_t dU/dy``, so the two are directly comparable.
    """
    shear = 1 - dns['y_over_delta'] - dns['dU_plus_dy_plus']
    if np.any(shear < -1e-3):
        raise ValueError('DNS momentum balance gives a negative turbulent shear stress')
    return np.maximum(shear, 0.)


def shear_stress_deviation(y_plus, modelled_shear, dns, *, low=30., high_fraction=.2, re_tau=None):
    """Normalised RMS error of the modelled turbulent shear stress against DNS.

    Normalisation is by the peak DNS turbulent shear stress inside the window, so
    the number is a fraction of the stress that carries the momentum flux, not a
    pointwise ratio that explodes where the DNS stress passes through a small
    value.
    """
    from scipy.interpolate import PchipInterpolator
    re_tau = dns['re_tau'] if re_tau is None else float(re_tau)
    y_plus = np.asarray(y_plus, dtype=float)
    modelled = np.asarray(modelled_shear, dtype=float)
    mask = (y_plus >= low) & (y_plus <= high_fraction * re_tau)
    if np.count_nonzero(mask) < 3:
        raise ValueError('log region holds fewer than three cells')
    reference = PchipInterpolator(dns['y_plus'], dns_reynolds_shear(dns))(y_plus[mask])
    scale = float(np.max(np.abs(reference)))
    if scale <= 0:
        raise ValueError('DNS shear stress vanishes inside the window')
    error = modelled[mask] - reference
    return dict(low=float(low), high=float(high_fraction * re_tau), cells=int(np.count_nonzero(mask)),
                normalised_rms=float(np.sqrt(np.mean(error ** 2)) / scale),
                normalised_max=float(np.max(np.abs(error)) / scale),
                dns_peak=scale)
