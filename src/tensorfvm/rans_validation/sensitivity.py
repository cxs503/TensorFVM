"""Model-constant sensitivity of the predicted friction coefficient.

The graded question is the robustness criterion of the validation matrix: if the
turbulence model's ``C_mu``-equivalent constant is deliberately mis-set by ten
per cent, how far does the converged skin-friction coefficient move?  The
``C_mu``-equivalent is ``beta*`` for the k-omega family (``nu_t = C_mu k^2/eps``
transforms into ``nu_t = k/omega`` with ``C_mu = beta*``) and ``C_mu`` itself for
standard k-epsilon.

Only the scanned constant is perturbed; every other constant, the discretisation,
the Reynolds number (fixed through ``Re_tau``) and the wall conditions stay
identical, so the reported change isolates the model constant.
"""
import numpy as np

from .channel import solve_channel
from .models import MODELS, constants_for

CMU_EQUIVALENT = {'sst': 'beta_star', 'komega': 'beta_star', 'kepsilon': 'c_mu'}


def constant_sweep(model, *, re_tau=180., n=256, changes=(-.2, -.1, 0., .1, .2), constant=None,
                   first_cell_y_plus=None, tolerance=1e-10):
    """Sweep one constant and report the change of ``Cf`` relative to the base run."""
    if model not in MODELS:
        raise ValueError('unknown model %r' % model)
    name = CMU_EQUIVALENT[model] if constant is None else constant
    base_constants = constants_for(model)
    if name not in base_constants:
        raise ValueError('unknown constant %r for model %r' % (name, model))
    base_value = base_constants[name]
    rows = []
    for change in changes:
        overrides = {name: base_value * (1 + change)}
        result = solve_channel(model, n, re_tau=re_tau, constants=overrides, first_cell_y_plus=first_cell_y_plus,
                               tolerance=tolerance)
        rows.append(dict(change=float(change), constant_value=float(base_value * (1 + change)),
                         Cf=float(result['Cf']), bulk_u_plus=float(result['bulk_u_plus']),
                         converged=bool(result['converged'])))
    reference = next((row for row in rows if row['change'] == 0.), None)
    if reference is None:
        reference = dict(Cf=float(solve_channel(model, n, re_tau=re_tau, first_cell_y_plus=first_cell_y_plus,
                                                tolerance=tolerance)['Cf']))
    for row in rows:
        row['Cf_relative_change'] = float(row['Cf'] / reference['Cf'] - 1)
    graded = [abs(row['Cf_relative_change']) for row in rows if abs(row['change']) <= .1 + 1e-12]
    return dict(model=model, constant=name, base_value=float(base_value), base_Cf=float(reference['Cf']),
                re_tau=float(re_tau), n=int(n), rows=rows,
                max_abs_change_within_ten_percent=float(max(graded)),
                sensitivity_dCfdC_over_Cf=float(np.polyfit([row['change'] for row in rows],
                                                           [row['Cf_relative_change'] for row in rows], 1)[0]),
                all_converged=bool(all(row['converged'] for row in rows)))