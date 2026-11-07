"""Published constant sets and pointwise equilibrium closures for three models.

Every closure is specialised to the equilibrium parallel-shear state used by the
validation matrix.  With unit density, unit friction velocity and the half
channel height as the length scale, the imposed pressure gradient fixes the
total shear

    (nu + nu_t) dU/dy = u_tau^2 (1 - y)                                   (1)

so the *same* call that evaluates the eddy viscosity also returns the shear that
solves (1).  This is what makes a 1-D channel a closed problem: the momentum
equation never has to be iterated.  Sources are per unit mass and are split into
a non-negative production and a sink that is linear in the transported variable,
so the implicit transport sweep stays positive without clipping the state.

Constant sets (read from this module only, never retyped downstream):

* ``sst``      -- SST-2003 with the exact production term, NASA TMR constants.
* ``komega``   -- Wilcox k-omega standard constants (alpha=13/25, beta=3/40,
  beta*=9/100, sigma=sigma*=1/2) including the ``sigma_d`` cross-diffusion term
  of the Wilcox 1998 form.  No low-Reynolds damping.
* ``kepsilon`` -- standard (Launder & Spalding 1974) high-Reynolds k-epsilon,
  ``sigma_epsilon`` *divides* the eddy viscosity in the epsilon flux.  This model
  has no viscous-sublayer solution and is therefore always paired with the
  equilibrium wall functions in :mod:`.channel`.

``implied_kappa`` returns the logarithmic-layer slope that the constant set
implies.  It is *not* a fit: substituting k = u_tau^2/sqrt(beta*),
omega = u_tau/(sqrt(beta*) kappa y) and the product-rule form of the diffusion
flux into the second transport equation and requiring the balance to vanish gives,
for k-omega type models,

    kappa^2 = (beta/sqrt(beta*) - gamma sqrt(beta*)) / sigma_omega           (2)

and, for the standard k-epsilon model,

    kappa^2 = (C2 - C1) sqrt(C_mu) sigma_epsilon .                           (3)

For SST-2003 (inner constants) (2) gives kappa = 0.408, for Wilcox k-omega
0.434 and for the standard k-epsilon 0.433; the wall law used by the rest of the
repository assumes 0.41.
"""
import numpy as np

MODELS = ('sst', 'komega', 'kepsilon')
VARIABLE = {'sst': 'omega', 'komega': 'omega', 'kepsilon': 'epsilon'}
WALL_TREATMENT = {'sst': 'low_re', 'komega': 'low_re', 'kepsilon': 'wall_function'}
WALL_LAW_KAPPA = .41

DEFAULT_CONSTANTS = {
    'sst': dict(beta_star=.09, beta1=.075, beta2=.0828, gamma1=5 / 9, gamma2=.44,
                sigma_k1=.85, sigma_k2=1., sigma_omega1=.5, sigma_omega2=.856,
                a1=.31, cd_floor=1e-10, production_limit=10.),
    'komega': dict(beta_star=.09, beta=.075, gamma=.52, sigma_k=.5, sigma_omega=.5,
                   sigma_d=.5, production_limit=10.),
    'kepsilon': dict(c_mu=.09, c1=1.44, c2=1.92, sigma_k=1., sigma_epsilon=1.3),
}


def _positive_finite(name, value):
    value = float(value)
    if not np.isfinite(value) or value <= 0:
        raise ValueError('%s must be positive and finite' % name)
    return value


def constants_for(model, overrides=None):
    """Constant set for ``model`` with optional, validated overrides."""
    if model not in MODELS:
        raise ValueError('unknown model %r; expected one of %s' % (model, MODELS))
    constants = dict(DEFAULT_CONSTANTS[model])
    if overrides:
        for name, value in overrides.items():
            if name not in constants:
                raise ValueError('unknown constant %r for model %r' % (name, model))
            constants[name] = _positive_finite(name, value)
    return constants


def implied_kappa(model, constants=None):
    """Log-layer slope implied by the constant set, closed forms (2) and (3)."""
    c = constants_for(model) if constants is None else dict(constants)
    if model == 'sst':
        root = np.sqrt(c['beta_star'])
        value = (c['beta1'] / root - c['gamma1'] * root) / c['sigma_omega1']
    elif model == 'komega':
        root = np.sqrt(c['beta_star'])
        value = (c['beta'] / root - c['gamma'] * root) / c['sigma_omega']
    else:
        value = (c['c2'] - c['c1']) * np.sqrt(c['c_mu']) * c['sigma_epsilon']
    if not value > 0:
        raise ValueError('constant set admits no logarithmic layer')
    return float(np.sqrt(value))


def wall_values(model, first_cell_distance, nu, constants=None):
    """Wall-adjacent state: ``(k, variable)`` for unit friction velocity.

    ``low_re`` models use their published near-wall conditions: k = 0 and
    omega = 60 nu/(beta1 d^2) for SST-2003 (TMR recommendation, ten times the
    asymptotic 6 nu/(beta d^2) to drive the modelled sublayer), and the exact
    Wilcox asymptotic omega = 6 nu/(beta d^2) for the undamped k-omega model.
    ``wall_function`` models take the equilibrium wall-function values
    k = u_tau^2/sqrt(C_mu) and epsilon = u_tau^3/(kappa y_1).
    """
    c = constants_for(model) if constants is None else dict(constants)
    d = _positive_finite('first cell distance', first_cell_distance)
    nu = _positive_finite('nu', nu)
    if WALL_TREATMENT[model] == 'wall_function':
        return 1 / np.sqrt(c['c_mu']), 1 / (WALL_LAW_KAPPA * d)
    if model == 'sst':
        return 0., 60 * nu / (c['beta1'] * d * d)
    return 0., 6 * nu / (c['beta'] * d * d)


def equilibrium_closure(model, y, k, variable, dk, dvariable, nu, constants=None, shear=None):
    """Closure state for equilibrium parallel shear, including shear from (1).

    ``y`` is the wall-normal distance (positive, half-channel heights), ``k`` the
    turbulence kinetic energy and ``variable`` omega or epsilon.  ``dk`` and
    ``dvariable`` are the wall-normal gradients used by the SST cross-diffusion
    and the Wilcox ``sigma_d`` term.  Returns per-unit-mass production/sink
    coefficients, diffusion coefficients and the eddy viscosity.

    ``shear`` optionally supplies the mean shear instead of solving (1).  That is
    what a manufactured log-layer state needs: the leading-order analysis assumes
    ``dU/dy = u_tau/(kappa y)``, whereas (1) carries the extra total-stress factor
    ``(1 - y)`` which is not part of the asymptotic balance.
    """
    c = constants_for(model) if constants is None else dict(constants)
    y, k, variable, nu = (np.asarray(a, dtype=float) for a in (y, k, variable, nu))
    if not (np.isfinite(y).all() and np.isfinite(k).all() and np.isfinite(variable).all()):
        raise ValueError('non-finite closure input')
    if np.any(y <= 0) or np.any(k < 0) or np.any(variable <= 0) or np.any(nu <= 0):
        raise ValueError('y, nu and variable must be positive, k non-negative')
    total = 1 - y if shear is None else np.asarray(shear, dtype=float)
    if model == 'sst':
        root = np.sqrt(c['beta_star'])
        a = np.sqrt(k) / (root * variable * y)
        b = 500 * nu / (y * y * variable)
        cd = np.maximum(2 * c['sigma_omega2'] * dk * dvariable / variable, c['cd_floor'])
        arg1 = np.minimum(np.minimum(np.maximum(a, b), 4 * c['sigma_omega2'] * k / (cd * y * y)), 10.)
        f1 = np.tanh(arg1 ** 4)
        f2 = np.tanh(np.minimum(np.maximum(2 * a, b), 20.) ** 2)
        blend = lambda inner, outer: f1 * inner + (1 - f1) * outer
        beta = blend(c['beta1'], c['beta2'])
        gamma = blend(c['gamma1'], c['gamma2'])
        sigma_k = blend(c['sigma_k1'], c['sigma_k2'])
        sigma_omega = blend(c['sigma_omega1'], c['sigma_omega2'])
        # Solve the eddy-viscosity limiter and (1) exactly: the shear is bounded
        # above by a1*omega/F2, so exceeding that bound switches to the branch
        # where nu_t is fixed by the limiter and the shear is measured directly.
        shear = total / (nu + k / variable) if shear is None else total
        limited = shear * f2 > c['a1'] * variable
        if np.any(limited):
            shear = np.where(limited, (total - c['a1'] * k / f2) / nu, shear)
        if np.any(shear < 0):
            raise ValueError('stress limiter leaves a negative shear; state inadmissible')
        denominator = np.maximum(c['a1'] * variable, shear * f2)
        nut = c['a1'] * k / denominator
        production = np.minimum(nut * shear ** 2, c['production_limit'] * c['beta_star'] * variable * k)
        variable_production = gamma * np.minimum(shear ** 2, c['production_limit'] * c['beta_star'] * variable * denominator / c['a1'])
        cross = 2 * (1 - f1) * c['sigma_omega2'] * dk * dvariable / variable
        return dict(nut=nut, shear=shear, F1=f1, F2=f2, beta=beta, gamma=gamma,
                    k_production=production, k_sink=c['beta_star'] * variable,
                    variable_production=variable_production + cross,
                    variable_sink=beta * variable,
                    k_diffusion=nu + sigma_k * nut, variable_diffusion=nu + sigma_omega * nut,
                    cross_diffusion=cross)
    if model == 'komega':
        nut = k / variable
        shear = total / (nu + nut) if shear is None else total
        production = np.minimum(nut * shear ** 2, c['production_limit'] * c['beta_star'] * variable * k)
        cross = c['sigma_d'] * dk * dvariable / np.maximum(variable, 1e-300)
        return dict(nut=nut, shear=shear,
                    k_production=production, k_sink=c['beta_star'] * variable,
                    variable_production=c['gamma'] * production / np.maximum(nut, 1e-300) + cross,
                    variable_sink=c['beta'] * variable,
                    k_diffusion=nu + c['sigma_k'] * nut, variable_diffusion=nu + c['sigma_omega'] * nut,
                    cross_diffusion=cross)
    nut = c['c_mu'] * k * k / variable
    shear = total / (nu + nut) if shear is None else total
    production = nut * shear ** 2
    ratio = variable / np.maximum(k, 1e-300)
    return dict(nut=nut, shear=shear,
                k_production=production, k_sink=ratio,
                variable_production=c['c1'] * ratio * production,
                variable_sink=c['c2'] * ratio,
                k_diffusion=nu + nut / c['sigma_k'], variable_diffusion=nu + nut / c['sigma_epsilon'],
                cross_diffusion=np.zeros_like(nut))