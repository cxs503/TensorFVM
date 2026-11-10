"""DES/IDDES hybrid RANS-LES length-scale limiter for SST k-omega.

Implements two hybrid RANS-LES approaches on top of the existing SST-2003
closure (native_sst3d.py):

1. DES97 (Spalart et al. 1997):
   L_DES = min(L_SST, C_DES * Delta)
   where L_SST = sqrt(k)/(beta_star * omega) is the SST length scale,
   Delta = max(dx, dy, dz) is the largest grid spacing, and C_DES is the
   DES model constant (C_DES_1 = 0.78 for SST, calibrated from log-law).

2. IDDES (Shur et al. 2008):
   Improved Delayed DES that protects the RANS boundary layer from premature
   switching by using a wall-distance-based shielding function:
   L_IDDES = l_RANS * (1 - f_d) + l_LES * f_d
   where f_d = 1 - tanh((8 * r_d)^3), r_d = (nu_t + nu) / (S * kappa^2 * d^2)
   and l_RANS = sqrt(k)/(beta_star * omega), l_LES = C_DES * Delta.

The limiter modifies the destruction term in the k-equation:
   D_k = rho * beta_star * omega * k  -->  rho * k^(3/2) / L_hybrid

This module provides:
- des97_length_scale(): pure DES97 length scale
- iddes_length_scale(): IDDES length scale with shielding
- des_shielding_function(): IDDES shielding function f_d
- modified_sst_destruction(): modified k-destruction for DES/IDDES
- HybridSSTDES class: wrapper that applies DES/IDDES to SST closure

References:
- Spalart et al. (1997) "A one-equation turbulence model for aerodynamic flows"
  (La Recherche Aerospatiale) - DES97
- Strelets (2001) "Detached Eddy Simulation of Massively Separated Flows"
  AIAA 2001-0879 - DES calibration
- Shur et al. (2008) "RANS-based delayed detached-eddy simulation of turbulent
  flow with application to aeroacoustics" Proc. 16th AIAA/CEAS Aeroacoustics
- Menter et al. (2003) "A SST-2003 based IDDES model" - SST variant

Scope: This module provides the length-scale modification. It does NOT
implement a full time-accurate LES solver; the underlying RANS infrastructure
provides the mean flow, and the DES limiter reduces turbulent dissipation in
separated regions to allow resolved unsteady structures.
"""
from __future__ import annotations
import numpy as np
from .native_sst3d import BETA_STAR, sst2003_closure

# DES constants for SST-based DES (calibrated from log-layer)
C_DES_SST_1 = 0.78   # Inner layer (SST blend F1=1)
C_DES_SST_2 = 0.61   # Outer layer (SST blend F1=0)
KAPPA = 0.41          # von Karman constant


def _validate_positive(name, value):
    """Check finite and positive."""
    v = np.asarray(value, dtype=float)
    if not np.isfinite(v).all():
        raise ValueError(f'{name} must be finite')
    if np.any(v <= 0):
        raise ValueError(f'{name} must be positive')
    return v


def des97_length_scale(k, omega, wall_distance, cell_size, *, nu=0., rho=1.,
                       velocity_gradient=None):
    """DES97 hybrid length scale: min(L_RANS, C_DES * Delta).

    Parameters
    ----------
    k : array
        Turbulent kinetic energy [m^2/s^2], k >= 0.
    omega : array
        Specific dissipation rate [1/s], omega > 0.
    wall_distance : array
        Nearest wall distance [m], d > 0.
    cell_size : array
        Maximum grid spacing Delta = max(dx, dy, dz) [m].
    nu : float or array
        Kinematic viscosity [m^2/s].
    rho : float or array
        Density [kg/m^3].
    velocity_gradient : array, optional
        Velocity gradient tensor for C_DES blending (uses F1 if available).

    Returns
    -------
    dict with keys:
        L_RANS: SST RANS length scale sqrt(k)/(beta_star*omega)
        L_LES: LES length scale C_DES * Delta
        L_DES: DES97 length scale min(L_RANS, L_LES)
        C_DES: blended DES constant
        ratio: L_DES / L_RANS (1 = pure RANS, <1 = LES mode)
    """
    k = np.asarray(k, dtype=float)
    omega = np.asarray(omega, dtype=float)
    d = _validate_positive('wall_distance', wall_distance)
    Delta = _validate_positive('cell_size', cell_size)

    if np.any(k < 0):
        raise ValueError('k must be non-negative')
    if np.any(omega <= 0):
        raise ValueError('omega must be positive')

    # RANS length scale
    L_RANS = np.sqrt(np.maximum(k, 0.0)) / (BETA_STAR * omega)

    # Blend C_DES using F1 from SST if velocity_gradient is provided
    if velocity_gradient is not None:
        F1 = _sst_f1(k, omega, d, nu, velocity_gradient)
        C_DES = F1 * C_DES_SST_1 + (1 - F1) * C_DES_SST_2
    else:
        C_DES = C_DES_SST_1  # Default to inner-layer value

    L_LES = C_DES * Delta

    # DES97: take minimum
    L_DES = np.minimum(L_RANS, L_LES)
    ratio = np.where(L_RANS > 0, L_DES / np.maximum(L_RANS, 1e-30), 1.0)

    return dict(L_RANS=L_RANS, L_LES=L_LES, L_DES=L_DES,
                C_DES=C_DES, ratio=ratio)


def iddes_shielding_function(k, omega, wall_distance, nu_t, nu, velocity_gradient):
    """IDDES shielding function f_d that protects the boundary layer.

    f_d = 1 - tanh((8 * r_d)^3)
    r_d = (nu_t + nu) / (S * kappa^2 * d^2)

    When f_d ≈ 0: RANS mode (boundary layer protected)
    When f_d ≈ 1: LES mode (separated region)

    Parameters
    ----------
    k, omega : arrays
        SST turbulence variables.
    wall_distance : array
        Nearest wall distance d > 0.
    nu_t : array
        Turbulent (eddy) viscosity [m^2/s].
    nu : float or array
        Molecular kinematic viscosity [m^2/s].
    velocity_gradient : array (..., 3, 3)
        Velocity gradient tensor.

    Returns
    -------
    dict with keys:
        f_d: shielding function [0, 1]
        r_d: nondimensional ratio
        S: strain rate magnitude
    """
    d = _validate_positive('wall_distance', wall_distance)
    nu_t = np.asarray(nu_t, dtype=float)
    nu = np.broadcast_to(np.asarray(nu, dtype=float), d.shape)
    G = np.asarray(velocity_gradient, dtype=float)

    # Strain rate magnitude
    S_tensor = 0.5 * (G + G.swapaxes(-1, -2))
    S = np.sqrt(2.0 * np.sum(S_tensor * S_tensor, axis=(-2, -1)))
    S = np.maximum(S, 1e-10)  # Avoid division by zero

    # r_d ratio
    r_d = (nu_t + nu) / (S * KAPPA**2 * d**2)
    r_d = np.clip(r_d, 0.0, 1e10)  # Prevent overflow

    # Shielding function
    f_d = 1.0 - np.tanh((8.0 * r_d)**3)
    f_d = np.clip(f_d, 0.0, 1.0)

    return dict(f_d=f_d, r_d=r_d, S=S)


def iddes_length_scale(k, omega, wall_distance, cell_size, nu_t, nu,
                       velocity_gradient):
    """IDDES hybrid length scale with boundary-layer shielding.

    l_hybrid = l_RANS * (1 - f_d) + l_LES * f_d
    but with the additional constraint that l_hybrid <= l_RANS when
    the wall distance is small relative to the grid (RANS protection).

    Parameters
    ----------
    Same as des97_length_scale plus nu_t and velocity_gradient.

    Returns
    -------
    dict with keys:
        L_RANS, L_LES, L_IDDES, f_d, r_d, C_DES, ratio
    """
    k = np.asarray(k, dtype=float)
    omega = np.asarray(omega, dtype=float)
    d = _validate_positive('wall_distance', wall_distance)
    Delta = _validate_positive('cell_size', cell_size)

    if np.any(k < 0):
        raise ValueError('k must be non-negative')
    if np.any(omega <= 0):
        raise ValueError('omega must be positive')

    # RANS length scale
    L_RANS = np.sqrt(np.maximum(k, 0.0)) / (BETA_STAR * omega)

    # Shielding function
    shield = iddes_shielding_function(k, omega, d, nu_t, nu, velocity_gradient)
    f_d = shield['f_d']

    # Blend C_DES
    F1 = _sst_f1(k, omega, d, nu, velocity_gradient)
    C_DES = F1 * C_DES_SST_1 + (1 - F1) * C_DES_SST_2
    L_LES = C_DES * Delta

    # IDDES length scale
    L_IDDES = L_RANS * (1.0 - f_d) + L_LES * f_d

    # Additional wall protection: ensure L_IDDES <= L_RANS near walls
    # when Delta > d (prevents premature LES switching)
    wall_protection = (Delta > d) & (f_d > 0)
    L_IDDES = np.where(wall_protection,
                       np.minimum(L_IDDES, L_RANS),
                       L_IDDES)

    ratio = np.where(L_RANS > 0, L_IDDES / np.maximum(L_RANS, 1e-30), 1.0)

    return dict(L_RANS=L_RANS, L_LES=L_LES, L_IDDES=L_IDDES,
                f_d=f_d, r_d=shield['r_d'], C_DES=C_DES, ratio=ratio,
                S=shield['S'])


def modified_k_destruction(k, L_hybrid, rho=1.0):
    """Modified k-destruction term using hybrid length scale.

    Standard SST: D_k = rho * beta_star * omega * k
    DES/IDDES:    D_k = rho * k^(3/2) / L_hybrid

    When L_hybrid = L_RANS = sqrt(k)/(beta_star*omega), these are identical.
    When L_hybrid < L_RANS (LES mode), destruction increases, reducing k
    and allowing resolved fluctuations.

    Parameters
    ----------
    k : array
        Turbulent kinetic energy.
    L_hybrid : array
        Hybrid length scale from DES97 or IDDES.
    rho : float
        Density.

    Returns
    -------
    array: Modified destruction rate per unit volume.
    """
    k = np.asarray(k, dtype=float)
    L = _validate_positive('L_hybrid', L_hybrid)
    rho = np.broadcast_to(np.asarray(rho, dtype=float), k.shape)

    # D_k = rho * k^(3/2) / L_hybrid
    # Use maximum(k, 0) to handle k=0 safely
    k_safe = np.maximum(k, 0.0)
    D_k = rho * k_safe * np.sqrt(k_safe) / L

    return D_k


def _sst_f1(k, omega, wall_distance, nu, velocity_gradient):
    """Compute SST blending function F1 (simplified, for C_DES blending).

    This is a lightweight version of the F1 from sst2003_closure, used only
    for blending the DES constant.
    """
    from .native_sst3d import BETA1, SIGMA_W2
    k = np.asarray(k, dtype=float)
    omega = np.asarray(omega, dtype=float)
    d = np.asarray(wall_distance, dtype=float)
    nu = np.broadcast_to(np.asarray(nu, dtype=float), d.shape)
    G = np.asarray(velocity_gradient, dtype=float)

    S2 = 2.0 * np.sum(
        (0.5 * (G + G.swapaxes(-1, -2)))**2, axis=(-2, -1))

    arg2 = np.maximum(2.0 * np.sqrt(k) / (BETA_STAR * omega * d),
                      500.0 * nu / (d**2 * omega))
    F2 = np.tanh(np.minimum(arg2, 20.0)**2)

    # For F1 we need the full argument; use simplified form
    arg1_inner = np.sqrt(k) / (BETA_STAR * omega * d)
    arg1_wall = 500.0 * nu / (d**2 * omega)
    arg1 = np.minimum(np.maximum(arg1_inner, arg1_wall), 10.0)
    F1 = np.tanh(arg1**4)

    return F1


class HybridSSTDES:
    """Wrapper that applies DES/IDDES length-scale limiting to SST closure.

    Usage:
        des = HybridSSTDES(mode='iddes')
        result = des.apply(k, omega, wall_distance, cell_size, nu_t, nu,
                          velocity_gradient)
        # result contains modified destruction and length scales

    The apply() method returns the modified k-destruction rate that should
    replace the standard SST destruction in the k-transport equation.
    """

    MODES = ('des97', 'iddes')

    def __init__(self, mode='iddes'):
        if mode not in self.MODES:
            raise ValueError(f'Unknown DES mode: {mode}. Choose from {self.MODES}')
        self.mode = mode

    def apply(self, k, omega, wall_distance, cell_size, nu_t, nu,
              velocity_gradient, rho=1.0):
        """Compute DES/IDDES-modified k-destruction.

        Returns
        -------
        dict with:
            D_k_modified: modified destruction rate
            D_k_standard: standard SST destruction (for comparison)
            L_hybrid: hybrid length scale
            L_RANS: SST RANS length scale
            ratio: L_hybrid / L_RANS
            mode: 'des97' or 'iddes'
            f_d: IDDES shielding function (only for iddes mode)
        """
        k = np.asarray(k, dtype=float)
        omega = np.asarray(omega, dtype=float)

        # Standard SST destruction
        D_k_standard = rho * BETA_STAR * omega * k

        if self.mode == 'des97':
            ls = des97_length_scale(k, omega, wall_distance, cell_size,
                                    nu=nu, rho=rho,
                                    velocity_gradient=velocity_gradient)
            L_hybrid = ls['L_DES']
            f_d = None
        else:  # iddes
            ls = iddes_length_scale(k, omega, wall_distance, cell_size,
                                    nu_t, nu, velocity_gradient)
            L_hybrid = ls['L_IDDES']
            f_d = ls['f_d']

        D_k_modified = modified_k_destruction(k, L_hybrid, rho=rho)

        result = dict(
            D_k_modified=D_k_modified,
            D_k_standard=D_k_standard,
            L_hybrid=L_hybrid,
            L_RANS=ls['L_RANS'],
            L_LES=ls['L_LES'],
            ratio=ls['ratio'],
            C_DES=ls['C_DES'],
            mode=self.mode,
        )
        if f_d is not None:
            result['f_d'] = f_d
            result['r_d'] = ls['r_d']

        return result

    def effective_nut(self, k, omega, wall_distance, cell_size, nu_t_rans,
                      nu, velocity_gradient, rho=1.0):
        """Compute effective eddy viscosity with DES/IDDES limiting.

        In LES mode, the eddy viscosity is reduced proportionally to the
        length scale ratio: nut_eff = nut_RANS * (L_hybrid / L_RANS)^2

        This follows the standard DES practice where mu_t ~ rho * k^(1/2) * L.
        """
        result = self.apply(k, omega, wall_distance, cell_size, nu_t_rans,
                           nu, velocity_gradient, rho)
        ratio = result['ratio']
        nut_eff = nu_t_rans * ratio**2
        return nut_eff, result
