"""Ice-steel conjugate heat transfer for hull plating and superstructure decks.

Two layers of very different conductivity sit on either side of one cell face:
steel (``k = 50 W/(m K)``) and ice (``k = 2.2 W/(m K)``), a ratio of 23.  This
module solves the resulting temperature field two ways, on purpose:

``layered_field``       single-field finite volume on a Cartesian grid that puts a
                        grid line exactly on the ice/steel interface, with the
                        distance-weighted harmonic face conductance of
                        :mod:`tensorfvm.cht.energy`.  Heat flux continuity then
                        holds *by construction*; the audit re-derives the two
                        one-sided fluxes from the solved field to prove it and to
                        catch a future regression.
``PartitionedIceSteel`` the two layers are solved **separately** (steel as a block
                        with a Dirichlet interface temperature, ice as a block with
                        a Neumann interface flux) and coupled by a
                        Dirichlet-Neumann iteration.  This is the discretisation a
                        production CHT code actually uses when the two materials
                        live in different solvers, and it is *not* conservative by
                        construction: the residual ``|q_steel - q_ice| / |q|``
                        measures how well the coupling converged.  It is the
                        quantity the verification uses as the interface-flux
                        residual, because it can actually fail.

On top of the conduction problem sits the icing side: a Makiranta/Lozowski-type
five-term surface heat balance (convection, evaporation/sublimation, long-wave
radiation, spray sensible heat, freezing latent heat) evaluated on the *same*
environmental inputs as the W4-N08 marine-icing model, so the two models can be
adjudicated against each other at identical conditions.  The five terms are
implemented here from the published correlations, independently of W4-N08's code;
the archived W4-N08 numbers are carried as *reference data*, never as a substitute
for a computation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .energy import Grid2D, solve_steady_energy
from .stefan import ICE_C, ICE_K, ICE_LATENT, ICE_RHO

__all__ = [
    "STEEL_K", "WATER_K", "STEEL_DENSITY", "STEEL_C",
    "layered_interface_analytic", "layered_field", "interface_flux_audit",
    "PartitionedIceSteel", "surface_heat_loss", "ice_growth_from_balance",
    "W4N08_REFERENCE_PANEL",
]

#: material data
STEEL_K = 50.0            # W/(m K)   marine structural steel
STEEL_DENSITY = 7850.0    # kg/m^3
STEEL_C = 490.0           # J/(kg K)
WATER_K = 0.56            # W/(m K)   sea water at ~0 C
AIR_K = 0.0236            # W/(m K)   air at -10 C
AIR_RHO = 1.342           # kg/m^3
AIR_CP = 1005.0           # J/(kg K)
AIR_NU = 1.296e-5         # m^2/s     air at -10 C
AIR_PR = 0.71
LV_VAPOUR = 2.834e6       # J/kg      latent heat of sublimation at ~0 C
LB_FUSION_SEAWATER = 3.337e5   # J/kg  latent heat of fusion (fresh-ice proxy)


# ------------------------------------------------------------------ analytics
def layered_interface_analytic(*, d_steel: float, d_ice: float, t_inner: float,
                               t_outer: float, k_steel: float = STEEL_K,
                               k_ice: float = ICE_K, h_inner: float | None = None,
                               h_outer: float | None = None) -> dict:
    """Exact 1-D series-resistance solution of the two-layer slab.

    ``h_inner`` / ``h_outer`` are optional convective conductances at the two ends
    (``None`` = Dirichlet).  Returns the interface temperature, the common flux
    and the individual resistances, so the discrete solution can be judged against
    a closed-form value rather than against another computation.
    """
    r_inner = 0.0 if h_inner is None else 1.0 / h_inner
    r_steel = d_steel / k_steel
    r_ice = d_ice / k_ice
    r_outer = 0.0 if h_outer is None else 1.0 / h_outer
    total = r_inner + r_steel + r_ice + r_outer
    q = (t_inner - t_outer) / total
    t_iface = t_inner - q * (r_inner + r_steel)
    return {
        "q_W_m2": q,
        "interface_temperature": t_iface,
        "resistance_m2K_W": {"inner": r_inner, "steel": r_steel, "ice": r_ice,
                             "outer": r_outer, "total": total},
    }


# ------------------------------------------------------------- single field FV
def layered_field(*, d_steel: float, d_ice: float, t_inner, t_outer, nx_steel: int = 20,
                  nx_ice: int = 30, nz: int = 1, k_steel: float = STEEL_K,
                  k_ice: float = ICE_K, h_inner: float | None = None):
    """Solve the two-layer conduction problem on one grid (heat flux continuous).

    ``t_inner`` / ``t_outer`` may be scalars (1-D) or arrays of length ``nz``
    (2-D: a strip of plate with a span-wise varying surface temperature).  The
    grid places a face exactly on the interface, so the harmonic-mean conductance
    acts across a steel/ice pair of cells and the flux is a single unknown.

    ``h_inner`` turns the inner boundary into a convective (Robin) condition.  It
    is realised by prepending one ghost layer whose resistance ``d/k_steel``
    equals ``1/h_inner`` and holding its outer face at ``t_inner``: the discrete
    operator is untouched, and the exact solution of that extended stack is the
    analytic series resistance.
    """
    t_inner = np.broadcast_to(np.asarray(t_inner, dtype=float), (nz,))
    t_outer = np.broadcast_to(np.asarray(t_outer, dtype=float), (nz,))
    d_ghost = 0.0 if h_inner is None else k_steel / float(h_inner)
    n_steel_cells = nx_steel + (1 if d_ghost > 0.0 else 0)
    pieces = []
    if d_ghost > 0.0:
        pieces.append(np.zeros(1))
    pieces.append(np.linspace(0.0, d_steel, nx_steel + 1) + d_ghost)
    pieces.append(np.linspace(0.0, d_ice, nx_ice + 1)[1:] + d_ghost + d_steel)
    x_faces = np.concatenate(pieces)
    z_faces = np.linspace(0.0, 1.0, nz + 1)
    grid = Grid2D(x_faces, z_faces)
    kappa = np.empty((nz, grid.nx))
    kappa[:, :n_steel_cells] = k_steel
    kappa[:, n_steel_cells:] = k_ice
    zero_e = np.zeros((nz, grid.nx + 1))
    zero_n = np.zeros((nz + 1, grid.nx))
    bc = {
        "west": {"type": "dirichlet", "values": t_inner},
        "east": {"type": "dirichlet", "values": t_outer},
    }
    field, info = solve_steady_energy(grid.dx, grid.dy, kappa, zero_e, zero_n, bc,
                                      t_init=None)
    if nz == 1:
        field = field[0]
    return {
        "grid": grid,
        "x": grid.xc,
        "T": field,
        "kappa": kappa,
        "interface_cell": n_steel_cells - 1,   # steel-side cell of the interface face
        "nx_steel": n_steel_cells,
        "dx": grid.dx,
        "ledger": info["ledger"],
        "d_steel": d_steel,
        "d_ice": d_ice,
        "h_inner": h_inner,
        "t_inner": t_inner,
        "t_outer": t_outer,
    }


def interface_flux_audit(field, *, k_steel: float = STEEL_K,
                         k_ice: float = ICE_K) -> dict:
    """Re-derive both one-sided interface fluxes from the solved field.

    The face temperature is reconstructed from the two adjacent cell centres with
    the same distance weighting that the face conductance used, then each side is
    differentiated with *its own* conductivity and half-cell distance.  For the
    single-field discretisation the two values agree to round-off: that is the
    point of the check (a regression that broke the harmonic mean would show up
    here immediately), and the number is reported rather than assumed.
    """
    dx = field["dx"]
    i = field["interface_cell"]                     # steel cell, face at its east
    s_steel = 0.5 * dx[i]
    s_ice = 0.5 * dx[i + 1]
    t_all = np.atleast_2d(field["T"])
    t_p = t_all[:, i]
    t_n = t_all[:, i + 1]
    # interface temperature consistent with the harmonic face conductance:
    # q = (T_p - T_n) / (s_p/k_p + s_n/k_n), then each side is differentiated with
    # its own conductivity.  A plain distance interpolation of T would put the
    # face value in the wrong place exactly when k jumps, which is the error this
    # audit is designed to expose.
    q_face = (t_p - t_n) / (s_steel / k_steel + s_ice / k_ice)   # +x = toward air
    t_face = t_p - q_face * s_steel / k_steel
    q_steel = k_steel * (t_p - t_face) / s_steel
    q_ice = k_ice * (t_face - t_n) / s_ice
    scale = max(float(np.max(np.abs(q_steel))), 1e-30)
    local = np.abs(q_steel - q_ice) / np.maximum(np.abs(q_steel), 1e-12)
    return {
        "interface_temperature": float(np.mean(t_face)),
        "interface_temperature_profile": t_face.tolist(),
        "q_face_profile_W_m2": np.asarray(q_face, dtype=float).tolist(),
        "q_steel_W_m2": float(np.mean(q_steel)),
        "q_ice_W_m2": float(np.mean(q_ice)),
        "local_residual_max": float(np.max(local)),
        "global_residual": float(abs(np.sum(q_steel) - np.sum(q_ice)) / scale),
        "interface_area_mean_flux": float(np.mean(0.5 * (q_steel + q_ice))),
        "q_face_W_m2": float(np.mean(q_face)),
    }


# ------------------------------------------------------- partitioned coupling
def _solve_block(a_sub, a_diag, a_sup, rhs):
    """Thomas solve (small blocks only: the partitioned case is 1-D)."""
    n = a_diag.size
    cp = np.empty(n)
    dp = np.empty(n)
    cp[0] = a_sup[0] / a_diag[0]
    dp[0] = rhs[0] / a_diag[0]
    for i in range(1, n):
        m = a_diag[i] - a_sub[i] * cp[i - 1]
        cp[i] = a_sup[i] / m
        dp[i] = (rhs[i] - a_sub[i] * dp[i - 1]) / m
    x = np.empty(n)
    x[-1] = dp[-1]
    for i in range(n - 2, -1, -1):
        x[i] = dp[i] - cp[i] * x[i + 1]
    return x


@dataclass
class PartitionedResult:
    iterations: int
    converged: bool
    interface_temperature: float
    q_steel: float
    q_ice: float
    flux_residual: float
    energy_residual: float
    history: list = field(default_factory=list)
    temperatures: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "iterations": self.iterations,
            "converged": self.converged,
            "interface_temperature": self.interface_temperature,
            "q_steel_W_m2": self.q_steel,
            "q_ice_W_m2": self.q_ice,
            "flux_residual": self.flux_residual,
            "energy_residual": self.energy_residual,
            "history_tail": self.history[-5:],
        }


class PartitionedIceSteel:
    """Separate steel and ice blocks coupled by a Dirichlet-Neumann iteration.

    Steel block: Robin (``h_water``, ``t_water``) at the sea side and a Neumann
    interface flux, solved for the temperature field.  Ice block: Dirichlet
    interface temperature and Robin (``h_air``, ``t_air``) at the air side, solved
    for its field and hence for the interface flux.

    That assignment direction matters.  Exchanging the flux from the steel block
    into a Dirichlet ice block diverges as ``R_ice / R_steel`` (here ~20), whereas
    feeding the ice block first and closing with the steel block contracts by
    ``R_steel / R_ice``.  The verification reports the contraction it actually
    observes instead of trusting the argument.

    The residual that is reported is the *change of the interface flux in the last
    coupling sweep*: it is a property of the coupled iteration, not an identity of
    the discretisation, so a loose tolerance or a broken conductance makes it
    large.
    """

    def __init__(self, *, d_steel: float = 0.020, d_ice: float = 0.030,
                 n_steel: int = 20, n_ice: int = 30, t_water: float = -1.8,
                 t_air: float = -10.0, h_water: float = 500.0, h_air: float = 25.0,
                 k_steel: float = STEEL_K, k_ice: float = ICE_K):
        self.d_steel, self.d_ice = float(d_steel), float(d_ice)
        self.n_steel, self.n_ice = int(n_steel), int(n_ice)
        self.t_water, self.t_air = float(t_water), float(t_air)
        self.h_water, self.h_air = float(h_water), float(h_air)
        self.k_steel, self.k_ice = float(k_steel), float(k_ice)
        self.dx_s = self.d_steel / self.n_steel
        self.dx_i = self.d_ice / self.n_ice

    # --- blocks ---------------------------------------------------------
    def _steel(self, q_interface):
        """Robin at the sea side, Neumann flux ``q_interface`` leaving at +x."""
        n, dx, k = self.n_steel, self.dx_s, self.k_steel
        g = k / (dx * dx)
        gw = 1.0 / (1.0 / self.h_water + 0.5 * dx / k)     # sea-side conductance
        a = np.full(n, -g)
        b = np.full(n, 2.0 * g)
        c = np.full(n, -g)
        rhs = np.zeros(n)
        b[0] = g + gw / dx
        rhs[0] = gw * self.t_water / dx
        a[0] = 0.0
        b[-1] = g
        rhs[-1] = -q_interface / dx
        c[-1] = 0.0
        return _solve_block(a, b, c, rhs)

    def _ice(self, t_interface):
        """Dirichlet at the interface (west), Robin at the air side (east)."""
        n, dx, k = self.n_ice, self.dx_i, self.k_ice
        g = k / (dx * dx)
        ge = 1.0 / (1.0 / self.h_air + 0.5 * dx / k)       # air-side conductance
        a = np.full(n, -g)
        b = np.full(n, 2.0 * g)
        c = np.full(n, -g)
        rhs = np.zeros(n)
        # west half-cell Dirichlet: its conductance is 2g, plus the interior east
        # face g, so the diagonal of the first row is 3g (not 2g)
        b[0] = 3.0 * g
        rhs[0] = 2.0 * g * t_interface
        a[0] = 0.0
        b[-1] = g + ge / dx
        rhs[-1] = ge * self.t_air / dx
        c[-1] = 0.0
        return _solve_block(a, b, c, rhs)

    # --- coupling -------------------------------------------------------
    def solve(self, *, relax: float = 0.8, max_iter: int = 100,
              tol: float = 1e-10, q_initial: float = 0.0) -> PartitionedResult:
        # deliberately *no* series-resistance initial guess: starting from the
        # closed-form answer would make any coupling look converged in one sweep
        q = float(q_initial)
        t_interface = float(np.mean([self.t_water, self.t_air]))
        history = []
        converged = False
        t_steel = self._steel(q)
        t_ice = self._ice(t_interface)
        for it in range(1, max_iter + 1):
            t_steel = self._steel(q)
            t_iface_steel = t_steel[-1] - 0.5 * self.dx_s * q / self.k_steel
            t_ice = self._ice(t_iface_steel)
            q_new = 2.0 * self.k_ice * (t_iface_steel - t_ice[0]) / self.dx_i
            delta = q_new - q
            history.append((it, float(t_iface_steel), float(q), float(delta)))
            q = q + relax * delta
            t_interface = t_iface_steel
            if abs(delta) <= tol * max(abs(q), 1e-30):
                converged = True
                break
        # final consolidated state
        t_steel = self._steel(q)
        t_iface_steel = t_steel[-1] - 0.5 * self.dx_s * q / self.k_steel
        t_ice = self._ice(t_iface_steel)
        q_new = 2.0 * self.k_ice * (t_iface_steel - t_ice[0]) / self.dx_i
        scale = max(abs(q), abs(q_new), 1e-30)
        # wall (not cell-centre) temperatures, so the two end fluxes are exact
        t_steel_wall = t_steel[0] + 0.5 * self.dx_s * q / self.k_steel
        t_ice_wall = t_ice[-1] - 0.5 * self.dx_i * q_new / self.k_ice
        q_water = self.h_water * (self.t_water - t_steel_wall)
        q_air = self.h_air * (t_ice_wall - self.t_air)
        energy_res = abs(q_water - q_air) / max(abs(q_water), abs(q_air), 1e-30)
        return PartitionedResult(
            iterations=len(history), converged=converged,
            interface_temperature=float(t_iface_steel), q_steel=float(q),
            q_ice=float(q_new), flux_residual=float(abs(q_new - q) / scale),
            energy_residual=float(energy_res),
            history=history,
            temperatures={"steel": t_steel.tolist(), "ice": t_ice.tolist(),
                          "q_water_W_m2": float(q_water),
                          "q_air_W_m2": float(q_air)},
        )


def _saturation_density_ice(t_celsius: float) -> float:
    """Saturation vapour density over ice (Magnus form, kg/m^3)."""
    e_hpa = 6.112 * math.exp(22.46 * t_celsius / (272.62 + t_celsius))
    return e_hpa * 100.0 / (461.5 * (t_celsius + 273.15))


def _sky_longwave(t_air_c: float) -> float:
    """Downward long-wave irradiance, Idso-Jackson clear sky (W/m^2)."""
    t_k = t_air_c + 273.15
    emis_sky = 1.0 - 0.261 * math.exp(-7.77e-4 * t_air_c * t_air_c)
    return 5.670374e-8 * t_k ** 4 * emis_sky


def _convective_coefficient(t_surface: float, t_air: float, wind: float,
                            length: float) -> float:
    """Forced flat-plate (with transition) and natural convection, cube-mean."""
    re = wind * length / AIR_NU
    if re < 5.0e5:
        nu = 0.664 * math.sqrt(max(re, 1e-12)) * AIR_PR ** (1.0 / 3.0)
    else:
        nu = 0.037 * re ** 0.8 - 871.0
    h_forced = nu * AIR_K / length
    delta_t = max(abs(t_surface - t_air), 1e-6)
    h_natural = 1.32 * (delta_t / length) ** 0.25
    return (h_forced ** 3 + h_natural ** 3) ** (1.0 / 3.0)


def surface_heat_loss(*, t_surface: float, t_air: float, wind: float,
                      relative_humidity: float = 0.8, t_sea: float = -1.8,
                      spray_flux: float = 0.0, length: float = 2.0,
                      emissivity: float = 0.97, cloud_cover: float = 0.0) -> dict:
    """Five-term surface heat balance of an iced panel (Makiranta/Lozowski type).

    Positive terms mean **heat leaving** the surface:

    * ``q_conv = h_c (T_s - T_a)`` with ``h_c`` from the flat-plate correlation
      above plus natural convection in quadrature;
    * ``q_evap = h_m L_s (rho_v,s - rho_v,a)``, ``h_m = h_c / (rho c_p)`` (Chilton-
      Colburn, Lewis number unity for vapour in air);
    * ``q_lw = eps (sigma T_s^4 - L_down)`` with ``L_down`` from Idso-Jackson;
    * ``q_spray = W c_pw (T_sea - T_s)``;
    * the freezing term ``q_freeze`` is what is left over.

    This is an independent implementation of the published correlation set, so
    comparing it with the archived W4-N08 numbers is a genuine cross-check of two
    codes rather than a tautology.
    """
    h_c = _convective_coefficient(t_surface, t_air, wind, length)
    q_conv = h_c * (t_surface - t_air)
    h_m = h_c / (AIR_RHO * AIR_CP)
    e_hpa = 6.112 * math.exp(17.62 * t_air / (243.12 + t_air))
    rho_v_air = relative_humidity * e_hpa * 100.0 / (461.5 * (t_air + 273.15))
    rho_v_surface = _saturation_density_ice(t_surface) if t_surface <= 0.0 else \
        e_hpa * 100.0 / (461.5 * (t_surface + 273.15))
    q_evap = h_m * LV_VAPOUR * (rho_v_surface - rho_v_air)
    sky = _sky_longwave(t_air)
    sky = sky * (1.0 + 0.26 * cloud_cover)
    q_lw = emissivity * (5.670374e-8 * (t_surface + 273.15) ** 4 - sky)
    q_spray = spray_flux * 4000.0 * (t_sea - t_surface)
    required = q_conv + q_evap + q_lw - q_spray
    return {
        "h_conv_W_m2K": h_c,
        "q_conv_W_m2": q_conv,
        "q_evap_W_m2": q_evap,
        "q_lw_W_m2": q_lw,
        "q_spray_W_m2": q_spray,
        "q_required_W_m2": required,
        "t_surface": t_surface,
        "t_air": t_air,
        "wind": wind,
        "length": length,
    }


def ice_growth_from_balance(*, balance: dict, spray_flux: float,
                            conduction_into_structure: float = 0.0,
                            latent: float = LB_FUSION_SEAWATER) -> dict:
    """Freezing fraction and ice mass flux from the surface balance.

    ``W`` is the impinging spray mass flux.  In *wet growth* the surface is pinned
    at the freezing point and only the part ``f = Q_req / (W L_f)`` of the supply
    freezes; in *dry growth* everything freezes and the surface temperature drops
    below freezing.  ``conduction_into_structure`` (positive = heat flowing *into*
    the ice/steel, i.e. away from the surface) is the term W4-N08 neglects and the
    CHT solution supplies.
    """
    q_req = balance["q_required_W_m2"] - conduction_into_structure
    available = spray_flux * latent
    if available <= 0.0:
        return {"regime": "no_spray", "freezing_fraction": 0.0,
                "ice_flux_kg_m2_s": 0.0, "q_req_W_m2": q_req}
    if q_req <= 0.0:
        return {"regime": "melting", "freezing_fraction": 0.0,
                "ice_flux_kg_m2_s": 0.0, "q_req_W_m2": q_req}
    fraction = min(1.0, q_req / available)
    return {
        "regime": "wet" if fraction < 1.0 else "dry",
        "freezing_fraction": fraction,
        "ice_flux_kg_m2_s": fraction * spray_flux,
        "q_req_W_m2": q_req,
    }


#: archived W4-N08 reference panel (same environmental inputs), carried as
#: *reference data* for the cross-adjudication.  Source: TensorPlatform worktree
#: ``m4759``, ``artifacts/marine_icing_w4n08/validation.json`` ->
#: ``results.reference_panel`` (node W4-N08, evidence level REAL).
W4N08_REFERENCE_PANEL = {
    "source": "artifacts/marine_icing_w4n08/validation.json:results.reference_panel",
    "char_length_m": 2.0,
    "t_air_C": -10.0,
    "wind_m_s": 10.0,
    "t_sea_C": -1.8,
    "relative_humidity": 0.8,
    "spray_flux_kg_m2_s": 0.002535607967,
    "regime": "wet",
    "surface_temperature_C": -1.8,
    "freezing_fraction": 0.543297,
    "ice_flux_kg_m2_s": 0.0013775882858117778,
    "icing_rate_mm_per_h": 5.572267,
    "q_conv_W_m2": 218.9928,
    "q_evap_W_m2": 142.5713,
    "q_lw_W_m2": 98.1371,
    "q_spray_W_m2": 0.0,
    "q_req_W_m2": 459.701211,
    "deck_group_mean_ice_flux_kg_m2_s": 0.001332121481,
}