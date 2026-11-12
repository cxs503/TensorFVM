"""Multi-species (mass-fraction) transport and mixture-fraction chemistry algebra.

Why this module exists
----------------------
``docs/verification-catalog.md`` §10.1 gap 4 ("Combustion: No species transport
verification") was the last untouched item of the catalog's gap list: the
compressible package carries at most *two-material* bookkeeping ("species" there
means "material id", see ``compressible/solver.py``) and nothing in the
repository transported a mass fraction ``Y_k``.  This module adds the missing
layer -- transport of species mass fractions, the conserved-scalar (mixture
fraction) algebra and the Burke--Schumann infinitely-fast-chemistry closure --
so that the gap can be *verified* instead of merely declared.

Scope, stated up front
----------------------
Supported and verified (see :mod:`tensorfvm.verification.species_transport`):

* Fickian diffusion of mass fractions with mixture-averaged diffusivities and
  the zero-sum ("correction velocity") flux fix;
* 1-D finite-volume transport, steady (sparse linear solve) and transient
  (explicit Euler), central or first-order upwind convection, Dirichlet or
  zero-gradient boundaries, optional linear volumetric source;
* Bilger's element-based mixture fraction ``Z`` (Bilger 1977) and the
  Burke--Schumann piecewise-linear-in-``Z`` structure with an
  enthalpy-consistent temperature (Burke & Schumann 1928).

Deliberately *not* supported (declared, not hidden):

* finite-rate Arrhenius chemistry, stiff splitting, detailed mechanisms -- the
  ``GAP-FVM-COMBUSTION-SPECIES`` proposal's flame-speed acceptance needs Cantera
  for its reference, and Cantera is not installable here, so that outer anchor
  stays WITHHELD;
* differential diffusion / non-unity Lewis numbers in the chemistry closure;
* multi-dimensional transport, variable density, Soret/Dufour effects.

Numerical conventions
---------------------
Variables are cell-centred on a uniform grid; faces sit half a cell away, so a
Dirichlet boundary value is imposed on the *face* and the ghost cell is the
exact linear reflection ``phi_ghost = 2*phi_bc - phi_0``.  For every species
``k`` the equation solved is

    d(rho Y_k)/dt + d(rho u Y_k)/dx = d/dx[ rho D_k dY_k/dx ] + S

with a first-order upwind or central face value for the convective flux.  A
mixture keeps ``sum_k Y_k = 1`` only if the diffusive fluxes sum to zero, which
Fickian diffusion with unequal ``D_k`` does not guarantee; :func:`diffusive_flux`
therefore returns the corrected fluxes ``j_k - Y_k sum_m j_m`` (the standard
correction velocity), and the drift is measured, not assumed away.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Mapping, Optional, Sequence, Tuple

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

__all__ = [
    "ATOMIC_MASS",
    "SPECIES_COMPOSITION",
    "CNUB",
    "molar_mass",
    "element_mass_fractions",
    "bilger_beta",
    "BilgerScalar",
    "Stoichiometry",
    "burke_schumann",
    "diffusive_flux",
    "solve_steady_species_1d",
    "solve_transient_mixture_1d",
]

# --------------------------------------------------------------------------
# Thermodynamic / stoichiometric data (CODATA atomic weights)
# --------------------------------------------------------------------------
ATOMIC_MASS: Dict[str, float] = {
    "C": 12.011,
    "H": 1.008,
    "O": 15.999,
    "N": 14.007,
}

#: Number of atoms of each element in a species (only what CH4/air needs).
SPECIES_COMPOSITION: Dict[str, Dict[str, int]] = {
    "CH4": {"C": 1, "H": 4},
    "O2": {"O": 2},
    "N2": {"N": 2},
    "CO2": {"C": 1, "O": 2},
    "H2O": {"H": 2, "O": 1},
}

#: Standard specific enthalpies of formation [J/kg] at 298.15 K (N2/O2 datum).
#: ``kJ/kmol / (kg/kmol) = kJ/kg``, hence the explicit 1e3: dividing J/mol by
#: the molar mass in g/mol (what :func:`molar_mass` returns) gives kJ/kg, not
#: J/kg -- forgetting the factor silently produced 300 K "flames".
CNUB: Dict[str, float] = {
    "CH4": -74.873e3 / 16.043 * 1e3,
    "CO2": -393.522e3 / 44.010 * 1e3,
    "H2O": -241.826e3 / 18.015 * 1e3,
    "O2": 0.0,
    "N2": 0.0,
}


def molar_mass(species: str) -> float:
    """Molar mass [kg/kmol] of a species in :data:`SPECIES_COMPOSITION`."""
    try:
        comp = SPECIES_COMPOSITION[species]
    except KeyError:  # pragma: no cover - defensive
        raise KeyError(f"unknown species {species!r}") from None
    return float(sum(ATOMIC_MASS[e] * n for e, n in comp.items()))


def element_mass_fractions(Y: Mapping[str, float]) -> Dict[str, float]:
    """Element mass fractions from species mass fractions.

    ``Z_e = sum_k Y_k * (atoms of e in k) * M_e / M_k``.  This is the map that
    makes reactions *element conserving* by construction, which is exactly the
    invariant the mixture fraction relies on.
    """
    out: Dict[str, float] = {}
    for name, comp in SPECIES_COMPOSITION.items():
        y = float(Y.get(name, 0.0))
        if y == 0.0:
            continue
        mm = molar_mass(name)
        for e, n in comp.items():
            out[e] = out.get(e, 0.0) + y * n * ATOMIC_MASS[e] / mm
    return out


def bilger_beta(Y: Mapping[str, float]) -> float:
    """Bilger's coupling function ``beta`` (Bilger 1977, Combust. Sci. Tech. 13:87).

    ``beta = 2 Z_C/M_C + Z_H/(2 M_H) - Z_O/M_O`` with ``Z_e`` the element mass
    fractions.  Its key property -- verified in
    ``tensorfvm.verification.species_transport`` -- is ``beta = 0`` for any
    stoichiometric CH4/air mixture, so the mixture fraction built from it
    reproduces the mass-stoichiometric value and is invariant under reaction.
    """
    Z = element_mass_fractions(Y)
    return (
        2.0 * Z.get("C", 0.0) / ATOMIC_MASS["C"]
        + Z.get("H", 0.0) / (2.0 * ATOMIC_MASS["H"])
        - Z.get("O", 0.0) / ATOMIC_MASS["O"]
    )


@dataclass(frozen=True)
class BilgerScalar:
    """Element-based mixture fraction ``Z = (beta - beta_ox)/(beta_f - beta_ox)``."""

    Y_fuel: Mapping[str, float]
    Y_ox: Mapping[str, float]

    @property
    def beta_fuel(self) -> float:
        return bilger_beta(self.Y_fuel)

    @property
    def beta_ox(self) -> float:
        return bilger_beta(self.Y_ox)

    def __call__(self, Y: Mapping[str, float]) -> float:
        denom = self.beta_fuel - self.beta_ox
        if denom == 0.0:  # pragma: no cover - defensive
            raise ZeroDivisionError("fuel and oxidizer streams have equal beta")
        return (bilger_beta(Y) - self.beta_ox) / denom

    def from_mixture(self, Y: Mapping[str, float]) -> float:
        return self(Y)


@dataclass(frozen=True)
class Stoichiometry:
    """Single-step irreversible ``CH4 + 2 O2 -> CO2 + 2 H2O`` closure data.

    ``Y_oxidant_ox`` is the oxidant mass fraction of the oxidizer stream
    (0.232 for standard dry air), the inert being the rest.  All derived
    quantities below are *closed forms* of the elemental composition, so the
    verification driver can compare two independent routes to them.
    """

    fuel: str = "CH4"
    oxidant: str = "O2"
    inert: str = "N2"
    Y_oxidant_ox: float = 0.232

    @property
    def nu_oxidant(self) -> float:
        """kmol O2 per kmol fuel (``a + b/4`` for C_a H_b, no fuel oxygen)."""
        comp = SPECIES_COMPOSITION[self.fuel]
        if "O" in comp:  # pragma: no cover - defensive
            raise NotImplementedError("fuel oxygen not supported")
        return comp["C"] + comp["H"] / 4.0

    @property
    def mass_stoichiometric_ratio(self) -> float:
        """``s`` = kg oxidant per kg fuel at stoichiometry."""
        return self.nu_oxidant * molar_mass(self.oxidant) / molar_mass(self.fuel)

    @property
    def Y_fuel_fuel(self) -> float:
        return 1.0

    @property
    def Z_st_coupling(self) -> float:
        """Stoichiometric ``Z`` from the coupling function ``s Y_F - Y_O``."""
        a = self.Y_oxidant_ox / self.mass_stoichiometric_ratio
        return a / (self.Y_fuel_fuel + a)

    @property
    def Z_st(self) -> float:
        """Stoichiometric ``Z`` from the **Bilger** scalar (the primary one here)."""
        sc = BilgerScalar({self.fuel: 1.0}, {self.oxidant: self.Y_oxidant_ox,
                                            self.inert: 1.0 - self.Y_oxidant_ox})
        Yst = self.stoichiometric_products()
        return sc(Yst)

    def stoichiometric_products(self) -> Dict[str, float]:
        """Mass fractions of a stoichiometric fuel/air mixture (all burnt)."""
        comp = SPECIES_COMPOSITION[self.fuel]
        n_c, n_h = comp["C"], comp["H"]
        m_fuel = molar_mass(self.fuel)
        m_prod = n_c * molar_mass("CO2") + (n_h / 2.0) * molar_mass("H2O")
        m_ox = self.nu_oxidant * molar_mass(self.oxidant)
        m_inert = m_ox * (1.0 - self.Y_oxidant_ox) / self.Y_oxidant_ox
        total = m_fuel + m_ox + m_inert
        return {
            "CO2": n_c * molar_mass("CO2") / total,
            "H2O": (n_h / 2.0) * molar_mass("H2O") / total,
            self.inert: m_inert / total,
        }

    @property
    def Y_products_st(self) -> float:
        """Product mass fraction of the stoichiometric mixture."""
        prod = self.stoichiometric_products()
        return prod["CO2"] + prod["H2O"]

    @property
    def product_split(self) -> Dict[str, float]:
        """Relative CO2 / H2O split of the product mass (sums to 1)."""
        comp = SPECIES_COMPOSITION[self.fuel]
        m_co2 = comp["C"] * molar_mass("CO2")
        m_h2o = (comp["H"] / 2.0) * molar_mass("H2O")
        tot = m_co2 + m_h2o
        return {"CO2": m_co2 / tot, "H2O": m_h2o / tot}


def burke_schumann(
    Z,
    stoich: Optional[Stoichiometry] = None,
    *,
    cp: float = 1200.0,
    T0: float = 298.15,
    T_fuel: float = 298.15,
    T_ox: float = 298.15,
) -> Dict[str, np.ndarray]:
    """Burke--Schumann infinitely-fast-chemistry structure, piecewise linear in ``Z``.

    For ``Z <= Z_st`` (lean) ``Y_F = 0`` and the oxidant burns linearly with
    ``Z``; for ``Z > Z_st`` (rich) ``Y_O = 0`` and the fuel survives linearly.
    Products are linear in ``Z`` on both sides and ``Y_inert = Y_inert_ox (1-Z)``.
    Temperature follows from the *enthalpy* of the mixture, which is itself a
    conserved scalar under the unity-Lewis assumption,

        h(Z) = (1-Z) h_ox + Z h_fuel,   h = sum_k Y_k (h_f,k + cp (T - T0)),

    i.e. ``T = T0 + (h(Z) - sum_k Y_k h_f,k)/cp``.  With a single ``cp`` this is
    the classical piecewise-linear flame-sheet temperature; dissociation and a
    temperature-dependent ``cp`` are out of scope (see the module docstring).
    """
    st = stoich or Stoichiometry()
    Z = np.asarray(Z, dtype=float)
    if np.any(Z < 0.0) or np.any(Z > 1.0):
        raise ValueError("mixture fraction must lie in [0, 1]")
    zst = st.Z_st
    Y_ox_ox = st.Y_oxidant_ox
    Y_in_ox = 1.0 - Y_ox_ox
    yp_st = st.Y_products_st
    split = st.product_split

    lean = Z <= zst
    Y_F = np.where(lean, 0.0, (Z - zst) / (1.0 - zst))
    Y_O = np.where(lean, Y_ox_ox * (1.0 - Z / zst), 0.0)
    Y_P = np.where(lean, yp_st * Z / zst, yp_st * (1.0 - Z) / (1.0 - zst))

    out: Dict[str, np.ndarray] = {
        "Y_F": Y_F,
        "Y_O": Y_O,
        "Y_CO2": Y_P * split["CO2"],
        "Y_H2O": Y_P * split["H2O"],
        "Y_N2": Y_in_ox * (1.0 - Z),
    }
    # Enthalpy route: conserved scalar h(Z) minus the chemical part.
    h_fuel = CNUB[st.fuel] + cp * (T_fuel - T0)
    h_ox = CNUB[st.oxidant] * Y_ox_ox + CNUB[st.inert] * Y_in_ox + cp * (T_ox - T0)
    h = (1.0 - Z) * h_ox + Z * h_fuel
    chem = (
        out["Y_F"] * CNUB[st.fuel]
        + out["Y_O"] * CNUB[st.oxidant]
        + out["Y_CO2"] * CNUB["CO2"]
        + out["Y_H2O"] * CNUB["H2O"]
        + out["Y_N2"] * CNUB[st.inert]
    )
    out["T"] = T0 + (h - chem) / cp
    out["h"] = h
    # species-name view (the flat "Y_F"/"Y_O" keys are plot-friendly shorthands;
    # mixing them up with element names is an easy trap -- Z_E / Z_O is oxygen)
    out["Y"] = {st.fuel: out["Y_F"], st.oxidant: out["Y_O"],
                "CO2": out["Y_CO2"], "H2O": out["Y_H2O"], st.inert: out["Y_N2"]}
    return out


# --------------------------------------------------------------------------
# Diffusive fluxes
# --------------------------------------------------------------------------
def diffusive_flux(
    Y: Mapping[str, np.ndarray],
    gradY: Mapping[str, np.ndarray],
    D: Mapping[str, float],
    rho=1.0,
    *,
    correct: bool = True,
) -> Dict[str, np.ndarray]:
    """Fickian mass fluxes ``j_k = -rho D_k grad Y_k`` with the zero-sum fix.

    Fick's law per species does not conserve total mass unless the ``D_k`` are
    equal: ``sum_k j_k = -rho sum_k D_k grad Y_k != 0``.  With ``correct=True``
    the standard correction velocity is applied,

        j_k <- j_k - Y_k sum_m j_m,

    after which ``sum_k j_k = 0`` to round-off (verified: see criterion C7 of
    the verification driver, and the quantified *uncorrected* drift).
    """
    raw = {k: -rho * D[k] * np.asarray(gradY[k], dtype=float) for k in Y}
    if not correct:
        return raw
    ysum = sum(np.asarray(Y[k], dtype=float) for k in Y)
    if np.any(np.abs(ysum - 1.0) > 1e-8):
        raise ValueError(
            "zero-sum diffusive-flux correction needs the complete species set "
            "(sum_k Y_k = 1); got max|sum Y - 1| = "
            f"{float(np.max(np.abs(ysum - 1.0))):.3e}.  For a single-species or "
            "single-scalar diffusion problem pass correct=False -- otherwise the "
            "correction multiplies the flux by (1 - Y) and silently destroys it."
        )
    total = sum(raw[k] for k in raw)
    return {k: raw[k] - np.asarray(Y[k], dtype=float) * total for k in raw}


# --------------------------------------------------------------------------
# 1-D finite-volume discretisation
# --------------------------------------------------------------------------
def _grid(n: int, L: float, x0: float = 0.0) -> Tuple[np.ndarray, float]:
    if n < 3:
        raise ValueError("need at least 3 cells")
    if L <= 0.0:
        raise ValueError("domain length must be positive")
    dx = L / n
    xc = x0 + (np.arange(n) + 0.5) * dx
    return xc, dx


def _ghosts(phi: np.ndarray, bc_left, bc_right) -> np.ndarray:
    """Extended array with one ghost cell per side (see module docstring)."""
    g = np.empty(phi.size + 2, dtype=float)
    g[1:-1] = phi
    kind_l, val_l = bc_left
    kind_r, val_r = bc_right
    if kind_l == "dirichlet":
        g[0] = 2.0 * val_l - phi[0]
    elif kind_l == "zerogradient":
        g[0] = phi[0]
    else:  # pragma: no cover - defensive
        raise ValueError(f"unknown bc kind {kind_l!r}")
    if kind_r == "dirichlet":
        g[-1] = 2.0 * val_r - phi[-1]
    elif kind_r == "zerogradient":
        g[-1] = phi[-1]
    else:  # pragma: no cover - defensive
        raise ValueError(f"unknown bc kind {kind_r!r}")
    return g


def _face_value(phi_g: np.ndarray, u: float, scheme: str) -> np.ndarray:
    """Interior face values of ``phi`` from the ghost-extended array."""
    if scheme == "central":
        return 0.5 * (phi_g[:-1] + phi_g[1:])
    if scheme == "upwind":
        if u >= 0.0:
            return phi_g[:-1]
        return phi_g[1:]
    raise ValueError(f"unknown scheme {scheme!r}")


def solve_steady_species_1d(
    *,
    n: int,
    L: float,
    D: float,
    bc_left: Tuple[str, Optional[float]],
    bc_right: Tuple[str, Optional[float]],
    u=0.0,
    rho: float = 1.0,
    scheme: str = "central",
    source_coeff: float = 0.0,
    source_const: float = 0.0,
) -> np.ndarray:
    """Steady 1-D finite-volume solution of one species mass fraction.

    Solves ``d(rho u phi)/dx = d/dx(rho D dphi/dx) + source_coeff*phi +
    source_const`` on a uniform grid by assembling the linear system and using a
    sparse direct solve.  ``bc_left/bc_right`` are ``("dirichlet", value)`` or
    ``("zerogradient", None)`` tuples evaluated on the *face*; the Dirichlet
    ghost cell is the exact linear reflection ``2*value - phi_0`` so a pure
    diffusion problem is reproduced to round-off (verified, criterion C0 of the
    verification driver).

    ``source_coeff`` is the linear volumetric source coefficient on ``phi``; it
    is the hook that lets a curved/stretched 1-D reduction carry the transverse
    convective term (see criterion C3 of the verification driver: omitting it
    is a quantified trap, not a detail).
    """
    xc, dx = _grid(n, L)
    if D <= 0.0:
        raise ValueError("diffusivity must be positive")
    if scheme not in ("central", "upwind"):
        raise ValueError(f"unknown scheme {scheme!r}")
    n_cells = n

    # Linear form: value = sum_j coeff[j]*phi_j + const.
    def _expr(kind: str, val: Optional[float]) -> Tuple[Dict[int, float], float]:
        if kind == "dirichlet":
            return {}, float(val)  # the face value of a Dirichlet BC is the BC
        if kind == "zerogradient":
            return {}, 0.0
        raise ValueError(f"unknown bc kind {kind!r}")  # pragma: no cover

    def _ghost_expr(is_right: bool, bc) -> Tuple[Dict[int, float], float]:
        kind, val = bc
        cell = n_cells - 1 if is_right else 0
        if kind == "dirichlet":
            return {cell: -1.0}, 2.0 * float(val)
        return {cell: 1.0}, 0.0

    def _add(terms, expr, scale):
        coeff, const = expr
        for j, a in coeff.items():
            terms[j] = terms.get(j, 0.0) + scale * a
        return scale * const

    rows: list = []
    cols: list = []
    vals: list = []
    rhs = np.zeros(n_cells)

    for i in range(n_cells):
        for sign, is_left_face in ((+1.0, False), (-1.0, True)):
            if is_left_face:
                left_ghost, right_ghost = (i == 0), False
                cell_left, cell_right = i - 1, i
            else:
                left_ghost, right_ghost = False, (i == n_cells - 1)
                cell_left, cell_right = i, i + 1

            phi_L = _ghost_expr(False, bc_left) if left_ghost else ({cell_left: 1.0}, 0.0)
            phi_R = _ghost_expr(True, bc_right) if right_ghost else ({cell_right: 1.0}, 0.0)

            # convective face value
            if scheme == "upwind":
                phi_face = phi_L if u >= 0.0 else phi_R
                if left_ghost and u >= 0.0 and bc_left[0] == "dirichlet":
                    phi_face = _expr(*bc_left)          # inflow face = BC exactly
                if right_ghost and u < 0.0 and bc_right[0] == "dirichlet":
                    phi_face = _expr(*bc_right)
            else:  # central
                phi_face = ({}, 0.0)
                coeff: Dict[int, float] = {}
                for e in (phi_L, phi_R):
                    for j, a in e[0].items():
                        coeff[j] = coeff.get(j, 0.0) + 0.5 * a
                phi_face = (coeff, 0.5 * (phi_L[1] + phi_R[1]))

            terms: Dict[int, float] = {}
            const = _add(terms, phi_face, rho * u)          # rho u phi_face
            const += _add(terms, phi_L, +rho * D / dx)      # -rho D (phi_R-phi_L)/dx
            const += _add(terms, phi_R, -rho * D / dx)

            scale = sign / dx
            for j, a in terms.items():
                if 0 <= j < n_cells:
                    rows.append(i)
                    cols.append(j)
                    vals.append(scale * a)
            rhs[i] -= scale * const

        # linear volumetric source on the cell value
        rows.append(i)
        cols.append(i)
        vals.append(-source_coeff)
        rhs[i] += source_const

    A = sp.csr_matrix((vals, (rows, cols)), shape=(n_cells, n_cells))
    phi = spla.spsolve(A.tocsc(), rhs)
    if not np.all(np.isfinite(phi)):  # pragma: no cover - defensive
        raise RuntimeError("steady species solve produced non-finite values")
    return np.asarray(phi, dtype=float)


def solve_transient_mixture_1d(
    Y0: Mapping[str, np.ndarray],
    *,
    L: float,
    D: Mapping[str, float],
    dt: float,
    n_steps: int,
    u=0.0,
    rho: float = 1.0,
    scheme: str = "central",
    bc_left=None,
    bc_right=None,
    correct_fluxes: bool = True,
    source_coeff: float = 0.0,
    store_every: int = 0,
) -> Tuple[Dict[str, np.ndarray], list]:
    """Explicit-Euler transient 1-D transport of a species mixture.

    ``Y0`` maps species name to its initial cell-centred mass-fraction array;
    ``D`` maps the same names to mixture-averaged diffusivities.  Boundaries are
    given per species as ``{name: ("dirichlet", value) | ("zerogradient", None)}``;
    a species missing from ``bc_left``/``bc_right`` gets a zero-gradient face.

    With ``correct_fluxes=True`` the diffusive fluxes are made to sum to zero via
    :func:`diffusive_flux`, which is what keeps ``sum_k Y_k = 1`` (criterion C8);
    with ``False`` the drift is the *measured* consequence of unequal ``D_k``.

    Returns ``(Y_final, history)`` where ``history`` holds ``(step, Y_snapshot)``
    pairs when ``store_every > 0``.
    """
    names = list(Y0)
    if not names:  # pragma: no cover - defensive
        raise ValueError("empty species mixture")
    n = np.asarray(Y0[names[0]], dtype=float).size
    xc, dx = _grid(n, L)
    Y = {k: np.asarray(Y0[k], dtype=float).copy() for k in names}
    for k in names:
        if Y[k].shape != (n,):  # pragma: no cover - defensive
            raise ValueError(f"species {k!r} has shape {Y[k].shape}, expected ({n},)")
    bc_left = {k: ("zerogradient", None) for k in names} if bc_left is None else bc_left
    bc_right = {k: ("zerogradient", None) for k in names} if bc_right is None else bc_right
    if dt <= 0.0:
        raise ValueError("dt must be positive")
    history: list = []
    for step in range(n_steps):
        # uncorrected Fickian fluxes, then optional zero-sum correction
        grad = {}
        Yg = {}
        for k in names:
            Yg[k] = _ghosts(Y[k], bc_left.get(k, ("zerogradient", None)),
                            bc_right.get(k, ("zerogradient", None)))
            grad[k] = (Yg[k][1:] - Yg[k][:-1]) / dx
        Yface = {k: _face_value(Yg[k], u, scheme) for k in names}
        j = diffusive_flux(Yface, grad, D, rho, correct=correct_fluxes)
        for k in names:
            conv = rho * u * Yface[k]
            total = conv + j[k]
            Y[k] = Y[k] - dt / (rho * dx) * (total[1:] - total[:-1])
            Y[k] = Y[k] + dt * source_coeff * Y[k]
        if store_every and (step + 1) % store_every == 0:
            history.append((step + 1, {k: Y[k].copy() for k in names}))
    return Y, history