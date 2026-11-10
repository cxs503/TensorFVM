"""Seakeeping evidence layer on top of the sigma-grid time-domain wave tank.

The tank in :mod:`tensorfvm.wave_tank_td.sigma_tank` gives a state-variable free
surface and an exact algebraic pressure projection, i.e. the *fluid* side of a
2-D vertical-plane numerical wave tank.  What was missing (documented honest FAIL
in ``docs/wave-tank-td/README.md``) was the measurement layer needed to turn a
tank run into seakeeping evidence: a floating section whose hydrostatics are
exact, the Goda two-point decomposition of the wave field, the added-resistance
bookkeeping, and the potential-flow reference to adjudicate against.

This module supplies that layer.  Nothing here is a new solver -- it is the
physics contract between the tank and the reference solution:

Hydrostatics (exact, verified against the potential energy Hessian)
--------------------------------------------------------------------
For a rigid 2-D section floating at the still-water line the hydrostatic
stiffness about the waterline reference point ``ref`` is the Hessian of

    U(heave, pitch) = m g z_G - rho g int_{submerged} z dA

so that ``K33 = rho g B_wl`` and ``K55 = rho g I_wl - rho g nabla (z_G - z_B)``
= ``rho g nabla GM``.  :func:`potential_energy_stiffness` evaluates the Hessian
numerically from the *rotated wetted polygon* alone, so the analytic formula and
the tank's own pressure integration (which only knows the rotated geometry) can
be compared with no shared algebra.

Wave field
----------
:func:`goda_reflection` (from :mod:`tensorfvm.wave_tank.theory`) splits two
probes seaward of the body into incident and reflected components;
:func:`far_field_drift_2d` is the 2-D mean drift force from the radiation-stress
balance

    F_x = (rho g / 2)(a_i^2 + a_r^2 - a_t^2) (2 n - 1/2),  n = c_g / c,

which is the same ``rho g/4`` deep-water coefficient TensorBEM uses in
``tensorbem.drift_force.two_d_far_field_drift`` (checked in the tests).  The
energy form of Gerritsma & Meijers (1980) is the potential-energy bookkeeping of
the same balance: :func:`gerritsma_meijers_drift` reports it with the explicit
factor so a reader can see which convention is in use instead of guessing.

Reference solution
------------------
:func:`reference_rao` solves the 2-DOF (heave, pitch) frequency-domain equation
of motion from *supplied* mass / added-mass / damping / restoring / excitation
coefficients, so the same routine adjudicates both the in-repo potential-flow
solver (:mod:`tensorfvm.wave_tank_td.potential2d`) and TensorBEM without this
module knowing either.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from ..wave_tank.theory import (GRAVITY, added_resistance_energy, goda_two_point,
                                harmonic_fit)
from .potential2d import wetted_polygon
from .sigma_tank import SigmaWaveTank, TankConfig, polygon_properties

__all__ = [
    "SectionGeometry", "floating_strip", "submerged_strip",
    "potential_energy", "hydrostatic_stiffness", "potential_energy_stiffness",
    "wave_energy_factor", "far_field_drift_2d", "gerritsma_meijers_drift",
    "reference_rao", "gci_two_grid", "run_seakeeping_case",
]


# ----------------------------------------------------------------- geometry
@dataclass(frozen=True)
class SectionGeometry:
    """A 2-D strip of the hull centreplane: length ``L``, draft ``T``.

    ``freeboard`` is *not* cosmetic.  A section whose deck sits exactly on the
    still-water line is neutrally stable in heave: sink it and the displaced
    volume cannot grow, so the Archimedes force stays constant and nothing
    pushes back.  The previous tank case used exactly that degenerate section
    and recorded heave running away -- see ``docs/wave-tank-td/README.md``.
    Every section here carries a freeboard of at least ``0.1 T``.
    """

    length: float
    draft: float
    freeboard: float = 0.0
    x0: float = 0.0

    def __post_init__(self):
        if self.length <= 0 or self.draft <= 0:
            raise ValueError("length and draft must be positive")
        if self.freeboard < 0:
            raise ValueError("freeboard must be non-negative")

    @property
    def polygon(self) -> np.ndarray:
        a = 0.5 * self.length
        x0, T, f = self.x0, self.draft, self.freeboard
        return np.array([[x0 - a, -T], [x0 + a, -T],
                         [x0 + a, f], [x0 - a, f]], dtype=float)

    @property
    def wetted_area(self) -> float:
        """Displaced area per unit width, ``L T`` (box section, Cb = 1)."""
        return self.length * self.draft

    @property
    def mass(self) -> float:
        """Mass per unit width for a section floating at the design draft."""
        return SECTION_DENSITY * self.wetted_area

    @property
    def waterplane(self) -> float:
        return self.length

    @property
    def inertia_about_cg(self) -> float:
        """Pitch inertia about the section CG (including freeboard)."""
        m = self.mass
        height = self.draft + self.freeboard
        # scale the inertia to the *floating* mass: the freeboard is structure,
        # not displacement, so use the equivalent mass distribution of the box
        return m * (self.length ** 2 + height ** 2) / 12.0

    @property
    def cg(self) -> np.ndarray:
        """CG of the floating box: uniform over the structure, not the draft."""
        height = self.draft + self.freeboard
        return np.array([self.x0, -self.draft + 0.5 * height])

    @property
    def ref(self) -> np.ndarray:
        """Pitch/heave reference: the section's vertical axis at the waterline."""
        return np.array([self.x0, 0.0])

    @property
    def inertia(self) -> float:
        """Pitch inertia about :attr:`ref` (parallel-axis)."""
        d = float(self.ref[1] - self.cg[1])
        return self.inertia_about_cg + self.mass * d * d

    def as_dict(self) -> dict:
        return {"length": self.length, "draft": self.draft,
                "freeboard": self.freeboard, "x0": self.x0,
                "wetted_area": self.wetted_area, "mass_per_length": self.mass,
                "inertia_about_ref": self.inertia,
                "inertia_about_cg": self.inertia_about_cg,
                "block_coefficient_2d": 1.0,
                "length_to_draft": self.length / self.draft,
                "cg": self.cg.tolist(), "ref": self.ref.tolist()}


SECTION_DENSITY = 1000.0


def floating_strip(length: float, draft: float, freeboard: float = 0.05,
                   x0: float = 0.0) -> SectionGeometry:
    return SectionGeometry(length=length, draft=draft, freeboard=freeboard, x0=x0)


def submerged_strip(length: float, height: float, top_depth: float,
                    x0: float = 0.0) -> np.ndarray:
    """Fully submerged rectangle with its top ``top_depth`` below the waterline.

    Used for the TensorBEM cross-check: that solver documents a *submerged
    closed body* as its domain, so the independent reference is built on a
    geometry both codes can represent exactly (no waterline, no freeboard).
    """
    if top_depth <= 0.0 or height <= 0.0 or length <= 0.0:
        raise ValueError("length, height and top_depth must be positive")
    a = 0.5 * length
    z_top, z_bot = -top_depth, -top_depth - height
    return np.array([[x0 - a, z_bot], [x0 + a, z_bot],
                     [x0 + a, z_top], [x0 - a, z_top]], dtype=float)


# ------------------------------------------------------------- hydrostatics
def potential_energy(poly, mass: float, heave: float, pitch: float, ref,
                     rho: float = SECTION_DENSITY, g: float = GRAVITY,
                     cg=None) -> float:
    """Hydrostatic potential energy of a rigid section at ``(heave, pitch)``.

    ``U = m g z_G - rho g int_{submerged} z dA``.  The rotation is taken about
    ``ref`` (the tank's own convention) and the mass centroid ``cg`` defaults to
    the geometric centroid of ``poly``; ``-d^2U`` is then the stiffness the
    equation of motion needs, and ``dU/dxi(0, 0) = 0`` for a section floating at
    its design draft.
    """
    poly = np.asarray(poly, dtype=float)
    ref = np.asarray(ref, dtype=float)
    c, s = math.cos(pitch), math.sin(pitch)
    rot = np.array([[c, s], [-s, c]])
    moved = (poly - ref) @ rot + ref + np.array([0.0, heave])
    wet = wetted_polygon(moved)
    area, _cx, cy, _ = polygon_properties(wet, rho)
    if cg is None:
        _a, cgx, cgz, _ = polygon_properties(poly, 1.0)
        cg = np.array([cgx, cgz])
    cg_moved = (np.asarray(cg, dtype=float) - ref) @ rot + ref + np.array([0.0, heave])
    return float(mass * g * cg_moved[1] - rho * g * area * cy)


def hydrostatic_stiffness(section: SectionGeometry,
                          rho: float = SECTION_DENSITY,
                          g: float = GRAVITY) -> dict:
    """Analytic ``K33``, ``K55`` (about :attr:`SectionGeometry.ref`), ``K35 = 0``.

    ``K33 = rho g B_wl`` and ``K55 = rho g nabla GM`` with
    ``GM = I_wl/nabla - BG``.  For the box section ``I_wl = L^3/12``,
    ``nabla = L T``, ``z_B = -T/2``, ``z_G = -T + (T+f)/2``.
    """
    L, T, f = section.length, section.draft, section.freeboard
    nabla = L * T
    i_wl = L ** 3 / 12.0
    z_b = -T / 2.0
    z_g = -T + 0.5 * (T + f)
    bm = i_wl / nabla
    bg = z_g - z_b
    gm = bm - bg
    return {
        "C33": rho * g * L,
        "C55": rho * g * nabla * gm,
        "C35": 0.0,
        "waterline": L, "I_waterline": i_wl, "displaced_area": nabla,
        "z_B": z_b, "z_G": z_g, "BM": bm, "BG": bg, "GM": gm,
        "metacentric_height": gm,
    }


def potential_energy_stiffness(poly, mass, ref, rho: float = SECTION_DENSITY,
                               g: float = GRAVITY, h: float = 1e-4) -> dict:
    """Numeric Hessian of :func:`potential_energy` -- an independent K check."""
    u0 = potential_energy(poly, mass, 0.0, 0.0, ref, rho, g)
    # U has a minimum at equilibrium, so the stiffness is +d^2U/dxi_i dxi_j
    k33 = (potential_energy(poly, mass, h, 0.0, ref, rho, g) - 2 * u0
           + potential_energy(poly, mass, -h, 0.0, ref, rho, g)) / h ** 2
    k55 = (potential_energy(poly, mass, 0.0, h, ref, rho, g) - 2 * u0
           + potential_energy(poly, mass, 0.0, -h, ref, rho, g)) / h ** 2
    k35 = ((potential_energy(poly, mass, h, h, ref, rho, g)
            - potential_energy(poly, mass, h, -h, ref, rho, g)
            - potential_energy(poly, mass, -h, h, ref, rho, g)
            + potential_energy(poly, mass, -h, -h, ref, rho, g)) / (4 * h * h))
    return {"C33": float(k33), "C55": float(k55), "C35": float(k35)}


# -------------------------------------------------------------- wave field
def wave_energy_factor(k: float, depth: float) -> float:
    """``2 n - 1/2`` with ``n = c_g/c = (1 + 2kh/sinh 2kh)/2``."""
    kh = k * depth
    if kh > 30.0:
        return 0.5
    n = 0.5 * (1.0 + 2.0 * kh / math.sinh(2.0 * kh))
    return 2.0 * n - 0.5


def far_field_drift_2d(a_i: float, a_r: float, a_t: float, k: float,
                       depth: float, rho: float = SECTION_DENSITY,
                       g: float = GRAVITY) -> float:
    """Mean drift force per unit width from the radiation-stress balance.

    ``F_x = (rho g/2)(a_i^2 + a_r^2 - a_t^2)(2n - 1/2)``.  In deep water
    ``2n - 1/2 = 1/2`` and this is ``(rho g/4)(a_i^2 + a_r^2 - a_t^2)``, the
    coefficient TensorBEM's ``two_d_far_field_drift`` documents.
    """
    factor = wave_energy_factor(k, depth)
    return 0.5 * rho * g * (a_i ** 2 + a_r ** 2 - a_t ** 2) * factor


def gerritsma_meijers_drift(a_i: float, a_r: float, a_t: float, k: float,
                            depth: float, rho: float = SECTION_DENSITY,
                            g: float = GRAVITY) -> dict:
    """Gerritsma & Meijers (1980) energy bookkeeping, spelled out.

    Per unit crest length the wave energy of a progressive train is
    ``e = rho g a^2 / 2`` and its flux is ``e c_g``.  Over one encounter the
    energy that does not leave the control volume is the added resistance work
    per unit advance, which for a *fixed* body reduces to the momentum balance
    of :func:`far_field_drift_2d`.  Both numbers are returned so a reader sees
    the convention:

    * ``energy_content``: ``(rho g/2)(a_i^2 + a_r^2 - a_t^2)`` -- energy per
      unit length lost from the wave train;
    * ``momentum_balance``: the same times ``(2n - 1/2)``, the force.

    The two differ by ``2 (2n - 1/2)`` = 1 in deep water: reporting the energy
    density as if it were a force is the common factor-of-two trap.
    """
    energy = 0.5 * rho * g * (a_i ** 2 + a_r ** 2 - a_t ** 2)
    factor = wave_energy_factor(k, depth)
    return {"energy_content": float(energy),
            "momentum_balance": float(energy * factor),
            "factor_2n_minus_half": float(factor),
            "ratio_energy_to_force": float(1.0 / factor)}


def gci_two_grid(fine: float, coarse: float, ratio: float = 1.4,
                 order: float = 2.0, safety: float = 1.25) -> dict:
    """Roache two-grid GCI with a stated (not fitted) observed order."""
    fine, coarse = float(fine), float(coarse)
    if fine <= 0.0 or coarse <= 0.0:
        raise ValueError("functionals must be positive for a relative GCI")
    diff = abs(coarse - fine) / fine
    if ratio <= 1.0:
        raise ValueError("refinement ratio must exceed 1")
    gci = safety * diff / (ratio ** order - 1.0)
    return {"fine": fine, "coarse": coarse, "ratio": float(ratio),
            "order": float(order), "safety_factor": float(safety),
            "relative_difference": float(diff), "gci_fine": float(gci)}


# ------------------------------------------------------------- reference RAO
def reference_rao(omega: float, mass: float, inertia: float, cg_offset: float,
                  added_mass, damping, stiffness, excitation,
                  wave_amp: float, k: float) -> dict:
    """Frequency-domain 2-DOF heave/pitch response from supplied coefficients.

    The mass matrix keeps the heave-pitch coupling of a CG that is not at the
    rotation reference (``cg_offset = z_ref - z_G``)::

        M = [[m, m d], [m d, I_ref]],   T = m h'^2/2 + m d h' th' + I_ref th'^2/2

    with ``A``, ``B``, ``K`` and ``F`` all referenced to the same point, so the
    result is comparable with the tank's time-domain equation of motion.
    """
    m = float(mass)
    i_ref = float(inertia)
    d = float(cg_offset)
    mmat = np.array([[m, m * d], [m * d, i_ref]])
    amat = np.asarray(added_mass, dtype=float)
    bmat = np.asarray(damping, dtype=float)
    kmat = np.asarray(stiffness, dtype=float)
    force = np.asarray(excitation, dtype=complex)
    lhs = -omega ** 2 * (mmat + amat) + 1j * omega * bmat + kmat
    xi = np.linalg.solve(lhs, force)
    return {
        "xi_heave": complex(xi[0]), "xi_pitch": complex(xi[1]),
        "rao_heave": float(abs(xi[0]) / wave_amp),
        "rao_pitch": float(abs(xi[1]) / (k * wave_amp)),
        "phase_heave": float(np.angle(xi[0])), "phase_pitch": float(np.angle(xi[1])),
        "natural_heave": float(math.sqrt(kmat[0, 0] / (m + amat[0, 0]))),
        "natural_pitch": float(math.sqrt(kmat[1, 1] / (i_ref + amat[1, 1]))),
        "mass_matrix": mmat.tolist(), "added_mass": amat.tolist(),
        "damping": bmat.tolist(), "stiffness": kmat.tolist(),
        "excitation": [[force[0].real, force[0].imag],
                       [force[1].real, force[1].imag]],
    }


# ------------------------------------------------------------------ the run
def _harmonic(t, signal, omega, keep_from):
    t = np.asarray(t)
    y = np.asarray(signal)
    keep = t >= keep_from
    amp, phase, mean, resid = harmonic_fit(t[keep], y[keep], omega)
    return {"amp": float(amp), "phase": float(phase), "mean": float(mean),
            "residual_rel": float(resid)}


def _complex_amp(fit: dict) -> complex:
    return complex(fit["amp"] * np.cos(fit["phase"]), fit["amp"] * np.sin(fit["phase"]))



def _radiation_from_forced(t, signal_force, moment, keep_from, omega, forced,
                           stiffness, section) -> dict:
    """Added mass and damping from a forced-oscillation run.

    The driven motion is ``xi = a cos(omega t)`` (phase zero, so the algebra
    stays readable).  The measured hydrodynamic force is

        F = (A omega^2 - K) a cos(omega t) + B omega a sin(omega t),

    i.e. the hydrostatic spring enters with the *opposite* sign to the added
    mass -- the distinction that makes ``A = (F_c/a + K)/omega^2`` rather than
    ``(F_c/a - K)/omega^2``.  ``K`` must therefore come from the verified
    hydrostatic stiffness, not from a fit.
    """
    if not forced:
        return None
    dof = int(forced["dof"])
    amp = float(forced["amp"])
    k_value = float(stiffness[0] if dof == 0 else stiffness[1])
    t = np.asarray(t)
    keep = t >= keep_from
    signal = np.asarray(signal_force if dof == 0 else moment)[keep]
    tt = t[keep]
    design = np.column_stack([np.cos(omega * tt), np.sin(omega * tt),
                              np.ones_like(tt)])
    coef, *_ = np.linalg.lstsq(design, signal, rcond=None)
    f_c, f_s, mean = (float(c) for c in coef)
    added = (f_c / amp + k_value) / omega ** 2
    damping = f_s / (amp * omega)
    return {"dof": dof, "amplitude": amp, "omega": omega,
            "force_cos": f_c, "force_sin": f_s, "force_mean": mean,
            "added_mass": added, "damping": damping,
            "stiffness_used": k_value,
            "fit_residual": float(np.std(signal - design @ coef) / max(np.std(signal), 1e-30))}

def _goda(s1, s2, x1, x2, k):
    """Goda two-point split of two seaward probes into incident + reflected."""
    a_i, a_r = goda_two_point(x1, x2, _complex_amp(s1), _complex_amp(s2), k)
    return {"A_I": a_i, "A_R": a_r, "C_R": abs(a_r) / abs(a_i) if abs(a_i) else float("inf")}


def run_seakeeping_case(*, amp: float = 0.02, period: float = 0.9,
                        nx: int = 200, ns: int = 22, length: float = 14.0,
                        depth: float = 1.0, spp: int = 120, periods: float = 12.0,
                        section: SectionGeometry | None = None,
                        added_mass=None, damping=None,
                        blend: float = 0.15, relax_waves: float = 1.0,
                        damp_waves: float = 1.5, damp_strength: float = 6.0,
                        motion: str = "free", forced: dict | None = None,
                        stiffness=None, polygon=None, ref=None, mass=None,
                        inertia=None, name: str = "seakeeping") -> dict:
    """One head-wave run of a floating section: RAO, reflection, drift force.

    The body is free in heave and pitch with surge fixed (the tank frame is the
    carriage frame).  ``added_mass`` / ``damping`` are *stabilisation* terms for
    the staggered time integration; the radiation reaction itself is carried by
    the fluid pressure, so passing zeros is physically complete and passing the
    potential-flow coefficients is the standard explicit-coupling remedy.  Both
    are measured, and the RAO must not depend on the choice.
    """
    if section is None:
        section = floating_strip(0.6, 0.2, freeboard=0.05, x0=6.0)
    cfg = TankConfig(length=length, depth=depth, nx=nx, ns=ns,
                     advection_blend=blend, relax_waves=relax_waves,
                     damp_waves=damp_waves, damp_strength=damp_strength)
    tank = SigmaWaveTank(cfg)
    wave = tank.set_wave(amp, period)
    tank.initialize_from_wave(0.0)
    am = np.zeros(2) if added_mass is None else np.asarray(added_mass, dtype=float)
    dp = np.zeros(2) if damping is None else np.asarray(damping, dtype=float)
    poly = np.asarray(section.polygon if polygon is None else polygon, dtype=float)
    reference_point = np.asarray(section.ref if ref is None else ref, dtype=float)
    body_mass = float(section.mass if mass is None else mass)
    body_inertia = float(section.inertia if inertia is None else inertia)
    tank.set_body(poly, mass=body_mass, inertia=body_inertia,
                  ref=reference_point, added_mass=am, damping=dp, motion=motion,
                  forced=forced)

    lam = wave.wavelength
    x1 = section.x0 - 2.0 * lam          # two seaward probes for Goda
    x2 = x1 + 0.25 * lam
    x_t = section.x0 + 1.5 * lam         # transmitted-wave probe
    keep_from = max(periods - 4.0, 1.0) * period
    dt = period / spp
    nsteps = int(round(periods * spp))

    ts, heave, pitch, fx, fy = [], [], [], [], []
    list_of_moment = []
    eta1, eta2, eta_t = [], [], []
    for _ in range(nsteps):
        t = tank.step(dt)
        ts.append(t)
        heave.append(float(tank.body["xi"][0]))
        pitch.append(float(tank.body["xi"][1]))
        fx.append(float(tank.body["force"][0]))
        fy.append(float(tank.body["force"][1]))
        list_of_moment.append(float(tank.body["moment"]))
        eta1.append(float(tank.eta_at(x1)))
        eta2.append(float(tank.eta_at(x2)))
        eta_t.append(float(tank.eta_at(x_t)))

    om = wave.omega
    has_wave = amp > 0.0
    h1 = _harmonic(ts, eta1, om, keep_from)
    h2 = _harmonic(ts, eta2, om, keep_from)
    ht = _harmonic(ts, eta_t, om, keep_from)
    hh = _harmonic(ts, heave, om, keep_from)
    hp = _harmonic(ts, pitch, om, keep_from)

    goda = _goda(h1, h2, x1, x2, wave.k) if has_wave else {"A_I": 0j, "A_R": 0j, "C_R": float("nan")}
    a_i = float(abs(goda["A_I"]))
    a_r = float(abs(goda["A_R"]))
    a_t = abs(complex(ht["amp"] * np.cos(ht["phase"]), ht["amp"] * np.sin(ht["phase"])))
    c_r = a_r / a_i if a_i > 0 else float("inf")
    drift_far = (far_field_drift_2d(a_i, a_r, a_t, wave.k, depth) if has_wave
                 else float("nan"))
    jm = (gerritsma_meijers_drift(a_i, a_r, a_t, wave.k, depth) if has_wave
          else {"energy_content": float("nan"), "momentum_balance": float("nan"),
                "factor_2n_minus_half": float("nan"), "ratio_energy_to_force": float("nan")})
    t_arr = np.asarray(ts)
    keep = t_arr >= keep_from
    r_direct, r_direct_se = added_resistance_energy(np.asarray(fx)[keep], t_arr[keep], period)

    return {
        "case": name, "name": name,
        "config": {"length": length, "depth": depth, "nx": nx, "ns": ns,
                   "spp": spp, "periods": periods, "advection_blend": blend,
                   "relax_waves": relax_waves, "damp_waves": damp_waves,
                   "damp_strength": damp_strength, "dx": length / nx,
                   "points_per_wavelength": float(wave.wavelength / (length / nx)),
                   "dt": dt},
        "wave": wave.as_dict(), "section": section.as_dict(),
        "stabilisation": {"added_mass": am.tolist(), "damping": dp.tolist()},
        "probes": {"x_goda": [x1, x2], "x_transmitted": x_t,
                   "x_goda_over_lambda": [x1 / lam, x2 / lam],
                   "x_transmitted_over_lambda": x_t / lam},
        "incident": {"amp_probe1": h1["amp"], "amp_probe2": h2["amp"],
                     "a_i": a_i, "a_r": a_r, "a_t": a_t},
        "reflection": {"C_R": float(c_r), "k_dx": float(wave.k * (x2 - x1)),
                       "A_I_re": goda["A_I"].real, "A_I_im": goda["A_I"].imag,
                       "A_R_re": goda["A_R"].real, "A_R_im": goda["A_R"].imag},
        "rao": {"heave_amp": hh["amp"], "pitch_amp": hp["amp"],
                "rao_heave": hh["amp"] / amp if has_wave else float("nan"),
                "rao_pitch": hp["amp"] / (wave.k * amp) if has_wave else float("nan"),
                "heave_residual": hh["residual_rel"],
                "pitch_residual": hp["residual_rel"],
                "heave_mean": hh["mean"], "pitch_mean": hp["mean"]},
        "drift": {"direct_mean_force": float(r_direct),
                  "direct_standard_error": float(r_direct_se),
                  "far_field": float(drift_far),
                  "energy_content": jm["energy_content"],
                  "factor_2n_minus_half": jm["factor_2n_minus_half"],
                  "relative_difference": float(abs(r_direct - drift_far)
                                               / max(abs(drift_far), 1e-30)
                                               if drift_far else float("inf"))},
        "numerics": {"div_residual_max": float(tank.div_residual),
                     "volume_drift_rel": float(tank.volume_drift / tank.volume0),
                     "eta_max_final": float(np.abs(tank.eta).max())},
        "motion": {"mode": motion, "forced": forced},
        "radiation": (_radiation_from_forced(ts, fx, list_of_moment, keep_from, wave.omega,
                                             forced or {}, stiffness,
                                             section) if motion == "forced" else None),
        "snapshot": {"x": tank.x.tolist(), "eta": tank.eta.tolist(),
                     "body": tank.body_polygon().tolist()},
        "series": {"t": ts, "heave": heave, "pitch": pitch, "fx": fx, "fy": fy,
                   "eta_goda1": eta1, "eta_goda2": eta2, "eta_t": eta_t},
    }