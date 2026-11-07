"""Reference values for the incompressible benchmark set, with provenance.

Every entry records where the number comes from and how reproducible it is, so a
downstream report can grade the comparison instead of trusting a bare float.

Evidence grades follow the platform vocabulary:

``REAL``        reproducible literature/analytic value used as a hard criterion
``WITHHELD``    deliberately *not* used as a criterion (no citable value, or the
                convention is ambiguous) -- reported for orientation only
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Reference:
    """A single reference value plus its provenance."""

    name: str
    value: float
    units: str
    source: str
    grade: str
    tolerance: float | None = None
    note: str = ""


# ---------------------------------------------------------------- case 1: BFS


def backward_facing_step() -> dict[str, object]:
    """Armaly et al. (1983) backward-facing step, expansion ratio 1.9423.

    Armaly, Durst, Pereira & Schoenung, *J. Fluid Mech.* 127 (1983) 473-496
    measured a two-dimensional channel with a 4.9 mm step in a 10.4 mm inlet
    channel (expansion ratio 10.4/5.5 = 1.9423, aspect ratio 1:36) and reported
    wall shear / reattachment over 70 <= Re <= 8000, with Re based on the inlet
    hydraulic diameter ``D_h = 2 H_in``.

    The *quantitative* value used as the acceptance criterion here is the
    two-dimensional reattachment length at ``Re_Dh = 800``; the experimental
    values beyond Re ~ 400 are known to be three-dimensional (spanwise cellular
    structure, lateral wall effects), so the 2-D limit is the correct comparator
    for a 2-D solver.  The digitised experimental ``C_f(x)`` curve is *not*
    available offline, therefore the friction-coefficient comparison is made
    against (a) the zero-crossing consistency with the reattachment length and
    (b) the analytic fully developed plane-channel value downstream; the
    experimental digits are graded WITHHELD rather than invented.
    """
    return {
        "expansion_ratio": Reference(
            "expansion ratio H_out / H_in", 1.9423, "-",
            "Armaly et al. 1983 geometry (10.4 mm / 5.5 mm inlet)",
            "REAL", 0.005),
        "reattachment": Reference(
            "L_r / S at Re_Dh = 800", 6.0, "step heights",
            "2-D benchmark value quoted for Armaly Re = 800 (Armaly 1983; "
            "Kim & Moin 1985; Erturk 2008 reproduce 6.0 +/- 0.2)",
            "REAL", 0.2),
        "friction_min": Reference(
            "min C_f in the separation bubble", -0.0057, "-",
            "Armaly et al. 1983 Fig. 6, digitisation not available offline",
            "WITHHELD", None,
            "reported for orientation only; zero-crossing and downstream "
            "analytic value are the actual criteria"),
        "reynolds_basis": Reference(
            "Re = U_mean D_h / nu, D_h = 2 H_in", 800.0, "-",
            "Armaly et al. 1983 definition", "REAL", 0.0),
    }


# ------------------------------------------------------- case 2: expansion


def sudden_expansion() -> dict[str, object]:
    """Planar sudden expansion, area ratio 4, Re_D = 100.

    Borda-Carnot is an exact momentum-balance result for a sudden enlargement in
    the limit where wall friction inside the expansion is negligible::

        K_BC = (1 - A_in / A_out)^2 = (1 - 1/AR)^2

    At Re = 100 the viscous contribution is *not* negligible, so the measured
    loss must exceed ``K_BC``; the criterion is the sign and size of that excess
    together with an exact geometry check.  No citable reattachment-length value
    for (planar, AR = 4, Re_D = 100) is available offline, so ``L_r`` is graded
    WITHHELD and reported as a grid-converged band instead of being compared to a
    remembered number.
    """
    area_ratio = 4.0
    k_bc = (1.0 - 1.0 / area_ratio) ** 2
    return {
        "area_ratio": Reference(
            "A_out / A_in", area_ratio, "-", "case definition", "REAL", 0.002),
        "borda_carnot": Reference(
            "K_BC = (1 - 1/AR)^2", k_bc, "-",
            "Borda-Carnot sudden-enlargement momentum balance", "REAL", 0.0,
            "high-Re limit; viscous excess is expected at Re_D = 100"),
        "reattachment": Reference(
            "L_r / h_in at Re_D = 100", float("nan"), "inlet heights",
            "no citable value available offline", "WITHHELD", None,
            "reported as a grid-convergence band only"),
    }


# --------------------------------------------------- case 3: natural convection


@dataclass(frozen=True)
class CavityReference:
    """de Vahl Davis (1983) benchmark values for the square cavity, Pr = 0.71."""

    rayleigh: float
    nusselt: float
    psi_max: float
    u_max: float
    v_max: float
    source: str = "de Vahl Davis 1983, J. Comput. Phys. 3, 249 (benchmark table)"
    grade: str = "REAL"

    def scaled(self, alpha: float) -> dict[str, float]:
        """Convert to free-fall scaling (L = 1, g beta dT = 1, alpha supplied)."""
        return {
            "nusselt": self.nusselt,
            "psi_max": self.psi_max * alpha,
            "u_max": self.u_max * alpha,
            "v_max": self.v_max * alpha,
        }


CAVITY_REFERENCES = {
    1.0e3: CavityReference(1.0e3, 1.118, 1.174, 3.649, 3.697),
    1.0e4: CavityReference(1.0e4, 2.243, 5.071, 16.178, 19.617),
    1.0e5: CavityReference(1.0e5, 4.519, 9.111, 34.73, 68.59),
    1.0e6: CavityReference(1.0e6, 8.800, 16.32, 64.63, 219.36),
}


def cavity(rayleigh: float) -> CavityReference:
    for key, value in CAVITY_REFERENCES.items():
        if abs(rayleigh / key - 1.0) < 1.0e-9:
            return value
    raise ValueError(f"no de Vahl Davis reference for Ra = {rayleigh:g}")