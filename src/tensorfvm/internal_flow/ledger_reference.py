"""Reference values used by the internal-flow ledger study, with evidence grades.

Grades (platform vocabulary):

``REAL``        a closed-form/analytic value, or a benchmark table that is
                reproduced identically by several independent sources;
``WITHHELD``    deliberately *not* used as a hard criterion: no citable number is
                available offline in this worktree, or the convention/geometry of
                the available data does not match this 2-D laminar case.  The value
                is reported as an orientation band only.

The hard gates of this study are therefore (i) the control-volume momentum
residual, (ii) the agreement between the pressure-tap Cp and the momentum-ledger
Cp (two routes over the same field), (iii) the grid-converged reattachment and
(iv) the cavity Nusselt numbers against the de Vahl Davis (1983) table, which is
reproduced from :mod:`tensorfvm.inc_bench.reference` so both surfaces quote the
same digits.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..inc_bench import reference as _inc


@dataclass(frozen=True)
class Anchor:
    name: str
    value: float
    units: str
    source: str
    grade: str
    tolerance: float | None = None
    note: str = ""

    def as_dict(self) -> dict[str, object]:
        return {"name": self.name, "value": self.value, "units": self.units,
                "source": self.source, "grade": self.grade,
                "tolerance": self.tolerance, "note": self.note}


def cavity(rayleigh: float) -> _inc.CavityReference:
    """de Vahl Davis (1983) square-cavity benchmark, ``Pr = 0.71``."""
    return _inc.cavity(rayleigh)


def cavity_anchor(rayleigh: float) -> Anchor:
    ref = cavity(rayleigh)
    return Anchor(
        name=f"Nu(Ra={rayleigh:g}) de Vahl Davis 1983", value=ref.nusselt, units="-",
        source=ref.source, grade=ref.grade, tolerance=0.03,
        note="Nu is the mean hot-wall Nusselt number; psi/u/v normalised by alpha")


def borda_carnot(area_ratio: float) -> Anchor:
    """Inviscid momentum-balance pressure recovery of a sudden expansion.

    ``Cp = 2 s (1 - s)`` with ``s = A_in/A_out = 1/AR`` follows from the
    control-volume momentum balance for uniform profiles, uniform corner pressure
    ``p = p_in`` and no wall friction (Borda-Carnot / momentum theory).
    """
    s = 1.0 / area_ratio
    return Anchor(
        name=f"Cp_BordaCarnot(AR={area_ratio:g})", value=2.0 * s * (1.0 - s), units="-",
        source="exact control-volume momentum balance (Borda-Carnot), uniform "
               "profiles, no wall friction", grade="REAL", tolerance=None,
        note="viscous excess at finite Re is quantified by the ledger terms "
             "dcp_viscous / dcp_step_face, not absorbed into this anchor")


def plane_channel_friction(reynolds_h: float) -> Anchor:
    """Fully developed plane-Poiseuille friction coefficient ``C_f = 12/Re_h``."""
    return Anchor(
        name="C_f fully developed plane channel", value=12.0 / reynolds_h, units="-",
        source="analytic plane Poiseuille profile, C_f = 12/Re_h (h = channel "
               "height)", grade="REAL", tolerance=0.15)


def armaly_reattachment(reynolds: float) -> Anchor:
    """Orientation band for the 2-D backward-facing-step reattachment length.

    Armaly et al. (1983) measured ``x_r/S`` in a channel of expansion ratio
    1.9423 with ``Re = U_mean 2 h_in / nu``; the two-dimensional limit up to
    ``Re ~ 400`` is the accepted comparator for a 2-D solver.  The commonly quoted
    2-D values (``x_r/S ~ 3.0`` at ``Re = 100`` and ``~ 8.1`` at ``Re = 400``) are
    **not** reproduced from an offline copy of the paper inside this worktree;
    they are therefore graded ``WITHHELD`` and used as a +/-10 % orientation band,
    never as a hard gate.  The hard gate for this quantity is the grid-converged
    (Richardson) value plus the agreement of the two wall diagnostics.
    """
    table = {100.0: 3.0, 400.0: 8.1}
    key = min(table, key=lambda k: abs(k - reynolds))
    return Anchor(
        name=f"x_r/S Armaly 1983 (Re={key:g})", value=table[key], units="step heights",
        source="Armaly, Durst, Pereira & Schoenung, JFM 127 (1983) 473",
        grade="WITHHELD", tolerance=0.10,
        note="orientation band only: digits not verified against an offline copy in "
             "this worktree; 3-D effects are documented to start above Re ~ 400")


def sudden_expansion_cp(area_ratio: float) -> Anchor:
    """Placeholder for a laminar 2-D sudden-expansion Cp measurement."""
    return Anchor(
        name=f"Cp measured, planar laminar sudden expansion AR={area_ratio:g}",
        value=float("nan"), units="-",
        source="no citable 2-D planar laminar dataset available offline",
        grade="WITHHELD", tolerance=None,
        note="the classic abrupt-enlargement loss data (Benedict et al. 1966) are "
             "turbulent pipe measurements; Macagno & Hung (1967) give laminar "
             "reattachment lengths, not pressure recovery. Neither is used as a "
             "hard gate; the gate is the two-route Cp agreement of this report")