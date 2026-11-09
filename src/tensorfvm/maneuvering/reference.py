"""Benchmark records for the KVLCC2 manoeuvring validation face -- with flags.

**Read this before quoting any number from this module.**  No provenance-verified
SIMMAN 2008 experimental table (turning circle, zig-zag, rudder derivative) exists
in this workspace: there is no primary workshop table on disk and no network route
to one.  Node IN40-N04 (``docs/pmm-cfd``) additionally audited the in-repository
``tensorfvm.simman2021_reference`` derivative table and found it internally
inconsistent (its own drift table implies derivatives 3.5-6.3x larger than its
derivative list, and its added-mass entry implies ``m22/(rho nabla) = 6.6 %``
where potential theory requires 50-100 %).  That table is therefore **not** usable
as a validation reference and is not used here.

What this module provides instead is an explicit, machine-readable record of the
*status* of each candidate reference:

``status="withheld_band"``
    A literature-recalled *band* for a full single-screw tanker
    (``Cb ~ 0.81``, ``Fn = 0.142``).  It is a plausibility band, not an
    experimental value: a prediction landing inside it is "consistent", never
    "validated".
``status="withheld"``
    A quantity with no reference at all in this workspace (the SIMMAN 2008 EFD
    turning/zig-zag/rudder table).  Criteria that need it are reported as
    WITHHELD, not PASS/FAIL.

Consumers must propagate ``status`` into their verdict, exactly as
``rans_ship_resistance.anchors`` requires for the resistance anchor.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

__all__ = [
    "ReferenceBand", "KVLCC2_TURNING_BAND", "KVLCC2_ZIGZAG_BAND",
    "SIMMAN2008_EFD", "TURNING_CRITERIA", "band_check", "criteria_table",
]


@dataclass(frozen=True)
class ReferenceBand:
    """A literature band with its provenance status."""

    quantity: str
    low: float
    high: float
    unit: str
    status: str
    basis: str
    note: str = ""

    @property
    def central(self) -> float:
        return 0.5 * (self.low + self.high)

    def contains(self, value: float) -> bool:
        return self.low <= value <= self.high

    def as_dict(self) -> Dict[str, object]:
        return {"quantity": self.quantity, "low": self.low, "high": self.high,
                "central": self.central, "unit": self.unit, "status": self.status,
                "basis": self.basis, "note": self.note}


#: Turning-circle bands for a full single-screw tanker at 35 deg rudder.
#: Recalled engineering spread (PNA Vol. III controllability chapter figures and
#: the general VLCC literature); **not** a digitized workshop table.
KVLCC2_TURNING_BAND = {
    "tactical_diameter_over_L": ReferenceBand(
        quantity="tactical_diameter_over_L", low=2.8, high=3.8, unit="-",
        status="withheld_band",
        basis="literature-recalled spread for Cb~0.8 single-screw tankers, 35 deg rudder",
        note="plausibility band only; no digitized SIMMAN 2008 EFD table in workspace"),
    "advance_over_L": ReferenceBand(
        quantity="advance_over_L", low=2.6, high=3.6, unit="-",
        status="withheld_band",
        basis="literature-recalled spread for Cb~0.8 single-screw tankers, 35 deg rudder",
        note="plausibility band only"),
    "transfer_over_L": ReferenceBand(
        quantity="transfer_over_L", low=1.2, high=1.9, unit="-",
        status="withheld_band",
        basis="literature-recalled spread for Cb~0.8 single-screw tankers, 35 deg rudder",
        note="plausibility band only"),
    "steady_drift_deg": ReferenceBand(
        quantity="steady_drift_deg", low=8.0, high=18.0, unit="deg",
        status="withheld_band",
        basis="literature-recalled spread for Cb~0.8 single-screw tankers",
        note="plausibility band only"),
}

#: 10/10 zig-zag bands (first/second overshoot).
KVLCC2_ZIGZAG_BAND = {
    "psi_1_deg": ReferenceBand(
        quantity="psi_1_deg", low=4.0, high=15.0, unit="deg",
        status="withheld_band",
        basis="literature-recalled spread for full tankers, 10/10 zig-zag",
        note="plausibility band only; second overshoot typically exceeds the first"),
    "psi_2_deg": ReferenceBand(
        quantity="psi_2_deg", low=6.0, high=22.0, unit="deg",
        status="withheld_band",
        basis="literature-recalled spread for full tankers, 10/10 zig-zag",
        note="plausibility band only"),
}

#: The reference this node *would* need, and why it is not used.
SIMMAN2008_EFD = {
    "status": "withheld",
    "reason": ("no primary SIMMAN 2008 (Stern & Agdrup) table for KVLCC2 turning "
               "circle / zig-zag / rudder force exists in this workspace and no "
               "network route to one was available"),
    "in_repo_substitute_rejected": ("tensorfvm.simman2021_reference -- audited by "
                                    "node IN40-N04 as internally inconsistent"),
}

#: Which acceptance criteria of the node can be adjudicated in this workspace.
TURNING_CRITERIA = {
    "a_self_propulsion_closure": {
        "target": "|T - R_T/(1-t)| / (R_T/(1-t)) <= 2 %",
        "reference": "internal (thrust balance)",
        "status": "adjudicable",
    },
    "b_rudder_derivative": {
        "target": "F_y_delta within 15 % of SIMMAN 2008 EFD",
        "reference": "SIMMAN 2008 EFD",
        "status": "withheld",
    },
    "c_tactical_diameter": {
        "target": "T_D/L within 20 % of EFD",
        "reference": "SIMMAN 2008 EFD",
        "status": "withheld",
    },
    "d_first_overshoot": {
        "target": "psi_1 within 15 % of EFD",
        "reference": "SIMMAN 2008 EFD",
        "status": "withheld",
    },
    "e_mmg_arbitration": {
        "target": "|T_D(MMG, CFD rudder) - T_D(direct CFD) | / T_D <= 10 %",
        "reference": "internal (two force models)",
        "status": "adjudicable",
    },
    "f_grid_convergence": {
        "target": "GCI <= 5 % (three meshes; time-step convergence for the manoeuvre)",
        "reference": "internal (Richardson)",
        "status": "adjudicable",
    },
    "g_negative_control": {
        "target": "delta = 0 => F_y = 0, M_z = 0 (symmetric flow)",
        "reference": "internal (symmetry)",
        "status": "adjudicable",
    },
}


def band_check(value: float, band: ReferenceBand) -> Dict[str, object]:
    """Locate ``value`` in a band and state what that does and does not mean."""
    inside = band.contains(value)
    return {
        "quantity": band.quantity, "value": value, "band": [band.low, band.high],
        "inside_band": bool(inside),
        "offset_from_central_pct": 100.0 * (value - band.central) / band.central,
        "reference_status": band.status,
        "verdict": ("consistent with literature band (NOT a validation)"
                    if inside else "outside the literature band"),
    }


def criteria_table() -> Dict[str, Dict[str, object]]:
    return {k: dict(v) for k, v in TURNING_CRITERIA.items()}