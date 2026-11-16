"""Flux-family registry: the four flux families plus the overset transfer.

m23215 found the flux operators only inside their own domains (VOF PLIC /
CICSAM / THINC, compressible slope limiting) and one interface-generic piece
(``overset_flux.PlanarFluxTransfer``) with **zero ``src`` references** -- an
orphan reachable only through its test.  Registering it here is the wiring:
this module now imports it, so the transfer is part of the kernel surface
instead of a dead end.  (``pmm_solver`` remains unwired by design -- see
``docs/m23813/CORE_SKELETON.md`` for the recorded reason.)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from ...registry import capability
from .protocol import FluxFunction

__all__ = [
    "FluxEntry",
    "OversetTransferFlux",
    "register_flux",
    "get_flux",
    "list_families",
    "FLUX_FAMILIES",
]


class _FunctionFlux:
    """Adapter wrapping a module-level flux function."""

    def __init__(self, family: str, func: Callable[..., Any], doc: str = ""):
        self.family = family
        self._func = func
        self.doc = doc

    def flux(self, *args: Any, **kwargs: Any) -> Any:
        return self._func(*args, **kwargs)


class OversetTransferFlux:
    """Adapt a ``PlanarFluxTransfer`` instance to the ``FluxFunction`` protocol.

    ``flux(integrated_flux)`` delegates to the transfer's ``transfer`` method;
    the conservation check stays available as ``balance``.
    """

    family = "overset_transfer"

    def __init__(self, transfer):
        self.transfer = transfer

    def flux(self, integrated_flux: Any) -> Any:
        return self.transfer.transfer(integrated_flux)

    def balance(self, donor_flux: Any, receiver_flux: Any) -> Any:
        return self.transfer.balance(donor_flux, receiver_flux)


@dataclass(frozen=True)
class FluxEntry:
    """One registered family: a factory producing a ``FluxFunction``."""

    family: str
    factory: Callable[..., FluxFunction]
    doc: str = ""


FLUX_FAMILIES: dict[str, FluxEntry] = {}


def register_flux(family: str, factory: Callable[..., FluxFunction], *, doc: str = "") -> None:
    """Register a flux family (idempotent per process)."""
    if family in FLUX_FAMILIES:
        raise ValueError(f"flux family {family!r} is already registered")
    FLUX_FAMILIES[family] = FluxEntry(family, factory, doc)


def list_families() -> list[str]:
    """Names of all registered flux families, sorted."""
    return sorted(FLUX_FAMILIES)


@capability('fvm.core.flux_registry',
            kind='flux',
            maturity='developing',
            test_path='tests/test_core_flux.py',
            wired=True,
            tested=True,
            verified=False,
            evidence={
                'grade': 'MECHANICS',
                'note': ('four flux families (vof_plic, vof_cicsam, vof_thinc, '
                         'compressible_slope) plus the previously orphaned '
                         'overset_transfer, all delegated, none rewritten'),
                'docs': 1,
            })
def get_flux(family: str, *args: Any, **kwargs: Any) -> FluxFunction:
    """Instantiate the flux operator of a registered family."""
    entry = FLUX_FAMILIES.get(family)
    if entry is None:
        raise KeyError(f"unknown flux family {family!r}; known: {sorted(FLUX_FAMILIES)}")
    return entry.factory(*args, **kwargs)


def _register_defaults() -> None:
    from ... import vof_cicsam, vof_plic
    from ...compressible import reconstruct
    from ...overset_flux import PlanarFluxTransfer
    from ...vof import thinc

    register_flux("vof_plic",
                  lambda: _FunctionFlux("vof_plic", vof_plic.advect_plic,
                                        "geometric PLIC advection of the volume fraction"),
                  doc="VOF PLIC: advect_plic(alpha, u, v, dt, dx, dy, ...)")
    register_flux("vof_cicsam",
                  lambda: _FunctionFlux("vof_cicsam", vof_cicsam.cicsam_face_value,
                                        "CICSAM blended face value (Ubbink & Issa 1999)"),
                  doc="VOF CICSAM: cicsam_face_value(alpha_d, alpha_a, alpha_u, cour, gamma)")
    register_flux("vof_thinc",
                  lambda: _FunctionFlux("vof_thinc", thinc.thinc_flux,
                                        "THINC analytic strip flux"),
                  doc="VOF THINC: thinc_flux(alpha, courant, sharpness=3.0)")
    register_flux("compressible_slope",
                  lambda: _FunctionFlux("compressible_slope", reconstruct.limited_slope,
                                        "TVD-limited cell-centred slope"),
                  doc="compressible slope limiting: limited_slope(q, axis, limiter='mc')")
    register_flux("overset_transfer",
                  lambda transfer: OversetTransferFlux(transfer),
                  doc="overset mortar transfer: flux(transfer_instance)(integrated_flux)")


_register_defaults()
