"""``GAP-FVM-TWOPHASE`` verification face: VOF interface capture, gated.

The repository already owns the *pieces* of a VOF interface-capture capability
(``tensorfvm.vof_plic`` geometric reconstruction, ``tensorfvm.twophase_bench``
algebraic CICSAM/HRIC transport and shape metrics, ``tensorfvm.two_phase``
PLIC dam break).  What was missing -- and what this package adds -- is a single
**runnable, machine-gated entry point** that turns those pieces into an audited
acceptance face with real artefacts:

===========  ===============================================================  ==========
Gate         Criterion                                                        Metric
===========  ===============================================================  ==========
Z1           Zalesak slotted disk, one revolution, declared ``N = 128``:     ``geometric_error``
             thresholded geometric error ``<= 5 %``
Z2           Zalesak mass conservation ``|A(T) - A(0)| / A(0) <= 0.1 %``       ``mass_drift_relative``
Z3           Zalesak perimeter change ``|P(T) - P(0)| / P(0) <= 10 %``         ``perimeter_error``
Z4           Zalesak boundedness: no cell outside ``[0, 1]`` beyond ``1e-12``  ``bound_violation``
Z5           geometric error is resolution-consistent: order ``>= 0.8``        ``convergence_order``
D1           dam break surge front vs Martin & Moyce (1952): ``<= 15 %``      ``front_max_rel_error``
D2           dam break column height vs Martin & Moyce (1952): ``<= 15 %``    ``height_max_rel_error``
D3           dam break liquid volume drift ``<= 1 %``                          ``volume_drift``
D4           dam break boundedness ``<= 1e-10``                                ``bound_violation``
D5           zero-gravity control: a resting interface must not move (``<=1e-14``)  ``control_max_abs_change``
B1           Hysing bubble terminal velocity vs ref (Eo=10, M=1e-3): ``<= 10 %``  ``terminal_velocity_rel_error``
B2           Hysing bubble minimum circularity vs ref: ``<= 10 %``             ``circularity_rel_error``
B3           Hysing bubble mass conservation: ``<= 1e-6``                      ``worst_step_drift``
B4           Hysing bubble convergence: order ``>= 0.5``                       ``convergence_order``
===========  ===============================================================  ==========

Every gate is *evaluated by the driver*, written to JSON next to the numbers it
judged, and re-read from that JSON when the report is rendered -- no number in
the report is typed by hand.

Entry points
------------
* ``python -m tensorfvm.twophase_gap`` (console script
  ``tensorfvm-twophase-gap``), see :mod:`.campaign` for the CLI;
* :meth:`tensorfvm.vof_phase0.solver.Phase0TwoPhaseSolver.verify_interface_transport`
  -- the production solver calls the Zalesak leg on its own VOF scheme before a
  two-phase run (integration caller).

Artefacts are written to ``artifacts/GAP-FVM-TWOPHASE`` (JSON + figures +
markdown); the report is ``docs/GAP-FVM-TWOPHASE/README.md``.
"""

from __future__ import annotations

from .campaign import (
    DEFAULTS,
    GATES,
    evaluate_gates,
    render_report,
    run_campaign,
    summarise,
)
from .dambreak_gate import (
    decompose_dambreak_error,
    quantisation_floor,
    run_dambreak_leg,
)
from .hysing_gate import run_hysing_leg
from .zalesak_gate import run_zalesak_leg

__all__ = [
    "DEFAULTS",
    "decompose_dambreak_error",
    "quantisation_floor",
    "GATES",
    "evaluate_gates",
    "render_report",
    "run_campaign",
    "run_dambreak_leg",
    "run_hysing_leg",
    "run_zalesak_leg",
    "summarise",
]
