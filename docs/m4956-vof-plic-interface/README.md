# IN14-D10 — two-phase VOF: PLIC interface transport, minimal verification surface

Module under test: [`src/tensorfvm/vof_plic.py`](../../src/tensorfvm/vof_plic.py)
Verification harness: [`src/tensorfvm/verification/vof_plic_interface.py`](../../src/tensorfvm/verification/vof_plic_interface.py)
Tests: [`tests/test_vof_plic_interface.py`](../../tests/test_vof_plic_interface.py) (24 cases)

```bash
python3 scripts/verify_vof_plic_interface.py --output docs/m4956-vof-plic-interface
pytest tests/test_vof_plic_interface.py -q
```

Artefacts in this directory: `report.md` (auto-generated, numbers injected
programmatically), `summary.json` (all raw metrics + criteria), and four figures
(`fig-1d-step-interface.png`, `fig-zalesak-fields.png`,
`fig-zalesak-metrics.png`, `fig-mass-ledger.png`).

## Why this node exists

Phase 0 audited two-phase flow as the largest gap of the four solver libraries.
The repository already had an *algebraic* donor-cell transport
(`two_phase_vof.VOFTransport`) plus a Zalesak rotation study
(`verification/vof_rotation.py`). Those artefacts show what the missing piece
costs on a discontinuous indicator field: peak volume fraction retained to only
19 % / 31 % / 52 % at 32²/64²/128² and relative L1 error 1.19–1.77 — the slotted
disk is smeared away. For interface-dominated ship hydrodynamics (resistance of a
partly immersed hull, sloshing, green water on deck) that is the dominant error,
not the momentum discretisation.

`vof_plic.py` supplies the missing ingredient: a **geometric** reconstruction
(Youngs normal + exact plane constant from the closed-form cut-area inverse) and
flux-form advection whose face flux is the exact reconstructed-liquid volume in
the swept strip of the upwind cell. Everything in this node is measured against an
exact reference, not asserted.

## The four things the node had to clear

| # | item | criterion | measured (PLIC) | donor-cell baseline |
|---|---|---|---|---|
| 1 | 1-D step interface sharpness | 10–90 % width ≤ 2 cells | **1 cell at N = 32/64/128**; L1 = 0.4·Δx | 6 / 9 / 13 cells, L1 0.120/0.084/0.060 |
| 2 | Zalesak, one revolution | area error < 5 % | **8.8e-15 (N=64), 6.4e-16 (N=128)** | 0 (exactly conservative) |
| 3 | global mass conservation | < 1e-6 per step | **6.6e-16 worst step (Zalesak), 0.0 (1-D, exact)** | 3.3e-16 |
| 4 | negative evidence | invalid input rejected | 7/7 guards raise `ValueError` | — |

Shape metrics that the area criterion cannot see (a scheme returning a uniform
`α = 0.5` field conserves volume exactly and passes it): peak retention
**1.000 at N=64 and N=128** for PLIC against 0.332 / 0.443 for donor cell, L1
6.1e-3 / 2.6e-3 against 6.3e-2 / 5.8e-2, distorted fraction 0.018 / 0.007 against
0.230 / 0.194. The report states this explicitly rather than letting the 5 %
criterion stand alone.

## Geometry口径 provenance (plan-13 P2 fluid leg)

The Zalesak geometry is **imported, not re-implemented**:
`verification/vof_rotation.py::zalesak_disk` (`domain = 1`, disk centre
`(0.5, 0.75)`, radius `0.15`, slot half-width `0.05`) and
`rotation_velocity` (angular rate `2π`, `Δx = 1/N`, discrete divergence exactly
zero). The harness asserts that the `Δx` of the rotation field equals the `Δx` of
the geometry, so the two legs of the plan cannot silently drift apart.

The donor-cell reference is the *stencil* of `VOFTransport(scheme="upwind")`
re-expressed with periodic boundaries. This fairness matters: the shipped
transport uses transmissive zero-gradient ghosts, and on the same 1-D step it
creates **+60 % volume** over 48 steps (measured, `summary.json` →
`library_upwind_transmissive`). That is a boundary-condition artefact, not a flux
error, and it is reported separately instead of being mixed into the scheme
comparison.

## Negative evidence

`validate_alpha` rejects (does not clip) a volume fraction outside `[0, 1]`,
`validate_velocity` rejects `dt ≤ 0`, non-finite fields, wrong staggered shapes
and directional CFL > 1, and `validate_directional_speed` rejects a negative
transport-rate magnitude. Signed face velocities (`u < 0`) remain **legal** —
they are physical back-flow; the module documents the distinction and a test
asserts the mirror symmetry of leftward advection.

## Honest limitations

* 2-D uniform Cartesian PLIC only. No 3-D polygon-clipping reconstruction, no
  unstructured-mesh variant: both are roadmap items.
* Dimensional (Lie) splitting with alternating sweep order — formally first order
  in time. A Strang split or an unsplit geometric flux is the next accuracy step.
* `LinF = 1.0` on Zalesak at N ≤ 128: the slot tip is a sub-cell feature, so one
  cell can be entirely wrong even though the mean error is 2.6e-3. Reported, not
  hidden.
* Surface tension (CSF) and phase change are **out of scope** here and declared
  roadmap items: CSF is audited in `docs/two-phase-csf-balance` and
  `docs/two-phase-surface-tension-audit`; phase change has no module yet.
* The scheme is verified for divergence-free velocity fields (rigid rotation) and
  for uniform 1-D translation; a strongly compressible/divergent field is not
  covered.