# Native 3D SUBOFF manufactured calculation

Reconstruction: **quadratic**. Initial state: `{'kind': 'previous numerical field; not analytic reference', 'sha256': 'a484e93d10e4f84c2ad619dee83d6254e4cae552998aa84e7d8c10bf4528ee94', 'origin': 'docs/verification-native-suboff-mms/tail-transition-transferred-seed', 'producer_converged': False, 'producer_result_sha256': '9ca766c5f9e471ab7595fbbed20a59597d8869c311c20b2fe27e2fb517395635'}`.

Actual mixed-cell body-fitted mesh: 9,068 control volumes.
Converged: **True**. Independent manufactured field gate: **True**.
Experimental drag/Cp qualification: **false**. Spatial refinement qualification: **false**.

| Relative volume-weighted L2 metric | Error | Strictly below 3% |
|---|---:|---|
| velocity | 0.0154511% | True |
| velocity_x | 0.0191457% | True |
| velocity_y | 0.00900536% | True |
| velocity_z | 0.0238297% | True |
| pressure | 0.746506% | True |

The manufactured solution prescribes nonzero velocity at the hull and analytic
continuum forcing; it is a discretization verification, not physical SUBOFF flow.
Pressure is solved from coupled momentum and continuity, not a velocity potential.
See `config.json` for normalization, `result.json` for current reconstructed mass
and momentum residuals, and `history.jsonl` for every completed nonlinear update.
The input mesh and imported numerical sources are frozen with SHA256 bindings.
`fields.npz` contains raw cell fields and native face fluxes/tractions.

Failure, if any: `None`.
CPU wall time: 607.702 seconds. GPU has not been tested.
