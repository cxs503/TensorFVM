# Native 3D SUBOFF manufactured calculation

Reconstruction: **quadratic**. Initial state: `{'kind': 'previous numerical field; not analytic reference', 'sha256': '09fb259c5a5fb4d4850ab9e1a1d1ecb404873777e8747843cec2ab885fc0f9fc', 'origin': 'docs/verification-native-suboff-mms/medium', 'producer_converged': True, 'producer_result_sha256': 'e3f326a01d041bd83e00d73b27245546778a20c363d3bd524212d1263add7964'}`.

Actual mixed-cell body-fitted mesh: 24,284 control volumes.
Converged: **True**. Independent manufactured field gate: **True**.
Experimental drag/Cp qualification: **false**. Spatial refinement qualification: **false**.

| Relative volume-weighted L2 metric | Error | Strictly below 3% |
|---|---:|---|
| velocity | 0.0076091% | True |
| velocity_x | 0.00944047% | True |
| velocity_y | 0.00432881% | True |
| velocity_z | 0.0118426% | True |
| pressure | 0.400184% | True |

The manufactured solution prescribes nonzero velocity at the hull and analytic
continuum forcing; it is a discretization verification, not physical SUBOFF flow.
Pressure is solved from coupled momentum and continuity, not a velocity potential.
See `config.json` for normalization, `result.json` for current reconstructed mass
and momentum residuals, and `history.jsonl` for every completed nonlinear update.
The input mesh and imported numerical sources are frozen with SHA256 bindings.
`fields.npz` contains raw cell fields and native face fluxes/tractions.

Failure, if any: `None`.
CPU wall time: 3995.766 seconds. GPU has not been tested.
