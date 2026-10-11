# Native 3D SUBOFF manufactured calculation

Reconstruction: **quadratic**. Initial state: `{'kind': 'previous numerical field; not analytic reference', 'sha256': 'a5524548e4ce43359630ab6699e4bc3f7cbcc845c5f2d9f2b3d71dcbc2444563', 'origin': 'docs/verification-native-suboff-mms/tail-transition-quadratic', 'producer_converged': True, 'producer_result_sha256': '723b0d2e9746f8127cc06de493e48b399cd119af76b4280999667316f179ae3b'}`.
Viscous reconstruction: **cubic** (when quadratic flow is selected).

Actual mixed-cell body-fitted mesh: 9,068 control volumes.
Converged: **True**. Independent manufactured field gate: **True**.
Experimental drag/Cp qualification: **false**. Spatial refinement qualification: **false**.

| Relative volume-weighted L2 metric | Error | Strictly below 3% |
|---|---:|---|
| velocity | 0.0135567% | True |
| velocity_x | 0.0165396% | True |
| velocity_y | 0.00755077% | True |
| velocity_z | 0.02164% | True |
| pressure | 0.861708% | True |

The manufactured solution prescribes nonzero velocity at the hull and analytic
continuum forcing; it is a discretization verification, not physical SUBOFF flow.
Pressure is solved from coupled momentum and continuity, not a velocity potential.
See `config.json` for normalization, `result.json` for current reconstructed mass
and momentum residuals, and `history.jsonl` for every completed nonlinear update.
The input mesh and imported numerical sources are frozen with SHA256 bindings.
`fields.npz` contains raw cell fields and native face fluxes/tractions.

Failure, if any: `None`.
CPU wall time: 739.100 seconds. GPU has not been tested.
