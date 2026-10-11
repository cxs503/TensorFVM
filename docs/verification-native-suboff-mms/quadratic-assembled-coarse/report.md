# Native 3D SUBOFF manufactured calculation

Reconstruction: **quadratic**. Initial state: `{'kind': 'previous numerical field; not analytic reference', 'sha256': 'eef0ab63909df3a2f7a5ea166297a858688d888d3419b7db7336754951dd4c43', 'origin': 'docs/verification-native-suboff-mms/coarse'}`.

Actual mixed-cell body-fitted mesh: 11,648 control volumes.
Converged: **True**. Independent manufactured field gate: **True**.
Experimental drag/Cp qualification: **false**. Spatial refinement qualification: **false**.

| Relative volume-weighted L2 metric | Error | Strictly below 3% |
|---|---:|---|
| velocity | 0.0112074% | True |
| velocity_x | 0.0149918% | True |
| velocity_y | 0.00681589% | True |
| velocity_z | 0.0154045% | True |
| pressure | 0.524758% | True |

The manufactured solution prescribes nonzero velocity at the hull and analytic
continuum forcing; it is a discretization verification, not physical SUBOFF flow.
Pressure is solved from coupled momentum and continuity, not a velocity potential.
See `config.json` for normalization, `result.json` for current reconstructed mass
and momentum residuals, and `history.jsonl` for every completed nonlinear update.
The input mesh and imported numerical sources are frozen with SHA256 bindings.
`fields.npz` contains raw cell fields and native face fluxes/tractions.

Failure, if any: `None`.
CPU wall time: 841.099 seconds. GPU has not been tested.
