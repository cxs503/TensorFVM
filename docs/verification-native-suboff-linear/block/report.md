# Native 3D SUBOFF manufactured calculation

Actual mixed-cell body-fitted mesh: 11,648 control volumes.
Converged: **False**. Independent manufactured field gate: **False**.
Experimental drag/Cp qualification: **false**. Spatial refinement qualification: **false**.

| Relative volume-weighted L2 metric | Error | Strictly below 3% |
|---|---:|---|
| velocity | 100% | False |
| velocity_x | 100% | False |
| velocity_y | 100% | False |
| velocity_z | 100% | False |
| pressure | 100% | False |

The manufactured solution prescribes nonzero velocity at the hull and analytic
continuum forcing; it is a discretization verification, not physical SUBOFF flow.
Pressure is solved from coupled momentum and continuity, not a velocity potential.
See `config.json` for normalization, `result.json` for current reconstructed mass
and momentum residuals, and `history.jsonl` for every completed nonlinear update.
The input mesh and imported numerical sources are frozen with SHA256 bindings.
`fields.npz` contains raw cell fields and native face fluxes/tractions.

Failure, if any: `{'type': 'RuntimeError', 'message': 'current coupled block GMRES failed: 60, 6000 iterations'}`.
CPU wall time: 472.116 seconds. GPU has not been tested.
