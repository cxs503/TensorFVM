# Native 3D SUBOFF manufactured calculation

Actual mixed-cell body-fitted mesh: 11,648 control volumes.
Converged: **False**. Independent manufactured field gate: **False**.
Experimental drag/Cp qualification: **false**. Spatial refinement qualification: **false**.

| Relative volume-weighted L2 metric | Error | Strictly below 3% |
|---|---:|---|
| velocity | 18.1385% | False |
| velocity_x | 25.9717% | False |
| velocity_y | 12.0341% | False |
| velocity_z | 20.3501% | False |
| pressure | 425.519% | False |

The manufactured solution prescribes nonzero velocity at the hull and analytic
continuum forcing; it is a discretization verification, not physical SUBOFF flow.
Pressure is solved from coupled momentum and continuity, not a velocity potential.
See `config.json` for normalization, `result.json` for current reconstructed mass
and momentum residuals, and `history.jsonl` for every completed nonlinear update.
The input mesh and imported numerical sources are frozen with SHA256 bindings.
`fields.npz` contains raw cell fields and native face fluxes/tractions.

Failure, if any: `None`.
CPU wall time: 154.309 seconds. GPU has not been tested.
