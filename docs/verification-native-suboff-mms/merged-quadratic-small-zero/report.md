# Native 3D SUBOFF manufactured calculation

Reconstruction: **quadratic**. Initial state: `None`.

Actual mixed-cell body-fitted mesh: 6,085 control volumes.
Converged: **False**. Independent manufactured field gate: **False**.
Experimental drag/Cp qualification: **false**. Spatial refinement qualification: **false**.

| Relative volume-weighted L2 metric | Error | Strictly below 3% |
|---|---:|---|
| velocity | 0.0388213% | True |
| velocity_x | 0.0443974% | True |
| velocity_y | 0.0323662% | True |
| velocity_z | 0.0482404% | True |
| pressure | 9.22286% | False |

The manufactured solution prescribes nonzero velocity at the hull and analytic
continuum forcing; it is a discretization verification, not physical SUBOFF flow.
Pressure is solved from coupled momentum and continuity, not a velocity potential.
See `config.json` for normalization, `result.json` for current reconstructed mass
and momentum residuals, and `history.jsonl` for every completed nonlinear update.
The input mesh and imported numerical sources are frozen with SHA256 bindings.
`fields.npz` contains raw cell fields and native face fluxes/tractions.

Failure, if any: `{'type': 'RuntimeError', 'message': 'Quadratic nonlinear step rejected: no residual decrease in 16 trials; previous state preserved'}`.
CPU wall time: 430.933 seconds. GPU has not been tested.
