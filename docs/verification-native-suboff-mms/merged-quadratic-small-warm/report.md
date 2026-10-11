# Native 3D SUBOFF manufactured calculation

Reconstruction: **quadratic**. Initial state: `{'kind': 'previous numerical field; not analytic reference', 'sha256': '01393e10b948bc8f3fbb197cb655631456509552c0d64d46aa7cf5c91bff7550', 'origin': 'docs/verification-native-suboff-mms/merged-linear-small', 'producer_converged': False, 'producer_result_sha256': 'cd870383c31d3af7ff15258aa7ac3b931234d2627930d25a7d826f65fc363105'}`.

Actual mixed-cell body-fitted mesh: 6,085 control volumes.
Converged: **True**. Independent manufactured field gate: **True**.
Experimental drag/Cp qualification: **false**. Spatial refinement qualification: **false**.

| Relative volume-weighted L2 metric | Error | Strictly below 3% |
|---|---:|---|
| velocity | 0.0158491% | True |
| velocity_x | 0.0202514% | True |
| velocity_y | 0.00923546% | True |
| velocity_z | 0.0236722% | True |
| pressure | 0.772845% | True |

The manufactured solution prescribes nonzero velocity at the hull and analytic
continuum forcing; it is a discretization verification, not physical SUBOFF flow.
Pressure is solved from coupled momentum and continuity, not a velocity potential.
See `config.json` for normalization, `result.json` for current reconstructed mass
and momentum residuals, and `history.jsonl` for every completed nonlinear update.
The input mesh and imported numerical sources are frozen with SHA256 bindings.
`fields.npz` contains raw cell fields and native face fluxes/tractions.

Failure, if any: `None`.
CPU wall time: 299.806 seconds. GPU has not been tested.
