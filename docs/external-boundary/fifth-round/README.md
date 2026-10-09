# Independent fifth-round diagnostic audit

`passive-report.json` independently reconstructs two actual FEM passive rigid
feedback cases and six balanced local moving-boundary cases using NumPy.
Raw artifacts, external solver sources and this auditor are SHA256 bound.
No external solver or audit function is imported.

The energy functional is combined fluid/body kinetic energy plus the fluid
isothermal density free energy, with SI scale rho_ref * thickness * dx^4 /
(3 dt^2). The reference density is 1 and the periodic integer disk mask is
reconstructed from raw center/radius. The initial total is 0.00225 J; the
fixed budget is 0.002252250001 J. Accepted final states remain below this
budget. First crossing candidates at 0.017/0.0175 s contain 0.25830021 /
1.03070647 J, of which 99.36% / 99.85% is density free energy, and are rejected.
Every body impulse and midpoint kinetic-work identity is independently checked.

Rollback/restart exactness comes from the producer's actual exact replay;
the independent audit checks their raw accepted/candidate history prefix,
clocks, energy values and attestations. It does not independently execute the
foreign solver's rollback implementation. Rejections stop before requested
0.3 s duration; they do not establish a repaired pressure boundary, complete
LBM entropy stability or physical qualification.

The six balanced-boundary cases reconstruct raw mass, momentum, wall and
conversion impulses. Conversion momentum is near roundoff, while the crossing
co-moving disk still accumulates a large artificial impulse. This new backend
is not qualified as a physical repair. Diffusive refinement results remain
explicit in the report.

```sh
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=src python scripts/audit_passive_feedback.py \
  --fem-root /data/TensorFEM --lbm-root ../TensorLBM
PYTHONPATH=src python -m pytest -q tests/test_passive_feedback_audit.py
```

Six tests check zero reference energy, body kinetic energy, positive compression
energy, invalid populations, false qualification and a fabricated rejection
within budget. Earlier fourth-round and 23-case reports remain unchanged.
