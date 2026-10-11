# Actual quadratic SUBOFF solve: first inner solve failed

The producer ran for 1281.207 CPU seconds and failed its first GMRES solve:
1,800 inner iterations, true scaled residual 1.78420924797e-5. There were
**zero completed nonlinear updates**. No `history.jsonl` was created; the
producer report's generic history link does not imply a missing successful solve.

The raw velocity and pressure arrays in `fields.npz` are exactly equal to the
explicitly copied baseline numerical initialization in
`initial-numerical-state.npz` (maximum absolute difference zero). Thus the
reported 5.49992% pressure error remains the old numerical starting field.
It is **not a converged quadratic FVM result**, and no new drag/Cp qualification
can be inferred from the saved forces.

The first preconditioner is already a fresh baseline coupled LU. Refreshing
cached LU or adding nonlinear backtracking cannot repair this first inner
failure. The next concrete gate is a preconditioner matching the quadratic
momentum/pressure operator, verified against its true residual on mixed meshes.
This mesh has maximum nonorthogonality 88.7745 degrees and minimum cell volume
6.26419e-10. More inner iterations alone is not established as a solution.

The original producer report, configuration, input mesh, raw fields and source
snapshot are preserved. This note describes the failed result after completion;
it does not amend the producer's recorded numerical outcome.
