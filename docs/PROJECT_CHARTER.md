# TensorFVM — finite-volume solver charter

**Canonical ecosystem vision:** [Tensor Ecosystem: Vision for the AI Era](https://github.com/cxs503/TensorPlatform/blob/main/docs/architecture/TENSOR_ECOSYSTEM_AI_ERA_VISION.md)

## Mission
Provide independently usable finite-volume numerical methods for fluid and related transport problems, with explicit formulations, reproducible verification cases, and honest statements of supported physics.

## Owns
- Finite-volume mesh/face/cell data models, fluxes, reconstruction, pressure-velocity coupling, boundary conditions, turbulence and transport models actually implemented.
- Solver-local diagnostics, convergence controls, conservation ledgers and numerical failure behavior.
- Reproducible analytical, manufactured-solution, canonical-flow and external-reference benchmarks.
- Solver-specific assumptions, limitations, and evidence for each advertised capability.

## Does not own
- General-purpose agent runtime, cross-domain orchestration, shared platform identity/governance, or ecosystem-wide task scheduling.
- All scientific acceptance policy for other solvers or domains.
- Claims of physical validation based only on convergence, a successful run, or agreement with a single unverified reference.

## Integration
Remain independently runnable and expose documented inputs, mesh/units, solver configuration, residuals, conservation diagnostics, outputs and run provenance. TensorPhysics owns scientific workflow coordination and domain-level acceptance; TensorPlatform owns shared domain-neutral infrastructure. Large computed results should be reproducible artifacts, not unnecessarily committed into source control.

## Quality gates
Distinguish code tests, equation/discretization verification, iterative convergence, grid/time-step sensitivity, conservation, comparison against analytical/manufactured solutions, and validation against trusted experimental or external solver references. Record reference provenance and limitations; do not claim a high-Re engineering case is validated merely because a lower-level case passes.

## Agent rules
Read this charter, README.rst and relevant verification records before work; state ownership and affected contracts; keep specialist numerical methods here; avoid private cross-repository imports; add reproducible tests and benchmark records; report exact commands, failed cases and untested paths; separate implementation from verification and physical validation.

## Maturity
This charter defines responsibility, not blanket validation of the repository. Use the individual benchmark records and their stated limitations to assess capability maturity.
