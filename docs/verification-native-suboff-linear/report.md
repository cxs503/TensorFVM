# Native SUBOFF linear solver diagnostics

Same 11,648-cell body-fitted grid and independent ABC continuum forcing.
These tests request **one nonlinear update only**; they do not compute a
converged flow or qualify its spatial/physical accuracy.

| Linear method | Outcome | Linear iterations | Total CPU wall time |
|---|---|---:|---:|
| Separate velocity ILU / orthogonal pressure preconditioner | Failed current GMRES after 60 restart cycles | 6,000 | 472.12 s |
| Exact velocity LU / actual sparse SIMPLE Schur approximation | First coupled linearization solved; true relative residual 5.30e-11 | 238 | 154.31 s |

The Schur approach uses the actual shared mass and conservative pressure operators
in its sparse approximation. The full coupled current matrix defines the solve;
its residual is independently checked. Momentum, reconstructed continuity and
field changes are still nonzero after the one update, so convergence is false.
The failed block approach remains opt-in and is not recommended for this grid.
These reference CPU methods do not establish scalability to 100,000 cells,
GPU acceleration, or a physical SUBOFF result. Input, settings, raw starting/one-step
fields, imported sources, and complete failure metadata are frozen in each case.
