# Historical rejected-candidate artifact

This run exposed the state-commit bug. It reports zero completed momentum steps
and initialization_only, but its saved velocity/pressure already contain the
rejected linear candidate (max speed 1245.325 U). It must not be interpreted as
an accepted physical iteration, a convergence result or an experimental metric.
The original fields/config/sources are retained byte for byte. See ../README.md
and ../suboff-state-preserved for the corrected failure semantics.
