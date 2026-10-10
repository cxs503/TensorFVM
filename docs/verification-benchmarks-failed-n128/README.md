# Retained 128-square-grid nonlinear failure

The original suite stopped at nx=ny=128, nz=4, dt=0.0125 s, mu=0.1 Pa s.
An isolated exact-configuration replay records every accepted velocity state,
then the rejected candidate/check/tentative/pressure arrays. Nonfinite candidate
values are deliberately retained in NPZ. Velocity, clock and history remain at
the last accepted state when this candidate is rejected; this does not mean no
earlier steps were accepted.

Preceding channel and 32/64 vortex raw results from the aborted suite remain.
The final formal spatial study uses 32/48/64 at fixed dt=0.0125. It cannot
overwrite this 128-grid failure. A smaller time step or another nonlinear solver
is required to qualify the 128 grid; this report asserts neither qualification.
Reproduce using scripts/reproduce_periodic_mac_failure.py with an output path.
