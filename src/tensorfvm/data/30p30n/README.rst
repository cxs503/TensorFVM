30P30N element coordinates
==========================

``slat.dat``, ``main.dat`` and ``flap.dat`` are the normalized two-dimensional
profile coordinates from the public `30P-30N Validation Case` repository:

https://github.com/linuxguy123/30P-30N-Validation-Case

The source repository is distributed under GNU GPL version 3.0 and identifies
the geometry as the MDA 30P-30N validation configuration, based on the
WolfDynamics validation case.  The coordinates are retained here so the
benchmark can run without downloading geometry at runtime.  Their order and
normalization are preserved; the solver treats the profiles as separate solid
boundaries.

The deployed x extent was normalized to one. The nominal stowed reference
chord in these coordinates is 1/1.216241; Reynolds numbers and force coefficients
must use that nominal chord. The legacy entry point remains a diagnostic case.

The original Wolf Dynamics table identifies Cl=2.167089 and Cd=0.033243 at
Re=5,000,000 and alpha=0. The derivative repository README swapped their labels.
The values were read from plots with unquantified uncertainty; neither residual
convergence nor matching a force coefficient grants pressure accuracy acceptance.

The separate 30p30n-hlpw4 directory contains the official HLPW4 geometry and
NASA LTPT Cp data at Re=9,000,000, M=0.2 and alpha=8.10/16.21/21.34/23.28 degrees.
Do not mix its conditions with this legacy Wolf Dynamics case.
