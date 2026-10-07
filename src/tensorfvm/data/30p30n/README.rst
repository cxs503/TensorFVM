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

The diagnostic case's Re=5,000,000 and 0-degree conditions follow the original
case description in https://github.com/coutinhola/30P30N.  Reference lift and
drag values are transcribed from the validation repository README; the drag
normalization is ambiguous there, so the code reports both Cd and 100*Cd and
does not use the comparison as an acceptance criterion.
