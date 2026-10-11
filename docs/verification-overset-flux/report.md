# Nonmatching Cartesian interface flux verification

A common planar interface is partitioned independently on two sides. The normal orientations oppose each other. The prescribed flux density is 1 + 0.2 sin(pi y) cos(pi z); donor integrals are analytic. Transfer uses actual rectangle intersection areas and uniform donor-face density. Receiver reference integrals are evaluated independently. This is interface verification, not a solved overset flow.

|Donor resolution|Donor faces|Receiver faces|Density relative L2|Oriented balance defect|Passed|
|---|---:|---:|---:|---:|---|
|4|16|25|1.189088%|1.11e-16|True|
|8|64|81|0.318913%|1.11e-16|True|
|16|256|289|0.080609%|0|True|

The balance test alone does not establish local flow accuracy. A shared surface must be explicitly defined in the same physical coordinate frame. Flux from moving surfaces additionally needs grid velocity and geometric conservation. No pressure correction, time advancement, populations, or fluid solver is executed here.

Reproduce: `PYTHONPATH=src python scripts/verify_overset_flux.py --output EMPTY_DIRECTORY`. Raw patches, fluxes, sparse intersections and producer sources are included.
