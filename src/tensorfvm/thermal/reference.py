"""Published anchors for the thermal-convection verification face.

All numbers are quoted in the non-dimensionalisation used throughout
:mod:`tensorfvm.thermal`: length ``H`` (the layer height), temperature ``dT``
(the plate-to-plate drop) and velocity ``U = sqrt(g beta dT H)``.  With that
choice the Rayleigh and Prandtl numbers are recovered exactly as

    Ra = g beta dT H^3 / (nu alpha),   Pr = nu / alpha,

and the momentum/energy equations become

    du/dt + div(u u) = -grad p + sqrt(Pr/Ra) lap(u) + theta e_y,
    dtheta/dt + div(u theta) = lap(theta) / sqrt(Pr Ra),

see :class:`tensorfvm.thermal.rayleigh_benard.RayleighBenardCase`.

Linear stability of the conduction state
----------------------------------------
The classical Rayleigh--Benard marginal curve follows from the linearised
Boussinesq operator (Chandrasekhar 1961, *Hydrodynamic and Hydromagnetic
Stability*, ch. II).  For a layer between **stress-free** plates the
eigenvalue problem is solvable in closed form,

    Ra_c = 27 pi^4 / 4 = 657.511,   k_c = pi / sqrt(2) = 2.2214,

and for **rigid** (no-slip) plates the numerically established values are

    Ra_c = 1707.762,                k_c = 3.11632,

the latter to the accuracy quoted by Drazin & Reid (1981, table 2 of ch. 4)
and by Chandrasekhar.  These two anchors bracket the whole verification:
the closed-form pair tests the *discretisation*, the rigid pair tests the
*no-slip wall treatment*, and the gap between them (a factor 2.6) proves the
wall condition is actually being felt by the eigen-solver.

Nusselt-number scaling
----------------------
Above onset the heat transport obeys, in the classical (Malkus/Howard)
regime, ``Nu ~ Ra^(1/3)``; the exponent is the acceptance anchor used by
:mod:`tensorfvm.thermal.scaling`.  The prefactor is not universal (it depends
on Pr and on the boundary conditions), so only the exponent is compared.
"""

from __future__ import annotations

#: Critical Rayleigh number of the conduction state, rigid (no-slip) plates.
RA_CRITICAL_RIGID = 1707.762
#: Critical horizontal wavenumber for rigid plates.
WAVENUMBER_CRITICAL_RIGID = 3.11632
#: Critical Rayleigh number, stress-free plates (exact: 27 pi^4 / 4).
RA_CRITICAL_FREE = 657.511
#: Critical horizontal wavenumber, stress-free plates (exact: pi / sqrt(2)).
WAVENUMBER_CRITICAL_FREE = 2.22144
#: Classical Nusselt-number exponent above onset.
NU_EXPONENT_CLASSICAL = 1.0 / 3.0
#: Prandtl number of air at ambient conditions, used for all runs here.
PRANDTL_AIR = 0.71