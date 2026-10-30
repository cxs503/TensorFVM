"""TMR validation cases (NASA Turbulence Modeling Resource) for TensorFVM.

Modules
-------
``linear``    LU-preconditioned linear solves for the collocated solver; the
              discretisation is unchanged, only the linear algebra accelerates.
``plot3d``    (see :mod:`tensorfvm.tmr_grid`) official TMR grid loading.
``run_flatplate`` / ``run_backstep`` case drivers.
``gate``      recomputation and acceptance gates (exit non-zero on failure).
"""