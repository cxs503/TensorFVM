# Gmsh external-flow verification evidence

19 regression tests passed. Independent source/artifact/input/continuation validation covers the saved physical meshes, complete equations, immutable wall postprocessing inputs and the strict <3% coefficient gates. Actual results and all failed coarse configurations are in the linked primary reports. This evidence does not establish an asymptotic refinement regime, GPU speedup or matched TensorLBM ranking.

- [Cylinder report](../verification-gmsh-cylinder-wall/report.pdf)
- [Reference-domain NACA report](../verification-gmsh-naca-matched/report.pdf)
- [Regression output](tests.txt)
- [Structured evidence and manifest hashes](audit.json)
