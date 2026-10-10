# TensorFVM analytic benchmarks and TensorLBM comparison

- [Complete Chinese report](report.md)
- [English manuscript PDF](report.pdf)
- [Printable HTML](report.html)
- [Editable LaTeX](report.tex)
- [Matched FVM/LBM comparison](comparison.md)
- [Comparison PDF](comparison.pdf)
- [Common workflow and usage](../benchmark-workflow.md)

Nine regular FVM runs pass every declared physical error below 3% plus numerical gates. Two negative controls are rejected. Three actual external TensorLBM BGK runs pass the velocity/pressure accuracy gate. Independent NumPy arithmetic reconstructs all 195 accepted MAC steps and 443 BGK steps.

Poiseuille's finest velocity L2 / pressure-gradient errors are 0.042848% / 0.078365%. Taylor-Green's finest face velocity / cell pressure L2 errors are 0.008024% / 0.224797%. Spatial and temporal observed orders are approximately two.

At 64x64, the compared TensorLBM BGK errors are 0.152073% / 0.814284%. At 32x32 the LBM pressure is slightly more accurate than FVM; this is retained. FVM face continuity/energy evidence and primary state structure are discussed separately from runtime. No speed or peak-memory superiority is claimed.

A separate 128x128x4, dt=0.0125 Picard failure at the third step remains in the [failure archive](../verification-benchmarks-failed-n128/README.md). Qualification applies only to the declared channel region and periodic analytic problem, not engineering icebreaking.

Raw states, curves, histories, source hashes, independent audit and validation logs accompany the report. Report regeneration changed publication code only; numerical/driver source and raw fields remained unchanged.
