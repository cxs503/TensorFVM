# SIMPLEC audit on the collocated developed-channel case

## What was measured

The 2-D collocated production iteration in `tensorfvm.solver` documented SIMPLEC as
unimplemented (`verification/simple_gpu.py`: "SIMPLEC/PISO/PIMPLE have no separate
production implementation").  SIMPLEC is now implemented as a mobility change only:

    d_SIMPLE  = A / (aP/alpha)
    d_SIMPLEC = A / max(aP/alpha - sum(aN), 0.1*aP)

so the pressure-correction operator, the Rhie-Chow surface, the boundary treatment
and the residual definitions are untouched.  Everything below is produced by
`python -m tensorfvm.verification.simplec_channel --output DIR`.

## Results

Resolution ny=12 (nx=3ny, Re_H=1, Ub=5e-4 m/s):

| configuration | alpha_u | alpha_p | iterations | converged | velocity_l2 | mass imbalance | divergence | gates |
|---|---|---|---|---|---|---|---|---|
| simple-default | 0.7 | 0.3 | 41 | yes | 0.00634 | 3.47e-16 | 9.50e-15 | all pass |
| simple-tuned | 0.8 | 0.3 | 43 | yes | 0.00634 | 3.47e-16 | 6.94e-15 | all pass |
| simplec-a07-p10 | 0.7 | 1.0 | 32 | yes | 0.00634 | 3.47e-16 | 5.96e-15 | all pass |
| simplec-a08-p10 | 0.8 | 1.0 | 49 | yes | 0.00634 | 0.00e+00 | 9.29e-15 | all pass |
| simplec-no-floor | 0.8 | 1.0 | 49 | yes | 0.00634 | 0.00e+00 | 9.29e-15 | all pass |

- Fixed-point agreement against `simple-default` (peak component difference / Ub):
  - simplec-a08-p10: u 6.684e-08, v 3.695e-08
  - simple-tuned: u 6.685e-08, v 8.666e-09
  - simple-default: u 0.000e+00, v 0.000e+00
  - simplec-a07-p10: u 2.080e-06, v 3.272e-07
  limit 0.0001, passed=True.
- SIMPLE with alpha_p=1 breaks down:
  `RuntimeError`
  after 12 iterations.
- SIMPLEC denominator-floor activity, recorded honestly: limited degrees of
  freedom = {"12/simple-default": 0, "12/simple-tuned": 0, "12/simplec-a07-p10": 0, "12/simplec-a08-p10": 0, "24/simple-default": 0, "24/simple-tuned": 0, "24/simplec-a07-p10": 0, "24/simplec-a08-p10": 0, "12/simplec-no-floor": 0};
  floor-free run equivalent to the floored run = True.
  On this smooth developed channel the floor never binds, so it is a safeguard
  whose necessity is *not* demonstrated here.
- SIMPLEC keeps every analytic gate: velocity L2 vs
  `6Ub(y/H)(1-y/H)` = 0.00634 (< 0.03),
  mass imbalance 0.00e+00,
  scaled divergence 9.29e-15.

## Iteration cost, honestly

At ny=12 SIMPLEC(alpha_u=0.8, alpha_p=1) needs
49 iterations against 41 for SIMPLE default
(-20% fewer) and 43 for a tuned SIMPLE
(-14% more).  The gain is
therefore *not* a universal speed-up: with a well-tuned alpha_p, SIMPLE is
competitive on this smooth developed-channel case.  What SIMPLEC buys is a
stable alpha_p=1 iteration, which is exactly the regime where SIMPLE loses
positive definiteness.

## Scope and limitations

Converged fields are identical within solver tolerance, so the scheme changes the
iteration path, not the discretization.  The audit covers one laminar,
single-block, wall-bounded case at two/three resolutions; it does not claim
general speed-ups, does not touch the 3-D native SIMPLEC path, and does not
implement PISO/PIMPLE.
