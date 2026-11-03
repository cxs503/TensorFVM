# Second-order verification of the annular laminar FVM backend

**Case**: fully developed laminar annular Poiseuille flow, production backend `tensorfvm.annular` (fitted diffusion kernel, straight-edge polygonal annulus, no analytic initialisation).

**Verdict**: PASS — every reported quantity is below the 3 % acceptance limit at the finest level and the velocity error converges at observed order 2.00.

## 1. Case, geometry and parameters

| quantity | symbol | value |
|---|---|---|
| inner radius | ri | 0.0500 m |
| outer radius | ro | 0.1000 m |
| radius ratio | ri/ro | 0.500 |
| axial pressure gradient (volumetric source) | G | 0.0100 Pa/m |
| dynamic viscosity | mu | 0.0010 Pa s |
| density | rho | 1.0 kg/m3 (irrelevant: Stokes limit) |
| exact flow rate | Q | 4.947382e-05 m3/s |
| exact peak velocity at r = 73.55 mm | u_max | 3.16594 mm/s |
| exact inner-wall shear | tau_i | 2.910106e-04 Pa |
| exact outer-wall shear | tau_o | -2.294947e-04 Pa |

Reynolds number based on the hydraulic diameter is 0.3: the case is a pure diffusion problem without advection, so the analytic solution is the exact reference and no turbulence or literature scatter enters the comparison.

## 2. Method and tool chain

1. **Reference is machine-verified first.** The closed-form profile is checked against the axisymmetric momentum equation with a fourth-order stencil, both walls against no-slip, the closed-form flow rate against independent `scipy.integrate.quad`, and the analytic wall shears against the global equilibrium identity. Worst relative residual: 3.35e-10 (limit 1e-09).
2. **Geometry is audited before the numerics** on the actual production mesh: total polygon area versus pi(ro^2-ri^2) and the chord deficit of both straight-edge walls.
3. **Production refinement study**: `tensorfvm.annular.solve` at nr = 8, 16, 32 with ntheta = 4 nr (h = (ro-ri)/nr), double precision, scipy CG to a true residual gate of 1e-11.
4. **Error budget**: L-infinity and volume-weighted L2 velocity error, flow rate, both wall shears, peak velocity and its radial position, plus the discrete axial equilibrium residual (pressure source versus total wall traction).

Reproduce with `python scripts/verify_annular_order.py` (total wall time 57.1 s on CPU).

## 3. Results

### 3.1 Geometry audit

| nr | ntheta | cells | area error | inner perimeter deficit | outer perimeter deficit | min cell area [m2] |
|---|---|---|---|---|---|---|
| 8 | 32 | 256 | -6.413e-03 | -1.606e-03 | -1.606e-03 | 6.478e-05 |
| 16 | 64 | 1024 | -1.606e-03 | -4.015e-04 | -4.015e-04 | 1.579e-05 |
| 32 | 128 | 4096 | -4.015e-04 | -1.004e-04 | -1.004e-04 | 3.893e-06 |

The area defect decays as h^2.00: the polygonal inner and outer walls are inscribed chords, so the numerical annulus is *smaller* than the true one by a second-order geometric term. This is the floor under any second-order error claim on this mesh family.

### 3.2 Error against the exact solution

| nr | cells | $\|u-u_{ex}\|_\infty/\max|u_{ex}|$ | volume-weighted $\|u-u_{ex}\|_2/\max|u_{ex}|$ | $|Q-Q_{ex}|/Q_{ex}$ | inner-wall shear $|\tau_i-\tau_{ex}|/|\tau_{ex}|$ | outer-wall shear $|\tau_o-\tau_{ex}|/|\tau_{ex}|$ | $|u_{max}-u_{ex,max}|/u_{ex,max}$ | wall equilibrium | solver residual |
|---|---|---|---|---|---|---|---|---|---|
| 8 | 256 | 4.017e-02 | 1.931e-02 | 1.642e-02 | 9.819e-03 | 1.643e-03 | 3.801e-03 | 0.0e+00 | 3.1e-14 |
| 16 | 1024 | 1.084e-02 | 4.877e-03 | 4.200e-03 | 2.464e-03 | 4.062e-04 | 1.925e-03 | 2.3e-16 | 7.3e-13 |
| 32 | 4096 | 2.811e-03 | 1.223e-03 | 1.056e-03 | 6.165e-04 | 1.013e-04 | 2.263e-04 | 4.6e-16 | 3.1e-12 |

### 3.3 Observed order of accuracy

| quantity | orders between successive levels | last |
|---|---|---|
| $\|u-u_{ex}\|_\infty/\max|u_{ex}|$ | 1.89, 1.95 | 1.95 |
| volume-weighted $\|u-u_{ex}\|_2/\max|u_{ex}|$ | 1.98, 2.00 | 2.00 |
| $|Q-Q_{ex}|/Q_{ex}$ | 1.97, 1.99 | 1.99 |
| inner-wall shear $|\tau_i-\tau_{ex}|/|\tau_{ex}|$ | 1.99, 2.00 | 2.00 |
| outer-wall shear $|\tau_o-\tau_{ex}|/|\tau_{ex}|$ | 2.02, 2.00 | 2.00 |
| $|u_{max}-u_{ex,max}|/u_{ex,max}$ | 0.98, 3.09 | 3.09 |

### 3.4 Integrated quantities at the finest level

| quantity | numerical | exact | relative error |
|---|---|---|---|
| flow rate Q [m3/s] | 4.952606e-05 | 4.947382e-05 | +1.056e-03 |
| inner-wall shear tau_i [Pa] | 2.908312e-04 | 2.910106e-04 | -6.165e-04 |
| outer-wall shear magnitude |tau_o| [Pa] | 2.294714e-04 | 2.294947e-04 | -1.013e-04 |
| peak velocity u_max [m/s] | 3.165226e-03 | 3.165942e-03 | -2.263e-04 |
| peak velocity radius [m] | 0.074199 | 0.073553 | +8.779e-03 |

Cross-check of the reference itself: `scipy.integrate.quad` gives 4.947381662033e-05 m3/s versus the closed form 4.947381662033e-05 m3/s, a relative difference of 1.9e-15.

## 4. Figures

![profile](profile.png)

*Figure 1 — radial velocity profiles at three angular positions against the exact solution (left) and the pointwise absolute error (right) at the finest level.*

![convergence](convergence.png)

*Figure 2 — log-log error versus spacing h with the nominal slope-2 guide (left) and the geometric area defect of the polygonal annulus (right).*

![field](field.png)

*Figure 3 — axial-velocity cloud map on the production mesh (left) and the absolute error field (right); the white disc is the no-slip inner cylinder.*

## 5. Notes, limitations and scope

- **Scope**: this verifies the production diffusion operator and no-slip boundary handling on a body-fitted curvilinear mesh, in the Stokes limit. It does not qualify advection, pressure-velocity coupling, turbulence or GPU paths.
- The straight-edge inner/outer walls are chords, not arcs, so the discrete domain differs from the analytic one by an O(h^2) area defect; the measured velocity error contains that geometric term and cannot fall below it on this mesh family. A curved-wall variant would be needed to isolate the operator error further.
- Wall shear is read from the exported boundary-face momentum flux, which is the same quantity the discrete momentum balance uses; it is therefore a property of the solved system, not an independent post-processing gradient.
- The peak velocity is a single-cell quantity: its error is already 2.3e-04 but its observed order is not monotone (the argmax jumps between cells), so no order claim is made for it; the smooth L2 and linfin norms carry the order claim.
- The discrete axial equilibrium residual (pressure source minus total wall traction) is machine-zero by construction; it is reported as an implementation-consistency check, not as an accuracy statement.
- Runtime 57.1 s, linear iterations at the finest level 36, matrix nonzeros 20224.

## 6. Error verdict

| quantity (finest level) | relative error | < 3 % ? |
|---|---|---|
| $\|u-u_{ex}\|_\infty/\max|u_{ex}|$ | 2.811e-03 | yes |
| volume-weighted $\|u-u_{ex}\|_2/\max|u_{ex}|$ | 1.223e-03 | yes |
| $|Q-Q_{ex}|/Q_{ex}$ | 1.056e-03 | yes |
| inner-wall shear $|\tau_i-\tau_{ex}|/|\tau_{ex}|$ | 6.165e-04 | yes |
| outer-wall shear $|\tau_o-\tau_{ex}|/|\tau_{ex}|$ | 1.013e-04 | yes |
| $|u_{max}-u_{ex,max}|/u_{ex,max}$ | 2.263e-04 | yes |
| velocity convergence order | 2.00 (nominal 2) | yes |

Overall: **PASS**.

