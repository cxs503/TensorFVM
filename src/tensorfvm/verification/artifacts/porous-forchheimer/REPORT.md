# Forchheimer porous regime: independently referenced verification

## 1. Case

Volume-averaged (Brinkman-extended Darcy--Forchheimer) channel flow, same geometry
and fluid as the shipped porous case: L = 0.12 m, H = 0.02 m,
U_b = 0.0005 m/s, rho = 1000.0 kg/m3, mu = 0.01 Pa s,
K = 1e-06 m2 (sqrt(K) = 1 mm), beta in {0, 2e3, 2e4} 1/m.
Governing equation mu u'' - (mu/K)u - rho beta |u|u = dp/dx, u(0)=u(H)=0, dp/dx fixed by the
prescribed bulk velocity.

## 2. Method

* Continuum reference: damped Newton on a 2001-point FD grid, dp/dx by Brent root finding on
  the bulk velocity; validated against the shipped closed forms (Darcy--Brinkman 1.4e-6,
  plane Poiseuille 1.4e-6, see evidence.json/self_check).
* Discrete reference: independent NumPy replica of the staggered u-face balance
  (diag = 2 mu dx/dy + dx dy (mu/K + rho beta |u|), wall rows 3 mu dx/dy, rhs = -dx dy dp/dx).
* Production: PorousChannelSolver (SIMPLE) at ny = 16, 32, nx = 3 ny, tolerance 1e-8.

## 3. Results

| ny | beta [1/m] | dy/sqrt(K) | L_inf profile err | L2 profile err | dp/dx SIMPLE | dp/dx exact | dp/dx err | iters |
|---|---|---|---|---|---|---|---|---|
| 16 | 0 | 1.25 | 5.65% | 2.48% | -5.4633 | -5.5556 | 1.66% | 53 |
| 32 | 0 | 0.62 | 2.84% | 0.87% | -5.5276 | -5.5556 | 0.50% | 58 |
| 16 | 2000 | 1.25 | 5.98% | 2.62% | -5.9971 | -6.1126 | 1.89% | 53 |
| 32 | 2000 | 0.62 | 3.12% | 0.94% | -6.0772 | -6.1126 | 0.58% | 58 |
| 16 | 20000 | 1.25 | 6.78% | 3.07% | -10.6473 | -10.9809 | 3.04% | 52 |
| 32 | 20000 | 0.62 | 5.18% | 1.46% | -10.8645 | -10.9809 | 1.06% | 57 |

Independent FV-replica audit at ny = 32: profile L_inf = 1.6e-14,
L2 = 9.1e-15, dp/dx = 1.9e-14 (machine precision).
Observed convergence orders (16->32): {"0": [0.9916438614178542], "2000": [0.9381480800133131], "20000": [0.3884542288163522]}.

## 4. Figures

`porous-forchheimer.png`: (left) velocity profile SIMPLE vs nonlinear BVP vs replica;
(middle) L_inf error vs ny with the 3% gate; (right) replica audit residuals.

## 5. Notes

* The lagged Picard sink is *consistent*: at convergence the SIMPLE field satisfies the
  nonlinear discrete equation to 2e-14. The residual-history checks cannot detect a lagged
  inconsistency; this replica can.
* The serious limitation is resolution, not the sink: with dy/sqrt(K) = 0.6--1.25 the
  Brinkman layer is only 1-2 cells thick, so the profile is 3--7% off the continuum solution and
  the observed order is 1.0/0.9/0.4 (not 2). The previous 5% *bulk-law* gate hid this because
  a plug-like core keeps the mean velocity right.
* Requirement: dy <= sqrt(K)/3 (ny >= 67 here) to get within ~1%.

## 6. Error verdict

| quantity | ny=16 | ny=32 | <3%? |
|---|---|---|---|
| profile L_inf (beta=0) | 5.65% | 2.84% | no at 16, yes at 32 |
| profile L_inf (beta=2e3) | 5.98% | 3.12% | no |
| profile L_inf (beta=2e4) | 6.78% | 5.18% | no |
| dp/dx (beta=2e4) | 3.04% | 1.06% | no at 16, yes at 32 |
| SIMPLE vs independent FV replica | -- | 2e-14 | yes (machine precision) |

Honest FAIL on the physical profile gate at production grids; the implementation itself is
verified. No result is reported as passing that does not.
