# fvm_backward_step — backward-facing step at Re_H = 800

**Status: FAIL (not converged within the compute budget of this node).**

| item | value |
|---|---|
| geometry | downstream height H = 1, step H/2 (expansion ratio 2), inlet duct 2H, outlet duct 10H |
| inlet | parabolic `u = 24(y-0.5)(1-y)` over the upper half, peak 1.5, bulk U_b = 0.5 |
| Reynolds | `Re_H = U_b H / nu = 800` (Gartling 1990 convention) |
| criterion | `L_r/H = 6.0 +- 0.2`, error < 10 % (Gartling 6.10, Armaly 1983 5.8–6.4) |
| grids | H/20, H/40, H/80 |
| code | `src/tensorfvm/incompressible_baselines.py::build_backward_facing_step` |

What works
- geometry, inlet mass flux (Q = 0.5), Reynolds definition and the wall-shear
  zero-crossing post-processing are unit-tested;
- the solve is stable and *physical*: a strong recirculation develops
  (`min u ~ -0.27`) and the reattachment point exists and grows monotonically in
  pseudo-time (L_r/H: 0.95 -> 1.83 -> 2.64 -> 3.37 over 600 iterations at H/20).

What does not work (honest)
- the pseudo-transient SIMPLE needs O(10^3–10^4) iterations to advect the
  recirculation to its steady length; the runs launched here did not reach a
  steady L_r within the session, so no verdict number is claimed.  The failure is
  **budget/robustness, not physics**: the same kernel reproduces the analytic
  Poiseuille pressure gradient to 0.15 % and the cavity Nusselt numbers to < 2 %.
- the kernel validation figure (`fig_kernel_channel.png`) belongs to this
  directory only because it was generated first; it documents the analytic
  checks used to sign off the kernel.
