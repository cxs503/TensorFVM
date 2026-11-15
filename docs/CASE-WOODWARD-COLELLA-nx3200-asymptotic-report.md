# TensorSPH / SPHinXsys CSS port: the hourglass pair order was the momentum leak

**Mission** m22163 (reopen of #856, single-phase completion) · **Repo** TensorSPH ·
**Anchor** XVAL-SPH-MIXTURE-MATERIAL-AND-SURFACE-TENSION · **Verdict** defect fixed, gated
numbers unchanged.

## 1. Case

The ported SPHinXsys continuous-surface-stress (CSS) surface-tension model
(`src/tensorsph/sphinxys_css.py`) carries an upstream "zero-surface-energy" hourglass
penalty `H_ij = 4.5 sigma/2 (T_i + T_j) M_ij /(r_ij + Eps)` with
`T = I - n n^T`, `M = -(g.e) g (x) e/(|g.e||g|)`, `g = (grad c_i + grad c_j)/2`.
Two scenes are used as exact anchors: a 2-D water disc (R = 0.1 m, sigma = 0.072 N/m,
rho = 1000 kg/m3, Wendland C2, h/dx = 1.3, reference `Delta p = sigma/R` = 0.7200 Pa) and a
mode-2 Lamb-perturbed interface (`r = R(1 + eps cos 2 theta)`,
`omega^2 = n(n^2-1) sigma/(rho R^3)`, `omega_exact = 0.6573 rad/s`).

## 2. How

Round 2's record stated that the upstream penalty "is non-conservative by construction"
(0.73 relative momentum residual) and therefore had to be switched off.  That is wrong.
`H_ij` is **even** under `i <-> j` (`T_i + T_j` and `M ∝ (g.e) g (x) e` both are), so the
scalar `H . grad W` is odd — exactly the parity of the symmetric stress flux
`(S_i + S_j) . grad W` — hence both terms must enter the antisymmetric pair bracket with the
same sign.  Upstream runs the symmetric loop (its `j` end *is* the other particle's loop) and
is exactly conservative.  The port assembled the `j` end as `-w (flux - hg)` (inner) and kept
an `e`-odd term un-flipped (contact), i.e. each pair added its penalty twice.

Fix: inner `-w (flux + hg)`, contact `hg_ji = -hg_ij`.  The removed assembly is
*reconstructed* from the current API (two runs with `hourglass_coeff = ±c`, the term being
linear in the coefficient) so the audit does not cite history: the reconstruction reproduces
the pre-fix residual 0.7534 exactly, and is itself verified against the model on a star of
pairs (agreement 4.6e-17).

## 3. Results

| configuration | before | after |
|---|---|---|
| inner penalty, momentum residual (relative) | 0.7534 | 9.2e-18 |
| contact penalty, momentum residual (two-phase) | 0.7260 | 8.0e-18 |
| penalty-off residual (gated config) | 5.1e-17 | 1.4e-18 |
| single-phase Laplace, penalty ON, R/dx = 32 | -375.10 % | -119.07 % |
| two-phase Laplace (both-phase reading), penalty ON | -5.82 % | -12.75 % |
| two-phase Laplace, penalty OFF | -5.80 % | -5.80 % |
| spurious mode-2 pressure on a circle, penalty ON (sigma/R) | 2.097 | 0.395 |
| two-phase Lamb omega, penalty ON, scope=all | -4.16 % | +49.69 % |
| two-phase Lamb omega, penalty ON, scope=contact | -4.15 % | -2.17 % |
| single-phase Laplace with closure, alpha* (gated) | +0.73 / 0.00 / -0.18 % | unchanged |

## 4. Figures

Panel (a): momentum residual of the removed vs corrected pair assembly; (b) Laplace ladder
with the penalty on/off and with the interface-completion closure; (c) two-phase Lamb
frequency: the inner penalty now visibly biases the mode, `scope="contact"` is the gated
setting.  All numbers are read from `out_single_phase_completion/gate.json` at plot time.

## 5. Pitfalls

* The correction *reveals* a bias that the leak had been partly cancelling: the two-phase
  Lamb frequency with the penalty on is now +49.7 % (a faithful-port statement), which
  downgrades a previously recorded -4.2 %.  That record was measuring the port's leak as much
  as the model; the gated configuration (`hourglass_scope="contact"`, -2.2 %) stands.
* Only penalty-ON numbers move.  Every gated result (alpha ladder, smoothing-ratio transfer,
  two-phase regression, penalty-off BEM/CSF channels) is bit-identical because those run with
  the penalty off.
* `out_mixture_laplace/gate.json` still holds penalty-ON rows measured before the fix; the
  README 2.5 correction flags this and the regeneration is the follow-up.
* Scope: 2-D, static Laplace + linear Lamb; no dynamic multiphase coupling.

## 6. Error judgement

Key quantities vs. their exact references: closure-pass Laplace error on the ladder
+0.73 / 0.00 / -0.18 % (pre-existing, unchanged) and smoothing sweep -1.01 ... +5.84 %;
two-phase Laplace with the penalty off -5.80 %; two-phase Lamb with `scope="contact"`
-2.17 %.  **Is the key quantity < 3 %?** Yes for the closure-pass Laplace ladder (worst
0.73 %) and for the two-phase Lamb (-2.17 %); No for the two-phase *static* reading at the
fit rung (-5.80 %, ~6 %, a documented model/estimator bias, unchanged by this work).  The
momentum residual is not an accuracy quantity: it is now machine zero (1e-17) in every
configuration, which is the property being repaired.

