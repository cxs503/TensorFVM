# CASE-FVM-CHANNEL-KMM180 / 3-D LES route: what the implicit-diffusion fix does and does not fix

Anchor: `CASE-FVM-CHANNEL-KMM180` (mean-velocity log law, `Re_tau = 180`). Route: `channel_les3d`.
Status of this note: **BLOCKED** - the anchor's 3-D LES reading is still `not_run`.

## 1. What changed in the solver

1. `implicit_wall_normal` (default on): the wall-normal viscous/SGS flux is solved as a
   tridiagonal system `(I - dt M_y) u^{n+1} = u^*` (vectorised Thomas, M-matrix, no pivoting).
   The matrix is *derived from the same expressions* the explicit stencil uses; the identity is
   asserted to round-off: `max|M_y^implicit f - M_y^explicit f| = 2.18e-11` (u),
   `5.46e-12` (v), `2.18e-11` (w) at `64x64x32`.
   The wall-row diffusion number of the production grid is `nu dt/dy^2 = 0.35` at `dt = 2e-3`
   and grows linearly with `dt`, so any usable time step leaves the explicit stability band
   (`< 0.5`) and the stiff part has to be solved, not integrated.
2. `time_scheme='rk3'` (default): SSP-RK3 for convection + transverse diffusion. Forward Euler with
   central convection has `|G|^2 = 1 + (lambda sin theta)^2 > 1` at every wavenumber.
3. Wall distance: `y+ = min(y, 2-y) Re_tau`. The grid spans the full channel `y in [0, 2]`, so the
   historical `y+ = y Re_tau` read the **top wall as `y+ = 359`** instead of
   `0.51`: van Driest damping was off in the upper half and the Reichardt initial profile
   was evaluated at twice its argument there.
4. `wall_shear('second_order')`: quadratic one-sided wall gradient next to the first-order
   half-cell conductance (`1.000000` vs analytic `1.0` on the
   laminar parabola, half-cell `0.995865`).

## 2. What is still broken (honest)

At `16x16x8` the solution still becomes non-finite, and the time-to-divergence depends on `dt` and
**not** on the closure or the integrator:

| configuration | steps to divergence |
|---|---|
| Cs=0.1 dt=2e-3 rk3 | 142 |
| Cs=0.1 dt=4e-3 rk3 | 74 |
| Cs=0.1 dt=2e-3 euler | 187 |
| Cs=0.0 dt=2e-3 rk3 | 131 |
| Cs=0.0 dt=4e-3 rk3 | 70 |

Two recorded diagnoses are therefore falsified by this table:

* "the wall-normal diffusion stiffness is the cause" - fixed above, and the run still diverges;
* "the divergence comes from the `nu_t` feedback" - `Cs = 0` diverges as well (`131` vs `142` steps).

The trend "halving `dt` roughly doubles the steps" is a per-step amplification `|G| = 1 + O(dt)`,
not a per-unit-time growth: lowering `dt` delays but never removes it. Locating the amplified mode
(per-row / per-mode growth, wall-row vs interior) is the next concrete step and is **not done here**.

## 3. Consequence for the anchor

The ledger criterion `u+(y+)` in `y+ > 30` vs the KMM1987-lineage table is **not delivered by this
route**: no production (`64x64x32`) statistics exist, so no `<3 %` or `<5 %` claim is made. The
two-dimensional/1-D route (SST equilibrium channel, `docs/case-fvm-channel-kmm180-pool`) is
unaffected by this note.

## 4. Reproduction

```bash
PYTHONPATH=src python3 docs/case-fvm-channel-kmm180-les3d-stability/make.py
PYTHONPATH=src python3 -m pytest tests/test_channel_les3d_implicit.py tests/test_channel_les3d.py -q
```

Figure: `fig_divergence_evidence.png` (left: steps-to-divergence per configuration; right: amplitude
trace before divergence).

## 5. Update: the "next concrete step" below has been done

Section 2 lists "locating the amplified mode (per-row / per-mode growth, wall-row vs
interior)" as the next step. That measurement now exists in
`docs/case-fvm-channel-kmm180-les-gridscale/REPORT.md`
(`src/tensorfvm/verification/kmm180_les_gridscale_stability.py`), and it changes the
attribution above:

* the amplification is **broadband** (largest single-mode share 16.6 %), so odd-even /
  Nyquist decoupling of the central convection is not the cause;
* the wall-normal operator and the channel mirror symmetry are exact, so the divergence
  is not a boundary/operator defect;
* the lifetime is set by the **SGS constant**: `Cs = 0.1` (the shipping default) diverges,
  `Cs >= 0.2` survives the 500-step cap, and refining at fixed `Cs` makes it *worse*
  (63 steps at `32x32x16` vs 121 at `16x16x8`);
* the step counts in the table above were taken with the default **laminar**
  `profile='parabolic'` initial field (peak `Re_tau/2 = 90 u_tau` at `Re_tau = 180`);
  under identical numerics the physical Reichardt state survives 3.3x longer.

Read "what is still broken" above as "the default configuration is under-dissipative on
this grid", not as "the scheme is unstable".
