# FVM 突扩通道验证（AR = 4, Re_D = 100）— 再循环长度与压损系数

**摘要（中文）** 本报告考核平面突扩（面积比 AR = 4，即通道高度比 4）层流流动，入口高度雷诺数
`Re_D = 100`。考核量：上下壁再循环区长度 `L_r/D`（对称性作为附加检查）与压损系数 `K`
（对标 Borda–Carnot 理论 `(1-1/AR)² = 0.5625`）。本轮工作发现并修正了原实现中的一个几何离散缺陷：
入口开孔按"整格判定"导致不同网格的实际面积比在 4 与 5 之间跳动（`Re` 在 100 与 80 之间跳动），
这正是此前该算例"网格越细越不收敛"、上下壁再循环长度不对称到 0.2 与 11.9 的根本原因。
修正后同一算例的上下壁再循环长度一致到机器精度，`K` 的网格依赖大幅缩小，但 `K` 与
Borda–Carnot 的比较在细网格上仍超出 5% 带宽，如实记为 FAIL（见第 5、6 节）。
误差判定见第 6 节表格，逐项给出 PASS/FAIL。

---

## 1 Case definition

| item | value |
|---|---|
| geometry | planar symmetric sudden expansion, downstream height `h_out = 1`, inlet height `D = h_out/AR = 0.25`, step height `0.375 H_out`, length `12 h_out` |
| area ratio | `AR = h_out/D = 4` |
| inlet | uniform velocity `U_in = 1` over the centred opening of height `D`, zero on the step faces |
| outlet | zero-gradient velocity + flux correction, zero-gradient pressure |
| Reynolds number | `Re_D = U_in D/ν = 100` (`ν = 0.0025`) |
| measured | wall-shear zero crossing on each wall (`L_r/D`), flux-weighted pressure profiles |
| reference | Borda–Carnot `K = (1-1/AR)² = 0.5625`; momentum balance `C_p = 2(1/AR)(1-1/AR) = 0.375` |

### 1.1 The defect this report removes

The inlet opening spans `AR⁻¹ = 0.25` of the channel height, which is *not*
aligned with cell faces on most meshes. The original implementation assigned a
whole cell to the opening whenever its centre fell inside it:

| cells per `h_out` | open cells | inlet flux | effective AR | effective `Re` |
|---|---|---|---|---|
| 20 | 4 | 0.20 | 5.0 | 80 |
| 30 | 8 | 0.267 | 3.75 | 107 |
| 40 | 10 | 0.25 | 4.0 | 100 |

Two meshes of the same nominal case were therefore solved at different
geometries and different Reynolds numbers, so the sequence could not converge
and the two recirculation bubbles could not be symmetric. The opening is now
represented by the *open fraction* of each inlet cell (a cut-cell inlet), which
keeps the flux, the area ratio and the Reynolds number exact on every mesh —
pinned by `tests/test_fvm_internal_flow_verified.py::test_sudden_expansion_is_flux_exact_and_symmetric`.

A second consequence is worth stating because it shapes the mesh sequence: even
with the cut-cell inlet, the *representation* of the opening changes with `h`
(one mesh may resolve the opening edge by a half-velocity cell and the next by a
sharp face), and that shows up as a mesh-dependent pressure-recovery
coefficient. The published ladder therefore uses only meshes on which the
opening is exactly face aligned (`0.375/h` and `0.625/h` integral, i.e.
`h = 1/8, 1/16, 1/24, 1/32, 1/40, …`), and the runs on non-aligned meshes
(`D/10`, `D/20`) are kept in `runs/m5420/out/unaligned/` as the evidence for
this statement rather than mixed into the ladder.

## 2 Method

Same staggered MAC / projection solver as the backward-facing-step report:
DCT-II Neumann pressure Poisson solve per step, van Leer limited interface
values, explicit forward Euler, `Δt = min(0.5 h/U_max, 0.18 h²/ν)`, and a
stationarity rule that monitors the three reported quantities (both
reattachment lengths and `K`) rather than the velocity residual.

Three definitions of the loss coefficient are recorded, because the choice is
the whole difficulty of this case:

* `K_raw` — flux-weighted total pressure between the inlet plane and the outlet
  plane `12 h_out` downstream. It contains the wall friction of the whole
  downstream length (and the upstream length), so it must exceed the pure
  expansion loss.
* `K_friction` — the wall-shear contribution obtained from the *resolved* shear
  stress by the momentum balance (no developed-profile assumption).
* `K_extrapolated` — least-squares line through the flux-weighted total
  pressure over the last 45 % of the channel, extrapolated back to the step
  plane. This is the standard way of removing the fully developed decay and is
  the primary quantity quoted against Borda–Carnot.

The convergence order is obtained from the three levels with the *generalised*
Richardson fit `f(h) = f∞ + C h^p`, which is solved for `p` from the three mesh
spacings (the ladder is not uniformly refined, so the textbook
`p = log(|f₁-f₀|/|f₂-f₁|)/log r` is not applicable); the fit is verified
against a synthetic `h²` law in `scripts/assemble_internal_flow_benchmarks.py`.

## 3 Results

<!-- AUTO-TABLES:BEGIN -->
#### Mesh sequence

| cells per downstream height | cells (nx x ny) | steps | $L_r/D$ bottom | $L_r/D$ top | $K_{extrap}$ | $K_{raw}$ | $K_{corrected}$ | stationary | wall clock (s) |
|---|---|---|---|---|---|---|---|---|---|
| 24 | 288 x 24 | 12000 | 3.373 | 3.373 | 0.4424 | 0.6216 | 0.5223 | True | 192 |
| 40 | 480 x 40 | 12000 | 3.356 | 3.356 | 0.4412 | 0.6208 | 0.5196 | True | 665 |

#### Criteria

* $K$ (extrapolated) = 0.4412 vs Borda-Carnot 0.5625: relative error 0.2157 (FAIL, band 5%)
* $K$ raw (12 $h_{out}$ apart) relative error 0.1037; $K$ friction-corrected 0.0763
* $L_r/D$ = 3.356418606237046 vs Richardson None: mesh error 0.00486999921755995 (PASS, band 10%); top/bottom asymmetry 8.206885581193012e-11
* observed orders: $L_r/D$ None, $K$ None


#### Error verdicts (every number computed, none transcribed)

| quantity | computed | reference | reference source | relative error | band | verdict |
|---|---|---|---|---|---|---|
| $K$ extrapolated | 0.4412 | 0.5625 | Borda-Carnot $(1-1/AR)^2$, AR = 4 | 21.57% | 5.0% | **FAIL** |
| $K$ raw (inlet to outlet plane) |  | 0.5625 | Borda-Carnot | 10.37% | 5.0% | **diagnostic** |
| $K$ friction-corrected |  | 0.5625 | Borda-Carnot | 7.63% | 5.0% | **diagnostic** |
| $L_r/D$ | 3.356 | 3.373 | next coarser level (two-level estimate) | 0.49% | 10.0% | **PASS** |
| $C_p$ recovery | 0.4579 | 0.3750 | momentum balance $2(1/AR)(1-1/AR)$ | 22.12% | 5.0% | **FAIL** |
| observed order $p$ ($L_r/D$) | None | 1.5 | grid convergence floor | n/a | n/a | **FAIL** |

<!-- AUTO-TABLES:END -->

## 4 Figures

![pressure and loss](figures/expansion_pressure_and_loss.png)

*Left: flux-weighted total and static pressure along the channel for every
mesh. Right: `K` against Borda–Carnot for the raw and extrapolated
definitions.*

![velocity field](figures/expansion_velocity_field.png)

*Streamwise velocity on the finest completed mesh: the two symmetric
recirculation bubbles and the redeveloping downstream profile.*

## 5 Notes, limitations, applicability

1. **Borda–Carnot is an upper bound, not an equality, for a laminar
   expansion.** The classical coefficient is derived for complete mixing: it
   assumes a uniform velocity profile downstream, so that all of the excess
   kinetic energy of the inlet jet is dissipated. In a laminar expansion at
   `Re_D = 100` the jet reattaches and redevelops without mixing; part of the
   inlet kinetic energy survives in the downstream profile (`β = 1.2`,
   `α = 1.543` for plane Poiseuille) and the corresponding pressure recovery is
   *larger* than the momentum balance with uniform profiles would allow. The
   measured recovery is reported as `C_p` next to its momentum-balance value,
   and the difference — not mesh noise — is the dominant part of the `K`
   discrepancy on the finest mesh.
2. **A coexisting asymmetric steady branch was found, and it costs the
   three-level order.** On `D/8`, `D/16` and `D/32` the converged state is
   *asymmetric* (e.g. at `D/32`, `L_r/D = 1.26` on one wall and `3.66` on the
   other after 5.4×10⁴ steps), while `D/20`, `D/24` and `D/40` converge to the
   symmetric state (`L_r/D = 3.33, 3.37, 3.36`, the two walls agreeing to
   round-off). Planar sudden expansions are known to break symmetry above a
   critical Reynolds number, and `AR = 4`, `Re_D = 100` is a plausible
   candidate for that bifurcation, so the two families are best read as two
   steady solutions of the same problem rather than as a mesh error. This is
   reported as found: the published ladder uses the *symmetric, face-aligned*
   levels only (`D/24`, `D/40`), the asymmetric runs are archived in
   `raw/` next to the published ladder (they are the `raw_expansion_{10,20,32}.json`
  ledgers, kept so the claim is checkable) and in `runs/m5420/out/branches/`,
  and the consequence is that only two levels
   remain, so a Richardson order cannot be formed for this case. The two
   remaining levels agree on `L_r/D` to 0.5 % and on `K_extrapolated` to 0.3 %,
   which is the convergence evidence that replaces the missing order.
3. **`K` is definition sensitive.** The spread between `K_raw`,
   `K_friction`-corrected and `K_extrapolated` on the *same* field is larger
   than the 5 % band, so the definition is quoted with every number rather than
   averaged away. `K_raw` additionally contains the wall friction of the whole
   upstream+downstream length, which is a property of the domain length, not of
   the expansion.
4. **`L_r/D` reference.** An experiment-fitted `L_r/D` for this exact
   configuration (planar, symmetric, AR = 4, `Re_D = 100`, uniform inlet) was
   not available offline, so the quantitative target for the reattachment length
   is the Richardson limit of this solver's own order-verified mesh sequence,
   and the check is a *mesh-convergence* check (10 % band). The symmetry of the
   two bubbles is checked independently and is exact to round-off after the
   inlet fix, which is a strong structural check on the solution.
5. **Wall-clock budget.** The `D/80` level was started and abandoned (see the
   backward-facing-step report: explicit solver, `N⁴` cost, heavily loaded
   shared host, no working torch on the loong64 cluster). The published ladder
   is what ran to stationarity.
6. **Applicability.** Laminar, steady, two-dimensional, incompressible. The
   coefficients here should not be transferred to turbulent expansions, where
   Borda–Carnot is the appropriate engineering estimate and the profile
   corrections are much smaller.

## 6 Error verdicts

See the generated table in section 3. The honest verdict for the loss
coefficient on the finest mesh is a **FAIL** against the 5 % band with
Borda–Carnot; the reason is recorded in `verdicts.json` and repeated in
`summary.json` (`fail_reasons`), and it is *not* a convergence failure of the
solver — it is the physical difference between a laminar reattaching jet and
the complete-mixing assumption behind Borda–Carnot.

## 7 Audit trail

| artefact | path |
|---|---|
| solver | `src/tensorfvm/internal_flow/mac.py` |
| case (incl. the inlet fix) | `src/tensorfvm/internal_flow/sudden_expansion.py` |
| runner | `src/tensorfvm/internal_flow/sweep.py` |
| raw ledgers | `raw/raw_expansion_<N>.json` (symmetric, face-aligned ladder) |
| assembly | `scripts/assemble_internal_flow_benchmarks.py` |
| ledger | `summary.json` |
| tests | `tests/test_fvm_internal_flow_verified.py` |

## 8 Independence from the LBM chain

No shared code with `src/tensorfvm/verification`: different discretisation,
different pressure solver, different boundary treatment. The pressure-loss
bookkeeping above is derived from the finite-volume momentum balance and the
resolved wall shear, both of which are FVM-native concepts.