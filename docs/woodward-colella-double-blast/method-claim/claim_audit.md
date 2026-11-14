# 做法登记一致性审计：CASE-WOODWARD-COLELLA / oss_solutions#438

claim text sha256 = `10e7671e3adffc9c…`

| # | 领域 | 登记声明 | 检查结果 | 状态 | 缺陷等级 |
|---|---|---|---|---|---|
| 1 | scheme | MUSCL-Hancock, 2nd order | CONFORMANT | ✅ | none |
| 2 | scheme | MC limiter | CONFORMANT | ✅ | none |
| 3 | scheme | HLLC flux | CONFORMANT | ✅ | none |
| 4 | scheme | SSP-RK time integration | NONCONFORMANT | ❌ | medium |
| 5 | scheme | CFL 0.4 | CONFORMANT | ✅ | none |
| 6 | case | W&C 1984 datum via compressible.cases.double_blast | CONFORMANT | ✅ | none |
| 7 | criteria | tolerance < 10 % | CONFORMANT | ✅ | none |
| 8 | criteria | peak density, peak position, L1, 4 steepest faces, window mass | CONFORMANT | ✅ | none |
| 9 | numbers | nx=800: 2.03 / 2.01 / 0.18 % , faces<=0.0012, PASS | CONFORMANT | ✅ | none |
| 10 | numbers | peak err 20.76/10.58/2.03 %, obs order ~1.8 | NONCONFORMANT | ❌ | low |
| 11 | numbers | independent WENO5/Rusanov/SSP-RK3 arm, zero shared discretisation | CONFORMANT | ✅ | none |

**verdict = NONCONFORMANT** (9/11 条符合)

不一致项：
- `scheme.time_integrator`（medium）: no SSP-RK anywhere in compressible/; EulerSolver accepts order in {1,2} and step() is a single MUSCL-Hancock predictor+corrector, so the registered description names a time integrator this solver does not have. MUSCL-Hancock and SSP-RK are alternative second-order-in-time integrations, not composable stages: the claim is self-contradictory as written. Accuracy content ('2nd order in time') is delivered by the Hancock predictor -- see probes; the defect is provenance only.
- `numbers.convergence`（low）: the three quoted errors reproduce exactly, but 'observed order ~1.8' is not reproduced by any standard estimator on the same numbers: local pair orders 0.97 (200->400) and 2.38 (400->800), 3-point least-squares 1.18, mean of the two local orders 1.68. The claim is directionally true (between first and second order, as shocks demand) but not reproducible without naming the estimator.
