# CASE-FVM-DAMBREAK-STOKER — 验证总览（Verification Summary）

**锚点**: `CASE-FVM-DAMBREAK-STOKER`（算例池·浅水溃坝 Ritter/Stoker 精确解）
**参考**: Ritter (1892) Z. Ver. Deutsch. Ing. 36:947; Stoker (1957) Water Waves ch.10
**容差**: <5%（波前位置、水位阶跃、L1 深度误差、x_f(t) 序列）
**状态**: ✅ PASS（n ≥ 800）
**生成时间**: 2026-11-15T07:00:00Z
**Git commit**: (由平台自动记录)

---

## 1. 验收项与实测结果

| 验收项 | 判据 | Ritter 干床 (n=1600) | Stoker 湿床 (n=1600) | 判定 |
|---|---|---:|---:|---|
| 波前/水跃位置 | <5% | 0.0173% | 0.0210% | ✅ |
| 深度剖面 L1 | <5% | 0.0772% | 0.0427% | ✅ |
| x_f(t) 序列最大误差 | <5% | 0.4247% | 0.8443% | ✅ |
| 水位阶跃 h2/h1 | <5% | — | 0.000058% | ✅ |
| 波形单调（内极值） | =0 | 0 | 0 | ✅ |
| 质量守恒 | 机器精度 | 3.2e-14 | 1.3e-14 | ✅ |
| 深度正性 | h_min ≥ 0 | ✅ | ✅ | ✅ |

**所有关键量误差均 <3%**，远低于 5% 容差。

## 2. 解析解自验证

| 检验项 | 残差 | 判定 |
|---|---:|---|
| Ritter h(dam)/h₀ = 4/9 | <1e-14 | ✅ |
| Ritter u(dam)/c₀ = 2/3 | <1e-14 | ✅ |
| Ritter 扇区质量守恒 | <1e-13 | ✅ |
| Stoker Riemann 不变量 | 0.0 | ✅ |
| Stoker R-H 质量跳跃 | 0.0 | ✅ |
| Stoker R-H 动量跳跃 | 1.1e-16 | ✅ |

解析解通过自洽检验，不依赖任何外部表格。

## 3. 网格收敛

| 情形 | n | depth_l1 | 波前/激波误差 | 观测阶(L1) |
|---|---:|---:|---:|---:|
| Ritter 干床 | 400 | 0.3083% | 0.1059% | 1.01 |
| Ritter 干床 | 800 | 0.1542% | 0.0391% | — |
| Ritter 干床 | 1600 | 0.0772% | 0.0173% | — |
| Stoker 湿床 | 400 | 0.1730% | 0.1597% | 0.99 |
| Stoker 湿床 | 800 | 0.0851% | 0.0433% | — |
| Stoker 湿床 | 1600 | 0.0427% | 0.0210% | — |

L1 收敛阶 ≈1.0 是 TVD 格式含间断解的理论预期。

## 4. 扩展验证（已完成）

| 扩展项 | 报告 | 状态 |
|---|---|---|
| 伽利略提升（u₀ ≠ 0） | `docs/CASE-FVM-DAMBREAK-STOKER-GALILEAN/REPORT.md` | ✅ PASS |
| 2D 通道确认 | `docs/case-fvm-dambreak-stoker-2d/REPORT.md` | ✅ PASS |
| 鲁棒性扫描 | `docs/case-fvm-dambreak-stoker-robustness/REPORT.md` | ✅ PASS |
| 亚格水跃位置 | `docs/case-fvm-dambreak-stoker/REPORT.md §7` | ✅ PASS |
| 无先验位置度量 | `tests/test_dry_bed_front_position_oracle_free.py` | ✅ PASS |
| 可复现性绑定 | `tests/test_dambreak_stoker_repro_verification.py` | ✅ PASS |
| 算例池登记 | `docs/oss_solutions/CASE-FVM-DAMBREAK-STOKER.json` | ✅ PASS |

## 5. 独立审计记录

| 审计轮次 | 发现 | 修复 |
|---|---|---|
| m17055 (F1-F6) | 波前度量定义不一致、序列未分级 | 已修复 |
| m18263 (F7) | 水跃位置为格心读数 | 已修复（亚格插值） |
| m20028 (F8) | 序列判据未进 verdict | 已修复 |

详见 `docs/case-fvm-dambreak-stoker/AUDIT.md`。

## 6. 测试证据

```
81 tests passed (核心 dam break + Galilean + 2D + oracle-free 位置度量)
- test_shallow_water_dam_break.py: 14 passed
- test_dambreak_stoker_series.py: 8 passed
- test_case_fvm_dambreak_stoker_pool.py: 4 passed
- test_dambreak_stoker_repro_verification.py: 3 passed
- test_shallow_water_galilean.py: 19 passed
- test_shallow_water_2d_dam_break.py: 12 passed
- test_dry_bed_front_position_oracle_free.py: 12 passed
- ... 及其他相关测试
```

## 7. 结论

**CASE-FVM-DAMBREAK-STOKER 验收通过**。所有锚点验收项（波前位置 x_f(t)、静水区水位阶跃、波形单调性）在 n ≥ 800 上均满足 <5% 容差，且关键量误差均 <3%。解析解通过自洽检验，网格收敛符合理论预期，扩展验证（伽利略提升、2D 确认、鲁棒性扫描）全部通过。三轮独立审计发现的缺陷均已修复并记录。

---

*本文件数字由 `tensorfvm.shallow_water.dam_break.run_case` 程序化生成，零手抄。*
*工件路径: `src/tensorfvm/verification/artifacts/shallow-water-dam-break/`*
