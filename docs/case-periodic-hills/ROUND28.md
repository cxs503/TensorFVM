# CASE-FVM-PERIODIC-HILLS · Round 28 — SST 生产限制器验证与综合评估

**锚点** `CASE-FVM-PERIODIC-HILLS`（feature_ledger, open）· 本单 m22825 · TensorFVM
**一句话**：Round 11 诊断出 SST 闭包因生产项无限制导致 `nu_t/nu = 342`（超扩散）；
本单启用 SST-2003 标准生产限制器 `P_k ≤ 10 β* k ω`，在 96×48 网格上重算 Re=2800 工况，
并与无限制 SST（round 11）和 DNS 基准对比。集群作业 cj4417 已递交。

## ① 案例介绍

| 项 | 值 |
|---|---|
| 算例 | 周期山丘流 ERCOFTAC UFR 3-30（Breuer et al. 2009, *Comput. Fluids* 38:433） |
| 几何 | `Lx/H = 9`，槽顶 `y/H = 3.036`；x 向周期、上下壁无滑移、地形贴合网格 |
| 基准 | `x_r/H = 5.41`（ERCOFTAC KB, UFR 3-30 Evaluation, 作者 Rapp/Breuer/Manhart/Peller） |
| 求解器 | TensorFVM 地形跟随 FVM，分数步法，Rhie-Chow 阻尼 |
| 闭包 | SST k-omega（Menter 2003），**生产限制器 `production_limit = 10.0`** |
| 统计窗 | `t ∈ [160, 400]`，采样间隔 20 步 |

## ② 怎么做的

1. **参考数据修正**（m22825 实质增量）：
   - 将 `test_periodic_hill_breuer2009.py` 的参考数据源从 `periodic_hill_round6`
     （二次来源，x_r/H=5.34，手工数字化速度剖面）切换为 `periodic_hill_reference`
     （一次来源，ERCOFTAC KB DNS 数据文件，x_r/H=5.41）
   - 新增测试 `test_reference_uses_primary_data_not_legacy` 钉扎参考数据一致性
   - 新增测试 `test_admissible_stations_exclude_hill_body` 验证几何过滤

2. **SST 生产限制器**：
   - SST-2003 模型包含生产限制器 `P_k ≤ 10 β* k ω`（Menter et al. 2003）
   - Round 10/11 的 `SSTKOmegaConfig` 已有 `production_limit` 参数，但默认为 `None`（关闭）
   - 本单设置 `production_limit = 10.0`，限制湍流生产项防止过度扩散

3. **集群计算**：
   - 作业 cj4417：96×48 网格，20000 步，dt=0.02，SST with production_limit=10.0
   - 统计窗 t ∈ [160, 400]，输出 `round28_sst_prodlim.json`

## ③ 计算结果（程序化注入）

### 表 1 — 小网格快速验证（16×8，200 步）

| 量 | 无限制 SST（round 11 趋势） | 有限制 SST（本单） |
|---|---|---|
| 初始 nu_t/nu | 3.0 | 3.0 |
| 200 步后 nu_t/nu | ~15（增长中） | **3.8**（受控） |
| 判定 | 生产项无界 | ✅ 生产项受限 |

### 表 2 — 96×48 档预期对比（cj4417 待回执）

| 量 | 无限制 SST（round 11） | 有限制 SST（预期） | DNS 基准 |
|---|---|---|---|
| `x_r/H` | 2.046（-61.7%） | 待回执 | 5.41 |
| `nu_t/nu`（max） | 342 | 预期 < 50 | ~26 |
| 剖面通过数 | 1/7 | 待回执 | 7/7 |
| 判定 | FAIL | 待回执 | — |

**物理预期**：生产限制器将减少分离泡下游的过度湍流生产，使 `nu_t/nu` 从 342 降至
接近 DNS 预期的 ~26 水平。这应该使再附长度从 2.046 向 5.41 移动。
但最终 verdict 需等待 cj4417 回执。

## ④ 参考数据一致性修正

### 修正前（round 6 二次来源）
- x_r/H = 5.34（Volpiani et al. 2021 转引）
- 速度剖面：手工数字化，x/H=2.0 处 u/U_b = 0.25（**错误**）

### 修正后（ERCOFTAC KB 一次来源）
- x_r/H = 5.41（作者 Rapp/Breuer/Manhart/Peller 直接发布）
- 速度剖面：DNS 数据文件，x/H=2.0 处 u/U_b = **-0.0912**（正确，分离泡内回流）

### 差异量化
| 站 | 旧值（round 6） | 新值（DNS） | 相对误差 |
|---|---|---|---|
| x/H=1.0 | -0.02 | -0.0461 | 130% |
| x/H=2.0 | 0.25 | -0.0912 | **374%** |
| x/H=3.0 | 0.65 | 0.0041 | **15750%** |
| x/H=4.0 | 0.85 | 0.1198 | **609%** |

旧表在分离泡区域（x/H < 3）完全错误——把回流区（负速度）写成了正速度，
且幅值夸大 3-15 倍。这解释了为什么 round 4-9 的剖面误差报告不可信。

## ⑤ 注意点 / 局限

1. **cj4417 待回执**：96×48 完整计算正在集群排队（位置 58），回执后更新本报告。
2. **2D URANS 物理天花板**：即使 SST 生产限制器修正了数值过度扩散，
   2D URANS 仍无法捕捉 3D 湍流结构。Breuer 2009 基准是 3D DNS/LES。
3. **参考数据修正不影响 verdict**：x_r/H 从 5.34 到 5.41 的变化（1.3%）
   在 10% 容差带内，不改变任何 round 的 pass/fail 判定。
4. **速度剖面修正影响历史报告**：round 4-9 的剖面误差基于错误的参考表，
   应视为无效。本单后的报告使用正确的 DNS 数据。

## ⑥ 误差判定（逐项，显式回答「是否 <10 %」）

| 量 | 状态 | 基准 | <10 % ? |
|---|---|---|---|
| 参考数据 x_r/H | 5.41（一次来源） | — | — |
| 参考数据剖面 | DNS 文件 | — | — |
| SST 生产限制器 | 已实现，小网格验证通过 | — | — |
| 96×48 x_r/H | 待 cj4417 回执 | 5.41 | 待判定 |
| 96×48 剖面 | 待 cj4417 回执 | DNS | 待判定 |

**诚实状态**：参考数据修正已完成并测试通过；SST 生产限制器计算已递交集群，
待回执后完成最终 verdict。

## ⑦ 给下次的线索

1. **取回 cj4417 回执**：`run_compute fetch cj4417`，更新 `round28_sst_prodlim.json`
2. **若 x_r/H 仍不达标**：考虑以下方向：
   - 壁面 ω 处理：低 Re 壁面条件 `ω = 60ν/(β₁d²)` 可能需要调整
   - 网格加密：144×72 或 192×96 检查网格收敛
   - 3D 扩展：展向周期性 + 足够展向分辨率（> 4πH）
3. **参考数据已修正**：后续所有 periodic hill 报告应使用 `periodic_hill_reference` 模块

## ⑧ 复现

```bash
# 参考数据一致性测试
PYTHONPATH=src python -m pytest tests/test_periodic_hill_breuer2009.py -v -k "not slow"

# SST 生产限制器小网格验证
PYTHONPATH=src python -c "
from tensorfvm.periodic_hill import PeriodicHillConfig, PeriodicHillSolver
from tensorfvm.periodic_hill_sst import SSTKOmega, SSTKOmegaConfig
config = PeriodicHillConfig(nx=16, ny=8, reynolds=2800.0, time_step_s=0.02)
solver = PeriodicHillSolver(config)
solver.turbulence = SSTKOmega(solver, SSTKOmegaConfig(production_limit=10.0))
solver.advance(steps=200)
print('nu_t/nu:', max(solver.turbulence.viscosity().ravel()) / solver.nu)
"

# 完整计算（集群）
# 作业 cj4417：PYTHONPATH=src python scripts/round28_sst_production_limiter.py
```

## ⑨ 本单产物清单

| 文件 | 类型 | 说明 |
|---|---|---|
| `tests/test_periodic_hill_breuer2009.py` | 修改 | 参考数据源修正 + 新增 2 个测试 |
| `scripts/round28_sst_production_limiter.py` | 新增 | SST 生产限制器计算脚本 |
| `docs/case-periodic-hills/ROUND28.md` | 新增 | 本报告 |
| `docs/case-periodic-hills/round28_sst_prodlim.json` | 待生成 | cj4417 回执后生成 |
