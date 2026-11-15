# XVAL-FVM-M18737-ROUND2: Certified <1% 基准闭合（nx=6400 达成）

> **锚点**: `XVAL-FVM-M18737-ROUND2`（TensorFVM 的 m18737-round2 做法）
> **验收标准**: 对标误差 <15%（owner 精度标准：对标误差 >15% 不可接受）
> **本单增量**: 将 `wc_certified_ref.py` 的定级阶梯扩展至 nx=6400，发现 **MUSCL/HLLC nx=6400 对独立 WENO5@12800 参考实现 certified <1%**（L1=0.893%，峰值=0.044%）；原 nx=3200 为绑定门（L1=1.776% > 1%）；更新回归测试 7 项全绿。

---

## ① 案例介绍（物理背景 / 几何 / 参数表）

Woodward & Colella (1984) 双爆轰波：一维闭管 [0,1]，三压力平台（左 p=1000 / 中 p=0.01 /
右 p=100），两束爆轰波相向对撞，积分至 t=0.038。γ=1.4，两端反射壁。

| 参数 | 值 | 出处 |
|---|---|---|
| 域 | x∈[0,1]，1D | W&C 1984 |
| 初始 (ρ,u,p) | (1,0,1000) x≤0.1 / (1,0,0.01) 0.1<x<0.9 / (1,0,100) x≥0.9 | Dumbser et al. JCP 2019 |
| γ / t_end / 壁面 | 1.4 / 0.038 / 两端反射 | 同上 |
| TensorFVM 格式 | MUSCL-Hancock 二阶 + MC 限幅 + HLLC + SSP-RK，CFL 0.4 | `compressible/cases.double_blast` |
| 独立参考 | WENO5 + SSP-RK3 + Rusanov（不共享离散） | `xcode/wc_crosscode.py` |
| 认证参考 | WENO5@12800（自认证：WENO5@6400 vs 12800 L1=0.544% <1%） | `ladder/data/` |

## ② 怎么做的（方法 / 流程 / 关键设置）

1. **扩展定级阶梯**: 在 `wc_certified_ref.py` 中将 `HLLC_RUNGS` 从
   `("nx800_hllc", "nx1600_hllc", "nx3200_hllc")` 扩展为包含 `"nx6400_hllc"`。
2. **动态最细档**:  verdict 改为取 `measured_tags[-1]`（最细已测档），而非硬编码 nx=3200。
3. **三道门不变**: 参考自认证 <1%、地板移除、守恒 <0.1% 判据保持不变。
4. **回归测试更新**: 新增 `test_muscl_hllc_nx6400_is_certified_under_1_percent`（验证 certified=True）、
   `test_nx3200_was_the_binding_gate`（记录历史绑定门）、
   `test_convergence_monotone_under_1`（验证 L1 单调收敛）。

## ③ 计算结果（程序化注入，零手抄）

### 定级阶梯（vs WENO5@12800 独立参考）

| 网格 (nx) | L1 误差 % | 峰值误差 % | 质量误差 % | 判定 (<1%) |
|---|---|---|---|---|
| 800 | 6.046 | 4.137 | 0.136 | ❌ |
| 1600 | 3.299 | 0.127 | 0.048 | ❌ |
| 3200 | 1.776 | 0.758 | 0.008 | ❌（L1 绑定门） |
| **6400** | **0.893** | **0.044** | **0.0002** | ✅ **PASS** |

### 三道门逐项判定

| 门 | 度量 | 实测 | 判据 | 判定 |
|---|---|---|---|---|
| 1 参考自认证 | WENO5@6400 vs 12800 L1 | 0.544 % | <1% | ✅ |
| 1 参考自认证 | WENO5@6400 vs 12800 峰值 | 0.023 % | <1% | ✅ |
| 2 地板移除 | nx3200 L1 vs 已发表参考 | 1.776 < 3.079 | 更小 | ✅ |
| 3 守恒（nx=6400） | 窗质量 ∫ρdx | 0.0002 % | <0.1% | ✅ |
| **certified** | nx6400 L1 + 峰值 | 0.893% + 0.044% | 均 <1% | ✅ **CERTIFIED** |

### 收敛性

| 相邻档 | L1 比值 | 观测阶 |
|---|---|---|
| 800→1600 | 1.83 | 0.87 |
| 1600→3200 | 1.86 | 0.89 |
| **3200→6400** | **1.99** | **0.99** |

nx=3200→6400 的 L1 比值 ≈2.0，观测阶 ≈1.0，表明 **MUSCL-HLLC 在 nx=6400 仍处于一阶渐近区**。

## ④ 结果图

（图由 `wc_certified_ref.py` 程序化生成，见 `docs/woodward-colella-double-blast/artifacts/`）

- `fig_certified_ref_profiles.png` — 被定级 MUSCL/HLLC 剖面 vs 独立 WENO5@12800 参考
- `fig_certified_ref_selfconv.png` — 参考自收敛（WENO5@6400 vs 12800）

## ⑤ 注意点（局限 / 踩过的坑 / 适用范围）

1. **nx=3200 是历史绑定门**: 原 certified 分析只到 nx=3200，L1=1.776% > 1% 导致 honest-FAIL。
   本单发现 nx=6400 时 L1=0.893% < 1%，certified  claim **达成**。
2. **nx=800 质量误差 0.136% > 0.1%**: 这是粗网格的预期行为，不影响 certified 判定（仅检查 nx≥1600）。
3. **收敛阶 ~1.0**: MUSCL-HLLC 在激波问题中表现为一阶（限幅导致），nx=6400 仍在渐近区。
4. **参考自认证通过**: WENO5@6400 vs 12800 自收敛 <1%，故 WENO5@12800 是可信的高分辨率参考。

## ⑥ 误差判定（逐项显式回答「是否 <15% / <1%」）

| 关键量 | 实测 | <15%（总账） | <1%（certified） |
|---|---|---|---|
| 参考自认证 L1 | 0.544 % | ✅ | ✅ |
| 参考自认证 峰值 | 0.023 % | ✅ | ✅ |
| 被定级解 L1（nx=6400） | **0.893 %** | ✅ | ✅ |
| 被定级解 峰值（nx=6400） | **0.044 %** | ✅ | ✅ |
| 窗质量 ∫ρdx（nx=6400） | 0.0002 % | ✅ | ✅ |
| **总账 <10%** | 全部良态 <10% | ✅ **PASS** | — |
| **certified <1%** | L1+峰值均 <1% | — | ✅ **CERTIFIED** |

**诚实结论**: 本锚点**总账 PASS**（全部良态 <10%）。**certified <1% 达成**——
MUSCL/HLLC nx=6400 对独立 WENO5@12800 参考的 L1=0.893% < 1%，峰值=0.044% < 1%，
参考自认证 L1=0.544% < 1%，地板移除确认。

---

## 复现命令

```bash
PYTHONPATH=src python docs/woodward-colella-double-blast/xval/wc_certified_ref.py
PYTHONPATH=src python -m pytest tests/test_compressible_wc_certified_ref.py -v  # 7 passed
PYTHONPATH=src python -m pytest tests/test_compressible_wc_xval.py \
    tests/test_compressible_wc_certified_ref.py \
    tests/test_compressible_wc_method_claim.py \
    tests/test_wc_weno7_ladder.py -q  # 40 passed
```

## 变更文件

- `docs/woodward-colella-double-blast/xval/wc_certified_ref.py` — 扩展 HLLC_RUNGS 至 nx=6400，动态最细档 verdict
- `tests/test_compressible_wc_certified_ref.py` — 更新 7 项回归门（新增 nx6400 certified 验证 + 历史绑定门记录 + 单调收敛检查）

---

## 勘误（round-12 self-audit，2026-11-16）—— `certified <1 %` 是点估计，不是证明

上文「certified <1 % 达成」的依据是**两级**门（WENO5@6400 vs WENO5@12800），两级既无观测阶
也无误差棒 ⇒ `L1(MUSCL@6400, WENO5@12800)=0.893 %` 只是点估计。补上参考侧误差棒后：

| 量 | 值 |
|---|---|
| 参考阶梯观测阶 `p_ref` | 0.8327 |
| `eps_ref`（参考侧误差，docs 实现 / 包实现） | 0.7076 % / 0.7181 % |
| 诚实上界 `L1 + eps_ref` | **1.601–1.611 % > 1 %** |
| 族内 Richardson（不依赖参考）自身误差 | **1.523 % > 1 %** |
| `certified` / `certified_point_estimate` | **False** / True |

判据未改（`<1 %`），改的是「两级门能支撑什么」：**诚实状态是「点估计 <1 %，可证明上界 1.61 %」**，
总账 `<15 %` 与 `<3 %` 仍 PASS。`wc_certified_ref.py` 的 `certified` 布尔已改为要求诚实上界
`<1 %`（新增 `certified_point_estimate` / `certified_provable` 两个显式字段），
回归门同步更新为 9 项。详见 `CERTIFIED-NX6400-ERRATA.md`（数字全部由
`certified_nx6400_errata.py` 程序写入）。
