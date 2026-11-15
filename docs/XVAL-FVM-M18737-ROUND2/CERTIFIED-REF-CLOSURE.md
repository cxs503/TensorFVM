# XVAL-FVM-M18737-ROUND2 — Certified-<1% 基准闭合（honest-FAIL 裁决）

> **锚点**: `XVAL-FVM-M18737-ROUND2`（TensorFVM 的 m18737-round2 做法）
> **本文件**: 对 round-2「给下一手」建议（换基准 → certified-<1%）的**最终数值裁决**。
> **前置**: `docs/CASE-WOODWARD-COLELLA-certified-ref-proposal.md`（工具链与三道门已就位，
> 但 WENO5@6400/12800 工件未回填 → 本单补齐并完成裁决）。

---

## ① 物理背景 / 参数表

Woodward & Colella (1984) 双爆轰波：一维闭管 [0,1]，三压力平台（左 p=1000 / 中 p=0.01 /
右 p=100），两束爆轰波相向对撞，积分至 t=0.038。γ=1.4，两端反射壁。
TensorFVM 用 MUSCL-Hancock 二阶 + MC 限幅 + HLLC + SSP-RK（CFL 0.4）；
独立参考用 **WENO5 / SSP-RK3 / Rusanov**（`xcode/wc_crosscode.py`，与 TensorFVM 不共享离散）。

**本单新增参考档**（独立格式，更细网格）：

| 网格 | 格式 | 峰值密度 | 窗 L1 (vs 已发表参考) | 质量漂移 |
|---|---|---|---|---|
| 6400 | WENO5/SSP-RK3/Rusanov | 6.4578 | 5.02 % | 9.7e-13 |
| 12800 | WENO5/SSP-RK3/Rusanov | 6.4595 | 5.48 % | 1.9e-12 |

## ② 怎么做的（三道门 + 定级）

`wc_certified_ref.py` 用独立 WENO5@12800 作**新高分辨率参考**，对 TensorFVM MUSCL/HLLC
解重新定级。三道门保证「certified」一词诚实：

1. **参考自认证**：WENO5@6400 vs WENO5@12800 须 <1%（L1+峰值）。
2. **跨族地板**：MUSCL/HLLC vs WENO5@12800 残差须**小于** vs 已发表参考残差。
3. **守恒**：窗质量 ∫ρdx 在 [0.4,1] 上 <0.1%。

**本单修复**：`wc_certified_ref.py::_load` 的 400/800 档回退路径（`artifacts/wc_blast_profile_*.npz`
用 `x_got/rho_got` 键）未适配 → 已修（`x_got` 优先）。**且**：`certified` 判定原只查
「参考自认证」，未查「被定级解本身 <1%」→ 已修（`certified = 参考自认证 ∧ 地板移除 ∧ 被定级解<1%`）。

## ③ 计算结果（程序化注入，零手抄）

### 三道门逐项判定

| 门 | 度量 | 实测 | 判据 | 判定 |
|---|---|---|---|---|
| 1 参考自认证 | WENO5@6400 vs 12800 L1 | **0.544 %** | <1% | ✅ |
| 1 参考自认证 | WENO5@6400 vs 12800 峰值 | **0.023 %** | <1% | ✅ |
| 1 参考自认证 | 窗质量 | **0.0005 %** | <0.1% | ✅ |
| 2 被定级解峰值 | MUSCL/HLLC nx3200 vs WENO5@12800 峰值 | **0.758 %** | <1% | ✅ |
| 2 被定级解 L1 | MUSCL/HLLC nx3200 vs WENO5@12800 L1 | **1.776 %** | <1% | ❌ **FAIL** |
| 2 地板移除 | 1.776 % vs 已发表参考 3.079 % | 1.776 < 3.079 | 更小 | ✅ |
| 3 守恒 | 窗质量 ∫ρdx | **0.008 %** | <0.1% | ✅ |

### 定级裁决

```
certified          = False   （L1 门 1.776% > 1%，绑定）
graded_under_1     = False   （L1 未达标；峰值 0.758% 已达标）
reference          = WENO5@12800，自认证 <1%（L1 0.544%，峰值 0.023%）
vs_published_ref   = L1 3.079%，峰值 3.000%（round-2 数字）
vs_certified_ref   = L1 1.776%，峰值 0.758%，质量 0.008%
floor_removed      = True
```

## ④ 结果图

- `docs/woodward-colella-double-blast/artifacts/fig_certified_ref_profiles.png` —
  被定级 MUSCL/HLLC 剖面 vs 独立 WENO5@12800 参考（+ 已发表参考）。
- `docs/woodward-colella-double-blast/artifacts/fig_certified_ref_selfconv.png` —
  参考自收敛（WENO5@6400 vs 12800）。

## ⑤ 注意点（局限 / 适用范围）

1. **「certified <1%」未达成（L1 门）**：被定级解峰值 0.758% <1%，但 **L1=1.776% >1%**。
   round-2「~3% vs 已发表参考」残差**大部分**是参考侧离散误差（换更细独立参考后
   L1 3.08%→1.78%、峰值 3.00%→0.76%），但**未完全**移除——L1 仍停在 1.78%。
   诚实上限是 **certified <2%**（L1 1.776%），不是 <1%。
2. **参考自认证通过**：WENO5@6400 vs 12800 自收敛 <1%（L1 0.544%），故 WENO5@12800
   是**可信**的高分辨率参考；被定级解的 1.78% L1 是**真实**的（非参考侧噪声）。
3. **峰值 vs L1 不对称**：激波主导的峰值收敛快（0.76%），但逐点 L1 受接触间断/平台
   区域拖累（1.78%）——L1 是更严苛的「certified」判据。
4. **`_load` 回退键名缺陷**：400/800 档旧工件用 `x_got/rho_got`，已修；不影响数值。

## ⑥ 误差判定（逐项显式回答「是否 <15% / <3% / <1%」）

| 关键量 | 实测 | <15%（总账） | <3% | <1%（certified） |
|---|---|---|---|---|
| 参考自认证 L1（WENO5@6400 vs 12800） | 0.544 % | ✅ | ✅ | ✅ |
| 参考自认证 峰值 | 0.023 % | ✅ | ✅ | ✅ |
| 被定级解 峰值（MUSCL/HLLC vs WENO5@12800） | 0.758 % | ✅ | ✅ | ✅ |
| 被定级解 **L1** | **1.776 %** | ✅ | ✅ | ❌（1.776%） |
| 窗质量 ∫ρdx | 0.008 % | ✅ | ✅ | ✅ |
| **总账 <10%** | 全部良态 <10% | ✅ **PASS** | — | — |

**诚实结论**：本锚点**总账 PASS**（全部良态 <10%）。但「certified <1%」的**硬判据未达成**——
被定级解 L1=1.776% >1%，是**绑定门**。峰值与参考自认证均 <1%，故诚实上限为
**certified <2%**。此结论与 round-2「残差 ~3% 主要由参考侧误差主导」**自洽**（更细独立
参考移除大部分，但未完全）。

## 复现命令

```bash
PYTHONPATH=src python docs/woodward-colella-double-blast/xval/wc_certified_ref.py
PYTHONPATH=src python -m pytest tests/test_compressible_wc_certified_ref.py -q  # 5 passed
PYTHONPATH=src python -m pytest tests/test_compressible_wc_xval.py -q            # 13 passed
```

## 给下一手的建议

1. **新账条目建议**：`XVAL-FVM-M18737-CERTIFIED-REF`（已由本单证伪「<1%」）——
   改验收为 **「MUSCL/HLLC nx=3200 vs 独立 WENO5@12800 参考，L1 <2% 且峰值 <1%；
   且 WENO5@6400 vs 12800 自收敛 <1%」**（量化验收，L1 门 2% 是诚实可达上限）。
2. **若想真 <1%**：需在接触间断/平台区域用更高阶（如 WENO7）或自适应加密，
   而非仅加密均匀网格（L1 在 6400→12800 已趋平，均匀加密收益递减）。
