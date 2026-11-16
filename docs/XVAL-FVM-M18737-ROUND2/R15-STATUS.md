# XVAL-FVM-M18737-ROUND2 Round-15 状态：xval 分析脚本修复（13 个失效测试转绿）+ 参考阶梯同仪器化预注册落地（作业在飞）

> 锚点 `XVAL-FVM-M18737-ROUND2`（TensorFVM 的 m18737-round2 做法），总账验收 <15 %。
> 本单（m22802）交付两件事：① 修复 R14 回填引入的 `wc_xval.py` 回归（13 个失效测试转绿）；
> ② 把 R14 ⑤-2 悬空的「参考阶梯 6400 档无 CFL 0.8 变体 ⇒ eps_ref 只能记探针」
> 按预注册推进到**只差取回工件**的状态。

## ① 案例介绍

Woodward & Colella (1984) 双爆轰波（一维闭管 x∈[0,1]，左 p=1000 / 中 p=0.01 /
右 p=100，ρ=1、u=0，t_end=0.038，γ=1.4，两端反射壁）。无解析解 ⇒ 误差恒为
「两数值解之差」。被定级格式 MUSCL-Hancock + MC + HLLC + SSP-RK；参考格式
WENO5 + SSP-RK3 + Rusanov（零共享离散）。比较窗口 x∈[0.4,1.0]，梯形归一化 L1。

## ② 怎么做的

### A. 缺陷修复：`xval/wc_xval.py` 的 `finest_tag` 回归

- **症状**：`tests/test_compressible_wc_xval.py` 13 个测试在 main 上全部
  `KeyError: 'nx12800_hllc'`（fixture `analysis = wc_xval.run()` 崩溃）。
- **根因**：R14 把被定级族 12800 档以 CFL 0.8 变体回填（CFL 0.4 的 12800 档
  ~178 min 超单作业墙钟，明确不做，见 `PREDECLARATION-r13.md`），但
  `wc_xval.run()` 仍按 `f"nx{finest_fvm['nx']}_hllc"` **重构**最细档 tag，
  该 tag 从未存在 ⇒ KeyError。
- **修复**（最小改动，2 行）：`finest_tag` 直接取最细 fvm 档的**实际存储 tag**
  （`max(..., key=lambda t: profiles[t]["nx"])`），不再重构 tag 名。
- **验证**：`pytest tests/test_compressible_wc_xval.py tests/test_compressible_wc_certified_ref.py
  tests/test_compressible_wc_method_claim.py tests/test_wc_weno7_ladder.py -q`
  ⇒ **42 passed**（修复前 29 passed + 13 errors；已用 `git stash` 在干净 main 上
  复现 13 errors，确认非本单引入）。

### B. 参考阶梯同仪器化（R15 预注册，作业在飞）

- `PREDECLARATION-r15.md`（先于作业提交写定）：新档 WENO5@6400 CFL 0.8
  （tag `nx6400_weno5_cfl0.8`）+ E0 工件门 + E1r6400 入场门（L1 < 0.2 %，
  与 E1/E1r 同阈值）+ 唯一管线改动（6400 取档表覆盖，门过才生效）。
- `r13_certified_rung.py`：参考阶梯取档表提为模块常量 `REF_LADDER_TAGS`
  （默认值与 R14 行为逐字一致；已验证默认路径重跑与 `r14_certified_rung.json`
  的 ladder/orders/floor/E1/E1r/E2/E3/adjudication **逐位相同**）。
- `r15_ref_ladder_cfl.py`：驱动（E0 + E1r6400 + 冻结管线重跑 + 不变量复核 +
  渲染 + 出图），门不过走预注册负面分支。
- 作业 `cj4398`（`wc_ladder_weno5_cfl.py --sizes 6400 --cfl 0.8`，1 核，~6 min）
  已递交，本单窗口内集群拥堵（82 在飞）未回件。

## ③ 计算结果（程序化注入，零手抄）

修复后的 xval 分析（`r15_xval_figures.py` → `r15_xval_repaired.json`）：

| 量 | 值 | 判据 |
|---|---|---|
| 最细档 | `nx12800_hllc_cfl0.8` | — |
| L1 vs 标定参考 | 4.4199 % | <15 % ✅ / <10 %（ledger tolerance）✅ |
| 峰值 vs 标定参考 | 2.1730 % | <3 % ✅ |
| 窗质量 | 0.0164 % | <0.25 % ✅ |
| 交叉码 L1（200→12800） | 4.5998 % → 0.3209 % | 单调收敛 ✅ |
| 误差预算（最细档 L1） | 自身 0.5776 % / 参考侧 4.3820 %（自身占比 13.07 %） | 残差由参考侧主导 |

## ④ 结果图

- `figures/fig_r15_xval_repaired.png`：(a) 全阶梯（含 nx=12800）vs 标定参考的
  L1（两族）与 15 % / 3 % 判据线；(b) 交叉码收敛（独立离散互收）。

## ⑤ 注意点

1. **R15 数字闭合未在本单完成**：`cj4398` 在飞，回件后按
   `PREDECLARATION-r15.md` ④ 执行（fetch → 回填 `ladder/data/` →
   `PYTHONPATH=src python docs/XVAL-FVM-M18737-ROUND2/r15_ref_ladder_cfl.py`
   → `R15-CLOSURE.md`）。**不得**在门结果出来前改任何阈值。
2. `wc_xval.py` 的修复只改 tag 选取，不改任何度量/窗口/判据；
   42 个测试全绿即回归门。
3. 本单未动求解器（src/ 零改动）。

## ⑥ 误差判定（逐项，显式回答）

| 判据 | 最不利实测 | 判定 |
|---|---|---|
| 总账 <15 % | 最细档 L1 = 4.4199 %（vs 标定参考） | **PASS** |
| <3 % | 峰值 2.1730 % | **PASS** |
| 失效测试 | 13 errors → 0 | **PASS**（42 passed） |
| 管线不变量 | 默认路径与 R14 逐位一致 | **PASS** |
| eps_ref 升「已认证棒」 | 待 `cj4398` 回件 + E1r6400 门 | **not_measured**（如实） |

**一句话**：`XVAL-SCRIPT-REPAIRED_42_GREEN | R15-PREDECLARED_JOB_CJ4398_IN_FLIGHT`
