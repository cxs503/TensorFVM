# XVAL-FVM-M18737-ROUND2 Round-15 闭合：参考阶梯 6400 档补 CFL 0.8 变体，eps_ref 升为已认证误差棒

> 锚点 `XVAL-FVM-M18737-ROUND2`（TensorFVM 的 m18737-round2 做法），总账验收 <15 %。
> 判据在 `PREDECLARATION-r15.md` 先于数字冻结；本轮唯一管线改动（6400 取档表覆盖）
> 以 E1r6400 门通过为前提，**不改任何阈值/口径**。
> 本文件由 `r15_ref_ladder_cfl.py` 从 `r15_certified_rung.json` 程序化渲染（零手抄）；
> 前轮文档（含 `R14-CLOSURE.md`）**不改写**。

## ① 案例介绍（物理背景 / 几何 / 参数表）

Woodward & Colella (1984) 双爆轰波：一维闭管 x∈[0,1]，三压力平台（左 p=1000 /
中 p=0.01 / 右 p=100），ρ=1、u=0，两束爆轰波相向对撞（t≈0.028），积分至 t=0.038，
γ=1.4，两端反射壁。**无解析解** ⇒ 任何「误差」都是两个数值解之差，参考解自身的
离散误差必须作为误差棒计入。

| 项 | 值 | 出处 |
|---|---|---|
| 被定级格式 | MUSCL-Hancock 二阶 + MC 限幅 + HLLC + SSP-RK | `compressible/cases.double_blast` |
| 参考格式 | WENO5 + SSP-RK3 + Rusanov（零共享离散） | `ladder/wc_ladder_weno5_cfl.py` |
| 比较窗口 / 度量 | x∈[0.4,1.0]，梯形归一化 L1（与前轮逐字一致） | `xval/wc_graded_refinement.py` |
| 冻结参考 | WENO5@12800（自认证 L1 0.544 % vs @25600） | 既有 |
| 本轮新增计算 | **1 个档**：WENO5@6400 CFL 0.8（tag `nx6400_weno5_cfl0.8`） | 见 ② |

## ② 怎么做的（方法 / 流程 / 关键设置）

1. **新档**：`wc_ladder_weno5_cfl.py --sizes 6400 --cfl 0.8`（作业 `cj4398`，
   1 核，~6 min），工件回填 `ladder/data/`（npz + metrics json 成对入库）。
2. **E0 工件门**（同 R13 三道，含「metrics 里 cfl 必须与档名一致」）。
3. **E1r6400 入场门**：同格 CFL 0.4 vs 0.8 的互相 L1 < 0.2 %（与 E1/E1r 同阈值）。
4. **冻结管线重跑**：`r13_certified_rung.py`（import，不复制）；门过 ⇒ 6400 取档表
   覆盖为 `("weno5_cfl0.8", "weno5")`（预声明改动），门不过 ⇒ 默认表（R14 行为）。
5. **不变量复核**：被定级族阶梯 / E1 / E1r / E2 / E3 与 `r14_certified_rung.json`
   逐位比对（同输入工件 ⇒ 必须逐位相同）：全部一致（被定级族阶梯 / E1 / E1r / E2 / E3 与 R14 逐位相同）。
6. 求解器一行未改。

## ③ 计算结果（程序化注入，零手抄）

### E0 工件门（新档）

| 项 | 值 |
|---|---|
| cells | 6400 |
| dx 相对误差 | 2.134e-14 |
| ρmin / ρmax | 0.1455 / 6.4589 |
| steps / 秒 | 12837 / 151.3 |
| cfl（metrics） | 0.8 |
| mass drift | 4.856115509710435e-13 |
| 通过 | ✅ |

### E1r6400 CFL 入场门（阈值 0.2 %，R15 预注册）

| E1r6400（参考族 @6400） | 6400 | `nx6400_weno5` vs `nx6400_weno5_cfl0.8` | 0.0231 | 0.0169 | ✅ 准入 |

### 参考阶梯（本轮取档）

| nx | tag | CFL | L1 vs 上一档 [%] | 观测阶 |
|---|---|---|---|---|
| 6400 | `nx6400_weno5_cfl0.8` | 0.8 | — | — |
| 12800 | `nx12800_weno5_cfl0.8` | 0.8 | 0.5408 | — |
| 25600 | `nx25600_weno5_cfl0.8` | 0.8 | 0.2951 | 0.8740 |

### E4 诚实上限（参考侧误差棒，licensed=true）

| 口径 | eps_ref [%] | 承重假设 |
|---|---|---|
| B0 | 0.3544 | measured order p is exact |
| B1 | 0.5454 | order uncertain by +-0.25, propagated |
| B2 | 0.4430 | ASME V&V-20: asymptotic range reached (R<1, p stable) |
| B3 | 1.0631 | Roache: observed order not yet asymptotic |
| B4 | 0.6495 | difference series geometric with non-increasing ratio |

| nx | L1 [%] | B0 | B1 | B2 | B3 | B4 |
|---|---|---|---|---|---|---|
| 800 | 6.0464 | 6.4008 | 6.5918 | 6.4894 | 7.1095 | 6.6959 |
| 1600 | 3.2992 | 3.6536 | 3.8446 | 3.7422 | 4.3623 | 3.9487 |
| 3200 | 1.7756 | 2.1300 | 2.3210 | 2.2186 | 2.8387 | 2.4251 |
| 6400 | 0.8931 | 1.2475 | 1.4385 | 1.3361 | 1.9562 | 1.5426 |
| 12800 | 0.3209 | 0.6753 | 0.8663 | 0.7639 | 1.3840 | 0.9704 |

| 口径 | <1.5 % | <1 % | 总账 <15 % |
|---|---|---|---|
| B0 | pass | pass | pass |
| B1 | pass | pass | pass |
| B2 | pass | pass | pass |
| B3 | pass | fail | pass |
| B4 | pass | pass | pass |

参考侧：e_pair = 0.2951 %，p_ref = 0.8740，
licensed = true。

## ④ 结果图

- `figures/fig_r15_ref_ladder.png`：(a) 参考阶梯三档同仪器（全 CFL 0.8）的相邻档
  L1 对差商与 eps_ref；(b) 五口径诚实上界（licensed 状态标注）；(c) 6400 档
  CFL 0.4 vs 0.8 剖面（E1r6400 控制）。

## ⑤ 注意点（坑 / 局限 / 适用范围）

1. **licensed 的含义**：三档同仪器 ⇒ eps_ref 可作误差棒引用；但「vs 自认证参考」
   的口径不变，对精确解的 <1 % 声明**仍不成立**（无解析解）。
2. **B3（非渐近 Fs=3）若仍越 1 % 线**：honest FAIL 不覆盖（预注册分支）。
3. 6400 档 CFL 0.4 工件**保留**（E1r6400 的对照侧 + 门不过时的回退档）。
4. 全部结论限于「vs 自认证 WENO5@12800」。

## ⑥ 误差判定（逐项，显式回答）

| 判据 | 最不利实测 | 判定 |
|---|---|---|
| 总账 <15 % | 最保守上界 B3 = 1.3840 % | **PASS** |
| <3 % | 同上 | **PASS** |
| <1 %（vs 自认证参考，点估计） | L1 = 0.3209 % | **PASS** |
| <1 %（诚实上界，B0/B1/B2/B4） | 0.6753 / 0.8663 / 0.7639 / 0.9704 % | **PASS** |
| <1 %（诚实上界，B3 非渐近） | 1.3840 % | **honest FAIL** |

**一句话裁决**：`CERTIFIED_UNDER_1PCT_AT_NX12800 | NO_FLOOR_DETECTED | HONEST_UNDER_1PCT_B0_B1_B2_B4`
