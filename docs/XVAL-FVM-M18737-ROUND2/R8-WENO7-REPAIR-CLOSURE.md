# XVAL-FVM-M18737-ROUND2 Round-8：修好 WENO7 探针（保正通量限制）再问归因

> 锚点 `XVAL-FVM-M18737-ROUND2`（TensorFVM 的 m18737-round2 做法），验收 <15 %。
> **前置**：round-6（`R6-ORDER-AUDIT-CLOSURE.md`，总账 D22）把 WENO7 探针**证伪**（nx=200/800 NaN、nx=1600 峰值 +3.9 %），并给出下一手动作①「修边界/正性，并用含激波收敛测试锁住」。本文执行该动作并重新提问。
> **状态**：`PARTIAL/HONEST-FAIL`；判据 S1–S5 见 ⑥，`certified-<1 %` **未推进**（仍 False）。

## ① 案例与仪器

Woodward & Colella (1984) 双爆轰波：x∈[0,1] 闭管，左/中/右 p=1000/0.01/100，ρ=1、u=0，γ=1.4，t_end=0.038，两端反射壁，无解析解 ⇒ 误差恒为两数值解之差。

| 角色 | 格式 | 档位 |
|---|---|---|
| 被定级 | MUSCL-Hancock + MC + HLLC + SSP-RK（TensorFVM），CFL 0.4 | 1600/3200/6400 |
| 参考族（审计过） | WENO5 + SSP-RK3 + Rusanov（有限差分） | 400…12800 |
| 本轮探针 | WENO7 + SSP-RK3 + Rusanov + **保正通量限制** | 400/800/1600/3200/6400 |

判定窗口 x∈[0.4,1.0]，比较用守恒限制（粗单元均值 = r 个细单元均值）——沿用 round 2–7 口径，本轮不改仪器。

## ② 缺陷定位（round-6 的 NaN 到底在哪）

对 nx=200、CFL 0.4 的重放加了诊断（`/tmp` 探针，结论写进本文件）：首个**不可容许态**出现在 **SSP-RK 第 1 级**的**单元均值**上（单格 p = −0.0054），而不是重构出的界面态。因此 round-6 的「正性回退」机制（只检查界面态）**结构上就看不见**这个问题：它修的是脸，坏的是格心。

## ③ 修复：通量级保正限制（守恒且可终止）

每个 RK 级写成 `U(θ) = P_lo + θ (dU_ho − dU_lo)`，`P_lo = P + dU_lo` 为一阶 LLF 步（CFL≤1/2 下可容许）。`θ` 取使**所有**单元可容许的最大值，闭式求解：`ρ(θ)` 仿射、`q = 2ρE − m²` 二次 ⇒ 逐格二次根，再对格取 min。三条性质是**结构性的**，不是运气：

* **守恒**：`θ` 只是重标一个通量差 ⇒ 通量望远镜消去（实测相对质量漂移见表 S1，~1e-14）。被否证的两个替代方案：状态混合（守恒破了，nx=200 实测 5.5e-5）、整步 dt 减半重试（守恒但对**空间**越界无效：dt 掉到 9.1e-13 后仍失败）。
* **可终止**：`θ=0` 即一阶 LLF 步，CFL 0.4≤1/2 下可容许；释放式对半衰减 ⇒ 必停。
* **光滑区惰性**：无风险时 `θ=1` 精确成立 ⇒ 熵波阶不受影响（判据 S2）。

## ④ 结果（程序化注入）

### S1 稳健性/守恒账

| nx | status | step_status | admissible | pp_failures | θ_min | 限幅级数 | 质量漂移 | 秒 |
|---|---|---|---|---|---|---|---|---|
| 400 | ok | complete | True | 0 | 1.0000 | 0 | 0.000 | 9.1 |
| 800 | ok | complete | True | 0 | 0.6193 | 11 | 0.000 | 22.6 |
| 1600 | not_measured | None | None | None | — | None | — | — |
| 3200 | not_measured | None | None | None | — | None | — | — |
| 6400 | not_measured | None | None | None | — | None | — | — |

**S1 判定：FAIL**

### S2 精度未被修复吃掉（光滑熵波，含限制器）

`not_measured`：entropy_order_r8.json absent (job not back) —— 不得读作 PASS。

### S3 同网格峰值一致性

| nx | peak WENO7 | peak WENO5 | Δ [%] | ≤1 % |
|---|---|---|---|---|
| 800 | 6.6287 | 6.2951 | 5.30 | ❌ |

**S3 判定：FAIL**（峰值是低阶收敛泛函：两族在粗网格上从不同侧逼近同一极限，1 % 的同网格门偏紧——此处照实记 FAIL，不移动判据。）

### S4 同网格跨阶差是否随加密收缩

| nx | L1(WENO7−WENO5) [%] |
|---|---|
| 400 | 5.7632 |
| 800 | 3.3978 |

相邻档观测阶：400→800: p=0.762。

**S4 判定：FAIL**（nx=3200 处 — % ≤ 1 %）

### S5 归因：高阶族人是否更贴近参考族极限

参考取 WENO5@12800（族内最细档）。

| nx | D7 = L1(WENO7−ref) [%] | D5 = L1(WENO5−ref) [%] | D7/D5 |
|---|---|---|---|
| 800 | 5.0196 | 5.0681 | 0.990 |

**S5 判定：PASS**

### 被定级解的读数（同一套窗口口径）

| nx | HLLC vs WENO7 同网格 [%] | HLLC vs WENO5 同网格 [%] | HLLC vs WENO5@12800 [%] |
|---|---|---|---|

## ⑤ 注意点（局限）

* WENO7 与 WENO5 共享 SSP-RK3 与 Rusanov 通量：它探的是**重构阶**，不是代码独立性。跨**代码**独立性仍由 TensorFVM（有限体积 MUSCL/HLLC）vs 有限差分 WENO5 提供。
* 限制器是通量级 `θ` 缩放（Zhang–Shu 型），比状态混合耗散更大但守恒；它保证**可容许**，不保证**无振荡**（S3 的 FAIL 就是振荡/抹平的残留）。
* 峰值、低平台这类泛函收敛阶远低于 L1：不要用峰值一致去替 L1 判据。
* 参考侧误差棒仍非 0：任何 `<1 %` 的措辞必须区分「点估计」与「可证明上界」。

## ⑥ 误差判定（逐项显式回答是否 <3 % / <1 %）

| 关键量 | 实测 | <15 % | <3 % | <1 % |
|---|---|---|---|---|
| HLLC@3200 vs WENO5@12800（另文 1.776 %） | 见表 | ✅ | ✅ | ❌ |
| D7@nx=800（对族极限） | 5.0196 % | ✅ | ❌ | ❌ |

**是否 <3 %？** 上表凡 ✅ 项成立；S3 的峰值同网格差 5.30 % 带 «<3 %» 语境时须注明是**峰值**泛函而非 L1。
**是否 <1 %？** certified-<1 % 结论：**未推进**；理由见 S6（冻结判据）。

## ⑦ 复现

```bash
# 探针（修好的）：本地 400/800，集群 1600/3200/6400
PYTHONPATH=src python docs/woodward-colella-double-blast/xcode/wc_weno7_ladder.py --rungs 400,800
PYTHONPATH=src python docs/woodward-colella-double-blast/xcode/wc_weno7_ladder.py --rungs 1600,3200,6400 --overwrite
# 熵波阶（判据 S2）
PYTHONPATH=src python docs/woodward-colella-double-blast/xcode/wc_weno7.py --entropy-order \
    --json docs/XVAL-FVM-M18737-ROUND2/entropy_order_r8.json
# 对账 + 图 + 渲染
PYTHONPATH=src python docs/XVAL-FVM-M18737-ROUND2/weno7_repair_r8.py --figures --render
# 回归门
PYTHONPATH=src python -m pytest tests/test_compressible_wc_weno7_repair.py -q
```
