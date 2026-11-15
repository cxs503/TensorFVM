# ROUND16 — CASE-FVM-PERIODIC-HILLS：ω 汇项线性化 A/B 的验收级复核

**锚点**：`CASE-FVM-PERIODIC-HILLS`（算例池·周期山丘流分离与再附长度，Re=2800）
**基准**：Breuer et al. 2009 (Comput. Fluids 38:433, doi:10.1016/j.compfluid.2008.05.002) / ERCOFTAC UFR 3-30
**容差**：10%（`x_r/H` 与 `y/H=0.5` 平均速度剖面）；参考值 `x_r/H = 5.41`、DNS 剖面取自仓内
`tests/data/periodic_hill/ercoftac_ufr330/`（`periodic_hill_reference`，单一真源）
**本单**：m21316 · 集群作业 cj3331（patankar）/ cj3332（newton2）
**结论一句话**：round 16 认定「ω 汇项用了 Newton 系数 `2 beta ω`，导致 ω 偏低 √2、ν_t 偏高"2x"、
再附泡过短」并已把系数改为模型一致的 Patankar 值——本单用 96×48 / 20000 步（统计窗 t=160–400，
约 66 个流动贯越）的**验收级 A/B** 复核：修正确实把 ν_t 降了 4 倍、`y/H=0.5` 剖面合格站从 1/8 提到 3/8，
但把 `x_r/H` 从 3.767（-30.4%）推到**整个统计窗内无再附交点（UNDEFINED）**——所以「汇项线性化是本算例
过短再附泡的根因」这一假设**被否定**，两条配置都不通过 10% 门，验收仍是 **honest FAIL**。

---

## ① 案例介绍

周期山丘（periodic hills）是分离-再附湍流的标准算例：流向周期 L=9H，下壁为六段三次曲线山丘（最大高度
H 于 x/H=0），上壁为平面 y=3.036H，展向统计二维。以体平均速度 `U_b`（ERCOFTAC 定义：丘顶以上
`1/2.035 ∫ u dy`）驱动，Re = U_b H/ν = 2800。

| 参数 | 值 |
|---|---|
| 几何 | `hill_profile`（ERCOFTAC UFR 3-30 六段三次曲线，`HILL_KAPPA=27.96`），L=9 m，H=1 m |
| 上壁 | y = 3.036 m |
| Re | 2800（`U_b=1`，ν=3.5714e-4） |
| 网格 | 96×48（body-fitted，地形跟随）；ons. trend 探针 32×16 |
| 闭合 | k-ω SST（`SSTKOmega`，a1=0.31，自由流强度 5%） |
| 时间推进 | dt=0.02，投影法 + 体动力控制器；统计窗 t=160–400（20000 步，burn-in 8000） |
| 参考 | `x_r/H = 5.41`（KB UFR 3-30 Evaluation，作者自述值）；`y/H=0.5` 十个流向站 DNS 文件 |

**要回答的问题**：rounds 10–15 的 ω 汇项缺陷（隐式对角用 Newton 系数 `2βω`，右端未补 `+βω_old²`，
不动点满足 `2βω² = γP_k/ν_t`，故 ω 偏低 √2、ν_t 偏高约 2 倍）被 round 16 修成 Patankar 系数 `βω` 后，
**验收量是否随之进入 10% 带**？round 16 只在 32×16/2000 步的趋势探针上回答过（且 `ROUND16.md` 从未产出、
96×48 验收运行从未提交），本单补上。

## ② 怎么做的

1. `scripts/round16_acceptance.py`：单配置单进程驱动。**变体是测出来的、不是声明的**——
   从 `SSTKOmega.source_terms()` 实测 `omega_destruction_linear / (β ω)`，写入 artefact
   `sink_coefficient_ratio`（1.0=Patankar，2.0=旧 Newton）。此设计直接针对 rounds 10–14 的
   「knob 记录了却没接线」失效模式。
2. `--variant newton2` 通过方法覆盖复现 rounds 10–15 的缺陷系数（HEAD 只有 Patankar 一条路径，
   不留未接线的开关）；覆盖后实测比值 2.0。
3. 两次运行网格、步数、dt、动量混合（0.75）、输运混合（0.0）完全相同，唯一差别是 ω 汇项系数。
4. 裁决走 round-15 的 `score_hill`（5.41 + DNS 剖面表），**不使用** payload 自带的 legacy 5.34 门
   （rounds 4–9 的手抄表已被 round 10 判定最多偏大 3.5 倍）。
5. 集群递交：cj3331（patankar，2638 s）、cj3332（newton2），单核，同节点 n330。

## ③ 计算结果（全部由 artefact 程序化注入，零手抄）

| 配置 | 实测 S_P/(βω) | x_r/H | 误差 vs 5.41 | 再附检出率 | 体动力 G | 时间平均 max ν_t/ν | y/H=0.5 合格站 | 统计窗稳定性 |
|---|---|---|---|---|---|---|---|---|
| **newton2**（rounds 10–15） | 2.00 | 3.7667 | **30.4 %** | 100 % | 0.016893 | 202.54 | **1/8** | spread 2.1e-6（稳定） |
| **patankar**（HEAD 修正） | 1.00 | **UNDEFINED** | — | **0 %** | 0.008746 | **49.42** | **3/8** | spread = ∞（无交点） |
| round-15 基线（同一 artefact 复核） | 2.00 | 3.7667069408020226 | 30.4 % | 100 % | 0.016893 | 202.54 | 1/8 | 2.1e-6 |

`newton2` 与 round-15 存档 **逐位相同**（3.7667069408020226），证明覆盖路径确实复现了旧缺陷，
A/B 的唯一变量成立。

`y/H=0.5` 逐站（DNS 值 → patankar / newton2）：

| x/H | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|---|---|
| DNS U/U_b | -0.046 | -0.091 | 0.004 | 0.120 | 0.218 | 0.288 | 0.314 | 0.224 |
| patankar | -0.060 ✓ | -0.044 ✓ | -0.005 ✓ | 0.044 ✗ | 0.098 ✗ | 0.143 ✗ | 0.168 ✗ | 0.114 ✗ |
| newton2 | -0.125 ✓ | 0.043 ✗ | 0.180 ✗ | 0.306 ✗ | 0.395 ✗ | 0.444 ✗ | 0.432 ✗ | 0.330 ✗ |

图：`fig_round16_acceptance.png`（左：`y/H=0.5` 平均剖面 vs DNS；右：`x_r/H` 对 5.41±10% 带）。

## ④ 注意点 / 局限

* **patankar 的 `x_r/H` 是 UNDEFINED 而不是大数**：统计窗内 0 % 快照在 `[1.9, 7.1]` 出现再附交点，
  即时间平均壁面剪切的回流区延伸到 x/H>7.1 之外。**没有**把它写成「≈7.1」或任何数字——压力门会 FAIL，
  但错误原因是缺量（WITHHELD），不是实测超限（REJECTED），二者在 `gate_summary` 里分开报。
* ν_t 降 4 倍（202→49）、体动力降一半（0.0169→0.0087）说明修正**确实按物理方向起了作用**；
  `y/H=0.5` 剖面从系统性过冲（+0.13…+0.20）变成系统性亏欠（-0.08…-0.15），合格站 1/8→3/8。
  也就是说：缺陷真实，但它不是「再附泡过短」的根因；2-D URANS+SST 在这个 Re 下的误差是
  **模型维度问题**（DNS 有三维湍流，2-D 时间平均没有），不是线性化问题。
* 32×16 趋势探针（round 16 原文）给 patankar 4.57 / newton2 4.62——两者差 1 %，**当时就已经不支持
  「根因」结论**，只是没人把它与「96×48 上差 30 %」并置。粗网格上数值扩散掩盖了差异。
* 本单只做了 96×48 一档网格（时间预算），**没有**网格收敛阶梯；因此 patankar 的 UNDEFINED 是否随加密
  消失，本轮不作断言。

## ⑤ 误差判定（逐项，显式回答「是否 <3%」）

| 量 | 计算 | 基准 | 误差 | <3 % ? |
|---|---|---|---|---|
| `x_r/H`（newton2 / rounds 10–15） | 3.7667 | 5.41 | 30.4 % | **否** |
| `x_r/H`（patankar / HEAD） | 未定义 | 5.41 | —（WITHHELD） | **否（无值）** |
| `y/H=0.5` 剖面，8 站最大偏差（patankar） | 见上表 | DNS 文件 | 最差站 0.146（x/H=4，|ref|=0.12 ⇒ 相对 122 %） | **否** |
| `y/H=0.5` 剖面，8 站最大偏差（newton2） | 见上表 | DNS 文件 | 最差站 0.186（x/H=3） | **否** |
| S_P/(βω) 实测 vs 模型定义 | 1.000 / 2.000 | 解析 | <1e-12 | **是**（离散恒等式，非物理量） |

**验收判定：FAIL（honest）**。`x_r/H` 与剖面两条主判据均未进 10 % 带，距 3 % 更远；唯一 <3 % 的是
线性化系数的实现恒等式，不代表任何物理精度。

## ⑥ 顺带修掉的缺陷（本单阻塞项）

`periodic_hill_round9.averaging_window` 在「没有任何窗口平稳」时把 `attempts` 列表挂到
`attempts[0]` 上——而 `attempts[0]` **就是**返回对象本身，于是
`worst["attempts"][0]["attempts"] -> worst` 构成引用环，`write_json`/`_clean` 递归爆炸
（`RecursionError`）。后果：**恰好在 honest-FAIL（非平稳）时写不出 artefact**，而 PASS 时写得出来——
门禁的失败表现为「没有文件」，与「没有运行」无法区分。已改为浅拷贝（保留每轮丢弃块的诊断信息，
去掉自引用），`tests/test_periodic_hill_round16_acceptance.py` 用三条断言钉住：可 `json.dumps`、
`attempts[0] is not out`、`blocks_dropped == [0,1,2]`。

## ⑦ 复现

```bash
PYTHONPATH=src python3 scripts/round16_acceptance.py --variant patankar \
    --nx 96 --ny 48 --steps 20000 --tag r16_patankar_96x48
PYTHONPATH=src python3 scripts/round16_acceptance.py --variant newton2 \
    --nx 96 --ny 48 --steps 20000 --tag r16_newton2_96x48
PYTHONPATH=src python3 scripts/round16_acceptance_analysis.py     # 表 + 图
PYTHONPATH=src pytest tests/test_periodic_hill_round16_acceptance.py -q   # 6 passed
```

artefact：`r16_patankar_96x48.json`、`r16_newton2_96x48.json`、`round16_acceptance.json`、
`fig_round16_acceptance.png`。集群回执：cj3331（2638 s）、cj3332。

## ⑧ 给下次自找活的线索

1. **网格收敛阶梯**（本单欠账）：96×48 → 144×72 → 192×96，只跑 patankar，判定「UNDEFINED 是否随加密消失」；
   本单已证明 96×48 单核 2638 s，144×72 约 5900 s，可在单作业预算内跑。
2. **再附判据的适用范围**：`zero_crossings_periodic` 的搜索窗 `[1.9, 7.1]` 是 round 5 为「过短泡」设的；
   patankar 情形回流区越界后该窗给出 UNDEFINED 而非「越界」的诊断码。建议加 `out_of_window` 状态位，
   把「没有交点」与「交点在窗外」分开——这是下一轮报告的诚实性前提。
3. 2-D URANS 的维度误差无法靠线性化或网格修补；若总账要求 10 %，正路是 LES/3-D 或换低 Re 的姊妹算例。