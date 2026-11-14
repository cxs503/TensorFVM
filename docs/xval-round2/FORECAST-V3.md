# FORECAST-V3 — 因果停止迭代预测 v3（锚点 XVAL-FVM-M18737-ROUND2）

> 本文件由 `scripts/gen_forecast_v3_report.py` 生成的报告正文落盘，数字全部来自
> `docs/xval-round2/causal_forecast_v3.json`（程序化注入，零手抄）。
> 图：`fig_forecast_v3_leads.png`、`fig_forecast_v3_cliff.png`。
> 表格级证据（逐算例/逐臂/覆盖率/崩塌证据）：`docs/xval-round2/causal_forecast_v3.md`。

# TensorFVM — causal stopping-iteration forecast v3 (XVAL-FVM-M18737-ROUND2)

## ① 案例介绍
迭代线性求解（5 点 Poisson、一阶迎风对流扩散，GMRES(32)/CG/Richardson，网格 32²/64²，
容差 1e-8/1e-12/relTol 1e-2）。工程意义：CFD 迭代求解常占 50–90% 机时，准确的「还剩多少步」
可用于预算调度与异常报警。验收标准（总账）：停止迭代预测误差 <15%。
数据：仓内已入库的 8 条残差历史（`docs/convergence-diagnosis-v2/histories`，最长 2001 点），
本轮不重跑求解器、不起集群。

## ② 怎么做的
v1 假设常数几何收敛率；round-2 对 `log rate` 做线性外推并 70/30 混合。上一轮审计
（`docs/xval-round2/CAUSAL-AUDIT.md`）证明 round-2 的「0.00%」是事后恒等式，因果口径下与 v1 无差别。
本单新增 **v3**：对 `log residual`（而非 `log rate`）拟合**到迭代步的二次模型**，曲率 c 即收敛加速度；
c<0 且斜率 b<0 且曲率 t 检验 t_c ≤ −2 才启用（否则退回常数率候选）；外推**锚定在观测残差**上；
最后一步收缩因子须落在 (0,1) 才发布；并给出参数不确定度区间（68%/90%）。
比较协议是**严格因果**的：每个算例按前缀逐点回放，误差在**同一前缀** k=round(stop·(1−L)) 上取，
未发布即不计分（不给「晚发布」加分）。四臂：v1 / v2（原样）/ v2g（原 hold-out 闸）/ v3。

## ③ 计算结果（程序化生成，零手抄）
各臂在固定提前量下的达标数（≤15%）：

| 提前量 | v1 | v2 | v2g | v3 |
|---|---|---|---|---|
| 50% | 2/5（最差 38.8%） | 2/5（38.8%） | 2/5（38.8%） | 2/5（34.9%） |
| 25% | 4/5（70.8%） | 4/5 | 4/5 | **2/6（70.8%）** |
| 10% | 5/6（20.8%） | 5/6 | 5/6 | **6/6（6.2%）** |
| 首次发布 | 1/6（最差 307.0%） | 1/6 | 1/6 | 0/6（最差 103.6%） |

逐算例（提前量 10% 时，v1 vs v3；末列给出 v3 在 25% 提前量时的误差以示并非全面改善）：

| 算例 | 真实停止迭代 | v1@10% | v3@10% | v3@25% |
|---|---|---|---|---|
| cd-pe10-gmres | 584 | 3.3% | 3.3% | 18.8% |
| cd-pe1000-gmres | 498 | 4.4% | 4.4% | 31.1% |
| poisson-n32-reltol1e-2 | 48 | 20.8% | 6.2% | 70.8% |
| poisson-n32-tol1e-8 | 188 | 10.1% | 1.6% | 9.6% |
| poisson-n64-tol1e-12 | 849 | 2.7% | 2.7% | 2.0% |
| poisson-n64-tol1e-8 | 541 | 0.7% | 0.7% | 15.3% |

区间覆盖率（名义 68%/90%）：11.8% / 19.4%
（2441 / 2381 个已发布前缀）——参数不确定度
远小于模型误差，这是本单的负面结论之一。

## ④ 结果图
图 1：固定提前量下的误差（同一前缀、严格计分）。图 2：`poisson-n32-reltol1e-2` 的残差轨迹——
前 36 步比值趋近 1，最后 12 步突然崩塌（GMRES 终末 breakdown），k=36 时任何光滑 `log residual`
模型都无法预见；k=43（再晚 7 步）时曲率显著，误差降到 6.2%。

## ⑤ 注意点（坑/局限/适用范围）
1. **模型族的能力边界已量化**：6 个衰变算例中 2 个（`poisson-n32-reltol1e-2`、`cd-pe1000-gmres`）
   在提前量 25% 处不可预测——残差前缀呈现「比值趋 1 后突然崩塌」，而历史文件只有
   `iteration/residual` 两列（无 Ritz 值/算子信息），信息不足。v3 不宣称修复它们。
2. **v3 在长提前量上更差**（25%：2/6 vs v1 的 4/5）：二次外推在中段会把噪声曲率当加速。
   试过的两种闸门（自预测视界比、跨窗一致性、持久性）都不稳定，见
   `docs/xval-round2/causal_forecast_v3.json` 与 §⑥，因此**不发布**隐藏调参。
3. 区间只覆盖参数不确定度，不覆盖模型误差，故覆盖率仅
   11.8%/19.4%；诚实标注为
   「参数区间」而非预测区间，预测区间需要模型误差项（残差轨迹本身给不出）。
4. 测试抓到过一个真实缺陷：发散但凹（增长减速）的序列会得到虚构的收敛预测——已用「斜率必须为负」
   并入闸门（`tests/test_convergence_forecast_v3.py::test_non_decaying_series_publishes_nothing`）。
5. 数据为已入库 JSON，未覆盖「诊断反馈到求解控制」的耦合行为；样本仅 6 个衰变算例（2D 线性算子）。

## ⑥ 误差判定
| 关键量 | 数值 | <15%？ | <3%？ |
|---|---|---|---|
| v3 提前量 10% 的停止迭代预测 | 6/6 通过，最差 6.2% | **是** | 否（最差 6.2% > 3%） |
| v1 提前量 10% | 5/6，最差 20.8% | 否 | 否 |
| v3 提前量 25% | 2/6，最差 70.8% | **否** | 否 |
| v3 首次发布 | 0/6，最差 103.6% | **否** | 否 |
| 区间 68%/90% 覆盖率 | 11.8%/19.4% | 名义未达成（诚实记录） | 否 |

**honest-FAIL**：锚点的「<15%」只在**提前量 ≤10%** 处成立（6/6，最差 6.2%）；全提前量口径**未达成**，
且 v3 在 25% 提前量上比 v1 差。总账条目不应被标为 done，建议改为
`GATE-FVM-CONV-FORECAST-V3`：验收「提前量 10% 时 6/6 ≤15%，且区间覆盖率 ≥ 名义值 80%」——
前半已达成，后半为欠账。

**复现命令**
```bash
PYTHONPATH=src python docs/xval-round2/causal_forecast_v3.py
PYTHONPATH=src python scripts/gen_forecast_v3_report.py
PYTHONPATH=src python -m pytest tests/test_convergence_forecast_v3.py \
    tests/test_convergence_causal_audit.py tests/test_convergence_xval_round2.py \
    tests/test_convergence_diagnosis.py -q     # 37 passed
```
