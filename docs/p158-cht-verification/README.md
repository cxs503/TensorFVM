# P158W3-N02 · TensorFVM 流-固共轭传热（CHT）验证面

| 产物 | 路径 |
|---|---|
| 论文级报告（数字全部程序化注入） | [`REPORT.md`](REPORT.md) |
| 可复用知识蒸馏（含回归守门） | [`LESSONS.md`](LESSONS.md) |
| 原始工件 JSON | [`artifacts/`](artifacts/)（`flatplate.json`、`cyl_*.json`、`summary.json`） |
| 图 | [`figures/`](figures/) |

## 复现

```bash
# 平板共轭锚（A1/A2/A4 + GCI/A5），~2 分钟
python scripts/p158_cht_validation.py --case flatplate

# 圆柱强制对流（A3），单算例；medium 网格 Re=40 约 30 分钟量级
python scripts/p158_cht_validation.py --case cylinder --re 40 --pr 0.7 --grid coarse

# 重新渲染报告与图（数字零手抄）
python scripts/p158_cht_report.py

# 验收测试（慢用例需 CHT_SLOW=1）
python -m pytest tests/test_cht_conjugate_anchors.py -q
```

## 判据状态（本轮，2026-11-11）

| 判据 | 要求 | 实测 | 结论 |
|---|---|---|---|
| A1 平板 Nu_x vs 均匀热流 Pohlhausen（5 站 Re_x∈[1e4,1e6]） | ≤5% | 0.047% | 通过 |
| A2 固体域温度 vs 解析共轭解 | ≤2% | 0.139% | 通过 |
| A3 圆柱平均 Nu vs 经验关联式（Re=40/100/200 × Pr=0.7/7） | ≤5% | 9.2%–13.0%（coarse，且 Cd 未达标） | **未通过（honest FAIL）** |
| A4 界面热流守恒（局部/全局） | ≤1e-6 | 1.4e-13 | 通过 |
| A5 GCI 观测收敛阶（自相似网格族） | ≥1.8 | 2.55 | 通过 |

A3 的根因诊断（方位离散数值扩散为主误差源；显式壁涡量把 dt 锁死在 `dt·ν/Δr² ≲ 0.3`）
与改进路径见 `REPORT.md` §5 与 `LESSONS.md` L3/L4。新写的谱方法圆柱求解器
(`src/tensorfvm/cht/cylinder_fourier.py`) 已通过纯导热极限（Nu=2/ln(R/r)，1e-3 级）
与 Poisson/速度场精确性检验，但其定常解尚未通过阻力基准，故**不进入**验收。

## 新增/修改的代码

- `src/tensorfvm/cht/similarity.py`：`solve_uniform_flux` / `uniform_flux_nusselt`（均匀热流相似解，自解 ODE）。
- `src/tensorfvm/cht/energy.py`：新增 `neumann` 边界类型（含能量账本项）。
- `src/tensorfvm/cht/flatplate.py`：底部定热流共轭模式 + 解析固体温度/界面全局守恒审计。
- `src/tensorfvm/cht/cylinder_fourier.py`（新）：θ 谱方法 + r 有限体积的 ψ–ω 与能量求解器。
- `src/tensorfvm/cht/polar.py`：二阶壁涡量、限幅二阶方位对流、外边界径向通量修正（缺陷修复）。