# P447W2-N01 — FVM 两相 VOF 三算例验证面（工件目录）

本目录是战役节点 `p447w2-n01-fvm-vof-two-phase` 的完整审计链。**唯一可引用的数字来源是
`vof_threecase.json`**（`render_report.py` 从它渲染 `../report.md`，零手抄）。

## 文件

| 路径 | 内容 | 谁写的 |
|---|---|---|
| `vof_threecase.json` | 验收记录：三算例全部记录 + 五条判据的机器裁决 + 环境指纹 | `render_report.py`（由各 shard JSON 合并） |
| `static.<key>.json` | 静止气泡单配置记录（含时程、Laplace 压差、饱和/发散标记） | `leg_runner.py --leg static` |
| `zalesak.<key>.json` | 单配置旋转记录（缺口空缺率、周长、L1、质量漂移） | `leg_runner.py --leg zalesak` |
| `dam.<ny>x<nx>.json` | 单网格溃坝记录（波前时程、判据时刻误差、质量账本） | `leg_runner.py --leg dam` |
| `fields/*.npz` | 场量快照（α、|u|、缺口掩膜、终态自由面），图直接读它 | 同上 |
| `figures/*.png` | 报告图（场量云图 + 与基准的比较曲线） | `render_report.py` |
| `logs/*.log` | 每次运行的原始 stdout（含发散/异常堆栈），审计用 | shell 重定向 |

## 复现

```bash
PYTHONPATH=src python docs/P447W2-N01-vof-three-case/leg_runner.py --leg dam --index 0 --count 3
PYTHONPATH=src python docs/P447W2-N01-vof-three-case/leg_runner.py --leg static --index 0 --count 4
PYTHONPATH=src python docs/P447W2-N01-vof-three-case/leg_runner.py --leg zalesak --index 0 --count 4
PYTHONPATH=src python docs/P447W2-N01-vof-three-case/render_report.py
PYTHONPATH=src python -m pytest tests/test_vof_threecase.py -q
```

`--index i --count n` 分片互不依赖：任一 shard 被 kill 只损失一个配置，`render_report.py`
用现存工件重建验收记录（缺失项在报告里显示为 `n/a` 而非被静默跳过）。

## 结论摘要（详见 `../report.md` 与 `vof_threecase.json` 的 `criteria`）

| 判据 | 结论 |
|---|---|
| C1 溃坝波前 vs Martin & Moyce（≤10 %） | **FAIL**（50x100 网格 t=0.1 s 误差 −40.2 %） |
| C2 静止气泡归一化虚假流速（≤0.01） | **FAIL**（本仓库该路径在 0.4~3.6 s 内发散，无法给出稳态值） |
| C3 Zalesak 缺口面积误差（360°≤5 %，1000°≤15 %） | **PASS**（cicsam N=128：1.79 % / 7.78 %） |
| C4 三算例质量守恒（≤0.1 %） | **PASS**（最差数量级 1e-13） |
| C5 三网格波前误差单调收敛 | **FAIL**（误差随加密先降后升，见报告 §3.3） |

FAIL 项**不覆盖、不粉饰**：报告第 5 节列出各自的可复现证据、候选机理与未完成部分。