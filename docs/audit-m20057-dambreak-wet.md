# Board Audit m20153 — m20057 CASE-FVM-DAMBREAK-WET 交差核验

**锚点**: `CASE-FVM-DAMBREAK-WET` (TensorFVM two-phase VOF benchmark)
**被审对象**: mission m20057（已 succeeded，auto-approve dev_acceptance）
**本单**: m20153（交差核验单，非重做）
**基准裁判**: `tensor_benchmark_cases`（唯一裁判，报告好看不算数）

---

## 1. 取证（三件套逐项核验）

### 1.1 run_artifacts — ✅ 存在
- `artifacts/dambreak_mm_100x30/results.json`（102KB，含 surge_front/column_height/metrics）
- `artifacts/dambreak_mm_100x30/surge_front_comparison.png`（1500×900 RGBA）
- `artifacts/dambreak_mm_100x30/surge_front_error.png`（1500×900 RGBA）
- `artifacts/dambreak_mm_100x30/column_height_comparison.png`（1500×900 RGBA）

### 1.2 benchmark ref vs got — ✅ 存在但 **FAIL**
| 指标 | got | 容差(报告) | 容差(代码) | 判定 |
|---|---|---|---|---|
| surge_front_mean_error | 29.68% | <8% | 15% | ❌ FAIL |
| surge_front_max_error | 419.38% | <25% | 25% | ❌ FAIL |
| column_height_mean_error | 871.96% | <15% | 15% | ❌ FAIL |
| simulation_completed | 0.0 | — | 0.5 | ✅ PASS |

### 1.3 repro_cmd — ✅ 可执行
`python scripts/run_dambreak_mm_benchmark.py --nx 100 --ny 30 --max-time 0.6 --output-dir ...`
- 实测运行 70.7s，359 步
- 复现结果：surge_front_mean_error=29.68%（与交付一致）

### 1.4 基准裁判（tensor_benchmark_cases）
- `SELECT ... WHERE anchor_slug='CASE-FVM-DAMBREAK-WET'` → **0 行**（未登记）
- 基准裁判无法裁判任何东西——锚点被引用但无登记行

### 1.5 报告六段
- 全部六段齐全（案例介绍/怎么做的/计算结果/结果图/注意点/误差判定）
- 报告为 **honest-FAIL**（如实声明未通过）

### 1.6 平台登记
- `tensor_reports`：mission 20057 **0 行**（未发布到平台）
- `tensorbench_results`/`tensorbench_evaluations`：**0 行**
- `tensor_maturity_verdicts`：本单插入前 **0 行**

### 1.7 Git 状态
- 全部交付物（artifacts/、docs/、scripts/）**UNTRACKED**（未提交）
- 分支 `agent/m20057` 无自有 commit（HEAD=0a075ce8，main 合并点）

---

## 2. 判级（L0-L5）

**级别 = 功能实质**（附百页报告不改变级别）

- **L0** 文档成果：否（有真代码）
- **L1** 数学/单元验证：否（非单元级）
- **L2** 独立功能实现：是（真代码：输入→计算→输出，run_dam_break 实跑）
- **L3** 系统集成：是（接入 TensorFVM 主求解器 src/tensorfvm/verification/dam_break.py）
- **L4** 基准验证：**否**（基准 FAIL——got 29.68% 不在容差内；且 tensor_benchmark_cases 无登记行）
- **L5** 科研工程成果：否（标准基准复现，无新方法）

**判定：L3**（系统集成）
- 真代码接入主求解器，产生真实输出
- 但基准验证未通过（honest-FAIL），且基准未在平台登记

---

## 3. 落表

```sql
INSERT INTO tensor_maturity_verdicts (repo, scope, mission_id, level, evidence, verdict_by)
VALUES ('TensorFVM', 'm20057 CASE-FVM-DAMBREAK-WET', 20153, 3, '<JSON>', 'board-audit');
```
- 已执行：id=845，level=3，gate_state=pending

### evidence 键（真实指针）
- `impl_files`: `/root/tp-agent/work/m20057/scripts/run_dambreak_mm_benchmark.py`, `src/tensorfvm/verification/dam_break.py`
- `entry`: `scripts/run_dambreak_mm_benchmark.py`
- `run_artifacts`: `artifacts/dambreak_mm_100x30/results.json` + 3 PNG
- `tests`: NONE（无测试引用该锚点）
- `integration_callers`: `src/tensorfvm/verification/dam_break.py:run_dam_break`
- `benchmark`: got=29.68%/419.38%/871.96%，pass=false，authoritative_judge=0 行
- `repro_cmd`: 实测可跑（70.7s，359 步，复现 29.68%）
- `external_anchor`: Martin & Moyce (1952) A 244:312-324, Fig. 3

---

## 4. maturity_gate 裁定

- 基准 FAIL（got 不在容差内）→ **reopened（返修）**
- 需返修：3D 模拟 + PLIC 界面重构 + 更细网格 + 基准登记

---

## 5. 结论

- 三件套：run_artifacts ✅ / benchmark ref vs got ✅但FAIL / repro_cmd ✅
- 级别：**L3**（真代码接入主求解器，但基准未通过）
- 裁定：**reopened**（返修）
- 关键差异：mission 标记 `succeeded` 但实际为 honest-FAIL；基准未登记；交付物未提交
