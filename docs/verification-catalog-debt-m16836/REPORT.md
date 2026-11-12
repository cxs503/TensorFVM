# 验证欠账清偿：§10.2 计划项审计盲区补闸 + 两项"计划中"能力的状态定案（m16836）

**Kind**: verification debt（把已合并能力对文献/解析基准复核）+ 缺陷修复（验证索引门 §10.2 盲区）
**Repo**: TensorFVM  **Branch**: agent/m16836  **Date**: 2026-11-12

## 1. 案例介绍（问题背景）

`docs/verification-catalog.md` 是本仓对外的"什么已验证 / 什么未验证"索引。
它的问题不是文风，而是**索引本身没有第三者复核**：条目里引用的测试改名、
模块搬走、能力交付后没回写，索引就静默地错，而索引恰恰被别人当证据引用
（"能力 X 未验证"）。

仓库对此已有机器门 `src/tensorfvm/verification/catalog_gate.py`（m7555 建）：
死指针、✅PASS 无活测试、§10.1 Current Gaps 无状态标注 → FAIL。

**本单扫描发现的真实缺陷（两条）**：

| # | 缺陷 | 证据 |
|---|---|---|
| A | 门只审 §10.1，**完全跳过 §10.2 Planned Additions** | `catalog_gate.parse_catalog` 里 `re.split(r"\n###\s+10\.2", block)[0]`；实测 §10.2 有 3 条无状态项而门判 PASS |
| B | 已合并的能力仍挂在"计划中"：**MRF/旋转机械**（`rotating_frame` 能力早已交付，18/18 判据 PASS）、**自然对流 de Vahl Davis**、**多相上升气泡** | 门未报；`docs/verification-rotating-frame/summary.json` 全 PASS 却无 `closed-by` 标注 |

另有一条**先于本单存在的红**：`tests/test_verification_catalog_gate.py::test_gap_status_annotation_parsing`
断言 `3D Lid-Driven Cavity == "open"`，而目录早已改成 `closed-by`（闭包滞后一天的历史动作）→ 该测试长期 FAIL，说明"索引-测试"一致性本身没有守住。

## 2. 怎么做的（方法/流程）

1. **复核（先证后判）**：不采信既有工件，重跑 `python -m tensorfvm.verification.rotating_frame`
   （12.6 s）→ 18/18 判据 PASS 逐条复现（C5 旋转槽道 1.270e-05 ≤ 3e-02；
   C8b MRF 坐标系不变性 4.146e-13；三条控制判据 C6/C7c/C10b = 1.0 证明锚点能判死错实现）。
   自然对流侧重跑 `tests/test_natural_convection_highra_gate.py` 等 35 测绿，
   并直接读 `docs/natural-convection-gci/summary.json` 的 `verdict` 字段定案。
2. **补闸**：`catalog_gate` 扩展为 §10.1 + §10.2 双节审计，并新增第三态
   `[[gap-status: partially-closed-by <paths>]]`（部分交付：部分判据 PASS、部分 honest FAIL/WITHHELD）。
   引入第三态的理由写在模块 docstring 里：二态会逼出两种失真——「假 open」藏掉已交付工作，
   「假 closed-by」夸大结论；这正是门要防的失效模式，所以门必须能说"部分"。
   新状态与 `closed-by` 一样**逐条校验引用路径存在**，最终写进 `to_dict()`（
   `planned_without_status` / `dangling_planned_closures` / `counts.planned_*`）与 CLI 报表。
3. **回写索引（状态定案，不粉饰）**：
   - Rotating Machinery → `closed-by`（18/18，文献锚 Johnston 1972 旋转槽道 / Taylor 力矩 / 坐标系不变性）；
   - Heat Transfer（自然对流）→ `partially-closed-by`（Ra=1e5 3% 内、Ra=1e6 honest FAIL）；
   - Multiphase（上升气泡）→ `partially-closed-by`（Hysing case 1 已交付，形状门受 `cpd=8` 分辨率限制）。
4. **零手抄**：新增 `scripts/generate_catalog_verification_entries.py`，§17/§18 两节的
   **每一个数字都由工件 JSON 程序化渲染**并以 `<!-- BEGIN/END generated -->` 标记拼进目录；
   `--check` 模式 + `tests/test_catalog_verification_entries.py` 构成漂移门：工件一变而目录没重渲染→测试红。

## 3. 计算结果（数字表，全部来自工件 JSON）

### 3.1 旋转坐标系 / MRF（重跑复现）

| 判据 | 值 | 门 | 判定 |
|---|---:|---:|:--:|
| C5 旋转槽道 vs 闭式解 | 1.270e-05 | 3.0e-02 | PASS |
| C5b 旋转槽道观测阶 | 1.974 | ≥1.8 | PASS |
| C6 锚点拒绝"无科氏项"求解器 | 1.000 | ≥0.5 | PASS |
| C7c 控制：错误科氏符号被拒 | 1.000 | ≥0.5 | PASS |
| C8b MRF 坐标系不变性 | 4.146e-13 | 1.0e-12 | PASS |
| C9 Couette 力矩 vs Taylor | 3.052e-06 | 3.0e-02 | PASS |
| C10a 坐标系径向平衡 | 1.981e-16 | 1.0e-14 | PASS |
| 合计 | **18/18 PASS** | — | PASS |

### 3.2 自然对流 vs de Vahl Davis 1983（半格 Nu 估计，门 3%）

| Ra | N=16 | N=32 | N=64 | 观测阶 p | Richardson 极限 | 判定 |
|---|---:|---:|---:|---:|---:|:--:|
| 1e5（ref 4.519） | +13.19 % | +3.39 % | **+0.89 %** | 1.97 | 4.5205（+0.03 %） | PASS |
| 1e6（ref 8.800） | +18.24 % | +11.11 % | **+3.07 %** | −0.18 | withheld | **honest FAIL** |

### 3.3 门本身（修复前后对照）

| 检查 | 修复前 | 修复后 |
|---|---:|---:|
| §10.2 计划项审计数 | 0（跳过） | 5 |
| §10.2 无状态项 | 3（门仍 PASS，静默） | 0 |
| §10.2 closed-by / partial | 2 / 0 | 3 / 2 |
| `3D Lid-Driven Cavity` 测试断言 | 红（断言 open，实为 closed-by） | 绿 |
| 门判定（本仓目录） | PASS（假绿） | PASS（§10.1 5/5 + §10.2 5/5 全标注） |

## 4. 结果图

`fig_catalog_debt.png`（`make_figure.py` 程序化生成，数字直接读工件 JSON）：
左图=旋转坐标系 18 条判据相对各自门的余量（对数十年，虚线右侧=门内，全绿=PASS）；
右图=方腔 Nu 相对误差随网格（log2 横轴）与 3% 验收带——Ra=1e5 在最细网格进入带内，
Ra=1e6 停在 +3.07% 带外，与 honest FAIL 表述一致。

## 5. 注意点（踩过的坑/局限）

1. **门只覆盖"有标注的行"是不够的**：真正的盲区是**没被扫到的整节**。旧门用
   `re.split(r"\n###\s+10\.2", block)[0]` 把 §10.2 切掉，于是三条已交付/已部分交付的
   计划项长期无人审计，而门一直 PASS——"测试绿 ≠ 索引对"。
2. **二态标注会诱导失真**：自然对流是"部分交付"的典型（Ra=1e5 过、Ra=1e6 不过）。
   逼成 open 会把已交付的网格收敛研究与 3% 闭合藏掉；逼成 closed-by 会把 +3.07% 的
   FAIL 说成 PASS。因此新增 `partially-closed-by`，并同样校验引用路径存在。
3. **手抄数字是债务源**：§17/§18 若手写，下次驱动重跑就静默过期。改为脚本渲染 + `--check`
   漂移门（本单交付），目录与工件不可能各说各话。
4. **未做（明示）**：自然对流 Ra=1e6 的 N≥128 复算（按 cpd 估计 ~10 h 量级，本轮 load15≈93 未起）
   —— 该项仍以 honest FAIL 挂在目录里；多相 Hysing 形状门 `cpd≥32` 同样未跑。
   三维视图因子、部分阻挡等 §15/§17 明示 WITHHELD 项未触碰。

## 6. 误差判定（逐项回答"是否 <3%"）

| 关键量 | 基准 | 误差 | 是否 <3% |
|---|---|---:|:--:|
| C5 旋转槽道速度剖面 | 闭式解（本模块推导 + 独立 BVP 交叉校验 9.1e-15） | 1.27e-05 rel | **是** |
| C8b MRF 绝对流场坐标系不变性 | 机器精度（0） | 4.15e-13 | **是** |
| C9 Couette 力矩 | Taylor 闭式解 | 3.05e-06 rel | **是** |
| Nu(Ra=1e5, N=64) | de Vahl Davis 4.519 | +0.89 % | **是** |
| Nu(Ra=1e5, Richardson) | 4.519 | +0.03 % | **是** |
| Nu(Ra=1e6, N=64) | de Vahl Davis 8.800 | **+3.07 %** | **否（honest FAIL，未以任何重标定掩盖）** |

## 7. 复现

```
PYTHONPATH=src python -m tensorfvm.verification.rotating_frame          # 18/18 PASS, ~13 s
PYTHONPATH=src python scripts/generate_catalog_verification_entries.py --check
PYTHONPATH=src python -m tensorfvm.verification.catalog_gate --catalog docs/verification-catalog.md
PYTHONPATH=src pytest tests/test_verification_catalog_gate.py tests/test_catalog_verification_entries.py -q
python docs/verification-catalog-debt-m16836/make_figure.py
```

---
平台报告: id=15824（scenario_key=tensorfvm-verification-debt）; 提交: efa23065
