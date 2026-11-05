# PR #174 CI 红修复：`MultiGridStudy` 缺 `fallback_reason`（TensorFVM）

## ① 案例介绍
物理背景：本修复不改动任何流场求解器，改动面是**验证与确认（V&V）工具链**——
Roache/Celik 网格收敛指数（GCI）估计器。它是"算出的阻力/升力能否被称为已收敛"的判据层，
所以它的接口一致性直接决定工程结论是否可引用。

几何与参数：无几何；被修的对象是 N≥3 层网格（spacing `h` 细→粗严格递增）上的
截断模型 `f(h) = L + C h^p`。合成算例取闭式族 `L = 1.0`, `C = 0.5`, `p = 2.0`，
`h_k = 0.25/2^k`（k = 0..N-1），解析极限已知，可作为估计器的真值基准。

红的原因（CI 观测）：PR #174 的 `test (3.11)` / `test (3.12)` 均 failure，
失败点 `tests/test_multi_grid_gci.py::test_reference_agreement_accepts_a_multi_grid_study`：

```
AttributeError: 'MultiGridStudy' object has no attribute 'fallback_reason'
  src/tensorfvm/verification/richardson.py:242 in reference_agreement
```

## ② 怎么做的
1. 定位：用 `pytest -q` 全量复现，把 -q 进度流的失败字符位映射回收集顺序（1401 项中的第 414 项），
   直接锁定失败用例，而不是猜；
2. 归因：`richardson.reference_agreement()` 被设计成**同时接受**三网格 `RefinementStudy`
   与 N 层 `MultiGridStudy`（函数文档明确承诺把 `asymptotic / fallback_reason /
   safety_factor` 三个限定词随结论一起传播），但 `MultiGridStudy` 这个 dataclass 漏了
   `fallback_reason` 字段——三网格路径写好了限定词，N 层路径没有；
3. 修复（最小、加法式）：
   - `src/tensorfvm/verification/multi_grid.py`：`MultiGridStudy` 增加 `fallback_reason: str`，
     在决定回退的同一点取值 `non-monotone` / `undetermined-order` / `none`，并写进返回值；
     模块文档补一条字段说明（与三网格同一套词汇表）；
   - 同时删掉 `multi_grid_order` 内两处**从未被读取**的 `res` 赋值（ruff F841）——
     该文件因本修复进入 CI 的"改动文件 lint 范围"，不修则 lint job 必红；
     并对两个受改文件跑 `ruff format`（纯格式，**数值逐位不变**，见下）；
4. 验证工具链：`pytest`（目标套件 + 受影响的 gci 族）、`ruff check` / `ruff format --check`、
   以及格式前后数值快照的逐位对比。

## ③ 计算结果（脚本注入，非手抄）

`artifacts/m4492/capture_numbers.py` 的输出（`logs/numbers_before_format.json` 与
`logs/numbers_after_format.json` 完全一致，`diff` 空）：

| N | 拟合阶 p | 极限 L 相对误差 | GCI | 带区间 | 带覆盖真值 | fallback_reason | 效率指数 |
|---|---|---|---|---|---|---|---|
| 3 | 2.000000 | 2.220e-16 | 0.002437 | [0.997563, 1.002437] | yes | none | 9.11e-14 |
| 4 | 2.000000 | 0.000e+00 | 0.000610 | [0.999390, 1.000610] | yes | none | 0.00e+00 |
| 5 | 2.000000 | 0.000e+00 | 0.000153 | [0.999847, 1.000153] | yes | none | 0.00e+00 |
| 6 | 2.000000 | 0.000e+00 | 0.000038 | [0.999962, 1.000038] | yes | none | 0.00e+00 |
| 7 | 2.000000 | 0.000e+00 | 0.000010 | [0.999990, 1.000010] | yes | none | 0.00e+00 |

红绿对照（同一用例，同一命令）：

| 状态 | 命令 | 结果 |
|---|---|---|
| 修复前 | `pytest -q tests/test_multi_grid_gci.py::test_reference_agreement_accepts_a_multi_grid_study` | `1 failed`（AttributeError） |
| 修复后 | `pytest -q tests/test_multi_grid_gci.py` | `36 passed` |
| 修复后（gci 全族） | `pytest -q tests/test_multi_grid_gci.py tests/test_richardson_gci.py tests/test_gci_fallback_band.py tests/test_gci_band_coverage.py tests/test_scalar_transient_diffusion.py` | `77 passed` |
| lint 门 | `ruff check` + `ruff format --check`（两个受改文件） | `All checks passed!` / `already formatted` |

## ④ 结果图
左：N=3..7 的 N 层最小二乘拟合极限与闭式极限 L=1.0（对数纵轴，阴影为 GCI 带）。
右：效率指数 `|L-ref|/GCI`（Roache：>1 表示带不是界）；橙色点为非单调阶梯的回退带
（`Fs = 3.0`，`fallback_reason = 'non-monotone'`，
指数 0.33，即回退带刻意保守）。

## ⑤ 注意点
* **这是主分支的潜在红**：`fallback_reason=study.fallback_reason` 这行和缺字段的
  `MultiGridStudy` 在 `origin/main` 上同时存在（`git show origin/main:...` 已核对），
  所以 #174 只是"继承"了红；本修复同时把 main 复原为绿，PR 内容本身（CI 门范围化）未被削弱。
* 爆炸半径：`multi_grid_order`/`MultiGridStudy` 的调用方仅 `tests/test_multi_grid_gci.py`
  与 `scripts/audit_multi_grid_gci.py`，且均为关键字构造，新增字段不破坏位置参数调用。
* `undetermined-order` 是本次新增的词汇（单调但拟合退化：`1` 与 `h**p` 两基列在
  允许的阶数区间内共线），它与 `non-monotone` 语义不同，故不复用旧词。
* 局限（honest）：本机 128 核被 20+ 个并发 agent 压到 load≈900，全量 `pytest` 两次
  后台跑均在 50–56% 处被预算截断，只有"截至 56% 仅此 1 个失败"的基线证据；
  因此 CI 绿的最终判定仍以 GitHub 上的 3.11/3.12 运行结果为准。

## ⑥ 误差判定
* 关键量 1：N 层拟合极限 vs 闭式真值 L=1.0 → 最大相对误差 2.220e-16
  （N=3..7 全部）→ **远小于 3%，达标**；带覆盖真值：全部覆盖。
* 关键量 2：接口约定（`reference_agreement` 对两种 study 都能给出限定词）→ 修复前
  `AttributeError`（功能性缺失，非数值误差），修复后 `fallback_reason` 与三网格路径
  逐字一致（`non-monotone`），断言通过。
* 关键量 3：格式化的数值无损 → 两个 JSON 快照逐字节 `diff` 为空 → **0% 偏差**。
