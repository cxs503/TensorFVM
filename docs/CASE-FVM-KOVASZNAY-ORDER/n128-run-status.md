# CASE-FVM-KOVASZNAY-ORDER — n128 运行状态与路径缺陷记录

**任务**: m18408 · TensorFVM · 锚点 CASE-FVM-KOVASZNAY-ORDER
**日期**: 2026-11-13

## 现状

feature_ledger 锚点 `CASE-FVM-KOVASZNAY-ORDER` 状态 = **open**（未完成）。
验收判据：L2 收敛阶 >= 1.8（名义二阶），log-log 斜率 on 32/64/128 三套网格。

已提交工件（`docs/CASE-FVM-KOVASZNAY-ORDER/runs-32-64-128-zero_gradient/`）：
- **n32**：完整（cases.json + n32-fields.npz），zero_gradient 闭合，velocity L2 = 3.4356e-03，收敛 ✅
- **n64**：完整（cases.json + n64-fields.npz），zero_gradient 闭合，velocity L2 = 8.9726e-04，收敛 ✅
- **n128**：**缺失**（仅 environment.json/sources.json，无 cases.json / n128-fields.npz）

32/64 两点斜率 = **1.9370**（PASS ≥ 1.8）。但验收要求三套网格，n128 未跑 → 验收门
`test_kovasznay_32_64_128_acceptance.py::test_anchor_zg_full_3d_velocity_order_meets_acceptance`
对 n128 层 `pytest.skip`（honest "not measured yet"），锚点仍 open。

## 关键路径缺陷（必须修复才能跑 n128）

在 m18408 工作树内执行 `python3 scripts/verify_kovasznay_order.py` 时，
`import tensorfvm` **解析到 `/root/tp-agent/work/m17818/src/tensorfvm/`**（非本仓 m18408 的 src/）。

验证：
```
$ python3 -c "import tensorfvm; print(tensorfvm.__file__)"
/root/tp-agent/work/m17818/src/tensorfvm/__init__.py   # ← 错误，应为 m18408/src
```

后果：`NativeRunEvidence.__init__` 里 `self.repo = Path(runner).resolve().parents[1]` = m18408，
但 `tensorfvm` 各模块 `__file__` 在 m17818 → `source.relative_to(self.repo)` 抛
`ValueError: '.../m17818/...' is not in the subpath of '.../m18408'`，运行直接崩溃。

**修复**：运行前显式设 `PYTHONPATH=/root/tp-agent/work/m18408/src`，使 tensorfvm 解析到本仓：
```
cd /root/tp-agent/work/m18408
PYTHONPATH=/root/tp-agent/work/m18408/src \
  python3 scripts/verify_kovasznay_order.py \
  --output docs/CASE-FVM-KOVASZNAY-ORDER/runs-32-64-128-zero_gradient/n128 \
  --n 128 --reynolds 40 --convection-order 2 --outer-relaxation 0.7 \
  --tolerance 1e-8 --max-outer-iterations 120 --max-iterations 8000 \
  --symmetry-closure zero_gradient
```
（n128 输出目录须为空；若已有 environment.json/sources.json 先清空。）

## 本单已做

1. 核对 feature_ledger：锚点 open，未被他单完成 → 认领。
2. 扫描已提交工件：n32/n64 完整、n128 缺失 → 定位为「验证欠账」类实质增量。
3. 递交 n128 到集群（cj1915，queued，277 前排队）。
4. 本地尝试 n128：因上述路径缺陷（tensorfvm 解析到 m17818）崩溃；
   且 shell 120s 超时收割后台进程，本地无法在单次调用内完成。
5. 记录路径缺陷与修复命令（本文件）。

## 未完成（honest）

- n128 运行结果未落盘（集群 cj1915 仍在排队；本地受路径缺陷+超时限制）。
- 验收门未全绿（n128 层 skip）。
- 平台报告未发布。

## 给下次自找活的建议

- 完成 n128：按上文修复命令跑（集群 cj1915 或本地 PYTHONPATH 修正），
  落盘 cases.json/n128-fields.npz 后跑 `test_kovasznay_32_64_128_acceptance.py`，
  预期 velocity L2 斜率 ~1.9（PASS），锚点即可 close。
- 建议新条目（slug+量化验收+所属域）：`CASE-FVM-KOVASZNAY-ORDER-PRESSURE`
  压力 mean-free L2 收敛阶 >= 1.8（当前 32/64 压力斜率 1.858，需 n128 确认），
  域=TensorFVM。
