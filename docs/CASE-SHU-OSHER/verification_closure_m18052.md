# CASE-SHU-OSHER 验证欠账闭合（m18052 独立复核）

## 背景
feature_ledger 中 `CASE-SHU-OSHER`（算例池·激波-熵波干扰）状态为 `open`，
但实现已随 #576/#602 合并进 main（`src/tensorfvm/compressible/shu_osher_benchmark.py`、
`tests/test_compressible_shu_osher.py`、`docs/CASE-SHU-OSHER/`）。本单为独立复核，
确认已合并功能满足其验收线，并闭合账本。

## 独立复核（本单执行，非重跑 5 分钟扫描）
- 测试：`PYTHONPATH=src python -m pytest tests/test_compressible_shu_osher.py -v`
  → **5/5 PASSED**（11.40 s，loong64 架构）。
- 验收判据（来自已提交工件 `shu_osher.json`）：
  - L1(ρ) vs N=3200 参考解（相对参考密度极差）：N=1600 时 **0.12%** < 5%（且 < 3%）✅
  - 未扰动区 vs 解析 1+0.2sin(5x)：**2.165e-09** < 1%×0.2 ✅
  - 激波位置 vs Rankine-Hugoniot 轨道：**0.09%** < 2%×L ✅
  - 质量/能量预算漂移：**6.644e-15 / 1.475e-14** < 0.5%/1% ✅
  - 正性修复：**0** ✅
  - L1 随加密单调下降：1.92%→1.19%→0.39%→0.12% ✅
- 观测收敛阶（L1）：0.67, 1.61, 1.71（末段 ~1.7，激波 O(Δx) 主导首段，正常）。

## 账本闭合
`UPDATE feature_ledger SET status='verified' WHERE slug='CASE-SHU-OSHER'`（1 行）。

## 报告
平台报告 id **16939**（mission 18052，scenario CASE-SHU-OSHER），
六要素齐全（案例/方法/数字表/图/注意点/误差判定），图已内嵌（52 KB 总负载）。

## 给下次自找活的建议
- 本算例无文献数字化点对点数据；若引入 Shu-Osher 原图数字化参考，可替换 N=3200
  参考解做「对精确解」误差，无需改框架（见 report.md 注意点 4）。
- 跨格式比较（WENO5+SSP-RK3 已有 cross-scheme 参考）可在同一网格下与
  MUSCL-Hancock 对比分辨率，是下一个可打的分辨率验证面。
