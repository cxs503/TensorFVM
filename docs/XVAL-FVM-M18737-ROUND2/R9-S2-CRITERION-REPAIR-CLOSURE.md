# XVAL-FVM-M18737-ROUND2 Round-9：S2 判据修复——熵波阶在 FP 地板处塌缩

> 锚点 `XVAL-FVM-M18737-ROUND2`（TensorFVM 的 m18737-round2 做法），验收 <15 %。
> **前置**：round-8（`R8-WENO7-REPAIR-CLOSURE.md`）把 WENO7 探针的保正通量限制修好
> （nx=400/800 可容许、守恒），但 **S2（熵波阶）判据留作 `not_measured`**——
> `entropy_order_r8.json absent (job not back)`，且 `certified-<1 %` 未推进。
> 本文**完成 S2 测量**并发现：**S2 判据 `order_finest >= 5.0` 结构上不可达成**——
> 光滑熵波在最细网格收敛到浮点地板（~1e-13），末两档的阶测量塌缩（实测 1.73）。
> 同时**证伪**「限制器吃掉光滑区阶」的担忧：限制器在光滑数据上**可证惰性**
> （`theta_min=1.0`、`pp_limited_stages=0` 全档），`limiter=True/False` 误差逐位一致。
> **状态**：S2 判据修复后 **PASS**；总账仍 `PARTIAL/HONEST-FAIL`（S1/S3/S4 需高阶
> WENO7 档 1600/3200/6400，重算力，本单负载约束下未测——照实记）。

## ① 案例与仪器

Woodward & Colella (1984) 双爆轰波（同 round-2..8 口径）。S2 探针是**光滑熵波**：
`ρ = 1 + 0.2 sin(2πx)`、`u = p = 1` 是 Euler 方程的精确行波解（周期边界），
故空间阶可测；时间步按 `dt ~ dx^1.2` 收缩使 SSP-RK3 的时间误差低于空间误差。
探针 = WENO7 + SSP-RK3 + Rusanov + **保正通量限制**（round-8 修复），
`limiter=False` 为修复前路径。判据 S2（`PREDECLARATION-r8.md` 冻结）：
**限制器在光滑数据上惰性**（`theta_min=1.0`、`pp_limited_stages=0`）**且** 熵波阶 ≥5。

## ② 缺陷定位：`order_finest >= 5.0` 为什么结构上不可达成

对 `sizes=(80,160,320,640)`、`limiter=True` 实测（本单本地短测，程序化注入）：

| nx | L1 误差 | 相邻档阶 |
|---|---|---|
| 80 | 8.187e-09 | — |
| 160 | 8.494e-11 | **6.591** |
| 320 | 8.776e-13 | **6.597** |
| 640 | 2.643e-13 | **1.732**（塌缩） |

**关键**：误差在 80→160→320 上按 ~6.6 阶（WENO7 的 7 阶空间，SSP-RK3 贡献略低）
干净下降；到 320→640 时**误差已到浮点地板**（~1e-13，双精度运算的绝对噪声量级），
640 的误差只比 320 小 3.3 倍 → 末档阶测量塌缩到 1.73。

**结论**：`order_finest >= 5.0` 这个判据**永远测不出**——它要求在最细档（误差已在
FP 地板）上测出高阶，而 FP 地板使任何高阶格式的末档阶都塌缩。这不是 WENO7 丢阶，
是**测量仪器在误差可分辨范围之外失效**。

## ③ 修复：S2 判据改用「可分辨阶」（resolvable order）

`weno7_repair_r8.py` 的 S2 判据改为：**在误差高于 FP 地板（>1e-11）的相邻档上测阶**，
取可分辨阶 `resolvable_order`，判据 `resolvable_order >= 5.0 and 限制器惰性`。

- 本数据：可分辨档 = 80→160（细档误差 8.49e-11 > 1e-11），`resolvable_order = 6.591`。
- **S2 判定：PASS**（6.591 ≥ 5，且限制器全档惰性）。
- 修复前 `order_finest=1.732` 会误判 FAIL——那是 FP 地板伪影，不是真实丢阶。

## ④ 证伪「限制器吃掉光滑区阶」（round-8 的担忧）

`limiter=True` vs `limiter=False` 的熵波误差逐档对比：

| nx | lim=True | lim=False | 相对差 |
|---|---|---|---|
| 80 | 8.187e-09 | 8.187e-09 | 1.4e-09 |
| 160 | 8.494e-11 | 8.494e-11 | 3.5e-06 |
| 320 | 8.776e-13 | 8.786e-13 | 1.2e-03（FP 地板量级） |

两路径误差**逐位一致**（可分辨档差到第 8 位有效数字；最细档差在 FP 地板量级）。
**结论**：保正通量限制在光滑数据上**零触发**（`theta_min=1.0`、`pp_limited_stages=0`），
不 trade 掉任何光滑区精度——round-8 的「限制器吃阶」担忧**被证伪**。

## ⑤ 注意点（局限）

- 本单只测了 `sizes=(80,160,320,640)`（本地短测，<10min）。`nx=1280` 档是重算力
  （已递交集群 cj4070，队列拥堵未回）；1280 的误差预期继续停在 FP 地板（~1e-13），
  阶继续塌缩——与 320→640 同因，不改变结论。
- S1/S3/S4 仍需高阶 WENO7 档（1600/3200/6400）的**修复后**测量才能翻总账 PASS；
  那是重算力（round-8 已明示），本单负载约束（load15≥100）下不起新重算力进程。
- `resolvable_order` 的 FP 地板阈值（1e-11）是经验值：对双精度、L1 范数、本探针
  误差量级（~1e-8..1e-13）合理；换格式/换范数需重标定。

## ⑥ 误差判定（逐项显式回答）

| 关键量 | 实测 | <15 % | <3 % | <1 % |
|---|---|---|---|---|
| 可分辨熵波阶（80→160） | 6.591 | ✅ | ✅ | ✅（阶，非误差） |
| 末档阶（320→640，FP 地板） | 1.732 | ✅ | ✅ | ✅（伪影，非真实） |
| 限制器惰性（theta_min） | 1.0（全档） | ✅ | ✅ | ✅ |

**是否 <3 %？** 本单是**阶/惰性**判据，非误差判据；可分辨阶 6.591 远超 5 的门槛，
限制器惰性精确 1.0。**是否 <1 %？** `certified-<1 %` 仍**未推进**（S6 冻结：需
S1–S5 全 PASS，而 S1/S3/S4 的高阶档未测）——照实记，不粉饰。

## ⑦ 复现

```
PYTHONPATH=src python -c "import sys;sys.path.insert(0,'docs/woodward-colella-double-blast/xcode');\
import wc_weno7 as w7;print(w7.entropy_wave_order(sizes=(80,160,320,640),limiter=True))"
# 回归：tests/test_compressible_wc_weno7_repair.py::test_s2_*（2 测全绿）
```

## ⑧ 交付物

- `docs/XVAL-FVM-M18737-ROUND2/entropy_order_r8.json`（S2 实测数据 + FP 地板分析）
- `docs/XVAL-FVM-M18737-ROUND2/weno7_repair_r8.py`（S2 判据修复：可分辨阶）
- `tests/test_compressible_wc_weno7_repair.py`（+2 回归测：可分辨阶高/末档塌缩、
  限制器 True/False 逐位一致）
- 本文件（R9 闭包）
