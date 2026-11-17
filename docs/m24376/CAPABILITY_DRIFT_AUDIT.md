# TensorFVM 能力注册漂移对账与收口（m24376 · CAP-REG-FVM）

## 1. 案例介绍（物理/工程背景与对象）

TensorFVM 在 m23215 建立了 `@capability` 自注册机制与两份总账文档
（`SOLVER_INVENTORY.md` / `KERNEL_ARCHITECTURE.md`），把"哪些数值能力存在、
由什么证据背书"从散文变成可机检的代码。但注册表是**活物**：m23215 之后
仓库继续合并了大量模块，注册表与代码树的漂移无人对账。本单对
CAP-REG-FVM 锚点做**漂移对账+收口**，不重做 m23215 已交付的两份文档本体。

对账发现三类漂移（全部程序化检出，见 §5）：

1. **红灯漂移**：`fvm.verification.annular_band`（m23215 之后并入）声称
   `verified=True`，但 `docs/m23215/error_bands.json` 中没有它的实测带——
   `test_registry_error_bands_equal_the_measured_bands` 在 main 上红。
   "quoted numbers are the measured numbers" 契约被后来的合并打破。
2. **端点漂移**：3 个 wired solver 能力（`fvm.core.annular_axial`、
   `fvm.core.ins2d_collocated`、`fvm.internal_flow.backward_step`）没有声明
   消费者可调用的 endpoint（`WIRED_SOLVER_WITHOUT_ENDPOINT` 告警 ×3）。
3. **覆盖漂移**：m23215 清单自认的一批高被引模块
   （`verification/core.py` 24 次、`native_mesh3d.py` 18 次、
   `rans_validation/channel.py` 16 次、`shallow_water/swe1d.py` 14 次、
   `panel_added_mass.py` 13 次、`shallow_water/dam_break.py` 13 次、
   `verification/kmm1987.py` 13 次、`native_hybrid_ns3d.py` 12 次——被 src
   内模块 import 次数，m23215 实测）始终未登记。

## 2. 怎么做的（方法/流程/关键设置）

* **红灯收口**：扩展 `scripts/m23215_error_bands.py`，新增
  `band_annular_verification()`——经 verification 入口
  `tensorfvm.verification.annular:run_case`（该能力自己声明的公开入口）对同一
  解析基准（环形轴向 Poiseuille 闭式解）重测误差带，角向扫掠
  ntheta=32/64/128（nr=32）+ 径向扫掠 nr=16/32（ntheta=32），与
  `fvm.core.annular_axial` 同协议；重生成 `error_bands.json`。实测
  max_error_pct 与装饰器原声明**逐位一致**（0.2813），声明无需改动。
* **端点收口**：3 个 wired solver 补 `endpoint='route:…'`，指向真实可导入的
  类/函数（`tensorfvm.annular:solve` 等），不新造 CLI、不改 pyproject。
* **覆盖收口**：8 个模块按 m23186/m23215 规范补 `@capability` 注册，
  一律保守申报（`developing` + `MECHANICS`/`XVALUE` 证据级，无一升
  `mature`/`verified`——升格必须先有 booked band，本单一并不做）。
* **验证闭环**：`registry.py --static` 对账 + 契约测试 17 项 +
  全部被改模块的单测；before/after 快照用 `git archive HEAD` pristine
  副本扫描（含 src/tests/docs/scripts，避免 test_path 解析假阳性）。

## 3. 计算结果（数字表，程序化注入）

**注册表总量**：before（HEAD）52 caps / 3 problems
→ after（本单）60 caps / 0 problems；
maturity 分布 mature 4 / developing 56；
契约测试 16 passed + 1 failed → **17 passed**。

**annular_band 实测带**（grade=ANALYTIC，参考=环形轴向 Poiseuille 闭式解）：

| 量 | 值 |
|---|---|
| max_error_pct（ntheta=128, nr=32） | 0.2813 |
| coarse_max_error_pct（ntheta=32, nr=16） | 3.0336 |
| 相对 L2 @nt32/nr16 | 2.532% |
| 相对 Linf @nt128/nr32 | 0.2813% |
| 流量误差 @nt128/nr32 | 0.1056% |
| 收敛阶 ntheta 32→64 / 64→128 | 1.985 / 1.929 |

**新登记 8 项**（id / kind / maturity / 证据级 / test_path）：

| id | kind | maturity | grade | test_path |
|---|---|---|---|---|
| `fvm.core.hybrid_ns3d` | solver | developing | MECHANICS | tests/test_native_hybrid_ns3d.py |
| `fvm.mesh.native3d` | mesh | developing | MECHANICS | tests/test_core_mesh.py |
| `fvm.rans.channel_validation` | verification | developing | MECHANICS | tests/test_channel_kmm180_loglaw.py |
| `fvm.seakeeping.panel_added_mass` | maneuvering | developing | MECHANICS | tests/test_added_mass_potential.py |
| `fvm.shallowwater.dambreak` | solver | developing | MECHANICS | tests/test_dam_break_front_metric.py |
| `fvm.shallowwater.swe1d` | solver | developing | MECHANICS | tests/test_dambreak_depth_modes.py |
| `fvm.verification.core` | verification | developing | MECHANICS | tests/test_richardson_gci.py |
| `fvm.verification.kmm1987` | verification | developing | XVALUE | tests/test_channel_kmm180_direct_anchor.py |

## 4. 结果图

* `fig_annular_band_convergence.png`：annular_band 实测误差带
  （相对 L2/Linf vs ntheta，log-log，含二阶参考斜线）——数据全部读自
  `error_bands.json`。
* `fig_registry_drift.png`：注册表 before/after 对比
  （mature/developing/contract problems 三组柱）——before 用 pristine
  HEAD 副本扫描。

（本单为注册表审计任务，无流场云图；两图分别为"与解析解比较曲线"与
"能力账目对比图"，符合纯审计任务的图表义务。）

## 5. 注意点（踩坑/局限/适用范围）

* **静态扫描器静默跳过语法损坏文件**：本单补丁过程曾把 3 个装饰器改坏
  （重复参数），`--static` 不报错而是少计 3 个能力——数量对不上账时
  （52+8≠57）靠总数守恒发现。已修复；建议后续给扫描器加
  "文件解析失败即 problem"（未在本单实施，避免扩大改动面）。
* **子包导入层级**：`shallow_water/`、`verification/`、`rans_validation/`
  内的模块必须 `from ..registry import capability`（双点），顶层模块用
  单点——写错即 NameError/ImportError。
* **pristine 快照必须含 tests/docs**：只导出 src 会让 test_path 解析全部
  失败，把 3 个真问题虚报成 50 个。
* **局限**：8 项新登记均为保守申报；其中 `panel_added_mass`（有 Lamb
  闭式锚）、`kmm1987`（有 XVALUE 参考）具备升 verified 的潜力，但升格
  必须先跑出 booked band，留给后续单。
* 存量失败（与本单无关，git stash 验证 main 上同样红）：
  `test_core_mesh.py::test_body_fitted3d`（MeshData 内部面缺 'internal'
  patch 标签）、`test_internal_flow_benchmarks.py` 的自然对流与台阶再附
  2 项。

## 6. 误差判定（vs 基准，逐项）

| 关键量 | 基准 | 误差 | <3%？ |
|---|---|---|---|
| annular_band 相对 L2（生产分辨率 nt128/nr32） | 环形 Poiseuille 闭式解 | 0.168% | **是** |
| annular_band 相对 Linf（同上） | 同上 | 0.2813% | **是** |
| annular_band 流量误差（同上） | 同上 | 0.1056% | **是** |
| annular_band coarse（nt32/nr16） | 同上 | 3.0336% | 否（>3%，已按协议披露 coarse 值；生产分辨率达标） |
| 注册表契约 problems | 0 容忍 | 3 → **0** | 是（按 0 容忍口径收口） |

## 7. 交付物与验证命令

```bash
python src/tensorfvm/registry.py --static            # 60 caps, 0 problems
PYTHONPATH=src python scripts/m23215_error_bands.py --json docs/m23215/error_bands.json
python -m pytest tests/test_capability_registry_fvm.py -q   # 17 passed
python scripts/m24376_drift_figures.py                      # 两张图程序化生成
```
