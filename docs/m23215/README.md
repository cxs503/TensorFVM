# m23215 · TensorFVM 能力注册 + 内核架构审计（CAP-REG-FVM）

本单把 TensorFVM「按案例纵向生长」的现状转成**可机读资产**，为重构定靶。派单三件交付：

| # | 交付物 | 内容 | 机读形态 |
|---|---|---|---|
| ① | `src/tensorfvm/registry.py` | `@capability` 装饰器 import 时自注册；`--static` AST 模式（免 torch）+ import 模式；`--parity` 双模式逐字段比对；反虚标纪律（`mature` ⇒ `wired and verified`，本地注册即抛 `MaturityDisciplineError`） | `docs/m23215/registry_dump.json` |
| ② | [`SOLVER_INVENTORY.md`](SOLVER_INVENTORY.md) | 46 项能力逐条：模块/入口/kind/maturity/A-B-C 证据等级/wired/tested/test_path/误差带 | 同左（表）+ `fvm_audit.json` |
| ③ | [`KERNEL_ARCHITECTURE.md`](KERNEL_ARCHITECTURE.md) | 子包级 import 依赖图（mermaid）、复制变体族谱的**证实/修正**、共享件真实复用关系、五接口缺口表 | `fvm_audit.json` |

## 实测结论（一句话版）

- 注册 46 项：mature **3**（annular 轴向 Poiseuille / PLIC 几何重构 / 线性诊断），developing 43，planned 0；wired 35，tested 41，verified 4，endpoint 3，error 级 problem **0**。
- 全仓 587 模块 / 198,768 行 / 24 子包 / 195 顶平铺模块，**只有 1 个 `Protocol`**（`mesh_api.Mesh2D`，仅 2D）、**39 个模块各自实现 `class.step()`**——五接口（mesh/flux/BC/linear/time）**一个都没统一**。
- 复制变体的派单描述**部分证伪**：vof 四代不是 4 份复制（精确重复函数体仅 2 处，且有 5 条下游 import 边，`vof_phase0` 是事实内核）；wave_tank 两支是「库 + 消费者」，不是平行实现。
- 孤儿共享件：`overset_flux.py` 与 `pmm_solver.py` 在 `src` 内**零引用**（只被测试调用）；`wave_tank.py` 被 `wave_tank/` 包**遮蔽**，是不可 import 的死代码。
- 未登记家底 541 模块：**没有能力声明**，本单不替它们背书。

## 复现（全部在本工作树可重算）

```bash
python src/tensorfvm/registry.py --static --strict-exit   # 免 torch 的 AST dump，error 退出码
python src/tensorfvm/registry.py --static --parity        # static↔import 逐字段一致
PYTHONPATH=src python scripts/m23215_error_bands.py       # A 级误差带现算（4 条）
python scripts/m23215_fvm_audit.py --check                # 装饰器↔注册表对账 + 全仓审计
python -m pytest tests/test_capability_registry_fvm.py -q # 契约测试
```

## 误差判定（平台标准：关键量 vs 基准 >15% 不可接受）

| 能力 | 参考 | 实测 max_err% | <3%? |
|---|---|---|---|
| `fvm.core.annular_axial` | 环形轴向 Poiseuille 闭式解 | 0.281（128×32）；粗网格 3.034 | 精化解 ✅ / 粗网格 ❌（3.03%，且**径向加密无效**，误差由角向分面主导） |
| `fvm.freesurface.plic_geometry` | `A(n, c(α)) = α` 代数恒等式 | 1.4e-11 | ✅ |
| `fvm.flux.overset_transfer` | 非匹配界面通量守恒（精确） | 1.2e-13 | ✅ |
| `fvm.linear.diagnostics` | 直接解的分量型后向误差 | 1.5e-14 | ✅ |

其余 42 项**没有**机器核验的误差带，因此 `verified=False`、不参与 mature——这是本单刻意的 honest-FAIL 边界。

## 目录约定

- `docs/m23215/` 成果（本文件、两份主文档、`fvm_audit.json`、`error_bands.json`、`registry_dump.json`）
- `runs/` 只放运行记录，不入 git（本单未产生 runs 工件）
