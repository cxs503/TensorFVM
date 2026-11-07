# TensorFVM 设备化债务普查（P8）

> 产物由 `tools/debt_census.py` + `tools/debt_report.py` 程序化生成，报告内所有数字均从
> `artifacts/p8/fvm_debt.json` 注入，无手抄。复跑：`python tools/debt_census.py && python tools/debt_report.py`。

- 扫描根：`src`（排除目录名：`__pycache__, example, examples, test, tests`）
- 扫描文件：**183**（解析成功 183，排除 0）
- AST 覆盖率：**100.0%**
- 源码内容指纹：`sha256:5ea58233f906cdfcd6d8fc2f680b518ddff7a4c5e0a37f5f76f7a8ea8c7af4fc`
- schema：`fvm-debt-census/1`

## 1. 结论一览

| 字段 | 数值 | 定义 |
|---|---|---|
| `numpy_only_count` | 97 | D1 numpy-only modules — modules with no device framework import |
| `sync_points` | 118 | D2 sync points — `.cpu()` + `.numpy()` tensor->host round trips |
| `no_device_constructs` | 84 | D3 no-`device=` constructs — tensor constructors with no device= and no inheritance |
| `host_loops` | 945 | D4 host loops — Python for/while statements (all run on the host interpreter) |

**物理/工程含义**：TensorFVM 目前有 97 个模块（占 53.0%）完全不经过任何设备框架，等价于「设备化零覆盖」；设备侧模块内部仍有 118 个 tensor→host 同步点与 84 处未声明 `device=` 的构造，前者会周期性排空加速器队列、后者会在 NPU 上触发隐式 H2D 拷贝或设备不一致错误。

## 2. 方法与判据（可复现）

四类债务均以 `ast` 模块静态判定（不执行被测代码，无副作用），定义冻结在
`tools/debt_census.py` 头部 docstring 与本文件第 2.1 节：

1. **D1 numpy_only**：模块内无 `torch`/`torch_npu`/`torch_sdaa`/`torch_mlu`/`jax`/`tensorflow`/`paddle`/`mindspore`/`cupy` 的 import（`ast.Import`/`ast.ImportFrom`，`level == 0`）。
2. **D2 sync_points**：`Call` 节点中方法名为 `cpu` 或 `numpy` 的站点数（含 `x.cpu().numpy()` 计 2 点）。`item`/`tolist`/`synchronize` 单列在 `breakdown.sync_extended`（同为同步语义，但不计入任务口径）。
3. **D3 no_device_constructs**：`torch.<factory>(...)` 构造调用满足：无 `device=` 关键字、无 `**kwargs` 展开、且不继承设备（`*_like` 系列继承首参设备，归入 `device_inherited_like`，不计债务）。`from_numpy`/`frombuffer` 属主机边界构造，计入债务但在 `no_device_host_boundary` 单列。带 `**opts` 展开的构造无法静态判定设备，归入 `device_unverified_spread`（不计入债务，如实标注为不可判定）。
4. **D4 host_loops**：`ast.For`/`ast.AsyncFor`/`ast.While` 语句节点总数——任何解释器级循环都在主机执行；循环体内含下标写（`a[i] = ...`/`+=`）或累加器 `.append/.extend/.update/.add` 的再计入 `breakdown.vectorizable_loops`（真正需要张量化的逐索引债务）。

### 2.1 覆盖率与诚实性

全部扫描文件均成功解析（`uncovered_files` 为空列表），不存在覆盖率缺口。

## 3. 交叉校验（独立第二实现）

同一个脚本内置 `tokenize` 词法级重算（与 AST 完全独立的代码路径、不受注释/字符串干扰）作为基准：

| 指标 | AST | tokenize 基准 | 偏差 | 是否 < 3% |
|---|---|---|---|---|
| `.cpu()` 站点 | 58 | 58 | 0.0000% | 是 |
| `.numpy()` 站点 | 60 | 60 | 0.0000% | 是 |

**误差判定**：D2 的 AST 计数与词法基准逐项完全一致（偏差 0.0000%，是，< 3%）。D1/D4 为结构性计数（import / 语句节点存在性），不存在「数值近似」意义上的误差量：判据为AST 可解析 + 两次运行字节级一致（见第 7 节）。D3 的口径误差无法用第二实现基准检验，其不确定性已显式隔离到 `device_unverified_spread` 桶中而非隐藏。

## 4. 分布与热点

### 4.1 张量构造分解

| 桶 | 调用数 |
|---|---|
| total factory calls | 415 |
| device= 显式给出 | 127 |
| 设备继承（*_like） | 148 |
| **无 device= 债务（合计）** | 84 |
| 　其中普通构造 | 70 |
| 　其中 from_numpy/frombuffer 主机边界 | 14 |
| **device 无法静态判定（**kwargs 展开）** | 56 |

### 4.2 循环形态

| 循环种类 | 数量 |
|---|---|
| `builtin:range` | 262 |
| `literal` | 191 |
| `host_sequence` | 181 |
| `builtin:enumerate` | 75 |
| `builtin:zip` | 66 |
| `host_index` | 56 |
| `method:items` | 54 |
| `call` | 23 |
| `while` | 16 |
| `builtin:sorted` | 10 |
| `method:splitlines` | 4 |
| `method:tolist` | 3 |
| `other` | 2 |
| `method:reshape` | 1 |
| `method:split` | 1 |

- 可张量化（循环体内有下标写/累加）的循环：**596**，占全部 host loops 的 63.1%。
- numpy-only 模块内的循环：**553**（占 58.5%，只能靠整体重写模块消除）。

### 4.3 热点模块（Top 10）

**同步点最多**

| module | sync_points | no_device | host_loops |
|---|---|---|---|
| `src/tensorfvm/cylinder3900_checkpoint.py` | 16 | 5 | 1 |
| `src/tensorfvm/cylinder.py` | 14 | 0 | 14 |
| `src/tensorfvm/benchmark_cylinder3d.py` | 11 | 0 | 3 |
| `src/tensorfvm/benchmark_30p30n.py` | 9 | 0 | 12 |
| `src/tensorfvm/verification/annular_order.py` | 8 | 0 | 2 |
| `src/tensorfvm/verification/porous_forchheimer.py` | 8 | 0 | 11 |
| `src/tensorfvm/verification/multi_element_ls.py` | 7 | 5 | 5 |
| `src/tensorfvm/verification/multi_element_skew.py` | 7 | 6 | 9 |
| `src/tensorfvm/body_fitted_transient3d.py` | 6 | 20 | 2 |
| `src/tensorfvm/mesh_cylinder3d.py` | 5 | 0 | 8 |

**无 device= 构造最多**

| module | no_device | sync_points | host_loops |
|---|---|---|---|
| `src/tensorfvm/body_fitted_transient3d.py` | 20 | 6 | 2 |
| `src/tensorfvm/verification/lid_cavity_extraction.py` | 10 | 0 | 4 |
| `src/tensorfvm/verification/fct_conservation_audit.py` | 7 | 0 | 3 |
| `src/tensorfvm/verification/multi_element_skew.py` | 6 | 7 | 9 |
| `src/tensorfvm/cylinder3900_checkpoint.py` | 5 | 16 | 1 |
| `src/tensorfvm/verification/multi_element_ls.py` | 5 | 7 | 5 |
| `src/tensorfvm/lid_cavity.py` | 4 | 2 | 7 |
| `src/tensorfvm/multi_element_coupled.py` | 3 | 0 | 3 |
| `src/tensorfvm/multi_element_flow.py` | 3 | 0 | 4 |
| `src/tensorfvm/verification/vof_rotation.py` | 3 | 1 | 11 |

**host 循环最多**

| module | host_loops | vectorizable_loops | sync_points |
|---|---|---|---|
| `src/tensorfvm/verification/vof_transport_audit.py` | 35 | 15 | 0 |
| `src/tensorfvm/overset_stokes.py` | 27 | 20 | 0 |
| `src/tensorfvm/body_fitted.py` | 26 | 22 | 0 |
| `src/tensorfvm/verification/report.py` | 26 | 6 | 0 |
| `src/tensorfvm/verification/native_viscous_passivity.py` | 21 | 18 | 0 |
| `src/tensorfvm/overset.py` | 18 | 13 | 2 |
| `src/tensorfvm/external_mesh_report.py` | 17 | 5 | 0 |
| `src/tensorfvm/hlpw30p30n.py` | 17 | 14 | 0 |
| `src/tensorfvm/verification/audit.py` | 17 | 8 | 0 |
| `src/tensorfvm/multi_element.py` | 16 | 16 | 0 |

## 5. 图

![FVM 设备化债务总览](../../artifacts/p8/fvm_debt_overview.png)

*图 1*：(a) 四类债务总量；(b) 183 个模块的设备能力分类（设备可用 / numpy-only / python-only），颜色块内为占比；(c) host 循环按可迭代对象形态的分布（Top 10），`builtin:range`/`host_sequence`/`literal` 三类即逐索引循环的主要来源，是张量化的首选目标。

## 6. 注意点与局限

1. **静态计数 ≠ 运行时热点**：本普查度量的是设备化工作量（代码债务面），不是性能代价。一次 `.cpu()` 落在每步内循环与落在 checkpoint 出口的代价差几个数量级，需配合 profiling 排序。
2. **D1 是粗粒度类别**：numpy-only 模块中相当部分是主机侧参考解/后处理/验收工具（`verification/*`、`*_reference.py`），设备化优先级应低于求解器核心；另有 14 个纯 Python（连 numpy 都不导入）模块单列未计入 D1。
3. **D3 的 `**opts` 三态**：`device_unverified_spread` 桶证明了「有 148 处靠 `*_like` 合法继承 + 127 处显式声明」与「56 处由调用方注入」并存，因此不能把 84 当作「全部缺陷」，只应作为待审清单。
4. **不覆盖 C 扩展/生成代码**：`ast` 只看 `.py`；`torch.compile` 生成物、triton kernel 字符串不在范围内。
5. **口径变更需同步**：任何一类判据调整都会改变全部历史数字，必须同时改 `tools/debt_census.py` docstring、`SCHEMA_VERSION` 与本报告，否则趋势不可比。

## 7. 复现与验证

```bash
python tools/debt_census.py --quiet     # 写 artifacts/p8/fvm_debt.json
python tools/debt_report.py             # 写 docs/p8/FVM_DEBT_REPORT.md + 图
python -m pytest tests/test_debt_census.py -q
```

判据：① JSON 五字段齐全；② 两次运行 `diff` 为 0（无时间戳/绝对路径）；③ 覆盖率 100.0% ≥ 70%（`uncovered_files` 为空列表，仍随 JSON 输出以支持 honest-FAIL 判读）。
