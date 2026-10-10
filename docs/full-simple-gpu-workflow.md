# 完整 SIMPLE / SA 的大网格 GPU 验证

本轮补齐的是完整贴体 SIMPLE 的弯曲通道制造解、大规模非正交网格、真实稳态 SA 以及原 128×112 SA 五步一致性失败的复核。SIMPLEC、PISO、PIMPLE 没有独立生产实现，不计入算法覆盖。

## 可复现入口

依赖 CUDA Torch、SciPy>=1.10、NumPy、matplotlib；配置输出必须为空。

```sh
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=src python -m tensorfvm.verification.simple_gpu --output /tmp/full-simple-new
PYTHONPATH=src python -m tensorfvm.verification.simple_gpu_audit /tmp/full-simple-new
PYTHONPATH=src python scripts/audit_steady_sa.py /tmp/full-simple-new
PYTHONPATH=src python scripts/publish_simple_profiles.py /tmp/full-simple-new
PYTHONPATH=src python -m pytest tests/test_sparse_simple.py -q
```

正式输出：[完整报告](verification-simple-gpu/report.md)、[PDF](verification-simple-gpu/report.pdf)、[定量结果](verification-simple-gpu/quantitative-results.json)、[独立审计](verification-simple-gpu/audit.json)。原 SA 五步复核：[报告](verification-sparse-sa-consistency/report.md)。

## 求解器与接口

`SparseBodyFittedSolver` 是可选的生产 `BodyFittedSolver` 子类，复用原 SIMPLE、Rhie–Chow、动量、非正交扩散与 SA 输运算子。矩阵组装结果每次与原压力算子进行数值核对。稀疏 CSR 和粗网格矩阵在 CPU 组装；Torch GMRES、压力行/粗网格预条件的 LU 与场更新在真实请求设备执行。这里是明确的混合组装路径，不宣称所有操作都驻留 GPU。

```python
from tensorfvm.solver import SolverConfig
from tensorfvm.sparse_simple import SparseBodyFittedSolver

cfg = SolverConfig(mesh_type="flat-plate", turbulence_model="spalart-allmaras",
                   nx=64, ny=56, length=1, height=.2,
                   cylinder_radius=None, reynolds=100000, device="cuda")
solver = SparseBodyFittedSolver(cfg)
solver.coupled_coarse_shape = (16, 14)
solver.sa_inner_iterations = 8
result = solver.solve()
```

默认 `SimpleSolver(flat-plate)` 仍调用历史后端；如需新后端应显式选用上面的类。原失败数据、原默认算法与对应源码均保留。制造解通过 `momentum_body_force` 指定连续方程体力；该力必须是每个控制体的二维力密度，与网格/设备一致。

## 问题与验收

弯曲通道边界为 `g(x)=0.2 sin⁴(πx/2)` 和 `g(x)+1`，`0≤x≤2`。制造流场 `u=w(y−g), v=g′w`、`w=6η(1−η)`、`p=12μ(2−x)`，由连续 NS 给出体力。动量有非零横向输运，压力由完整 SIMPLE 求解。制造解不是无外力自然弯管流动，也不替代实验工程验证。

网格为 32×16、64×32、128×64、256×128；最大 32,768 控制体、98,304 个主要流动未知量。细网格只从实际计算的粗网格场插值启动，无解析初始化。多边形边界的顶点位于曲线上，边界面为弦段，几何近似随网格加密；面法向和控制体连心线不正交，包含生产非正交修正。迎风对流会贡献一阶误差，报告实际加密阶数。

速度/压力 L2 与最大误差分别严格 `<3%`；最终动量、连续性、显式收敛与线性真实残差另设门；真实稳态 Rhie–Chow 通量与保存通量的相对差必须小于 1e−7。CPU/CUDA 主场相对 L2 差统一 `≤1e−6`。独立 NumPy 审计重建几何、梯度、完整末态动量/质量及 SA 非线性残差和壁面牵引。

SA 平板保留原 Re=100000、原模型、首层贴壁积分，并实际计算到稳态门。Cf 的完全湍流经验关联与最大局部 y+ 单独列出；残差收敛和 GPU 一致不等于湍流物理 3% 通过。

## 计时

CPU 单线程、float64、RTX3090。CPU 与 CUDA 分别做完整稳态计算；完整时间各一次，不是重复中位数。随后从同一实际中间场重启，完整预热，分别三次执行五步完整 SIMPLE。窗口计时包括 CPU 组装、传输、所有动量/压力/SA 解及外迭代，排除重启复制与绘图。窗口结果不能作为整个稳态求解加速比。

首轮小网格合格、128×64 外迭代缓慢的部分试算已保存在 [初始归档](verification-simple-gpu-initial/README.md)，含当时源码快照；调整 Anderson 量纲缩放/历史长度及内迭代残差预算后仍发现长波误差缓慢，因此增加粗网格动量—压力耦合残差校正后重算，未放宽门槛。

插值与粗校正之后采用常 mobility 的辅助速度投影，保持物理压力；末态必须重新满足物理 Rhie–Chow 固定点。历史残差记录 SIMPLE 主步的状态，最终接受步不再追加粗校正。

粗网格校正也严格核对原动量/压力梯度/稳态 Rhie–Chow 的耦合 Picard 响应，实际 LU 在请求设备执行；最后一次矩阵、右端和解均导出供独立重算残差。该加速器正式覆盖本轮逻辑矩形曲壁通道和平板网格，不能自动推广为任意 O/C-grid 或三维网格已验证。

按最新优先级，先完成上述四级案例验证；512×256（131,072 控制体）为后续可选扩展，未计入本轮正式结果。压力预条件用双线性粗空间；较细曲壁网格的耦合粗网格采用32×16。

## 后续物理参考匹配

公开 [Turbulence Modeling Resource 平板工况](https://tmbwg.github.io/turbmodels/flatplate.html) 使用 Ma=0.2、单位长度 Re=5×10⁶，实际平板长2，因此尾端 Reₓ=10⁷。本轮 Re=10⁵、长1的经验 Cf 对照并未匹配此工况；即使加密后的 Cf 接近经验公式，也不能称 NASA 平板验证完成。下一步需要匹配网格前缘、边界、SA版本及公开局部 Cf/速度数据，明确不可压缩近似的适用性。

## 案例加密与 GPU 的分工

[SA 三网格研究](verification-sa-refinement/report.md)的64×56、128×112由CPU完成，256×224由CUDA完成；最细级不声称双端一致性。未收敛的最细CPU试算及源码在[部分记录](verification-sa-refinement-cpu-partial/README.md)。网格加密首先回答摩阻误差与首层y+的问题，不以设备性能作为验收目标。

```sh
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=src python scripts/verify_sa_mesh_refinement.py
PYTHONPATH=src python scripts/audit_steady_sa.py docs/verification-sa-refinement
PYTHONPATH=src python scripts/audit_sa_refinement.py docs/verification-sa-refinement
PYTHONPATH=src python scripts/audit_simple_acceptance.py docs/verification-sa-refinement
PYTHONPATH=src python scripts/publish_sa_profiles.py docs/verification-sa-refinement
PYTHONPATH=src python scripts/validate_sparse_sa_consistency.py
PYTHONPATH=src python scripts/audit_sparse_sa_consistency.py
```

三网格研究复用仓库内经审计的两级原场；完整从零复现时应先执行完整 SIMPLE/SA 入口，再依部分记录对应源码复算128×112级。各文件哈希和复用来源明确保存。原SA五步配置复核每端只执行一次，保留冷启动时钟，不将其称为重复性能加速比。
