三维与分布式架构说明
======================

当前正确性阻塞（2026-10-09）
---------------------------

定向审查发现压力泊松七点算子与中心差分散度/梯度的组合不一致。
PCG 残差收敛不能证明投影后连续性满足：真实 CPU/CUDA 单步连续性
分别约 0.892/0.580（无量纲 max|div u| D/U）。现有 smoke 只检查
末步有限值、CFL 和 PCG，不检查连续性或全部历史。三维物理资格仍未通过，
首要任务应是兼容面通量投影和每步连续性验收，之后再做性能扩展。
详见 ``../current-status-2026-10-09.md``。

目标与边界
----------

三维开发的首个物理目标是 Re_D=3900 外流圆柱，而不是把二维 O 网格简单复制
为“3D”。该问题包含非定常尾迹、跨展向相关性和小尺度耗散；可靠定量结果需要
三维网格、时间统计、网格/时间步独立性以及 DES/LES 或同等分辨能力。

为避免把实验性原型包装成工程软件，代码分为以下层次：

* ``runtime.py``：设备、进程组发现、全局归约及 z-slab 分区；不在导入时初始化
  分布式进程组，兼容 ``torchrun`` 和单进程运行。
* ``mesh3d.py``：笛卡尔三维网格几何和流体掩码，不含任何离散或求解器状态。
* ``body_fitted3d.py``：三维圆柱 O-grid 的局部 z-slab 几何、六面体连接和曲面度量；
  不包含离散或求解器状态。
* ``solver3d.py``：当前笛卡尔三维速度、压力、时间推进、SGS 及力积分；结果对象
  不依赖 CLI。
* ``benchmark_cylinder3d.py``：笛卡尔案例配置、可移植输出和 smoke 验收。
* ``mesh_cylinder3d.py``：贴体 O-grid 的质量审计和可移植节点/单元/壁面导出。
* 既有 ``body_fitted.py``：二维贴体 SIMPLE/SA；它不会被三维原型隐式复用，
  以免把二维假设扩散到三维内核。

当前三维基线
------------

``Cylinder3DSolver`` 实现了一个可执行的结构化三维投影基线：x/y 为开边界，
z 为周期方向，圆柱由笛卡尔固体掩码表示；提供 Smagorinsky 代数 SGS 黏度，
并导出中展向场、历史与压力近似力。单 rank 运行命令为::

    python -m tensorfvm.benchmark_cylinder3d --output results/cylinder-3d

这个案例只验收有限值、低 CFL 和压力线性残差收敛的三维推进，**不**验收 Cd、
Cl 或 St；阶梯圆柱、中心差分和短时间窗口不能构成 Re=3900 的 LES 验证。

三维贴体 O-grid：几何与拓扑阶段
--------------------------------

``BodyFittedCylinderMesh3D`` 已实现独立的三维圆柱贴体网格层。它复用经过验证的
二维圆柱 O-grid 横截面：内边界是圆柱多边形，外边界严格落在矩形远场域；随后沿
z 线性挤出。网格提供每个本地 z slab 的：

* ``(local_nz + 1, ny + 1, nx + 1)`` 节点坐标及 theta 周期接缝；
* 正体积六面体连接、单元中心和精确的“横截面面积乘 dz”体积；
* 所有横向面、面中心、面面积向量、owner/neighbor 拓扑和边界标签；
* z 平面中心/面积，以及圆柱曲面 quad、曲面面积向量和全局节点编号所需的局部
  连接关系。

径向方向可通过 ``body_fitted_stretching`` 向壁面指数聚集。网格生成器可以单进程
或经 ``torchrun`` 按当前 z-slab 分区构建，并输出单 rank 的 ``nodes.csv``、
``cells.csv``、``cylinder-faces.csv``；多 rank 时输出带 rank 后缀的分片文件，
且 z 接口节点与 cell id 使用一致的全局编号。例如::

    tensorfvm-mesh-cylinder-3d --output results/cylinder-3d-o-grid \
        --nx 48 --ny 24 --nz 12 --stretching 2.5

当前环境已实际生成 32 x 8 x 6 网格：所有 1536 个六面体为正体积，体积守恒误差
约 ``1.6e-16``，theta 接缝误差为零；多 rank 的 3/3 z slab 导出也已运行。
32 个周向面得到的多边形圆柱曲面面积相对光滑圆柱的误差为约 ``0.002``，这是
几何逼近指标，而不是流动误差。

**数值内核边界：** ``Cylinder3DSolver`` 仍只接受 ``mesh_type="cartesian"``。
若请求 ``mesh_type="body-fitted"``，它会显式拒绝，而不是把贴体坐标送入笛卡尔
中心差分算子。这一阶段交付的是可审计、可分区的贴体几何和拓扑；其上的
守恒曲线坐标动量、Rhie--Chow 面通量、压力投影和分布式 PCG 算子仍需作为独立
的下一数值阶段实现。

多 GPU / 分布式执行
-------------------

三维内核现在使用沿 z 的真实 slab 分解。每个 rank 只分配自己的
``(local_nz, ny, nx)`` 速度、压力和掩码场；不会重算或常驻整个三维域。
``DistributedRuntime.periodic_z_halos`` 用有标签的点对点通信交换一层周期
halo，并正确覆盖只有两个 rank 时上、下邻居相同的情况。该 halo 被用于：

* 所有展向一阶导数和拉普拉斯模板；
* Smagorinsky 速度梯度；
* 每一次矩阵无关 PCG 压力算子应用，故压力跨分区连续耦合；
* 展向周期首尾 rank 的互相通信。

压力泊松求解器
--------------

固定次数 Jacobi 已替换为矩阵无关、对角预条件的共轭梯度（PCG）。出口零压提供
gauge；入口与 y 远场的齐次 Neumann 值在 Krylov 未知量中消元，使独立流体单元的
``-Laplacian`` 算子保持对称。每个 PCG 算子应用进行一次 z halo 交换；内积和
残差范数使用全局 FP64 sum 归约。配置中的 ``pressure_iterations`` 是最大迭代数，
``pressure_relative_tolerance`` 与 ``pressure_absolute_tolerance`` 共同定义停止
目标。每个物理步的初始残差、最终残差、目标、迭代数和收敛标记写入
``history.json``；若最大迭代数后未收敛，信息会保留，3-D smoke benchmark 返回
非通过状态，不能将固定迭代次数伪装成压力收敛。

连续性、CFL、速度变化和 SGS 黏度使用全局 max；圆柱压力力使用全局 sum。
输出时拥有全局中展向平面的 rank 写 ``midspan.csv``，rank 0 在同步后写入
全局规约的 ``history.json``、``forces.json`` 和 ``summary.json``，避免多 rank
竞争同一输出文件。

``discover`` 仍然没有副作用，方便库调用者自行管理进程组；可执行 benchmark
在检测到 ``torchrun`` 的多 rank 环境时显式调用
``DistributedRuntime.initialize_from_environment``。CPU 使用 Gloo，CUDA 使用
NCCL 并绑定 ``LOCAL_RANK``。例如两 GPU 运行方式为::

    torchrun --standalone --nproc-per-node=2 -m tensorfvm.benchmark_cylinder3d \
        --device cuda --nx 48 --ny 32 --nz 12 --output results/cylinder-3d-2gpu

全局 ``nz`` 必须不小于进程数；不均衡余数由 ``partition_slab`` 分配到低 rank。
当前环境已以两个 Gloo CPU rank 实际运行默认的 48 x 32 x 12、50 步 PCG
smoke case：末步使用 141 次 PCG 迭代，残差 ``1.9205e-08`` 小于目标
``2.4160e-08``；也运行了 32 x 24 x 5 的不均衡 3/2 slab case，以及三个 rank 的
32 x 24 x 6、两步 case。自动回归在两 rank 和单 rank 间比较重组速度（绝对容差
``1e-13``）、压力（``1e-12``）、全局诊断和
力历史；它还要求两端的 PCG 都达到全局残差目标。浮点规约顺序不同会造成约
机器精度的残差差异，故标量按 ``2e-12`` 相对/绝对容差比较。这证明单/多 rank 的重组与通信结果一致，**不是** 离散 D/G 与压力算子的
兼容性或物理精度验证。2026-10-09 环境已具备一张 RTX 3090，单卡实际一步
可执行，但连续性仍未达标；NCCL 多 GPU 和性能扩展尚未验证。

限制与后续验收
--------------

当前 PCG 的对角预条件仅是可靠的第一层线性求解基础：每个 Krylov 迭代仍需
一次 halo 交换和若干全局规约，在大规模 GPU 上会受延迟和条件数限制。它没有
代数/几何多重网格、粗网格通信、重叠或通信-计算重叠，也还未在 NCCL 硬件上
性能验证。下一阶段应按以下顺序推进：

1. 为 PCG 添加分布式几何多重网格或等效可扩展预条件，报告迭代数与残差；
2. 在已有贴体 O-grid 上实现守恒曲线坐标动量、Rhie--Chow 面通量、压力投影和
   z-slab halo，而不是复用笛卡尔中心差分；
3. 采用二阶时间与受限二阶对流，再完成 Re=3900 足够长的统计采样；
4. 对比公开三维数据中的时均 Cd、Cl RMS、St、回流长度、Cp 与速度剖面；
5. 在至少三个网格、两个时间步和单/多 GPU 下发布误差、强扩展和弱扩展报告。
