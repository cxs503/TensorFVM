三维与分布式架构说明
======================

目标与边界
----------

三维开发的首个物理目标是 Re_D=3900 外流圆柱，而不是把二维 O 网格简单复制
为“3D”。该问题包含非定常尾迹、跨展向相关性和小尺度耗散；可靠定量结果需要
三维网格、时间统计、网格/时间步独立性以及 DES/LES 或同等分辨能力。

为避免把实验性原型包装成工程软件，代码分为以下层次：

* ``runtime.py``：设备、进程组发现、全局归约及 z-slab 分区；不在导入时初始化
  分布式进程组，兼容 ``torchrun`` 和单进程运行。
* ``mesh3d.py``：三维网格几何和流体掩码，不含任何离散或求解器状态。
* ``solver3d.py``：三维速度、压力、时间推进、SGS 及力积分；结果对象不依赖 CLI。
* ``benchmark_cylinder3d.py``：案例配置、可移植输出和 smoke 验收。
* 既有 ``body_fitted.py``：二维贴体 SIMPLE/SA；它不会被三维原型隐式复用，
  以免把二维假设扩散到三维内核。

当前三维基线
------------

``Cylinder3DSolver`` 实现了一个可执行的结构化三维投影基线：x/y 为开边界，
z 为周期方向，圆柱由笛卡尔固体掩码表示；提供 Smagorinsky 代数 SGS 黏度，
并导出中展向场、历史与压力近似力。单 rank 运行命令为::

    python -m tensorfvm.benchmark_cylinder3d --output results/cylinder-3d

这个案例只验收有限值和低 CFL 的三维推进，**不**验收 Cd、Cl 或 St；阶梯圆柱、
中心差分/固定 Jacobi 压力求解和短时间窗口不能构成 Re=3900 的 LES 验证。

多 GPU / 分布式执行
-------------------

三维内核现在使用沿 z 的真实 slab 分解。每个 rank 只分配自己的
``(local_nz, ny, nx)`` 速度、压力和掩码场；不会重算或常驻整个三维域。
``DistributedRuntime.periodic_z_halos`` 用有标签的点对点通信交换一层周期
halo，并正确覆盖只有两个 rank 时上、下邻居相同的情况。该 halo 被用于：

* 所有展向一阶导数和拉普拉斯模板；
* Smagorinsky 速度梯度；
* 每一次 Jacobi 压力泊松更新，故压力跨分区连续耦合；
* 展向周期首尾 rank 的互相通信。

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
当前环境已以两个 Gloo CPU rank 实际运行 32 x 24 x 4、两步的 smoke case，
也运行了 32 x 24 x 5 的不均衡 3/2 slab case。自动回归通过逐元素容差
``1e-13`` 比较两 rank 重组后的速度/压力与单 rank 两步结果，同时比较每步全局
指标和力历史。这证明当前 stencil/压力耦合的一致性，**不是** GPU 性能或物理
精度验证；本环境没有可用于验证 NCCL/CUDA 的 GPU。

限制与后续验收
--------------

当前压力方程是同步的固定迭代 Jacobi。它在数值上跨分区耦合，但每次迭代都会
交换 halo，因而不适合作为大规模扩展的最终压力算法。它还没有残差驱动的停止
准则、Krylov 预条件或多重网格。下一阶段应按以下顺序推进：

1. 替换为带全局残差的分布式 Krylov/几何多重网格压力求解，并报告迭代数；
2. 使用三维 body-fitted 或 cut-cell 圆柱网格替代阶梯掩码；
3. 采用二阶时间与受限二阶对流，再完成 Re=3900 足够长的统计采样；
4. 对比公开三维数据中的时均 Cd、Cl RMS、St、回流长度、Cp 与速度剖面；
5. 在至少三个网格、两个时间步和单/多 GPU 下发布误差、强扩展和弱扩展报告。
