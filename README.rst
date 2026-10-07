TensorFVM
=========

基于 PyTorch 的二维不可压缩有限体积求解器，采用 SIMPLE
压力－速度耦合。支持层流，以及实验性的 Spalart--Allmaras（SA）一方程
RANS 闭合；网格包括笛卡尔交错网格、圆柱 O 型贴体网格、NACA 翼型 C
型网格及高 Re 平板边界层网格；另含实验性的三维圆柱投影/SGS 基线。
它不是已经验证的工程 CFD 软件。

安装
----

需要 Python 3.10 或以上。仅依赖 PyTorch；测试使用标准库 unittest。
在仓库根目录运行::

    python -m venv .venv
    source .venv/bin/activate
    python -m pip install -e .

仅使用 CPU 时可先从 PyTorch 官方 CPU 索引安装，避免下载 CUDA 运行库::

    python -m pip install "torch>=2.14.1" --index-url https://download.pytorch.org/whl/cpu
    python -m pip install -e .

CUDA 需安装与硬件兼容的 PyTorch，并通过 ``--device cuda`` 选择。
计算使用 float64；不包含可微分求解或训练接口。

圆柱绕流案例
------------

运行默认低雷诺数案例::

    tensorfvm-cylinder --output results/cylinder

或者::

    python -m tensorfvm --mesh-type body-fitted --nx 80 --ny 40 --reynolds 20 --max-iterations 1000

命令行默认采用 ``body-fitted`` 贴体 O 型网格，``nx`` 为周向单元数
（至少 8，且为 4 的倍数），``ny`` 为径向层数（至少 4）。
内边界节点位于圆柱上，外边界严格保持原矩形通道并包含四个角点，
周向接缝周期连接。圆柱边界由节点间的直线面片构成，仍需周向加密
控制圆的多边形近似误差，但不再使用阶梯状固体掩码。
原交错网格可通过 ``--mesh-type cartesian`` 选择，此时 ``nx``、``ny``
仍为 x、y 方向的单元数。

NACA 四位数翼型可使用 C 型网格计算，例如 NACA 0012 基准::

    python -m tensorfvm --mesh-type c-grid --airfoil-code 0012 --reynolds 100 \
        --nx 160 --ny 40 --domain-length 10 --domain-height 8 \
        --airfoil-x 2 --airfoil-y 4 --output results/naca0012

此模式 ``nx`` 为周向单元数（至少 16，且为 4 的倍数），``ny`` 为从翼型
表面到外边界的径向层数。四位数 NACA 外形按解析公式生成，尾缘闭合；
周向接缝从尾缘延伸到下游出口中点并作为内部连接处理。翼型表面无滑移，
径向间距向翼型表面作指数聚集；入口及远场边界指定均匀来流，出口压力为零。
``airfoil-x``、``airfoil-y``
指定前缘和弦线位置，Re 以弦长为特征长度；``angle-of-attack`` 以度为单位，
通过旋转来流方向设置攻角。可用 ``domain-length``、``domain-height`` 和
翼型位置控制外边界距离。

默认计算域为 4 × 2，圆柱中心 (1, 1)、半径 0.2，入口速度 1，
密度 1，Re = 20，以圆柱直径作为特征长度，
动力黏度由 ``mu = rho * U * (2 * radius) / Re`` 确定。
入口给定均匀水平速度；上下壁面及圆柱无滑移；
出口压力为零，速度预测采用零法向梯度，出口通量参与压力修正。
因此这是有壁面约束的绕流，并非无限域圆柱绕流。

命令行提供网格、Re、迭代上限、容差、设备、CPU 线程数和输出目录选项，
可运行 ``tensorfvm-cylinder --help`` 查看。
默认单线程适合小网格；GPU 对小网格未必更快。
返回码 0 表示收敛，2 表示达到迭代上限但已导出结果，1 表示计算或输出失败。
不要将仅满足连续性方程的中间结果视为稳态收敛。

输出
----

输出目录包含以下文件（再次运行会覆盖同名文件）：

* ``fields.csv``：单元中心 x、y、u、v、p 及 fluid 标记；
  fluid 为 0 的固体单元不应参与流场后处理。
* ``history.json``：每次 SIMPLE 迭代的残差记录。
* ``summary.json``：配置、迭代次数、收敛标记及最终残差。
* 贴体模式另含 ``nodes.csv`` 和 ``cells.csv``：节点坐标与四边形连接关系，
  单元编号按 ``fields.csv`` 的行顺序排列；接缝两侧节点坐标重合。
  ``fields.csv`` 使用真实物理单元中心坐标，所有贴体单元均为流体。
* C 型网格另含 ``airfoil.csv``（翼型表面面片坐标、邻接单元压力和压力系数 ``Cp``）；
  ``summary.json`` 包含由离散压力及壁面剪切积分得到的升力、阻力系数。
* 启用 SA 时另含 ``turbulence.csv``（单元中心 ``nu_tilde``、运动学涡黏度
  ``nu_t``），``summary.json`` 记录壁面合力和 SA 残差。
* 物理时间推进的 URANS 结果另含 ``forces.json``：逐时间步的 ``drag``、
  ``lift`` 及物理时间，可用于计算升力 RMS 和 Strouhal 数。
* 贴体 O/C 及平板网格另含 ``velocity.svg``、``pressure.svg``；
  按物理四边形单元绘制速度模和无量纲表压，不需要绘图库。

压力为以出口为参考的表压。结果可以直接用 CSV 工具后处理，
没有额外绘图库依赖，也不使用 pickle 保存结果。

离散与算法
----------

* 笛卡尔模式使用均匀交错网格：压力在单元中心，u、v 分别在 x、y 面上。
* 贴体模式使用四边形有限体积几何、物理坐标下的单元中心速度及压力，
  通过 Rhie–Chow 面通量耦合避免同位网格的压力棋盘格，
  并用最小二乘梯度重构处理非正交修正；
  内部面的通量在相邻单元中符号相反，壁面法向通量为零。
* 有限体积面通量守恒；对流采用一阶迎风，黏性扩散采用中心离散。
* 依次求解欠松弛动量方程、压力修正方程，修正压力、速度及面通量；
  压力修正系数使用欠松弛后的动量对角系数。
  笛卡尔压力修正方程采用矩阵无关、对角预条件共轭梯度求解；
  贴体非正交压力修正及动量方程采用对角预条件 BiCGSTAB。
* 圆柱、翼型、平板及通道壁面采用无滑移条件；C 型网格和平板顶部的远场
  速度固定为来流值。
* SA 模式求解原始完全湍流 Spalart--Allmaras 工作变量；湍流黏度反馈到
  动量扩散，壁面 ``nu_tilde=0``，入口/远场用 ``sa_freestream_ratio`` 指定
  ``nu_tilde / nu``。它使用墙面解析而非壁函数。
* 设置 ``time_step`` 可启用一阶隐式 Euler URANS：每一物理时间步执行
  ``inner_iterations`` 次 SIMPLE 子迭代，速度和 SA 方程保留真实时间惯性；
  它不能与 ``pseudo_time_step`` 同时使用。
* 同时监测连续性和动量收敛；SA 模式还监测输运方程残差；达到迭代上限仍明确报告未收敛。

笛卡尔历史中的 ``continuity`` 为最大单元散度乘以通道高度、除以入口速度；
``momentum`` 为最大稳态离散动量方程缺陷除以对角系数和入口速度。
贴体模式中 ``continuity`` 为最大单元净质量通量除以入口质量流量，
``momentum`` 为两个速度分量中较大的全域动量缺陷绝对值之和，
除以 ``max(rho * U² * Lref, mu * U)``；通道算例 ``Lref=height``，
C 型网格 ``Lref=airfoil_chord``。该标准随加密可能更严格。
``mass_imbalance`` 在圆柱模式下为进出口流量差相对入口流量的绝对值；
C 型网格为所有外边界净质量流量相对 ``rho * U * height`` 的绝对值。
三者均小于配置容差才报告收敛，``velocity_change`` 仅作辅助诊断。
程序接口可设置 ``pseudo_time_step`` 添加伪时间阻尼，但稳态动量残差
不包含伪时间项，避免将小步长造成的小变化误判为稳态解。

笛卡尔模式仍通过单元中心圆形掩码表示阶梯状圆柱；
贴体模式的圆柱为内接多边形，细化周向网格也会改变有效圆柱面积。
一阶迎风会产生数值耗散，贴体网格不等于高阶精度或已验证的工程结果。
当前不包含切割单元、转捩、壁函数、可压缩修正、二阶时间格式、DES 或 LES。
SA 仅是实验性、完全湍流的一方程 RANS 闭合。虽然可用 ``time_step`` 进行
二维 URANS 并记录圆柱升阻力历史，但 Re=3900 尾迹本质上三维；二维 SA-URANS
可能衰减到对称解或错误预测 Cd/St，不能以增加物理步数替代 DES/LES 或三维验证。
定量使用前应进行网格无关性、充分长采样时间及公开基准验证。
NACA C 型网格的压力系数和升阻力仅作数值实验输出；当前层流翼型基准不等于
SA 高 Re 翼型验证，远场位置、近壁 y+、自由流湍流量和网格分辨率会影响结果。

程序接口与测试
--------------

``tensorfvm`` 导出 ``SolverConfig``、``SimpleSolver``、``BodyFittedMesh``,
``CGridMesh`` 和 ``BodyFittedSolver``。程序接口为兼容旧代码，
``SolverConfig`` 默认 ``mesh_type="cartesian"``；设置
``mesh_type="body-fitted"``、``mesh_type="c-grid"`` 或
``mesh_type="flat-plate"`` 后 ``SimpleSolver(config)`` 自动选择贴体/同位求解器；
设置 ``turbulence_model="spalart-allmaras"`` 可启用实验性的 SA RANS 闭合。
修改配置可以设置几何、网格、物性、欠松弛参数及设备。
``SimpleSolver(config).solve()`` 返回压力、速度、流体掩码、
残差历史及收敛状态；``cell_center_velocity()`` 在两种模式下均提供
单元中心速度。贴体结果的 ``u``、``v``、``x``、``y`` 均为
``(ny, nx)`` 数组；笛卡尔结果仍保留交错面速度与一维坐标。
贴体结果通过 ``mesh`` 提供网格几何。
将圆柱半径设置为 0 可计算无障碍通道，此时 Re 的特征长度为通道高度。
无障碍通道仅支持笛卡尔模式；贴体 O 型网格要求正半径且圆柱严格位于通道内部。

解析 benchmark（误差 < 3%）
--------------------------

可运行三档网格的平行板 Poiseuille 验证::

    python -m tensorfvm.benchmark --output results/benchmark

独立检查充分发展截面的速度 L2、最大速度归一化 Linf 误差和压力梯度误差，
仅在所有网格收敛且每项误差严格小于 3% 时返回 0，否则返回 2。
输出包含完整数值场、残差、机器可读误差表、速度对标 CSV、
速度/压力 SVG 云图及中文 ``report.rst``。不增加绘图库依赖。
已运行的说明和计算云图见 `docs/benchmark/report.rst <docs/benchmark/report.rst>`_。
这一验证仅适用于无障碍笛卡尔通道，不代表圆柱、翼型或贴体求解器精度已达标。

圆柱公开基准（DFG 2D-1）
-----------------------

运行 Schäfer–Turek Re=20 圆柱基准，入口采用抛物线速度剖面::

    python -m tensorfvm.benchmark_dfg --output results/dfg

算例为 2.2 × 0.41 通道、直径 0.1 的圆柱，入口截面平均速度 0.2；
在三档 O 型网格上对比公开阻力系数 ``Cd=5.579535``；前两档记录网格
收敛趋势，最细 256×96 网格的误差必须严格小于 3% 且 SIMPLE 收敛才通过。
``--inlet-profile parabolic`` 也可用于直接运行 O 网格算例；入口速度参数表示
截面平均速度。输出包括各网格的场、网格、收敛历史、阻力指标、SVG 云图及
汇总报告。

NACA 0012 公开低雷诺数基准
--------------------------

稳态翼型定量验证采用 Di Ilio 等人（2020，arXiv:2006.10487）的二维层流
NACA 0012、Re=1000、攻角 4° 数据。该攻角低于文献给出的 8° 非稳态起始点，
与本项目稳态模型相容；``Cl=0.205``、``Cd=0.120`` 两项相对误差分别严格
小于 3% 才通过::

    python -m tensorfvm.benchmark_airfoil --output results/airfoil-benchmark

输出包含完整流场、表面 ``Cp``、升阻力、残差、机器可读验收 JSON 和报告。
匹配的 4° 文献数据没有机器可读 Cp 表；文献给出的 Cp 曲线属于 8° 工况，
已处于非稳态起始点，故本稳态验收不伪造 Cp 的 3% 声明。NASA 高 Re 实验
数据也不能直接作为当前层流模型的合格目标。

高 Re 平板 SA RANS 基准（实验性）
---------------------------------

SA 模型以无壁函数、近壁积分方式接入贴体求解器，支持 ``body-fitted``、
``c-grid`` 与专用 ``flat-plate`` 网格。当前首先采用 Re_L=100000 的二维、
零压梯度、从前缘即完全湍流的光滑平板验证；高 Re 圆柱另有下述 URANS
诊断，但两者都不替代三维湍流验证::

    python -m tensorfvm.benchmark_flat_plate --output results/flat-plate

该算例以 Schlichting 平滑完全湍流平板平均摩擦关联式
``Cf = 0.074 Re_L^(-1/5)`` 为参考，要求平均 ``Cf`` 误差严格小于 20%，
第一单元 ``y+ < 1``，并同时收敛连续性、动量、质量平衡和 SA 输运残差。
``turbulence.csv`` 导出 SA 工作变量 ``nu_tilde`` 及运动学涡黏度 ``nu_t``。
这是初始模型回归而不是翼型、圆柱的高 Re 工程验证；尚不包括转捩、壁函数、
可压缩/曲率修正，也尚未完成高 Re 翼型公开数据对标。一次实际运行的指标见
`平板 SA benchmark 报告 <docs/flat-plate/report.rst>`_。

Re=3900 外流圆柱 SA-URANS 诊断
------------------------------

圆柱 O 网格可设置 ``outer_boundary="far-field"``，将上下外边界从通道无滑移
壁面改为均匀来流远场；``body_fitted_stretching`` 用于向圆柱表面集中径向网格。
物理时间推进用隐式 Euler，输出逐步升阻力并从升力频谱计算 Strouhal 数::

    python -m tensorfvm.benchmark_cylinder_urans --output results/cylinder-urans

默认工况为 D=1、Re=3900、20D × 12D 外流域和 SA-URANS。报告比较
``Cd=1.12``、``St=0.20`` 的公开 Re=3900 参考值，并且只有在解析出非零周期
升力、每步内残差满足容差且 Cd/St 达到阈值时才返回 0。若二维一方程 RANS
衰减到对称解，benchmark 会保留 ``forces.json`` 等诊断输出并返回 2；这不是
可通过放宽判据掩盖的失败，而是需要更高阶对流、DES/LES 或三维计算的信号。
当前 48×24、15 个对流时间单位的实际诊断结果已如实记录在
`Re=3900 圆柱 URANS 报告 <docs/cylinder-urans/report.rst>`_。

三维圆柱与多 GPU 投影基线
--------------------------

三维代码以独立的 ``runtime``、``mesh3d``、``solver3d`` 与 benchmark 模块构建，
避免将二维贴体假设隐式复制到三维。``Cylinder3DSolver`` 是可执行的三维笛卡尔
投影/Smagorinsky SGS 基线，展向周期、圆柱固体掩码和中展向可移植输出均已具备。
单 rank 运行方式为::

    python -m tensorfvm.benchmark_cylinder3d --output results/cylinder-3d

三维贴体圆柱 O-grid 已作为独立网格层提供：它从二维 O-grid 横截面挤出正体积
六面体，保留圆柱壁面、矩形远场、theta 接缝、曲面面积向量及按 z-slab 分区的
全局节点/单元编号。可导出网格并审计体积和曲面逼近误差::

    tensorfvm-mesh-cylinder-3d --output results/cylinder-3d-o-grid \
        --nx 48 --ny 24 --nz 12 --stretching 2.5

当前 ``Cylinder3DSolver`` 仍明确只支持 ``mesh_type="cartesian"``；请求贴体
模式会 fail-fast，而不会用笛卡尔差分伪装成曲线坐标求解。贴体网格上的守恒动量、
Rhie--Chow 和压力投影是后续数值阶段，故该命令仅生成/验证网格，不能宣称已求解
贴体三维 Re=3900 流动。

多 rank 使用真实 z-slab 分解：每个 rank 只保留本地速度/压力场；展向导数、
SGS 梯度和分布式压力 PCG 算子都通过一层周期 halo 点对点交换跨分区耦合。
压力求解使用矩阵无关、对角预条件 PCG，出口 gauge 与入口/远场 Neumann 值在
独立未知量中处理；每步输出全局初始/最终压力残差、目标、迭代数和收敛标记。
``pressure_iterations`` 是上限，``pressure_relative_tolerance`` 与
``pressure_absolute_tolerance`` 控制停止目标。以两 GPU 启动的示例为::

    torchrun --standalone --nproc-per-node=2 -m tensorfvm.benchmark_cylinder3d \
        --device cuda --nx 48 --ny 32 --nz 12 --output results/cylinder-3d-2gpu

CPU 可用相同命令配合 ``--device cpu``，使用 Gloo；CUDA 使用 NCCL 和
``LOCAL_RANK`` 设备绑定。全局 ``nz`` 必须不小于 rank 数。输出目录由中展向
所有者写 ``midspan.csv``，rank 0 写全局历史和汇总，因而不会发生并行写冲突。
两 rank Gloo 的自动回归会重组场并与单 rank 比较，且要求两端 PCG 达到残差目标。
3-D smoke case 只在有限值、低 CFL、压力残差收敛和并行数值一致性同时满足时
通过，**不**对 Re=3900 的 Cd、Cl 或 St 声称精度；阶梯圆柱、中心差分和短时间
窗口仍不足以构成 LES/DES 或 GPU 扩展验证。当前 PCG 尚没有多重网格预条件，
故不能将它的可执行性视为大规模性能结论。详见
`三维与分布式架构说明 <docs/architecture/three-dimensional.rst>`_.

测试
----

运行测试::

    python -m unittest discover -s tests -v

测试覆盖输入检查、交错网格及边界、贴体几何与周期接缝、连续性和质量平衡、
无障碍通道及小网格圆柱案例，并检查真实坐标、网格导出和命令行状态。

许可证
------

GNU GPL v3，见 LICENSE。
