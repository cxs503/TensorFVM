联合平台 SI 通道解析验证
======================

该验证复用 TensorFVM 的真实 SIMPLE/MAC 求解器，不以解析场初始化或替换数值结果。
H=0.02 m、L=0.12 m、rho=1000 kg/m³、nu=1e-5 m²/s、U=0.0005 m/s，Re_H=1。
独立参考为 u(y)=6U(y/H)(1-y/H)、dp/dx=-0.15 Pa/m、壁面剪切幅值 0.0015 Pa。
入口速度均匀，出口表压为零；上下壁半单元无滑移，因此只在 x≈5H 对比速度，
在 3H≤x≤5H 对比压力斜率、壁面力和剪切，不把入口发展区当充分发展流动。

实际结果
--------

.. list-table:: 实际三网格结果，误差单位为百分比
   :header-rows: 1

   * - 网格
     - 迭代
     - 速度 L2
     - 压力梯度
     - 壁面剪切 Linf
   * - 36×12
     - 41
     - 0.634358
     - 1.369853
     - 1.369855
   * - 72×24
     - 87
     - 0.161971
     - 0.344019
     - 0.344020
   * - 144×48
     - 282
     - 0.042848
     - 0.078365
     - 0.078369

三网格均通过速度、压力梯度及剪切误差 <3% 和原求解器残差门槛。
入口质量流量为 0.01 kg/(s·m)，最细网格出口相对不平衡 3.47e-16。
充分发展区域压力推力和壁面对流体阻力的相对差为 3.37e-8；这些力均按单位展向宽度报告。
剪切采用求解器自身半单元壁面扩散离散，不是更高阶壁面梯度重构。
这只验证静态通道；不能推导一般流固载荷映射、自由液面或移动碎冰能力已验证。

数据与复现
----------

在仓库根目录执行::

    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=src python -m tensorfvm.benchmark_suite_channel
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python scripts/compare_lbm_channel_reference.py --lbm-repo /path/to/TensorLBM

benchmark.json 保存配置、误差和字段 SHA256。
每个网格的 si-fields.json 保存实际中心速度、压力、MAC 面速度及上下壁面牵引；
fields.csv、history.json、summary.json 保留原求解器导出。
场数组按 [y][x]，SI 单位；壁面力为壁面对流体施加的 x 向力，沿 +x 流动时为负。
最大单元散度根据保存的 MAC 面速度计算。未收敛案例仍输出字段，并返回不达标状态。

LBM 独立参考的边界差异
----------------------

脚本直接调用未修改的 TensorLBM benchmarks/verified/poiseuille_2d/run.py。
BGK tau=0.8，Re_H=1，初始静止，压力入口/出口与半程反弹；该原有案例 L/H=3，
FVM 则 L/H=6、均匀速度入口。映射 dx=H_SI/H_LB、dt=nu_LB dx²/nu_SI，
压力尺度 rho_SI (dx/dt)²/3，保留实际测量斜率，不校正其 nx 与 nx-1 长度差。
原 runner 只输出剖面与压力诊断，故 LBM 不声称保留完整二维流场。
两者分别和共同解析解比较，并非完全相同边界问题，更非联合 FSI 对照。
当前 TensorFVM 不被登记为移动冰体的合格后端；其作用是独立流体基础交叉核验。

实际 LBM 独立参考结果（未隐藏失败）
---------------------------------

H_LB=12：4000 步，原 runner 判稳；速度 L2 误差 2.391864%，实测压力梯度误差 2.856227%。
H_LB=24：23600 步，原 runner 判稳；速度 L2 误差 1.293972%，实测压力梯度误差 1.407548%。
H_LB=48：30000 步，速度 L2 误差 0.735145%，压力梯度误差 0.669594%，
**未满足原 runner 的稳态判据，保留为未通过**；不能仅依据低解析误差认证收敛。
因此独立 LBM 对照总体未通过，FVM 通道验证通过，二者状态分别报告。
LBM 为原脚本 float32，与 FVM float64 不同；数值精度、边界和长度定义均影响差异。
此组证据不用于排名求解器，也不用于认证 LBM 的其他通道或联合破冰求解器。
原始记录在 lbm-reference/H*-raw.json，映射结果与版本在 comparison.json。

原始字段审计::

    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=src python scripts/validate_suite_channel.py

审计无需重新求解，重建中心速度、压力斜率、MAC 连续性、壁面剪切与力，
并核对 FVM 发布字段 SHA256。12 项相关测试已通过。
