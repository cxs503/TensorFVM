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

延长最细 LBM 参考及证据审计
--------------------------

本轮新增 lbm-extended-reference；原 lbm-reference 的 30000 步失败记录保留。
采用同一个未修改的 LBM runner，把 H=48 上限延长至 120000 步，从静止重新求解，
仍要求原判据：每 200 步的最大速度，最近十个采样的相对极差 <1e-5。
解析误差门槛仍为速度 L2 和压力梯度各 <3%；不会仅因误差小而判为稳态。
原 runner 在主循环结束后额外平均 200 步，该阶段不计入 n_steps。

复现::

    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python scripts/compare_lbm_channel_reference.py \
        --lbm-repo /path/to/TensorLBM --heights 48 --max-steps 120000 \
        --output docs/suite-channel/lbm-extended-reference
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python scripts/validate_lbm_channel_reference.py

comparison.json 保存原始 JSON 的 SHA256 和五个 LBM 求解依赖文件的 SHA256。
独立审计重建 SI 速度剖面、压力梯度、尺度、误差和通过决定；篡改原始数据或 SI 摘要会被拒绝。
原 runner 不导出其稳态采样历史，审计只能核对保存的 steady 布尔值和组合判定，
不能单凭最终剖面独立证实其时间漂移判据，亦不声称审核完整二维 LBM 流场。

同边界与精度对照仍未完成：该 LBM runner 将初始化密度明确设为 float32，
固定压力入口/出口、L/H=3；FVM 使用 float64、均匀速度入口和 L/H=6。
本轮只延长运行，不修改默认精度或入口，不把两组输出当成同一边界值问题。
未来需要单独开发边界一致的案例并保存完整场和稳态历史，而不是修改已发布结果的物理解释。

120000 步的实际 H48 原 float32 结果仍未判稳：速度 L2 误差 0.738202%，
压力梯度误差 0.695203%，steady=false，主循环耗时 178.4 s。
新增哈希与 SI 重建审计通过，但求解参考本身仍未通过；两种状态分别报告。

float64 隔离诊断
----------------

新增脚本通过 AST 提取同一上游 run_case，仅将密度初始化和零数组的 dtype
显式改为 float64，并导出真实每 200 步最大速度历史、全精度平均速度剖面、
最终二维 rho/ux/uy。碰撞、传播、压力边界、几何、Re、静止初态及判稳门槛不变。
变换后的函数和脚本各有 SHA256；该证据是新的精度诊断，不能替换原 float32 失败记录。
该适配保留上游源码不变，不影响 TensorLBM 项目中的既有算例。

复现与审计::

    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python scripts/run_lbm_float64_diagnostic.py \
        --lbm-repo /path/to/TensorLBM
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python scripts/validate_lbm_float64_diagnostic.py

新审计重建每个实际十点时间窗口，核对首次判稳的位置、最终漂移、原始场的有限性与形状、
中高处密度剖面和压力斜率及 SI 指标。速度剖面是最后额外 200 步的平均，
因此不会宣称它可由单个最终二维速度场精确重建。
LBM 压力边界与 FVM 速度边界的差异依然存在；这项实验只诊断精度和稳态采样。

实际 float64 结果：H24 在 8600 步首次满足原门槛，末窗口漂移 8.45857e-6，
速度 L2 误差 1.288946%，压力梯度误差 1.408451%。H48 在 27400 步满足，
漂移 9.55155e-6，速度 L2 误差 0.665727%，压力梯度误差 0.699301%。
两组均通过原稳态与 <3% 解析误差门槛。相同离散算法提高存储精度后判稳，
与 float32 H48 在 120000 步仍未判稳相比，支持精度对漂移监测有显著影响。
这是诊断推断：原 float32 runner 未输出漂移序列，本轮没有据此宣称已量出其噪声频谱或噪声下限。
完整监测与宏观场审计也已通过；新判稳记录不能把原 float32 记录改写为通过。
