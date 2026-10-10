# TensorFVM 新增解析 benchmark：三维 ABC 涡与对流剪切波

## 1. 问题介绍与解析依据

域为 [0,2π]³，周期边界，ρ=1 kg/m³，μ=0.1 Pa·s，双精度。没有固体、自由液面或外力。

**ABC / Beltrami 涡：** u₀=(sin z+cos y, sin x+cos z, sin y+cos x)，u=u₀ exp(−νt)，p=−ρ|u|²/2+常数。
解析依据可直接代入不可压 Navier–Stokes：div u₀=0，curl u₀=u₀，Δu₀=−u₀；
(u·∇)u=∇(|u|²/2)−u×curl u，因此非线性项由压力梯度抵消，剩余黏性指数衰减。
所有分量非零，场依赖 x/y/z，是真正三维周期流，但不是湍流统计、圆柱或工程绕流。
正式 n=24/32/48，Δt=0.005 s，t=0.1 s；n=16 保留为压力不达标负控制。

**对流剪切波：** u=sin(y−0.7t) exp(−νt)，v=0.7，w=0，p=0。
直接代入得 ∂t u+0.7∂y u=ν∂yy u，验证输运相位、衰减和非零平均动量。
正式 n=16/24/32，z=4 层，Δt=0.01 s，t=0.5 s；该案例是二维场嵌入三维网格。

## 2. 软件使用与离散方法

使用现有 PeriodicMACSolver：面动量共享中心通量、完整对称应力、隐式 midpoint/Picard、相容周期 FFT 压力投影。
速度参考取各分量真实面坐标和末时刻；ABC 压力取 cell 坐标、最后 midpoint 时刻 t−Δt/2，均值规范为零。
所有声明物理误差严格 <3%，另检查全部接受步的连续性、真实非线性残差、动量与能量账本。
三网格观察阶要求 1.8–2.2。每步原始状态保存并由独立 NumPy 算术重建。

复现（仓库根目录）：

~~~bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=src python -m tensorfvm.verification.extended --output results/extended --lbm-repo ../TensorLBM
~~~

## 3. TensorLBM 对照与误差定义

剪切波对照使用外部 TensorLBM 原生产 D2Q9/BGK 平衡、碰撞、迁移函数：32²/64²，目标初始 Mach=0.05，整数步精确达到 t=0.5。
使用同一物理 ν、平均速度及周期长度。初始化为解析速度的平衡 populations，没有质量补偿。
LBM 保留原 float32 格式权重常数，populations 为 float64；初始化缺少非平衡应力可能贡献早期误差。
D2Q9 无三维能力，因此没有将 ABC 与二维 LBM 冒充同条件比较。

剪切波解析压力恒零，不能计算相对零参考误差；使用 max|p|/[0.5ρ(1²+0.7²)] 的绝对压力动态尺度误差，要求 <3%。
速度同时检查全部分量的 L2/L∞ 与单独剪切分量 L2，防止平均流掩盖输运误差。
LBM 速度取实际节点，FVM 速度取实际面，各自与解析值比较；压力由 EOS 的均值规范值导出。
单次 CPU 用时含设置、诊断和原场采样，不推论普遍速度优势；四层 FVM 与二维 LBM 的总内存也不能直接比较。

## 4. 计算结果

| Case | Grid | Role | Velocity L2 % | Pressure metric % | Passed |
|---|---|---|---:|---:|---|
| abc | 24x24x24 | spatial | 0.005699 | 1.692779 | True |
| abc | 32x32x32 | spatial | 0.003209 | 0.954533 | True |
| abc | 48x48x48 | spatial | 0.001427 | 0.424980 | True |
| advected-shear | 16x16x4 | spatial | 0.620363 | 0.000000 | True |
| advected-shear | 24x24x4 | spatial | 0.276903 | 0.000000 | True |
| advected-shear | 32x32x4 | spatial | 0.156024 | 0.000000 | True |
| abc | 16x16x16 | negative-control | 0.012786 | 3.782032 | False |
| advected-shear-lbm | 32x32 | comparison | 0.230176 | 0.000000 | True |
| advected-shear-lbm | 64x64 | comparison | 0.054381 | 0.000000 | True |

观察阶：

- abc-velocity_l2: 1.996579, 1.998281
- abc-pressure_l2: 1.991451, 1.995685
- advected-shear-shear_velocity_l2: 1.989409, 1.994080

FVM 正式案例资格：True；LBM 对照精度：True。

同为 n=32 的剪切分量 L2 误差：FVM 0.225186%，BGK 0.332209%。这组实测值反映本工况下的相位与衰减误差，不能推广为所有流动的精度排名。

## 5. 云图、曲线与证据

### abc：32x32x32

![pressure](abc-32/figures/pressure.png)

![speed](abc-32/figures/speed.png)

![velocity-profile](abc-32/figures/velocity-profile.png)

![pressure-curve](abc-32/figures/pressure-curve.png)

### abc：48x48x48

![pressure](abc-48/figures/pressure.png)

![speed](abc-48/figures/speed.png)

![velocity-profile](abc-48/figures/velocity-profile.png)

![pressure-curve](abc-48/figures/pressure-curve.png)

### advected-shear：32x32x4

![pressure](shear-32/figures/pressure.png)

![speed](shear-32/figures/speed.png)

![velocity-profile](shear-32/figures/velocity-profile.png)

![pressure-curve](shear-32/figures/pressure-curve.png)

### advected-shear-lbm：32x32

![pressure](lbm-shear-32/figures/pressure.png)

![speed](lbm-shear-32/figures/speed.png)

![velocity-profile](lbm-shear-32/figures/velocity-profile.png)

![pressure-curve](lbm-shear-32/figures/pressure-curve.png)

## 6. 讨论、限制与复现材料

本轮扩展验证了三维周期非线性压力平衡以及非零平均流的输运，资格限定于这些参数与网格。
各目录包含完整三维/二维原场 NPZ、逐步 history、压力/速度曲线 CSV 和 300dpi PNG/PDF/SVG。
原始采样时间、面/节点布局、零压力归一化和负控制均明确保存。audit.json 由独立 NumPy 重建每个接受步。
manifest.json 绑定所有原场、报告及当前源码。旧 Poiseuille/Taylor–Green 报告与源文件保持其原始证据版本。
DFG 圆柱和翼型的旧 runner 不因本轮周期案例通过而获得认证；SUBOFF/破冰仍需真实固壁、移动边界与自由液面验证。

本文件与 report.pdf 是可复现研究稿，尚未完成期刊投稿或同行评审。
