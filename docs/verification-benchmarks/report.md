# TensorFVM 不可压有限体积方法的解析基准验证
## 摘要
本研究通过共性 benchmark 模块，对平行板 Poiseuille 流和二维 Taylor–Green 衰减涡开展空间及时间加密。所有正式算例的预声明速度、压力与壁面相关误差均要求严格小于 3%，并分别满足质量、求解残差和适用的能量门。总体验收：**通过**。失败对照单独列出并保留原始数据。本文验证解析子问题，不能外推三维湍流、复杂曲面或上浮破冰。

## 1. 问题介绍与数学模型
常密度不可压方程为 ∇·u=0，ρ(∂u/∂t+∇·(u⊗u))=−∇p+∇·[μ(∇u+∇uᵀ)]。本研究无体力。全部单位为 SI，二维力/流量按单位展向深度计算。

### 1.1 Poiseuille 通道
L=0.12 m，H=0.02 m，ρ=1000 kg/m³，ν=10⁻⁵ m²/s，Ub=5×10⁻⁴ m/s，Re_H=1。入口规定均匀速度，壁面无滑移，出口零表压。均匀入口存在发展段；解析验证只在 3H≤x≤5H 的压力/壁面窗口，以及 x 最接近 5H 的速度剖面进行。全域云图展示实际解，入口区没有被当作充分发展解析解。
解析解：u(y)=6Ub(y/H)(1−y/H)，dp/dx=−12μUb/H²，|τw|=6μUb/H。压力曲线只在比较窗内对齐均值，不把任意表压零点计入误差。

### 1.2 Taylor–Green 衰减涡
周期域 [0,2π]³ m，ρ=1 kg/m³，μ=0.1 Pa·s，U0=1 m/s，波数 k=1 m⁻¹。二维解析涡嵌入四层展向三维网格，w=0；不是三维湍流。
u=sin(x)cos(y)e^(−2νt)，v=−cos(x)sin(y)e^(−2νt)，p=ρ[cos(2x)+cos(2y)]e^(−4νt)/4。
速度比较位于实际错位面、t=0.5 s。投影压力是末步 midpoint 压力，比较时间为 t−dt/2，压力采用零均值规范。云图中的中心速度仅用于展示，验收使用权威面速度。

## 2. 软件与共性模块
案例适配器提供配置、求解器、参考公式和原始场；共性 core 模块统一误差、验收、导出、来源绑定与观察阶；report 模块统一云图、曲线、报告；audit 模块独立读取原场重算指标。没有复制流体求解器，也没有把解析解作为计算结果输出。

安装：python -m pip install -e '.[benchmark,dev]'
运行：OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m tensorfvm.verification --output docs/verification-benchmarks
审计：python -m tensorfvm.verification.audit --directory docs/verification-benchmarks
失败返回非零退出码；--max-channel-iterations 1 可运行非收敛检查。输出目录须为空，防止覆盖既有失败证据。

## 3. 数值方法与实验设计
通道采用交错有限体积 SIMPLE、半单元无滑移黏性壁面离散，求解 tolerance=10⁻⁶，最多1000迭代。三网格36×12、72×24、144×48。空间误差以解析速度、压力梯度和压力曲线、壁面剪切及力平衡独立评估。
周期 MAC 采用权威面速度、共享中心动量通量、完整对称黏性应力、离散相容 FFT 投影和隐式 midpoint；非线性真实残差约10⁻¹² m/s。空间32²/48²/64²×4、dt=0.0125 s；时间32²×4、dt=0.1/0.05/0.025 s，全部至t=0.5 s。时间观察阶使用固定网格离散特征值的精确指数参考，避免混合空间误差。该参考不作为连续物理误差。
误差采用 ||q−q_ref||₂/||q_ref||₂ 与峰值归一化 L∞；无逐点除以参考零值。观察阶 p=ln(e₁/e₂)/ln(h₁/h₂)，来自实际解析误差，不使用假定阶或两网格 GCI 宣称验证完成。
正式算例全部需通过，失败对照不计入正例资格。非收敛对照与16²粗网格压力失败仅用于证明门能拒绝错误，绝不改成通过。

## 4. 计算结果
| Case | Study | Grid | dt (s) | Velocity L2 (%) | Pressure metric (%) | All gates |
|---|---|---|---|---:|---:|---|
| poiseuille | spatial | 36x12 | — | 0.634358 | 1.369853 | PASS |
| poiseuille | spatial | 72x24 | — | 0.161971 | 0.344019 | PASS |
| poiseuille | spatial | 144x48 | — | 0.042848 | 0.078365 | PASS |
| taylor-green | spatial | 32x32x4 | 0.0125 | 0.032086 | 0.897811 | PASS |
| taylor-green | spatial | 48x48x4 | 0.0125 | 0.014267 | 0.399544 | PASS |
| taylor-green | spatial | 64x64x4 | 0.0125 | 0.008024 | 0.224797 | PASS |
| taylor-green | temporal | 32x32x4 | 0.1 | 0.031761 | 0.894261 | PASS |
| taylor-green | temporal | 32x32x4 | 0.05 | 0.032009 | 0.898033 | PASS |
| taylor-green | temporal | 32x32x4 | 0.025 | 0.032071 | 0.898174 | PASS |
| poiseuille | negative-control | 36x12 | — | 25.356201 | 74.139719 | FAIL |
| taylor-green | negative-control | 16x16x4 | 0.0125 | 0.127928 | 3.562683 | FAIL |
| taylor-green-lbm | comparison | 32x32 | 0.0056179775280898875 | 0.633476 | 0.861731 | PASS |
| taylor-green-lbm | comparison | 64x64 | 0.002824858757062147 | 0.152073 | 0.814284 | PASS |
| taylor-green-lbm | comparison | 32x32 | 0.002824858757062147 | 0.640030 | 1.683112 | PASS |

空间与时间观察阶：{"channel_velocity": [1.969562587297749, 1.9184359357168919], "channel_pressure_gradient": [1.9934624306292263, 2.134205788719911], "tg_velocity": [1.9989589105828733, 2.000226557168189], "tg_pressure": [1.9968077082851627, 1.999166343016587], "tg_time": [2.0000627255879495, 2.000015610195456]}。
每个 result.json 包含全部门槛、指标、时间和计算配置；history.json/CSV 保存实际迭代/步历史；fields.npz 保存原始面场、中心展示场、参考与采样曲线，Taylor–Green 还保存全部接受步面场。

## 5. 压力场、速度场与剖面
周期涡压力曲线取中心平面 y=(floor(n/2)+1/2)Δy；速度剖面取u面 x=(floor(n/4)+1)Δx。通道剖面位置见每个reference记录。

### poiseuille，144x48

![实际压力场，单位 Pa](channel-48/figures/pressure.png)

图：实际压力场，单位 Pa。

![实际速度幅值，单位 m/s](channel-48/figures/speed.png)

图：实际速度幅值，单位 m/s。

![实际水平速度](channel-48/figures/u.png)

图：实际水平速度。

![实际垂向速度](channel-48/figures/v.png)

图：实际垂向速度。

![压力曲线与解析参考；使用上述区域、规范和时钟](channel-48/figures/pressure-curve.png)

图：压力曲线与解析参考；使用上述区域、规范和时钟。

![速度剖面与解析参考](channel-48/figures/velocity-profile.png)

图：速度剖面与解析参考。

![实际收敛或能量历史](channel-48/figures/history.png)

图：实际收敛或能量历史。

### taylor-green，64x64x4

![实际压力场，单位 Pa](tg-space-64/figures/pressure.png)

图：实际压力场，单位 Pa。

![实际速度幅值，单位 m/s](tg-space-64/figures/speed.png)

图：实际速度幅值，单位 m/s。

![实际水平速度](tg-space-64/figures/u.png)

图：实际水平速度。

![实际垂向速度](tg-space-64/figures/v.png)

图：实际垂向速度。

![压力曲线与解析参考；使用上述区域、规范和时钟](tg-space-64/figures/pressure-curve.png)

图：压力曲线与解析参考；使用上述区域、规范和时钟。

![速度剖面与解析参考](tg-space-64/figures/velocity-profile.png)

图：速度剖面与解析参考。

![实际收敛或能量历史](tg-space-64/figures/history.png)

图：实际收敛或能量历史。

![poiseuille-spatial](figures/poiseuille-spatial.png)

![taylor-green-spatial](figures/taylor-green-spatial.png)

![taylor-green-temporal](figures/taylor-green-temporal.png)

## 6. 讨论与适用边界
3% 是本项目预声明的解析验证门，并非论文、行业或工程安全的通用标准。多项解析误差与数值门均通过后，只认证本文的边界条件和评估区域。通道入口不是充分发展解；周期涡无固壁、自由液面和移动体。二维涡嵌入3D不能验证真实3D湍流。本研究不认证翼型/圆柱升阻力、SUBOFF、双向FSI或冰破坏。网格加密使用相同物理时间；未将小残差等同于物理误差。
另一次128²×4、dt=0.0125 s试验因Picard候选非有限被拒绝，完整失败场在相邻 verification-benchmarks-failed-n128 目录保留。它不计入正式32/48/64序列，不能被该序列通过覆盖。
历史根目录 tensorlbm 实验代码没有作为本报告求解后端；其投影尺度、裁剪和守恒问题仍须独立处理。此工作没有放宽旧失败门或覆盖旧证据。

## 7. 可复现性与数据可用性
源码 SHA256、git 基线、Python/Torch/NumPy、设备与线程见 summary.json；全部报告、图和数据摘要见 manifest.json。检查源码哈希与文件哈希后，再从原始场独立重算验收。图以300 dpi PNG、矢量PDF及SVG提供；压力/速度曲线同时导出CSV。报告PDF、可打印HTML及LaTeX稿可供论文编排；投稿前仍需期刊模板、作者信息和同行审查。
环境：{"matplotlib": "3.11.2", "git_base": "38c48dfe386ac822f4f70e44726fa0a610c753af", "python": "3.11.15", "torch": "2.14.1+cu130", "numpy": "2.4.6", "platform": "Linux-6.8.0-137-generic-x86_64-with-glibc2.39", "torch_threads": 1, "device": "cpu", "source_sha256": {"pyproject.toml": "7222720c67aacd63d49197202dea8e1a457b8b78a53a8cba31640846a5498172", "src/tensorfvm/__init__.py": "9f8d4d3a249d695e4e841e4c9d11d7158db8a2d33e2928d3565794718e2aba3c", "src/tensorfvm/backend_registry.py": "e68b95fdb7c457b921b2f1c9e3920f80ebdcdb104ca8077e10e96d2a4182a13a", "src/tensorfvm/body_fitted.py": "4e633fa42c1171cf57aca5bf930827f97a30dada4e405bf3ac9c1068ba1bb7a7", "src/tensorfvm/mesh_api.py": "72f4f9d724084df335dca0e561877c5423c47a7f8bfffe950d631845e2dce22d", "src/tensorfvm/periodic_mac.py": "b6b823c2c7972368a30fbba14f07d1222d062d00a9fe03c769c844407bfde5eb", "src/tensorfvm/runtime.py": "ff383730f34f82030f28da424819eea670f28c663d5be834f30bebc6c74a2da5", "src/tensorfvm/solver.py": "7c3613f23791c6a22825bd2ba0467e2bb84013ecaedc974901452b1864a2868e", "src/tensorfvm/verification/__init__.py": "2e93edad1f2e0ed881602c2f72592f32070e99f9a75433e6fa0ce4c485f52652", "src/tensorfvm/verification/__main__.py": "bac1da0b8ea13da1d7e8de391d0da9fe5a69f8658d55c57d55fc9c861eb28096", "src/tensorfvm/verification/audit.py": "7df05206579b07718441e0ae894e60d8a7b2543fd8e557bd1f7fb7d8586ea414", "src/tensorfvm/verification/cases.py": "6cf52e2cf76334ae3ce47c77c9d4bd33e7d83ab08f2d96b2123d356b0702260e", "src/tensorfvm/verification/core.py": "088114b4c8ec39a0724213bfc00f83a4356bcaca1dd2398376ef35426866e12e", "src/tensorfvm/verification/lbm.py": "0ad5bf5d1022752755f5f0ff47df597baf00dca3e64bf66e576014e6930be9a4", "src/tensorfvm/verification/report.py": "d38464cbba0f152d4410589e6a66c8fe05f9f84da06ecab62d3895a8f77ead62"}}。

## 8. 参考依据
解析参考由本文控制方程直接推导：通道积分 μu''=dp/dx 且满足无滑移与平均流量；周期涡代入不可压动量方程得到指数衰减及压力。时间参考来自离散相容拉普拉斯 Fourier 特征值。未采用无法追溯的经验升阻力值。

[1] G. I. Taylor and A. E. Green, Mechanism of the production of small eddies from large ones, Proceedings of the Royal Society A 158 (1937), 499–521. DOI: https://doi.org/10.1098/rspa.1937.0036 。本文使用其经典涡验证思想的二维解析衰减变体。
[2] NASA NPARC Alliance, Examining Spatial (Grid) Convergence, https://www.grc.nasa.gov/www/wind/valid/tutorial/spatconv.html 。用于网格研究方法背景，不将该网页当作本文3%阈值的来源。

## 9. FVM 与 LBM 对照

# FVM 与 TensorLBM 的匹配解析对照

## 1. 公平性与共同条件
两者求解同一周期二维 Taylor–Green 连续问题：L=2π m、ρ=1 kg/m³、ν=0.1 m²/s、U0=1 m/s、终止时间0.5 s、CPU单线程、float64状态。同样的L2/峰值归一化L∞定义与严格3%精度门；解析值在各自实际自由度位置计算。
FVM 是四层展向的MAC原型，物理解无展向变化；LBM是二维D2Q9 BGK，调用外部仓库原始 equilibrium、collide_bgk、stream、macroscopic 共性模块，没有在benchmark复制碰撞/迁移算法。LBM根据dx/dt映射SI，并使整数步恰好到0.5 s；τ=0.5+3νdt/dx²，Ma按实际dt记录。初始密度由解析压力和EOS确定，populations采用生产equilibrium初始化，可能产生有限启动误差；未做人为质量调平。
FVM压力属于最后midpoint时刻；LBM压力属于步末。各自对照对应时刻的解析压力，没有把两个不同时钟的压力直接相减。LBM节点和FVM错位面不同，比较解析误差而非未经插值的数组差。
LBM使用float64 populations，但当前生产D2Q9权重由float32常数形成；本报告保留该实现并测量质量漂移，不把它归因于所有LBM方法。对照仅涉及此BGK基线，不能代表TensorLBM的MRT、cumulant或GPU优化能力。

## 2. 实际结果
| N | LBM Ma | FVM u L2 (%) | LBM u L2 (%) | FVM p L2 (%) | LBM p L2 (%) | FVM / LBM steps | FVM / LBM observed CPU seconds |
|---:|---:|---:|---:|---:|---:|---|---|
| 32 | 0.04956 | 0.03209 | 0.63348 | 0.89781 | 0.86173 | 40 / 89 | 0.8630 / 0.0747 |
| 64 | 0.04984 | 0.00802 | 0.15207 | 0.22480 | 0.81428 | 40 / 177 | 1.7436 / 0.2948 |
| 32 | 0.02492 | 0.03209 | 0.64003 | 0.89781 | 1.68311 | 40 / 177 | 0.8630 / 0.1473 |

![Accuracy comparison](figures/fvm-lbm-errors.png)

## 3. FVM 的优势及证据边界
- 压力由不可压约束求解，不依靠弱可压状态方程或人为声速；本例记录相容面散度与实际压力误差。
- MAC 面速度直接满足离散连续性，并保留每步动量、压力功、黏性功与能量账本。LBM的中心差分散度是可压宏观诊断，与MAC权威通量散度不是同一个离散算子；只并列展示，不据此判定LBM错误。
- 物理时间步不由格子声速映射决定。本例FVM可以直接做0.1/0.05/0.025 s时间加密；隐式非线性求解仍可能失败，绝不宣称任意大步长。
- 按一层物理平面统计，FVM主未知量3速度+1压力为4N²个double，D2Q9为9N²个double，比例4/9。此数只描述主状态存储，不含FVM非线性临时量、FFT、LBM流索引、报告缓存；不是峰值内存或总成本优势。
- 新通道报告另验证压力梯度、壁面剪切与力平衡，给后续FEM牵引映射提供明确SI量。不能据此声称已有曲面、移动体或工程FSI优势。

## 4. 性能与LBM自身优势
此处运行时间来自单次CPU完整执行，包含初始化、诊断与原场采样；FVM实际推进四个z平面，LBM只有一个二维网格，而且时间步数不同。未进行重复统计、暖启动、峰值RSS或GPU公平性能测量。因此不宣称FVM速度更快或峰值内存更低。LBM的局部碰撞/迁移、无需全局压力Poisson，以及成熟融合GPU路径具有不同成本结构，应在后续相同精度、相同物理维度的基准中实测。
两者的失败、质量漂移和压力误差均保留，不能通过筛选指标来制造优势。未来应增加相同固壁边界的压力驱动通道和同一几何外流，在完整3%门与守恒约束下比较达到指定精度的总成本。

## 5. 数据与复现
summary.json各LBM运行记录外部仓库提交号、共性源码SHA、Mach、τ、dt、密度变化与质量历史。原始populations及每一步完整状态可独立重建BGK和迁移。comparison.json绑定此次测得指标。主报告的FVM资格与LBM比较资格分别记录；某一基线失败不能被另一个基线的成功覆盖。
