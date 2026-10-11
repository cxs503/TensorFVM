# SUBOFF：SIMPLE + SST + 壁面函数接入与实测

## 结论

已实现原生三维贴体网格的 **SIMPLE 分离式求解 + SST-2003 + Spalding 壁面函数**。
SUBOFF 运行器默认使用这一组合，每次 SIMPLE 动量/压力修正后更新 k、ω。
**小通道原方程收敛通过；SUBOFF 尚未收敛，Cp、Cd 未达到 3%。**

[完整物理报告：网格、压力/速度场、Cp 实验对照、阻力分解、残差和 y+](physical-report/report.md)。
[42 项针对性测试](tests.txt)覆盖独立仿射解、偏斜网格、守恒通量、壁面牵引、SST 输运和原方程收敛。
测试通过不计入合格物理 benchmark 数量。

## 实现内容与模型约定

- SIMPLE：正交扩散和迎风对流隐式求解动量，完整非正交/对称应力差额延迟修正；速度方程松弛；紧凑压力修正；压力松弛与守恒面通量修正。CPU SciPy 分别求解动量和压力矩阵，未运行 GPU。
- Rhie–Chow 使用松弛后的动量对角响应。本离散的稳定化系数依赖松弛参数；它与此前全耦合离散并非同一固定点声明。当前全动量残差、存储/重构通量散度和面通量一致性共同决定收敛。
- 复用 SST-2003 内部闭合与 k、ω 输运；动量求解 pi=p+2ρk/3，报告同时还原物理压力及各向同性雷诺应力。
- Spalding 全 y+ 平衡壁面律由局部切向速度、真实近壁距离和分子黏度反求摩擦速度。切向牵引为 −ρuτ² U_t/|U_t|；作为隐式 Robin 阻力进入动量方程，力积分使用同一牵引。壁面法向质量通量严格为零，保留法向黏性牵引。
- 壁面函数模式下 k 为零法向梯度；ω 在壁邻单元施加约束，并对一单元多壁面按面积平均。采用本项目明确的平衡组合：ω_vis=6ν/(β1 y²)，ω_log=√k/(β*^(1/4) κy)，ω_target=√(ω_vis²+ω_log²)。壁邻 k 生成项为 [1−exp(−(y+/11)²)]uτ³/(κy)，仍受 SST 生成上限约束。这是近壁模型选择，不等同于低 Re 原始壁面条件，也不宣称逐式复制 OpenFOAM。
- ω 约束单元的输运反力单列进守恒账本；收敛要求非约束输运残差及 ω 约束残差均通过。不会把约束行删除后的小残差当成原输运方程无源守恒。
- 旧的 `--algorithm coupled --wall-treatment resolved` 路径保留。`--disable-skew-correction` 是线性面插值诊断选项，其精度需另做网格检验；并未成为默认方案。

近壁约束方法参考 [OpenFOAM 官方 omegaWallFunction 实现](https://github.com/OpenFOAM/OpenFOAM-dev/blob/master/src/MomentumTransportModels/momentumTransportModels/derivedFvPatchFields/wallFunctions/omegaWallFunctions/omegaWallFunction/omegaWallFunctionFvPatchScalarField.C)；内部 SST 方程依据 [NASA TMR SST-2003](https://tmbwg.github.io/turbmodels/sst.html#sst-2003)。常数未按 SUBOFF 参考数据拟合。

## 数值检查

27 单元非零来流通道，ν=0.01，SIMPLE 速度/压力松弛 0.4/0.2，每轮最多 10 个 SIMPLE 步，SST 伪时间步 1，湍流松弛 0.7。
[通道迭代记录](channel-convergence.json)，由 [复现脚本](verify_channel.py) 生成：20 轮收敛；当前动量残差 1.76e-09，重构散度 9.65e-09，k 残差 6.64e-08，ω 残差 7.93e-08。
对应可复现断言在 `tests/test_native_simple3d.py::test_simple_sst_wall_function_converges_current_equations`；偏斜修正开/关均检查。该算例验证求解器收敛能力，未与湍流通道实验比对。

## SUBOFF 实际尝试

相同 AFF-1 共形贴体混合网格：9,068 单元；旧的 3 层近壁网格。
来流 U=1、ρ=1；Cp 对照 Re=1.2e7，阻力对照 Re=13,218,056.809580836。
所有尝试均保留源代码快照、配置、原场和每步日志，速度保护阈值 5U，未裁剪解。

| 目录 | 设置区别 | 动量步 | 完整 SST 更新 | 停止时最大速度/U |
|---|---|---:|---:|---:|
| cp-pilot | 初版壁面阻力延迟处理 | 1 | 0 | 5.247 |
| cp-projected-implicit | 壁面阻力隐式处理；初始通量投影 | 3 | 0 | 8.969 |
| cp-interleaved | 每步更新 SST；松弛 0.05/0.03 | 2 | 1 | 18.336 |
| cp-linear-faces | 关闭压力/速度面偏斜修正 | 2 | 1 | 28.029 |
| drag-interleaved | 与 cp-interleaved 同设置，匹配阻力 Re | 2 | 1 | 18.388 |

这不是严格的单变量消融表，各例完整差异以 config.json 为准；早期快照保留当时实现，不覆盖成最终源代码。
主报告使用 cp-interleaved / drag-interleaved。未收敛场 Cp 相对 L2 偏差 **1485.52%**，Cd=**5.8243066**，参考 Cd=**0.093120813**，阻力相对偏差 **6154.57%**。
这些数值仅表示失败迭代场与参考的偏差，不能称为收敛解精度或改善率。

最终存储面通量散度接近舍入量级，但重构通量与存储通量严重不一致，故所有 SUBOFF 数值收敛标志为 false。
Cp 主诊断的 y+ 约 113–246，阻力诊断约 123–268；它们来自停止时的牵引，不是收敛近壁解的资格。
匹配 y+ 范围、完成一次 SST 更新或换成 SIMPLE 均不能替代完整收敛和网格独立性。

## 使用与后续

从仓库根目录运行，输出目录必须为空：

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=src \
/data/TensorFEM/.venv/bin/python scripts/run_native_suboff_rans.py \
  --mesh docs/verification-suboff-single-domain-mesh/coarse-tail-transition/suboff-hybrid.msh \
  --output results/suboff-simple-reproduction --reynolds 12000000 \
  --algorithm simple --wall-treatment spalding \
  --inner-iterations 1 --outer-iterations 20 --pseudo-dt .0001 \
  --convection-order 1 --relaxation .05 --pressure-relaxation .03 \
  --maximum-speed 5 --project-initial-flux
```

该命令复现失败诊断，退出码 2 表示未收敛。实验比较及报告命令见 `physical-report/report.md`。
下一步定位尾部混合网格的压力修正/单元速度响应，首先解决面通量与单元速度一致性；随后才扩大网格、提高空间阶数并开展壁面模型/三档网格独立性研究。

## 证据复核

[audit_evidence.py](audit_evidence.py) 已独立核对 140 个计算生产源文件哈希、5 组原始面力积分与通量散度，以及 2 项实验误差指标。[审计记录](audit-results.json) 通过；物理验证标志仍为 false。
`implementation-sha256.json` 绑定本次最终实现与验证源文件，`campaign-manifest.json` 绑定全部归档文件。
