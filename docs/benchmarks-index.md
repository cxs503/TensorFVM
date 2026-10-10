# TensorFVM benchmark 总览

## 已发布的正式解析验证

| 物理问题 | 正式计算矩阵 | 已验证内容 | 报告 |
|---|---|---|---|
| Poiseuille 层流通道 | 36×12、72×24、144×48 | 速度、压力梯度/曲线、壁面剪切、质量/力平衡 | [原始共性报告](verification-benchmarks/report.md) |
| Taylor–Green 涡衰减 | 空间32²/48²/64²；固定32²的3档时间步；z=4 | 速度、压力、空间/时间约二阶、逐步质量/动量/能量；3组BGK对照 | [原始共性报告](verification-benchmarks/report.md) |
| **三维 ABC/Beltrami 涡衰减** | **24³、32³、48³** | 真正三维的非线性压力平衡、速度/压力误差、空间约二阶、逐步守恒 | [新增报告](verification-extended/report.md) |
| **带平均流的对流剪切波** | **16²/24²/32²，z=4** | 输运相位与衰减、非零平均动量、零压力动态尺度误差；32²/64² BGK对照 | [新增报告](verification-extended/report.md) |

新增 [贴体环形管](verification-engineering/report.md)：16,384 / 65,536 / 262,144 个完整矩阵未知量，速度二阶收敛，全部严格 <3%。该问题为三维充分发展流的二维截面约化。

以上共18个正式 FVM 运行、5个实际 TensorLBM BGK 物理对照。
各正式配置全部声明物理误差严格 <3%，另有独立逐步 NumPy 原场审计。
ABC最细速度/压力 L2 误差为0.001427%/0.424980%。
剪切波32²的单独输运分量误差为FVM 0.225186%、BGK 0.332209%。
新增详细数据和同网格比较：[结果表 PDF](verification-extended/results-tables.pdf)。

## 已保留的失败与诊断

- 单次 SIMPLE 迭代的通道、16² Taylor–Green 压力、16³ ABC 压力负控制：均明确拒绝。
- 128² Taylor–Green 非线性候选失败：[完整复现证据](verification-benchmarks-failed-n128/README.md)。
- DFG Re20 圆柱64×24：SIMPLE残差收敛，但阻力误差约13.36%，未获3%物理资格。
  [圆柱原场与诊断报告](verification-dfg/report.md)；增加升力/前后压力差门及真实压力/黏性牵引和质量面通量证据。
  已提供三网格共性入口，尚未完成该入口的全部细化计算，不能称圆柱已网格收敛。

## 专项数值验证与工程范围

相容压力投影、周期MAC动量/能量、变黏度应力制造解已有专项归档，
它们的资格不能自动赋予其他几何/边界案例。
NACA0012、DFG其他网格、SA平板、Re3900 URANS和30P30N仍按各自历史审查状态处理。
当前尚未发布达标的SUBOFF、自由液面、移动体或上浮破冰流体benchmark。

复现入口与模块约定：[共性工作流](benchmark-workflow.md)。

## GPU、壁面函数与湍流推进

[实际 CPU/CUDA 重复计时和场一致性](verification-gpu/report.md)：RTX 3090，10 项，完整预热后三次计时；最大环形网格约 7.29 倍，小网格存在减速。
[壁面公开关联曲线](verification-wall-law-reference/report.md)在指定对数区通过；完整壁面函数 RANS 尚未认证。
[SA 1800 次迭代严格诊断](verification-sa-diagnostic/report.md)摩阻误差 7.22%，未收敛。
[完整推进路线与复现](engineering-gpu-roadmap.md)。
