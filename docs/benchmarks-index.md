# TensorFVM benchmark 总览

## 已发布的正式解析验证

| 物理问题 | 正式计算矩阵 | 已验证内容 | 报告 |
|---|---|---|---|
| Poiseuille 层流通道 | 36×12、72×24、144×48 | 速度、压力梯度/曲线、壁面剪切、质量/力平衡 | [原始共性报告](verification-benchmarks/report.md) |
| Taylor–Green 涡衰减 | 空间32²/48²/64²；固定32²的3档时间步；z=4 | 速度、压力、空间/时间约二阶、逐步质量/动量/能量；3组BGK对照 | [原始共性报告](verification-benchmarks/report.md) |
| **三维 ABC/Beltrami 涡衰减** | **24³、32³、48³** | 真正三维的非线性压力平衡、速度/压力误差、空间约二阶、逐步守恒 | [新增报告](verification-extended/report.md) |
| **带平均流的对流剪切波** | **16²/24²/32²，z=4** | 输运相位与衰减、非零平均动量、零压力动态尺度误差；32²/64² BGK对照 | [新增报告](verification-extended/report.md) |

新增 [贴体环形管](verification-engineering/report.md)：16,384 / 65,536 / 262,144 个完整矩阵未知量，速度二阶收敛，全部严格 <3%。该问题为三维充分发展流的二维截面约化。

新增 [完整贴体曲壁 SIMPLE](verification-simple-gpu/report.md)：32×16、64×32、128×64、256×128，实际求解两个速度分量与压力；最细32,768控制体/98,304主要未知量，速度L2 0.037391%、压力L2 0.029793%，四级误差全部严格小于3%，独立原场验收通过。含非正交曲壁及连续 NS 体力制造解，不能外推无体力弯管工程验证。

以上共22个正式 FVM 运行、5个实际 TensorLBM BGK 物理对照。
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

## 完整 SIMPLE 与案例优先验证

[共性入口、算法范围与复现](full-simple-gpu-workflow.md)；[完整曲壁/稳态 SA 报告](verification-simple-gpu/report.md)。新可选稀疏后端实际 CPU/CUDA 一致性通过，包括原 128×112 五步失败配置的 [独立复核](verification-sparse-sa-consistency/report.md)。复核是单次正确性检查，不是重复性能资格。原默认后端与原失败记录保留。

SA64×56现在78步达到稳态，但经验平均Cf误差6.61%，仍不合格；[网格加密报告（两级CPU、最细级CUDA）](verification-sa-refinement/report.md)另行检查分辨率与物理误差。首层贴壁 SA、完整壁面函数 RANS 与匹配公开湍流数据是不同验收范围。GPU作为兼顾验证；主任务是物理误差、网格收敛和边界/方程残差。

SA三网格（64×56 CPU、128×112 CPU、256×224 CUDA）全部达到稳态门，但经验摩阻误差分别6.614%/5.894%/5.059%，均未达到3%，不增加正式合格计数。最细网格57,344控制体/229,376未知量，最大首层y+=0.736；Cf相邻网格变化没有持续减小，不能宣称渐近网格收敛或直接归因于模型误差。

## 圆柱与翼型三网格实际验证

新增统一入口 `tensorfvm.verification.external_flow`，通过共性 Metric/evaluate/save_run 保存真实流场、面质量通量、压力/黏性牵引和全部线性残差。CPU 可选稀疏直接解，CUDA保留Torch Krylov；离散方程相同。回归测试比较两个CPU后端实际五步场。

- [DFG Re20 三网格报告](verification-external-cylinder/report.md)：64×24、128×48、256×96，逐项检查 Cd、Cl、前后压力差。
- [NACA0012 Re1000、4°三网格报告](verification-external-naca/report.md)：64×26、128×52、256×104，分别检查 Cl、Cd，展示原场压力、速度与 Cp。
- [旧翼型通过结论撤回](airfoil-benchmark/REVIEW.md)：原报告的0.205/0.120参考已替换为原论文矢量曲线读取事实及 ±0.0001 保守区间；旧档案保持可追溯。

本节诊断计算不增加22个正式合格运行的计数。数值收敛、物理误差和渐近网格收敛分别审查，具体状态以实际报告为准。

圆柱三网格已通过独立NumPy完整层流方程及物理Rhie–Chow固定点复核，但最细Cd误差4.040%、Cl误差33.297%、Δp误差4.410%，仍不合格。[完整入口与推进次序](external-flow-benchmark-workflow.md)解释求解后端、逐项验收、历史撤回和后续二阶输运/网格误差分解。

NACA三网格（64×26、128×52、256×104）全部达到稳态并通过独立完整方程/物理通量复核；最细546步，Cd=0.14480160、Cl=0.23290890，保守参考区间误差15.928%/15.954%，仍不合格。最细采用当前矩阵GMRES+缓存LU预条件及既有Anderson混合，前两级复用实际CPU直接解并保留各自生产源快照。新增六个诊断网格不增加22个正式合格运行计数。
