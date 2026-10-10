# DFG 2D-1 Re20 圆柱：未达标诊断

[完整报告](report.md) / [PDF](report.pdf) / [完整字段及积分审计](64x24/audit.json) / [独立压力与质量审计](independent-pressure-audit.json)。

64×24 贴体网格在975次SIMPLE迭代后收敛，残差与质量门通过。
但是 Cd=6.32479236、Cl=0.015101714、前后壁面压力差0.130991489 Pa；
相对公开参考误差分别为13.35698%、42.21478%、11.46299%，三个3%物理门均失败。
没有将残差收敛解释为物理达标，也未宣布网格收敛。

原场包含完整贴体网格、cell压力/速度、权威face质量通量、压力/黏性分力、
壁面修正梯度、压力外推曲线、残差和来源/原场SHA。
独立NumPy先积分压力/黏性牵引，再重建least-squares压力梯度、壁面外推及前后插值、质量和残差门。
审计通过说明证据一致，不改变物理误差失败的结果。

复现与后续细化：

~~~bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=src python -m tensorfvm.verification.dfg --output results/dfg --grids 64x24
# 三网格入口已经实现；本轮仅发布64×24，后两档尚未执行：
# --grids 64x24 128x48 256x96
PYTHONPATH=src python scripts/audit_dfg_pressure.py --directory results/dfg
~~~

一次迭代的16×8拒绝控制也保留在[控制归档](../verification-dfg-control/report.md)。
下一步分解网格/对流/壁面牵引/压力重建误差，并完成细化；目前没有同条件LBM圆柱对照。
