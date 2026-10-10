# 三段翼表面压力系数

![三段翼Cp曲线](cp-comparison.png)

Cp=(p_wall-p_inf)/(0.5 rho U_inf²)，p_inf=0为出口给定的表压参考。每个翼段在最前/最后x坐标顶点处将闭合壁面拆成两条连续路径；弦线法向平均坐标较高者定义为上表面，按真实壁面连接顺序连线；横轴为全局参考弦长归一化坐标，纵轴按气动惯例反向。Cp目前采用相邻单元压力，尚未验证壁面压力外推精度。

当前为两步未收敛SIMPLE/SA诊断。没有同工况可核验的实验/独立参考Cp数据，图中计算上下表面曲线不能替代实验对比，实验误差为空，物理精度验收为False。

[逐壁面原始数据](cp-surfaces.csv) · [对比状态](cp-comparison.json) · [PDF](cp-report.pdf)

复现：`PYTHONPATH=src python scripts/compare_three_element_cp.py docs/verification-30p30n-flow-preview`。若有参考文件，添加`--reference DATA.csv --reference-metadata META.json`。入口校验工况、归一化、每个翼段上下表面覆盖，拒绝外推及含糊的多值横坐标；来源和输入哈希随报告保存。
