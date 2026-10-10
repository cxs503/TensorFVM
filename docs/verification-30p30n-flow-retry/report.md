# 30P30N 实际流动重跑：未通过

19,023个贴体单元，Re=5,000,000，攻角0°，二维不可压缩SIMPLE＋Spalart–Allmaras，CPU双精度。采用现有默认松弛与伪时间步0.02，残差门槛1e-5。

最初1000步入口运行约六分钟后仍在压力修正线性迭代，且没有中间输出；主动终止后改用有实时记录和120秒上限的诊断入口。两次运行从相同网格、初始场独立启动，未续算。

限时运行完成2步，第三步中达到120秒上限。第一/第二步动量残差16.6765/12.7180，湍流残差0.01080/0.01486；连续性残差5.54e-13/1.14e-12，质量不平衡1.96e-11/1.36e-10。数值稳态未收敛，物理精度未验收。少量迭代不能判断最终能否收敛，也不能据此判定算法发散。

超时步的部分更新场已丢弃，没有将它导出为压力/速度/Cp或升阻力结果。当前证据支持下一步优先检查压力线性求解器、对流离散与松弛策略。三段翼静态overset标量扩散通过不代表本实际流动算例通过。

完整历史与源码指纹：[diagnostic.json](diagnostic.json)。复现：

```bash
PYTHONPATH=src python scripts/run_three_element_flow_diagnostic.py --mesh-file docs/verification-30p30n-flow-retry/mesh.msh --output NEW_DIR --iterations 20 --seconds 120
```

## 三段翼几何与实际流场图

[几何、速度和压力图](../verification-30p30n-flow-preview/visual-report.md)。流场来自19,023单元单套Gmsh贴体网格上两个完整SIMPLE/SA迭代步，仍未收敛，不能作为已达标的物理解。此图不是overset标量制造解，也不是overset Navier–Stokes结果。

![三段翼几何](../verification-30p30n-flow-preview/geometry.png)

![未收敛速度场](../verification-30p30n-flow-preview/speed.png)

![未收敛压力场](../verification-30p30n-flow-preview/p.png)
