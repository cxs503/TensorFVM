# 原矩阵残差修正与完整 Newton 更新验收

2026-10-13。针对 11,648 单元 SUBOFF 的第四次线性修正失败开发。前次真实相对残差 1.83378e-9，超过 1e-9；失败记录仍保留，不能追认通过。

## 数值修改

初始分块 AMG–GMRES 保留 restart=80、最多 60 个周期、rtol=1e-13。仅当候选有限、原方程门槛未满足且初始状态允许使用时，对同一个当前矩阵计算扩展精度残差，再用原预条件器解修正方程；不修改动量、连续性或压力规范方程。每次修正最多 10 个 GMRES 周期、rtol=1e-8，最多三次；只保留原方程综合误差改善的候选，否则立即停止。修正求解的较松内部目标不能替代最终验收。

最终 float64 原方程相对残差仍须 <=1e-9、逐单元连续性缺陷/体积 <=1e-8、压力规范缺陷 <=1e-9。初始 GMRES 返回值和所有修正记录分别保存。严格初始化仍不接受迭代超限；只有显式 `--inexact-newton` 才允许初始正的迭代超限状态，负的 breakdown 始终拒绝。

新增 `--verify-initial-correction`，即使重启初始残差已小，也必须真正求解并验证一次 Newton 修正。主线还修复阻尼导致更新量假小的风险：收敛检查采用完整未阻尼修正，历史同时保存接受步的变化量与完整方向的变化量。默认的初始已解状态快捷路径保留；本次失败后重启必须显式启用完整修正检查。

## 验证范围

- 修正分支 28 项测试通过：闭合/开放压力条件、原矩阵修正、局部质量门槛、无改善拒绝、初始失败状态政策、完整重启检查。
- 与任意法向约束、Robin、SST 耦合主线合并后 79 项回归通过。
- 加入完整更新量判据及旋转坐标修正后，40 项针对性回归通过。反例证明大修正不能因线搜索减小而误判收敛；旋转切向基底保持零法向通量、原物理力与能量收支。
- 以上测试集合有重叠，不相加当作独立测试总数。

实际细网格复算使用冻结的 `f0ee8cb`，正在运行。该版本强制进行初始修正；若最终步长不是 1，则本次旧版记录不能作为完整更新验证通过。主线报告器已对此加守卫。没有将小网格测试结果视为实际 SUBOFF 达标。

复算命令（在修正分支，前次失败原场保留）：

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=src python scripts/run_native_hybrid_suboff.py \
  --mesh /home/jsyc/tensor-suite-development/TensorFVM/docs/verification-suboff-hybrid-mesh/coarse/suboff-hybrid.msh \
  --reynolds 10 --linear-method block_amg --inexact-newton --verify-initial-correction \
  --initial-run results/native-hybrid-suboff-fine-re10-inexact \
  --output results/native-hybrid-suboff-fine-re10-refined-terminal
```

Re=10 是无体力层流验证。SST/壁面函数的高 Re SUBOFF、Cp/Cd 对匹配实验小于 3% 的最终目标仍未完成。
