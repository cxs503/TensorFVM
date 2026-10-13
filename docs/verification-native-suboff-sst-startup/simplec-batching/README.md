# 原生 SST 的 SIMPLEC 内层修正批次

2026-10-13。此项改进迭代配置，不改变稳态动量、连续性或 SST 方程。

## 实测依据

[固定同一 SST 涡黏性的真实 SUBOFF 对照](../frozen-momentum-comparison/README.md)中，对角松弛 α=0.9、每次更新对流后最多 20 次修正，10 次外迭代、172 次内层修正收敛。当前动量残差 2.1647e-10，全局动量收支 4.5588e-10，能量收支 2.3016e-10。与同一输入的原生 Newton 解相比，速度最大绝对差 5.7563e-10，修正压力最大绝对差 7.3267e-9。全套输入 SHA256 一致；原始场、源码和残差图均已归档。两者都是原生 TensorFVM，无 OpenFOAM 替代。

此网格当前动量矩阵的绝对行和/对角线之比：中位数 3.6923，95 分位 15.0775，最大 20.5090。原行和松弛 α=0.5 的有效对角松弛为 α/该比值。这证明其松弛比标准对角松弛更强，但不能据此认定所有工况都应关闭行和稳定化。

单次对角松弛 SIMPLEC 步的 cProfile：总计 21.177 s（含导入、网格和审计），NS solve 13.456 s，其中两次稀疏 LU 分解合计 13.116 s。瓶颈为反复分解；批次内沿用同一冻结系统的两个分解。这些时间来自该次本机运行，不宣称通用加速倍数。

## 接口与验证

`NativeHybridRans3D.solve(momentum_coupling="sweeps", momentum_sweeps=1, momentum_inner_iterations=20)`：每个 RANS 外迭代更新一次动量对流矩阵，在该冻结系统上最多执行 20 次 SIMPLE/SIMPLEC 修正。每次线性求解仍检查原始矩阵残差；批次后重新检查当前动量以及更新湍流后的完整耦合残差。内层次数默认仍为 1；非 sweeps 模式拒绝非默认值，防止参数被忽略。

启动脚本增加 `--momentum-relaxation-basis diagonal` 和 `--momentum-inner-iterations 20`。默认保留 absolute_row_sum；对角选项未获高 Re 稳定性验证。历史区分 `momentum_outer_iterations`、实际 `momentum_iterations`、内层上限及松弛基准。

15 项 RANS 回归通过，包含新增测试：8 次内层修正只调用两次 LU；不把冻结动量收敛当作完整 RANS 收敛；在无体力剪切流上，批次 SIMPLEC 的速度、压力、k、ω 和壁面力与独立调用的原生 Newton 耦合结果一致；非法参数被拒绝。测试与源文件哈希见本目录。这是数值验证，不是 SUBOFF 实验误差验收。

## 完整 SST 验证

正在从同一已归档第 5 步状态启动完整 SST 计算：dt=1、SIMPLEC、对角松弛 0.9、每个 RANS 外迭代 1 次对流更新和最多 20 次压力修正，最多 40 个外迭代。使用明确声明的 relative 标量工作收支策略，不能宣称旧绝对工作门槛已通过。此处不预判完整 SST 已收敛。

复现代码分支：`fix/native-simplec-batching`，提交 `56ac096b576c4b43bdb73c8c5dd33a36a24e2695`。

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=src python scripts/verify_native_suboff_sst_startup.py \
  --laminar-run docs/verification-native-hybrid-suboff-startup/re10-newton \
  --restart-run docs/verification-native-suboff-sst-startup/precision-and-work-policy/simplec-five-completed \
  --output results/native-suboff-sst-batched-diagonal \
  --steps 40 --dt 1 --scalar-work-balance-policy relative \
  --momentum-method simplec --momentum-coupling sweeps --momentum-sweeps 1 \
  --momentum-inner-iterations 20 --momentum-relaxation .9 --momentum-relaxation-basis diagonal
```

当前改进尚不提供高 Re 壁面函数、Cp/Cd <3% 或网格收敛结论。

补充：原行和松弛两组各 100 步已结束，详见[完整四组对照](../frozen-momentum-four-way/README.md)。α=0.5/0.9 的动量残差分别为 4.6027e-7、1.4929e-7，均未收敛。批次接口及复现脚本选项已合入 main；所有运行仍保留各自原始源码快照。
