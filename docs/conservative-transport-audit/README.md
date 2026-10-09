# 守恒动量后端的独立审核归档

独立 reviewer 未修改生产源码，以 NumPy 重建共享对流、完整 variable μ 应力、压力面通量、壁面分力/力矩及动量外边界账本。实际 CPU/CUDA 三步独立实验与生产50步原场核查均通过。应力平滑 MMS 三网格约二阶；这些子集不代表 LES 或完整 MAC 已通过。

- `tensorfvm-conservative-raw-audit.json`：CPU/CUDA 各50步原场、实际应力/压力/对流通量及动量账本，source/study/auditor 绑定。
- `tensorfvm-conservative-independent-review.json`：真实 CPU/CUDA 三步与跨分量 μ 制造解，完整 NumPy 重建，`*.npz` 保存最终原场。
- `tensorfvm-conservative-distributed-default-failure.json`：原失败保留。
- `tensorfvm-conservative-distributed-default-replay.json`：另一次精确复现，相同误差，完整配置和 auditor SHA。32×24×4、3步、dt=.001、PCG500预算、rtol=1e-8/atol=1e-11，严格1e-9门失败（压力7.70214e-9、力矩9.36006e-9）。
- `tensorfvm-conservative-distributed-review.json`：更紧 rtol=1e-12/atol=1e-13、1000预算通过同一严格门（压力6.706e-13、力矩2.27e-12）；未覆盖默认失败。

`manifest.json` 绑定冻结生产源码、所有归档原始数据/报告/完整审核脚本。脚本是实际审核时的原件，其输出路径为 `/tmp`，raw auditor 使用本工作区 `/home/jsyc/tensor-suite-development/TensorFVM`。在该工作区复跑可将 `scripts/*.py` 原件复制到 `/tmp`，然后在仓库根目录执行：

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=src:/tmp \
  python /tmp/tensorfvm_conservative_raw_audit.py
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=src:/tmp \
  python /tmp/tensorfvm_conservative_distributed_default_review.py
```

旧生产投影源码、旧通过和失败原场保持字节不变；本后端是 opt-in。中心速度散度最大0.653、center/face仍不相容，能量、时间/圆柱空间细化、LES、运动体、自由液面与柔性FSI资格仍为false。
