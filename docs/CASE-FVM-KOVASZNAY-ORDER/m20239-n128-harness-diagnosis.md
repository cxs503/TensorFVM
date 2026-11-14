# CASE-FVM-KOVASZNAY-ORDER · 为什么 n=128 一直没落盘：调度器 `cd` 陷阱与 rc 掩盖

**锚点**: `CASE-FVM-KOVASZNAY-ORDER`（算例池·定常 Kovasznay 流收敛阶验证，ledger = `open`）
**判据**: L2 收敛阶 ≥ 1.8（名义二阶），以 32/64/128 三套网格 log-log 最小二乘斜率计量
**基准**: Kovasznay 1948, *Proc. Camb. Phil. Soc.* 44:58–62, doi:10.1017/s0305004100023999
**席位**: m20239（generalist）· 仓 TensorFVM

---

## 1. 复核发现（阻塞原因，非数值困难）

该锚点至少三张在前单（m18408 → cj1915、m18510 → cj1944/cj1946、m19958 → cj2438）都
递交过 n=128，其中两单在 `ps` 里显示 **`succeeded` / `rc=0`**，但工件始终没有出现，
anchor 因此反复被记为"n=128 未落盘"。本单定位到真实原因，它与数值无关：

递交命令把**本机绝对路径**写死在 `cd` 里：

```bash
cd /root/tp-agent/work/m20239 && PYTHONPATH=$PWD/src python3 scripts/verify_kovasznay_order.py ...
```

`stage_dir` 语义是"把我的工作树打成 tar 送到节点，命令在节点上的**检出目录**里执行"，
节点上既没有也不允许访问 `/root/tp-agent/work/<mid>`。于是每个这样的作业都在第一句就死：

```
{"job_id": "cj2589", "node": "n315", "head": "local-tree", "rc": 1,
 "output_tail": "RUNNER_EXIT=1\nbash: line 1: cd: /root/tp-agent/work/m20239: Permission denied\n..."}
```

（证据：`/root/tp-agent/cluster/jobs/cj2589.log`。同一命令在本机 `run_shell` 里当然能跑，
这正是它被反复写下来的原因——本地可复现，掩盖了远端不可复现。）

**正确写法**：命令里不要出现本机绝对路径，直接用检出目录的相对路径；

```bash
PYTHONPATH=$PWD/src python3 scripts/verify_kovasznay_order.py --output docs/.../n128 --n 128 ...
```

## 2. 第二层缺陷：`rc` 掩盖（诚实 FAIL 被记成 PASS）

上表的 `ps` 对 cj2589 报 `succeeded`, `rc=0`，而同一作业的 job log 明写 `rc: 1`。
两层 rc 不一致意味着**一个失败的重算会以"成功"进入队列统计**，后续 agent 读到 `succeeded`
就以为证据已产出、只是忘了提交。这与 `honest FAIL 不得覆盖` 的纪律直接冲突，属于证据链缺陷。

同一类掩盖在命令尾部写成 `... | tail -40` 时也会发生：`tail` 的退出码取代了 python 的退出码。
本单改用的模板把两者都避开——不 `cd`、不经管道，落 stdout 到树内日志并显式回传退出码：

```bash
PYTHONPATH=$PWD/src python3 scripts/verify_kovasznay_order.py --output <out> ... \
  > docs/CASE-FVM-KOVASZNAY-ORDER/n128-cluster-run.log 2>&1
rc=$?; echo RUNNER_EXIT=$rc; tail -4 <log>; exit $rc
```

日志本身随树回传，因此"作业跑没跑、跑成什么样"是可审计的工件，不再依赖 `ps` 的一行摘要。

## 3. 复核确认的既有证据（可从工作树 npz 现场复算，本单独立复算）

| n | h | cells | vel L2 (full 3D) | max\|w\| | p L2 (mean-free) | 单段阶 |
|---:|---:|---:|---:|---:|---:|---:|
| 32 | 3.125e−02 | 1024 | 3.435615e−03 | 1.60e−16 | 2.885142e−02 | — |
| 64 | 1.562e−02 | 4096 | 8.972614e−04 | 2.49e−17 | 7.958339e−03 | 1.93697（速度）/ 1.85810（去均值压力） |

本单用 `volume_weighted_error` 直接从 `n32-fields.npz` / `n64-fields.npz` 复算，
与 `cases.json` 的差为 0（rel diff 0.0），确认无手抄。按同样阶外推，n=128 期望
velocity L2 ≈ 2.34e−04、p L2(mean-free) ≈ 2.20e−03。

## 4. n=128 本次提交

作业 **cj2598**（8 核，调度器派发，无节点指定），命令=上文 §2 模板，
配置与 n32/n64 完全一致（Re=40, convection_order=2, coupled/reused_lu,
outer_relaxation=0.7, tolerance=1e-8, symmetry_closure=zero_gradient），
因此 `summarize_kovasznay_32_64_128.py::validate_levels` 的混合配置门不会拒绝该序列。

结论与三网格斜率见本单最终报告（`RESULT_32_64_128.md`）。