# CASE-FVM-KOVASZNAY-ORDER · n=256 两次递交取回为空：不是"交付过滤器丢工件"，是 checkout 就连不上 GitHub

**锚点**: `CASE-FVM-KOVASZNAY-ORDER` · 判据: 32/64/128 速度 L2 斜率 ≥ 1.8（本文件不改判据）
**席位 / 仓**: m22772 · TensorFVM · 分支 `agent/m22772`
**结论一句话**: `cj4359`（n=256 实算，16 核）与 `cj4360`（n=256 单迭代探针，8 核）都在
**作业 checkout 阶段**失败——`git clone https://github.com/cxs503/TensorFVM.git` 连接超时，
命令从未执行；取回 tar 为 **0 KB** 是这个原因，而不是"交付通道把二进制滤掉"。

## 1. 现象

| 作业 | 内容 | 队列可见状态 | 取回 | 每作业日志 `stage` |
|---|---|---|---|---|
| `cj2912`（上轮 m20758） | n=256 实算 8 核 | succeeded / rc=0 | `n256/` 空目录、无日志 | 未保留 |
| `cj4359`（本轮） | n=256 实算 16 核，带 `--evidence-markdown` | succeeded / rc=0 | — | **FAILED** |
| `cj4360`（本轮） | n=256 `--max-outer-iterations 1` 内存/时间探针 | succeeded / rc=0 | `size_kb = 0` | **FAILED** |

`cj4359`/`cj4360` 的每作业日志原文（`/srv/tensor/agent/cluster/jobs/<jid>.log`）：

```
{"job_id": "cj4360", "node": "n330", "stage": "FAILED",
 "detail": "fatal: unable to access 'https://github.com/cxs503/TensorFVM.git/':
  Failed to connect to github.com port 443 after 134817 ms: Couldn't connect to server"}
```

两作业在同一节点 `n330`、同一时段、同样的 134.8 s 超时，说明这是节点的出网状态，不是算例问题。

## 2. 被否证的假设（负面结果也入账）

`ORDER-32-64-128-256-ATTEMPT.md` §3.1 与台账 D10 续把 `cj2912` 的空取回记为两种可能且不可判别：
(i) 求解在节点上早退；(ii) 工件被"交付过滤器"滤掉。本轮的探针给出第三条、也是真正的解释：

> **作业根本没有进入命令执行阶段**。`run_compute` 可见状态 `rc=0` 是调度层回执，不代表命令成功；
> 判定要看每作业日志的 `stage` 字段与取回 tar 的大小（`size_kb`）。

因此 (ii)"交付过滤器丢工件"作为一个**机制**未被任何证据支持：本轮探针的取回 tar 是 0 KB，
里面连 `.md` 纯文本都没有，过滤器的行为无从谈起。同一时期仓内 `docs/**` 的普通提交正常，
把工件的丢失归因于"过滤器"会掩盖真正原因（出网检查）。`cj2912` 是否也栽在同一处，
从保留物无法回看，但同一节点的同一签名使其成为最简解释。

## 3. 下次（可执行，已验证到工具层）

1. **不要依赖 GitHub checkout**：`run_compute --stage_dir <工作树>` 把本地检出 tar 直达节点，
   checkout 不走出网。本轮两次递交都用了默认 `ref=main`，因此必然踩到这个坑。
2. **判作业成败别只看 `rc`**：先看每作业日志的 `stage`，再看取回 `size_kb`；
   `state=succeeded` + 0 KB 的取回必须继续查日志，不得写成"跑完了"。
3. 节点恢复出网前，把"跑一次 n=256"从算例判据里摘出来：它只是**阶的不确定度**（§4.5 欠账），
   不影响 32/64/128 的台账判据（`summary.json` 里 `verdict.pass = true`，且本轮未改动一位）。

## 4. 本文件不做的事

不生成 `summary-32-64-128-256/`、不写四层阶、不引用任何 n=256 的数字。四层证据仍为
**not measured**，`tests/test_kovasznay_256_ladder.py` 的工件门继续 skip（not measured ≠ pass）。
