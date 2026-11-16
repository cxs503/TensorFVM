# FVM-SLIM-GUARD 注入试验（D6, R4）：实测拦截有效性

> 生成：2026-11-16T20:25:43.761917+00:00 ｜ 脚本 `scripts/guard/injection_test.py`
> 方法：在**临时一次性 git 仓**上接线**真实**守卫（`.githooks/pre-commit` +
> `scripts/guard/repo_guard.py` + `.gitattributes`，`core.hooksPath=.githooks`），
> 注入大/小/白名单 blob 后跑 `git commit`，看钩子是否真的拦截。
> 机器记录：`docs/m24008/injection-test.json`。

## 1. 判据（冻结）

| 判据 | 值 |
|---|---|
| 512 KiB 拦截阈值 | 524288 B（512 KiB） |
| 注入大 blob | 600 KiB（> 阈值，应拦截） |
| 注入小 blob | 100 B（< 阈值，应放行） |
| 白名单属性 | `guard-allow-large`（`.gitattributes`） |

## 2. 结果（程序化注入，零手抄）

| 用例 | 路径 | 大小 | 钩子行为 | 期望 | 判定 |
|---|---|---|---|---|---|
| large-unlisted | big.bin | 600.0 KiB | blocked | block | PASS |
| small | small.txt | 0.1 KiB | passed | pass | PASS |
| large-whitelisted-npz | docs/verification-x/f/fields.npz | 600.0 KiB | passed | pass | PASS |
| large-unlisted-docs | docs/verification-x/f/fields.bin | 600.0 KiB | blocked | block | PASS |

## 3. live 体积门（真实仓，只读）

| 量 | 值 |
|---|---|
| live 体积（unique HEAD 引用 blob） | 188.0 MiB（197122081 B） |
| live 门状态 | OK |

## 4. 结论

- **全部 4 用例 通过**：真实钩子接线
  在注入大 blob 时确实拦截（rc≠0），小 blob 放行，白名单大工件放行，
  未列入白名单的大 blob 仍拦截。
- 拦截有效性 = **2/4**
  用例按预期触发钩子。
- 本试验**不触碰真实仓历史**（全部在临时仓完成）；真实仓只做只读 `live` 探测。
