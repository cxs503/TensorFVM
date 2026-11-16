| 项 | 方案基线(m23216) | 现在实测 | 取法 |
|---|---|---|---|
| 提交数（全引用可达） | 2184 | 2196 | `git rev-list --all --count` |
| 本地 heads | 833 | 838 | `git for-each-ref` |
| 关联工作树 | 326 | 331 | `git worktree list` |
| in-pack 对象数 | 38350 | 38633 | `git count-objects -vH` |
| size-pack（含不可达） | 1367.7 MiB | 1.33 GiB | `git count-objects -vH` |
