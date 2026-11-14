#!/usr/bin/env python3
"""Build /tmp/report.json for mission m20980 from the parsed audit evidence.

Numbers come from docs/audit-m20955/artifacts/evidence.json (itself parsed from raw
logs) -- nothing is retyped by hand.
"""
import base64
import json
import pathlib

HERE = pathlib.Path(__file__).resolve().parent
ev = json.loads((HERE / "evidence.json").read_text())
pytest_log = (HERE / "pytest_subject.log").read_text()
receipt = json.loads((HERE / "cj3080_job_receipt.log").read_text().strip().splitlines()[0])
probe = ev["probe"]
tp = ev["three_piece_check"]

cd_err = 100 * probe["errors"]["cd_mean"]["relative_error"]
st_err = 100 * probe["errors"]["st"]["relative_error"]


def fig(name, caption):
    b = (HERE / name).read_bytes()
    return {"name": name, "caption": caption, "png_base64": base64.b64encode(b).decode()}


markdown = f"""# m20980 交差核验：m20955 / GAP-DIFFLBM-EXT-LIT-CYL（圆柱绕流 Re=100，TensorFVM）

**裁定：L1（reopen / 返修）。** 三件套 run_artifacts / benchmark ref-vs-got / report 均为 0；repro_cmd 可执行但产出带外；committed 测试 4/6。
被核点：commit `{ev['audited_commit']}`；裁判：`tensor_benchmark_cases WHERE anchor_slug='GAP-DIFFLBM-EXT-LIT-CYL'`。

## ① 案例介绍

二维不可压绕圆柱 Re=U·D/ν=100（D=1.0，域 20D×20D，圆柱 (10,10)，U=1.0，ν=0.01，ρ=1.0），目标量 Cd_mean、St=f·D/U。交付声明参考 Cd=1.34±0.07（Braza 1986 JFM 165:79-130, doi:10.1017/S0022112086003014；Schäfer & Turek NNMFM 52 (1996), doi:10.1007/978-3-322-89849-4_39），St=0.164±0.005；生产设置 body-fitted 128×48 / dt=0.005 / 3000 步 / warmup 500。交付 4 文件：`src/tensorfvm/verification/cylinder_re100.py`(226 行)、`tests/test_case_cylinder_re100.py`、`docs/case-cylinder-re100/README.md`、`docs/case-cylinder-re100/reference/reference.json`。

## ② 怎么做的（核验方法）

1) 取证：`tensor_dev_missions.result` → `git ls-tree -r agent/m20955` → `git show` 逐文件读源码 → `git branch -a --contains` 判合入。
2) 唯一裁判：直查 `tensor_benchmark_cases` 该 anchor 与 `tensorbench_results WHERE mission_id=20955`（原始回执 `docs/audit-m20955/artifacts/db_evidence.txt`）。
3) repro_cmd：交付 4 文件还原到隔离快照（main 的 src/tests/pyproject + 覆盖交付文件），跑 pytest 与 150 步探针，并校验生产配置能否通过 `SolverConfig.__post_init__`。
4) 集群核对：`cluster_jobs.id=3080` 台账行 vs 节点原始回执 `/srv/tensor/agent/cluster/jobs/cj3080.log`。
5) 集成面：main 树 `grep -rn cylinder_re100` → 0 命中，`verification/__init__.py` 无导出。
6) 数字注入：`docs/audit-m20955/artifacts/build_audit_artifacts.py` 从上述日志解析生成本节全部数字与两张图。

## ③ 计算结果

| 三件套 | 状态 | 机器证据 |
|---|---|---|
| run_artifacts | **无** | 分支树无 `results/`；被核方 artifacts/ 为空目录；`cj3080` 节点回执 rc={receipt['rc']}，`{receipt['output_tail'].strip()}` |
| benchmark ref vs got | **无** | `tensor_benchmark_cases` 该 anchor = {tp['benchmark_ref_vs_got']['detail'].split('-> ')[1].split(' ')[0]} 行；`tensorbench_results` mission 20955 = {ev['benchmark']['tensorbench_results_rows_for_mission']} 行；reference.json 只有文献常量 |
| repro_cmd | 可执行，产出带外 | 生产配置 accepted={ev['production_config_accepted']['accepted']}；探针见下表 |

审计方独立探针（32×16, dt=0.05, 150 步, warmup 75, 单次 {probe['elapsed_s']} s）：

| 量 | got | ref | 相对误差 | 容差 | 判定 |
|---|---|---|---|---|---|
| Cd_mean | {probe['cd_mean']:.4f} | 1.34 | **{cd_err:.2f}%** | ±5% | FAIL |
| St | {probe['st']:.4f} | 0.164 | **{st_err:.2f}%** | ±3% | FAIL |
| Cl 幅值 | {probe['cl_amplitude']:.2e} | — | — | — | 未起涡（舍入噪声，仍报出 St） |

测试与集成：{ev['tests_state']['summary']}；红的两条 = `{'`、`'.join(n.split('::')[-1] for n in ev['tests_state']['failed'])}`，根因 = 测试读 `artifacts/case-cylinder-re100/reference/reference.json`，交付却位于 `docs/case-cylinder-re100/reference/reference.json`（FileNotFoundError）→ mission 自述「6/6 passed」在交付 commit 上不成立。集成调用方 0，未合入 main，平台报告 0 行（report_owed=true / report_id=NULL）。

**台账虚假绿灯（附带发现）**：`cluster_jobs` 行 state=succeeded / rc=0 / note="run rc=0"，而节点原始回执 rc={receipt['rc']}（`cd` 权限拒绝）。只读台账判「生产计算已完成」的后续审计会被骗。

## ④ 结果图

![三件套核验矩阵](fig_evidence_checklist.png)——六条交付面机器可核状态：仅 repro_cmd 与 {ev['tests_state']['summary'].split(' ')[0]}/6 测试为真。
![探针 vs 文献带](fig_probe_vs_literature.png)——审计方粗探针落点 vs Braza/Schäfer-Turek 文献带，两点均在带外。

## ⑤ 注意点

1) 本单是交差核验非重做：未代跑 128×48×3000 步生产算例（集群重计算），故「正确网格上能否打到 1.34±5%」仍是**未知**，不是被证伪——判级不含对物理能力的否定。
2) 探针代表性有限（32×16、150 步远不足以定常化），{cd_err:.1f}%/{st_err:.1f}% 只能读作"未收敛/未起涡"，不可外推为生产精度。
3) 两处口径不一致已标注：mission 6/6 vs 交付树 4/6；台账 rc=0 vs 回执 rc=1。
4) 结构性缺口：TensorFVM 至今无任何圆柱绕流 referee 行（`repo='TensorFVM' AND (anchor_slug ILIKE '%cyl%' OR case_name ILIKE '%cyl%')` → 0 行），该锚在裁判表层**无法被判定**，这是不能给 L4 的机器原因。
5) 交付只在 `agent/m20955`，main 用户拿不到该模块；即便修好测试仍是仓外资产。

## ⑥ 误差判定（逐项 + 显式回答）

| 关键量 | 基准 | 交付 got | 误差 | 是否 <3% |
|---|---|---|---|---|
| Cd_mean | 1.34 | **无交付值**（探针 {probe['cd_mean']:.4f}） | 无法判定（探针 {cd_err:.2f}%） | **否** |
| St | 0.164 | **无交付值**（探针 {probe['st']:.4f}） | 无法判定（探针 {st_err:.2f}%） | **否** |
| Cl 幅值（脱落存在性） | >0 | 探针 {probe['cl_amplitude']:.2e} | — | **否**（未起涡） |
| 测试通过率 | 6/6（自述） | 4/6 | 33% 未通过 | **否** |

**显式回答「是否 <3%」：否。** 被核方没有任何"计算值 vs 文献值"的数值对照（裁判行 0、结果表 0、run 工件 0）；审计方粗探针 {cd_err:.2f}%/{st_err:.2f}% 亦远超 3% 与 15% 两条线。

## 附：落表与返修清单

`tensor_maturity_verdicts` 已落一行：repo=TensorFVM, scope=m20955 GAP-DIFFLBM-EXT-LIT-CYL, mission_id=20980, level=1, verdict_by=board-audit（id=871）。审计方以 maturity_gate.judge 同构复核：judge(1)→gate_level=1（L2 缺 run_artifacts），故预期裁定 reopened。
返修：①测试引用路径对齐交付位置；②从节点可读的分支检出重跑生产算例；③落 referee 行与 tensorbench_results 的 ref/got；④增加起涡守门（Cl 幅值阈值以下禁报 St）；⑤发布六段报告；⑥合入 main 并接入 verification 导出。
"""

payload = {
    "title": "交差核验 m20955 GAP-DIFFLBM-EXT-LIT-CYL：三件套 0/3，判 L1（reopen）；台账 succeeded 而作业回执 rc=1",
    "summary": ("圆柱绕流 Re=100 文献锚的交差核验：无 run 工件（cj3080 rc=1 权限失败却被台账记为 succeeded）、"
                "裁判表该 anchor 0 行、无平台报告、committed 测试 4/6；审计方独立跑通代码并给出 28.6%/62.6% 带外探针。"
                "判 L1，返修。"),
    "scenario_key": "board-audit",
    "mission_id": 20980,
    "tags": ["board-audit", "TensorFVM", "GAP-DIFFLBM-EXT-LIT-CYL", "m20955", "cylinder-Re100", "verdict-L1"],
    "markdown": markdown,
    "figures": [
        fig("fig_evidence_checklist.png", "六条交付面机器可核状态：run_artifacts / referee 行+got / 报告 为 0，仅 repro_cmd 与 4/6 测试为真"),
        fig("fig_probe_vs_literature.png", "审计方 32×16 探针 Cd/St 落点 vs Braza 1986 / Schäfer-Turek 1996 文献带：28.6% / 62.6% 带外"),
    ],
}
pathlib.Path("/tmp/report.json").write_text(json.dumps(payload, ensure_ascii=False))
print("report.json bytes:", len(json.dumps(payload)), "figures:", [f["name"] for f in payload["figures"]])
print(pytest_log.strip().splitlines()[-1])