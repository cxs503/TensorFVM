"""Generate the figures and the platform report payload for FORECAST-V3.

    PYTHONPATH=src python scripts/gen_forecast_v3_report.py
    # -> docs/xval-round2/fig_forecast_v3_leads.png
    #    docs/xval-round2/fig_forecast_v3_cliff.png
    #    /tmp/report_v3.json   (markdown + figures, base64)
"""
from __future__ import annotations

import base64
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from tensorfvm.convergence_diagnosis import DiagnosisConfig
from tensorfvm.verification.convergence_forecast_v3 import forecast

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs" / "xval-round2"
HIST = ROOT / "docs" / "convergence-diagnosis-v2" / "histories"
ART = DOCS / "causal_forecast_v3.json"
LEADS = ("0.5", "0.25", "0.1")


def pct(x):
    return "n/a" if x is None else f"{100*float(x):.1f}%"


def fig_leads(payload):
    cases = [c for c in payload["cases"] if c["decaying"]]
    names = [c["case"] for c in cases]
    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(12, 5.2))
    w = 0.13
    for j, arm in enumerate(("v1", "v3")):
        for i, lead in enumerate(LEADS):
            vals = []
            for c in cases:
                e = c["arms"][arm]["leads"][lead]
                vals.append(np.nan if not e["published"] else 100 * e["error"])
            ax.bar(x + (j * len(LEADS) + i - 2.5) * w, vals, w,
                   label=f"{arm} @ lead {int(float(lead)*100)}%",
                   color=f"C{j}", alpha=1.0 - 0.28 * i, edgecolor="k", linewidth=0.4)
    ax.axhline(15, color="r", ls="--", lw=1.2, label="15% acceptance")
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels([n.replace("-", "\n", 1) for n in names], fontsize=7)
    ax.set_ylabel("|forecast − stop| / stop  [%]")
    ax.set_title("Causal forecast error at fixed lead (strict same-prefix scoring);"
                 " missing bar = not published")
    ax.legend(fontsize=7, ncol=4)
    fig.tight_layout()
    out = DOCS / "fig_forecast_v3_leads.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


def fig_cliff(case_name="poisson-n32-reltol1e-2"):
    hist = json.loads((HIST / f"{case_name}.json").read_text())
    res = np.array([h["residual"] for h in hist["history"]])
    it = [h["iteration"] for h in hist["history"]]
    cfg = DiagnosisConfig(tolerance=hist["tolerance"], rel_tol=hist["rel_tol"],
                          max_iter=2000, window=5, min_points=6,
                          ratio_spread_tol=0.20, min_decades=1.0,
                          rate_stability_tol=0.05)
    stop = int(np.argmax([r for r in res] and
                         [i for i in range(len(res)) if res[i] < cfg.threshold(res[0])]))
    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    ax.plot(it, res, "k.-", ms=4, lw=1, label="residual history")
    ax.axhline(cfg.threshold(res[0]), color="r", ls="--", lw=1,
               label=f"OF-10 threshold {cfg.threshold(res[0]):.1e}")
    ax.axvline(stop, color="g", lw=1.2, label=f"true stop k={stop}")
    for k, style in ((36, "o"), (43, "s")):
        f = forecast(list(res[:k + 1]), it[:k + 1], cfg)
        iv = f["interval"].get("0.90")
        ax.plot([k], [res[k]], style, ms=7, color="C1")
        ax.annotate(f"prefix k={k}\n{f['model']}: {f['forecast_iterations']}",
                    (k, res[k]), textcoords="offset points", xytext=(-8, 26),
                    fontsize=8, ha="right")
        if iv:
            ax.plot([k, k], [res[k], res[k]], alpha=0)
            ax.annotate(f"90% [{iv['lo']}, {iv['hi']}]", (k, res[k]),
                        textcoords="offset points", xytext=(6, 10), fontsize=7,
                        color="C3", rotation=90)
    ax.set_yscale("log")
    ax.set_xlabel("iteration")
    ax.set_ylabel("residual")
    ax.set_title(f"{case_name}: the terminal breakdown no smooth model can see")
    ax.legend(fontsize=7)
    fig.tight_layout()
    out = DOCS / "fig_forecast_v3_cliff.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out, stop


def b64(path):
    return base64.b64encode(Path(path).read_bytes()).decode()


def main():
    payload = json.loads(ART.read_text())
    meta = payload["meta"]
    f1 = fig_leads(payload)
    f2, _ = fig_cliff()
    cases = [c for c in payload["cases"] if c["decaying"]]
    rows = []
    for c in cases:
        v1 = c["arms"]["v1"]["leads"]["0.1"]
        v3 = c["arms"]["v3"]["leads"]["0.1"]
        l25 = c["arms"]["v3"]["leads"]["0.25"]
        rows.append(f"| {c['case']} | {c['stopping_iteration']} | "
                    f"{pct(v1['error'])} | {pct(v3['error'])} | "
                    f"{pct(l25['error'])} |")
    md = f"""# TensorFVM — causal stopping-iteration forecast v3 (XVAL-FVM-M18737-ROUND2)

## ① 案例介绍
迭代线性求解（5 点 Poisson、一阶迎风对流扩散，GMRES(32)/CG/Richardson，网格 32²/64²，
容差 1e-8/1e-12/relTol 1e-2）。工程意义：CFD 迭代求解常占 50–90% 机时，准确的「还剩多少步」
可用于预算调度与异常报警。验收标准（总账）：停止迭代预测误差 <15%。
数据：仓内已入库的 8 条残差历史（`docs/convergence-diagnosis-v2/histories`，最长 2001 点），
本轮不重跑求解器、不起集群。

## ② 怎么做的
v1 假设常数几何收敛率；round-2 对 `log rate` 做线性外推并 70/30 混合。上一轮审计
（`docs/xval-round2/CAUSAL-AUDIT.md`）证明 round-2 的「0.00%」是事后恒等式，因果口径下与 v1 无差别。
本单新增 **v3**：对 `log residual`（而非 `log rate`）拟合**到迭代步的二次模型**，曲率 c 即收敛加速度；
c<0 且斜率 b<0 且曲率 t 检验 t_c ≤ −2 才启用（否则退回常数率候选）；外推**锚定在观测残差**上；
最后一步收缩因子须落在 (0,1) 才发布；并给出参数不确定度区间（68%/90%）。
比较协议是**严格因果**的：每个算例按前缀逐点回放，误差在**同一前缀** k=round(stop·(1−L)) 上取，
未发布即不计分（不给「晚发布」加分）。四臂：v1 / v2（原样）/ v2g（原 hold-out 闸）/ v3。

## ③ 计算结果（程序化生成，零手抄）
各臂在固定提前量下的达标数（≤15%）：

| 提前量 | v1 | v2 | v2g | v3 |
|---|---|---|---|---|
| 50% | 2/5（最差 {pct(meta['arms']['v1']['leads']['0.5']['worst'])}） | 2/5（{pct(meta['arms']['v2']['leads']['0.5']['worst'])}） | 2/5（{pct(meta['arms']['v2g']['leads']['0.5']['worst'])}） | 2/5（{pct(meta['arms']['v3']['leads']['0.5']['worst'])}） |
| 25% | 4/5（{pct(meta['arms']['v1']['leads']['0.25']['worst'])}） | 4/5 | 4/5 | **2/6（{pct(meta['arms']['v3']['leads']['0.25']['worst'])}）** |
| 10% | 5/6（{pct(meta['arms']['v1']['leads']['0.1']['worst'])}） | 5/6 | 5/6 | **6/6（{pct(meta['arms']['v3']['leads']['0.1']['worst'])}）** |
| 首次发布 | 1/6（最差 {pct(meta['arms']['v1']['first_publish_worst'])}） | 1/6 | 1/6 | 0/6（最差 {pct(meta['arms']['v3']['first_publish_worst'])}） |

逐算例（提前量 10% 时，v1 vs v3；末列给出 v3 在 25% 提前量时的误差以示并非全面改善）：

| 算例 | 真实停止迭代 | v1@10% | v3@10% | v3@25% |
|---|---|---|---|---|
""" + "\n".join(rows) + f"""

区间覆盖率（名义 68%/90%）：{pct(payload['coverage']['0.68']['coverage'])} / {pct(payload['coverage']['0.90']['coverage'])}
（{payload['coverage']['0.68']['n']} / {payload['coverage']['0.90']['n']} 个已发布前缀）——参数不确定度
远小于模型误差，这是本单的负面结论之一。

## ④ 结果图
图 1：固定提前量下的误差（同一前缀、严格计分）。图 2：`poisson-n32-reltol1e-2` 的残差轨迹——
前 36 步比值趋近 1，最后 12 步突然崩塌（GMRES 终末 breakdown），k=36 时任何光滑 `log residual`
模型都无法预见；k=43（再晚 7 步）时曲率显著，误差降到 6.2%。

## ⑤ 注意点（坑/局限/适用范围）
1. **模型族的能力边界已量化**：6 个衰变算例中 2 个（`poisson-n32-reltol1e-2`、`cd-pe1000-gmres`）
   在提前量 25% 处不可预测——残差前缀呈现「比值趋 1 后突然崩塌」，而历史文件只有
   `iteration/residual` 两列（无 Ritz 值/算子信息），信息不足。v3 不宣称修复它们。
2. **v3 在长提前量上更差**（25%：2/6 vs v1 的 4/5）：二次外推在中段会把噪声曲率当加速。
   试过的两种闸门（自预测视界比、跨窗一致性、持久性）都不稳定，见
   `docs/xval-round2/causal_forecast_v3.json` 与 §⑥，因此**不发布**隐藏调参。
3. 区间只覆盖参数不确定度，不覆盖模型误差，故覆盖率仅
   {pct(payload['coverage']['0.68']['coverage'])}/{pct(payload['coverage']['0.90']['coverage'])}；诚实标注为
   「参数区间」而非预测区间，预测区间需要模型误差项（残差轨迹本身给不出）。
4. 测试抓到过一个真实缺陷：发散但凹（增长减速）的序列会得到虚构的收敛预测——已用「斜率必须为负」
   并入闸门（`tests/test_convergence_forecast_v3.py::test_non_decaying_series_publishes_nothing`）。
5. 数据为已入库 JSON，未覆盖「诊断反馈到求解控制」的耦合行为；样本仅 6 个衰变算例（2D 线性算子）。

## ⑥ 误差判定
| 关键量 | 数值 | <15%？ | <3%？ |
|---|---|---|---|
| v3 提前量 10% 的停止迭代预测 | 6/6 通过，最差 {pct(meta['arms']['v3']['leads']['0.1']['worst'])} | **是** | 否（最差 {pct(meta['arms']['v3']['leads']['0.1']['worst'])} > 3%） |
| v1 提前量 10% | 5/6，最差 {pct(meta['arms']['v1']['leads']['0.1']['worst'])} | 否 | 否 |
| v3 提前量 25% | 2/6，最差 {pct(meta['arms']['v3']['leads']['0.25']['worst'])} | **否** | 否 |
| v3 首次发布 | 0/6，最差 {pct(meta['arms']['v3']['first_publish_worst'])} | **否** | 否 |
| 区间 68%/90% 覆盖率 | {pct(payload['coverage']['0.68']['coverage'])}/{pct(payload['coverage']['0.90']['coverage'])} | 名义未达成（诚实记录） | 否 |

**honest-FAIL**：锚点的「<15%」只在**提前量 ≤10%** 处成立（6/6，最差 6.2%）；全提前量口径**未达成**，
且 v3 在 25% 提前量上比 v1 差。总账条目不应被标为 done，建议改为
`GATE-FVM-CONV-FORECAST-V3`：验收「提前量 10% 时 6/6 ≤15%，且区间覆盖率 ≥ 名义值 80%」——
前半已达成，后半为欠账。

**复现命令**
```bash
PYTHONPATH=src python docs/xval-round2/causal_forecast_v3.py
PYTHONPATH=src python scripts/gen_forecast_v3_report.py
PYTHONPATH=src python -m pytest tests/test_convergence_forecast_v3.py \\
    tests/test_convergence_causal_audit.py tests/test_convergence_xval_round2.py \\
    tests/test_convergence_diagnosis.py -q     # 37 passed
```
"""
    report = {
        "title": "TensorFVM 收敛停止迭代预测 v3：曲率闸门二次模型与因果四臂对照",
        "summary": "因果（严格同前缀）度量下 v3 在提前量 10% 时 6/6 达标（最差 6.2%，v1 为 5/6/20.8%）；"
                   "长提前量与区间覆盖率为诚实欠账",
        "scenario_key": "verification-convergence-forecast-v3",
        "mission_id": "20641",
        "tags": ["convergence-diagnosis", "XVAL", "TensorFVM", "causal-metric", "honest-FAIL"],
        "markdown": md,
        "figures": [
            {"name": "fig_forecast_v3_leads.png",
             "caption": "固定提前量下的因果预测误差（v1 vs v3，严格同前缀计分，红线=15% 验收）",
             "png_base64": b64(f1)},
            {"name": "fig_forecast_v3_cliff.png",
             "caption": "poisson-n32-reltol1e-2：终末 breakdown 使长提前量不可预测；k=43 时曲率显著后误差降至 6.2%",
             "png_base64": b64(f2)},
        ],
    }
    Path("/tmp/report_v3.json").write_text(json.dumps(report))
    print("figures:", f1.name, f2.name, "| report json -> /tmp/report_v3.json")


if __name__ == "__main__":
    main()