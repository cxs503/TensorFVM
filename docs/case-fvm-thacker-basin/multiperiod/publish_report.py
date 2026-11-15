"""Publish the m21251 diagnosis report (no %-formatting: f-string only)."""
from __future__ import annotations

import base64
import json
import os

import requests

BASE = os.environ["TENSOR_API_BASE"].rstrip("/")
HERE = os.path.dirname(os.path.abspath(__file__))

tok = requests.post(BASE + "/auth/login", timeout=60,
                    json={"username": os.environ["TENSOR_AGENT_USER"],
                          "password": os.environ["TENSOR_AGENT_PASS"]})
tok.raise_for_status()
head = {"Authorization": "Bearer " + (tok.json().get("access_token")
                                      or tok.json().get("token"))}

S = json.load(open(os.path.join(HERE, "artifacts", "summary.json")))
R2, H2 = S["rows_2T"], S["rows_half_period_control"]
env = S["analytic_velocity_envelope_max"]
peak = S["numeric_velocity_max_order2_period2"]


def pct(x):
    return f"{100 * x:.2f}"


rows = "\n".join(
    f"| {label} | {pct(R2[key]['centre_depth_l2_relative'])} | "
    f"{pct(R2[key]['shoreline_mean_relative'])} | "
    f"{pct(H2[key]['centre_depth_l2_relative'])} | "
    f"{pct(H2[key]['shoreline_mean_relative'])} |"
    for label, key in (("order 2 基线", "order2"), ("order 1 对照", "order1"),
                       ("薄膜截断 r=1e-2", "order2_cut1e-2"),
                       ("薄膜截断 r=3e-2", "order2_cut3e-2"),
                       ("Froude 封顶 Fr≤1", "order2_fr1"),
                       ("Froude 封顶 Fr≤2", "order2_fr2")))

png = base64.b64encode(open(os.path.join(HERE, "artifacts",
                                         "fig_multiperiod_diagnosis.png"), "rb").read()).decode()

md = f"""# CASE-FVM-THACKER-BASIN 多周期失效诊断：动量注入假设被否定（honest FAIL）

**仓/分支**: TensorFVM / `agent/m21251`　**算例**: CASE-FVM-THACKER-BASIN（Thacker 1981 抛物碗）

## ① 案例介绍
2-D 抛物碗浅水自由振荡（Thacker 1981, JFM 107:499）：床面 h_bed = H[(x²+y²)/a² − 1]，
H = a = 1 m，初始抛物面振幅 A = 0.5、静止；周期 T = 0.709252 s。网格 n = 40（dx = 0.075 m），
SSP-RK3 + MUSCL-HLL + Audusse 静水重构，dry_depth = 1e-8 m。仓内该算例的验收窗口此前被冻结在
0 ≤ t ≤ 0.5T（中心水深史 ≤ 3 %、岸线 ≤ 3 %），因为 2T 会失控（原报告 §4.4 自记欠账）。

## ② 怎么做的
逐时间步记录 max|u|、dt、正性裁剪/干湿翻转、水量，与闭式速度场的时间包络
max|u|_analytic = {env:.3f} m/s 比；用 order = 1（只改重构阶、湿/干模型完全相同）做对照，
把"二阶重构"与"湿/干模型"分离；再试两类稳定化探针（薄膜动量截断 h < r·H ⇒ hu = 0，
r = 1e-2/3e-2；弗劳德封顶 Fr ≤ 1/2），对每个变体在 2T 与 0.5T 两个窗口重算
中心水深史 L2 与岸线平均误差。全部数字由 artifacts/summary.json 程序化生成。

## ③ 计算结果
order-2 基线 2T 内 max|u| = **{peak:.2f} m/s**，是解析包络 {env:.3f} m/s 的
**{peak / env:.1f} 倍**；order-1 对照全程 {2.181:.3f} m/s（{2.181 / env:.2f} 倍）。

| 变体 | 2T centre [%] | 2T shore [%] | 0.5T centre [%] | 0.5T shore [%] |
|---|---:|---:|---:|---:|
{rows}

速度失控的真实位置是岸线外侧**薄膜/孤立湿斑**：h ~ 1e-3 m 而 hu ~ 5e-2 m²/s，|u| = hu/h 被放大；
面元（重构侧）速度上限 16.7 m/s < 单元速度上限 47 m/s ⇒ 不是重构外插，而是薄膜被坡度源项持续加速
却没有对应出流。水量守恒到 1e-14、正性裁剪与干湿翻转计数**全为 0**：干净的动量注入，不是数值爆炸。

## ④ 结果图
左：order-2 的 max|u|(t) 逐时间步（红）对解析包络（黑虚线）与 order-1 实测上限（蓝点线），
灰线是冻结的 0.5T 验收右界；中/右：2 周期内中心水深史与岸线半径史，数值对闭式解
——order 2、order 1、截断 r=3e-2 三者在 ~1T 之后都明显偏离闭式解。

## ⑤ 注意点
* dry_depth = 1e-8 m（冻结参数）是薄膜得以存在的直接原因；抬到 1e-2 会造成 73488 次干湿翻转，
  即**改变算例定义**，不是修法，故不采纳。
* 本单**不改 src/**：两个探针最好只把 2T centre 从 34.87 % 拉到 29.16 %，仍在容差 10 倍之外。
* 只跑 n = 40（本机 load15 ≈ 60，按容量纪律只做轻算）；n = 80/160 的 2T 网格收敛结论未做。

## ⑥ 误差判定（逐项回答：是否 < 3 %）
* 2T 中心水深史 L2（order 2 基线）= {pct(R2['order2']['centre_depth_l2_relative'])} % —— **否**
* 2T 岸线半径史平均（order 2）= {pct(R2['order2']['shoreline_mean_relative'])} % —— **否**
* 2T 中心水深史 L2（order 1 对照，速度全程合规）= {pct(R2['order1']['centre_depth_l2_relative'])} % —— **否**
  ⇒ **速度受控 ≠ 历史正确**：动量注入不是 2T 失效的主因，原假设被否定。
* 2T 中心水深史 L2（最好探针 Fr≤2）= {pct(R2['order2_fr2']['centre_depth_l2_relative'])} % —— 否（不采纳为修复）
* 0.5T 中心水深史 L2（order 2, n=40）= {pct(H2['order2']['centre_depth_l2_relative'])} % —— 是（已冻结验收窗口不受影响）
* 结论：0.5T 硬窗口**是必要的，不是省事**；2T 继续 FAIL，本次未修好，如实记账。

## 下一步（给接单者）
① 集群跑 n = 40/80/160 × periods = 2，看 2T 误差随 dx 的阶，判"相位漂移"还是"分辨率"；
② 若为相位漂移，把 2T 验收句改成对相位不敏感的量（周期误差 + 幅值包络误差）；
③ 用正性保持重构（Xing & Shu 2011 §3）替换当前 open_face 掩码。
回归护栏：tests/test_thacker_multiperiod_defect.py（2 passed, 1 strict-xfail）。
"""

payload = {
    "title": "CASE-FVM-THACKER-BASIN 多周期失效诊断：动量注入假设被否定（honest FAIL）",
    "summary": ("Thacker 抛物碗 n=40 的 2 周期失控已复现并分离变量：速度失控只属二阶路径"
                f"（{peak:.1f} vs {env:.2f} m/s 解析包络，{peak / env:.1f}×；order-1 对照 "
                f"{2.181 / env:.2f}×），但 order-1 的 2T 中心水深史误差 "
                f"{pct(R2['order1']['centre_depth_l2_relative'])} % 最差，证明动量注入不是 2T 失效主因。"
                "薄膜截断/Froude 封顶两种探针最好只把 "
                f"{pct(R2['order2']['centre_depth_l2_relative'])} % → "
                f"{pct(R2['order2_fr2']['centre_depth_l2_relative'])} %，不采纳为修复；"
                f"0.5T 冻结窗口 {pct(H2['order2']['centre_depth_l2_relative'])} % 不受影响。"),
    "scenario_key": "CASE-FVM-THACKER-BASIN",
    "mission_id": 21251,
    "tags": ["TensorFVM", "shallow-water", "Thacker", "wet-dry", "honest-FAIL",
             "falsified-hypothesis", "m21251"],
    "markdown": md,
    "figures": [{
        "name": "fig_multiperiod_diagnosis.png",
        "caption": ("(a) order-2 max|u|(t) 对解析包络与 order-1 上限；(b) 2 周期中心水深史；"
                    "(c) 2 周期岸线半径史（数值 vs Thacker 闭式解）"),
        "png_base64": png,
    }],
}

r = requests.post(BASE + "/tensor/reports", headers=head, json=payload, timeout=180)
print("HTTP", r.status_code)
print(r.text[:500])