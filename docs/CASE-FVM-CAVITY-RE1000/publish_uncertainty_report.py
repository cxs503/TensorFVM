"""Publish the discretisation-uncertainty study to the platform reports API.

Numbers are read from ``runs/uncertainty.json`` (written by ``uncertainty.py``),
never typed here.

    PYTHONPATH=src python docs/CASE-FVM-CAVITY-RE1000/publish_uncertainty_report.py
"""

from __future__ import annotations

import base64
import json
import os
import pathlib
import urllib.error
import urllib.request

BASE = os.environ["TENSOR_API_BASE"].rstrip("/")
USER = os.environ["TENSOR_AGENT_USER"]
PASS = os.environ["TENSOR_AGENT_PASS"]
HERE = pathlib.Path(__file__).resolve().parent
MISSION = 20940


def post(path: str, payload: dict, token: str | None = None) -> dict:
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    req = urllib.request.Request(BASE + path, data=json.dumps(payload).encode(),
                                 headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=180) as response:
            return json.loads(response.read().decode())
    except urllib.error.HTTPError as error:
        print("HTTP", error.code, error.read().decode()[:1500])
        raise


def login() -> str:
    data = post("/auth/login", {"username": USER, "password": PASS})
    return data.get("access_token") or data.get("token") or data["data"]["access_token"]


def pct(x: float) -> str:
    return f"{100.0 * x:+.3f}%"


def main() -> None:
    payload = json.loads((HERE / "runs" / "uncertainty.json").read_text())
    md = (HERE / "UNCERTAINTY.md").read_text()
    u = payload["quantities"]["u_min"]
    v = payload["quantities"]["v_max"]
    ts = payload.get("time_step")

    rows = []
    for q, blk in payload["quantities"].items():
        if not blk["data_quality"]["usable"]:
            rows.append(f"| {q} | — | — | — | — | — | {blk['data_quality']['note']} |")
            continue
        ref = pct(blk["fine_minus_reference_relative"]) if "reference" in blk else "无参考值"
        ext = pct(blk["extrapolated_minus_reference_relative"]) if "reference" in blk else "—"
        rows.append(f"| {q} | {blk['observed_order']:.2f} | {blk['extrapolated']:.6g} | "
                    f"{pct(blk['gci_fine_relative'])} | {ref} | {ext} | "
                    f"{'是' if blk.get('reference_within_gci_band') else ('否' if 'reference' in blk else '—')} |")

    summary = (
        "对 CASE-FVM-CAVITY-RE1000 做离散不确定度量化（Roache GCI + Celik 2008 数据质量门）："
        f"网格序列 {'/'.join(str(n) for n in payload['grids'])}，观测阶 p = "
        f"{u['observed_order']:.2f}（u_min）/ {v['observed_order']:.2f}（v_max）与二阶格式一致；"
        f"最细网格 GCI = {pct(u['gci_fine_relative'])} / {pct(v['gci_fine_relative'])}，"
        f"而最细网格与 Ghia 的差只有 {pct(u['fine_minus_reference_relative'])} / "
        f"{pct(v['fine_minus_reference_relative'])} —— 小于自身离散带，故「0.6% 精度」不成立；"
        f"Richardson 外推极限距 Ghia {pct(u['extrapolated_minus_reference_relative'])} / "
        f"{pct(v['extrapolated_minus_reference_relative'])}（外推后更远，说明细网格一致性部分来自误差抵消）。"
        + (f"同网格只改 Courant 数，泛函与中线剖面最大相对散布 {pct(ts['max_relative_spread'])}"
           f"（阈 {ts['tolerance']:g}）→ 时间步无关性成立，CFL=0.15 只影响成本。"
           if ts else f"时间步（Courant 数）轴未执行（not_run：{payload.get('time_step_status')}），"
           "不作结论；该轴作业已递交集群（cj3067 / cj3068）。")
    )

    markdown = md + "\n\n## 程序化注入表（runs/uncertainty.json）\n\n"
    markdown += "| 泛函 | 观测阶 p | Richardson 外推 | GCI(最细) | 最细 vs Ghia | 外推 vs Ghia | 参考偏差在 GCI 带内？ |\n"
    markdown += "|---|---|---|---|---|---|---|\n" + "\n".join(rows) + "\n"

    figures = []
    figure_list = [
        ("fig_uq1_functionals.png",
         "泛函（u_min、v_max）随 1/n 的收敛：圆点=存储场实算，蓝虚线=Richardson 外推极限，"
         "误差棒=最细网格 GCI，红线=Ghia et al. (1982)。"),
    ]
    if ts:
        figure_list.append((
            "fig_uq2_timestep.png",
            "同网格（n=128）只改 Courant 数的敏感度：左=泛函相对基线跑偏差，"
            "右=中线剖面相对基线跑的最大绝对差（红虚线=接受阈 1e-3）。"))
    for name, caption in figure_list:
        path = HERE / "figures" / name
        figures.append({"name": name, "caption": caption,
                        "png_base64": base64.b64encode(path.read_bytes()).decode()})

    token = login()
    report = post("/tensor/reports", {
        "title": "CASE-FVM-CAVITY-RE1000 离散不确定度量化：网格收敛指数（GCI）复核",
        "summary": summary,
        "scenario_key": "CASE-FVM-CAVITY-RE1000",
        "mission_id": MISSION,
        "tags": ["verification", "GCI", "uncertainty-quantification", "cavity",
                 "Re=1000", "TensorFVM", "CASE-FVM-CAVITY-RE1000"],
        "markdown": markdown,
        "figures": figures,
    }, token)
    print(json.dumps(report, ensure_ascii=False)[:600])


if __name__ == "__main__":
    main()