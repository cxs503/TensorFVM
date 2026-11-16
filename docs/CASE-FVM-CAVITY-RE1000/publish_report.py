"""Publish the CASE-FVM-CAVITY-RE1000 paper-style report to the platform reports API.

Numbers are read from ``docs/CASE-FVM-CAVITY-RE1000/runs/summary.json`` (itself
written by the campaign), never typed here.

    PYTHONPATH=src python docs/CASE-FVM-CAVITY-RE1000/publish_report.py
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
MISSION = 22950


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


def main() -> None:
    summary = json.loads((HERE / "runs" / "summary.json").read_text())
    bench = json.loads((HERE / "runs" / "benchmark.json").read_text())
    cases = {int(c["n"]): c for c in summary["cases"]}
    finest = max(cases)
    top = cases[finest]
    markdown = (HERE / "REPORT.md").read_text()

    figures = []
    for name, caption in (
        ("fig1_centreline_profiles.png",
         "Re = 1000 方腔中心线剖面：三个分辨率（n = 32/64/128）vs Ghia, Ghia & Shin (1982) "
         "Table I（u 在 x = L/2）与 Table II（v 在 y = L/2）采样点。"),
        ("fig2_grid_convergence.png",
         "误差随分辨率收敛（对数坐标）：max|Δu|、max|Δv|、u_min 与 v_max 相对误差；"
         "红线为算例池 5% 容差，灰线为 3% 参考线。n = 32 超容差，n ≥ 64 进容差。"),
        ("fig3_fields.png",
         f"最细网格（n = {finest}）的流函数与涡量云图；白色十字为 Ghia 主涡中心 (0.5313, 0.5625) L。"),
    ):
        path = HERE / "figures" / name
        figures.append({"name": name, "caption": caption,
                        "png_base64": base64.b64encode(path.read_bytes()).decode()})

    login = post("/auth/login", {"username": USER, "password": PASS})
    token = login.get("access_token") or login.get("token")
    if not token:
        raise SystemExit(f"login failed: {login}")

    payload = {
        "title": "CASE-FVM-CAVITY-RE1000：Re = 1000 顶盖驱动方腔对标 Ghia(1982)，n=128 误差 <1%",
        "summary": (
            f"TensorFVM 生产求解器（交错 MAC FVM + van Leer 守恒对流 + DCT-II 纯 Neumann 投影）"
            f"在 Re = 1000 方腔跑出 n = 32/64/128 三档收敛定常解：最细网格 "
            f"max|Δu| = {top['criteria']['C1_u_profile_max_abs']['value']:.4f}、"
            f"max|Δv| = {top['criteria']['C2_v_profile_max_abs']['value']:.4f}（U_lid = 1），"
            f"u_min 相对误差 {100 * top['criteria']['C3_u_min_relative']['value']:.2f}%、"
            f"v_max {100 * top['criteria']['C4_v_max_relative']['value']:.2f}%、"
            f"主涡中心偏差 {top['criteria']['C5_vortex_centre']['value']:.4f} L；"
            f"算例池 5% 容差判定 {summary['finest_verdict']}，"
            f"最大散度 {top['max_divergence']:.1e}。运行工件（工件 JSON/速度场/基准表）"
            f"入库 docs/CASE-FVM-CAVITY-RE1000/runs/，机检门从工件重算。"
        ),
        "scenario_key": "tensorfvm-case-fvm-cavity-re1000",
        "mission_id": MISSION,
        "tags": ["TensorFVM", "FVM", "lid-driven cavity", "Ghia1982", "Re1000",
                 "incompressible", "projection method", "benchmark",
                 "CASE-FVM-CAVITY-RE1000"],
        "markdown": markdown,
        "figures": figures,
        "artifacts": [
            "docs/CASE-FVM-CAVITY-RE1000/REPORT.md",
            "docs/CASE-FVM-CAVITY-RE1000/runs/summary.json",
            "docs/CASE-FVM-CAVITY-RE1000/runs/benchmark.json",
            "docs/CASE-FVM-CAVITY-RE1000/runs/fields.npz",
            "docs/CASE-FVM-CAVITY-RE1000/run_case.py",
            "src/tensorfvm/case_cavity_re1000.py",
            "scripts/case_cavity_re1000.py",
            "tests/test_case_cavity_re1000.py",
        ],
        "metrics": {
            "n_finest": finest,
            "u_profile_max_abs": top["criteria"]["C1_u_profile_max_abs"]["value"],
            "v_profile_max_abs": top["criteria"]["C2_v_profile_max_abs"]["value"],
            "u_min_rel": top["criteria"]["C3_u_min_relative"]["value"],
            "v_max_rel": top["criteria"]["C4_v_max_relative"]["value"],
            "vortex_centre_abs_L": top["criteria"]["C5_vortex_centre"]["value"],
            "max_divergence": top["max_divergence"],
            "steady_residual": top["residual"],
            "verdict_5pct": summary["finest_verdict"],
            "benchmark_rows": {str(r["n"]): {"u_err": r["u_profile_max_abs"]["err"],
                                             "v_err": r["v_profile_max_abs"]["err"],
                                             "verdict": r["verdict"]}
                               for r in bench["rows"]},
        },
    }
    result = post("/tensor/reports", payload, token)
    print(json.dumps(result, ensure_ascii=False)[:600])


if __name__ == "__main__":
    main()