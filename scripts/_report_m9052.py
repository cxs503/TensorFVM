"""Publish the P475W1-N06 paper-style report to the platform reports API."""
import base64
import json
import os
import urllib.error
import urllib.request

BASE = os.environ["TENSOR_API_BASE"].rstrip("/")
USER = os.environ["TENSOR_AGENT_USER"]
PASS = os.environ["TENSOR_AGENT_PASS"]

REPO = "/root/tp-agent/work/m9052"
DOCS = f"{REPO}/docs/vof-chain-anchor"


def post(path, payload, token=None):
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


login = post("/auth/login", {"username": USER, "password": PASS})
token = login.get("access_token") or login.get("token")
if not token:
    raise SystemExit(f"login failed: {login}")

summary = json.load(open(f"{DOCS}/summary.json"))
gates = summary["gates"]
markdown = open(f"{DOCS}/report.md").read()

figures = []
for name, caption in (
    ("fig1-hydrostatic-negative-control.png",
     "G1 negative control: max|u| of a resting two-layer column vs interface "
     "offset. Closed-form profile (dashed) leaves a rho*g imbalance at partial "
     "cells (|u| up to 1.9e-2 m/s); the balanced discrete profile (solid) holds "
     "to 7.5e-16 m/s."),
    ("fig4-analytic-anchors.png",
     "Analytic anchors. Left: gate-surface free fall h = 2a - g t^2/2 vs the "
     "measured h* for the tall column. Right: shallow dam-break front vs the "
     "Ritter solution 2 sqrt(g h0) t (measured -62%)."),
    ("fig3-dam-break-field.png",
     "Liquid-fraction field of every dam-break leg at the final time with the "
     "alpha = 0.5 contour; shows the surge sharpness each resolution transports."),
):
    with open(f"{DOCS}/{name}", "rb") as handle:
        figures.append({"name": name, "caption": caption,
                        "png_base64": base64.b64encode(handle.read()).decode()})

payload = {
    "title": "P475W1-N06 两相流 VOF 链：静水平衡负控 + 溃坝解析锚（TensorFVM）",
    "summary": (
        "TensorFVM 两相 VOF 链：新增离散平衡水静压初始化（静置 |u| 7.5e-16 vs "
        "闭式剖面 1.9e-2 m/s），并建立五道验收门（静水静置/界面不迁移/PLIC 质量"
        "守恒/自由落体表面锚/Ritter+Martin&Moyce 波前锚）。G1/G2/G3/G4a 通过，"
        "G4b/G4c honest FAIL：重力流启动滞后约 0.34 a/c0，气相密度、初始水头、"
        "分辨率与 pressure_form 均已用测量排除，根因未定。"
    ),
    "scenario_key": "tensorfvm-two-phase-vof-chain",
    "mission_id": 9052,
    "tags": ["TensorFVM", "two-phase", "VOF", "PLIC", "CSF", "hydrostatic",
             "dam-break", "Ritter", "P475W1-N06"],
    "markdown": markdown,
    "figures": figures,
    "artifacts": ["docs/vof-chain-anchor/report.md",
                  "docs/vof-chain-anchor/summary.json",
                  "src/tensorfvm/verification/vof_chain.py",
                  "src/tensorfvm/verification/vof_chain_anchor.py",
                  "tests/test_vof_chain_hydrostatic.py"],
    "metrics": {
        "G1_max_velocity_balanced_m_s": gates["G1_hydrostatic_velocity"]["worst_max_velocity_discrete_m_s"],
        "G1_max_velocity_closed_form_m_s": gates["G1_hydrostatic_velocity"]["worst_max_velocity_analytic_m_s"],
        "G2_interface_displacement_balanced": gates["G2_interface_immobility"]["worst_displacement_discrete"],
        "G3_worst_final_volume_drift_plic": max(
            abs(e["final_volume_drift"]) for e in gates["G3_mass_conservation"]["legs"]
            if "upwind" not in e["name"]),
        "G4a_freefall_worst_relative_error": gates["G4a_freefall_surface"]["worst_relative_error"],
        "G4b_ritter_worst_relative_error": gates["G4b_ritter_front"]["worst_relative_error"],
        "G4c_martin_moyce_worst_relative_error": gates["G4c_martin_moyce_front"]["worst_relative_error"],
        "all_pass": gates["all_pass"],
    },
}
result = post("/tensor/reports", payload, token)
print(json.dumps(result)[:600])