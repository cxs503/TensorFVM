#!/usr/bin/env python3
"""Publish m15222 audit report to tensor platform (login + POST /tensor/reports)."""
import base64, json, os, urllib.request

BASE = os.environ["TENSOR_API_BASE"]
USER = os.environ["TENSOR_AGENT_USER"]
PASS = os.environ["TENSOR_AGENT_PASS"]

def post(path, payload, token=None):
    req = urllib.request.Request(BASE + path, data=json.dumps(payload).encode(),
                                  headers={"Content-Type": "application/json"}, method="POST")
    if token:
        req.add_header("Authorization", "Bearer " + token)
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read().decode())

tok = post("/auth/login", {"username": USER, "password": PASS})
token = tok.get("access_token") or tok.get("token") or (tok.get("data") or {}).get("access_token")
print("login ok, token?", bool(token))

md = open("/root/tp-agent/work/m15222/docs/audit-m15142-sod1978.md").read()

figs = []
for name, cap in [
    ("sod_profiles_t0.14.png", "Sod t=0.14 rho/u/p profiles: TensorFVM HLLC (nx=400, order2) vs exact solution (published baseline comparison, m15142 artifact)"),
    ("sod_convergence.png", "L1 error vs grid: Sod solution (0.62-0.66/0.86-0.98, literal criterion FAIL) and smooth entropy wave design order (0.99/2.05-2.13, within 2% band)"),
]:
    p = "/root/tp-agent/work/m15142/docs/repro-fvm-sod1978/artifacts/" + name
    figs.append({"name": name, "caption": cap,
                 "png_base64": base64.b64encode(open(p, "rb").read()).decode()})

payload = {
    "title": "交差审计 m15142 REPRO-FVM-SOD1978：Sod(1978) 复现三件套核验 + L4 判级落表（verdict id=261）",
    "summary": "审计 m15142（Sod 1978 复现，TensorFVM，PR#419 已并 main）：run_artifacts 双处存在；C1 四波位置 4/4 PASS（max 0.365%<2%）、C2 星区 4/4 PASS（max 0.014%<5%）、C3 L1 阶字面判据 honest FAIL（0.62-0.66/0.86-0.98<1，光滑熵波设计阶 1.00/2.05 达标）；21/21 测试重跑绿；repro_cmd 独立重跑确定性复现（8/8 项一致）；报告 14285 七段齐全、数字与工件零偏差。判级 L4（基准验证：完整算例+误差分析+可复现）。缺口：tensor_benchmark_cases 无本锚点行（台账缺口，建议补登记）。verdict 已落表 id=261。",
    "scenario_key": "board-audit",
    "mission_id": 15222,
    "tags": ["board-audit", "REPRO-FVM-SOD1978", "TensorFVM", "maturity-verdict", "L4"],
    "markdown": md,
    "figures": figs,
}
res = post("/tensor/reports", payload, token)
print("report result:", json.dumps(res, ensure_ascii=False)[:400])
