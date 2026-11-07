"""Publish the mission report to the platform API (figures downscaled to fit the payload limit)."""
import base64
import io
import json
import os
import pathlib
import urllib.request

from PIL import Image

REPORT = pathlib.Path("/tmp/report_5539.json")
TOKEN_FILE = pathlib.Path("/tmp/login.json")


# ---- build the report from the machine-generated artefact (no hand-copied numbers)
data = json.loads(pathlib.Path("docs/pmm-cfd/pmm_derivatives_summary.json").read_text())
hull, dev, aud, quad = data["hull"], data["deviations"], data["reference_audit"], data["quadrature_convergence"]
f = lambda x, n=4: f"{x:.{n}f}"
rows = []
for k in ("Yv", "Nv", "Yvdot", "Nvdot", "Yr", "Nr", "Yrdot", "Nrdot"):
    d = dev[k]
    slope = f(d["reference_drift_slope"]) if "reference_drift_slope" in d else "\u2014"
    serr = f"{100*d['rel_error_drift_slope']:+.1f} %" if "rel_error_drift_slope" in d else "\u2014"
    rows.append(f"| {k} | {f(d['computed'])} | {f(d['reference_table'])} | {100*d['rel_error']:+.1f} % | {slope} | {serr} |")
rows = "\n".join(rows)

MD = f"""# KVLCC2 \u7eaf\u6a2a\u8361/\u7eaf\u9996\u6447 PMM \u865a\u62df\u8bd5\u9a8c\uff1aCFD \u57fa\u51c6\u8fdb\u5c55\u4e0e\u5ba1\u8ba1\uff08IN40-N04\uff09

## \u2460 \u6848\u4f8b\u4ecb\u7ecd
KVLCC2 \u6cb9\u8f6e\uff08\u6a21\u578b\u5c3a\u5ea6 1:31.6\uff09\uff1aL=7.000 m, B=0.936 m, T=0.342 m, Cb=0.708, LCB=\u22121.688 %L,
Cwp=0.9005, Cm=0.9976\uff1bU=0.771 m/s\uff08Fn=0.142\uff09, \u03c1=998 kg/m\u00b3, \u03bd=8.9e\u22127 m\u00b2/s\uff1bT_osc=10 s\uff08\u03c9'=\u03c9L/U=5.70\uff09\u3002
\u76ee\u6807\uff1a\u7eaf\u6a2a\u8361 y=A\u00b7sin \u03c9t\u3001\u7eaf\u9996\u6447 \u03c8=A\u00b7sin \u03c9t \u63d0\u53d6 Y_v/N_v/Y_vv/N_vv/Y_r/N_r/Y_rr/N_rr \u5e76\u5bf9\u6807 SIMMAN 2021\u3002

**\u51e0\u4f55\u5fc5\u987b\u5148\u8fc7\u95e8**\uff1a\u4ed3\u5185 simplified \u578b\u503c\u8868\u79ef\u5206 Cb=0.6355\uff08\u76ee\u6807 0.708\uff0c\u504f\u4f4e 10.2%\uff09\u21d2 \u4e0d\u53ef\u8fdb CFD\u3002\u672c\u8f6e\u91cd\u5efa\u53c2\u6570\u5316\u8239\u578b\uff0c
\u4ec5\u7528\u51e0\u4f55\u5f62\u72b6\u53c2\u6570\u4e09\u91cd\u4e8c\u5206\u6807\u5b9a\uff08\u4e0d\u62df\u5408\u4efb\u4f55\u6c34\u52a8\u529b\uff09\u547d\u4e2d\u516c\u5f00\u79ef\u5206\u91cf\uff1a

| \u91cf | \u8ba1\u7b97 | \u76ee\u6807 | \u504f\u5dee |
|---|---|---|---|
| Cb | {f(hull['Cb_computed'])} | 0.708 | {100*(hull['Cb_computed']-0.708)/0.708:+.2f} % |
| Cwp | {f(hull['Cwp_computed'])} | 0.9005 | {100*(hull['Cwp_computed']-0.9005)/0.9005:+.2f} % |
| LCB | {f(hull['LCB_computed'],5)} | \u22120.01688 | {1000*(hull['LCB_computed']+0.01688):+.3f} \u2030 L |
| \u2207 | {f(hull['volume'])} m\u00b3 | 1.5865 | {100*(hull['volume']-1.5865)/1.5865:+.2f} % |

## \u2461 \u600e\u4e48\u505a\u7684
1. \u53c2\u6570\u5316 KVLCC2\uff08\u5e73\u884c\u4e2d\u4f53 + \u629b\u7269\u578b\u8276\u8276 + \u8d85\u692d\u5706\u5256\u9762\u65cf\uff09\uff0c\u79ef\u5206\u81ea\u68c0 Cb/LCB/Cwp\uff0c\u5e76\u505a\u79bb\u6563\u6536\u655b\uff08120\u00d760\u00d740\u2192240\u00d7120\u00d780\uff1aCb 0.7139\u21920.7121\uff0c\u89e3\u6790 0.7094\uff09\u3002
2. \u5256\u9762\u9644\u52a0\u8d28\u91cf\u7528**\u7cbe\u786e 2D \u52bf\u6d41** m=\u03c1\u03c0T\u00b2/\u5355\u4f4d\u957f\u5ea6\uff08Lamb 1932 art.71\uff09+ \u7ec6\u957f\u4f53\u6761\u5e26\u79ef\u5206\uff1b\u72ec\u7acb\u6821\u9a8c\u7528 3D \u957f\u692d\u7403\u89e3\u6790\u89e3\uff08Lamb art.114\uff09\uff1a\u7403\u6781\u9650 0.5\u03c1\u2207\u3001\u7ec6\u957f\u6781\u9650 1.0\u03c1\u2207\uff08\u6d4b\u8bd5\u65ad\u8a00\u901a\u8fc7\uff09\u3002
3. \u7c98\u6027\u963b\u5c3c\u4e3a**\u6a2a\u6d41\u963b\u529b** 0.5\u03c1CdT(x)v|v|\uff08Cd=1.0\uff09\uff0c\u65e0\u6d41\u573a\u6c42\u89e3 \u21d2 \u8bc1\u636e\u964d\u7ea7 EMPIRICAL\u3002
4. PMM \u8fd0\u52a8\u5b66\uff1a\u7eaf\u6a2a\u8361 v=A\u03c9cos \u03c9t\uff08\u6cbf x \u5747\u5300\uff09\uff1b\u7eaf\u9996\u6447 r=A\u03c9cos \u03c9t\u3001v(x)=(x\u2212L/2)r\u3002
5. \u5bfc\u6570\u63d0\u53d6\uff1a\u672b 3 \u5468\u671f\u4e00\u9636\u8c10\u6ce2\u6700\u5c0f\u4e8c\u4e58\uff0cY_v=Y_cos/(A\u03c9)\u3001Y_v\u0307=\u2212Y_sin/(A\u03c9\u00b2)\uff08\u9996\u6447\u540c\u7406\u5f97 Y_r,N_r,Y_\u1e59,N_\u1e59\uff09\uff1b\u63d0\u53d6\u5668\u4ee5\u5408\u6210\u4fe1\u53f7\u6ce8\u5165\u771f\u503c\u9a8c\u8bc1\uff0c\u6062\u590d\u8bef\u5dee <1e\u22126\uff0c3\u03c9 \u566a\u58f0\u4e0d\u4e32\u5165\u3002
6. \u626b\u63cf\uff1aA/L=0.005/0.01/0.02\uff1b\u7ad9\u4f4d 21/41/81\uff1b\u53e6\u52a0\u53c2\u8003\u6570\u636e\u4e00\u81f4\u6027\u5ba1\u8ba1\u3002\u5de5\u5177\u94fe Python 3.11 + numpy + pytest\uff0823 \u9879\u5168\u901a\u8fc7\uff09\uff0c\u56fe\u7531 matplotlib \u7a0b\u5e8f\u5316\u751f\u6210\uff08\u6570\u5b57\u7a0b\u5e8f\u6ce8\u5165\uff0c\u96f6\u624b\u6284\uff09\u3002

## \u2462 \u8ba1\u7b97\u7ed3\u679c\uff08A/L=0.01\uff0c81 \u7ad9\u4f4d\uff0cITTC \u65e0\u91cf\u7eb2\u5316\uff09

| \u5bfc\u6570 | \u672c\u5de5\u4f5c | \u4ed3\u5185\u53c2\u8003\u8868 | \u504f\u5dee | \u53c2\u8003\u6f02\u79fb\u659c\u7387(\u4fee\u590d) | \u504f\u5dee |
|---|---|---|---|---|---|
{rows}

### \u5173\u952e\u5ba1\u8ba1\u53d1\u73b0\uff08\u786e\u5b9a\u6027\u7ed3\u8bba\uff09
- \u4ed3\u5185 `simman2021_reference` **\u5bfc\u6570\u8868\u4e0e\u81ea\u8eab\u6f02\u79fb\u8868\u77db\u76fe**\uff1a\u6f02\u79fb\u8868\u4e00\u9636\u659c\u7387 \u2202Y'/\u2202v'={f(aud['drift_slope_Yv'])}\u3001\u2202N'/\u2202v'={f(aud['drift_slope_Nv'])}\uff0c\u5bfc\u6570\u8868\u7ed9 {f(aud['table_Yv'])} / {f(aud['table_Nv'])}\uff0c\u6bd4\u503c **{f(aud['ratio_Y'],2)}\u00d7 / {f(aud['ratio_N'],2)}\u00d7**\u3002
- \u7531 Y_v\u0307'=\u22120.0125 \u53cd\u63a8 m22={f(aud['m22_reference_implied'],1)} kg \u21d2 **m22/\u03c1\u2207={f(aud['m22_reference_over_rho_nabla'],3)}**\uff1b\u52bf\u6d41\u7406\u8bba\u5bf9\u6cb9\u8f6e\u4e3a 50\u2013100 %\uff08\u7ec6\u957f\u4f53\u4e0a\u9650 {f(aud['m22_slender_over_rho_nabla'],3)}\uff09\u3002\u53c2\u8003\u503c\u4f4e\u4e00\u4e2a\u91cf\u7ea7\u4ee5\u4e0a \u21d2 **\u4e0d\u53ef\u4f5c\u57fa\u51c6**\u3002
- \u7ed3\u8bba\uff1a\u4ed3\u5185 `pmm_virtual_test`\uff08\u529b\u6a21\u578b\u7cfb\u6570\u76f4\u63a5\u53d6\u81ea\u53c2\u8003\u8868\uff09\u6240\u62a5\u201c\u543b\u5408\u201d\u5c5e**\u5faa\u73af\u8bba\u8bc1\uff08MOCK\uff09**\u3002

## \u2463 \u7ed3\u679c\u56fe
fig1 \u51e0\u4f55\uff08\u6c34\u7ebf+\u5256\u9762\uff09\uff1bfig2 \u7eaf\u6a2a\u8361 Y(t)/\u7eaf\u9996\u6447 N(t)\uff1bfig3 \u5bfc\u6570\u5bf9\u6bd4\uff08symlog \u663e\u793a\u91cf\u7ea7\u77db\u76fe\uff09\uff1bfig4 \u632f\u5e45\u654f\u611f\u6027\u4e0e\u9644\u52a0\u8d28\u91cf\u53ef\u4fe1\u533a\u95f4\u5ba1\u8ba1\u3002

## \u2464 \u6ce8\u610f\u70b9\uff08\u5751\u4e0e\u5c40\u9650\uff09
- **\u672c\u8f6e\u5c1d\u8bd5\u5e76\u56de\u9000\u7684\u4e24\u4e2a CFD \u5f15\u64ce\uff08\u7559\u8bc1\uff09**\uff1a(a) 3D \u53cc\u4f53\u52bf\u6d41\u9636\u68af\u58c1\u9762 Neumann \u9644\u52a0\u8d28\u91cf \u2014\u2014 \u7403\u89e3\u6790\u68c0\u9a8c\u4ec5 0.4\u00d7\u89e3\u6790\u4e14\u8870\u51cf\u6307\u6570\u4e0d\u5bf9\uff08Dirichlet \u9489\u6b7b\u9000\u5316\u5230\u5e73\u51e1\u89e3 \u03c6\u2261\u03c8\uff09\uff1b(b) 2D \u4f2a\u8c31 Brinkman \u7f5a NS \u53d7\u8feb\u632f\u8361 \u2014\u2014 Stokes \u5e73\u677f\u89e3\u6790\uff08\u03c4=\u03c1U\u221a(\u03bd\u03c9)\uff0c\u76f8\u4f4d +45\u00b0\uff09\u7ed9\u51fa 36\u00d7 \u504f\u5927\u3001\u76f8\u4f4d \u221291\u00b0\uff0c\u5168\u57df\u52a8\u91cf\u6cd5\u88ab\u95ed\u76d2\u6392\u6d41\u6c61\u67d3 4\u20137\u00d7\u3002\u4e24\u8005**\u672a\u53d1\u5e03**\uff0c\u6839\u56e0\u89c1 `docs/pmm-cfd/README.md`\u3002
- \u6761\u5e26\u6a21\u578b\u56fa\u6709\u7f3a\u9677\uff1a\u5bf9\u79f0\u5256\u9762 \u21d2 \u7eaf\u6a2a\u8361 N_v\u22610\u3001\u7eaf\u9996\u6447 Y_r\u22610\uff1b\u65e0 3D \u6709\u9650\u957f\u4fee\u6b63\uff08\u7ec6\u957f\u4f53\u6bd4 3D \u5b9e\u9645\u9ad8\u7ea6 2\u00d7\uff09\uff1b\u7c98\u6027\u4ec5\u5355\u4e00 Cd\u3002
- \u9002\u7528\u8303\u56f4\uff1a\u51e0\u4f55\u95e8 / \u63d0\u53d6\u94fe / \u5ba1\u8ba1\u53ef\u7528\uff1b**\u4e0d\u53ef**\u7528\u4e8e\u5de5\u7a0b\u5bfc\u6570\u4ea4\u4ed8\u3002

## \u2465 \u8bef\u5dee\u5224\u5b9a\uff08\u663e\u5f0f\u56de\u7b54\u201c\u662f\u5426 <3%\u201d\uff09
- **\u662f\uff08<3%\uff09**\uff1aCb +0.20 %\u3001Cwp +0.08 %\u3001LCB 0.027 %L\u3001\u692d\u7403\u9644\u52a0\u8d28\u91cf\u7403\u6781\u9650 <0.2 %\u3001\u5bfc\u6570\u63d0\u53d6\u5668 <1e\u22126\u3002
- **\u5426**\uff1aY_v 11.3 %\uff08>3 %\uff0c\u4e14\u5176\u201c\u63a5\u8fd1\u53c2\u8003\u8868\u201d\u662f\u4e8c\u6b21\u963b\u529b\u5728\u8be5\u632f\u5e45\u7684\u5076\u7136\u7b49\u6548\uff0c\u4e0d\u53ef\u4fe1\uff09\uff1bN_v/Y_r/Y_v\u0307/N_v\u0307/Y_\u1e59 \u504f\u5dee \u2212100 %\u2026+18857 %\uff08\u7f3a 3D \u4e0e\u7eb5\u5411\u4e0d\u5bf9\u79f0\u6548\u5e94\uff09\u3002
- **\u65e0\u57fa\u51c6\uff08WITHHELD\uff09**\uff1a\u529b/\u529b\u77e9\u76f8\u4f4d vs \u5b9e\u9a8c \u2014\u2014 \u4ed3\u5185\u65e0 SIMMAN \u76f8\u4f4d\u6570\u636e\uff0c\u65e0\u6cd5\u5224\u5b9a\uff0c\u672a\u7c89\u9970\u3002
- \u673a\u5668\u5224\u636e\u73b0\u72b6\uff1a`geometry_gate=True`\uff1bC1/C2/C3/C4 **\u5168\u90e8 False**\uff08C3 \u4ec5\u6709\u7ad9\u4f4d\u79ef\u5206\u6b63\u4ea4\u6536\u655b {quad.get('Yvdot',0):.2e}\uff0c\u975e CFD \u7f51\u683c\u6536\u655b\uff09\u3002

---
\u4ea7\u7269\uff1a`src/tensorfvm/pmm_cfd.py`\u3001`tests/test_pmm_manoeuvring.py`(23 passed)\u3001`scripts/pmm_report_figures.py`\u3001`docs/pmm-cfd/README.md`\u3001`docs/pmm-cfd/pmm_derivatives_summary.json`\u3001`docs/pmm-cfd/figures/*.png`\uff1bcommit a9ce7fcb\u3002
"""

rep = {"title": "KVLCC2 PMM \u865a\u62df\u8bd5\u9a8c CFD \u57fa\u51c6\uff08IN40-N04\uff09\uff1a\u51e0\u4f55\u95e8 PASS\u3001\u63d0\u53d6\u94fe\u53ef\u81ea\u68c0\u3001CFD \u5f15\u64ce\u672a\u95ed\u5408\uff08honest FAIL\uff09+ SIMMAN \u53c2\u8003\u6570\u636e\u5ba1\u8ba1",
       "summary": ("KVLCC2 \u51e0\u4f55\u91cd\u5efa Cb 0.7095 / Cwp 0.9012 / LCB \u22121.715%L\uff08\u504f\u5dee 0.2% / 0.08% / 0.027%L\uff0c\u51e0\u4f55\u95e8 PASS\uff09\uff1b"
                   "\u5256\u9762\u9644\u52a0\u8d28\u91cf\u7528\u7cbe\u786e 2D \u52bf\u6d41\u5e76\u7ecf 3D \u692d\u7403\u89e3\u6790\u6821\u9a8c\uff1b\u5bfc\u6570\u63d0\u53d6\u94fe\u4ee5\u5408\u6210\u4fe1\u53f7\u9a8c\u8bc1\u5230 1e-6\u3002"
                   "\u5ba1\u8ba1\u53d1\u73b0\u4ed3\u5185 SIMMAN \u5bfc\u6570\u8868\u4e0e\u81ea\u8eab\u6f02\u79fb\u8868\u77db\u76fe 3.5x/6.3x\u3001\u4e14 m22/rho*nabla=6.6%\uff08\u7406\u8bba 50-100%\uff09\uff0c\u4ed3\u5185\u6821\u51c6\u5f0f PMM \u5c5e\u5faa\u73af\u8bba\u8bc1\u3002"
                   "\u4e24\u4e2a CFD \u5f15\u64ce\uff083D \u9636\u68af\u52bf\u6d41\u9644\u52a0\u8d28\u91cf\u30012D \u7f5a\u6d78\u5165 NS\uff09\u672a\u901a\u8fc7\u9a8c\u8bc1\u5e76\u5df2\u56de\u9000\u7559\u8bc1\uff0c\u6545 C1-C4 \u5224\u636e\u5168\u90e8\u5982\u5b9e FAIL\uff0c\u76f8\u4f4d\u5224\u636e WITHHELD\u3002"),
       "scenario_key": "IN40-N04",
       "mission_id": 5539,
       "tags": ["pmm", "manoeuvring", "kvlcc2", "simm2021", "cfd-audit", "honest-fail"],
       "markdown": MD,
       "figures": []}

caps = {
    "fig1_geometry.png": "KVLCC2 \u91cd\u5efa\u51e0\u4f55\uff1a\u8bbe\u8ba1\u6c34\u7ebf\u4e0e\u5178\u578b\u6a2a\u5256\u9762\uff08Cwp=0.901\uff0cCb=0.7095\uff09",
    "fig2_pmm_forces.png": "PMM \u5f3a\u8feb\u8fd0\u52a8\u529b/\u529b\u77e9\u65f6\u95f4\u5386\u7a0b\uff08\u7eaf\u6a2a\u8361 Y(t)\uff0c\u7eaf\u9996\u6447 N(t)\uff09",
    "fig3_derivative_comparison.png": "\u5bfc\u6570\u5bf9\u6bd4\uff1a\u672c\u5de5\u4f5c vs \u4ed3\u5185 SIMMAN \u53c2\u8003\u8868\uff08symlog\uff09",
    "fig4_amplitude_plausibility.png": "\u5de6\uff1a\u5bfc\u6570\u968f\u632f\u5e45\u7684\u654f\u611f\u6027\uff1b\u53f3\uff1a\u9644\u52a0\u8d28\u91cf\u53ef\u4fe1\u533a\u95f4\u5ba1\u8ba1 m22/rho*nabla",
}
figs = []
for name, cap in caps.items():
    im = Image.open(pathlib.Path("docs/pmm-cfd/figures") / name).convert("RGB")
    w = 600
    im = im.resize((w, int(im.height * w / im.width)), Image.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, format="PNG", optimize=True)
    figs.append({"name": name, "caption": cap,
                 "png_base64": base64.b64encode(buf.getvalue()).decode()})
    print(name, len(buf.getvalue()) // 1024, "KB")
rep["figures"] = figs
payload = json.dumps(rep).encode()
print("payload KB", len(payload) // 1024)
REPORT.write_bytes(payload)

token = json.loads(TOKEN_FILE.read_text())["access_token"]
base = os.environ["TENSOR_API_BASE"].rstrip("/")
req = urllib.request.Request(
    f"{base}/tensor/reports", data=payload, method="POST",
    headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"})
with urllib.request.urlopen(req, timeout=90) as resp:
    body = resp.read().decode()
    print("HTTP", resp.status)
    print(body[:400])