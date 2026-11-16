import base64, json, os, requests

DOC = "docs/XVAL-FVM-M18737-ROUND2/R15-CLOSURE.md"
PNG = "docs/XVAL-FVM-M18737-ROUND2/figures/fig_r15_ref_ladder.png"
RUNG = "docs/XVAL-FVM-M18737-ROUND2/r15_certified_rung.json"

v = json.load(open(RUNG))
e1 = v["E1r6400_cfl_reference"]
e4 = v["E4_honest_ceiling"]
bars = e4["bars"]
fin = max(int(k) for k in v["ladder"])
l1 = v["ladder"][str(fin)]["l1_percent"]

md = open(DOC).read()
md += (
    "\n\n---\n\n**产物与证据指针**：`docs/XVAL-FVM-M18737-ROUND2/` 下 "
    "`PREDECLARATION-r15.md`（判据冻结）、`R15-CLOSURE.md`（本报告全文）、"
    "`r15_ref_ladder_cfl.py`（驱动，含本单 shadowing 缺陷修复）、"
    "`r15_certified_rung.json`（程序化注入）、`figures/fig_r15_ref_ladder.png`；"
    "回归门 `tests/test_wc_r15_ref_ladder_closure.py`（5 测全绿）；"
    "全量 47 测全绿（42 既有 + 5 新增）。\n"
)

summary = (
    "Woodward-Colella 双爆轰波（一维闭管三压力平台，无解析解）交叉验证欠账闭环："
    "R14 遗留唯一开口——参考阶梯 6400 档仍为冻结 CFL 0.4 仪器，E4 licensed=False、"
    "eps_ref 只能记探针——由本单补 WENO5@6400 CFL 0.8 档（cj4398）并重跑冻结 R13 管线。"
    f"E1r6400 同格 CFL 0.4 vs 0.8 互相 L1={e1['l1_percent']:.4f}%<0.2% 阈值（准入），"
    "6400 档取档表覆盖为 CFL 0.8 仪器，E4 licensed 升 True，"
    f"eps_ref={bars['B0']['eps_percent']:.4f}% 成为已认证误差棒；"
    "被定级族阶梯/E1/E1r/E2/E3 与 R14 逐位相同。"
    "误差判定：总账<15% 最保守上界 B3=1.384% PASS；<3% PASS；"
    "<1% 点估计 L1=0.321% PASS；<1% 诚实上界 B0/B1/B2/B4 全 PASS；"
    "B3 非渐近 1.384% honest FAIL（不覆盖）。"
    "驱动脚本 shadowing 缺陷（局部 e1r6400 遮蔽模块级门函数致 UnboundLocalError）已修复，"
    "新增 5 测回归门，全量 47 测全绿。"
)
# hard trim to <500
summary = summary[:498]

rep = {
    "title": "XVAL-FVM-M18737-ROUND2 R15 闭合：参考阶梯 6400 档补 CFL 0.8 变体，"
             "eps_ref 升为已认证误差棒（E4 licensed）",
    "summary": summary,
    "scenario_key": "xval-fvm-m18737-round2-r15-ref-ladder",
    "mission_id": 23274,
    "tags": ["TensorFVM", "verification", "Woodward-Colella", "reference-ladder",
             "CFL", "certified", "eps_ref", "honest-FAIL", "defect-fix"],
    "markdown": md,
    "figures": [{
        "name": "fig_r15_ref_ladder.png",
        "caption": ("(a) 参考阶梯 6400/12800/25600 同仪器（CFL 0.8）相邻档 L1 对差商与 "
                    "eps_ref（12800→25600）；(b) 五口径诚实上界（licensed=True 标注）；"
                    "(c) 6400 档 CFL 0.4 vs 0.8 剖面（E1r6400 准入控制，L1=0.023%）。"),
        "png_base64": base64.b64encode(open(PNG, "rb").read()).decode(),
    }],
}

tok = open("/tmp/tok").read().strip()
r = requests.post(os.environ["TENSOR_API_BASE"] + "/tensor/reports",
                  headers={"Authorization": f"Bearer {tok}"}, json=rep, timeout=180)
print(r.status_code, r.text[:400])
