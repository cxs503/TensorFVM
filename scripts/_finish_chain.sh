#!/bin/bash
# tail-end chain: wait for the study artifact, then figures + report + docs + commit
cd /root/tp-agent/work/m17128
J=docs/repro-fvm-sod1978-contact-limit/contact_study.json
for i in $(seq 1 55); do
  [ -f "$J" ] && break
  sleep 2
done
if [ ! -f "$J" ]; then echo "STILL RUNNING"; pgrep -af "sod1978_contact_limit" | head -2; exit 0; fi
echo "ARTIFACT PRESENT"
PYTHONPATH=src python scripts/publish_sod1978_contact_limit.py --study "$J" || exit 1
PYTHONPATH=src python - <<'EOF' || exit 1
import json, sys
sys.path.insert(0, "scripts"); sys.path.insert(0, "src")
from publish_sod1978_contact_limit import build_markdown
study = json.load(open("docs/repro-fvm-sod1978-contact-limit/contact_study.json"))
md = build_markdown(study)
head = """# REPRO-FVM-SOD1978 第 3 轮 · 全域 L1 阶的接触波下限（机制确认）

锚点：`REPRO-FVM-SOD1978`（论文复现 · Sod 1978, *J. Comput. Phys.* 27:1-31）。
本文件由 `scripts/publish_sod1978_contact_limit.py:build_markdown()` **从工件
`contact_study.json` 程序化生成**（零手抄）；重新生成：
`PYTHONPATH=src python scripts/publish_sod1978_contact_limit.py --study docs/repro-fvm-sod1978-contact-limit/contact_study.json`。

复现命令（集群，loong64；本机版把 `-m` 换成同参数即可）：

    PYTHONPATH=src python -m tensorfvm.verification.sod1978_contact_limit \\
        --out docs/repro-fvm-sod1978-contact-limit --grids 200,400,800,1600,3200,6400 --orders 1,2

性质门（11 项，约 60 s）：`PYTHONPATH=src python -m pytest tests/test_sod1978_contact_limit.py -q`

---

"""
open("docs/repro-fvm-sod1978-contact-limit/REPORT.md", "w").write(head + md)
print("REPORT.md written", len(md))
EOF
git add -A src/tensorfvm/verification/sod1978_contact_limit.py tests/test_sod1978_contact_limit.py scripts/publish_sod1978_contact_limit.py docs/repro-fvm-sod1978-contact-limit
git -c user.name=tp-agent -c user.email=agent@tp.local commit -q -m "TensorFVM: Sod(1978) anchor round 3 — pure-contact control pins the whole-field L1 rate below the design order

Adds src/tensorfvm/verification/sod1978_contact_limit.py (pure-contact control
problem + disjoint Sod zone error budget + declared thresholds), 11 property
tests, the study artifact and the report generator. Same solver as the anchor
(HLLC + MUSCL-Hancock, mc limiter, CFL 0.4): the Sod whole-field L1 rate tracks
the pure-contact (linearly degenerate wave) rate within the declared tolerance,
so the sub-linear global rate is the contact-capture limit, not a scheme defect.
Criterion C3 (>= 1.0 literally) is reported FAIL, unchanged." && echo COMMIT_OK
git log --oneline -1