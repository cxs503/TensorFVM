"""Fetch an open-access PDF (arXiv) and record its sha256."""
import hashlib
import sys
import urllib.request
from pathlib import Path

UA = {"User-Agent": "TensorFVM-verification/1.0 (research)"}
for aid in sys.argv[1:]:
    url = f"https://arxiv.org/pdf/{aid}"
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=180) as fh:
        raw = fh.read()
    out = Path("pdf") / f"{aid.replace('/', '_')}.pdf"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(raw)
    print(aid, len(raw), hashlib.sha256(raw).hexdigest(), out)