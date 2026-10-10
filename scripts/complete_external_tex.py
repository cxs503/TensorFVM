"""Ensure the editable manuscript includes global pressure, velocity and Cp plots."""
import json
from pathlib import Path
from tensorfvm.verification.core import sha,write_json,artifact_manifest
p=Path('docs/verification-external-cylinder');summary=json.loads((p/'summary.json').read_text());path=p/'report.tex';text=path.read_text();extra=''
for row in summary['runs']:
 for name in ('pressure','velocity','surface-pressure','residuals'):
  figure=row['directory']+'/figures/'+name+'.pdf'
  if figure not in text:extra+='\\begin{figure}[p]\\centering\\includegraphics[width=.9\\linewidth]{'+figure+'}\\caption{'+row['resolution']+' '+name.replace('-',' ')+'}\\end{figure}\n'
text=text.replace('\\section{Discussion and limits}',extra+'\\section{Discussion and limits}');path.write_text(text)
m=json.loads((p/'manifest.json').read_text());m['source_sha256']['scripts/complete_external_tex.py']=sha(Path(__file__));m['artifacts_sha256']=artifact_manifest(p);write_json(p/'manifest.json',m)
