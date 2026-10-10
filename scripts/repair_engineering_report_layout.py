"""Republish the frozen numerical report with table notes below the table."""
import json
from pathlib import Path
from matplotlib.axes import Axes
from tensorfvm.verification.engineering import publish
from tensorfvm.verification.core import artifact_manifest, write_json, sha
root=Path(__file__).resolve().parents[1];p=root/'docs/verification-engineering'
original=Axes.text
def text(self,x,y,s,*args,**kwargs):
    if x==0 and y==.65 and s.startswith('All physical gates'):y=.45
    return original(self,x,y,s,*args,**kwargs)
Axes.text=text
original_mkdir=Path.mkdir
def mkdir(self,*args,**kwargs):
    if self==p/"figures":kwargs["exist_ok"]=True
    return original_mkdir(self,*args,**kwargs)
Path.mkdir=mkdir
publish(p,json.loads((p/'summary.json').read_text()))
m=json.loads((p/'manifest.json').read_text());m['publication_layout_script_sha256']=sha(Path(__file__));m['artifacts_sha256']=artifact_manifest(p);write_json(p/'manifest.json',m)
