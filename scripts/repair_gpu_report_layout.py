"""Keep GPU report table and its scope notes on separate parts of the page."""
import json
from pathlib import Path
from matplotlib.axes import Axes
from tensorfvm.verification.gpu import publish
from tensorfvm.verification.core import artifact_manifest, write_json, sha
root=Path(__file__).resolve().parents[1];p=root/'docs/verification-gpu'
original_text=Axes.text
original_mkdir=Path.mkdir
original_table=Axes.table
def table(self,*args,**kwargs):
    if kwargs.get("colLabels",[])==["Case","CPU s","CUDA s","CPU/CUDA","Equivalent","Physical pass"]:
        kwargs["colWidths"]=[.30,.14,.14,.14,.14,.14]
    return original_table(self,*args,**kwargs)
def text(self,x,y,s,*args,**kwargs):
    if x==0 and y==.45 and s.startswith('Float64;'):y=.2
    return original_text(self,x,y,s,*args,**kwargs)
def mkdir(self,*args,**kwargs):
    if self==p/'figures':kwargs['exist_ok']=True
    return original_mkdir(self,*args,**kwargs)
Axes.text=text;Axes.table=table;Path.mkdir=mkdir
publish(p,json.loads((p/'summary.json').read_text()))
report=p/'report.md'
report.write_text(report.read_text()+'\n## 验收结论\n\n九项通过 CPU/CUDA 场一致性；SA 五次迭代未通过：压力相对 L2 差 0.575%，速度相对 L2 差 1.3344e−6，统一门限 1e−6。整体 correctness_passed=False。SA 时间比 6.917 仅是测量值，不能作为通过正确性认证的加速结果。原场及失败标志完整保留，后续定位压力线性求解停止条件、真实残差与硬件归约敏感性。\n')
m=json.loads((p/'manifest.json').read_text());m['publication_layout_script_sha256']=sha(Path(__file__));m['artifacts_sha256']=artifact_manifest(p);write_json(p/'manifest.json',m)
