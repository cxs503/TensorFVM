"""Compare unmodified wall constants with NASA TMR public Coles correlation data."""
import argparse
import json
from pathlib import Path
import urllib.request
import numpy as np
from tensorfvm.wall_functions import u_plus_at_y_plus
from tensorfvm.verification.core import Metric,evaluate,relative_error,write_json,sha,artifact_manifest
from tensorfvm.verification.report import plt,figure

URL='https://tmbwg.github.io/turbmodels/FlatPlate_validation/u+y+.dat'


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--reference-file',type=Path);a=p.parse_args()
    if a.output.exists() and any(a.output.iterdir()):p.error('empty output required')
    a.output.mkdir(parents=True,exist_ok=True)
    if a.reference_file:data=a.reference_file.read_bytes()
    else:
        with urllib.request.urlopen(URL,timeout=30) as response:data=response.read()
    (a.output/'coles-reference.dat').write_bytes(data);rows=[]
    for line in data.decode().splitlines():
        try:yp,up=map(float,line.split());rows.append((yp,up))
        except ValueError:pass
    ref=np.asarray(rows);pred=u_plus_at_y_plus(ref[:,0]);mask=(ref[:,0]>=50)&(ref[:,0]<=200)
    if mask.sum()<10:raise ValueError('reference log-window samples missing')
    metrics,passed=evaluate([Metric('log_window_uplus_l2',relative_error(pred[mask],ref[mask,1]),.03,'specified 50<=y+<=200; NASA Coles/van-Driest correlation, Re_theta=10000'),Metric('log_window_uplus_linf',relative_error(pred[mask],ref[mask,1],'linf'),.03,'reference-peak normalized maximum error in the same log window')])
    write_json(a.output/'result.json',dict(metrics=metrics,passed=passed,matching_scope='equilibrium wall-law versus independent published correlation only; not a RANS/SA flow or DNS qualification',reference_url=URL,reference_sha256=sha(a.output/'coles-reference.dat'),parameters=dict(kappa=.41,E=9.8),comparison_window_yplus=[50,200],window_samples=int(mask.sum()),producer_sha256=sha(Path(__file__))))
    np.savetxt(a.output/'comparison.csv',np.column_stack((ref[:,0],ref[:,1],pred,mask.astype(int))),delimiter=',',header='yplus,coles_reference_uplus,spalding_uplus,in_declared_log_window',comments='')
    fig,ax=plt.subplots(figsize=(7,5));ax.semilogx(ref[:,0],ref[:,1],label='NASA TMR Coles/van-Driest correlation');ax.semilogx(ref[:,0],pred,'--',label='Spalding: kappa=0.41 E=9.8');ax.axvspan(50,200,color='grey',alpha=.2,label='Declared log-window gate');ax.set(xlabel='y+',ylabel='u+',title='Independent correlation comparison; not a CFD flow run');ax.legend();ax.grid(alpha=.3);figure(fig,a.output,'wall-law-reference')
    report='# 壁面函数与公开关联曲线对照\n\n参考：[NASA TMR 的 Coles 速度律、含近壁 van-Driest 阻尼](https://tmbwg.github.io/turbmodels/flatplate_val.html)，Re_theta=10000。原始数据随包保存，来自官方新地址。\n\n固定Spalding参数κ=0.41、E=9.8，未拟合参考。声明对数区50≤y+≤200；全曲线同时绘制，区外不宣称3%适用。两种近壁/外层模型不同，不能把该窗口通过当成分离/外层湍流或完整RANS准确性。\n\n'
    for q in metrics:report+=f"- {q['name']}: {100*q['value']:.6f}%，严格<3%：{q['passed']}\n"
    report+='\n![reference](wall-law-reference.png)\n\n本模块反解与CPU/CUDA实际硬件验证另见大型网格与GPU报告。尚未接入原SA输运边界。\n';(a.output/'report.md').write_text(report);write_json(a.output/'manifest.json',dict(artifacts_sha256=artifact_manifest(a.output),source_sha256={'scripts/compare_wall_law_coles.py':sha(Path(__file__))}));print('Independent correlation window passed:',passed)


if __name__=='__main__':main()
