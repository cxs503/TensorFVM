"""Publish detailed tables and matched shear plots from immutable saved evidence."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from tensorfvm.verification.core import metric_value, artifact_manifest, sha, write_json
from tensorfvm.verification.report import figure, plt
from matplotlib.backends.backend_pdf import PdfPages


def main():
    p=argparse.ArgumentParser();p.add_argument('--directory',type=Path,required=True);a=p.parse_args();out=a.directory
    manifest=json.loads((out/'manifest.json').read_text());root=Path(__file__).resolve().parents[1]
    for name,digest in manifest['artifacts_sha256'].items():
        if sha(out/name)!=digest:raise ValueError('stored artifact changed: '+name)
    for name,digest in manifest['source_sha256'].items():
        if sha(root/name)!=digest:raise ValueError('stored source changed: '+name)
    s=json.loads((out/'summary.json').read_text());rows=s['runs'];table=[]
    for r in rows:
        pm='pressure_l2' if r['case']=='abc' else 'pressure_dynamic_scaled_linf'
        table.append([r['case'],r['resolution'],r['role'],f"{100*metric_value(r,'velocity_l2'):.6f}",f"{100*metric_value(r,pm):.6g}",'-' if r['case']=='abc' else f"{100*metric_value(r,'shear_velocity_l2'):.6f}",str(r['passed'])])
    columns=['Case','Grid','Role','Velocity L2 (%)','Pressure metric (%)','Shear L2 (%)','Pass']
    with (out/'results-table.csv').open('w',newline='') as stream:w=csv.writer(stream);w.writerow(columns);w.writerows(table)
    fvm=next(r for r in rows if r['case']=='advected-shear' and r['resolution'].startswith('32x'))
    lbm=next((r for r in rows if r['case']=='advected-shear-lbm' and r['resolution'].startswith('32x')),None)
    if lbm:
        fig,ax=plt.subplots(figsize=(6,4));names=['All velocity L2','Transported shear L2'];pos=np.arange(2)
        ax.bar(pos-.18,[100*metric_value(fvm,k) for k in ['velocity_l2','shear_velocity_l2']],width=.36,label='FVM n=32')
        ax.bar(pos+.18,[100*metric_value(lbm,k) for k in ['velocity_l2','shear_velocity_l2']],width=.36,label='TensorLBM BGK n=32')
        ax.set_xticks(pos,names);ax.set(ylabel='Relative error (%)',title='Matched advected shear: t=0.5 s, nu=0.1');ax.legend();ax.grid(axis='y',alpha=.3);figure(fig,out/'figures','shear-lbm-comparison')
        f=np.load(out/fvm['directory']/'fields.npz');l=np.load(out/lbm['directory']/'fields.npz');fig,ax=plt.subplots(figsize=(6,4))
        ax.plot(f['profile_y_m'],f['profile_u_m_s'],'o',ms=3,label='FVM');ax.plot(l['profile_y_m'],l['profile_u_m_s'],'x',ms=4,label='BGK');ax.plot(f['profile_y_m'],f['reference_u_profile_m_s'],'-',label='Analytic')
        ax.set(xlabel='y (m)',ylabel='u (m/s)',title='Matched transport phase and attenuation');ax.legend();ax.grid(alpha=.3);figure(fig,out/'figures','shear-lbm-profile')
    with PdfPages(out/'results-tables.pdf') as pdf:
        fig,ax=plt.subplots(figsize=(11.69,8.27));ax.axis('off');ax.set_title('Additional analytic benchmarks: complete quantitative results',pad=25)
        t=ax.table(cellText=table,colLabels=columns,loc='upper center',colWidths=[.19,.12,.14,.15,.15,.14,.09]);t.auto_set_font_size(False);t.set_fontsize(8);t.scale(1,1.8)
        ax.text(0,.43,'ABC pressure: relative L2 at final midpoint. Shear pressure: max absolute pressure / initial dynamic pressure.\nAll declared physical errors <3% on formal grids; n16 ABC is an intentional pressure failure control.\nShear component error is checked separately to avoid dilution by constant mean flow.\nABC is a genuine 3D periodic flow; shear is 2D in four z layers. No solid/free-surface qualification.\nLBM uses unchanged D2Q9 BGK production functions and upstream float32 weights.\nDirect comparison is n32 shear only: same continuum parameters, actual face/node reference sampling.\nRun duration includes diagnostics and raw sampling; this is not a repeated performance benchmark.',va='top',fontsize=9,linespacing=1.6)
        pdf.savefig(fig,bbox_inches='tight');plt.close(fig)
        if lbm:
            fig,axes=plt.subplots(1,2,figsize=(11.69,5))
            for ax,name in zip(axes,['shear-lbm-comparison','shear-lbm-profile']):ax.imshow(plt.imread(out/'figures'/(name+'.png')));ax.axis('off')
            fig.tight_layout();pdf.savefig(fig);plt.close(fig)
    write_json(out/'tables-publication.json',dict(producer='scripts/summarize_extended_benchmarks.py',source_sha256=sha(Path(__file__)),summary_sha256=sha(out/'summary.json'),scope='additional tables/plots only; numerical fields and primary report remain unchanged'))
    manifest['artifacts_sha256']=artifact_manifest(out);write_json(out/'manifest.json',manifest)


if __name__=='__main__':main()
