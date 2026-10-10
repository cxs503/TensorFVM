"""Ordered per-element surface Cp and optional independently supplied reference.
Reference CSV: element,surface,x_over_c,cp. Metadata JSON must accompany it.
Owner-cell pressure is an explicit wall-pressure approximation, not extrapolation.
"""
import argparse,csv,json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from tensorfvm.multi_element import ThreeElementMesh
from tensorfvm.benchmark_30p30n import PROFILE_DIRECTORY,_read_profile
from tensorfvm.verification.core import sha,write_json,artifact_manifest


def contiguous(mask):
    indices=np.flatnonzero(mask)
    return np.split(indices,np.flatnonzero(np.diff(indices)>1)+1) if len(indices) else []


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('output',type=Path);parser.add_argument('--reference',type=Path);parser.add_argument('--reference-metadata',type=Path);args=parser.parse_args();out=args.output
    if bool(args.reference)!=bool(args.reference_metadata):parser.error('Reference CSV and metadata are both required')
    torch.set_num_threads(1);summary=json.loads((out/'benchmark.json').read_text());config=summary['config'];data=np.genfromtxt(out/'fields.csv',delimiter=',',names=True)
    mesh=ThreeElementMesh(SimpleNamespace(mesh_file=out/'mesh.msh',device='cpu'));pressure=data['p'];dynamic=.5*config['density']*config['inlet_velocity']**2;pinf=0.;chord=config['airfoil_chord']
    if len(pressure)!=mesh.volumes.numel() or not np.isfinite(pressure).all():raise ValueError('Invalid pressure field')
    old=np.genfromtxt(out/'surface.csv',delimiter=',',names=True,dtype=None,encoding='utf-8');np.testing.assert_allclose(old['cp'],(old['p']-pinf)/dynamic,rtol=1e-12,atol=1e-12)
    ref=None;reference_record=dict(status='missing',reason='No independently verified matching-condition surface Cp dataset available',accuracy_accepted=False)
    if args.reference:
        meta=json.loads(args.reference_metadata.read_text());expected=dict(configuration='30P30N',reynolds=config['reynolds'],alpha_deg=config['angle_of_attack'],reference_chord=chord)
        for key,value in expected.items():
            if key not in meta or (meta[key]!=value if isinstance(value,str) else not np.isclose(meta[key],value,rtol=1e-8,atol=1e-10)):raise ValueError('Reference condition mismatch: '+key)
        for key in ('source_url','cp_definition','boundary_conditions','geometry_description'):
            if not meta.get(key):raise ValueError('Reference metadata requires '+key)
        if meta['cp_definition']!='(p-p_inf)/(0.5*rho*U_inf^2)':raise ValueError('Reference Cp normalization mismatch')
        ref=np.genfromtxt(args.reference,delimiter=',',names=True,dtype=None,encoding='utf-8');ref=np.atleast_1d(ref)
        if not {'element','surface','x_over_c','cp'}<=set(ref.dtype.names):raise ValueError('Invalid reference columns')
        if not np.isfinite(ref['x_over_c']).all() or not np.isfinite(ref['cp']).all():raise ValueError('Nonfinite reference data')
        if not set(ref['element'])<= {'slat','main','flap'} or not set(ref['surface'])<= {'upper','lower'}:raise ValueError('Unknown reference surface')
        reference_record=dict(status='provided_conditions_declared_match',metadata=meta,input_sha256=sha(args.reference),accuracy_accepted=False)
    rows=[];curves={};metrics=[]
    for name in ('slat','main','flap'):
        face_ids=torch.nonzero(mesh.masks[name]).flatten().numpy();edges=mesh.face_vertices[face_ids].numpy();links={tuple(a):i for i,(a,b) in enumerate(edges)};order=[0]
        while len(order)<len(edges):order.append(links[tuple(edges[order[-1],1])])
        if len(set(order))!=len(edges) or not np.array_equal(edges[order[-1],1],edges[order[0],0]):raise ValueError('Wall loop is not closed')
        face_ids=face_ids[order];centers=mesh.face_centers[face_ids].numpy();owners=mesh.owner[face_ids].numpy();S=mesh.face_area_vectors[face_ids].numpy();length=mesh.face_lengths[face_ids].numpy();cp=(pressure[owners]-pinf)/dynamic
        profile=np.array(_read_profile(PROFILE_DIRECTORY/f'{name}.dat'));leading=profile[np.argmin(profile[:,0])];trailing=profile[np.argmax(profile[:,0])];tangent=trailing-leading;tangent/=np.linalg.norm(tangent);normal=np.array([-tangent[1],tangent[0]])
        # Split the closed wall contour into two continuous LE-to-TE branches.
        # Local normal signs alone would fragment the slat and cove surfaces.
        wall_vertices=mesh.face_vertices[face_ids].numpy()[:,0];le=int(np.argmin(wall_vertices[:,0]));te=int(np.argmax(wall_vertices[:,0]));branch=np.zeros(len(face_ids),bool);j=le
        while j!=te:branch[j]=True;j=(j+1)%len(branch)
        coordinate=(centers-leading)@normal
        first=np.average(coordinate[branch],weights=length[branch]);second=np.average(coordinate[~branch],weights=length[~branch]);upper=branch if first>second else ~branch
        arc=(np.cumsum(length)-.5*length)/chord
        curves[name]=(centers[:,0]/chord,cp,upper)
        for i in range(len(cp)):rows.append([name,'upper' if upper[i] else 'lower',int(face_ids[i]),int(owners[i]),float(centers[i,0]/chord),float(centers[i,1]/chord),float(arc[i]),float(length[i]/chord),float(pressure[owners[i]]),float(cp[i])])
        for side,mask in (('upper',upper),('lower',~upper)):
            item=dict(element=name,surface=side,faces=int(mask.sum()),cp_min=float(cp[mask].min()),cp_max=float(cp[mask].max()),reference_relative_l2_error=None)
            if ref is not None:
                r=ref[(ref['element']==name)&(ref['surface']==side)]
                if len(r)<3:raise ValueError('Reference needs at least three points per surface')
                x=centers[mask,0]/chord;y=cp[mask];sort=np.argsort(x);x=x[sort];y=y[sort]
                if np.any(np.diff(x)<1e-10):raise ValueError('Multi-valued x on '+name+' '+side+'; supply an arclength-based reference adapter instead of averaging wall points')
                if r['x_over_c'].min()<x.min() or r['x_over_c'].max()>x.max():raise ValueError('Reference lies outside supported face-center range; extrapolation refused')
                computed=np.interp(r['x_over_c'],x,y);denominator=np.linalg.norm(r['cp'])
                if denominator<1e-12:raise ValueError('Reference norm too small')
                item['reference_relative_l2_error']=float(np.linalg.norm(computed-r['cp'])/denominator)
            metrics.append(item)
    if args.reference:
        (out/'cp-reference.csv').write_bytes(args.reference.read_bytes());write_json(out/'cp-reference-metadata.json',meta)
    with (out/'cp-surfaces.csv').open('w',newline='') as stream:
        writer=csv.writer(stream);writer.writerow(['element','surface','face','owner','x_over_c','y_over_c','s_over_c','face_length_over_c','pressure_owner','cp']);writer.writerows(rows)
    status='UNCONVERGED DIAGNOSTIC; reference accuracy unverified' if not summary['converged'] else 'Numerically converged; reference accuracy unverified'
    with PdfPages(out/'cp-report.pdf') as pdf:
        fig,axes=plt.subplots(1,3,figsize=(15,5),constrained_layout=True)
        for ax,(name,(x,cp,upper)) in zip(axes,curves.items()):
            for side,mask,color in (('upper',upper,'tab:blue'),('lower',~upper,'tab:orange')):
                for j,run in enumerate(contiguous(mask)):
                    ax.plot(x[run],cp[run],color=color,linewidth=1.2,label=f'Computed {side}' if j==0 else None)
                if ref is not None:
                    r=ref[(ref['element']==name)&(ref['surface']==side)];ax.scatter(r['x_over_c'],r['cp'],s=20,marker='o' if side=='upper' else 's',facecolors='none',edgecolors=color,label=f'Reference {side}')
            ax.invert_yaxis();ax.set_title(name);ax.set_xlabel('x / C (global reference chord)');ax.set_ylabel('Cp');ax.grid(alpha=.3);ax.legend(fontsize=8)
        fig.suptitle(f'30P30N surface pressure coefficients; Re={config["reynolds"]:g}, alpha={config["angle_of_attack"]:g} deg\n{status}',color='darkred' if not summary['converged'] else 'black');fig.savefig(out/'cp-comparison.png',dpi=300);fig.savefig(out/'cp-comparison.svg');pdf.savefig(fig);plt.close(fig)
    record=dict(converged=summary['converged'],physics_accepted=False,iterations=summary['iterations'],cp_definition='(p-p_inf)/(0.5*rho*U_inf^2)',pressure_reference=pinf,pressure_reference_description='Prescribed outlet zero gauge pressure',wall_pressure_method='Adjacent owner-cell pressure; no validated wall extrapolation',surface_definition='Closed wall split into two LE-to-TE branches; upper has larger length-weighted chord-normal coordinate',surface_faces=len(rows),reference=reference_record,metrics=metrics,inputs_sha256={key:sha(out/key) for key in ('fields.csv','surface.csv','mesh.msh','benchmark.json')});write_json(out/'cp-comparison.json',record)
    (out/'cp-report.md').write_text('# 三段翼表面压力系数\n\n![三段翼Cp曲线](cp-comparison.png)\n\nCp=(p_wall-p_inf)/(0.5 rho U_inf²)，p_inf=0为出口给定的表压参考。每个翼段在最前/最后x坐标顶点处将闭合壁面拆成两条连续路径；弦线法向平均坐标较高者定义为上表面，按真实壁面连接顺序连线；横轴为全局参考弦长归一化坐标，纵轴按气动惯例反向。Cp目前采用相邻单元压力，尚未验证壁面压力外推精度。\n\n当前为两步未收敛SIMPLE/SA诊断。没有同工况可核验的实验/独立参考Cp数据，图中计算上下表面曲线不能替代实验对比，实验误差为空，物理精度验收为False。\n\n[逐壁面原始数据](cp-surfaces.csv) · [对比状态](cp-comparison.json) · [PDF](cp-report.pdf)\n\n复现：`PYTHONPATH=src python scripts/compare_three_element_cp.py docs/verification-30p30n-flow-preview`。若有参考文件，添加`--reference DATA.csv --reference-metadata META.json`。入口校验工况、归一化、每个翼段上下表面覆盖，拒绝外推及含糊的多值横坐标；来源和输入哈希随报告保存。\n')
    manifest=json.loads((out/'manifest.json').read_text());key='scripts/compare_three_element_cp.py';manifest['source_sha256'][key]=sha(Path(__file__));target=out/'source-snapshot'/key;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(Path(__file__).read_bytes());manifest['artifacts_sha256']=artifact_manifest(out);write_json(out/'manifest.json',manifest)
    print(json.dumps(dict(surface_faces=len(rows),reference_status=reference_record['status'],converged=summary['converged'],physics_accepted=False)))

if __name__=='__main__':main()
