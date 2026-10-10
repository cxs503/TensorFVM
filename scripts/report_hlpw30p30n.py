"""Actual HLPW4 three-element fields versus original LTPT Cp, without forcing.

All experimental taps are plotted. Quantitative interpolation refuses ambiguous
multi-valued abscissae and out-of-support taps; incomplete coverage cannot pass.
"""
import argparse,csv,json,shutil
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib.backends.backend_pdf import PdfPages
from tensorfvm.multi_element import ThreeElementMesh
from tensorfvm.hlpw30p30n import DATA,metadata,reference_zones
from tensorfvm.verification.core import sha,write_json,artifact_manifest


def loops(mesh):
    result={}
    for name in ('slat','main','flap'):
        ids=torch.nonzero(mesh.masks[name]).flatten().numpy();edges=mesh.face_vertices[ids].numpy();links={tuple(a):i for i,(a,b) in enumerate(edges)}
        order=[0]
        while len(order)<len(ids):
            order.append(links[tuple(edges[order[-1],1])])
        if len(set(order))!=len(ids) or not np.array_equal(edges[order[-1],1],edges[order[0],0]):raise ValueError('Invalid closed wall contour')
        ids=ids[order];vertices=mesh.face_vertices[ids].numpy()[:,0];centers=mesh.face_centers[ids].numpy();length=mesh.face_lengths[ids].numpy();le=vertices[:,0].argmin();te=vertices[:,0].argmax();branch=np.zeros(len(ids),bool);i=le
        while i!=te:branch[i]=True;i=(i+1)%len(ids)
        tangent=vertices[te]-vertices[le];normal=np.array([-tangent[1],tangent[0]]);position=centers@normal
        if np.average(position[branch],weights=length[branch])<np.average(position[~branch],weights=length[~branch]):branch=~branch
        result[name]=(ids,vertices,centers,branch)
    return result


def crossing_interpolation(x,y,target):
    # Only interpolate along adjacent contour faces, never after sorting x.
    candidates=[]
    for i in range(len(x)-1):
        if abs(x[i+1]-x[i])<1e-13:continue
        t=(target-x[i])/(x[i+1]-x[i])
        if 0<=t<=1:candidates.append(float((1-t)*y[i]+t*y[i+1]))
    if len(candidates)==1:return candidates[0],None
    return None,'ambiguous_x' if candidates else 'outside_face_center_support'


def runs(mask):
    i=np.flatnonzero(mask)
    return np.split(i,np.flatnonzero(np.diff(i)>1)+1)


def validate_polygon_walls(wall):
    errors={}
    for name,(_,vertices,_,_) in wall.items():
        canonical=np.loadtxt(DATA/(name+'.dat'))
        edges=canonical[1:]-canonical[:-1];start=canonical[:-1];d=vertices[:,None]-start
        t=np.clip(np.einsum('nki,ki->nk',d,edges)/np.einsum('ki,ki->k',edges,edges),0,1)
        errors[name]=float(np.linalg.norm(d-t[:,:,None]*edges,axis=-1).min(1).max())
        if errors[name]>1e-8:raise ValueError('Actual mesh differs from official sampled '+name+' polygon')
    return dict(representation='official sampled polygon',max_wall_node_distance_over_c=errors)


def validate_cad_walls(wall,cad_path,expected_sha):
    """Check actual wall vertices against the original STEP BSpline curves."""
    import gmsh
    if sha(cad_path)!=expected_sha:raise ValueError('Original STEP SHA mismatch')
    errors={};gmsh.initialize()
    try:
        gmsh.option.setNumber('General.Terminal',0)
        gmsh.model.occ.importShapes(str(cad_path),highestDimOnly=False)
        gmsh.model.occ.synchronize()
        groups={'main':range(1,6),'flap':range(6,9),'slat':range(9,12)}
        if len(gmsh.model.getEntities(1))!=11:raise ValueError('Unexpected original STEP topology')
        for name,(_,vertices,_,_) in wall.items():
            xyz=np.column_stack([vertices*457.2,np.zeros(len(vertices))]);best=np.full(len(vertices),np.inf)
            for tag in groups[name]:
                closest,parameters=gmsh.model.getClosestPoint(1,tag,xyz.ravel().tolist())
                closest=np.asarray(closest).reshape(-1,3);parameters=np.asarray(parameters)
                low,high=gmsh.model.getParametrizationBounds(1,tag)
                valid=(parameters>=float(low[0])-1e-10)&(parameters<=float(high[0])+1e-10)
                distances=np.linalg.norm(closest-xyz,axis=1)
                distances[~valid]=np.inf
                for parameter in [float(low[0]),float(high[0])]:
                    endpoint=np.asarray(gmsh.model.getValue(1,tag,[parameter]))
                    distances=np.minimum(distances,np.linalg.norm(xyz-endpoint,axis=1))
                best=np.minimum(best,distances)
            errors[name]=float(best.max()/457.2)
            if errors[name]>1e-8:raise ValueError('Actual mesh differs from original STEP '+name+' curves')
    finally:gmsh.finalize()
    return dict(representation='official STEP BSpline',cad_sha256=expected_sha,
                imported_units='mm',nominal_chord_mm=457.2,max_wall_node_distance_over_c=errors)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('output',type=Path);args=p.parse_args();out=args.output;torch.set_num_threads(1)
    summary=json.loads((out/'benchmark.json').read_text());config=summary['config'];meta=metadata()
    settings=json.loads((out/'producer.json').read_text())['numerical_settings']
    gradient_method=settings.get('pressure_gradient','unspecified')
    stress_method=settings.get('viscous_stress','scalar variable-viscosity Laplacian')
    sa_method='experimental Newton update' if settings.get('sa_newton') else 'standard positive-production Picard update'
    if config['reynolds']!=meta['reynolds'] or config['airfoil_chord']!=1:raise ValueError('HLPW reference requires Re9e6 based on nominal C=1')
    ref=reference_zones(config['angle_of_attack']);mesh=ThreeElementMesh(SimpleNamespace(mesh_file=out/'mesh.msh',device='cpu'));wall=loops(mesh);data=np.genfromtxt(out/'fields.csv',delimiter=',',names=True)
    if len(data)!=mesh.volumes.numel() or not np.isfinite(data['p']).all():raise ValueError('Invalid actual field')
    mesh_producer=out/'mesh-producer-mesh-metadata.json'
    mesh_metadata=json.loads(mesh_producer.read_text()) if mesh_producer.exists() else {}
    if mesh_metadata.get('geometry_representation')=='official-step-bspline':
        cad_path=DATA/'original/Geometry/2010_30p30n_thik_te_18inches.stp'
        geometry_validation=validate_cad_walls(wall,cad_path,mesh_metadata['cad_sha256'])
    else:geometry_validation=validate_polygon_walls(wall)
    write_json(out/'geometry-validation.json',geometry_validation)
    dynamic=.5*config['density']*config['inlet_velocity']**2;cp=data['p']/dynamic;metrics=[];rows=[];excluded=[];curves={}
    pressure_method='Adjacent owner-cell pressure; prescribed outlet p_inf=0'
    wall_face_cp=None
    if (out/'wall-face-pressure.csv').exists():
        face_data=np.genfromtxt(out/'wall-face-pressure.csv',delimiter=',',names=True)
        face_ids=face_data['face'].astype(int)
        if set(face_ids)!=set(torch.nonzero(mesh.masks['wall']).flatten().numpy()) or len(set(face_ids))!=len(face_ids):
            raise ValueError('Incomplete/duplicated reconstructed wall pressure')
        wall_face_cp=np.full(len(mesh.owner),np.nan);wall_face_cp[face_ids]=face_data['p']/dynamic
        if not np.isfinite(wall_face_cp[face_ids]).all():raise ValueError('Invalid reconstructed wall pressure')
        pressure_method='Actual solver skew-corrected wall face pressure; prescribed outlet p_inf=0'
    # A first-cell wall-shear estimate diagnoses resolution; it is not a
    # replacement for the solver's reconstructed viscous force integration.
    wall_resolution={};resolution_rows=[]
    nu=config['inlet_velocity']*config['airfoil_chord']/config['reynolds']
    velocity=np.column_stack([data['u'],data['v']]);cell_centers=mesh.centers.reshape(-1,2).numpy()
    for name,(ids,_,centers,_) in wall.items():
        owner=mesh.owner[ids].numpy();normal=mesh.face_normals[ids].numpy()
        distance=np.abs(np.einsum('ij,ij->i',centers-cell_centers[owner],normal))
        tangent_velocity=velocity[owner]-np.einsum('ij,ij->i',velocity[owner],normal)[:,None]*normal
        tangential_speed=np.linalg.norm(tangent_velocity,axis=1)
        yplus=np.sqrt(tangential_speed*distance/nu)
        wall_resolution[name]=dict(first_cell_normal_distance_min=float(distance.min()),first_cell_normal_distance_max=float(distance.max()),y_plus_estimate_max=float(yplus.max()),y_plus_estimate_p95=float(np.percentile(yplus,95)),method='Molecular first-cell tangential velocity / normal distance; diagnostic only')
        resolution_rows.extend([name,int(face),float(x),float(y),float(d),float(yp)] for face,(x,y),d,yp in zip(ids,centers,distance,yplus))
    with (out/'wall-resolution.csv').open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['element','face','x_over_c','y_over_c','normal_distance_over_c','y_plus_first_cell_estimate']);w.writerows(resolution_rows)
    write_json(out/'wall-resolution.json',wall_resolution)
    for name,(ids,vertices,centers,upper) in wall.items():
        values=cp[mesh.owner[ids].numpy()] if wall_face_cp is None else wall_face_cp[ids];curves[name]=(centers[:,0],values,upper)
        r=ref[name];split=int(r[:,0].argmin())
        for side,mask,indices in [('upper',upper,np.arange(split+1)),('lower',~upper,np.arange(split+1,len(r)))]:
            computed=[];reference=[]
            for j in indices:
                # Multiple disjoint wrapped runs are treated independently.
                candidates=[];reasons=[]
                for run in runs(mask):
                    value,reason=crossing_interpolation(centers[run,0],values[run],r[j,0])
                    if reason is None:candidates.append(value)
                    else:reasons.append(reason)
                reason=None if len(candidates)==1 and 'ambiguous_x' not in reasons else 'ambiguous_x' if len(candidates)>1 or 'ambiguous_x' in reasons else 'outside_face_center_support'
                value=candidates[0] if reason is None else None
                rows.append([name,side,int(j),r[j,0],r[j,1],value,reason or 'supported'])
                if value is None:excluded.append(dict(element=name,surface=side,tap=int(j),x_over_c=float(r[j,0]),reason=reason))
                else:computed.append(value);reference.append(r[j,1])
            error=float(np.linalg.norm(np.array(computed)-reference)/np.linalg.norm(reference)) if reference else None
            metrics.append(dict(element=name,surface=side,total_taps=len(indices),supported_taps=len(reference),relative_l2_error=error))
    with (out/'cp-reference-comparison.csv').open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['element','surface','tap','x_over_c','cp_reference','cp_computed','registration_status']);w.writerows(rows)
    with (out/'cp-experiment-original.dat').open('wb') as f:f.write((DATA/'original/Exp_Data_LTPT_Re9million/cp_exp.dat').read_bytes())
    write_json(out/'reference-metadata.json',meta)
    status='CONVERGED' if summary['converged'] else 'UNCONVERGED'
    with PdfPages(out/'hlpw-report.pdf') as pdf:
        fig,ax=plt.subplots(figsize=(12,4),constrained_layout=True)
        for name,(_,body,_,_) in wall.items():
            ax.fill(body[:,0],body[:,1],facecolor='lightgray',edgecolor='black',linewidth=.8)
            ax.annotate(name,body.mean(0),xytext=(0,25),textcoords='offset points',ha='center')
        ax.set_aspect('equal');ax.set_xlabel('x/C');ax.set_ylabel('y/C');ax.set_title('Official HLPW4 30P30N geometry: actual numerical wall contours')
        fig.savefig(out/'geometry.png',dpi=300);pdf.savefig(fig);plt.close(fig)
        fig,axes=plt.subplots(1,3,figsize=(15,5),constrained_layout=True)
        for ax,name in zip(axes,wall):
            x,y,upper=curves[name]
            for side,mask,color in [('upper',upper,'tab:blue'),('lower',~upper,'tab:orange')]:
                for j,run in enumerate(runs(mask)):ax.plot(x[run],y[run],color=color,label='TensorFVM '+side if j==0 else None)
            r=ref[name];ax.scatter(r[:,0],r[:,1],s=18,facecolors='none',edgecolors='black',label='NASA LTPT experiment')
            ax.invert_yaxis();ax.set_title(name);ax.set_xlabel('x / C (nominal stowed chord)');ax.set_ylabel('Cp');ax.grid(alpha=.25);ax.legend(fontsize=8)
        fig.suptitle(f'30P30N: Re=9 million, alpha={config["angle_of_attack"]:g} deg; experiment M=0.2\nTensorFVM incompressible SA, {status}; accuracy NOT accepted',color='darkred');fig.savefig(out/'cp-comparison.png',dpi=300);fig.savefig(out/'cp-comparison.svg');pdf.savefig(fig);plt.close(fig)
        vertices=mesh.vertices.numpy();centers=mesh.centers.reshape(-1,2).numpy();near=(centers[:,0]>-.5)&(centers[:,0]<1.7)&(centers[:,1]>-.7)&(centers[:,1]<.8);ids=np.flatnonzero(near);polys=[vertices[list(mesh.cell_nodes[i])] for i in ids]
        for field,label in [('speed','Velocity magnitude / U_inf'),('p','Pressure / (rho U_inf²)')]:
            fig,ax=plt.subplots(figsize=(12,4),constrained_layout=True);collection=PolyCollection(polys,array=data[field][ids],edgecolors='none',cmap='turbo',rasterized=True);ax.add_collection(collection);fig.colorbar(collection,ax=ax,label=label)
            for name,(_,body,_,_) in wall.items():ax.fill(body[:,0],body[:,1],color='white',edgecolor='black',linewidth=.6)
            ax.set_xlim(-.2,1.4);ax.set_ylim(-.35,.25);ax.set_aspect('equal');ax.set_title(f'Actual Gmsh cell field: {status}; accuracy NOT accepted');ax.set_xlabel('x/C');ax.set_ylabel('y/C');fig.savefig(out/(field+'.png'),dpi=300);pdf.savefig(fig);plt.close(fig)
        history=summary['final_residuals'];all_history=json.loads((out/'history.json').read_text());fig,ax=plt.subplots(figsize=(10,5),constrained_layout=True)
        for key in ['momentum','turbulence','continuity','rhie_chow_flux_defect']:
            if all_history and key in all_history[-1]:ax.semilogy([r['iteration'] for r in all_history],[max(r.get(key,np.nan),1e-18) for r in all_history],label=key)
        ax.axhline(config['tolerance'],color='gray',linestyle='--',label='Outer tolerance');ax.set_xlabel('Completed iteration');ax.set_ylabel('Residual / defect');ax.legend();ax.grid(alpha=.25);fig.savefig(out/'convergence.png',dpi=250);pdf.savefig(fig);plt.close(fig)
    alpha=config['angle_of_attack'];cldata=np.loadtxt(DATA/'original/Exp_Data_LTPT_Re9million/clalpha_exp.dat',skiprows=1);cl=float(cldata[np.isclose(cldata[:,0],alpha),1][0]);computed=summary['aerodynamic_coefficients']['lift'];clerror=abs(computed-cl)/abs(cl)
    record=dict(geometry_validation=geometry_validation,converged=summary['converged'],physics_accepted=False,reference=meta,reference_mach=.2,computed_compressibility='incompressible',reference_cl=cl,computed_cl=computed,lift_relative_error=clerror,pressure_metrics=metrics,excluded_taps=excluded,full_pressure_coverage=not excluded,wall_resolution=wall_resolution,pressure_method=pressure_method,acceptance_limit=.03,blockers=['Compressible experiment versus incompressible model','Geometry exception: thicker BANC slat trailing edge','Experimental three-dimensional effects/unknown uncertainty','Grid independence has not been demonstrated']+(['Numerical residuals have not converged'] if not summary['converged'] else [])+(['Incomplete unambiguous pressure tap registration'] if excluded else []),inputs_sha256={n:sha(out/n) for n in ['mesh.msh','fields.csv','benchmark.json','cp-experiment-original.dat']})
    write_json(out/'cp-validation.json',record)
    table='\n'.join(f"| {r['element']} | {r['surface']} | {r['supported_taps']}/{r['total_taps']} | {100*r['relative_l2_error']:.3f}% |" if r['relative_l2_error'] is not None else f"| {r['element']} | {r['surface']} | 0/{r['total_taps']} | 未比较 |" for r in metrics)
    text=f'''# 30P30N 三段翼：实际计算与 NASA LTPT 压力数据比较

## 问题与参考来源

官方 HLPW4 辅助案例包提供 BANC 三段翼几何，以及 NASA Langley LTPT 原始实验数据。参考为名义收起弦长 C=1，Re=9×10⁶，M=0.2，攻角 {alpha:g}°。几何坐标由官方 18 英寸曲线除以 18 得到，壁面表示为 {geometry_validation['representation']}，节点到原始几何的距离见 geometry-validation.json；保留缝道和尾缘厚度。官方规定远场半径 500C。原包特别说明 BANC 缝翼尾缘较原实验厚，实验存在三维效应，数据只能作为粗略指南。

[官方归档]({meta['source_url']})：`{meta['archive_member']}`；原始数据及 SHA256 见 reference-metadata.json。

## 软件与计算方法

共享 Mesh2D 有限体积模块，Gmsh 壁面结构化层与外部三角网格；不可压 SIMPLE、全湍流 Spalart–Allmaras、受限线性迎风动量重构。压力梯度={gradient_method}；粘性应力={stress_method}；SA 更新={sa_method}。每次压力矩阵与实际离散算子独立核对，线性方程检验真实残差。远场上游给定速度、下游给定零表压，这是当前不可压近似边界条件。启动阶段的迎风混合系数见残差记录；尚未进入完整重构阶段的检查点不构成目标离散格式的收敛解。

网格包含 {summary['mesh']['cells']:,} 个真实单元，完成 {summary['iterations']} 次迭代。数值收敛={summary['converged']}。未将实验 Cp、Cl 或 Cd 输入求解方程。

## 实际计算结果

![实际网格三段翼几何](geometry.png)

![三翼段 Cp：计算线与全部实验点](cp-comparison.png)

![速度场](speed.png)

![压力场](p.png)

![真实残差](convergence.png)

实验 Cl={cl:.6g}；实际计算 Cl={computed:.6g}；相对误差={100*clerror:.3f}%。这不能单独替代 Cp 与网格收敛验收。原 Wolf Dynamics 的 Cl=2.167089、Cd=0.033243 属于另一工况，未用于本表。

| 翼段 | 表面 | 无歧义支持点/实验点 | 支持点 Cp 相对 L2 误差 |
|---|---|---|---|
{table}

全部实验点均显示，定量误差仅使用相邻真实壁面压力可无歧义插值的位置。未对多值 x 的缝翼凹腔排序平均，未外推到网格支持范围外。被排除点逐一记录，支持点误差不能称为完整表面误差。壁面压力方法={pressure_method}；仍需细化验证。

## 验收结论

物理精度验收=False，尚未证明完整 Cp 误差低于 3%。

'''+ '\n'.join('- '+b for b in record['blockers'])+'\n\n[完整 PDF](hlpw-report.pdf) · [逐测点对比](cp-reference-comparison.csv) · [验收记录](cp-validation.json)\n'
    (out/'hlpw-report.md').write_text(text)
    manifest=json.loads((out/'manifest.json').read_text());root=Path(__file__).resolve().parents[1]
    for source in [Path(__file__),root/'src/tensorfvm/hlpw30p30n.py',*(DATA.rglob('*'))]:
        if not source.is_file():continue
        name=str(source.relative_to(root));manifest['source_sha256'][name]=sha(source);target=out/'source-snapshot'/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,target)
    manifest['artifacts_sha256']=artifact_manifest(out);write_json(out/'manifest.json',manifest)
    print(json.dumps(dict(iterations=summary['iterations'],converged=summary['converged'],lift_relative_error=clerror,excluded_taps=len(excluded),physics_accepted=False)))

if __name__=='__main__':main()
