"""Common paper-style publication for structured or unstructured external meshes."""
import argparse
import csv
import json
import textwrap
from pathlib import Path
import numpy as np
from matplotlib.collections import PolyCollection, LineCollection
from matplotlib.backends.backend_pdf import PdfPages
from .verification.core import sha, write_json, artifact_manifest
from .verification.report import plt, figure


def polygons(fields):
    if 'cell_polygons_m' in fields:
        return fields['cell_polygons_m']
    v=fields['vertices_m']
    return np.stack([v[:-1,:-1],v[1:,:-1],v[1:,1:],v[:-1,1:]],axis=2).reshape(-1,4,2)


def publish(output):
    output=Path(output)
    summary=json.loads((output/'summary.json').read_text())
    rows=summary['runs'];case=rows[0]['case'];pages=[]
    equation_audit=json.loads((output/'equations-audit.json').read_text()) if (output/'equations-audit.json').exists() else None
    title='Body-fitted finite-volume '+('DFG 2D-1 cylinder' if case=='cylinder' else 'NACA0012')+' verification'
    text='# '+title+'\n\n'
    text+='## Problem and reference\n\n'
    if case=='cylinder':
        text+='DFG 2D-1: steady incompressible laminar flow, Re=20, channel 2.2 by 0.41 m, cylinder center (0.2,0.2) m and radius 0.05 m. The mean inlet speed is 0.2 m/s, density 1 kg/m3 and dynamic viscosity 0.001 Pa s. No-slip applies to the cylinder and channel walls. The inlet parabolic velocity is integrated exactly over each physical face; outlet pressure is zero with zero-gradient velocity. Drag, lift and front/rear pressure difference are compared with the [FeatFlow reference](https://wwwold.mathematik.tu-dortmund.de/~featflow/en/benchmarks/cfdbenchmarking/flow/dfg_benchmark1_re20.html).\n\n'
    elif not rows[0].get('reference_domain_matched'):
        text+='Steady incompressible laminar NACA0012 flow at Re=1000 and incidence 4 degrees, unit chord, domain 20 by 16 m and leading edge (6,8) m. The freestream vector is (cos(4 deg),sin(4 deg)) m/s. The body is no-slip; prescribed velocity is applied at the inlet and top/bottom far-field boundaries, and outlet pressure is zero. Drag/lift use aerodynamic axes. Comparisons use vector-digitized present-study markers in [Di Ilio et al. (2020), figures 10 and 11](https://arxiv.org/html/2006.10487), with +/-0.0001 graph-reading intervals and worst endpoint relative errors. The paper specifies open top/bottom boundaries, whereas this diagnostic retains prescribed freestream there. The paper also uses a 36C by 16C domain and places the quarter-chord point 12C from the inlet. Both boundary and domain differences must be resolved before claiming a matched reference benchmark. No matched 4-degree tabulated Cp reference is available.\n\n'
    else:
        text+='NACA0012 at Re=1000, incidence 4 degrees, unit chord and the reference 36C by 16C domain. The quarter-chord point is (12C,8C). The body is rotated clockwise by 4 degrees about that point, while inlet velocity is (1,0) m/s; drag and lift are global x/y forces in these physical freestream axes. The inlet has uniform velocity; the outlet and open top/bottom boundaries have zero pressure, zero-gradient outflow velocity, and freestream velocity on backflow. The paper specifies open boundaries but does not give an exact FV formula, so this pressure-open implementation is an explicit modeling choice. Geometry/domain values are independently read from [Di Ilio et al. figure 3](https://arxiv.org/html/2006.10487). Figures 10 and 11 give digitized coefficient references and +/-0.0001 graph-reading intervals; worst endpoint errors are used. The domain and boundary review is archived in docs/naca-reference-domain-review.json.\n\n'
    text+='## Mesh and discretization\n\n'
    text+='Gmsh 4.15.2 frontal Delaunay generates conforming triangles with physical boundary tags; cylinder CAD arcs or a closed NACA0012 interpolating spline define the body. Physical wall faces are linear chords. See the [Gmsh official manual](https://gmsh.info/doc/texinfo/gmsh.html) for algorithm 6 and Distance/Threshold fields. Neither Cartesian obstacle masks nor staircase boundaries are used. Positive volumes and projected face distances and a maximum nonorthogonality below 70 degrees are required before solving. The 70-degree project gate is a screening criterion, not proof of sufficient accuracy.\n\n'
    scheme=rows[0].get('discretization',{})
    text+='The recorded discretization is `'+json.dumps(scheme,sort_keys=True)+'`. Linear upwind uses the least-squares gradient of the upstream cell and the actual face-to-cell displacement; its conservative deferred correction is inspired by [OpenFOAM linearUpwind](https://github.com/OpenFOAM/OpenFOAM-dev/blob/master/src/finiteVolume/interpolation/surfaceInterpolation/schemes/linearUpwind/linearUpwind.C). This is unlimited reconstruction, so no bounded/TVD claim is made. Gauss pressure gradients use geometric face interpolation and the same boundary face pressure used by pressure traction. Nonorthogonal diffusion, physical Rhie-Chow flux and no-slip reconstructed molecular stress are retained. Cylinder acceleration uses a full-grid coupled Picard matrix with an independent response check; NACA uses SIMPLE and Anderson mixing. CPU sparse solves check true residuals. No reference value enters the equations, initialization or relaxation.\n\n'
    if rows[0].get('wall_postprocessed_without_resolving'):
        text+='## Wall-traction comparison\n\nThe numerical solution is reused unchanged from verification-gmsh-cylinder. This report changes only viscous wall-traction postprocessing; cell velocity, pressure, mass flux, steady histories and pressure traction are bitwise identical to the archived input. The source field SHA256 and original numerical producer snapshot are supplied. At a stationary incompressible no-slip wall, tangential velocity derivatives vanish and the normal derivative of normal velocity is zero. The reconstruction therefore estimates only the normal derivative of tangential velocity from its owner-cell value and normal distance. It is a first-order wall-shear estimate; the finite cell-to-wall reconstruction remainder contains higher-order normal velocity and is not the prescribed wall velocity. A separate divergence-free circular swirl with exact wall shear gives relative L2 errors of 5.005%, 2.508% and 1.342% on these meshes; this analytic reconstruction test does not replace the solved external-flow fields. Original general-gradient traction results remain archived.\n\nReproduction: `PYTHONPATH=src python scripts/postprocess_incompressible_wall.py` after generating the original cylinder study. The commands below reproduce the upstream flow solve.\n\n'
    text+='## Reproduction and acceptance\n\n```bash\nuv pip install --python /path/to/python gmsh numpy scipy matplotlib\nOMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=src python -m '+('tensorfvm.verification.external_matched_naca' if rows[0].get('reference_domain_matched') else 'tensorfvm.verification.external_gmsh --case '+case)+' --scales '+' '.join(str(r['mesh_scale']) for r in rows)+('' if rows[0].get('reference_domain_matched') else ' --solver '+rows[0]['solver'])+' --max-iterations '+str(rows[0]['config']['max_iterations'])+' --output results/'+case+'\nPYTHONPATH=src python scripts/audit_external_equations.py results/'+case+'\nPYTHONPATH=src python -m tensorfvm.verification.external_mesh_report results/'+case+'\n```\n\n'
    text+='Raw mesh/field files, velocity/pressure arrays, face mass fluxes, separate tractions, full histories, linear residuals and SHA256 source/artifact manifests accompany this report. Each physical relative error must be strictly below 3%; steady equation residuals must be below 1e-6 and the physical flux fixed-point defect below 1e-7. Source-preserving independent NumPy replay checks full configured equations. Three runs do not establish an asymptotic grid regime; qualification of the finest run does not qualify the entire refinement sequence. GPU and matched TensorLBM runs have not been performed for this new unstructured path.\n\n'
    text+='## Results\n\n|Method|Mesh scale|Actual cells|Max nonorthogonality (deg)|Iterations|Steady|Cd|Cl|Maximum physical error|Run accepted|\n|---|---:|---:|---:|---:|---|---:|---:|---:|---|\n'
    for row in rows:
        error=max(q['value'] for q in row['metrics'] if q['name'].endswith('relative_error'))
        text+=f"|{row['solver']}|{row['mesh_scale']:g}|{row['mesh_quality']['cells']}|{row['mesh_quality']['nonorthogonality_max_deg']:.3f}|{row['iterations']}|{row['converged']}|{row['computed']['drag']:.9g}|{row['computed']['lift']:.9g}|{error*100:.4f}%|{row['passed']}|\n"
    text+='\nPer-metric errors and all reference intervals are in `summary.json`; independent replay is in `equations-audit.json`.\n\n'
    with PdfPages(output/'report.pdf') as pdf:
        fig=plt.figure(figsize=(8.27,11.69))
        fig.text(.07,.94,title,fontsize=14)
        lines=['Actual body-fitted Gmsh meshes; strict physical relative error <3%.','Scheme: '+str(scheme),'CPU calculations; no new GPU or matched TensorLBM result.','Positive mesh geometry and nonorthogonality gate precede all solves.']
        for row in rows:
            lines+=['',f"{row['resolution']}: {row['mesh_quality']['cells']} cells; converged={row['converged']}; accepted={row['passed']}",str(row['computed']),str({q['name']:round(q['value'],7) for q in row['metrics'] if q['name'].endswith('relative_error')})]
        lines+=['','Full independently replayed equations: '+str(equation_audit['passed'] if equation_audit else 'not yet audited'),'Refinement sequence accepted: '+str(summary['passed']),'Asymptotic grid regime qualified: False']
        if case=='naca' and not rows[0].get('reference_domain_matched'):lines+=['','Reference paper uses open top/bottom boundaries.','Current diagnostic imposes freestream there; matching remains unresolved.']
        fig.text(.07,.87,'\n'.join(lines),va='top',fontsize=8);pdf.savefig(fig);plt.close(fig)
        prose=text.split('## Results')[0]
        manuscript_lines=[]
        for paragraph in prose.split('\n'):
            manuscript_lines.extend(textwrap.wrap(paragraph,width=100) or [''])
        frontpages=1
        for start in range(0,len(manuscript_lines),65):
            fig=plt.figure(figsize=(8.27,11.69));fig.text(.07,.95,'Problem, numerical method and reproducibility',fontsize=13)
            fig.text(.07,.91,'\n'.join(manuscript_lines[start:start+65]),va='top',fontsize=8.5,linespacing=1.4)
            pdf.savefig(fig);plt.close(fig);frontpages+=1
        for row in rows:
            directory=output/row['directory'];figs=directory/'figures';figs.mkdir(exist_ok=True)
            with np.load(directory/'fields.npz') as f:
                cells=polygons(f);u=f['cell_velocity_m_s'];pressure=f['cell_pressure_pa'];c=row['config']
                if case=='cylinder':bounds=(.05,.65,.02,.39)
                else:bounds=(c['airfoil_x']-.2,c['airfoil_x']+1.8,c['airfoil_y']-.5,c['airfoil_y']+.5)
                for name,values,label in [('pressure',pressure,'Pressure (Pa)'),('velocity',np.linalg.norm(u,axis=1),'Speed (m/s)'),('u',u[:,0],'u (m/s)'),('v',u[:,1],'v (m/s)')]:
                    views=['global','local'] if name in ('pressure','velocity') else ['local']
                    for view in views:
                        fig,ax=plt.subplots(figsize=(9,4))
                        collection=PolyCollection(cells,array=values,cmap='viridis',rasterized=True)
                        ax.add_collection(collection);ax.autoscale_view();ax.set_aspect('equal')
                        if view=='local':ax.set_xlim(bounds[:2]);ax.set_ylim(bounds[2:])
                        ax.set(xlabel='x (m)',ylabel='y (m)',title=f"{case}: {row['mesh_quality']['cells']} cells")
                        fig.colorbar(collection,ax=ax,label=label)
                        basename=name+'-'+view;figure(fig,figs,basename);pdf.savefig(fig);plt.close(fig)
                        pages.append(row['directory']+'/figures/'+basename)
                fig,ax=plt.subplots(figsize=(9,4))
                edges=np.concatenate([cells,np.roll(cells,-1,axis=1)],axis=-1).reshape(-1,2,2)
                ax.add_collection(LineCollection(edges,colors='black',linewidths=.12,rasterized=True))
                ax.set_xlim(bounds[:2]);ax.set_ylim(bounds[2:]);ax.set_aspect('equal');ax.set(xlabel='x (m)',ylabel='y (m)',title='Actual body-fitted mesh')
                figure(fig,figs,'mesh-local');pdf.savefig(fig);plt.close(fig);pages.append(row['directory']+'/figures/mesh-local')
                centers=f['surface_centers_m'];wallp=f['surface_wall_pressure_pa']
                fig,ax=plt.subplots(figsize=(7,4))
                if case=='cylinder':
                    x=np.mod(np.arctan2(centers[:,1]-.2,centers[:,0]-.2),2*np.pi);order=np.argsort(x)
                    ax.plot(np.degrees(x[order]),wallp[order]);ax.set(xlabel='Angle (deg)',ylabel='Wall pressure (Pa)')
                else:
                    q=np.array([c['airfoil_x']+.25,c['airfoil_y']])
                    angle=-np.radians(row.get('mesh_generation',{}).get('geometric_incidence_deg',0))
                    rotation=np.array([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]])
                    body_centers=(centers-q)@rotation+q
                    for upper in (True,False):
                        idx=np.flatnonzero((body_centers[:,1]>=c['airfoil_y'])==upper);idx=idx[np.argsort(body_centers[idx,0])]
                        ax.plot(body_centers[idx,0]-c['airfoil_x'],wallp[idx]/.5,label='Upper' if upper else 'Lower')
                    ax.set(xlabel='x/c',ylabel='Cp (outlet pressure reference)');ax.invert_yaxis();ax.legend()
                ax.grid(alpha=.3);figure(fig,figs,'surface-pressure');pdf.savefig(fig);plt.close(fig);pages.append(row['directory']+'/figures/surface-pressure')
            history=json.loads((directory/'history.json').read_text());fig,ax=plt.subplots(figsize=(7,4))
            for key in ('momentum','continuity','mass_imbalance','rhie_chow_flux_defect'):
                if key in history[-1]:ax.semilogy([entry[key] for entry in history],label=key)
            ax.set(xlabel='Iteration',ylabel='Actual post-update residual');ax.legend();ax.grid(alpha=.3)
            figure(fig,figs,'residuals');pdf.savefig(fig);plt.close(fig);pages.append(row['directory']+'/figures/residuals')
        fig,ax=plt.subplots(figsize=(7,4));refinement={}
        for metric in rows[0]['reference']:
            errors=[next(q['value'] for q in row['metrics'] if q['name']==metric+'_relative_error') for row in rows]
            values=[row['computed'][metric] for row in rows]
            changes=[abs(b-a)/max(abs(b),1e-30) for a,b in zip(values,values[1:])]
            refinement[metric]=dict(values=values,relative_errors=errors,successive_relative_changes=changes)
            if summary.get('study_type')=='discretization-control':
                ax.semilogy(np.arange(len(rows)),100*np.array(errors),'o-',label=metric)
            else:
                ax.loglog([row['mesh_quality']['cells'] for row in rows],100*np.array(errors),'o-',label=metric)
        ax.axhline(3,color='red',linestyle='--',label='Strict 3% gate');ax.set(xlabel='Actual fluid cells',ylabel='Physical relative error (%)');ax.legend();ax.grid(alpha=.3)
        if summary.get('study_type')=='discretization-control':
            ax.set_xticks(np.arange(len(rows)),[row['solver'].replace('-', '\n') for row in rows]);ax.set_xlabel('Discretization on the identical mesh')
        figure(fig,output,'refinement');pdf.savefig(fig);plt.close(fig);pages.append('refinement')
    write_json(output/'refinement-audit.json',dict(metrics=refinement,grid_convergence_qualified=False,reason='Physical acceptance and a demonstrable asymptotic regime are separate gates.'))
    with (output/'results.csv').open('w',newline='') as stream:
        writer=csv.writer(stream);writer.writerow(['mesh_scale','actual_cells','iterations','converged','drag','lift','pressure_difference_pa','passed','nonorthogonality_max_deg','elapsed_s'])
        for row in rows:writer.writerow([row['mesh_scale'],row['mesh_quality']['cells'],row['iterations'],row['converged'],row['computed']['drag'],row['computed']['lift'],row['computed'].get('pressure_difference_pa'),row['passed'],row['mesh_quality']['nonorthogonality_max_deg'],row['elapsed_s']])
    for name in pages:text+='\n!['+name+']('+name+'.png)\n'
    text+='\n## Discussion\n\nMesh validity, mesh quality, equation convergence and reference accuracy are reported separately. The refined physical coefficients and successive changes above determine the next development step; no failed case is relabeled as qualified. Wall geometry consists of linear chords, and viscous traction is reconstructed from no-slip data. Pressure-gradient and transport changes require full equation replay, not just a lower internal residual. The original radial-grid studies and their failures remain archived for comparison.\n'
    (output/'report.md').write_text(text)
    tex=r'\documentclass[11pt]{article}\usepackage[a4paper,margin=22mm]{geometry}\usepackage{graphicx,booktabs,hyperref}\begin{document}\title{'+title+r'}\maketitle\section{Problem and method}'+('DFG 2D-1, Re=20; parabolic inlet and no-slip circular cylinder in a channel.' if case=='cylinder' else 'NACA0012, Re=1000, incidence 4 degrees. Reference domain and pressure-open boundaries are used.' if rows[0].get('reference_domain_matched') else 'NACA0012, Re=1000, incidence 4 degrees. Domain size and prescribed far-field velocity differ from the reference study.')+r' Conforming Gmsh triangles, conservative linear upwind transport, geometric interpolation and Gauss face-pressure integration are verified using raw fields and independent NumPy equation replay. Strict physical errors are below 3\% only for runs explicitly accepted below. Full problem, software commands, references and limits are in report.md.\section{Results}\begin{tabular}{rrrrl}\toprule Cells & Iterations & $C_D$ & $C_L$ & Accepted\\\midrule '
    for row in rows:tex+=f"{row['mesh_quality']['cells']} & {row['iterations']} & {row['computed']['drag']:.9g} & {row['computed']['lift']:.9g} & {row['passed']} \\\\\n"
    tex+=r'\bottomrule\end{tabular}\section{Computed fields and convergence}'
    for name in pages:tex+=r'\begin{figure}[p]\centering\includegraphics[width=.95\linewidth]{'+name+r'.pdf}\caption{'+name.replace('/',' ').replace('-',' ')+r'}\end{figure}'
    tex+=r'\end{document}'
    (output/'report.tex').write_text(tex+'\n')
    manifest=json.loads((output/'manifest.json').read_text())
    root=Path(__file__).resolve().parents[2]
    source_key=str(Path(__file__).relative_to(root))
    manifest['source_sha256'][source_key]=sha(Path(__file__))
    if manifest.get('source_snapshot_root'):
        import shutil
        target=output/manifest['source_snapshot_root']/source_key;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(Path(__file__),target)
    manifest['artifacts_sha256']=artifact_manifest(output);write_json(output/'manifest.json',manifest)
    print(output,'published',len(pages)+frontpages,'PDF pages',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('outputs',type=Path,nargs='+');args=parser.parse_args()
    for output in args.outputs:publish(output)
