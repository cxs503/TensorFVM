"""Independent NumPy geometry, momentum flux and reference reconstruction."""
import json
import math
from pathlib import Path
import numpy as np
from .core import sha
from .audit import close


def annular_metrics(f,c):
    ri,ro,mu,G=c['inner_radius_m'],c['outer_radius_m'],c['viscosity_pa_s'],c['pressure_gradient_pa_m'];center=f['cell_centers_m'].reshape(-1,2);vol=f['cell_volumes_m2'];w=f['axial_velocity_m_s'].ravel();r=np.sqrt(np.sum(center*center,1));K=(ro*ro-ri*ri)/math.log(ro/ri)
    ref=G/(4*mu)*(ro*ro-r*r-K*np.log(ro/r));close(ref,f['reference_axial_velocity_m_s'].ravel())
    o,ne,inside=f['face_owner'],f['face_neighbor'],f['interior_face_mask'];S=f['face_area_vectors_m'];disp=f['face_centers_m']-center[o];disp[inside]=center[ne[inside]]-center[o[inside]]
    coefficients=mu*np.sum(S*S,1)/np.sum(S*disp,1);close(coefficients,f['diffusion_coefficient_pa_s'],1e-11)
    flux=coefficients*w[o];flux[inside]=coefficients[inside]*(w[o[inside]]-w[ne[inside]]);close(flux,f['axial_momentum_flux_n_m'],1e-12)
    net=np.zeros(len(vol));np.add.at(net,o,flux);np.add.at(net,ne[inside],-flux[inside]);source=G*vol;close(source,f['source_force_n_m'])
    force=flux[f['wall_face_mask']].sum();fi=flux[f['inner_wall_face_mask']].sum();fo=flux[f['outer_wall_face_mask']].sum();ri_force=G*np.pi/2*(K-2*ri**2);ro_force=G*np.pi/2*(2*ro**2-K)
    Q=float(np.sum(w*vol));qr=np.pi*G/(8*mu)*(ro**4-ri**4-(ro*ro-ri*ri)**2/math.log(ro/ri))
    # Geometry closure is checked for every CV and periodic seam, not only total area.
    closure=np.zeros((len(vol),2));np.add.at(closure,o,S);np.add.at(closure,ne[inside],-S[inside]);close(closure,0.,1e-12)
    return dict(velocity_volume_l2=float(np.sqrt(np.sum(vol*(w-ref)**2)/np.sum(vol*ref**2))),velocity_linf=float(np.max(abs(w-ref))/np.max(abs(ref))),flow_rate_relative_error=abs(Q-qr)/abs(qr),inner_wall_force_relative_error=abs(fi-ri_force)/abs(ri_force),outer_wall_force_relative_error=abs(fo-ro_force)/abs(ro_force),momentum_force_balance=abs(force-source.sum())/abs(source.sum()),true_linear_residual=float(np.linalg.norm(net-source)/np.linalg.norm(source)),area_relative_error=abs(vol.sum()-np.pi*(ro**2-ri**2))/(np.pi*(ro**2-ri**2)))


def audit(directory):
    root=Path(__file__).resolve().parents[3];directory=Path(directory);s=json.loads((directory/'summary.json').read_text());cases=[]
    for name,digest in s['provenance']['source_sha256'].items():
        if sha(root/name)!=digest:raise ValueError('source changed: '+name)
    for row in s['annular_runs']:
        d=directory/row['directory'];f=np.load(d/'fields.npz',allow_pickle=False);values=annular_metrics(f,row['config'])
        if f['axial_velocity_m_s'].size!=row['diagnostics']['full_matrix_unknowns']:raise ValueError('unknown count mismatch')
        if set(values)!={q['name'] for q in row['metrics']}:raise ValueError('incomplete annular metric audit')
        for q in row['metrics']:
            close(q['value'],values[q['name']],2e-10)
            limit=2e-10 if q['name']=='true_linear_residual' else 1e-9 if q['name']=='momentum_force_balance' else .03
            if q['limit']!=limit or q['passed']!=(values[q['name']]<limit):raise ValueError('false annular gate')
        if row['passed']!=all(q['passed'] for q in row['metrics']):raise ValueError('false annular badge')
        cases.append(dict(directory=row['directory'],raw_reconstruction_passed=True,physical_passed=row['passed'],unknowns=int(f['axial_velocity_m_s'].size)))
    if len(cases)!=3 or [r['config']['ntheta'] for r in s['annular_runs']]!=[256,512,1024] or [r['config']['nr'] for r in s['annular_runs']]!=[64,128,256]:raise ValueError('required engineering refinement matrix missing')
    orders=[]
    for a,b in zip(s['annular_runs'][:-1],s['annular_runs'][1:]):
        av=next(q['value'] for q in a['metrics'] if q['name']=='velocity_volume_l2');bv=next(q['value'] for q in b['metrics'] if q['name']=='velocity_volume_l2');orders.append(math.log(av/bv)/math.log(2))
    close(orders,s['velocity_observed_orders']);passed=all(c['physical_passed'] for c in cases) and all(1.8<v<2.2 for v in orders)
    if s['annular_qualified']!=passed:raise ValueError('false suite badge')
    # SciPy scalar bracketing provides an independent wall-law inverse and traction check.
    from scipy.optimize import brentq
    row=s['wall_law'];f=np.load(directory/'wall-law/fields.npz');yp=f['y_plus'];up=[]
    for value in yp:
        def fn(u):
            z=.41*u;return u+(math.exp(z)-1-z-z*z/2-z*z*z/6)/9.8-value
        up.append(brentq(fn,0.,min(float(value),1000.) if value<1000 else 1000.,xtol=1e-13))
    close(up,f['u_plus_reference'],1e-9)
    speed=np.asarray(up)*f['reference_friction_velocity_m_s'];tau=np.linalg.norm(f['computed_traction_pa'],axis=1);expected=f['density_kg_m3']*f['reference_friction_velocity_m_s']**2
    error=float(np.max(abs(tau-expected)/expected));close(error,row['maximum_traction_relative_error'],1e-9)
    if error>=1e-8 or row['passed']!=(error<1e-8):raise ValueError('wall-law inverse or badge failed')
    if np.any(np.sum(f['computed_traction_pa']*f['relative_tangential_velocity_m_s'],axis=1)>0):raise ValueError('wall traction adds energy')
    return dict(raw_reconstruction_passed=True,annular_qualified=passed,annular_cases=cases,wall_law_constitutive_verification=True,wall_law_physical_validation=False,summary_sha256=sha(directory/'summary.json'),auditor_sha256=sha(Path(__file__)),scope='full NumPy geometry/face coefficient, momentum balance and analytic flow; independent SciPy wall-law roots; no SA/industrial-flow certification')
