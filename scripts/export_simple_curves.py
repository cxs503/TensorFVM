"""Export the raw numerical values plotted in the full SIMPLE profiles."""
import argparse,json
from pathlib import Path
import numpy as np
from tensorfvm.verification.core import sha,write_json,artifact_manifest
p=argparse.ArgumentParser();p.add_argument('directory',type=Path);a=p.parse_args();d=a.directory;s=json.loads((d/'summary.json').read_text())
for case in s['cases']:
 c=case['cuda']['config'];ny,nx=c['ny'],c['nx'];f=np.load(d/case['name']/'cuda/fields.npz');x=f['cell_centers_m'].reshape(ny,nx,2);u=f['velocity_m_s'].reshape(ny,nx,2);pressure=f['pressure_pa'].reshape(ny,nx);j=ny//2;i=nx//2
 q=np.column_stack((x[j,:,0],x[j,:,1],pressure[j]));head='x_m,y_m,pressure_pa'
 if 'reference_pressure_pa' in f:q=np.column_stack((q,f['reference_pressure_pa'].reshape(ny,nx)[j]));head+=',reference_pressure_pa'
 np.savetxt(d/case['name']/'pressure-curve.csv',q,delimiter=',',header=head,comments='');q=np.column_stack((x[:,i,0],x[:,i,1],u[:,i]));head='x_m,y_m,u_m_s,v_m_s'
 if 'reference_velocity_m_s' in f:q=np.column_stack((q,f['reference_velocity_m_s'].reshape(ny,nx,2)[:,i]));head+=',reference_u_m_s,reference_v_m_s'
 np.savetxt(d/case['name']/'velocity-profile.csv',q,delimiter=',',header=head,comments='')
m=json.loads((d/'manifest.json').read_text());m['curve_export_source_sha256']=sha(Path(__file__));m['artifacts_sha256']=artifact_manifest(d);write_json(d/'manifest.json',m)
