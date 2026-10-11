"""Frozen Stokes correction audit; no physical benchmark qualification."""
from pathlib import Path
import sys,json
import numpy as np
from scipy import sparse
repo=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(repo/'src'),str(repo)]
from tests.test_native_collocated3d import skew_mesh
from tensorfvm.native_simple3d import NativeSimpleSolver3D
from tensorfvm.native_collocated3d import NativeSteadySettings
rows=[]
for outlet_enabled in (False,True):
    m=skew_mesh(3);boundary=m.neighbor<0
    outlet=boundary&(m.face_centers[:,0]>1.) if outlet_enabled else np.zeros(len(m.owner),bool)
    x,y,z=m.cell_centers.T;force=np.column_stack((np.sin(3*y),np.cos(2*x),x*z))
    s=NativeSimpleSolver3D(m,boundary&~outlet,np.zeros((len(m.owner),3)),outlet,
        body_force=force,settings=NativeSteadySettings(viscosity=.1,convection=False,max_iterations=1,relaxation=.5,pressure_relaxation=1.))
    r=s.solve();row=r['history'][-1]
    assert row['native_divergence_max']<1e-11
    assert row['current_reconstructed_divergence_max']<1e-11
    assert row['face_flux_consistency_max']<1e-11
    rows.append(dict(pressure_outlet=outlet_enabled,**row))
print(json.dumps(dict(cells=27,frozen_coefficients=True,pressure_relaxation=1.,checks_passed=True,physical_validation_passed=False,rows=rows),indent=2))
