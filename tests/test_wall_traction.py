"""Verify physical no-slip wall identities and curved-wall consistency."""
from pathlib import Path
import json
import numpy as np
import torch
from tensorfvm.wall_traction import incompressible_wall_gradient


def test_flat_incompressible_wall_exact_shear():
    # u=(2y,0) has no slip at y=0 and exact wall-normal derivative du/dy=2.
    velocity=torch.tensor([[.2,0.],[.6,0.]],dtype=torch.float64)
    d=torch.tensor([[0.,-.1],[.1,-.3]],dtype=torch.float64)
    area=torch.tensor([[0.,-.4],[0.,-.2]],dtype=torch.float64)
    g=incompressible_wall_gradient(velocity,d,area)
    expected=torch.tensor([[0.,0.],[2.,0.]],dtype=torch.float64).expand(2,-1,-1)
    torch.testing.assert_close(g,expected,rtol=0,atol=1e-14)
    torch.testing.assert_close(torch.diagonal(g,dim1=1,dim2=2).sum(1),torch.zeros(2,dtype=torch.float64),rtol=0,atol=1e-14)


def test_circular_manufactured_wall_shear_converges():
    # Incompressible swirl u_theta=r-R has exact circular-wall shear 1.
    root=Path(__file__).resolve().parents[1]/'docs/verification-gmsh-cylinder'
    summary=json.loads((root/'summary.json').read_text());errors=[]
    for row in summary['runs']:
        with np.load(root/row['directory']/'fields.npz') as f:
            mask=f['surface_mask'];center=f['cell_centers_m'].reshape(-1,2)[f['face_owner'][mask]]
            radial=center-[.2,.2];r=np.linalg.norm(radial,axis=1);er=radial/r[:,None]
            velocity=(r-.05)[:,None]*np.c_[-er[:,1],er[:,0]]
            g=incompressible_wall_gradient(torch.from_numpy(velocity),torch.from_numpy(f['surface_displacement_m']),torch.from_numpy(f['face_area_vectors_m'][mask]))
            magnitude=torch.linalg.matrix_norm(g).numpy()
            errors.append(float(np.linalg.norm(magnitude-1)/np.sqrt(len(magnitude))))
    assert all(a>b for a,b in zip(errors,errors[1:]))
    assert errors[-1]<.03
    assert np.log(errors[0]/errors[1])/np.log(2)>.8
