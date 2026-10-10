"""Physical mesh and reconstruction contracts for the external Gmsh path."""
import importlib.util
from dataclasses import replace
from pathlib import Path
import numpy as np
import pytest
import torch
from tensorfvm.verification.external_flow import configuration
from tensorfvm.gmsh_external import GmshExternalMesh
from tensorfvm.external_high_order import ConservativeLinearUpwindCoupledSolver
from tensorfvm.mesh_api import validate_mesh2d
from tensorfvm.mesh_quality import mesh_quality


@pytest.fixture(scope='module')
def cylinder(tmp_path_factory):
    pytest.importorskip('gmsh')
    torch.set_num_threads(1)
    root=Path(__file__).resolve().parents[1]
    spec=importlib.util.spec_from_file_location('generator',root/'scripts/generate_external_gmsh.py')
    generator=importlib.util.module_from_spec(spec);spec.loader.exec_module(generator)
    path=tmp_path_factory.mktemp('gmsh')/'cylinder.msh'
    generator.generate('cylinder',path,4)
    return replace(configuration('cylinder',64,24,'cpu',100),nx=1,ny=1,mesh_type='gmsh-cylinder',mesh_file=str(path))


def test_mesh_is_conforming_and_quality_checked(cylinder):
    mesh=validate_mesh2d(GmshExternalMesh(cylinder),required_masks=('cylinder','inlet','outlet','wall'))
    assert mesh_quality(mesh)['quality_gate_passed']
    wall=mesh.face_vertices[mesh.masks['cylinder']].numpy()
    np.testing.assert_allclose(np.linalg.norm(wall-[.2,.2],axis=-1),.05,rtol=0,atol=1e-12)
    closure=torch.zeros((mesh.volumes.numel(),2),dtype=torch.float64)
    closure.index_add_(0,mesh.owner,mesh.face_area_vectors)
    closure.index_add_(0,mesh.neighbor[mesh.interior],-mesh.face_area_vectors[mesh.interior])
    assert float(closure.abs().max())<1e-14


def test_affine_upwind_reconstruction_and_conservation(cylinder):
    s=ConservativeLinearUpwindCoupledSolver(cylinder)
    centers=s.mesh.centers.reshape(-1,2)
    gradient=torch.tensor([[1.7,-.4],[.6,2.1]],dtype=torch.float64)
    velocity=centers@gradient+torch.tensor([.3,-.2])
    supplied=gradient.expand(s.count,-1,-1)
    for sign in (1,-1):
        flux=sign*torch.ones_like(s.mass_flux)
        correction=s.linear_upwind_correction(velocity,flux,supplied)
        upstream=torch.where(flux[s.f]>0,s.oi,s.ni)
        reconstructed=velocity[upstream]+correction[s.f]
        exact=s.mesh.face_centers[s.f]@gradient+torch.tensor([.3,-.2])
        torch.testing.assert_close(reconstructed,exact,rtol=0,atol=2e-14)
        assert float(s._sum(flux[:,None]*correction).sum(0).abs().max())<1e-12
        assert not torch.any(correction[~s.f])


def test_conservative_pressure_and_coupled_operator(cylinder):
    s=ConservativeLinearUpwindCoupledSolver(cylinder)
    p=torch.sin(torch.arange(s.count,dtype=torch.float64)*.19)
    face=s._interpolate(p);face[s.mesh.masks['outlet']]=0
    gradient=s._gradient(p,pressure=True)
    total=(gradient*s.volume[:,None]).sum(0)
    boundary=(face[s.mesh.boundary,None]*s.S[s.mesh.boundary]).sum(0)
    torch.testing.assert_close(total,boundary,rtol=1e-12,atol=1e-12)
    matrix,defect=s._coupled_jacobian()
    assert matrix.shape==(3*s.count,3*s.count)
    assert defect<2e-11


def test_open_airfoil_mesh_and_public_result(tmp_path):
    pytest.importorskip('gmsh')
    from tensorfvm.external_open_boundary import OpenBoundaryNacaSolver
    root=Path(__file__).resolve().parents[1]
    spec=importlib.util.spec_from_file_location('generator',root/'scripts/generate_external_gmsh.py')
    generator=importlib.util.module_from_spec(spec);spec.loader.exec_module(generator)
    path=tmp_path/'naca.msh'
    generator.generate('naca',path,6,length=36,height=16,leading_x=11.75,leading_y=8,geometric_incidence_deg=4,open_boundaries=True)
    c=replace(configuration('naca',64,26,'cpu',2),nx=1,ny=1,length=36,height=16,airfoil_x=11.75,airfoil_y=8,angle_of_attack=0,mesh_type='gmsh-airfoil-open',mesh_file=str(path))
    s=OpenBoundaryNacaSolver(c)
    assert torch.all(s.mesh.masks['outlet'][s.mesh.masks['open-boundary']])
    assert not torch.any(s.mesh.masks['far-field'])
    assert mesh_quality(s.mesh)['quality_gate_passed']
    result=s.solve()
    assert 'airfoil' in result.surface_forces
    assert 'cylinder' not in result.surface_forces
    force=s._surface_force(s.mesh.masks['airfoil'])
    assert abs(result.aerodynamic_coefficients['drag']-float(force[0]/.5))<1e-12
    assert abs(result.aerodynamic_coefficients['lift']-float(force[1]/.5))<1e-12
    assert s.history[-1]['rhie_chow_flux_defect']>=0
