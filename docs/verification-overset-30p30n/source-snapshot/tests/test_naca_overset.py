"""Actual sharp-tail NACA donor coverage and nonorthogonal two-grid diffusion."""
from dataclasses import replace
import importlib.util
from pathlib import Path
import numpy as np
import pytest
import torch
from tensorfvm.gmsh_external import GmshExternalMesh
from tensorfvm.overset import OversetGrid,cartesian_grid,build_overset
from tensorfvm.overset_scalar import solve_manufactured_diffusion
from tensorfvm.verification.external_flow import configuration


def connectivity(tmp_path,limit):
    pytest.importorskip('gmsh');torch.set_num_threads(1)
    path=Path(__file__).resolve().parents[1]/'scripts/generate_external_gmsh.py';spec=importlib.util.spec_from_file_location('naca_generator',path);generator=importlib.util.module_from_spec(spec);spec.loader.exec_module(generator)
    file=tmp_path/'component.msh';generator.generate('naca',file,4,length=4,height=3,leading_x=1.25,leading_y=1.5,geometric_incidence_deg=4,hybrid=True,outer_size_limit=limit)
    mesh=GmshExternalMesh(replace(configuration('naca',64,26,'cpu',1),mesh_type='gmsh-airfoil',mesh_file=str(file)))
    outer=np.zeros(mesh.volumes.numel(),bool);outer[mesh.owner[mesh.boundary&~mesh.masks['airfoil']].numpy()]=True
    component=OversetGrid.from_mesh(mesh,outer_boundary_mask=outer);edges=mesh.face_vertices[mesh.masks['airfoil']].numpy();links={tuple(a):tuple(b) for a,b in edges};body=[tuple(edges[0,0])]
    while len(body)<len(edges):body.append(links[body[-1]])
    assert links[body[-1]]==body[0]
    return build_overset(cartesian_grid((-2,6,-2,5),32,28),component,np.array(body),np.array([[.5,.5],[3.5,.5],[3.5,2.5],[.5,2.5]]))


def test_naca_actual_two_grid_equations_and_tail_donors(tmp_path):
    conn=connectivity(tmp_path,.24);diagnostics=conn.diagnostics()
    assert diagnostics['orphan_receivers']==0
    assert diagnostics['max_affine_coordinate_error']<1e-12
    assert all(g['fringe']>0 for g in diagnostics['grids'])
    affine=[np.c_[g.centers[:,0]+2*g.centers[:,1],3-g.centers[:,0]] for g in conn.grids]
    exchanged=conn.interpolate(affine)
    for k,state in enumerate(conn.states):np.testing.assert_allclose(exchanged[k][state==2],affine[k][state==2],atol=1e-12,rtol=0)
    result=solve_manufactured_diffusion(conn);m=result.metrics
    assert max(m['grid_0_relative_l2_error'],m['grid_1_relative_l2_error'])<.03
    assert m['physical_global_conservation_defect_relative']<.03
    assert m['relative_algebraic_residual']<1e-10
    assert m['maximum_donor_constraint_residual']<1e-10
    assert not m['navier_stokes_verified'] and not m['strict_local_conservation']
    assert np.max(abs(result.fields[1]-result.exact_fields[1]))>1e-4


def test_coarse_component_exterior_cannot_silently_leave_orphans(tmp_path):
    with pytest.raises(ValueError,match='Orphan overset receivers'):
        connectivity(tmp_path,None)
