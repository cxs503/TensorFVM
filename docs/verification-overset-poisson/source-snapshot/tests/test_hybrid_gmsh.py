"""Generated wall layers and conforming mixed-cell operators."""
import importlib.util
from dataclasses import replace
from pathlib import Path
import numpy as np
import pytest
import torch
from tensorfvm.verification.external_flow import configuration
from tensorfvm.gmsh_external import GmshExternalMesh
from tensorfvm.hybrid_mesh_quality import hybrid_mesh_quality
from tensorfvm.external_high_order import ConservativeLinearUpwindCoupledSolver


@pytest.mark.parametrize('case', ['cylinder','naca'])
def test_actual_hybrid_layers_and_conservative_faces(tmp_path,case):
    pytest.importorskip('gmsh');torch.set_num_threads(1)
    path=Path(__file__).resolve().parents[1]/'scripts/generate_external_gmsh.py'
    spec=importlib.util.spec_from_file_location('hybrid_generator',path)
    generator=importlib.util.module_from_spec(spec);spec.loader.exec_module(generator)
    meshfile=tmp_path/(case+'.msh')
    metadata=generator.generate(case,meshfile,2,hybrid=True)
    c=replace(configuration(case,64,24 if case=='cylinder' else 26,'cpu',2),nx=1,ny=1,mesh_type='gmsh-cylinder' if case=='cylinder' else 'gmsh-airfoil',mesh_file=str(meshfile))
    mesh=GmshExternalMesh(c);quality=hybrid_mesh_quality(mesh)
    assert quality['hybrid_gate_passed']
    assert quality['quad_layers']['min']>=3
    assert abs(quality['measured_layer_growth']['median']-1.18)<.01
    distance=quality['wall_cell_normal_distance_m']
    assert .45*metadata['first_layer_target_m']<distance['min']
    assert distance['max']<.6*metadata['first_layer_target_m']
    # Summing every oriented face of every physical cell verifies interface closure.
    closure=torch.zeros((mesh.volumes.numel(),2),dtype=torch.float64)
    closure.index_add_(0,mesh.owner,mesh.face_area_vectors)
    closure.index_add_(0,mesh.neighbor[mesh.interior],-mesh.face_area_vectors[mesh.interior])
    assert float(closure.abs().max())<1e-13
    if case=='cylinder':
        solver=ConservativeLinearUpwindCoupledSolver(c)
        _,defect=solver._coupled_jacobian()
        assert defect<2e-11
