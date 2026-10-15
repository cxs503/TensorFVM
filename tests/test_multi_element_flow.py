"""Independent conservative reconstruction and current equation checks."""
import tempfile
import unittest
import numpy as np
from pathlib import Path
import torch
from tests.test_three_element import _TINY_GMSH
from tensorfvm.solver import SolverConfig
from tensorfvm.multi_element_flow import MultiElementFlowSolver
from tensorfvm.hlpw30p30n import DATA,layer_zones,per_element,profile_directory,reference_zones,resample_profile,write_mesh_geo


def _two_row_mesh():
    nodes=[(x+.12*y*(2-y),y) for y in range(3) for x in range(4)]
    edges=[(4+i,i+1,i+2) for i in range(3)]
    edges += [(3,9+i,10+i) for i in range(3)]
    edges += [(1,1,5),(1,5,9),(2,4,8),(2,8,12)]
    elements=[f'{j+1} 1 2 {tag} 1 {a} {b}' for j,(tag,a,b) in enumerate(edges)]
    for y in range(2):
        for x in range(3):
            a=y*4+x+1
            elements.append(f'{len(elements)+1} 3 2 7 1 {a} {a+1} {a+5} {a+4}')
    header=_TINY_GMSH.split('$Nodes')[0]
    return header+'$Nodes\n12\n'+'\n'.join(f'{i+1} {x} {y} 0' for i,(x,y) in enumerate(nodes))+'\n$EndNodes\n$Elements\n16\n'+'\n'.join(elements)+'\n$EndElements\n'


class MultiElementFlowTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        mesh=Path(self.temp.name)/'mesh.msh';mesh.write_text(_TINY_GMSH)
        self.solver=MultiElementFlowSolver(SolverConfig(nx=1,ny=1,length=3,height=1,
            cylinder_radius=None,mesh_type='three-element',mesh_file=str(mesh),
            reynolds=100,turbulence_model='spalart-allmaras',max_iterations=1))

    def test_limited_flux_is_bounded_and_conservative(self):
        s=self.solver;u=torch.tensor([[.2,.1],[.8,.3],[.4,-.2]],dtype=torch.float64)
        gradient=torch.full((3,2,2),20.,dtype=u.dtype)
        flux=torch.ones_like(s.mass_flux)
        correction=s.linear_upwind_correction(u,flux,gradient)
        up,_=s._upwind_geometry(flux);face=u[up]+correction[s.f]
        self.assertTrue(bool((face>=torch.tensor([0.,-.2])-1e-14).all()))
        self.assertTrue(bool((face<=torch.tensor([1.,.3])+1e-14).all()))
        torch.testing.assert_close(s._sum(flux[:,None]*correction).sum(0),torch.zeros(2,dtype=u.dtype),atol=1e-14,rtol=0)
        self.assertTrue(bool((correction[s.mesh.boundary]==0).all()))

    def test_taylor_reconstruction_affine_exact_when_unlimited(self):
        s=self.solver;s.bounded=False
        gradient=torch.tensor([[2.,-1.],[3.,4.]],dtype=torch.float64).expand(3,2,2)
        centers=s.mesh.centers.reshape(-1,2);u=centers@gradient[0]
        flux=torch.ones_like(s.mass_flux);up,_=s._upwind_geometry(flux)
        face=u[up]+s.linear_upwind_correction(u,flux,gradient)[s.f]
        torch.testing.assert_close(face,s.mesh.face_centers[s.f]@gradient[0])

    def test_pressure_operator_and_sa_preconditioner_have_true_checks(self):
        s=self.solver;s.step()
        pressure=[r for r in s.linear_history if r['kind']=='pressure']
        self.assertTrue(pressure)
        self.assertLess(pressure[-1]['assembled_operator_relative_defect'],2e-12)
        self.assertTrue(any(r['kind']=='spalart-allmaras' for r in s.linear_history))
        # Momentum factors are keyed per velocity component ('momentum0',
        # 'momentum1'); the SA solve reuses the 'momentum' kind in its own cache.
        momentum=[v for k,v in s._lu_cache.items() if k.startswith('momentum')]
        self.assertTrue(momentum);self.assertIn('momentum',s._sa_lu_cache)
        self.assertFalse(any(v is s._sa_lu_cache['momentum'] for v in momentum))
        self.assertTrue(all(r['true_residual']<=1.05*r['target'] for r in s.linear_history))

    def test_sa_uses_exactly_one_implicit_relaxation(self):
        from dataclasses import replace
        import numpy as np
        s=MultiElementFlowSolver(replace(self.solver.config,turbulence_relaxation=.4))
        s.sa_inner_iterations=1;old=s.nu_tilde.clone()
        d,a,b,rhs=s._sa_system(old);relaxed=d/.4
        expected=np.linalg.solve(s._matrix(relaxed,a,b).toarray(),(rhs+.6*relaxed*old).numpy())
        residual,_=s._turbulence_step()
        torch.testing.assert_close(s.nu_tilde,torch.from_numpy(expected).clamp_min(0),atol=2e-13,rtol=2e-12)
        d,a,b,rhs=s._sa_system(s.nu_tilde)
        actual=float((s._matvec(s.nu_tilde,d,a,b)-rhs).abs().max()/(s.config.density*s.config.inlet_velocity*s.config.reference_length*s.kinematic_viscosity))
        self.assertAlmostEqual(residual,actual,places=13)

    def test_stored_metrics_use_current_fields_and_local_flux_defect(self):
        s=self.solver;s.anderson_depth=3
        metrics=s.step()
        self.assertAlmostEqual(metrics['momentum'],max(metrics['u_residual'],metrics['v_residual']),places=13)
        d,_,_,_=s._momentum(s.velocity,s.p,s.mass_flux)
        physical,_=s._rhie_chow(s.velocity,s.p,s.volume/d)
        expected=float(((physical-s.mass_flux).abs()/(s.config.density*s.config.inlet_velocity*s.mesh.face_lengths)).max())
        self.assertAlmostEqual(metrics['rhie_chow_max_normal_velocity_defect'],expected,places=13)

    def test_coupled_frozen_reconstruction_operator(self):
        from tensorfvm.multi_element_coupled import CoupledMultiElementSolver
        s=CoupledMultiElementSolver(self.solver.config)
        matrix,defect=s.coupled_matrix()
        self.assertEqual(matrix.shape,(9,9))
        self.assertLess(defect,2e-11)

    def test_variable_viscosity_full_stress_response(self):
        s=self.solver
        s.turbulent_kinematic_viscosity=torch.tensor([.03,.07,.02],dtype=s.p.dtype)
        u=torch.tensor([[.2,.1],[.8,.3],[.4,-.2]],dtype=s.p.dtype)
        probe=torch.tensor([[.31,-.2],[-.17,.45],[.73,-.11]],dtype=s.p.dtype)
        blocks=s.stress_correction_matrices()
        expected=s.stress_correction(u+probe)-s.stress_correction(u)
        import numpy as np
        actual=np.stack([sum(blocks[j][k]@probe[:,k].numpy() for k in (0,1)) for j in (0,1)],axis=1)
        torch.testing.assert_close(torch.from_numpy(actual),expected,atol=1e-13,rtol=1e-12)

    def test_skew_pressure_face_force_and_assembled_gradient(self):
        from dataclasses import replace
        from tensorfvm.multi_element_skew_pressure import SkewPressureMultiElementSolver
        mesh=Path(self.temp.name)/'two-row.msh';mesh.write_text(_two_row_mesh())
        s=SkewPressureMultiElementSolver(replace(self.solver.config,mesh_file=str(mesh),height=2))
        field=torch.sin(torch.arange(s.count,dtype=s.p.dtype)*.37)
        H,I,G=s._geometry()
        gradient=s._gradient(field,pressure=True)
        import numpy as np
        actual=np.stack([g@field.numpy() for g in G],axis=1)
        torch.testing.assert_close(torch.from_numpy(actual),gradient,atol=2e-14,rtol=1e-13)
        face=s.reconstructed_pressure_faces(field)
        torch.testing.assert_close((gradient*s.volume[:,None]).sum(0),(face[s.mesh.boundary,None]*s.S[s.mesh.boundary]).sum(0),atol=2e-14,rtol=1e-13)
        s.step()
        pressure=[r for r in s.linear_history if r['kind']=='pressure']
        self.assertLess(pressure[-1]['assembled_operator_relative_defect'],2e-12)

    def test_full_momentum_response_uses_nonconstant_reconstruction(self):
        s=self.solver
        s.velocity.copy_(torch.tensor([[.2,.1],[.8,.3],[.4,-.2]],dtype=torch.float64))
        matrices,defect=s.momentum_correction_matrices(s.velocity,s.mass_flux)
        self.assertEqual(len(matrices),2);self.assertLess(defect,2e-11)
        s.step()
        momentum=[r for r in s.linear_history if r['kind']=='momentum']
        self.assertEqual(len(momentum),2)
        self.assertTrue(all(r['momentum_assembled_operator_relative_defect']<2e-11 for r in momentum))

    def test_exact_sa_newton_jacobian(self):
        from tensorfvm.sa_newton import NewtonSaMixin
        class Solver(NewtonSaMixin,MultiElementFlowSolver):pass
        s=Solver(self.solver.config)
        matrix,defect=s.sa_jacobian(s.nu_tilde)
        self.assertEqual(matrix.shape,(3,3));self.assertLess(defect,2e-10)
        for values in [[0.,0.,0.],[100.,20.,0.]]:
            _,boundary_defect=s.sa_jacobian(torch.tensor(values,dtype=torch.float64)*s.kinematic_viscosity)
            self.assertLess(boundary_defect,2e-10)
        s._turbulence_step()
        self.assertLessEqual(s.sa_newton_history[-1]['nonlinear_l2_after'],s.sa_newton_history[-1]['nonlinear_l2_before'])

    def test_primary_reference_requires_supported_conditions(self):
        zones=reference_zones(8.1);self.assertEqual(set(zones),{'slat','main','flap'})
        self.assertGreater(len(zones['main']),40)
        with self.assertRaises(ValueError):reference_zones(0.)
        geo=write_mesh_geo(Path(self.temp.name)/'official.geo').read_text()
        self.assertEqual(geo.count('Circle('),4)
        self.assertIn('500',geo)
        self.assertIn('Physical Curve("far-field")',geo)

    def test_spline_resampling_smooths_facets_but_keeps_corners_and_vertices(self):
        def turns(points):
            closed=points[:-1];edges=np.diff(np.vstack([closed,closed[:1]]),axis=0)
            t=edges/np.linalg.norm(edges,axis=1)[:,None]
            cross=t[:-1,0]*t[1:,1]-t[:-1,1]*t[1:,0];dot=(t[:-1]*t[1:]).sum(1)
            return np.degrees(np.arctan2(cross,dot))
        for name,corners in [('slat',2),('main',3),('flap',1)]:
            original=np.loadtxt(DATA/(name+'.dat'));resampled=resample_profile(original,max_turn=1.5)
            self.assertTrue(np.allclose(resampled[0],resampled[-1]))
            smooth=np.abs(turns(resampled));self.assertEqual(int((smooth>30).sum()),corners)
            self.assertLess(smooth[smooth<=30].max(),1.5+0.3)  # discrete spline turn, small tolerance
            self.assertGreater(np.abs(turns(original))[np.abs(turns(original))<=30].max(),10.)
            # every official vertex is kept exactly and the signed area (orientation) is unchanged
            from scipy.spatial import cKDTree
            self.assertLess(cKDTree(resampled).query(original)[0].max(),1e-12)
            area=lambda q:0.5*np.sum(q[:-1,0]*q[1:,1]-q[1:,0]*q[:-1,1])
            self.assertLess(abs(area(resampled)-area(original)),1e-5)  # LE sagitta of the 12-degree facets
        geo=write_mesh_geo(Path(self.temp.name)/'resampled.geo',max_turn={'slat':1.5,'main':1.5,'flap':2.}).read_text()
        self.assertGreater(geo.count('Point('),645+209+205+16)
        self.assertTrue((Path(self.temp.name)/'profiles'/'main.dat').exists())

    def test_zoned_quad_layers_keep_gaps_and_fans_thin(self):
        out=Path(self.temp.name)/'zoned'
        geo=write_mesh_geo(out/'mesh.geo',thick_layer=.04,max_turn={'slat':1.5,'main':1.5,'flap':2.}).read_text()
        self.assertEqual(geo.count('BoundaryLayer;'),2);self.assertIn('Field[7].thickness = 0.04;',geo)
        self.assertIn('Field[3].thickness = 0.0005;',geo)
        import re
        thin=set(map(int,re.search(r'Field\[3\]\.CurvesList = \{([^}]*)\}',geo).group(1).split(',')))
        thick=set(map(int,re.search(r'Field\[7\]\.CurvesList = \{([^}]*)\}',geo).group(1).split(',')))
        fans=set(map(int,re.search(r'Field\[3\]\.FanPointsList = \{([^}]*)\}',geo).group(1).split(',')))
        zones=layer_zones(out/'profiles',.0005,.04)
        self.assertEqual(thin|thick,set(zones));self.assertFalse(thin&thick)
        self.assertEqual(thick,{k for k,v in zones.items() if v==.04})
        # the facets on both sides of every fan corner stay in the thin field
        n_slat=len(np.loadtxt(out/'profiles'/'slat.dat'))-1
        for f in fans:self.assertIn(f,thin)
        # geometry of the zones: slat entirely thin, main thick facets only on the upper surface
        # clear of the slat/main and main/flap gaps, flap thick facets downstream of the main TE
        main=np.loadtxt(out/'profiles'/'main.dat')[:-1];flap=np.loadtxt(out/'profiles'/'flap.dat')[:-1]
        self.assertTrue(all(k>n_slat for k in thick))
        mids={i+n_slat+1:(main[i]+main[(i+1)%len(main)])/2 for i in range(len(main))}
        mids.update({i+n_slat+len(main)+1:(flap[i]+flap[(i+1)%len(flap)])/2 for i in range(len(flap))})
        for k in thick:
            x,y=mids[k]
            if k<=n_slat+len(main):self.assertTrue(.06<x<.855 and y>-.01,(k,x,y))
            else:self.assertTrue(.905<x<1.115,(k,x,y))
        self.assertGreater(len(thick),300);self.assertGreater(len(thin),300)
        with self.assertRaises(ValueError):write_mesh_geo(out/'bad.geo',thick_layer=.0004)

    def test_facet_length_cap_subdivides_long_flap_facets_only_where_asked(self):
        official=np.loadtxt(DATA/'flap.dat')
        base=resample_profile(official,max_turn=2.);capped=resample_profile(official,max_turn=2.,max_length=.0015)
        seg=lambda q:np.linalg.norm(np.diff(q,axis=0),axis=1)
        self.assertGreater(seg(base).max(),.003);self.assertLessEqual(seg(capped).max(),.0015+1e-12)
        self.assertGreater(len(capped),len(base))
        from scipy.spatial import cKDTree
        self.assertLess(cKDTree(capped).query(official)[0].max(),1e-12)  # official vertices kept
        edges=base[1:]-base[:-1];d=capped[:,None]-base[:-1]
        t=np.clip(np.einsum('nki,ki->nk',d,edges)/np.einsum('ki,ki->k',edges,edges),0,1)
        self.assertLess(np.linalg.norm(d-t[:,:,None]*edges,axis=-1).min(1).max(),2e-5)  # same spline (sagitta), finer sampling
        out=Path(self.temp.name)/'capped'
        profiles=profile_directory({'slat':1.5,'main':1.5,'flap':2.},out,max_length={'slat':None,'main':None,'flap':.0015})
        self.assertEqual(len(np.loadtxt(profiles/'flap.dat')),len(capped))
        self.assertEqual(len(np.loadtxt(profiles/'main.dat')),len(resample_profile(np.loadtxt(DATA/'main.dat'),max_turn=1.5)))
        self.assertEqual(per_element({'slat':1,'main':2,'flap':3},'flap'),3);self.assertEqual(per_element(.5,'slat'),.5)
        with self.assertRaises(ValueError):per_element({'slat':1},'slat')
        with self.assertRaises(ValueError):profile_directory(None,out,max_length=.001)

if __name__=='__main__':unittest.main()
