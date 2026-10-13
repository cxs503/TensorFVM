"""Internal-cell least-squares pressure trace, independent of wall pressure data.

This postprocessor does not change momentum/traction or qualify physical Cp.
Connected fluid-cell stencils avoid fitting across a solid body. Affine
consistency is necessary; curved-flow accuracy still requires refinement.
"""
import numpy as np


class NativeBoundaryPressureTrace3D:
    def __init__(self,mesh,boundary_mask,*,max_rings=3,minimum_singular_ratio=1e-10):
        mask=np.asarray(boundary_mask,bool)
        if mask.shape!=(len(mesh.owner),) or np.any(mask & (mesh.neighbor>=0)) or not mask.any():
            raise ValueError('nonempty boundary-only face mask required')
        if not isinstance(max_rings,int) or max_rings<1 or not 0<minimum_singular_ratio<1:
            raise ValueError('positive ring count and singular ratio in (0,1) required')
        self.mesh=mesh;self.face_indices=np.flatnonzero(mask);self.stencils=[]
        adjacency=[set() for _ in mesh.cells]
        for owner,neighbor in zip(mesh.owner,mesh.neighbor):
            if neighbor>=0:adjacency[owner].add(int(neighbor));adjacency[neighbor].add(int(owner))
        for owner in np.unique(mesh.owner[mask]):
            visited={int(owner)};frontier={int(owner)}
            for ring in range(1,max_rings+1):
                frontier=set().union(*(adjacency[i] for i in frontier))-visited
                visited.update(frontier);neighbors=np.array(sorted(visited-{int(owner)}),dtype=int)
                if len(neighbors)<3:continue
                dx=mesh.cell_centers[neighbors]-mesh.cell_centers[owner]
                distances=np.linalg.norm(dx,axis=1)
                if np.any(distances<=0):raise ValueError('coincident cell centers in pressure stencil')
                u,s,vt=np.linalg.svd(dx/distances[:,None],full_matrices=False)
                ratio=float(s[-1]/s[0])
                if ratio<=minimum_singular_ratio:continue
                inverse=(vt.T/s)@u.T/distances[None,:]
                positions=np.flatnonzero(mesh.owner[self.face_indices]==owner)
                offsets=mesh.face_centers[self.face_indices[positions]]-mesh.cell_centers[owner]
                self.stencils.append((int(owner),neighbors,inverse,positions,offsets,ring,ratio))
                break
            else:raise ValueError(f'rank-deficient connected pressure stencil at cell {owner}')

    def reconstruct(self,cell_pressure):
        p=np.asarray(cell_pressure,float)
        if p.shape!=(len(self.mesh.cells),) or not np.isfinite(p).all():
            raise ValueError('one finite pressure per cell required')
        values=np.empty(len(self.face_indices));gradients=[]
        for owner,neighbors,inverse,positions,offsets,_,_ in self.stencils:
            gradient=inverse@(p[neighbors]-p[owner]);gradients.append(gradient)
            values[positions]=p[owner]+offsets@gradient
        return dict(face_indices=self.face_indices.copy(),pressure=values,
            owner_cells=np.array([s[0] for s in self.stencils]),gradient=np.asarray(gradients),
            stencil_rings=np.array([s[5] for s in self.stencils]),
            singular_ratio=np.array([s[6] for s in self.stencils]),
            wall_Cp_certified=False,changes_momentum_equations=False)
