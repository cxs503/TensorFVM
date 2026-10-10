"""Generate tagged, conforming external meshes using Gmsh's frontal Delaunay.

Gmsh official API/manual: https://gmsh.info/doc/texinfo/gmsh.html
No Cartesian obstacle rasterization or cut cells are used.
"""
import argparse
import json
import hashlib
from pathlib import Path
import numpy as np


def generate(case, output, scale=1., *, length=None, height=None, leading_x=6., leading_y=8., geometric_incidence_deg=0., open_boundaries=False, hybrid=False, outer_size_limit=None):
    import gmsh
    if not np.isfinite(scale) or scale<=0:
        raise ValueError('Mesh scale must be finite and positive')
    output=Path(output);output.parent.mkdir(parents=True,exist_ok=True)
    gmsh.initialize()
    try:
        gmsh.option.setNumber('General.Terminal',0)
        gmsh.option.setNumber('General.NumThreads',1)
        gmsh.option.setNumber('Mesh.RandomSeed',1)
        gmsh.model.add(case)
        geo=gmsh.model.geo
        L,H=(2.2,.41) if case=='cylinder' else (20,16)
        L=L if length is None else length
        H=H if height is None else height
        wall_size=(.003 if case=='cylinder' else .008)*scale
        outer_size=(.025 if case=='cylinder' else .4)*scale
        if outer_size_limit is not None:
            if not np.isfinite(outer_size_limit) or outer_size_limit<=0:raise ValueError('Outer size limit must be positive')
            outer_size=min(outer_size,outer_size_limit)
        corners=[geo.addPoint(x,y,0,outer_size) for x,y in [(0,0),(L,0),(L,H),(0,H)]]
        outer=[geo.addLine(corners[i],corners[(i+1)%4]) for i in range(4)]
        if case=='cylinder':
            center=geo.addPoint(.2,.2,0,wall_size)
            points=[geo.addPoint(.2+.05*np.cos(t),.2+.05*np.sin(t),0,wall_size) for t in np.arange(4)*np.pi/2]
            body=[geo.addCircleArc(points[i],center,points[(i+1)%4]) for i in range(4)]
        else:
            x=(1-np.cos(np.linspace(0,np.pi,161)))/2
            y=.6*(.2969*np.sqrt(x)-.126*x-.3516*x*x+.2843*x**3-.1036*x**4)
            angle=-np.radians(geometric_incidence_deg)
            def point(xx,yy):
                return geo.addPoint(leading_x+.25+(xx-.25)*np.cos(angle)-yy*np.sin(angle),leading_y+(xx-.25)*np.sin(angle)+yy*np.cos(angle),0,wall_size)
            upper=[point(xx,yy) for xx,yy in zip(x,y)]
            lower=[upper[0]]+[point(xx,-yy) for xx,yy in zip(x[1:-1],y[1:-1])]+[upper[-1]]
            body=[geo.addSpline(list(reversed(upper))),geo.addSpline(lower)]
        surface=geo.addPlaneSurface([geo.addCurveLoop(outer),geo.addCurveLoop(body)])
        geo.synchronize()
        for label,curves in [('inlet',[outer[3]]),('outlet',[outer[1]]),('wall' if case=='cylinder' else 'open-boundary' if open_boundaries else 'far-field',[outer[0],outer[2]]),('cylinder' if case=='cylinder' else 'airfoil',body)]:
            tag=gmsh.model.addPhysicalGroup(1,curves);gmsh.model.setPhysicalName(1,tag,label)
        tag=gmsh.model.addPhysicalGroup(2,[surface]);gmsh.model.setPhysicalName(2,tag,'fluid')
        distance=gmsh.model.mesh.field.add('Distance')
        gmsh.model.mesh.field.setNumbers(distance,'CurvesList',body)
        gmsh.model.mesh.field.setNumber(distance,'Sampling',400)
        threshold=gmsh.model.mesh.field.add('Threshold')
        for key,value in [('InField',distance),('SizeMin',wall_size),('SizeMax',outer_size),('DistMin',.025 if case=='cylinder' else .08),('DistMax',.15 if case=='cylinder' else 1.5)]:
            gmsh.model.mesh.field.setNumber(threshold,key,value)
        background = threshold
        if hybrid:
            # Ordered wall-normal layers, with a triangular outer region.
            layer = gmsh.model.mesh.field.add('BoundaryLayer')
            gmsh.model.mesh.field.setNumbers(layer, 'CurvesList', body)
            first_height = (.0002 if case=='cylinder' else .0005)*scale
            thickness = .008 if case=='cylinder' else .025
            for key, value in [('Size',first_height),('SizeFar',outer_size),
                               ('Ratio',1.18),('Thickness',thickness),('Quads',1)]:
                gmsh.model.mesh.field.setNumber(layer,key,value)
            if case=='naca':
                gmsh.model.mesh.field.setNumbers(layer,'FanPointsList',[upper[-1]])
                gmsh.model.mesh.field.setNumbers(layer,'FanPointsSizesList',[20])
                trailing = gmsh.model.mesh.field.add('Distance')
                gmsh.model.mesh.field.setNumbers(trailing,'PointsList',[upper[-1]])
                tip = gmsh.model.mesh.field.add('Threshold')
                for key,value in [('InField',trailing),('SizeMin',.001*scale),
                                  ('SizeMax',outer_size),('DistMin',.025),('DistMax',.25)]:
                    gmsh.model.mesh.field.setNumber(tip,key,value)
                te = gmsh.model.getValue(0,upper[-1],[])
                wake = gmsh.model.mesh.field.add('Box')
                for key,value in [('XMin',te[0]),('XMax',te[0]+3),
                                  ('YMin',te[1]-.08),('YMax',te[1]+.08),
                                  ('VIn',.025*scale),('VOut',outer_size),('Thickness',.15)]:
                    gmsh.model.mesh.field.setNumber(wake,key,value)
                background=gmsh.model.mesh.field.add('Min')
                gmsh.model.mesh.field.setNumbers(background,'FieldsList',[threshold,tip,wake])
            gmsh.model.mesh.field.setAsBoundaryLayer(layer)
        gmsh.model.mesh.field.setAsBackgroundMesh(background)
        gmsh.option.setNumber('Mesh.Algorithm',6)
        gmsh.option.setNumber('Mesh.MshFileVersion',2.2)
        gmsh.option.setNumber('Mesh.Binary',0)
        gmsh.option.setNumber('Mesh.MeshSizeExtendFromBoundary',0)
        gmsh.option.setNumber('Mesh.MeshSizeFromCurvature',0)
        gmsh.option.setNumber('Mesh.Smoothing',10)
        gmsh.model.mesh.generate(2)
        gmsh.write(str(output))
        metadata=dict(case=case,scale=scale,domain_m=[L,H],leading_edge_unrotated_m=[leading_x,leading_y] if case=='naca' else None,geometric_incidence_deg=geometric_incidence_deg,open_boundaries=open_boundaries,gmsh_version=gmsh.__version__,algorithm='Frontal-Delaunay (Gmsh algorithm 6)',wall_target_size_m=wall_size,outer_target_size_m=outer_size,geometry='Exact circle CAD arcs' if case=='cylinder' else 'Closed NACA0012 upper/lower interpolating CAD splines through 161 cosine-spaced points',mesh_sha256=hashlib.sha256(output.read_bytes()).hexdigest(),generator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),documentation='https://gmsh.info/doc/texinfo/gmsh.html')
        if hybrid:
            metadata.update(mesh_layout='quadrilateral wall layers / triangular outer region',first_layer_target_m=first_height,layer_growth_ratio=1.18,layer_thickness_target_m=thickness,trailing_edge_fan_elements=20 if case=='naca' else 0)
        output.with_suffix('.json').write_text(json.dumps(metadata,indent=2)+'\n')
        return metadata
    finally:
        gmsh.finalize()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case',choices=['cylinder','naca'],required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--scale',type=float,default=1.)
    parser.add_argument('--hybrid',action='store_true')
    args=parser.parse_args()
    print(json.dumps(generate(args.case,args.output,args.scale,hybrid=args.hybrid),indent=2))
