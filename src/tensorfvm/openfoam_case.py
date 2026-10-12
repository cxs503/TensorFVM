"""Export native shared faces and an explicit OpenFOAM Foundation 12 RANS case.

This is a diagnostic interoperability baseline, not a validated SUBOFF recipe.
Native cells are retained; face order changes to OpenFOAM's internal-first order.
"""
from pathlib import Path
import re
import numpy as np


def header(name, kind='dictionary', location=None):
    where = f'location "{location}"; ' if location else ''
    return f'FoamFile {{ version 2.0; format ascii; class {kind}; {where}object {name}; }}\n'


def export_native_mesh(mesh, case):
    case = Path(case); folder = case/'constant/polyMesh'
    folder.mkdir(parents=True, exist_ok=False)
    internal = np.flatnonzero(mesh.neighbor >= 0)
    internal = internal[np.lexsort((mesh.neighbor[internal], mesh.owner[internal]))]
    labels = sorted(set(str(s) for s in mesh.patch_labels[mesh.neighbor < 0]))
    if any(not re.fullmatch(r'[A-Za-z_][A-Za-z0-9_]*', label) for label in labels):
        raise ValueError('OpenFOAM patch labels must be simple identifiers')
    boundary = [np.flatnonzero((mesh.neighbor < 0)&(mesh.patch_labels == label)) for label in labels]
    order = np.concatenate([internal]+boundary)
    def write(name, kind, entries):
        (folder/name).write_text(header(name,kind,'constant/polyMesh')+str(len(entries))+'\n(\n'+'\n'.join(entries)+'\n)\n')
    write('points','vectorField',['('+ ' '.join(f'{x:.17g}' for x in p)+')' for p in mesh.vertices])
    faces = []
    for face in order:
        ids = list(mesh.face_vertices[face]); points = mesh.vertices[ids]
        area = .5*np.sum(np.cross(points,np.roll(points,-1,axis=0)),axis=0)
        if np.dot(area,mesh.area_vectors[face]) < 0:
            ids.reverse()
        faces.append(f'{len(ids)}('+ ' '.join(map(str,ids))+')')
    write('faces','faceList',faces)
    write('owner','labelList',[str(int(x)) for x in mesh.owner[order]])
    write('neighbour','labelList',[str(int(x)) for x in mesh.neighbor[internal]])
    patches = []; start = len(internal)
    for label, ids in zip(labels,boundary):
        kind = 'wall' if label == 'wall' else 'patch'
        patches.append(f'{label}\n{{ type {kind}; nFaces {len(ids)}; startFace {start}; }}')
        start += len(ids)
    write('boundary','polyBoundaryMesh',patches)
    np.save(case/'native-face-order.npy',order)
    return labels


def write_rans_case(mesh, case, *, reynolds=12e6, iterations=300,
                    convection='linearUpwind', nonorthogonal_correctors=2,
                    patch_labels=('farfield', 'inlet', 'outlet', 'wall')):
    """Write dimensionless AFF-1 dictionaries; ``mesh=None`` prepares meshing.

    Inlet turbulence is an explicit project assumption. The force reference
    area is the AFF-1 frontal area divided by hull length squared.
    """
    case = Path(case)
    if case.exists() and any(case.iterdir()):
        raise ValueError('case directory must be empty')
    if not np.isfinite(reynolds) or reynolds <= 0 or iterations < 1:
        raise ValueError('positive Reynolds number and iteration count required')
    if convection not in ('upwind','linearUpwind'):
        raise ValueError('supported convection: upwind or linearUpwind')
    labels = (sorted(set(map(str, mesh.patch_labels[mesh.neighbor < 0])))
              if mesh is not None else list(patch_labels))
    if not {'wall','inlet','outlet'} <= set(labels):
        raise ValueError('external RANS case requires wall, inlet and outlet patches')
    if any(not re.fullmatch(r'[A-Za-z_][A-Za-z0-9_]*', label) for label in labels):
        raise ValueError('OpenFOAM patch labels must be simple identifiers')
    if mesh is not None:
        export_native_mesh(mesh,case)
    else:
        (case/'constant').mkdir(parents=True, exist_ok=True)
    (case/'0').mkdir(); (case/'system').mkdir()
    k = 1.5*.005**2; omega = np.sqrt(k)/(.09**.25*.05)
    fields = {'U': ('volVectorField','0 1 -1 0 0 0 0','(1 0 0)'),
              'p': ('volScalarField','0 2 -2 0 0 0 0','0'),
              'k': ('volScalarField','0 2 -2 0 0 0 0',f'{k:.17g}'),
              'omega': ('volScalarField','0 0 -1 0 0 0 0',f'{omega:.17g}'),
              'nut': ('volScalarField','0 2 -1 0 0 0 0','0')}
    for name,(kind,dimension,value) in fields.items():
        patches = []
        for label in labels:
            if name == 'U':
                entry = 'type noSlip;' if label == 'wall' else (
                    f'type inletOutlet; inletValue uniform {value}; value uniform {value};' if label == 'outlet'
                    else f'type fixedValue; value uniform {value};')
            elif name == 'p':
                entry = 'type fixedValue; value uniform 0;' if label == 'outlet' else 'type zeroGradient;'
            elif name == 'nut':
                entry = f'type {"nutUSpaldingWallFunction" if label == "wall" else "calculated"}; value uniform 0;'
            elif label == 'wall':
                entry = f'type {"omegaWallFunction" if name == "omega" else "kqRWallFunction"}; value uniform {value};'
                if name == 'omega': entry += ' blended true;'
            elif label == 'outlet':
                entry = f'type inletOutlet; inletValue uniform {value}; value uniform {value};'
            else:
                entry = f'type fixedValue; value uniform {value};'
            patches.append(label+' { '+entry+' }')
        (case/'0'/name).write_text(header(name,kind,'0')+f'dimensions [{dimension}];\ninternalField uniform {value};\nboundaryField\n{{\n'+'\n'.join(patches)+'\n}\n')
    (case/'constant/physicalProperties').write_text(header('physicalProperties')+f'viscosityModel constant;\nnu [0 2 -1 0 0 0 0] {1/reynolds:.17g};\n')
    (case/'constant/momentumTransport').write_text(header('momentumTransport')+'simulationType RAS;\nRAS { model kOmegaSST; turbulence on; printCoeffs on; }\n')
    velocity_scheme = 'bounded Gauss upwind' if convection == 'upwind' else 'bounded Gauss linearUpwindV grad(U)'
    (case/'system/fvSchemes').write_text(header('fvSchemes')+f'''
ddtSchemes {{ default steadyState; }}
gradSchemes {{ default Gauss linear; grad(U) cellLimited Gauss linear 1; }}
divSchemes
{{
    default none;
    div(phi,U) {velocity_scheme};
    div(phi,k) bounded Gauss upwind;
    div(phi,omega) bounded Gauss upwind;
    div((nuEff*dev2(T(grad(U))))) Gauss linear;
}}
laplacianSchemes {{ default Gauss linear limited 0.5; }}
interpolationSchemes {{ default linear; }}
snGradSchemes {{ default limited 0.5; }}
wallDist {{ method meshWave; }}
''')
    (case/'system/fvSolution').write_text(header('fvSolution')+f'''
solvers
{{
    p {{ solver GAMG; smoother GaussSeidel; tolerance 1e-9; relTol 0.01; }}
    "(U|k|omega)" {{ solver smoothSolver; smoother symGaussSeidel; tolerance 1e-9; relTol 0.05; }}
}}
SIMPLE
{{
    nNonOrthogonalCorrectors {nonorthogonal_correctors};
    consistent yes;
}}
relaxationFactors
{{
    fields {{ p 0.3; }}
    equations {{ U 0.5; k 0.5; omega 0.5; }}
}}
''')
    area = np.pi*(.508/4.356)**2/4
    (case/'system/controlDict').write_text(header('controlDict')+f'''
application foamRun;
solver incompressibleFluid;
startFrom startTime; startTime 0;
stopAt endTime; endTime {iterations}; deltaT 1;
writeControl timeStep; writeInterval {iterations};
purgeWrite 0; writeFormat ascii; writePrecision 12;
writeCompression off; timeFormat general; timePrecision 10;
runTimeModifiable false;
functions
{{
    forces
    {{
        type forces; libs ("libforces.so");
        patches (wall); rho rhoInf; rhoInf 1; CofR (0 0 0);
        writeControl timeStep; writeInterval 1; log false;
    }}
    forceCoeffs
    {{
        type forceCoeffs; libs ("libforces.so");
        patches (wall); rho rhoInf; rhoInf 1; CofR (0 0 0);
        dragDir (1 0 0); liftDir (0 0 1); pitchAxis (0 1 0);
        magUInf 1; lRef 1; Aref {area:.17g};
        writeControl timeStep; writeInterval 1; log false;
    }}
    yPlus
    {{
        type yPlus; libs ("libfieldFunctionObjects.so");
        writeControl timeStep; writeInterval {iterations};
        log false;
    }}
}}
''')
