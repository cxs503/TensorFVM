"""Run nonmatching planar interface flux verification, not a coupled flow solve."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import numpy as np
from tensorfvm.overset_flux import PlanarFluxTransfer
from tensorfvm.verification.core import Metric, evaluate


def patches(n, shift):
    nodes = np.linspace(0, 1, n+1)
    if shift:
        nodes[1:-1] += .17*np.sin(np.pi*nodes[1:-1])/n
    return np.array([[[a,c],[b,d]] for a,b in zip(nodes[:-1],nodes[1:]) for c,d in zip(nodes[:-1],nodes[1:])])


def exact_integrals(boxes):
    a,b = boxes[:,0], boxes[:,1]
    area = np.prod(b-a,axis=1)
    return area + .2*(np.cos(np.pi*a[:,0])-np.cos(np.pi*b[:,0]))/np.pi*(np.sin(np.pi*b[:,1])-np.sin(np.pi*a[:,1]))/np.pi


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()): parser.error('Output must be empty')
    args.output.mkdir(parents=True,exist_ok=True)
    rows=[]
    for n in (4,8,16):
        op=PlanarFluxTransfer.build(patches(n,False),patches(n+1,True))
        q=exact_integrals(op.donor_bounds); mapped=op.transfer(q)
        truth=-exact_integrals(op.receiver_bounds)
        error=float(np.sqrt(np.sum((mapped-truth)**2/op.receiver_measure)/np.sum(truth**2/op.receiver_measure)))
        defect=float(abs(op.balance(q,mapped)))
        metrics,passed=evaluate([Metric('flux_density_relative_l2',error,.03,'Actual receiver integrals against independent analytic flux'), Metric('oriented_integral_balance',defect,1e-12,'Opposing normal total flux balance')])
        rows.append(dict(n=n,donors=len(q),receivers=len(mapped),relative_l2_flux_density_error=error,integrated_flux_balance_absolute=defect,metrics=metrics,passed=passed))
        np.savez_compressed(args.output/f'level-{n}.npz',donor_bounds=op.donor_bounds,receiver_bounds=op.receiver_bounds,donor_flux=q,receiver_flux=mapped,reference_receiver_flux=truth,overlap_data=op.overlap.data,overlap_indices=op.overlap.indices,overlap_indptr=op.overlap.indptr)
    root=Path(__file__).resolve().parents[1]
    for name in ('scripts/verify_overset_flux.py','src/tensorfvm/overset_flux.py'):
        target=args.output/'source-snapshot'/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(root/name,target)
    summary=dict(case='nonmatching-planar-integrated-flux',normal_sign=-1,rows=rows,passed=all(r['passed'] for r in rows) and all(rows[i+1]['relative_l2_flux_density_error']<rows[i]['relative_l2_flux_density_error'] for i in range(2)),flow_solved=False,overset_navier_stokes_qualified=False,lbm_coupling_executed=False)
    (args.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    text='# Nonmatching Cartesian interface flux verification\n\nA common planar interface is partitioned independently on two sides. The normal orientations oppose each other. The prescribed flux density is 1 + 0.2 sin(pi y) cos(pi z); donor integrals are analytic. Transfer uses actual rectangle intersection areas and uniform donor-face density. Receiver reference integrals are evaluated independently. This is interface verification, not a solved overset flow.\n\n|Donor resolution|Donor faces|Receiver faces|Density relative L2|Oriented balance defect|Passed|\n|---|---:|---:|---:|---:|---|\n'
    for r in rows: text+=f"|{r['n']}|{r['donors']}|{r['receivers']}|{100*r['relative_l2_flux_density_error']:.6f}%|{r['integrated_flux_balance_absolute']:.3g}|{r['passed']}|\n"
    text+='\nThe balance test alone does not establish local flow accuracy. A shared surface must be explicitly defined in the same physical coordinate frame. Flux from moving surfaces additionally needs grid velocity and geometric conservation. No pressure correction, time advancement, populations, or fluid solver is executed here.\n\nReproduce: `PYTHONPATH=src python scripts/verify_overset_flux.py --output EMPTY_DIRECTORY`. Raw patches, fluxes, sparse intersections and producer sources are included.\n'
    (args.output/'report.md').write_text(text)
    manifest={str(p.relative_to(args.output)):hashlib.sha256(p.read_bytes()).hexdigest() for p in args.output.rglob('*') if p.is_file()}
    (args.output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(summary,indent=2))
    return 0 if summary['passed'] else 2


if __name__=='__main__': raise SystemExit(main())
