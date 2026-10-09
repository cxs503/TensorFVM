"""Isolate the upstream pressure channel algorithm with explicit float64 storage.

Only AST edits: torch.float32 -> float64; torch.zeros gets dtype=float64;
result includes the actual monitor history and full precision final fields.
No changes are written into TensorLBM, and its old evidence is preserved.
"""
import argparse
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import torch
from compare_lbm_channel_reference import map_case


class Float64AndEvidence(ast.NodeTransformer):
    def visit_Attribute(self,node):
        self.generic_visit(node)
        if isinstance(node.value,ast.Name) and node.value.id=='torch' and node.attr=='float32':
            node.attr='float64'
        return node

    def visit_Call(self,node):
        self.generic_visit(node)
        if (isinstance(node.func,ast.Attribute) and isinstance(node.func.value,ast.Name)
            and node.func.value.id=='torch' and node.func.attr=='zeros'
            and not any(k.arg=='dtype' for k in node.keywords)):
            node.keywords.append(ast.keyword(arg='dtype',value=ast.Attribute(value=ast.Name(id='torch',ctx=ast.Load()),attr='float64',ctx=ast.Load())))
        return node

    def visit_Assign(self,node):
        self.generic_visit(node)
        if any(isinstance(t,ast.Name) and t.id=='result' for t in node.targets):
            return [node]+ast.parse('''
result['storage_dtype'] = str(f.dtype)
result['monitor_interval_steps'] = 200
result['umax_history'] = umax_hist
result['last_monitor_relative_drift'] = (max(umax_hist[-10:])-min(umax_hist[-10:])) / max(abs(sum(umax_hist[-10:])/len(umax_hist[-10:])),1e-12)
result['u_profile'] = [float(v) for v in u_num]
result['rho_profile_x'] = rho_x.tolist()
result['final_rho'] = rho.cpu().tolist()
result['final_ux'] = ux.cpu().tolist()
result['final_uy'] = macroscopic(f)[2].cpu().tolist()
''').body
        return node


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--lbm-repo',type=Path,required=True)
    parser.add_argument('--heights',type=int,nargs='+',default=[24,48])
    parser.add_argument('--max-steps',type=int,default=120000)
    parser.add_argument('--output',type=Path,default=Path('docs/suite-channel/lbm-float64-diagnostic'))
    args=parser.parse_args()
    repo=args.lbm_repo.resolve();source=repo/'benchmarks/verified/poiseuille_2d/run.py'
    sys.path.insert(0,str(repo/'src'))
    spec=importlib.util.spec_from_file_location('lbm_channel_float64',source)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    tree=ast.parse(source.read_text());function=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='run_case')
    modified=Float64AndEvidence().visit(ast.Module(body=[function],type_ignores=[]));ast.fix_missing_locations(modified)
    namespace=dict(module.__dict__);exec(compile(modified,str(source)+' [float64 diagnostic]','exec'),namespace)
    torch.set_num_threads(1);args.output.mkdir(parents=True,exist_ok=True)
    cases=[];artifacts={}
    for height in args.heights:
        path=args.output/f'H{height}-raw.json'
        raw=namespace['run_case'](height,.8,.15/height,'bgk',4000,args.max_steps,str(path),compile_mode=None)
        case=map_case(raw);case['monitor_relative_drift']=raw['last_monitor_relative_drift'];cases.append(case)
        artifacts[path.name]=hashlib.sha256(path.read_bytes()).hexdigest()
        print(json.dumps({k:v for k,v in case.items() if not k.startswith('profile')}),flush=True)
    report={'schema':'tensor-suite.channel-float64-diagnostic/1','cases':cases,'passed':all(c['passed'] for c in cases),
            'scope':'isolated dtype diagnostic of LBM pressure channel, NOT same-boundary FVM comparison or FSI',
            'upstream_source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
            'transformed_function_sha256':hashlib.sha256(ast.unparse(modified).encode()).hexdigest(),
            'diagnostic_script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'lbm_git_base':subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip(),
            'changes':['float32 density ramp and implicit zeros replaced with explicit float64','actual maximum velocity monitor history exported','full precision averaged profile and final rho/ux/uy fields exported'],
            'unchanged':['pressure boundary','L/H=3','collision BGK tau=.8','initial rest','Re_H=1','min_steps=4000','every200 steps ten sample relative drift<1e-5'],
            'thresholds':{'velocity_l2_relative':.03,'pressure_gradient_relative':.03,'monitor_relative_drift':1e-5},
            'artifact_sha256':artifacts}
    (args.output/'comparison.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    return 0 if report['passed'] else 2


if __name__=='__main__':
    raise SystemExit(main())
