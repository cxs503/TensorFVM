#!/usr/bin/env python3
"""Archive verified fixed-viscosity momentum replays; never qualify full RANS."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reference',type=Path,required=True)
    p.add_argument('--runs',type=Path,nargs='+',required=True)
    p.add_argument('--seed',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists() and any(a.output.iterdir()):
        p.error('output must be empty')
    seed=read(a.seed/'summary.json')
    if digest(a.seed/'fields.npz')!=seed['fields_sha256']:
        raise ValueError('seed fields changed')
    mesh=np.load(a.seed/'fields.npz',allow_pickle=False)
    reference=read(a.reference/'summary.json')
    if read(a.reference/'settings.json')['algorithm']!='newton' or not reference['numerically_converged']:
        raise ValueError('converged native Newton reference required')
    if len({p.name for p in [a.reference,*a.runs]})!=1+len(a.runs):
        raise ValueError('distinct archive names required')
    target=np.load(a.reference/'fields.npz',allow_pickle=False)
    records=[]
    for path in [a.reference,*a.runs]:
        s=read(path/'summary.json');cfg=read(path/'settings.json')
        if not s['source_integrity_verified'] or s['full_rans_converged'] or s['physical_suboff_benchmark_passed']:
            raise ValueError('verified fixed-viscosity scope required')
        for name,sha in read(path/'sources.json').items():
            if digest(path/'source-snapshot'/name)!=sha:
                raise ValueError('source snapshot changed')
        for name in ('fields','inputs'):
            if digest(path/(name+'.npz'))!=s[name+'_sha256']:
                raise ValueError(name+' changed')
        for name,sha in cfg['input_bindings'].items():
            if digest(a.seed/name)!=sha:
                raise ValueError('seed input binding changed')
        if s['inputs_sha256']!=reference['inputs_sha256']:
            raise ValueError('momentum inputs must be identical')
        f=np.load(path/'fields.npz',allow_pickle=False)
        if not all(np.isfinite(f[key]).all() for key in f.files):
            raise ValueError('nonfinite raw result')
        flux=np.sum(f['face_velocity']*mesh['area_vectors'],axis=1)
        np.testing.assert_allclose(flux,f['flux'],rtol=0,atol=1e-15)
        mass=np.zeros(len(mesh['cell_volumes']));internal=mesh['neighbor']>=0
        np.add.at(mass,mesh['owner'],flux)
        np.add.at(mass,mesh['neighbor'][internal],-flux[internal])
        momentum=np.zeros((len(mass),3))
        np.add.at(momentum,mesh['owner'],f['momentum_flux_integrated'])
        np.add.at(momentum,mesh['neighbor'][internal],-f['momentum_flux_integrated'][internal])
        boundary_force_balance=f['momentum_flux_integrated'][~internal].sum(0)
        cd=float(-2*f['total_traction_integrated'][mesh['patch_labels']=='wall',0].sum())
        np.testing.assert_allclose(cd,s['Cd_L2'],rtol=0,atol=1e-14)
        np.testing.assert_allclose(abs(momentum.sum(0)).max(),s['global_momentum_balance_max'],rtol=0,atol=1e-12)
        differences={key:dict(max_abs=float(abs(f[key]-target[key]).max()),
            relative_l2=float(np.linalg.norm(f[key]-target[key])/max(np.linalg.norm(target[key]),1e-30)))
            for key in ('velocity','face_velocity','pressure','total_traction_integrated')}
        h=read(path/'history.json')
        records.append(dict(name=path.name,source=str(path),settings=cfg,summary=s,
            outer_iterations=len(h),linear_corrections=sum(r.get('inner_iterations',1) for r in h),
            replay_divergence_max=float(abs(mass/mesh['cell_volumes']).max()),
            replay_boundary_force_balance_max=float(abs(boundary_force_balance).max()),
            replay_local_momentum_balance_max=float(abs(momentum).max()),
            difference_from_native_newton=differences,history=h))
    a.output.mkdir(parents=True,exist_ok=True)
    for row in records:
        shutil.copytree(row['source'],a.output/row['name'])
    shutil.copy2(__file__,a.output/'reporter.py')
    result=dict(scope=__doc__,seed=str(a.seed),seed_fields_sha256=seed['fields_sha256'],
        identical_inputs_sha256=reference['inputs_sha256'],reference=str(a.reference),runs=records,
        full_rans_converged=False,physical_suboff_benchmark_passed=False)
    (a.output/'comparison.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    fig,ax=plt.subplots(figsize=(8,4.5))
    for row in records:
        h=row['history'];label=row['settings']['algorithm']
        if label!='newton':
            label+=f" / {row['settings'].get('relaxation_basis','absolute_row_sum')} / inner {row['settings'].get('inner_iterations',1)}"
        ax.semilogy([x['iteration'] for x in h],[x['current_momentum_residual'] for x in h],'.-',label=label)
    ax.axhline(1e-9,color='k',linestyle=':',label='current residual target')
    ax.set(xlabel='Nonlinear outer iteration (different work per iteration)',ylabel='Current momentum residual',
           title='Actual SUBOFF Re=10, fixed saved SST viscosity')
    ax.grid(True,alpha=.25);ax.legend(fontsize=8);fig.tight_layout()
    for suffix in ('png','pdf'):fig.savefig(a.output/('momentum-residual.'+suffix),dpi=180)
    plt.close(fig)
    lines=['# SUBOFF 固定 SST 涡黏性的动量子问题对照','',
        '本页验证原生动量求解器。固定同一保存状态的黏性系数，不推进 k、ω，不能作为完整 RANS 或实验精度达标证据。',
        '',f"输入 NPZ SHA256：`{reference['inputs_sha256']}`。各运行完全一致；源码、原始场和输入绑定均已核对。",'',
        'Re=10，1,331 单元，3,556 面，无滑移壁面。使用实际 SUBOFF 数值状态，无制造体力。压力数组为修正压力 p*；Cd 参考面积为 L²。', '',
        '| 求解设置 | 外迭代 / 修正数 | 当前动量残差 | 全局动量收支 | 能量收支 | 耗时/s |',
        '|---|---:|---:|---:|---:|---:|']
    for r in records:
        s=r['summary'];c=r['settings']
        name=c['algorithm'] if c['algorithm']=='newton' else f"SIMPLEC，{c.get('relaxation_basis','absolute_row_sum')}，α={c['relaxation']}，内层上限 {c.get('inner_iterations',1)}"
        lines.append(f"| {name} | {r['outer_iterations']} / {r['linear_corrections']} | {s['current_momentum_residual']:.4e} | {s['global_momentum_balance_max']:.4e} | {s['energy_balance_error']:.4e} | {s['runtime_seconds']:.2f} |")
    lines+=['','计数说明：SIMPLEC 每次修正包含动量预测与压力校正；Newton 为联合线性修正。两者不能按外迭代数直接比较计算成本。耗时为各进程记录值，同机并发负载可能影响结果。',
        '', '![当前动量残差](momentum-residual.png)', '',
        '原始面速度重新计算通量，独立散射质量和动量通量，重新积分壁面力。详细字段差异见 comparison.json；此审计不等于另一套离散方法的验证。', '',
        '**仍未完成：完整 SST 稳态收敛、实验雷诺数与壁面函数、同工况 Cp/Cd <3%、网格收敛。**']
    (a.output/'README.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(output=str(a.output),runs=len(records),inputs_verified=True)))


if __name__=='__main__':
    main()
