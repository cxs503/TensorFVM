#!/usr/bin/env python3
"""Build a source-bound SUBOFF report from actual separate Cp and drag runs."""
import argparse
import json
import os
from pathlib import Path
import shutil

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from plot_native_suboff_fields import plot_fields
from run_native_suboff_benchmark import digest, write


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('cp-run','drag-run','cp-comparison','drag-comparison','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args();out=args.output
    if out.exists() and any(out.iterdir()):parser.error('output must be empty')
    out.mkdir(parents=True,exist_ok=True)
    runs={'cp':args.cp_run,'drag':args.drag_run}
    comparisons={'cp':args.cp_comparison,'drag':args.drag_comparison}
    records={};bindings={}
    link=lambda path:os.path.relpath(path,out)
    for name,run in runs.items():
        result=json.loads((run/'result.json').read_text())
        config=json.loads((run/'config.json').read_text())
        compared=json.loads((comparisons[name]/'comparison.json').read_text())
        if compared['numerically_converged']!=result['numerically_converged']:
            raise ValueError('comparison and run status differ')
        for path,sha in compared['source_sha256'].items():
            candidate=run/Path(path).name
            if digest(candidate)!=sha:raise ValueError('comparison does not bind this run')
        records[name]=dict(result=result,config=config,comparison=compared)
        for path in (run/'result.json',run/'config.json',run/'fields.npz',run/'input.msh',comparisons[name]/'comparison.json'):
            bindings[link(path)]=digest(path)
        plot_fields(run/'input.msh',run/'fields.npz',out/(name+'-fields'),
            status_text='Numerically converged; physical validation pending' if result['numerically_converged'] else 'UNCONVERGED diagnostic; not an accuracy qualification',case_kind='physical')
    cp=records['cp']['comparison']['metrics'];drag=records['drag']['comparison']['metrics']
    shutil.copy2(comparisons['cp']/'cp-comparison.png',out/'cp-comparison.png')
    shutil.copy2(comparisons['cp']/'cp-taps.csv',out/'cp-taps.csv')
    fig,axes=plt.subplots(2,2,figsize=(12,8))
    for name,record in records.items():
        result=record['result'];inner=result.get('momentum_history',[]);outer=result['history']
        for ax,key in zip(axes[0],['momentum_residual','current_reconstructed_divergence_max']):
            ax.semilogy(np.arange(1,len(inner)+1),[max(row[key],1e-30) for row in inner],label=name)
            ax.set(xlabel='Completed momentum iteration',ylabel=key.replace('_',' '));ax.grid(alpha=.25)
        for ax,key in zip(axes[1],['k_current_nonlinear_residual','omega_current_nonlinear_residual']):
            if outer:ax.semilogy([row['iteration'] for row in outer],[max(row[key],1e-30) for row in outer],label=name)
            ax.set(xlabel='Completed coupled iteration',ylabel=key.replace('_',' '));ax.grid(alpha=.25)
    for ax in axes.ravel():
        ax.axhline(1e-7,color='black',ls='--',lw=.8)
        if ax.lines and ax.get_legend_handles_labels()[0]:ax.legend()
    fig.suptitle('Actual current-equation residuals; dashed line = outer tolerance')
    fig.tight_layout();fig.savefig(out/'residuals.png',dpi=180);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,4))
    for name,record in records.items():
        fields=np.load(runs[name]/'fields.npz');wall=fields['wall_face_indices']
        from tensorfvm.native_mesh3d import NativeMesh3D
        mesh=NativeMesh3D.from_gmsh(runs[name]/'input.msh',triangulate_faces=True)
        axes[0].scatter(mesh.face_centers[wall,0],fields['wall_yplus'],s=3,alpha=.35,label=name)
    axes[0].axhline(1,color='black',ls='--');axes[0].set(xlabel='x/L',ylabel='Actual wall y+',title='All native wall faces');axes[0].legend()
    components=drag['axial_components'];names=list(components)
    axes[1].bar(np.arange(len(names)),[components[name]['Cd'] for name in names],label='Calculated components')
    axes[1].axhline(drag['Cd_reference'],color='black',ls='--',label='Reference TOTAL Cd')
    axes[1].set_xticks(np.arange(len(names)),['Pressure','Viscous','Isotropic Reynolds'],rotation=12)
    axes[1].set(ylabel='Frontal-area Cd',title='Diagnostic force decomposition');axes[1].legend()
    fig.tight_layout();fig.savefig(out/'wall-and-force.png',dpi=180);plt.close(fig)
    table='\n'.join(f"| {name} | {r['result']['cells']:,} | {r['result']['Re_L']:.9g} | {r['result']['completed_momentum_iterations']} | {len(r['result']['history'])} | {r['result']['numerically_converged']} | {r['result']['low_Re_wall_resolution']['maximum_yplus']:.6g} |" for name,r in records.items())
    failures='\n'.join(f"- {name}: `{r['result']['failure']}`; state `{r['result']['solution_state']}`." for name,r in records.items())
    sources='\n'.join(f"- {name}: [raw calculation]({link(runs[name]/'report.md')}), [experimental comparison]({link(comparisons[name]/'report.md')})." for name in records)
    summary=dict(physical_validation_passed=False,
        Cp_relative_L2_error=cp['nominal_relative_L2_error'],Cp_absolute_RMSE=cp['absolute_RMSE'],
        Cd_relative_error=drag['relative_error'],Cd_calculated=drag['Cd_calculated'],Cd_reference=drag['Cd_reference'],
        both_numerically_converged=all(r['result']['numerically_converged'] for r in records.values()),
        spatial_convergence_qualified=False,input_sha256=bindings)
    write(out/'summary.json',summary)
    (out/'report.md').write_text(f'''# SUBOFF AFF-1：纯贴体 FVM 物理基线

## 问题与验收目标

裸艇、零攻角、真实无滑移壁面；以 19 个表面压力测点的 Cp 曲线相对 L2 误差和总阻力 Cd 相对误差均小于 3% 为主要目标。
**本报告物理验收未通过。未收敛场的误差仅表示当前迭代场偏差，不是已收敛解的精度。**

## 工况、模型与软件设置

Cp 对照风洞 Re_L=1.2e7；阻力采用文献复现工况 U=3.046 m/s、rho=998.2 kg/m³、mu=0.001002 Pa s、L=4.356 m、D=0.508 m。
两项独立运行，未混用雷诺数。SST-2003，低 Re 近壁积分，无壁面函数；入流湍流强度 0.5%、长度尺度 0.05L 为建模假设。
未显式建立实验绊线，也未使用转捩模型。边界为来流速度、远场速度、物理压力出口 p=0；压力出口回流使用显式给定的来流速度和湍流量。
物理压力由 pi=p+2rho k/3 转换；Cd 包含压力和全部黏性/雷诺应力。运行配置与源代码快照保存于各计算目录。

| 工况 | 单元数 | Re_L | 动量步数 | 完整耦合步数 | 数值收敛 | 最大 y+ |
|---|---:|---:|---:|---:|---|---:|
{table}

## 主要对照结果

| 指标 | 当前结果 | 目标 |
|---|---:|---:|
| Cp 曲线相对 L2 偏差 | {100*cp['nominal_relative_L2_error']:.6g}% | <3% |
| Cp 绝对 RMSE | {cp['absolute_RMSE']:.6g} | 补充指标 |
| 计算 Cd | {drag['Cd_calculated']:.8g} | — |
| 参考 Cd | {drag['Cd_reference']:.8g} | — |
| 总阻力相对偏差 | {100*drag['relative_error']:.6g}% | <3% |
| 计算总阻力 | {drag['total_drag_N']:.8g} N | 参考 87.4 N |

![Cp 与参考测点](cp-comparison.png)

[逐点 Cp、差值及实际采样面编号](cp-taps.csv)。测点按真实三角壁面就近采样，未根据实验调整压力常数、平滑曲线或缩放结果。
Cp 参考为已保存的二级转录候选表，图中 ±0.015 为文献报告的绝对不确定度；表内修正差异另作敏感性比较。
阻力参考来自实验，密度和黏度属于公开 CFD 复现条件；Cd 使用迎流截面积，与湿表面积 Ct 不混用。

## 网格与真实场

表面结构化层、端部楔形、外部四面体，共形单域。图像使用真实单元与 z=0 平面的交线和原始单元值，无点云插值或平滑。

![Cp 工况网格、压力与速度](cp-fields/fields.png)

![阻力工况网格、压力与速度](drag-fields/fields.png)

## 收敛、近壁与力分解

![原方程残差](residuals.png)

![实际 y+ 与阻力分解](wall-and-force.png)

{failures}

残差为当前原方程残差；线性求解成功或小迭代增量不能替代耦合收敛。低 Re SST 的 y+ 检查必须同时具备已收敛状态。
尚未完成三档网格或计算域独立性验证；CPU SciPy 运行，未开展本例 GPU 验证。

## 复现与原始证据

使用 `scripts/run_native_suboff_rans.py` 分别运行两个 Re；用 `scripts/compare_native_suboff_experiment.py --case cp/drag` 生成对应比较；本报告由 `scripts/report_suboff_physical_baseline.py` 从冻结数据生成。
具体迭代、离散、初始化与线性求解参数以每例 config.json 为准。summary.json 和 manifest.json 绑定输入及报告文件 SHA256。

{sources}

参考：[Liu & Huang 实验报告](https://ntrl.ntis.gov/NTRL/dashboard/searchResults/titleDetail/ADA359226.xhtml)、[Cp 数据候选转录](https://zenodo.org/records/22692726)、[Morse & Mahesh 2023](https://doi.org/10.1017/jfm.2023.777)、[阻力复现工况](https://academic.oup.com/ooenergy/article/doi/10.1093/ooenergy/oiaf014/8405192)。
''')
    shutil.copy2(__file__,out/'report-source.py')
    write(out/'manifest.json',{str(p.relative_to(out)):digest(p) for p in sorted(out.rglob('*')) if p.is_file() and p!=out/'manifest.json'})
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
