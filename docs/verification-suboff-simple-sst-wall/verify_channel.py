"""Reproduce the small-channel solver verification using the tested mesh fixture."""
from pathlib import Path
import json
import sys
import numpy as np
root=Path(__file__).resolve().parent
repo=root.parents[1]
sys.path[:0]=[str(repo/'src'),str(repo)]
from tests.test_native_rans3d import channel
from tensorfvm.native_rans3d import NativeSimpleSSTRansSolver3D,NativeRansSettings
from tensorfvm.native_collocated3d import NativeSteadySettings
m=channel();fixed=(m.neighbor<0)&(m.patch_labels!='outlet')
bd=np.tile([1.,0.,0.],(len(m.owner),1));bd[m.patch_labels=='wall']=0.
s=NativeSimpleSSTRansSolver3D(m,fixed,bd,m.patch_labels=='outlet',
    settings=NativeSteadySettings(viscosity=.01,relaxation=.4,pressure_relaxation=.2,convection_order=1,max_iterations=10),
    rans_settings=NativeRansSettings(max_outer_iterations=150,pseudo_time_step=1.,turbulence_relaxation=.7,wall_treatment='spalding'))
s.u[:,0]=1.
r=s.solve_rans(.001,1.,inlet_k=.001,inlet_omega=1.)
assert r['numerically_converged']
(root/'channel-convergence.json').write_text(json.dumps(dict(converged=r['numerically_converged'],history=r['history']),indent=2)+'\n')
last=r['history'][-1]
p=root/'README.md';text=p.read_text();start=text.index('[通道迭代记录]');end=text.index('\n对应可复现断言',start)
text=text[:start]+f"[通道迭代记录](channel-convergence.json)，由 [复现脚本](verify_channel.py) 生成：{last['iteration']} 轮收敛；当前动量残差 {last['momentum_current_residual']:.3g}，重构散度 {last['current_reconstructed_divergence_max']:.3g}，k 残差 {last['k_current_nonlinear_residual']:.3g}，ω 残差 {last['omega_current_nonlinear_residual']:.3g}。"+text[end:]
p.write_text(text)
print('CHANNEL_CONVERGED',last['iteration'])
