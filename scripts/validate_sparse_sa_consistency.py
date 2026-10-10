"""Repeat the previously failed SA configuration with the opt-in sparse solver."""
import json,statistics,time
from pathlib import Path
import numpy as np
import torch
from tensorfvm.solver import SolverConfig
from tensorfvm.sparse_simple import SparseBodyFittedSolver,host
from tensorfvm.verification.core import sha,write_json,artifact_manifest

root=Path(__file__).resolve().parents[1];output=root/'docs/verification-sparse-sa-consistency';output.mkdir(exist_ok=False);torch.set_num_threads(1)
old=root/'docs/verification-gpu/SA-128x112-five-iterations';c=json.loads((old/'cpu-result.json').read_text())['config'];summary=dict(previous_comparison_sha256=sha(old/'comparison.json'),source_sha256={name:sha(root/name) for name in ['src/tensorfvm/body_fitted.py','src/tensorfvm/sparse_simple.py','src/tensorfvm/solver.py']},hardware=torch.cuda.get_device_name(0),producer_sha256=sha(Path(__file__)),devices={})
fields={}
for device in ['cpu','cuda']:
    samples=[]
    for repeat in range(1):
        cfg=SolverConfig(**dict(c,device=device))
        if device=='cuda':torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
        start=time.perf_counter();s=SparseBodyFittedSolver(cfg)
        for _ in range(5):s.step()
        if device=='cuda':torch.cuda.synchronize()
        elapsed=time.perf_counter()-start
        samples.append(elapsed)
        print(device,repeat,elapsed,flush=True)
    fields[device]=dict(velocity=host(s.velocity),pressure=host(s.p),nu_tilde=host(s.nu_tilde));np.savez_compressed(output/(device+'-fields.npz'),**fields[device]);write_json(output/(device+'-result.json'),dict(config=dict(c,device=device),history=s.history,linear_history=s.linear_history,actual_solver_device=str(s.p.device)))
    summary['devices'][device]=dict(samples_s=samples,median_s=statistics.median(samples),peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated() if device=='cuda' else None)
summary['field_differences']={k:float(np.linalg.norm(fields['cpu'][k]-fields['cuda'][k])/np.linalg.norm(fields['cpu'][k])) for k in fields['cpu']};summary['equivalent']=all(v<=1e-6 for v in summary['field_differences'].values());summary['cpu_over_cuda']=summary['devices']['cpu']['median_s']/summary['devices']['cuda']['median_s'];summary['physical_passed']=None;summary['timing_scope']='one cold full five-step execution per device; correctness diagnostic, not a qualified performance benchmark';write_json(output/'summary.json',summary)
(output/'report.md').write_text('# 原失败 SA 配置的稀疏后端复核\n\n严格沿用原 128×112、Re=100000、pseudo_time_step=0.02、松弛系数与五次实际迭代。改变的是可选线性后端：原压力算子准确组装，行/粗网格预条件 GMRES，真实残差检查。矩阵CPU组装，线性迭代CPU或真实CUDA。本轮优先案例正确性，各设备只执行一次真实五步求解；单次同步时钟含设置/组装/求解，含首次设备初始化，不作为重复性能 benchmark。没有解析初始化，没有SA内迭代或Anderson。\n\nCPU/CUDA 主场相对差：'+str(summary['field_differences'])+'；一致性通过：'+str(summary['equivalent'])+'；CPU/CUDA 时间比：'+str(summary['cpu_over_cuda'])+'。\n\n这是五次迭代的一致性修复，不能称稳态湍流或3%物理验证通过。旧失败证据保留在 verification-gpu。另见完整稳态 SA 报告 verification-simple-gpu。\n')
write_json(output/'manifest.json',dict(source_sha256=summary['source_sha256'],artifacts_sha256=artifact_manifest(output)));print(summary['field_differences'],summary['equivalent'])
