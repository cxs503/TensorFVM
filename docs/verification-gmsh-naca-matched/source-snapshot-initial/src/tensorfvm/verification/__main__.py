"""Run source-bound CFD benchmarks through shared export, gates and reporting."""
import argparse
from pathlib import Path
import sys
import torch
from .cases import channel, taylor_green
from .core import provenance, save_run, write_json, observed_orders, artifact_manifest, metric_value


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--max-channel-iterations",type=int,default=1000)
    parser.add_argument("--lbm-repo",type=Path,help="external TensorLBM checkout; enables actual matched periodic comparison")
    args=parser.parse_args(argv)
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("output directory must be empty; preserve prior run evidence")
    args.output.mkdir(parents=True,exist_ok=True);torch.set_num_threads(1)
    from .report import render_case, render_study, write_report, comparison_report
    summary=dict(schema="tensorfvm.analytic-benchmark-suite/1",error_limit=.03,
                 provenance=provenance(),runs=[],passed=False,engineering_qualified=False)
    def execute(name,fn):
        directory=args.output/name
        try:
            raw=fn();raw["directory"]=name;result=save_run(directory,raw)
        except Exception as error:
            write_json(args.output/"execution-failure.json",
                       dict(run=name,type=type(error).__name__,message=str(error)))
            raise
        summary["runs"].append(result)
        write_json(args.output/"execution-status.json",dict(completed=name,runs=len(summary["runs"])))
        render_case(directory,result)
        print(name,result["passed"],flush=True)
    for ny in (12,24,48):
        execute(f"channel-{ny}",lambda ny=ny:channel(ny,args.max_channel_iterations))
    for n in (32,48,64):
        execute(f"tg-space-{n}",lambda n=n:taylor_green(n=n))
    for dt in (.1,.05,.025):
        execute(f"tg-time-{dt}",lambda dt=dt:taylor_green(n=32,dt=dt,role="temporal"))
    execute("control-unconverged-channel",lambda:dict(channel(12,1),role="negative-control"))
    execute("control-coarse-tg-pressure",lambda:dict(taylor_green(16),role="negative-control"))
    if args.lbm_repo:
        from .lbm import taylor_green_lbm
        for n,mach in ((32,.05),(64,.05),(32,.025)):
            execute(f"lbm-tg-{n}-ma-{mach}",lambda n=n,mach=mach:taylor_green_lbm(args.lbm_repo,n,mach))
    spatial=lambda case:[r for r in summary["runs"] if r["case"]==case and r["role"]=="spatial"]
    temporal=[r for r in summary["runs"] if r["role"]=="temporal"]
    summary["observed_orders"]={
      "channel_velocity":observed_orders(spatial("poiseuille"),"velocity_l2"),
      "channel_pressure_gradient":observed_orders(spatial("poiseuille"),"pressure_gradient"),
      "tg_velocity":observed_orders(spatial("taylor-green"),"velocity_l2"),
      "tg_pressure":observed_orders(spatial("taylor-green"),"pressure_l2"),
      "tg_time":observed_orders(temporal,"temporal_velocity_l2")}
    regular=[r for r in summary["runs"] if r["role"] not in ("negative-control","comparison")]
    controls=[r for r in summary["runs"] if r["role"]=="negative-control"]
    summary["fvm_all_regular_passed"]=all(r["passed"] for r in regular)
    summary["negative_controls_rejected"]=all(not r["passed"] for r in controls)
    summary["second_order_tg_passed"]=all(1.8<p<2.2 for key in ("tg_velocity","tg_pressure","tg_time") for p in summary["observed_orders"][key])
    summary["passed"]=summary["fvm_all_regular_passed"] and summary["negative_controls_rejected"] and summary["second_order_tg_passed"]
    summary["comparison_passed"]=all(r["passed"] for r in summary["runs"] if r["role"]=="comparison") if args.lbm_repo else None
    write_json(args.output/"summary.json",summary)
    render_study(args.output,summary)
    if args.lbm_repo:comparison_report(args.output,summary)
    write_report(args.output,summary)
    # Independent NumPy replay runs before final artifact hashing.
    from .audit import audit
    write_json(args.output/"audit.json",audit(args.output,source_check=True))
    write_json(args.output/"manifest.json",dict(schema="tensorfvm.benchmark-manifest/1",
       source_sha256=summary["provenance"]["source_sha256"],artifacts_sha256=artifact_manifest(args.output)))
    print("FVM qualification:",summary["passed"],"LBM accuracy comparison:",summary["comparison_passed"],flush=True)
    return 0 if summary["passed"] else 2


if __name__=="__main__":
    sys.exit(main())
