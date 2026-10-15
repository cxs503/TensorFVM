"""Plot lift/drag and residual histories of one or more 30P30N steady runs.

Reads ``history-live.json`` (written every iteration) and the per-iteration
coefficient lines of the run log when present, so it works on running jobs.
"""
import argparse,json,re
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def coefficients(run):
    log=run.with_suffix('.log')
    if not log.exists():return [],[],[]
    pattern=re.compile(r'"iteration": (\d+).*?"drag": ([-0-9.e]+), "lift": ([-0-9.e]+)')
    iterations,lift,drag=[],[],[]
    for line in log.read_text().splitlines():
        match=pattern.search(line)
        if match:iterations.append(int(match.group(1)));drag.append(float(match.group(2)));lift.append(float(match.group(3)))
    return iterations,lift,drag


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('runs',type=Path,nargs='+');p.add_argument('--output',type=Path,required=True);p.add_argument('--reference-lift',type=float,default=3.1354,help='LTPT experiment Cl at alpha=8.1 deg (HLPW4 30P30N)');args=p.parse_args()
    fig,axes=plt.subplots(1,3,figsize=(16,4.5),constrained_layout=True)
    for run in args.runs:
        history=json.loads((run/'history-live.json').read_text()) if (run/'history-live.json').exists() else json.loads((run/'history.json').read_text())
        iterations=[h['iteration'] for h in history]
        axes[0].semilogy(iterations,[h['velocity_change'] for h in history],label=run.name+' max |du|/U')
        axes[0].semilogy(iterations,[h['momentum'] for h in history],'--',label=run.name+' momentum L1')
        axes[0].semilogy(iterations,[h.get('turbulence',float('nan')) for h in history],':',label=run.name+' SA defect')
        it,lift,drag=coefficients(run)
        if it:
            # Skip the start-up transient when choosing the axis range.
            start=min(len(it)-1,max(5,len(it)//10))
            axes[1].plot(it,lift,label=run.name);axes[2].plot(it,drag,label=run.name)
            for ax,values in ((axes[1],lift[start:]),(axes[2],drag[start:])):
                low,high=min(values),max(values);margin=.1*(high-low) or .05
                previous=ax.get_ylim() if ax.lines[:-1] else (low-margin,high+margin)
                ax.set_ylim(min(previous[0],low-margin),max(previous[1],high+margin))
    low,high=axes[1].get_ylim();axes[1].set_ylim(min(low,args.reference_lift-.1),max(high,args.reference_lift+.1))
    axes[1].axhline(args.reference_lift,color='k',ls='--',lw=.8,label='LTPT experiment')
    for ax,title,ylabel in zip(axes,('Residual history','Lift coefficient','Drag coefficient'),('normalized residual','Cl','Cd')):
        ax.set_title(title);ax.set_xlabel('SIMPLE iteration');ax.set_ylabel(ylabel);ax.grid(alpha=.3);ax.legend(fontsize=7)
    fig.savefig(args.output,dpi=140)
    print(args.output)


if __name__=='__main__':main()
