"""Generate scientific figures from retained numerical-study CSV rows.

Center: median of three timings within an instance, then median across seeds.
Bands: empirical interquartile range across instance medians, not confidence
intervals. All expected rows and statuses are audited by summarize_results.py.
"""
import argparse,csv,statistics
from collections import defaultdict
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

NAMES={'solve_cases':'Algorithm 2','solve_uniform':'Algorithm 3','lp':'HiGHS LP (total)','gurobi':'Gurobi LP (total)'}
COLORS={'solve_cases':'#1b6a93','solve_uniform':'#c05b26','lp':'#555555','gurobi':'#5a8f29'}
SOLVERS=('lp','gurobi')

def read(path):
    with path.open(encoding='utf-8',newline='') as f:return list(csv.DictReader(f))

def seconds(row):
    return float(row['end_to_end_seconds'] if row['method'] in SOLVERS else row['function_seconds'])

def valid(row):
    return (row.get('status') if row['method'] in SOLVERS else row.get('method_status'))=='optimal'

def qtile(vals,q):
    s=sorted(vals); p=(len(s)-1)*q; i=int(p)
    return s[i]+(s[min(i+1,len(s)-1)]-s[i])*(p-i)

def save(fig,out,name):
    fig.savefig(out/(name+'.png'),dpi=200,bbox_inches='tight')
    fig.savefig(out/(name+'.pdf'),bbox_inches='tight')
    plt.close(fig)

def run(results,out):
    out.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
    perf=read(results/'study_performance.csv')+read(results/'study_large.csv')
    cells=defaultdict(list)
    for r in perf:
        if valid(r):cells[(r['family'],int(r['edges']),r['method'],r['label'])].append(seconds(r))
    groups=defaultdict(list); seeds=defaultdict(set)
    for r in perf: seeds[r['family'],int(r['edges'])].add(r['label'])
    for (fam,m,method,label),times in cells.items():
        if len(times)!=(1 if method=='gurobi' and m==100000 else 3):raise AssertionError(f'incomplete timing cell {label}/{method}')
        groups[fam,m,method].append(statistics.median(times))
    # A point is drawn only when every seed of the cell was solved; otherwise
    # the median would be conditioned on the solver finishing.
    for key in [k for k in groups if len(groups[k])<len(seeds[k[0],k[1]])]: del groups[key]
    fig,grid=plt.subplots(2,3,figsize=(7.2,5.0),sharey=True)
    axs=grid.ravel()
    for ax,fam in zip(axs,['chain','star','balanced','comb','random']):
        for method in NAMES:
            ms=sorted(m for f,m,h in groups if f==fam and h==method)
            vs=[groups[fam,m,method] for m in ms]
            if not ms:continue
            med=[statistics.median(v) for v in vs]
            lo=[qtile(v,.25) for v in vs]; hi=[qtile(v,.75) for v in vs]
            assert all(0<a<=b<=c for a,b,c in zip(lo,med,hi))
            ax.plot(ms,med,'o-',label=NAMES[method],color=COLORS[method],markersize=3)
            ax.fill_between(ms,lo,hi,color=COLORS[method],alpha=.14)
        ax.set(xscale='log',yscale='log',title=fam.capitalize(),xlabel='Edges')
        ax.grid(alpha=.2)
    axs[0].set_ylabel('Wall time (seconds)');axs[3].set_ylabel('Wall time (seconds)')
    handles,labels=axs[0].get_legend_handles_labels()
    axs[-1].axis('off');axs[-1].legend(handles,labels,loc='center',fontsize=9,frameon=False)
    fig.tight_layout();save(fig,out,'runtime_scaling')

    sens=read(results/'study_sensitivity.csv');cells=defaultdict(list)
    for r in sens:
        if valid(r):cells[r['factor'],int(r['level']),r['method'],r['label']].append(seconds(r))
    groups=defaultdict(list)
    for (factor,level,method,label),times in cells.items():
        if len(times)!=3:raise AssertionError('incomplete sensitivity timings')
        groups[factor,level,method].append(statistics.median(times))
    factors=['costspread','boundwidth','targetdepth']
    labels=['Nominal cost ratio','Bound half-width','Target depth quantile']
    ticks=[['1','10','100','1000'],['0.25','0.5','1','2'],['0','0.33','0.67','1']]
    fig,axs=plt.subplots(1,3,figsize=(7.2,2.9),sharey=True)
    for ax,factor,label,tick in zip(axs,factors,labels,ticks):
        for method in NAMES:
            vals=[groups[factor,l,method] for l in range(4)]
            ax.plot(range(4),[statistics.median(v) for v in vals],'o-',color=COLORS[method],label=NAMES[method])
            ax.fill_between(range(4),[qtile(v,.25) for v in vals],[qtile(v,.75) for v in vals],alpha=.14,color=COLORS[method])
        ax.set(yscale='log',xlabel=label,xticks=range(4),xticklabels=tick);ax.grid(alpha=.2)
    axs[0].set_ylabel('Wall time (seconds)');axs[-1].legend(fontsize=6.5)
    fig.tight_layout();save(fig,out,'sensitivity')

    branch_rows=read(results/'study_branches.csv');cells=defaultdict(list)
    for r in branch_rows:
        if valid(r):cells[r['family'],int(r['edges']),r['label'],r['method']].append(seconds(r))
    ratios=defaultdict(list)
    for (fam,m,label,method),times in cells.items():
        if method=='solve_cases':
            ratios[fam,m].append(statistics.median(cells[fam,m,label,'solve_uniform'])/statistics.median(times))
    fig,grid=plt.subplots(2,2,figsize=(7.2,5.0),sharey=True)
    axs=grid.ravel()
    for ax,fam in zip(axs,['middle','up','hard-zero','hard-positive']):
        ms=[100,1000,10000];vs=[ratios[fam,m] for m in ms]
        ax.plot(ms,[statistics.median(v) for v in vs],'o-',color=COLORS['solve_cases'])
        ax.fill_between(ms,[qtile(v,.25) for v in vs],[qtile(v,.75) for v in vs],color=COLORS['solve_cases'],alpha=.15)
        ax.axhline(1,color='gray',ls='--');ax.set(xscale='log',yscale='log',xlabel='Edges',title=fam);ax.grid(alpha=.2)
    axs[0].set_ylabel('Algorithm 3 / Algorithm 2 time');axs[2].set_ylabel('Algorithm 3 / Algorithm 2 time')
    fig.suptitle('Targeted branches: time ratios above 1 favor Algorithm 2',fontsize=10)
    fig.tight_layout();save(fig,out,'branch_runtime_ratios')

    points=read(results/'value_function_counterexample.csv')
    fig,ax=plt.subplots(figsize=(5.5,3.4))
    b=[i/100 for i in range(401)]
    ax.plot(b,[max(6-2*x,5-1.5*x) for x in b],color=COLORS['solve_cases'],label=r'$V(B)=\max\{6-2B,5-1.5B\}$')
    p=[r for r in points if float(r['budget'])<=4]
    ax.scatter([float(r['budget']) for r in p],[float(r['lp_value']) for r in p],marker='x',color='black',label='Independent LP')
    ax.axvline(2,color=COLORS['solve_uniform'],ls='--',label='Change of slope at B = 2')
    ax.set(xlabel='Budget B',ylabel='Minimum target-path length',title='An interior value-function breakpoint')
    ax.legend(fontsize=8);ax.grid(alpha=.2);fig.tight_layout();save(fig,out,'value_function_counterexample')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--results',type=Path,default=Path(__file__).parent/'results');p.add_argument('--out',type=Path,default=Path(__file__).parent/'figures');a=p.parse_args();run(a.results,a.out)
