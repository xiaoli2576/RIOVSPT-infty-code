"""Predeclared targeted runtime comparisons within each Algorithm 2 branch.

Four explicit feasible tree templates, 3 sizes, 3 seeds; no outcome filtering.
Run only after all other timing groups finish.
"""
import argparse,random
from pathlib import Path
from study_instances import Instance
import study_runner
from study_runner import solve_methods_batch,method_record,_lp_checked,solve_lp,solve_gurobi,write_csv

BRANCHES=['middle','up','hard-zero','hard-positive']

def instance(branch,m,seed):
    r=random.Random(seed)
    if branch=='hard-zero':
        parent=[0,1,2,3,3,1]+[1]*(m-6)
        w=l=[0.0]*m;u=[3.0]*m;u[3]=0.0;c=[1.0]*6+[r.uniform(.8,1.2) for _ in range(m-6)]
        target=4
    elif branch in ('middle','up'):
        parent=[0]+[1]*(m-1);w=l=[0.0]*m;u=[3.0]*m
        c=[1.0,.5 if branch=='middle' else 4.0]+[r.uniform(.8,1.2) for _ in range(m-2)]
        target=2
    else:
        parent=[0]+[1]*(m-1);w=[2.0,4.0]+[1.0]*(m-2)
        l=[v-10 for v in w];u=[v+10 for v in w];c=[1.0,1.0]+[r.uniform(2,4) for _ in range(m-2)]
        target=2
    return Instance(parent,w[:],l[:],u,c,target,2.0,f'branch-{branch}-m{m}-s{seed}')

def run(out):
    study_runner.IN_PROCESS=True
    out.mkdir(parents=True,exist_ok=True);rows=[]
    for branch in BRANCHES:
        for m in [100,1000,10000]:
            for k in range(3):
                seed=850000+10000*BRANCHES.index(branch)+m+k
                ins=instance(branch,m,seed)
                base=dict(family=branch,expected_branch=branch,edges=m,seed_index=k,actual_seed=seed,label=ins.label)
                runs=solve_methods_batch(ins,['solve_uniform']*4+['solve_cases']*4)
                for method,start in [('solve_uniform',0),('solve_cases',4)]:
                    for repeat,ans in enumerate(runs[start+1:start+4]):
                        row=base|{'repeat':repeat}|method_record(ins,method,ans)
                        if method=='solve_cases':row['branch_matches_design']=ans.get('branch')==branch
                        rows.append(row)
                solve_lp(ins)
                for repeat in range(3):rows.append(base|{'method':'lp','repeat':repeat}|_lp_checked(ins))
                solve_gurobi(ins)
                for repeat in range(3):rows.append(base|{'method':'gurobi','repeat':repeat}|_lp_checked(ins,solve_gurobi))
    write_csv(out/'study_branches.csv',rows)
    print(f'Completed branch comparison: {len(rows)} measured rows / 36 instances.')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--scipy-path');p.add_argument('--out',type=Path,default=Path(__file__).parent/'results');a=p.parse_args();run(a.out)
