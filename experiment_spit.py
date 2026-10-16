"""Independent original-LP validation of Algorithm 1's parametric subproblem."""
import argparse,csv,json,math,random,sys,warnings
from pathlib import Path
import numpy as np
from scipy.optimize import linprog,OptimizeWarning
from algorithms import Tree,interdiction,spit0
from study_instances import Instance,shape_parents,path_edges


def evaluate(ins,budgets=None):
    t=Tree(ins); edges=list(range(t.m)); side=[v for v in t.leaves if v!=ins.target]
    k=0
    for j,e in enumerate(t.path):
        if len(t.children[t.parent[e]])>1: k=j
    chi=t.path[k:]; chiset=set(chi)
    aux=[e for e in edges if e not in chiset]
    bh=interdiction(t,aux,side)
    cap=max([bh]+[c*max(w-l,u-w) for c,w,l,u in zip(ins.c,ins.w,ins.l,ins.u)])
    if budgets is None: budgets=[bh+(cap-bh)*f for f in [0,.25,.75,1]]
    hat=ins.w[:]
    for e in aux: hat[e]=min(ins.u[e],ins.w[e]+bh/ins.c[e])
    for e in chi: hat[e]=max(ins.l[e],ins.w[e]-bh/ins.c[e])
    obj=np.zeros(t.m); obj[t.path]=1
    A=[]
    for v in side:
        a=np.zeros(t.m); a[path_edges(ins,v)]=-1; A.append(a)
    rows=[]
    for B in budgets:
        x,V,witnesses=spit0(t,edges,side,chi,hat,bh,B)
        L,U=t.bounds(B)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore',OptimizeWarning)
            lp=linprog(obj,A_ub=np.array(A) if A else None,b_ub=[-ins.D]*len(A) if A else None,
                       bounds=list(zip(L,U)),method='highs',options={'threads':1,'time_limit':30})
        paths=[math.fsum(x[e] for e in path_edges(ins,v)) for v in side]
        violation=max([0]+[L[e]-x[e] for e in edges]+[x[e]-U[e] for e in edges]+[ins.D-d for d in paths])
        error=abs(V-lp.fun) if lp.status==0 else math.inf
        rows.append(dict(label=ins.label,edges=t.m,budget=B,auxiliary_budget=bh,V=V,lp_status=lp.status,
                         lp_value=float(lp.fun) if lp.status==0 else None,error=error,violation=violation,
                         witnesses=witnesses,passed=bool(lp.status==0 and error<=2e-6 and violation<=2e-6)))
    return rows


def run(out):
    out.mkdir(parents=True,exist_ok=True); rows=[]; records=[]
    for shape in ['balanced','comb','random']:
        for m in [3,6,12,24]:
            for rep in range(2):
                seed=730000+['balanced','comb','random'].index(shape)*1000+m*10+rep
                r=random.Random(seed)
                # Add a common root gate, making every leaf a gate-subtree leaf.
                parent=[0]+[p+1 for p in shape_parents(m-1,shape,r)]
                parent[2]=1  # Explicitly ensure two branches below the gate.
                w=[r.uniform(-2,4) for _ in range(m)]
                l=[z-r.uniform(1,5) for z in w]; u=[z+r.uniform(3,8) for z in w]
                c=[10**r.uniform(-1,1) for _ in range(m)]
                provisional=Instance(parent,w,l,u,c,m,0,'')
                children=set(parent); leaves=[v for v in range(1,m+1) if v not in children]
                target=max(leaves,key=lambda v:len(path_edges(provisional,v)))
                side=[v for v in leaves if v!=target]
                if not side: raise AssertionError('side-leaf generator invariant')
                low=min(math.fsum(w[e] for e in path_edges(provisional,v)) for v in side)
                high=min(math.fsum(u[e] for e in path_edges(provisional,v)) for v in side)
                D=low+(high-low)*(.2 if rep==0 else .7)
                ins=Instance(parent,w,l,u,c,target,D,f'spit-{shape}-m{m}-s{seed}')
                records.append(vars(ins)); rows.extend(evaluate(ins))
    counter=Instance([0,1,1],[2,4,1],[-8,-6,-9],[12,14,11],[1,1,2],2,2,'value-function-counterexample')
    curve=evaluate(counter,[0,.5,1,1.5,1.9,2,2.1,2.5,3,5,10,20])
    for name,data in [('algorithm1_validation.csv',rows),('value_function_counterexample.csv',curve)]:
        with (out/name).open('w',newline='',encoding='utf-8') as f:
            writer=csv.DictWriter(f,fieldnames=data[0]); writer.writeheader(); writer.writerows(data)
    (out/'algorithm1_instances.json').write_text(json.dumps(records,indent=2),encoding='utf-8')
    summary=dict(instances=len(records),parametric_cases=len(rows),passed=sum(r['passed'] for r in rows),
                 max_error=max(r['error'] for r in rows),max_violation=max(r['violation'] for r in rows),
                 counterexample_points=len(curve),counterexample_passed=sum(r['passed'] for r in curve))
    (out/'algorithm1_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(json.dumps(summary))

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--out',type=Path,default=Path(__file__).resolve().parent/'results'); a=p.parse_args(); run(a.out)
