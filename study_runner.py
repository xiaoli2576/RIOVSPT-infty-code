"""Common LP baseline, serial method runner, and study plans.

No formal group is launched by default.  Once algorithms.py is present, use
the explicit CLI flags from a quiet machine so timed methods remain serial.
"""
from __future__ import annotations
import argparse,csv,json,math,os,subprocess,sys,tempfile,time,traceback
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]; HERE=Path(__file__).resolve().parent
if '--scipy-path' in sys.argv:
    # Batch children run with cwd=HERE, so normalize a caller-relative path
    # before passing it back to their command line.
    _sp=Path(sys.argv[sys.argv.index('--scipy-path')+1]).resolve()
else:
    _sp=(ROOT/'work'/'numerical_study'/'deps').resolve()  # local convenience default; not required by the source tree
sys.path.insert(0,str(_sp)); sys.path.insert(0,str(HERE))
os.environ.setdefault('OMP_NUM_THREADS','1'); os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import lil_matrix
from study_instances import Instance,SHAPES,correctness_random,named_cases,structured_performance,sensitivity_instance,instance_record,instance_from_record,children,leaves,path_edges

TOL=2e-6; DEADLINE_SECONDS=30; LARGE_LIMIT=600
IN_PROCESS=False

def validate_solution(ins,x,objective):
    """Independent checks of bounds, target equality, side paths and objective."""
    m=len(ins.parent)
    if len(x)!=m or not math.isfinite(objective) or not all(math.isfinite(v) for v in x): return {'valid':False,'reason':'nonfinite-or-wrong-x','max_violation':float('inf'),'actual_objective':None}
    dist=[0.0]*(m+1); maxv=0.0
    for i in range(m):
        maxv=max(maxv,ins.l[i]-x[i],x[i]-ins.u[i]); dist[i+1]=dist[ins.parent[i]]+x[i]
    maxv=max(maxv,abs(dist[ins.target]-ins.D))
    for v in leaves(ins):
        if v!=ins.target: maxv=max(maxv,ins.D-dist[v])
    actual=max((ins.c[i]*abs(x[i]-ins.w[i]) for i in range(m)),default=0.0)
    return {'valid':maxv<=TOL and actual<=objective+TOL,'max_violation':max(0.0,maxv),'actual_objective':actual,'budget_gap':objective-actual}

def build_lp(ins):
    """Sparse original epigraph LP with node-length variables (no path matrix).
    Variable order: edge weights x (m), node distances d (m+1), budget z."""
    m=len(ins.parent); n=m+1; z=m+n; q=z+1
    Aeq=lil_matrix((m+2,q)); beq=np.zeros(m+2); Aeq[0,m]=1
    for i,p in enumerate(ins.parent): Aeq[i+1,i]=-1; Aeq[i+1,m+i+1]=1; Aeq[i+1,m+p]=-1
    Aeq[m+1,m+ins.target]=1; beq[m+1]=ins.D
    side=[v for v in leaves(ins) if v!=ins.target]; Aub=lil_matrix((2*m+len(side)+1,q)); bub=np.zeros(2*m+len(side)+1); row=0
    for i in range(m):
        Aub[row,i]=ins.c[i]; Aub[row,z]=-1; bub[row]=ins.c[i]*ins.w[i]; row+=1
        Aub[row,i]=-ins.c[i]; Aub[row,z]=-1; bub[row]=-ins.c[i]*ins.w[i]; row+=1
    for v in side: Aub[row,m+v]=-1; bub[row]=-ins.D; row+=1
    Aub[row,z]=-1
    bounds=[(ins.l[i],ins.u[i]) for i in range(m)]+[(None,None)]*n+[(0,None)]
    return np.r_[np.zeros(z),1.0],Aub.tocsr(),bub,Aeq.tocsr(),beq,bounds

def solve_lp(ins,limit=DEADLINE_SECONDS):
    """HiGHS (SciPy linprog) on the original epigraph LP."""
    m=len(ins.parent); started=time.perf_counter()
    cost,Aub,bub,Aeq,beq,bounds=build_lp(ins); built=time.perf_counter()
    res=linprog(cost,A_ub=Aub,b_ub=bub,A_eq=Aeq,b_eq=beq,bounds=bounds,method='highs',options={'time_limit':limit,'threads':1})
    ended=time.perf_counter(); status={0:'optimal',1:'time_limit_or_iteration',2:'infeasible',3:'unbounded',4:'numerical_or_other'}.get(res.status,'unknown')
    return {'status':status,'scipy_status':int(res.status),'objective':float(res.fun) if res.status==0 else None,'x':res.x[:m].tolist() if res.status==0 else None,'build_seconds':built-started,'solve_seconds':ended-built,'end_to_end_seconds':ended-started,'highs_reported_seconds':getattr(res,'execution_time',None),'message':str(res.message).split('\n')[0]}

_GRB_ENV=None
def solve_gurobi(ins,limit=DEADLINE_SECONDS):
    """Gurobi on the same epigraph LP: one thread, barrier, tolerances 1e-9,
    DualReductions=0 so that infeasibility is reported unambiguously."""
    global _GRB_ENV
    import gurobipy as gp
    from gurobipy import GRB
    if _GRB_ENV is None:
        _GRB_ENV=gp.Env(empty=True); _GRB_ENV.setParam('OutputFlag',0); _GRB_ENV.start()
    m=len(ins.parent); started=time.perf_counter()
    cost,Aub,bub,Aeq,beq,bounds=build_lp(ins)
    lb=np.array([-GRB.INFINITY if a is None else a for a,_ in bounds]); ub=np.array([GRB.INFINITY if b is None else b for _,b in bounds])
    mod=gp.Model(env=_GRB_ENV)
    # Method=2 (barrier + crossover) was the fastest of primal/dual/barrier in a pilot on 10^4/10^5-edge chain and comb trees.
    for k,v in (('Threads',1),('Method',2),('TimeLimit',limit),('FeasibilityTol',1e-9),('OptimalityTol',1e-9),('DualReductions',0)): mod.setParam(k,v)
    y=mod.addMVar(len(cost),lb=lb,ub=ub,obj=cost)
    mod.addMConstr(Aub,y,'<',bub); mod.addMConstr(Aeq,y,'=',beq); mod.update(); built=time.perf_counter()
    mod.optimize(); ended=time.perf_counter()
    st={GRB.OPTIMAL:'optimal',GRB.INFEASIBLE:'infeasible',GRB.UNBOUNDED:'unbounded',GRB.TIME_LIMIT:'time_limit_or_iteration'}.get(mod.Status,'numerical_or_other')
    ok=st=='optimal'
    ans={'status':st,'gurobi_status':int(mod.Status),'objective':float(mod.ObjVal) if ok else None,'x':y.X[:m].tolist() if ok else None,'build_seconds':built-started,'solve_seconds':ended-built,'end_to_end_seconds':ended-started,'gurobi_runtime_seconds':mod.Runtime,'simplex_iterations':mod.IterCount}
    mod.dispose(); return ans

def _worker(input_path,output_path,method):
    try:
        import algorithms
        ins=instance_from_record(json.loads(Path(input_path).read_text(encoding='utf-8')))
        result=getattr(algorithms,method)(ins)
        Path(output_path).write_text(json.dumps(result),encoding='utf-8')
    except Exception:
        Path(output_path).write_text(json.dumps({'status':'error','error':traceback.format_exc()}),encoding='utf-8')

def _batch_worker(input_path,output_path):
    """One import per instance; function_seconds deliberately excludes IPC/startup."""
    try:
        import algorithms
        payload=json.loads(Path(input_path).read_text(encoding='utf-8')); ins=instance_from_record(payload['instance'])
        results=[]
        for method in payload['calls']:
            t=time.perf_counter(); result=getattr(algorithms,method)(ins); result['function_seconds']=time.perf_counter()-t; results.append(result)
            # Persist after every completed call so an outer batch guard can
            # distinguish completed work from calls never started.
            Path(output_path).write_text(json.dumps(results),encoding='utf-8')
    except Exception:
        Path(output_path).write_text(json.dumps([{'status':'error','error':traceback.format_exc()}]),encoding='utf-8')

def solve_methods_batch(ins,calls,limit_each=DEADLINE_SECONDS):
    """Serial worker with a batch guard; completed calls over 30s are flagged.

    This avoids repeated SciPy imports. It is intentionally not described as
    a hard per-call kill: a call exceeding 30s becomes limit_exceeded, while
    a batch kill preserves completed calls and labels the remainder not_run.
    """
    if IN_PROCESS:
        # Formal local mode: algorithms are already independently canaried and
        # the limit is soft.  This avoids Windows child-process import/IPC cost.
        import algorithms
        ans=[]
        for method in calls:
            try:
                t=time.perf_counter(); result=getattr(algorithms,method)(ins); result['function_seconds']=time.perf_counter()-t
                if result['function_seconds']>limit_each:
                    result['underlying_status']=result.get('status'); result['status']='limit_exceeded'
            except Exception:
                result={'status':'error','error':traceback.format_exc()}
            ans.append(result)
        return ans
    with tempfile.TemporaryDirectory(prefix='riovspt_batch_') as d:
        inp=Path(d)/'input.json'; out=Path(d)/'output.json'; inp.write_text(json.dumps({'instance':instance_record(ins),'calls':calls}),encoding='utf-8')
        cmd=[sys.executable,str(HERE/'study_runner.py'),'--scipy-path',str(_sp),'--batch-worker',str(inp),str(out)]; start=time.perf_counter()
        timed_out=False
        try: proc=subprocess.run(cmd,cwd=str(HERE),timeout=limit_each*len(calls)+5,check=False,capture_output=True,text=True)
        except subprocess.TimeoutExpired: timed_out=True
        if not out.exists():
            detail='batch-worker-no-output'
            if not timed_out:
                detail+=f'; returncode={proc.returncode}; stderr={proc.stderr[-1200:]}'
            return [{'status':'error','error':detail} for _ in calls]
        ans=json.loads(out.read_text(encoding='utf-8'))
        for result in ans:
            if result.get('function_seconds',0)>limit_each:
                result['underlying_status']=result.get('status'); result['status']='limit_exceeded'
        if len(ans)<len(calls): ans.extend([{'status':'not_run_after_timeout' if timed_out else 'error','error':'batch-incomplete'} for _ in calls[len(ans):]])
        if len(ans)>len(calls): return [{'status':'error','error':'batch-worker-wrong-count'} for _ in calls]
        return ans

def solve_method_deadline(ins,method,limit=DEADLINE_SECONDS):
    """Real wall deadline: each method is isolated in a fresh worker process."""
    with tempfile.TemporaryDirectory(prefix='riovspt_') as d:
        inp=Path(d)/'input.json'; out=Path(d)/'output.json'; inp.write_text(json.dumps(instance_record(ins)),encoding='utf-8')
        cmd=[sys.executable,str(HERE/'study_runner.py'),'--scipy-path',str(_sp),'--worker',str(inp),str(out),method]; start=time.perf_counter()
        try: subprocess.run(cmd,cwd=str(HERE),timeout=limit,check=False,capture_output=True,text=True)
        except subprocess.TimeoutExpired: return {'status':'timeout','wall_seconds':time.perf_counter()-start}
        wall=time.perf_counter()-start
        if not out.exists(): return {'status':'error','wall_seconds':wall,'error':'worker-no-output'}
        result=json.loads(out.read_text(encoding='utf-8')); result['wall_seconds']=wall
        return result

def correctness_plan():
    rows=[]
    for shape in SHAPES:
        for m in (1,5,20,100):
            for seed_index in range(20):
                actual_seed=100000+m*1000+seed_index+17*SHAPES.index(shape); rows.append((shape,m,seed_index,actual_seed,correctness_random(shape,m,actual_seed)))
    for i,ins in enumerate(named_cases()): rows.append(('named',len(ins.parent),i,None,ins))
    return rows

def performance_plan():
    rows=[]
    for shape in SHAPES:
        for m in (100,1000,10000):
            for seed in range(5):
                mode='down' if seed%2==0 else 'up'; actual_seed=200000+m*100+seed+31*SHAPES.index(shape); rows.append((shape,m,seed,actual_seed,mode,structured_performance(shape,m,actual_seed,mode)))
    return rows

def sensitivity_plan():
    rows=[]
    factors=('costspread','boundwidth','targetdepth')
    for factor in factors:
        for level in range(4):
            for seed_index in range(5):
                # Pair the generated tree and costs across levels of a factor.
                actual_seed=300000+10000*factors.index(factor)+seed_index
                rows.append((factor,level,seed_index,actual_seed,sensitivity_instance(factor,level,actual_seed)))
    return rows

def large_plan():
    rows=[]
    for shape in SHAPES:
        for seed_index in range(3):
            actual_seed=400000+31*SHAPES.index(shape)+seed_index
            mode='down' if seed_index%2==0 else 'up'
            rows.append((shape,100000,seed_index,actual_seed,mode,structured_performance(shape,100000,actual_seed,mode)))
    return rows

def write_plan(out):
    cp=correctness_plan(); pp=performance_plan(); sp=sensitivity_plan(); xp=large_plan()
    plan={'correctness':{'random_instances':400,'manual_instances':len(named_cases()),'records':[{'family':a,'edges':b,'seed_index':c,'actual_seed':d,'label':e.label} for a,b,c,d,e in cp]},'performance':{'instances':75,'records':[{'family':a,'edges':b,'seed_index':c,'actual_seed':d,'mode':e,'label':f.label} for a,b,c,d,e,f in pp]},'large':{'instances':15,'records':[{'family':a,'edges':b,'seed_index':c,'actual_seed':d,'mode':e,'label':f.label} for a,b,c,d,e,f in xp]},'sensitivity':{'instances':60,'records':[{'factor':a,'level':b,'seed_index':c,'actual_seed':d,'label':e.label} for a,b,c,d,e in sp]},'timing':'Serial execution. Algorithm functions are warmed once then measured three times; function_seconds excludes process/IPC startup. A batch guard retains completed calls; a completed call over 30 seconds is limit_exceeded, while calls never begun after a batch timeout are not_run_after_timeout. LP records model-build and linprog wrapper wall time, with SciPy HiGHS time_limit=30 seconds.'}
    plan['timing']='Formal runs use --in-process, serially. Each timed method has one warmup and three measurements. Algorithm timings include internal preprocessing and exclude input generation, independent checking, and process/IPC startup. The 30-second algorithm limit is soft: a completed over-limit call is marked limit_exceeded. HiGHS uses one thread and an internal 30-second limit; LP end-to-end time includes model construction and the linprog wrapper call. The retained child-worker mode is diagnostic only.'
    Path(out).mkdir(parents=True,exist_ok=True); (Path(out)/'study_plan.json').write_text(json.dumps(plan,indent=2),encoding='utf-8')

def method_record(ins,method,run):
    # Instance reconstruction is deterministic from family/seed/label; never
    # repeat the O(m)-sized instance or solution vector in every CSV row.
    rec={'method':method,'method_status':run.get('status'),'function_seconds':run.get('function_seconds'),'reported_objective':run.get('objective'),'branch':run.get('branch'),'method_error':run.get('error')}
    if run.get('status')=='optimal': rec.update(validate_solution(ins,run.get('x',[]),float(run['objective'])))
    else: rec.update({'valid':None,'max_violation':None,'actual_objective':None,'budget_gap':None})
    return rec

def write_csv(path,rows):
    if not rows: return
    keys=[]
    for row in rows:
        for k in row:
            if k not in keys: keys.append(k)
    with Path(path).open('w',newline='',encoding='utf-8') as f: w=csv.DictWriter(f,fieldnames=keys); w.writeheader(); w.writerows(rows)

def _lp_checked(ins,solver=None):
    lp=(solver or solve_lp)(ins); check=validate_solution(ins,lp['x'],lp['objective']) if lp['status']=='optimal' else {'valid':None,'max_violation':None,'actual_objective':None,'budget_gap':None}
    # Do not emit LP x: at m=100000 it would make the study CSV enormous.
    lp.pop('x',None)
    return lp|{'lp_validation_valid':check['valid'],'lp_max_violation':check['max_violation'],'lp_actual_objective':check['actual_objective'],'lp_budget_gap':check['budget_gap']}

def run_correctness(out):
    rows=[]
    for shape,m,seed_index,actual_seed,ins in correctness_plan():
        base={'family':shape,'edges':m,'seed_index':seed_index,'actual_seed':actual_seed,'label':ins.label}; lp=_lp_checked(ins); rows.append(base|{'method':'lp'}|lp); rows.append(base|{'method':'gurobi'}|_lp_checked(ins,solve_gurobi))
        for method,run in zip(('solve_uniform','solve_cases'),solve_methods_batch(ins,['solve_uniform','solve_cases'])):
            rec=base|method_record(ins,method,run)
            if lp['status']=='optimal' and run.get('status')=='optimal': rec['reported_lp_error']=abs(run['objective']-lp['objective']); rec['actual_lp_error']=abs(rec['actual_objective']-lp['objective']); rec['correctness_pass']=bool(rec['valid'] and rec['reported_lp_error']<=TOL and rec['actual_lp_error']<=TOL)
            elif lp['status']=='infeasible': rec['reported_lp_error']=None; rec['actual_lp_error']=None; rec['correctness_pass']=(run.get('status')=='infeasible')
            else: rec['reported_lp_error']=None; rec['actual_lp_error']=None; rec['correctness_pass']=False
            rows.append(rec)
    write_csv(Path(out)/'study_correctness.csv',rows)

def run_performance(out):
    rows=[]
    for shape,m,seed_index,actual_seed,mode,ins in performance_plan():
        base={'family':shape,'edges':m,'seed_index':seed_index,'actual_seed':actual_seed,'mode':mode,'label':ins.label}
        # One warmup, then three timed serial calls. All method initialization is inside each call.
        calls=['solve_uniform']*4+['solve_cases']*4; ans=solve_methods_batch(ins,calls)
        for method,offset in (('solve_uniform',0),('solve_cases',4)):
            for rep,run in enumerate(ans[offset+1:offset+4]): rows.append(base|{'method':method,'repeat':rep}|method_record(ins,method,run))
        # LP is also retained for all sizes; any 30s status is recorded, not discarded.
        solve_lp(ins) # warmup; excluded from timing results
        for rep in range(3): rows.append(base|{'method':'lp','repeat':rep}|_lp_checked(ins))
        solve_gurobi(ins) # warmup
        for rep in range(3): rows.append(base|{'method':'gurobi','repeat':rep}|_lp_checked(ins,solve_gurobi))
    write_csv(Path(out)/'study_performance.csv',rows)

def run_sensitivity(out):
    rows=[]
    for factor,level,seed_index,actual_seed,ins in sensitivity_plan():
        base={'factor':factor,'level':level,'edges':len(ins.parent),'seed_index':seed_index,'actual_seed':actual_seed,'label':ins.label}
        ans=solve_methods_batch(ins,['solve_uniform']*4+['solve_cases']*4)
        for method,offset in (('solve_uniform',0),('solve_cases',4)):
            for rep,run in enumerate(ans[offset+1:offset+4]): rows.append(base|{'method':method,'repeat':rep}|method_record(ins,method,run))
        solve_lp(ins)
        for rep in range(3): rows.append(base|{'method':'lp','repeat':rep}|_lp_checked(ins))
        solve_gurobi(ins) # warmup
        for rep in range(3): rows.append(base|{'method':'gurobi','repeat':rep}|_lp_checked(ins,solve_gurobi))
    write_csv(Path(out)/'study_sensitivity.csv',rows)

def run_large(out):
    rows=[]
    for shape,m,seed_index,actual_seed,mode,ins in large_plan():
        base={'family':shape,'edges':m,'seed_index':seed_index,'actual_seed':actual_seed,'mode':mode,'label':ins.label}
        ans=solve_methods_batch(ins,['solve_uniform']*4+['solve_cases']*4)
        for method,offset in (('solve_uniform',0),('solve_cases',4)):
            for rep,run in enumerate(ans[offset+1:offset+4]): rows.append(base|{'method':method,'repeat':rep}|method_record(ins,method,run))
        # HiGHS stays out of scope at 100000 edges.  Gurobi is the LP oracle
        # here: a single validation call with a longer limit, no warmup.
        rows.append(base|{'method':'gurobi','repeat':0}|_lp_checked(ins,lambda z:solve_gurobi(z,LARGE_LIMIT)))
    write_csv(Path(out)/'study_large.csv',rows)

def main():
    global IN_PROCESS
    p=argparse.ArgumentParser(); p.add_argument('--worker',nargs=3); p.add_argument('--batch-worker',nargs=2); p.add_argument('--scipy-path'); p.add_argument('--in-process',action='store_true'); p.add_argument('--write-plan',action='store_true'); p.add_argument('--run-correctness',action='store_true'); p.add_argument('--run-performance',action='store_true'); p.add_argument('--run-sensitivity',action='store_true'); p.add_argument('--run-large',action='store_true'); p.add_argument('--out',type=Path,default=HERE); a=p.parse_args()
    IN_PROCESS=a.in_process
    if a.worker: return _worker(*a.worker)
    if a.batch_worker: return _batch_worker(*a.batch_worker)
    if a.run_correctness: run_correctness(a.out)
    if a.run_performance: run_performance(a.out)
    if a.run_sensitivity: run_sensitivity(a.out)
    if a.run_large: run_large(a.out)
    if a.write_plan: write_plan(a.out)
    if not (a.write_plan or a.run_correctness or a.run_performance or a.run_sensitivity or a.run_large): print('Plans ready; no formal experiment launched.')
if __name__=='__main__': main()
