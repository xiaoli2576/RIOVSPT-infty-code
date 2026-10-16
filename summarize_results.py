"""Independent reconciliation and descriptive aggregation of study CSV rows."""
import argparse,csv,json,math,statistics
from collections import Counter,defaultdict
from pathlib import Path

def read(p):
    with p.open(encoding='utf-8',newline='') as f:return list(csv.DictReader(f))
def number(r,*fields):
    for key in fields:
        if r.get(key) not in (None,''):return float(r[key])
    return None
SOLVERS=('lp','gurobi')
def status(r):return r.get('status') if r['method'] in SOLVERS else r.get('method_status')
def valid(r):return r.get('lp_validation_valid')=='True' if r['method'] in SOLVERS else r.get('valid')=='True'
def obj(r):return number(r,'objective' if r['method'] in SOLVERS else 'reported_objective')
def actual(r):return number(r,'lp_actual_objective' if r['method'] in SOLVERS else 'actual_objective')
def secs(r):return number(r,'end_to_end_seconds' if r['method'] in SOLVERS else 'function_seconds')
def native(r):return number(r,'gurobi_runtime_seconds' if r['method']=='gurobi' else 'solve_seconds' if r['method']=='lp' else 'function_seconds')
UNRESOLVED=('time_limit_or_iteration',)
def maximum(xs):return max((x for x in xs if x is not None),default=0)
def write(p,rows):
    with p.open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def run(root):
    plan=json.loads((root/'study_plan.json').read_text(encoding='utf-8'))
    nc=plan['correctness']['random_instances']+plan['correctness']['manual_instances']
    # expected rows per instance: methods x repeats (large: 2 algorithms x 3 + one Gurobi oracle call)
    counts={'correctness':(nc,4),'performance':(75,12),'large':(15,7),'sensitivity':(60,12),'branches':(36,12)}
    summary={};medians=[];problems=[]
    for group,(n,per) in counts.items():
        rows=read(root/f'study_{group}.csv')
        expected=n*per
        keys=[(r['label'],r['method'],r.get('repeat','0')) for r in rows]
        duplicates=len(keys)-len(set(keys))
        if len(rows)!=expected or duplicates or len(set(r['label'] for r in rows))!=n:
            raise AssertionError(f'{group} count mismatch: {len(rows)}/{expected}, duplicates={duplicates}')
        by_instance=defaultdict(list)
        for r in rows:by_instance[r['label']].append(r)
        maxerr=0;branch=Counter();passed_instances=0;unresolved=Counter();oracle_checked=0
        for label,rr in by_instance.items():
            known=[r for r in rr if r['method']=='lp'] or [r for r in rr if r['method']=='gurobi' and status(r) not in UNRESOLVED] or [r for r in rr if r['method']=='solve_uniform']
            ref=known[0];ok=True
            if ref['method'] in SOLVERS:oracle_checked+=1
            for r in rr:
                if r['method'] in SOLVERS and status(r) in UNRESOLVED:
                    unresolved[r['method']]+=1;continue
                if group=='branches' and r['method']=='solve_cases' and r['branch']!=r['expected_branch']:ok=False
                if status(r)!=status(ref) or status(r) not in ('optimal','infeasible'):ok=False
                if status(r)=='optimal':
                    if not valid(r):ok=False
                    error=max(abs(obj(r)-obj(ref)),abs(actual(r)-obj(ref)))
                    maxerr=max(maxerr,error)
                    if not math.isfinite(error) or error>2e-6:ok=False
            if ok:passed_instances+=1
            else:problems.append({'group':group,'label':label,'rows':rr})
            case=next((r for r in rr if r['method']=='solve_cases'),None)
            if case:branch[case['branch']]+=1
        summary[group]={'expected_rows':expected,'rows':len(rows),'duplicates':duplicates,'instances':n,
                        'passed_instances':passed_instances,'status_counts':dict(Counter(status(r) for r in rows)),
                        'branch_counts':dict(branch),'max_objective_discrepancy':maxerr,
                        'solver_unresolved_calls':dict(unresolved),'instances_with_solver_reference':oracle_checked,
                        'max_constraint_violation':maximum(number(r,'lp_max_violation','max_violation') for r in rows),
                        'max_function_seconds':maximum(secs(r) for r in rows)}
        if group=='correctness':
            summary[group]['lp_feasible']=sum(r['method']=='lp' and status(r)=='optimal' for r in rows)
            summary[group]['lp_infeasible']=sum(r['method']=='lp' and status(r)=='infeasible' for r in rows)
            summary[group]['gurobi_infeasible']=sum(r['method']=='gurobi' and status(r)=='infeasible' for r in rows)
            continue
        cells=defaultdict(list)
        for r in rows:cells[r['label'],r['method']].append(r)
        for (label,method),rr in cells.items():
            if len(rr)!=(1 if group=='large' and method=='gurobi' else 3):raise AssertionError('missing repetition')
            r=rr[0];good=all(status(z)=='optimal' and valid(z) for z in rr)
            medians.append(dict(group=group,label=label,method=method,family=r.get('family',''),
                edges=r['edges'],factor=r.get('factor',''),level=r.get('level',''),branch=r.get('branch',''),
                median_seconds=statistics.median(secs(z) for z in rr) if good else None,
                median_native_seconds=statistics.median(native(z) for z in rr) if good else None,
                status=status(r),
                objective=obj(r),actual_seed=r.get('actual_seed','')))
    write(root/'instance_medians.csv',medians)
    comparisons={}
    for group in ['performance','large','sensitivity','branches']:
        cells=defaultdict(dict)
        for r in medians:
            if r['group']==group and r['median_seconds'] is not None:cells[r['label']][r['method']]=r['median_seconds']
        pairs=[('solve_cases','solve_uniform'),('gurobi','solve_cases'),('gurobi','solve_uniform')]+([] if group=='large' else [('lp','solve_cases'),('lp','solve_uniform')])
        for a,b in pairs:
            vals=[(label,x[a]/x[b]) for label,x in cells.items() if a in x and b in x]
            ratios=[v for _,v in vals]
            if not ratios:continue
            comparisons[f'{group}:{a}/{b}']={'instances':len(vals),'median_ratio':statistics.median(ratios),
                'min_ratio':min(ratios),'max_ratio':max(ratios),'numerator_faster':sum(v<1 for v in ratios),
                'denominator_faster':sum(v>1 for v in ratios),'ties':sum(v==1 for v in ratios),
                'numerator_faster_labels':[l for l,v in vals if v<1]}
    summary['paired_comparisons']=comparisons
    branch_comparisons={}
    for branch in ['middle','up','hard-zero','hard-positive']:
        paired=defaultdict(dict)
        for r in medians:
            if r['group']=='branches' and r['family']==branch and r['median_seconds'] is not None:
                paired[r['label']][r['method']]=r['median_seconds']
        ratios=[v['solve_uniform']/v['solve_cases'] for v in paired.values() if 'solve_uniform' in v and 'solve_cases' in v]
        branch_comparisons[branch]={'instances':len(ratios),'median_A3_over_A2':statistics.median(ratios),
            'A2_faster':sum(v>1 for v in ratios),'A3_faster':sum(v<1 for v in ratios)}
    summary['branch_runtime_comparisons']=branch_comparisons
    table=[]
    for group in ['performance','large']:
        for fam in ['chain','star','balanced','comb','random']:
            sizes=sorted({int(r['edges']) for r in medians if r['group']==group and r['family']==fam})
            for m in sizes:
                row={'family':fam,'edges':m}
                for method in ['solve_cases','solve_uniform','lp','gurobi']:
                    cell=[r for r in medians if r['group']==group and r['family']==fam and int(r['edges'])==m and r['method']==method]
                    vals=[r['median_seconds'] for r in cell if r['median_seconds'] is not None]
                    row[method+'_median_seconds']=statistics.median(vals) if vals else None
                    row[method+'_instances']=len(vals)
                    if method in ('lp','gurobi'):
                        nat=[r['median_native_seconds'] for r in cell if r['median_native_seconds'] is not None]
                        row[method+'_native_median_seconds']=statistics.median(nat) if nat else None
                        row[method+'_unresolved']=sum(r['status'] in UNRESOLVED for r in cell)
                table.append(row)
    write(root/'runtime_table.csv',table)
    summary['problem_count']=len(problems)
    (root/'verification_problems.json').write_text(json.dumps(problems,indent=2),encoding='utf-8')
    (root/'full_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in summary.items() if k!='paired_comparisons'},indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--results',type=Path,default=Path(__file__).parent/'results');a=p.parse_args();run(a.results)
