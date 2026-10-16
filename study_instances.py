"""Shared deterministic instances for the full RIOVSPT numerical study."""
from __future__ import annotations
from dataclasses import asdict,dataclass
import json,random

@dataclass
class Instance:
    parent:list[int]       # edge i connects parent[i] to node i+1
    w:list[float]; l:list[float]; u:list[float]; c:list[float]
    target:int; D:float; label:str='unnamed'

def children(ins):
    out=[[] for _ in range(len(ins.parent)+1)]
    for i,p in enumerate(ins.parent): out[p].append(i+1)
    return out
def leaves(ins): return [v for v,a in enumerate(children(ins)) if not a]
def path_edges(ins,v):
    ans=[]
    while v: ans.append(v-1); v=ins.parent[v-1]
    return ans[::-1]
def depths(ins):
    d=[0]*(len(ins.parent)+1)
    for i,p in enumerate(ins.parent): d[i+1]=d[p]+1
    return d
def validate(ins):
    m=len(ins.parent)
    if not (len(ins.w)==len(ins.l)==len(ins.u)==len(ins.c)==m): raise ValueError('edge-vector lengths')
    if not all(0<=p<=i for i,p in enumerate(ins.parent)): raise ValueError('parent order')
    if not all(a<=b<=d and q>0 for a,b,d,q in zip(ins.l,ins.w,ins.u,ins.c)): raise ValueError('bounds/cost')
    if ins.target not in leaves(ins): raise ValueError('target must be a leaf')
    return ins
def instance_record(ins): return asdict(ins)
def instance_from_record(d): return validate(Instance(**d))
def dumps(ins): return json.dumps(instance_record(ins),sort_keys=True,separators=(',',':'))

SHAPES=('chain','star','balanced','comb','random')
def shape_parents(m,shape,rng):
    if m<1: return []
    if shape=='chain': return list(range(m))
    if shape=='star': return [0]*m
    if shape=='balanced': return [i//2 for i in range(m)]
    if shape=='comb':
        ans=[]; spine=0
        for i in range(m):
            ans.append(spine)
            if i==0 or i%2==0: spine=i+1
        return ans
    if shape=='random': return [rng.randrange(i+1) for i in range(m)]
    raise ValueError(shape)

def correctness_random(shape,m,seed):
    """Unscreened signed instance; D is deliberately drawn beyond likely ranges."""
    r=random.Random(seed); p=shape_parents(m,shape,r)
    w=[r.uniform(-5,5) for _ in range(m)]
    lo=[r.uniform(0,4) for _ in range(m)]; hi=[r.uniform(0,4) for _ in range(m)]
    l=[w[i]-lo[i] for i in range(m)]; u=[w[i]+hi[i] for i in range(m)]
    for i in range(m):
        if r.random()<.12: l[i]=u[i]=w[i]
    c=[10**r.uniform(-1.2,1.2) for _ in range(m)]
    temp=Instance(p,w,l,u,c,0,0,'')
    target=r.choice(leaves(temp)); q=path_edges(temp,target)
    # Do not use any feasibility theorem here. This broad D draw preserves all
    # feasible and infeasible outcomes for independent algorithm/LP checking.
    center=sum(w[e] for e in q); spread=max(3.0,sum(abs(u[e]-l[e]) for e in q)+2.0)
    D=center+r.uniform(-1.25*spread,1.25*spread)
    return validate(Instance(p,w,l,u,c,target,D,f'correctness-{shape}-m{m}-s{seed}'))

def named_cases():
    """Small hand cases: singleton, endpoint, shared-prefix and signed/boundary cases."""
    raw=[
      ([],[],[],[],[],0,0.,'singleton-feasible'),
      ([],[],[],[],[],0,1.,'singleton-infeasible'),
      ([0],[2.],[1.],[4.],[3.],1,1.,'one-edge-lower'),
      ([0],[2.],[1.],[4.],[3.],1,4.,'one-edge-upper'),
      ([0,0],[1.,4.],[0.,0.],[7.,7.],[1.,5.],1,3.,'root-split-upward'),
      ([0,0],[4.,1.],[0.,0.],[7.,7.],[2.,1.],1,.5,'root-split-downward'),
      ([0,1,1],[2.,3.,0.],[0.,3.,0.],[10.,3.,1.],[1.,3.,7.],2,3.,'shared-prefix-infeasible'),
      ([0,1,1],[-3.,2.,-1.],[-4.,1.,-3.],[-1.,4.,6.],[.3,7.,2.],2,-3.,'signed-target-lower'),
      ([0,1,1],[-3.,2.,-1.],[-4.,1.,-3.],[-1.,4.,6.],[.3,7.,2.],2,3.,'signed-target-upper'),
      ([0,1,1],[1.,2.,3.],[1.,0.,3.],[1.,5.,3.],[1.,.5,9.],2,3.,'zero-width-edges'),
      ([0,1,1],[0.,0.,0.],[0.,0.,0.],[3.,3.,3.],[1.,.5,1.],2,2.,'positive-Bhat-middle'),
      ([0,1,2,3,3,1],[0.]*6,[0.]*6,[3.,3.,3.,0.,3.,3.],[1.]*6,4,2.,'hard-zero-positive-Bhat'),
      # Regression for the value-function breakpoint claim: B*=2, while the
      # active affine expression switches there; margin search remains valid.
      ([0,1,1],[2.,4.,1.],[-8.,-6.,-9.],[12.,14.,11.],[1.,1.,2.],2,2.,'value-function-breakpoint-regression'),
    ]
    return [validate(Instance(*x)) for x in raw]

def structured_performance(shape,m,seed,mode):
    """Feasible, positive-budget, heterogenous many-breakpoint instance.
    mode='down' lowers the target; mode='up' raises it.  The algorithm's
    reported branch remains the authoritative branch classification.
    """
    r=random.Random(seed); p=shape_parents(m,shape,r)
    w=[1.0+r.random()*2.0 for _ in range(m)]
    l=[0.0]*m; u=[8.0+r.random()*5.0 for _ in range(m)]
    c=[.13+10.0*r.random() for _ in range(m)]
    temp=Instance(p,w,l,u,c,0,0,''); dep=depths(temp); target=min(leaves(temp),key=lambda v:(dep[v],v)); q=path_edges(temp,target)
    base=sum(w[e] for e in q)
    D=max(.25,base-.75) if mode=='down' else base+.75
    # Every root-to-leaf upper length is >=8; selected D is well below this.
    return validate(Instance(p,w,l,u,c,target,D,f'performance-{shape}-m{m}-{mode}-s{seed}'))

def sensitivity_instance(factor,level,seed,m=1000):
    """Feasible-by-witness sensitivity instance; no theorem-based filtering."""
    r=random.Random(seed); p=shape_parents(m,'random',r); temp=Instance(p,[1.]*m,[0.]*m,[2.]*m,[1.]*m,0,0,''); dep=depths(temp)
    lf=sorted(leaves(temp),key=lambda v:(dep[v],v)); quant=(0.,.33,.67,1.)
    q=quant[level] if factor=='targetdepth' else .5; target=lf[round(q*(len(lf)-1))]; path=set(path_edges(temp,target)); d=max(1,dep[target]); D=10.
    width=(.25,.5,1.,2.)[level] if factor=='boundwidth' else 1.
    spread=(0.,1.,2.,3.)[level] if factor=='costspread' else 1.
    w=[(D+.5)/d if i in path else 2*D for i in range(m)]; l=[v-width for v in w]; u=[v+width for v in w]
    c=[10**r.uniform(-spread/2,spread/2) for _ in range(m)]
    # Explicit witness: lower target-path weights by .5/d; any side leaf has
    # an off-path edge of weight 2D and therefore remains at least D.
    return validate(Instance(p,w,l,u,c,target,D,f'sensitivity-{factor}-L{level}-s{seed}'))
