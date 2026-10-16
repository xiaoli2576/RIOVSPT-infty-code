"""Research implementations of manuscript Algorithms 1, 2 and 3.

All input preprocessing is included in public solver calls. Algorithm 2's
auxiliary interdiction problems use monotone upper-bound margins, followed by
the explicit canonical level-cap vector required by the manuscript proof.
No LP solver is used here. Floating point tolerance is 1e-10 in budget tests.
"""
from __future__ import annotations
import heapq
import math

EPS = 1e-10


class Tree:
    def __init__(self, ins):
        self.ins = ins
        self.m = len(ins.parent)
        self.parent = ins.parent
        self.children = [[] for _ in range(self.m + 1)]
        for e, p in enumerate(ins.parent):
            if not 0 <= p <= e:
                raise ValueError('parent array must be in topological order')
            self.children[p].append(e + 1)
        self.leaves = [i for i, ch in enumerate(self.children) if not ch]
        if ins.target not in self.leaves:
            raise ValueError('target must be a leaf')
        for w,l,u,c in zip(ins.w,ins.l,ins.u,ins.c):
            if not all(map(math.isfinite,(w,l,u,c))) or not l<=w<=u or c<=0:
                raise ValueError('invalid weights/bounds/costs')
        if not math.isfinite(ins.D): raise ValueError('nonfinite target')
        self.path = []
        v = ins.target
        while v:
            self.path.append(v-1)
            v=ins.parent[v-1]
        self.path.reverse()
        self.onpath = [False]*self.m
        self.pdepth = [0]*(self.m+1)
        for k,e in enumerate(self.path):
            self.onpath[e]=True
            self.pdepth[e+1]=k+1
        self.merge=[0]*(self.m+1)
        self.gate=[0]*(self.m+1)
        for e,p in enumerate(ins.parent):
            v=e+1
            self.merge[v]=v if self.onpath[e] else self.merge[p]
            self.gate[v]=v if p==0 else self.gate[p]

    def prefix(self, x, edges=None):
        d=[0.0]*(self.m+1)
        for e in range(self.m) if edges is None else edges:
            d[e+1]=d[self.parent[e]]+x[e]
        return d

    def margins(self, L, U, edges=None, leaves=None):
        ins=self.ins
        ls=self.prefix(L,edges); us=self.prefix(U,edges)
        low=ls[ins.target]
        out=[ins.D-low]
        for v in self.leaves if leaves is None else leaves:
            out.append(us[v]-ins.D)
            if v!=ins.target:
                k=self.merge[v]
                out.append((us[v]-us[k])-(low-ls[k]))
        return out

    def bounds(self,B):
        ins=self.ins
        return ([max(l,w-B/c) for l,w,c in zip(ins.l,ins.w,ins.c)],
                [min(u,w+B/c) for u,w,c in zip(ins.u,ins.w,ins.c)])


def threshold(points, evaluate):
    """First attainable budget: binary search edge kinks, affine root inside."""
    bp=sorted(set(points))
    ga=evaluate(bp[0])
    if min(ga,default=0)>=-EPS: return bp[0]
    if min(evaluate(bp[-1]),default=0)<-EPS:
        raise ArithmeticError('last breakpoint is not attainable')
    lo,hi=0,len(bp)-1
    while hi-lo>1:
        mid=(lo+hi)//2
        if min(evaluate(bp[mid]),default=0)>=-EPS: hi=mid
        else: lo=mid
    a,b=bp[lo],bp[hi]
    ga,gb=evaluate(a),evaluate(b)
    roots=[]
    for x,y in zip(ga,gb):
        if x < -EPS:
            if y<=x: raise ArithmeticError('nonincreasing violated margin')
            roots.append(a+(b-a)*(-x)/(y-x))
    if not roots: raise ArithmeticError('empty crossing set')
    return max(roots)


def full_budget(t,edges=None,leaves=None):
    ins=t.ins
    ii=range(t.m) if edges is None else edges
    pts=[0.0]
    for e in ii:
        pts.extend((ins.c[e]*(ins.w[e]-ins.l[e]),ins.c[e]*(ins.u[e]-ins.w[e])))
    return threshold(pts,lambda B:t.margins(*t.bounds(B),edges,leaves))


def result(x,B,branch,**meta):
    return dict(status='optimal',x=x,objective=B,branch=branch,metadata=meta)


def infeasible():
    return dict(status='infeasible',x=[],objective=math.inf,branch='infeasible',metadata={})


def solve_uniform(ins):
    t=Tree(ins)
    if min(t.margins(ins.l,ins.u)) < -EPS: return infeasible()
    if not t.m: return result([],0.0,'singleton')
    B=full_budget(t)
    L,U=t.bounds(B)
    x=U[:]
    remaining=ins.D-math.fsum(L[e] for e in t.path)
    for e in t.path:
        step=min(max(remaining,0.0),U[e]-L[e])
        x[e]=L[e]+step
        remaining-=step
    return result(x,B,'zero' if B==0 else 'threshold',construction_residual=remaining)


def interdiction(t,edges,leaves):
    """Auxiliary MCSPIT: all selected root-leaf paths must reach D."""
    ins=t.ins
    if not leaves: return 0.0
    pts=[0.0]+[ins.c[e]*(ins.u[e]-ins.w[e]) for e in edges]
    def margins(B):
        d=[0.0]*(t.m+1)
        for e in edges:
            d[e+1]=d[t.parent[e]]+min(ins.u[e],ins.w[e]+B/ins.c[e])
        return [d[v]-ins.D for v in leaves]
    return threshold(pts,margins)


def spit0(t,edges,side,chi,hat,Bhat,B):
    """Algorithm 1, implicit active-leaf heap and a forward path pointer."""
    ins=t.ins; extra=max(0.0,B-Bhat)
    x=hat[:]
    for e in edges:
        if not t.onpath[e]: x[e]=min(ins.u[e],hat[e]+extra/ins.c[e])
    for e in chi: x[e]=max(ins.l[e],hat[e]-extra/ins.c[e])
    floor=[max(ins.l[e],ins.w[e]-B/ins.c[e]) for e in t.path]
    dist=t.prefix(x,edges)
    heap=[(dist[v],v,t.pdepth[t.merge[v]]) for v in side]
    heapq.heapify(heap)
    processed=0; shortening=0.0; witnesses=0
    while heap:
        while heap and heap[0][2]<=processed: heapq.heappop(heap)
        if not heap: break
        initial,leaf,end=heap[0]
        excess=max(0.0,initial-shortening-ins.D)
        removed=0.0
        # Every path edge is visited in one iteration only. After the partial
        # edge the remaining edges in this segment keep their initial values.
        for k in range(processed,end):
            e=t.path[k]
            step=min(max(0.0,excess-removed),max(0.0,x[e]-floor[k]))
            x[e]-=step; removed+=step
        if removed>=excess-EPS: witnesses+=1
        shortening+=removed
        processed=end
    # Once no active side leaf remains the suffix can reach its lower bounds.
    for k in range(processed,len(t.path)): x[t.path[k]]=floor[k]
    return x,math.fsum(x[e] for e in t.path),witnesses


def solve_cases(ins):
    t=Tree(ins)
    if min(t.margins(ins.l,ins.u)) < -EPS: return infeasible()
    if not t.m: return result([],0.0,'singleton')
    gate=t.path[0]+1
    inside=[e for e in range(t.m) if t.gate[e+1]==gate]
    outside=[e for e in range(t.m) if t.gate[e+1]!=gate]
    gleaves=[v for v in t.leaves if t.gate[v]==gate]
    oleaves=[v for v in t.leaves if t.gate[v]!=gate]
    side=[v for v in gleaves if v!=ins.target]
    last_branch=0
    for k,e in enumerate(t.path):
        if len(t.children[t.parent[e]])>1: last_branch=k
    chi=t.path[last_branch:]
    chiset=set(chi)
    auxiliary=[e for e in inside if e not in chiset]
    Boutside=interdiction(t,outside,oleaves)
    Bhat=interdiction(t,auxiliary,side)
    hat=ins.w[:]
    for e in auxiliary: hat[e]=min(ins.u[e],ins.w[e]+Bhat/ins.c[e])
    delta=ins.D-math.fsum(hat[e] for e in t.path)
    low=math.fsum(max(ins.l[e]-ins.w[e],-Bhat/ins.c[e]) for e in chi)
    high=math.fsum(min(ins.u[e]-ins.w[e],Bhat/ins.c[e]) for e in chi)
    witnesses=0
    if low-EPS<=delta<=high+EPS:
        branch='middle'; B=Bhat; x=hat[:]
        remaining=abs(delta)
        direction=1 if delta>=0 else -1
        for e in chi:
            room=(min(ins.u[e]-ins.w[e],Bhat/ins.c[e]) if direction>0
                  else min(ins.w[e]-ins.l[e],Bhat/ins.c[e]))
            step=min(remaining,room); x[e]+=direction*step; remaining-=step
    elif delta>high:
        branch='up'
        pts=[Bhat]+[ins.c[e]*(ins.u[e]-ins.w[e]) for e in t.path
                    if ins.c[e]*(ins.u[e]-ins.w[e])>Bhat]
        B=threshold(pts,lambda b:[math.fsum(min(ins.u[e],ins.w[e]+b/ins.c[e]) for e in t.path)-ins.D])
        x=hat[:]
        for e in t.path: x[e]=min(ins.u[e],ins.w[e]+B/ins.c[e])
    else:
        for e in chi: hat[e]=max(ins.l[e],ins.w[e]-Bhat/ins.c[e])
        x0,V0,witnesses=spit0(t,inside,side,chi,hat,Bhat,Bhat)
        if V0<=ins.D+EPS:
            branch='hard-zero'; B=Bhat
            h0=math.fsum(hat[e] for e in t.path)
            alpha=(h0-ins.D)/(h0-V0) if h0>V0 else 1.0
            alpha=min(1.0,max(0.0,alpha))
            x=[a+alpha*(b-a) for a,b in zip(hat,x0)]
        else:
            branch='hard-positive'; B=full_budget(t,inside,gleaves)
            x,V,witnesses=spit0(t,inside,side,chi,hat,Bhat,B)
    for e in outside: x[e]=min(ins.u[e],ins.w[e]+Boutside/ins.c[e])
    return result(x,max(Boutside,B),branch,auxiliary_budget=Bhat,
                  outside_budget=Boutside,gate_budget=B,path_edges=len(t.path),
                  terminal_edges=len(chi),gate_edges=len(inside),witnesses=witnesses)
