"""Draw the five test-tree families on 14 edges (figure for the paper)."""
import argparse,random
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from study_instances import SHAPES,shape_parents

NAMES={'chain':'chain','star':'star','balanced':'balanced binary','comb':'comb','random':'random recursive'}

def layout(parent):
    """Tidy layered layout: leaves get consecutive x slots, parents are centred."""
    n=len(parent)+1; ch=[[] for _ in range(n)]
    for i,p in enumerate(parent): ch[p].append(i+1)
    x=[0.0]*n; y=[0]*n; slot=[0]
    def visit(v,d):
        y[v]=-d
        if not ch[v]: x[v]=slot[0]; slot[0]+=1; return
        for c in ch[v]: visit(c,d+1)
        x[v]=(x[ch[v][0]]+x[ch[v][-1]])/2
    visit(0,0); return x,y,ch

def target_leaf(parent):
    """Shallowest leaf, smallest index: the target used by the timing generator."""
    n=len(parent)+1; dep=[0]*n
    for i,p in enumerate(parent): dep[i+1]=dep[p]+1
    has=set(parent); lv=[v for v in range(n) if v not in has]
    return min(lv,key=lambda v:(dep[v],v))

def main(out,m=14):
    fig,axes=plt.subplots(1,5,figsize=(11,2.9))
    for ax,shape in zip(axes,SHAPES):
        parent=shape_parents(m,shape,random.Random(7)); x,y,ch=layout(parent); t=target_leaf(parent)
        path=set(); v=t
        while v: path.add(v); v=parent[v-1]
        for i,p in enumerate(parent):
            on=(i+1) in path
            ax.plot([x[p],x[i+1]],[y[p],y[i+1]],color='#c0392b' if on else '#555555',lw=2.2 if on else 1.0,zorder=1)
        leaves=[v for v in range(m+1) if not ch[v]]
        ax.scatter([x[v] for v in range(m+1)],[y[v] for v in range(m+1)],s=16,color='#222222',zorder=2)
        ax.scatter([x[v] for v in leaves],[y[v] for v in leaves],s=26,facecolor='white',edgecolor='#222222',zorder=3)
        ax.scatter([x[0]],[y[0]],s=40,marker='s',color='#222222',zorder=3)
        ax.scatter([x[t]],[y[t]],s=44,color='#c0392b',zorder=4)
        ax.set_title(NAMES[shape],fontsize=10); ax.axis('off')
    fig.tight_layout(); out.mkdir(parents=True,exist_ok=True)
    for ext in ('pdf','png'): fig.savefig(out/f'tree_families.{ext}',dpi=200,bbox_inches='tight')

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--out',type=Path,default=Path(__file__).parent/'figures'); a=p.parse_args(); main(a.out)
