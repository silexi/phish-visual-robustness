import numpy as np, json, re, sys
from pathlib import Path
from collections import Counter
from PIL import Image
R=Path(__file__).resolve().parent
cmap=json.load(open(R/"clusters_f.json"))["clusters"]
ids=[str(x) for x in np.load(R/"feat_A0.npz",allow_pickle=True)["ids"]]
print("=== 24. dedup threshold actually used ===")
def dhash(p,size=16):
    im=Image.open(p).convert("L").resize((size+1,size),Image.Resampling.LANCZOS)
    a=np.asarray(im,dtype=np.int16); return (a[:,1:]>a[:,:-1]).ravel().astype(np.int8)
B=np.stack([dhash(R/"clean_sub"/f"{s}.png") for s in ids])
def nclust(t):
    n=len(B); par=list(range(n))
    def f(x):
        while par[x]!=x: par[x]=par[par[x]]; x=par[x]
        return x
    D=(B[:,None,:]!=B[None,:,:]).sum(-1)
    I,J=np.where(np.triu(D<=t,1))
    for i,j in zip(I,J):
        a,b=f(int(i)),f(int(j))
        if a!=b: par[max(a,b)]=min(a,b)
    lab=[f(i) for i in range(n)]; c=Counter(lab)
    return len(c),sorted(c.values(),reverse=True)[:6],sum(1 for v in c.values() if v==1)
for t in [8,12,16,20,24,32]:
    k,sz,si=nclust(t); print(f"  maxdist={t:3d} -> {k} clusters, top {sz}, singletons {si}")
print("  paper/clusters_f.json has 172 clusters, top [129,107,31,30,25,17], singletons 70")
print()
print("=== 25. BRAND COVERAGE: page-weighted vs cluster-weighted ===")
def norm(b):
    if b is None: return None
    b=re.sub(r"[^a-z0-9]+","",str(b).lower()); return b or None
cl={}
for line in open(R/"vlm/desc_A0.jsonl"):
    r=json.loads(line); pa=r.get("parsed"); sid=Path(r["image"]).stem
    cl[sid]=None if not pa else norm(pa.get("brand"))
ok=[i for i,v in cl.items()]
named=[i for i in ok if cl[i]]
print(f"pages {len(ok)}, named {len(named)} = {len(named)/len(ok):.3f} (paper: 148/196=.755 or 148/200=.740)")
g=Counter(cmap.get(i,i) for i in ok); gn=Counter(cmap.get(i,i) for i in named)
rates=[gn.get(k,0)/v for k,v in g.items()]
print(f"clusters {len(g)}, cluster-weighted naming rate {np.mean(rates):.3f}")
print(f"brands among named: {Counter(cl[i] for i in named).most_common(8)}")
print(f"share of named pages that are meta/facebook: {sum(1 for i in named if cl[i] in (chr(109)+chr(101)+chr(116)+chr(97),chr(102)+chr(97)+chr(99)+chr(101)+chr(98)+chr(111)+chr(111)+chr(107)))/len(named):.3f}")
print()
print("=== 26. WHAT S1 ACTUALLY HIDES (logo_els per page) ===")
le=[json.loads(l) for l in open(R/"pert/manifest_S1.jsonl")]
v=np.array([r.get("logo_els",0) or 0 for r in le])
print(f"logo_els: median={np.median(v)} mean={v.mean():.1f} max={v.max()} zero={int((v==0).sum())} >5={int((v>5).sum())}")
ap=np.array([bool(r.get("applied")) for r in le])
print(f"applied where logo_els==0: {int((ap&(v==0)).sum())}   not-applied where logo_els>0: {int((~ap&(v>0)).sum())}")
