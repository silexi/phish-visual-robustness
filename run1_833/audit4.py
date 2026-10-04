import numpy as np, json, hashlib
from pathlib import Path
from collections import Counter
R=Path(__file__).resolve().parent
cl=dict(np.load(R/"feat_A0.npz",allow_pickle=True))
ids=[str(x) for x in cl["ids"]]
cmap=json.load(open(R/"clusters_f.json"))["clusters"]
meta=json.load(open(R/"gated_all_meta.json"))
ccl=np.array([cmap.get(s,s) for s in ids])
size=Counter(ccl.tolist())

print("=== 9. EXACT DUPLICATES among the 833 clean renders ===")
for key in ["px_gray256","clip","px_triv8","px_color"]:
    A=cl[key].reshape(len(ids),-1)
    hh=[hashlib.sha1(np.ascontiguousarray(a).tobytes()).hexdigest() for a in A]
    c=Counter(hh)
    print(f"{key:<12} distinct={len(c):4d}/{len(ids)}  pages_with_a_duplicate={sum(v for v in c.values() if v>1):4d} "
          f"largest_dup_group={max(c.values())}")
A=cl["clip"]; g=Counter([hashlib.sha1(np.ascontiguousarray(a).tobytes()).hexdigest() for a in A])
print("  clip dup-group size histogram:", sorted(Counter(g.values()).items()))

print()
print("=== 10. ARE THE GIANT CLUSTERS REAL KITS? (dhash single-linkage chaining) ===")
from PIL import Image
def dhash(p,size=16):
    im=Image.open(p).convert("L").resize((size+1,size),Image.Resampling.LANCZOS)
    a=np.asarray(im,dtype=np.int16); return (a[:,1:]>a[:,:-1]).ravel().astype(np.int8)
src=None
for cand in ["clean_sub","clean_f/png","clean_f","pixel"]:
    d=R/cand
    if d.exists() and (d/f"{ids[0]}.png").exists(): src=d;break
print("clean png dir:",src)
for k in [kk for kk,_ in size.most_common(4)]+["k9"]:
    mem=[s for s in ids if cmap.get(s)==k]
    B=np.stack([dhash(src/f"{s}.png") for s in mem])
    D=(B[:,None,:]!=B[None,:,:]).sum(-1)
    doms={meta[s]["domain"] for s in mem if s in meta}
    tit=Counter(meta[s]["title"][:40] for s in mem if s in meta)
    print(f"{k:<6} n={len(mem):4d} dhash-dist: max={D.max():3d} mean={D[np.triu_indices(len(mem),1)].mean():5.1f} "
          f"frac_pairs_within_16={(D[np.triu_indices(len(mem),1)]<=16).mean():.3f} domains={len(doms)} titles={len(tit)}")
    print(f"       top titles: {tit.most_common(3)}")

print()
print("=== 11. RE-CLUSTER FROM CLIP (non-pixel grouping) and re-score ===")
Z=dict(np.load(R/"audit_hits.npz",allow_pickle=True))
E=cl["clip"]; S=E@E.T
def linkage(S,t):
    n=len(S); par=list(range(n))
    def f(x):
        while par[x]!=x: par[x]=par[par[x]]; x=par[x]
        return x
    I,J=np.where(np.triu(S>=t,1))
    for i,j in zip(I,J):
        a,b=f(int(i)),f(int(j))
        if a!=b: par[max(a,b)]=min(a,b)
    lab=np.array([f(i) for i in range(n)])
    return lab
lo,hi=0.5,0.9999
for _ in range(40):
    t=(lo+hi)/2; lab=linkage(S,t); nk=len(set(lab.tolist()))
    if nk<172: lo=t
    else: hi=t
lab=linkage(S,(lo+hi)/2); nk=len(set(lab.tolist()))
cs=Counter(lab.tolist())
print(f"CLIP threshold={(lo+hi)/2:.4f} -> {nk} clusters; sizes {sorted(cs.values(),reverse=True)[:8]}; singletons {sum(1 for v in cs.values() if v==1)}")
clip_cl=np.array([f"c{v}" for v in lab])
dom_cl=np.array([meta[s]["domain"] if s in meta else s for s in ids])
print("domain grouping ->",len(set(dom_cl.tolist())),"groups; sizes",sorted(Counter(dom_cl.tolist()).values(),reverse=True)[:8])
p2={s:i for i,s in enumerate(ids)}
RUNGS=["P_color","P5_triv8","P3_hash","P_ssim","P0_fusion","P_lbp","P1_repaired","E1_clip","P4_hog"]
for name,grp in [("dhash(paper)",ccl),("CLIP",clip_cl),("domain",dom_cl)]:
    print(f"-- grouping={name}")
    print("cond        "+"".join(f"{r:>12}" for r in RUNGS))
    for c in ["A0n","A3","S1","S7","A4","S3"]:
        tag=f"{c}_all" if c.startswith("A0") else f"{c}_filt"
        qids=[str(x) for x in Z[tag+"__ids"]]; qi=np.array([p2[s] for s in qids])
        rowp=[];rowc=[]
        for r in RUNGS:
            t1=Z[f"{tag}__top1__{r}"]
            h=(grp[t1]==grp[qi]).astype(float); rowp.append(h.mean())
            ks=sorted(set(grp[qi].tolist())); rowc.append(np.mean([h[grp[qi]==k].mean() for k in ks]))
        print(f"{c:<5} page "+"".join(f"{v:12.4f}" for v in rowp))
        print(f"{c:<5} grp  "+"".join(f"{v:12.4f}" for v in rowc))
