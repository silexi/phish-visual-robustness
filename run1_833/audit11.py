import numpy as np, json
from pathlib import Path
from collections import Counter
R=Path(__file__).resolve().parent
Z=dict(np.load(R/"audit_hits.npz",allow_pickle=True))
clean_ids=[str(x) for x in Z["clean_ids"]]
cmap=json.load(open(R/"clusters_f.json"))["clusters"]
ccl=np.array([cmap.get(s,s) for s in clean_ids]); p2={s:i for i,s in enumerate(clean_ids)}
sz=Counter(ccl.tolist())
print("median cluster size (over 172 clusters):",np.median(sorted(Counter(cmap.values()).values())))
RUNGS=["P_color","P5_triv8","P3_hash","P_ssim","P0_fusion","P_lbp","P1_repaired","E1_clip","P4_hog"]
print()
print("=== 27. RECOMMENDED REPLACEMENT TABLE: cluster-weighted, applied-only, tie-tolerant (1e-4) ===")
print("cond        "+"".join(f"{r:>12}" for r in RUNGS))
for c in ["A0n","A1","A3","A4","S1","S3","S7"]:
    tag=f"{c}_all" if c.startswith("A0") else f"{c}_filt"
    ids=[str(x) for x in Z[tag+"__ids"]]; qi=np.array([p2[s] for s in ids]); cl=ccl[qi]
    ks=sorted(set(cl.tolist())); row=[]
    for r in RUNGS:
        h=(ccl[Z[f"{tag}__top1__{r}"]]==cl).astype(float)
        h=np.maximum(h,((Z[f"{tag}__maxsim__{r}"]-Z[f"{tag}__selfsim__{r}"])<1e-4).astype(float))
        row.append(np.mean([h[cl==k].mean() for k in ks]))
    print(f"{c:<5}{len(ks):<4} "+"".join(f"{v:12.4f}" for v in row))
print()
print("=== 28. ABSTRACT CLAIM CHECK: A4 decoy, loss vs A0n per rung ===")
base={}
tag="A0n_all"; ids=[str(x) for x in Z[tag+"__ids"]]; qi=np.array([p2[s] for s in ids]); cl=ccl[qi]
for r in RUNGS: base[r]=(ccl[Z[f"{tag}__top1__{r}"]]==cl).astype(float).mean()
tag="A4_filt"; ids=[str(x) for x in Z[tag+"__ids"]]; qi=np.array([p2[s] for s in ids]); cl=ccl[qi]
for r in RUNGS:
    v=(ccl[Z[f"{tag}__top1__{r}"]]==cl).astype(float).mean()
    print(f"  {r:<12} A0n={base[r]:.3f} A4={v:.3f} loss={base[r]-v:+.3f}")
