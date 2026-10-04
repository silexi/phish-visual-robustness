import numpy as np, json
from pathlib import Path
from collections import Counter
R=Path(__file__).resolve().parent
Z=dict(np.load(R/"audit_hits.npz",allow_pickle=True))
clean_ids=[str(x) for x in Z["clean_ids"]]
cmap=json.load(open(R/"clusters_f.json"))["clusters"]
clean_cl=np.array([cmap.get(s,s) for s in clean_ids])
size=Counter(clean_cl.tolist())
big=[k for k,v in size.most_common(2)]
RUNGS=["P_color","P5_triv8","P3_hash","P_ssim","P0_fusion","P_lbp","P1_repaired","E1_clip","P4_hog"]
CONDS=["A0n","A1","A3","A4","S1","S3","S7"]
def get(tag,rn):
    ids=[str(x) for x in Z[tag+"__ids"]]
    top1=Z[f"{tag}__top1__{rn}"]
    qcl=np.array([cmap.get(s,s) for s in ids])
    return ids,qcl,(clean_cl[top1]==qcl).astype(float)
def tagof(c): return f"{c}_all" if c.startswith("A0") else f"{c}_filt"

print("=== 3. PAGE-WEIGHTED (paper) vs CLUSTER-WEIGHTED (mean of per-cluster means) ===")
print("cond           "+"".join(f"{r:>12}" for r in RUNGS))
for c in CONDS:
    tag=tagof(c); pw=[];cw=[]
    for r in RUNGS:
        ids,qcl,h=get(tag,r); pw.append(h.mean())
        ks=sorted(set(qcl.tolist())); cw.append(np.mean([h[qcl==k].mean() for k in ks]))
    print(f"{c:<5} page  "+"".join(f"{v:12.4f}" for v in pw))
    print(f"{c:<5} clust "+"".join(f"{v:12.4f}" for v in cw))

print()
print("=== 4. SHARE OF ALL HITS CONTRIBUTED BY THE 2 GIANT KITS (%) ===")
print(f"giant clusters {big} sizes {[size[k] for k in big]}")
print("cond           "+"".join(f"{r:>12}" for r in RUNGS))
for c in CONDS:
    tag=tagof(c); row=[];row2=[]
    for r in RUNGS:
        ids,qcl,h=get(tag,r)
        m=np.isin(qcl,big)
        tot=h.sum(); row.append(100*h[m].sum()/max(tot,1e-9))
        row2.append(h[m].sum())
    print(f"{c:<5} %hits "+"".join(f"{v:12.1f}" for v in row))
print()
print("=== per-giant-cluster hit rates ===")
for c in ["A3","S1","S7","A4"]:
    tag=tagof(c)
    for r in ["P3_hash","P0_fusion","P4_hog","E1_clip","P_lbp"]:
        ids,qcl,h=get(tag,r)
        s=" ".join(f"{k}:{h[qcl==k].mean():.3f}(n={int((qcl==k).sum())})" for k in big)
        print(f"{c:<4}{r:<12}{s}   rest={h[~np.isin(qcl,big)].mean():.3f}")
