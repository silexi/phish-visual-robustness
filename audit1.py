import numpy as np, json
from pathlib import Path
from collections import Counter
R=Path(__file__).resolve().parent
Z=dict(np.load(R/"audit_hits.npz",allow_pickle=True))
clean_ids=[str(x) for x in Z["clean_ids"]]
cmap=json.load(open(R/"clusters_f.json"))["clusters"]
clean_cl=np.array([cmap.get(s,s) for s in clean_ids])
size=Counter(clean_cl.tolist())
RUNGS=["P_color","P5_triv8","P3_hash","P_ssim","P0_fusion","P_lbp","P1_repaired","E1_clip","P4_hog"]
CONDS=["A0n","A1","A3","A4","S1","S3","S7"]
def get(tag,rn):
    ids=[str(x) for x in Z[tag+"__ids"]]
    top1=Z[f"{tag}__top1__{rn}"]
    qcl=np.array([cmap.get(s,s) for s in ids])
    return ids,qcl,(clean_cl[top1]==qcl).astype(float),(top1==Z[tag+"__truth"]).astype(float)

print("=== 1. REPRODUCTION of Table 1 (filt = author pipeline) ===")
print("cond  n    "+"".join(f"{r:>12}" for r in RUNGS))
for c in CONDS:
    tag=f"{c}_all" if c.startswith("A0") else f"{c}_filt"
    row=[]
    for r in RUNGS:
        _,_,h,_=get(tag,r); row.append(h.mean())
    print(f"{c:<5}{len(Z[tag+chr(95)+chr(95)+chr(105)+chr(100)+chr(115)])!s:<5}"+"".join(f"{v:12.4f}" for v in row))

print()
print("=== 2. CLUSTER-SIZE STRATIFIED cluster-rank1 (author filt subsets) ===")
bands=[(1,1),(2,4),(5,20),(21,200)]
for c in ["A0n","A3","A4","S1","S3","S7"]:
    tag=f"{c}_all" if c.startswith("A0") else f"{c}_filt"
    print(f"-- {c}")
    print(f"   {chr(39)}band{chr(39)}    n  "+"".join(f"{r:>12}" for r in RUNGS))
    for lo,hi in bands:
        ids,qcl,_,_=get(tag,RUNGS[0])
        m=np.array([lo<=size[x]<=hi for x in qcl])
        if m.sum()==0: continue
        row=[]
        for r in RUNGS:
            _,_,h,_=get(tag,r); row.append(h[m].mean())
        print(f"   {lo}-{hi:<6}{int(m.sum()):<5}"+"".join(f"{v:12.4f}" for v in row))
    # exact-match
    row=[]
    for r in RUNGS:
        _,_,_,e=get(tag,r); row.append(e.mean())
    print(f"   EXACT  {len(qcl):<5}"+"".join(f"{v:12.4f}" for v in row))
