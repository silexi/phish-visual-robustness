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
app={}
for c in CONDS:
    d={}
    for l in open(R/f"pert/manifest_{c}.jsonl"):
        r=json.loads(l); d[r["safe"]]=bool(r.get("applied"))
    app[c]=d
print("=== 5. WHICH CLUSTERS DO THE NOT-APPLIED CELLS COME FROM? ===")
for c in CONDS:
    bad=[s for s in clean_ids if not app[c].get(s,False)]
    cc=Counter(cmap.get(s,s) for s in bad)
    print(f"{c}: {len(bad)} not-applied over {len(cc)} clusters; top: {cc.most_common(5)}")
print()
print("=== 6. INTENT-TO-TREAT: all 833 scored, no-op cells scored as they actually fall ===")
print("cond        "+"".join(f"{r:>12}" for r in RUNGS))
for c in CONDS:
    ta=f"{c}_all"
    row=[]
    for r in RUNGS:
        ids=[str(x) for x in Z[ta+"__ids"]]; top1=Z[f"{ta}__top1__{r}"]
        qcl=np.array([cmap.get(s,s) for s in ids]); h=(clean_cl[top1]==qcl).astype(float)
        row.append(h.mean())
    print(f"{c:<5} ITT  "+"".join(f"{v:12.4f}" for v in row))
print()
print("=== 7. COMMON SUBSET: pages where EVERY condition applied ===")
common=[s for s in clean_ids if all(app[c].get(s,False) for c in ["A1","A3","A4","S1","S3","S7"])]
print("n_common",len(common),"clusters",len({cmap.get(s,s) for s in common}))
print("cond        "+"".join(f"{r:>12}" for r in RUNGS))
for c in CONDS:
    ta=f"{c}_all"; ids=[str(x) for x in Z[ta+"__ids"]]
    idx=np.array([i for i,s in enumerate(ids) if s in set(common)])
    row=[];row2=[]
    for r in RUNGS:
        top1=Z[f"{ta}__top1__{r}"]; qcl=np.array([cmap.get(s,s) for s in ids])
        h=(clean_cl[top1]==qcl).astype(float)[idx]; q2=qcl[idx]
        row.append(h.mean()); ks=sorted(set(q2.tolist())); row2.append(np.mean([h[q2==k].mean() for k in ks]))
    print(f"{c:<5} page "+"".join(f"{v:12.4f}" for v in row))
    print(f"{c:<5} clus "+"".join(f"{v:12.4f}" for v in row2))
print()
print("=== 8. TIES in the null condition A0n (how many clean pages share the row max) ===")
for r in RUNGS:
    nt=Z[f"A0n_all__nties__{r}"]; ss=Z[f"A0n_all__selfsim__{r}"]; mx=Z[f"A0n_all__maxsim__{r}"]
    ids=[str(x) for x in Z["A0n_all__ids"]]; top1=Z[f"A0n_all__top1__{r}"]
    qcl=np.array([cmap.get(s,s) for s in ids]); h=(clean_cl[top1]==qcl)
    tie_at_max=(ss>=mx-1e-12)
    print(f"{r:<12} median_nties={np.median(nt):5.0f} mean={nt.mean():7.1f} frac_rows_with_ties={ (nt>1).mean():.3f} "
          f"self_is_at_max={tie_at_max.mean():.4f} misses={int((~h).sum())} of_which_self_tied_at_max={int((tie_at_max&~h).sum())}")
