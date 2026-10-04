import numpy as np, json
from pathlib import Path
R=Path(__file__).resolve().parent
Z=dict(np.load(R/"audit_hits.npz",allow_pickle=True))
clean_ids=[str(x) for x in Z["clean_ids"]]
cmap=json.load(open(R/"clusters_f.json"))["clusters"]
ccl=np.array([cmap.get(s,s) for s in clean_ids]); p2={s:i for i,s in enumerate(clean_ids)}
print("=== 19. ARE THE MISSES DECIDED BY FLOAT NOISE?  gap = max_sim - self_sim over misses ===")
print(f"{chr(39)}cond rung{chr(39):<20}  misses  frac_gap<1e-4  frac_gap<1e-2   median_gap")
for c in ["A0n","A1","A4","S7","S3","S1","A3"]:
    tag=f"{c}_all" if c.startswith("A0") else f"{c}_filt"
    ids=[str(x) for x in Z[tag+"__ids"]]; qi=np.array([p2[s] for s in ids])
    for r in ["P_color","P_lbp","P3_hash","P4_hog","E1_clip","P5_triv8"]:
        t1=Z[f"{tag}__top1__{r}"]; h=(ccl[t1]==ccl[qi])
        ss=Z[f"{tag}__selfsim__{r}"]; mx=Z[f"{tag}__maxsim__{r}"]
        g=(mx-ss)[~h]
        if len(g)==0:
            print(f"{c:<5}{r:<14} 0 misses"); continue
        print(f"{c:<5}{r:<14}{len(g):7d}  {np.mean(g<1e-4):13.3f}  {np.mean(g<1e-2):13.3f}   {np.median(g):.3e}")
print()
print("=== 20. COLOUR RUNG RESCORED WITH A 1e-4 TOLERANCE (tie -> credit if own cluster is at the max) ===")
for c in ["A0n","A1","A4","S7","S3","S1","A3"]:
    tag=f"{c}_all" if c.startswith("A0") else f"{c}_filt"
    ids=[str(x) for x in Z[tag+"__ids"]]; qi=np.array([p2[s] for s in ids])
    out=[]
    for r in ["P_color","P_lbp","P3_hash","P4_hog","E1_clip"]:
        t1=Z[f"{tag}__top1__{r}"]; h=(ccl[t1]==ccl[qi]).astype(float)
        ss=Z[f"{tag}__selfsim__{r}"]; mx=Z[f"{tag}__maxsim__{r}"]
        h2=np.maximum(h,((mx-ss)<1e-4).astype(float))
        out.append(f"{r}:{h.mean():.3f}->{h2.mean():.3f}")
    print(f"{c:<5} "+"  ".join(out))
