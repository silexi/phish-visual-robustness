import numpy as np, json, sys
from pathlib import Path
from collections import Counter
R=Path(__file__).resolve().parent
sys.path.insert(0,str(R)); sys.path.insert(0,str(R/"pixel"))
import config
print("WEIGHTS:",config.WEIGHTS)
Z=dict(np.load(R/"audit_hits.npz",allow_pickle=True))
clean_ids=[str(x) for x in Z["clean_ids"]]
cmap=json.load(open(R/"clusters_f.json"))["clusters"]
ccl=np.array([cmap.get(s,s) for s in clean_ids]); p2={s:i for i,s in enumerate(clean_ids)}
RUNGS=["P_color","P5_triv8","P3_hash","P_ssim","P0_fusion","P_lbp","P1_repaired","E1_clip","P4_hog"]
app={c:{json.loads(l)["safe"]:bool(json.loads(l).get("applied")) for l in open(R/f"pert/manifest_{c}.jsonl")}
     for c in ["A0n","A1","A3","A4","S1","S3","S7"]}
print()
print("=== 15. WHAT IS THE A0n FLOOR MADE OF? ===")
ids=[str(x) for x in Z["A0n_all__ids"]]; ch=np.array([app["A0n"].get(s,False) for s in ids])
print("pages whose pixels the inert style DID change:",int(ch.sum()))
for r in RUNGS:
    t1=Z[f"A0n_all__top1__{r}"]; qi=np.array([p2[s] for s in ids]); h=(ccl[t1]==ccl[qi])
    miss=~h
    print(f"{r:<12} misses={int(miss.sum()):3d}  of which pixel-changed={int((miss&ch).sum()):3d}  "
          f"retention|unchanged={h[~ch].mean():.4f}  retention|changed={h[ch].mean() if ch.sum() else float(chr(110)+chr(97)+chr(110)):.4f}")
print()
print("=== 16. BOOTSTRAP: author (page-mean, cluster-resampled) vs cluster-mean; PAIRED contrasts ===")
def cboot(fn,cl,B=4000,seed=20260922):
    rng=np.random.default_rng(seed); keys=np.array(sorted(set(cl.tolist())))
    idx={k:np.flatnonzero(cl==k) for k in keys}; out=[]
    for _ in range(B):
        take=np.concatenate([idx[k] for k in rng.choice(keys,len(keys),replace=True)])
        out.append(fn(take))
    v=np.array(out); return v.mean(),np.percentile(v,2.5),np.percentile(v,97.5)
def cboot_cw(h,cl,B=4000,seed=20260922):
    rng=np.random.default_rng(seed); keys=np.array(sorted(set(cl.tolist())))
    idx={k:np.flatnonzero(cl==k) for k in keys}; out=[]
    for _ in range(B):
        ks=rng.choice(keys,len(keys),replace=True)
        out.append(np.mean([h[idx[k]].mean() for k in ks]))
    v=np.array(out); return v.mean(),np.percentile(v,2.5),np.percentile(v,97.5)
for c in ["A3","A4","S1","S3"]:
    tag=f"{c}_filt"; ids=[str(x) for x in Z[tag+"__ids"]]; qi=np.array([p2[s] for s in ids]); cl=ccl[qi]
    print(f"-- {c}")
    for r in ["P0_fusion","P4_hog","P_color","P_lbp","E1_clip","P3_hash"]:
        h=(ccl[Z[f"{tag}__top1__{r}"]]==cl).astype(float)
        _,l1,h1=cboot(lambda t:h[t].mean(),cl); _,l2,h2=cboot_cw(h,cl)
        print(f"   {r:<11} page={h.mean():.3f} CI[{l1:.2f},{h1:.2f}] (paper)   cluster={np.mean([h[cl==k].mean() for k in sorted(set(cl.tolist()))]):.3f} CI[{l2:.2f},{h2:.2f}]")
    for a,b in [("P4_hog","E1_clip"),("P4_hog","P3_hash"),("P_lbp","P0_fusion")]:
        ha=(ccl[Z[f"{tag}__top1__{a}"]]==cl).astype(float); hb=(ccl[Z[f"{tag}__top1__{b}"]]==cl).astype(float)
        d=ha-hb; _,l,hh=cboot(lambda t:d[t].mean(),cl)
        print(f"   PAIRED {a}-{b}: delta={d.mean():+.3f} CI[{l:+.3f},{hh:+.3f}]  (paired, same pages)")
print()
print("=== 17. P1_repaired / P5_triv8 BATCH DEPENDENCE (normalisation uses the eval batch) ===")
for c in ["A3","S1"]:
    f_ids=[str(x) for x in Z[f"{c}_filt__ids"]]; a_ids=[str(x) for x in Z[f"{c}_all__ids"]]
    pos_a={s:i for i,s in enumerate(a_ids)}; sel=np.array([pos_a[s] for s in f_ids])
    for r in ["P1_repaired","P5_triv8","P0_fusion"]:
        hf=(ccl[Z[f"{c}_filt__top1__{r}"]]==ccl[[p2[s] for s in f_ids]]).astype(float)
        ha=(ccl[Z[f"{c}_all__top1__{r}"]][sel]==ccl[[p2[s] for s in f_ids]]).astype(float)
        print(f"{c} {r:<12} same pages, scored in the 698/714-row batch: {hf.mean():.4f}  vs in the 833-row batch: {ha.mean():.4f}  n_flipped={int((hf!=ha).sum())}")
