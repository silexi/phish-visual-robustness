import numpy as np, json, re
from pathlib import Path
from collections import Counter
R=Path(__file__).resolve().parent
Z=dict(np.load(R/"audit_hits.npz",allow_pickle=True))
clean_ids=[str(x) for x in Z["clean_ids"]]
cmap=json.load(open(R/"clusters_f.json"))["clusters"]
ccl=np.array([cmap.get(s,s) for s in clean_ids])
size=Counter(ccl.tolist()); p2={s:i for i,s in enumerate(clean_ids)}
RUNGS=["P_color","P5_triv8","P3_hash","P_ssim","P0_fusion","P_lbp","P1_repaired","E1_clip","P4_hog"]
app={}
for c in ["A0n","A1","A3","A4","S1","S3","S7"]:
    app[c]={json.loads(l)["safe"]:bool(json.loads(l).get("applied")) for l in open(R/f"pert/manifest_{c}.jsonl")}
def norm(b):
    if b is None: return None
    b=re.sub(r"[^a-z0-9]+","",str(b).lower()); return b or None
def load(p):
    out={}
    for line in open(p,encoding="utf-8"):
        r=json.loads(line); pa=r.get("parsed"); sid=Path(r["image"]).stem
        out[sid]={"pe":True} if not pa else {"pe":False,"brand":norm(pa.get("brand")),"raw":pa.get("brand"),
            "pt":(pa.get("page_type") or "").strip().lower() or None,"cred":pa.get("asks_for_credentials"),
            "logo":pa.get("logo_present")}
    return out
clean=load(R/"vlm/desc_A0.jsonl")
V=[i for i in clean]
print("=== 12. THE 200-PAGE VLM SUBSET: composition ===")
cc=Counter(cmap.get(s,s) for s in V)
print("n",len(V),"clusters",len(cc),"top",cc.most_common(6))
print("pages from the 2 giant kits:",sum(v for k,v in cc.items() if k in ("k2","k22")),
      " share=%.2f"%(sum(v for k,v in cc.items() if k in ("k2","k22"))/len(V)))
print("in-clean:",sum(1 for s in V if s in p2))
print()
print("=== 13. PIXEL/CLIP RUNGS RESTRICTED TO THE VLM SUBSET (concern 3) ===")
VS=set(V)
print("cond           n  "+"".join(f"{r:>12}" for r in RUNGS))
for c in ["A0n","A3","A4","S1","S7"]:
    tag=f"{c}_all" if c.startswith("A0") else f"{c}_filt"
    qids=[str(x) for x in Z[tag+"__ids"]]
    idx=np.array([i for i,s in enumerate(qids) if s in VS]); qi=np.array([p2[qids[i]] for i in idx])
    rowp=[];rowc=[]
    for r in RUNGS:
        t1=Z[f"{tag}__top1__{r}"][idx]; h=(ccl[t1]==ccl[qi]).astype(float); rowp.append(h.mean())
        ks=sorted(set(ccl[qi].tolist())); rowc.append(np.mean([h[ccl[qi]==k].mean() for k in ks]))
    print(f"{c:<5} page {len(idx):<4}"+"".join(f"{v:12.4f}" for v in rowp))
    print(f"{c:<5} clus {len(set(ccl[qi].tolist())):<4}"+"".join(f"{v:12.4f}" for v in rowc))
print()
print("=== 14. VLM TABLE 2 RECOMPUTED: applied-only, cluster-weighted, wrong-brand decomposed ===")
for c in ["A0n","A3","A4","S1","S7"]:
    p=R/f"vlm/desc_{c}.jsonl"
    if not p.exists(): continue
    att=load(p)
    ids=[i for i in att if i in clean and not clean[i]["pe"]]
    elig=[i for i in ids if clean[i]["brand"] is not None]
    def rates(sub):
        n=len(sub); ret=sum(1 for i in sub if not att[i]["pe"] and att[i]["brand"]==clean[i]["brand"])
        ab=sum(1 for i in sub if not att[i]["pe"] and att[i]["brand"] is None)
        pe=sum(1 for i in sub if att[i]["pe"])
        wrong=n-ret-ab-pe
        return n,ret/n,ab/n,wrong/n,pe/n
    n,r1,a1,w1,pe1=rates(elig)
    ea=[i for i in elig if app[c].get(i,False)] if not c.startswith("A0") else elig
    n2,r2,a2,w2,pe2=rates(ea) if ea else (0,0,0,0,0)
    cl=np.array([cmap.get(i,i) for i in elig]); v=np.array([1.0 if (not att[i]["pe"] and att[i]["brand"]==clean[i]["brand"]) else 0.0 for i in elig])
    ks=sorted(set(cl.tolist())); cw=np.mean([v[cl==k].mean() for k in ks])
    print(f"{c:<4} ALL      n={n:<4} brand={r1:.3f} abstain={a1:.3f} wrong={w1:.3f} parse_fail={pe1:.3f}  cluster_wtd_brand={cw:.3f} ({len(ks)} clusters)")
    print(f"{c:<4} APPLIED  n={n2:<4} brand={r2:.3f} abstain={a2:.3f} wrong={w2:.3f} parse_fail={pe2:.3f}")
    # false naming on pages the clean run left unnamed
    un=[i for i in ids if clean[i]["brand"] is None]
    fn=[i for i in un if not att[i]["pe"] and att[i]["brand"] is not None]
    logo_off=[i for i in ids if not att[i]["pe"] and clean[i]["logo"] and not att[i]["logo"]]
    print(f"     clean-unnamed n={len(un)} -> named by attack: {len(fn)} ({[att[i][chr(114)+chr(97)+chr(119)] for i in fn][:6]})   logo_present flipped ON->OFF: {len(logo_off)}/{sum(1 for i in ids if clean[i][chr(108)+chr(111)+chr(103)+chr(111)])}")
