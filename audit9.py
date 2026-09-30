import numpy as np, json, re
from pathlib import Path
from collections import Counter
R=Path(__file__).resolve().parent
def norm(b):
    if b is None: return None
    b=re.sub(r"[^a-z0-9]+","",str(b).lower()); return b or None
def load(p):
    out={}
    for line in open(p,encoding="utf-8"):
        r=json.loads(line); pa=r.get("parsed"); sid=Path(r["image"]).stem
        out[sid]={"pe":True} if not pa else {"pe":False,"brand":norm(pa.get("brand")),
            "pt":(pa.get("page_type") or "").strip().lower() or None,"cred":pa.get("asks_for_credentials")}
    return out
clean=load(R/"vlm/desc_A0.jsonl")
ids0=[i for i in clean if not clean[i]["pe"]]
print("=== 21. LABEL CARDINALITY / CHANCE AGREEMENT of each semantic channel (clean marginals) ===")
for ch in ["brand","pt","cred"]:
    vals=[clean[i][ch] for i in ids0]
    c=Counter(map(str,vals)); n=len(vals)
    pe=sum((v/n)**2 for v in c.values())
    print(f"{ch:<6} distinct={len(c):3d}  top3={c.most_common(3)}  chance_agreement={pe:.3f}")
print()
print("=== 22. OBSERVED vs CHANCE-CORRECTED (kappa) SEMANTIC RETENTION ===")
for c in ["A0n","A3","A4","S1","S7"]:
    p=R/f"vlm/desc_{c}.jsonl"
    if not p.exists(): continue
    att=load(p); ids=[i for i in att if i in clean and not clean[i]["pe"]]
    out=[]
    for ch in ["pt","cred"]:
        pairs=[(clean[i][ch],att[i][ch]) for i in ids if not att[i]["pe"]]
        po=np.mean([a==b for a,b in pairs])
        ca=Counter(str(a) for a,_ in pairs); cb=Counter(str(b) for _,b in pairs); n=len(pairs)
        pe=sum(ca[k]*cb.get(k,0) for k in ca)/n**2
        k=(po-pe)/(1-pe) if pe<1 else float("nan")
        out.append(f"{ch}: obs={po:.3f} chance={pe:.3f} kappa={k:.3f}")
    elig=[i for i in ids if clean[i]["brand"] is not None and not att[i]["pe"]]
    pairs=[(clean[i]["brand"],att[i]["brand"]) for i in elig]
    po=np.mean([a==b for a,b in pairs])
    ca=Counter(str(a) for a,_ in pairs); cb=Counter(str(b) for _,b in pairs); n=len(pairs)
    pe=sum(ca[k]*cb.get(k,0) for k in ca)/n**2
    out.append(f"brand: obs={po:.3f} chance={pe:.3f} kappa={(po-pe)/(1-pe):.3f}")
    print(f"{c:<5} "+"   ".join(out))
print()
print("=== 23. IS THERE ANY BENIGN MEASUREMENT OF THE SEMANTIC RUNG? ===")
print(sorted(p.name for p in (R/"vlm").glob("*")))
print("benign feature npz:",[p.name for p in R.glob("feat_benign*")])
print("score.py output (TPR table) conditions:", list(json.load(open(R/"results_pixel_clip.json"))["conditions"].keys()))
d=json.load(open(R/"results_pixel_clip.json"))
print("taus:",{k:round(v,4) for k,v in d["taus"].items()},"n_ref",d["n_ref"],"n_cal",d["n_cal"])
for c,e in d["conditions"].items():
    print(f"  {c}: "+" ".join(f"{k}={v[chr(116)+chr(112)+chr(114)]:.3f}" for k,v in e["rungs"].items()))
