import json,re
from pathlib import Path
import numpy as np
R=Path(__file__).resolve().parent
def norm(b):
    if b is None: return None
    b=re.sub(r"[^a-z0-9]+","",str(b).lower()); return b or None
def load(p):
    out={}
    for line in open(p,encoding="utf-8"):
        r=json.loads(line); pa=r.get("parsed"); sid=Path(r["image"]).stem
        out[sid]={"pe":True} if not pa else {"pe":False,"brand":norm(pa.get("brand")),"raw":pa.get("brand"),"logo":pa.get("logo_present")}
    return out
clean=load(R/"vlm/desc_A0.jsonl")
ids=[i for i in clean if not clean[i]["pe"]]
elig=[i for i in ids if clean[i]["brand"]]; nel=[i for i in ids if not clean[i]["brand"]]
print(f"eligible={len(elig)} logo_present={sum(1 for i in elig if clean[i][chr(108)+chr(111)+chr(103)+chr(111)])} "
      f"| not-eligible={len(nel)} logo_present={sum(1 for i in nel if clean[i][chr(108)+chr(111)+chr(103)+chr(111)])}")
print()
print("=== 29. IS THE CONDITIONING A SELECTION ON LOGO PRESENCE? brand retention by clean logo_present ===")
for c in ["A0n","A3","A4","S1","S7"]:
    att=load(R/f"vlm/desc_{c}.jsonl")
    for lab,sub in [("logo=T",[i for i in elig if clean[i]["logo"]]),("logo=F",[i for i in elig if not clean[i]["logo"]])]:
        r=np.mean([1.0 if (i in att and not att[i]["pe"] and att[i]["brand"]==clean[i]["brand"]) else 0.0 for i in sub])
        print(f"  {c:<4}{lab}  n={len(sub):<4} brand_retention={r:.3f}")
    # unconditional misattribution: named a brand that is not the clean brand, over ALL pages
    mis=[i for i in ids if i in att and not att[i]["pe"] and att[i]["brand"] and att[i]["brand"]!=clean[i]["brand"]]
    print(f"  {c:<4}UNCONDITIONAL misattribution (any page now carrying a wrong/new brand name): "
          f"{len(mis)}/{len(ids)} = {len(mis)/len(ids):.3f}   [paper reports {chr(39)}wrong brand{chr(39)} over eligible only]")
