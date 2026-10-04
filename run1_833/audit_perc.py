import numpy as np, json, sys
from pathlib import Path
from PIL import Image
from skimage.color import deltaE_ciede2000, rgb2lab
from skimage.metrics import structural_similarity as ssim
R=Path(__file__).resolve().parent
ids=[x.strip() for x in (R/"ids_f.txt").read_text().split() if x.strip()]
rng=np.random.default_rng(20260922)
ids=list(rng.choice(ids,200,replace=False))
out={}
for c in ["A3","S1","S7","A4"]:
    app={json.loads(l)["safe"]:bool(json.loads(l).get("applied")) for l in open(R/f"pert/manifest_{c}.jsonl")}
    rows=[]
    for sid in ids:
        cp=R/"clean_sub"/f"{sid}.png"; ap=R/"pert"/c/f"{sid}.png"
        if not (cp.exists() and ap.exists()): continue
        a=np.asarray(Image.open(cp).convert("RGB"),dtype=np.uint8); b=np.asarray(Image.open(ap).convert("RGB"),dtype=np.uint8)
        if a.shape!=b.shape: continue
        s=ssim(a,b,channel_axis=2,data_range=255,gaussian_weights=True,sigma=1.5,use_sample_covariance=False)
        dE=deltaE_ciede2000(rgb2lab(a/255.0),rgb2lab(b/255.0))
        rows.append({"id":sid,"applied":app.get(sid,False),"ssim":float(s),"dE_mean":float(dE.mean()),
                     "dE_p99":float(np.percentile(dE,99)),"f_JND":float((dE>1.0).mean())})
    out[c]=rows
    for lab,sel in [("ALL",rows),("APPLIED",[r for r in rows if r["applied"]]),("NOTAPPLIED",[r for r in rows if not r["applied"]])]:
        if not sel: continue
        f=lambda k: np.median([r[k] for r in sel])
        print(f"{c:<4}{lab:<12}n={len(sel):<4} SSIM_med={f(chr(115)+chr(115)+chr(105)+chr(109)):.4f} dEmean_med={f(chr(100)+chr(69)+chr(95)+chr(109)+chr(101)+chr(97)+chr(110)):.3f} "
              f"fJND_med={f(chr(102)+chr(95)+chr(74)+chr(78)+chr(68)):.4f} fJND_mean={np.mean([r[chr(102)+chr(95)+chr(74)+chr(78)+chr(68)] for r in sel]):.4f} "
              f"fJND_p90={np.percentile([r[chr(102)+chr(95)+chr(74)+chr(78)+chr(68)] for r in sel],90):.4f}",flush=True)
json.dump(out,open(R/"audit_perc.json","w"))
