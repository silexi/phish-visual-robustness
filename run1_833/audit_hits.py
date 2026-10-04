import numpy as np, json, sys
from pathlib import Path
R = Path(__file__).resolve().parent
sys.path.insert(0, str(R)); sys.path.insert(0, str(R/"pixel"))
import config
from retention import rung_matrices

clean = dict(np.load(R/"feat_A0.npz", allow_pickle=True))
clean_ids = [str(x) for x in clean["ids"]]
pos = {s:i for i,s in enumerate(clean_ids)}
cl_map = json.load(open(R/"clusters_f.json"))["clusters"]

conds = ["A0n","A1","A3","A4","S1","S3","S7"]
store = {"clean_ids": np.array(clean_ids)}
meta = {}
for c in conds:
    q = dict(np.load(R/f"feat_{c}.npz", allow_pickle=True))
    qids = [str(x) for x in q["ids"]]
    applied = {}
    for l in open(R/f"pert/manifest_{c}.jsonl"):
        r = json.loads(l); applied[r["safe"]] = bool(r.get("applied"))
    for mode in ["filt","all"]:
        if mode == "filt" and c.startswith("A0"):
            continue
        keep = [i for i,s in enumerate(qids)
                if s in pos and (mode=="all" or applied.get(s))]
        qsub = {k:(v[keep] if hasattr(v,"__len__") and len(v)==len(qids) else v)
                for k,v in q.items()}
        kept = [qids[i] for i in keep]
        mats = rung_matrices(qsub, clean, config.WEIGHTS)
        tag = f"{c}_{mode}"
        store[tag+"__ids"] = np.array(kept)
        store[tag+"__truth"] = np.array([pos[s] for s in kept])
        store[tag+"__applied"] = np.array([applied.get(s,False) for s in kept])
        for rn, M in mats.items():
            store[f"{tag}__top1__{rn}"] = M.argmax(1).astype(np.int32)
            mx = M.max(1)
            store[f"{tag}__nties__{rn}"] = (M >= mx[:,None]-1e-12).sum(1).astype(np.int32)
            store[f"{tag}__selfsim__{rn}"] = M[np.arange(len(kept)), store[tag+"__truth"]]
            store[f"{tag}__maxsim__{rn}"] = mx
        meta[tag] = {"n": len(kept), "n_applied": int(sum(applied.get(s,False) for s in kept))}
        print(tag, meta[tag], flush=True)
np.savez_compressed(R/"audit_hits.npz", **store)
json.dump(meta, open(R/"audit_meta.json","w"), indent=1)
print("done")
