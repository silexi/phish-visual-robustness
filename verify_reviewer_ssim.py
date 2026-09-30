import json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, ".")
sys.path.insert(0, "pixel")
from retention2 import rung_matrices, hist_corr64
from score import _ssim_matrix
import config

clean = dict(np.load("feat_A0.npz", allow_pickle=True))
cid = [str(x) for x in clean["ids"]]
pos = {s: i for i, s in enumerate(cid)}
g = clean["px_gray256"]
print("px_gray256 range:", float(g.min()), float(g.max()), g.dtype)

# null condition: A0n, non-applied rows should be bit-identical re-renders
q = dict(np.load("feat_A0n.npz", allow_pickle=True))
qids = [str(x) for x in q["ids"]]
mf = {json.loads(l)["safe"]: json.loads(l).get("applied") for l in open("pert/manifest_A0n.jsonl", encoding="utf-8")}
rows = [i for i, s in enumerate(qids) if s in pos and not mf.get(s)][:20]
ident = [bool(np.array_equal(q["px_gray256"][i], g[pos[qids[i]]])) for i in rows]
print("A0n non-applied rows, thumbnail bit-identical to own clean:", sum(ident), "/", len(rows))
qsub = {k: (v[rows] if hasattr(v, "__len__") and len(v) == len(qids) else v) for k, v in q.items()}
S = _ssim_matrix(qsub["px_gray256"], g)
print("_ssim_matrix output dtype:", S.dtype, "shape:", S.shape)
S = S.astype(np.float64)
selfv = np.array([S[r, pos[qids[i]]] for r, i in enumerate(rows)])
print("self SSIM (identical inputs):", np.round(selfv, 6).tolist())
print("SSIM matrix max:", S.max(), "| n entries > 1:", int((S > 1).sum()), "| n entries > 1+1e-4:", int((S > 1 + 1e-4).sum()))
rowmax = S.max(1, keepdims=True)
for eps in [1e-4, 1e-3, 1e-2]:
    print(f"  eps={eps}: mean #gallery entries within eps of rowmax:", float((S >= rowmax - eps).sum(1).mean()))

# same on the perturbed A1 condition: how many near-ties per row at 1e-4 vs 1e-3 for ssim / fusion / hash
q1 = dict(np.load("feat_A1.npz", allow_pickle=True))
q1ids = [str(x) for x in q1["ids"]]
rows1 = [i for i, s in enumerate(q1ids) if s in pos][:20]
qs1 = {k: (v[rows1] if hasattr(v, "__len__") and len(v) == len(q1ids) else v) for k, v in q1.items()}
mats = rung_matrices(qs1, clean, config.WEIGHTS)
for rn in ["P_ssim_global", "P0_fusion", "P3_hash", "P_color", "P_lbp", "P4_hog", "E1_clip", "P5_triv8"]:
    M = mats[rn]
    rm = M.max(1, keepdims=True)
    n4 = (M >= rm - 1e-4).sum(1); n3 = (M >= rm - 1e-3).sum(1)
    print(f"A1 {rn:<14} max={M.max():.5f} ties@1e-4 median={np.median(n4):.0f} ties@1e-3 median={np.median(n3):.0f} rows_where_differ={(n4 != n3).sum()}/20")

# hist_corr64 self-consistency
C = hist_corr64(qsub["px_color"], clean["px_color"])
print("hist_corr64 self:", np.round([C[r, pos[qids[i]]] for r, i in enumerate(rows)], 8).tolist()[:5], "max:", C.max())
