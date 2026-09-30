import json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, ".")
sys.path.insert(0, "pixel")
from retention2 import rung_matrices
import config

clean = dict(np.load("feat_A0.npz", allow_pickle=True))
cid = [str(x) for x in clean["ids"]]
pos = {s: i for i, s in enumerate(cid)}
q = dict(np.load("feat_A1.npz", allow_pickle=True))
qids = [str(x) for x in q["ids"]]
print("query order == gallery order (first 833 in-gallery ids):",
      [s for s in qids if s in pos] == cid)
rows = [i for i, s in enumerate(qids) if s in pos][:3]
qsub = {k: (v[rows] if hasattr(v, "__len__") and len(v) == len(qids) else v) for k, v in q.items()}
mats = rung_matrices(qsub, clean, config.WEIGHTS)
for rn, M in mats.items():
    self_cols = [pos[qids[i]] for i in rows]
    diag = [float(M[r, c]) for r, c in enumerate(self_cols)]
    rowmax = M.max(1)
    print(f"{rn:<14} shape={M.shape} finite={np.isfinite(M).all()} "
          f"self={np.round(diag,4).tolist()} rowmax={np.round(rowmax,4).tolist()} "
          f"min={M.min():.4f} max={M.max():.4f}")
