#!/usr/bin/env python
"""Leakage diagnostic for dup_excl: how many applied queries still face a
near-identical render of their own clean page after exact-bytes dedup, and
what the numbers look like under a pixel-RMSE dedup instead."""
import sys, json, collections
from pathlib import Path
import numpy as np
HOME = Path(__file__).resolve().parent
sys.path.insert(0, str(HOME)); sys.path.insert(0, str(HOME / "pixel"))
import config
from retention2 import rung_matrices, cluster_stats
EPS = 1e-4
RUNGS = ["P_color", "P5_triv8", "P_ssim_global", "P3_hash", "P_lbp", "P0_fusion", "E1_clip", "P4_hog"]
clean = dict(np.load(HOME / "feat_A0.npz", allow_pickle=True))
q_all = dict(np.load(HOME / "feat_A3.npz", allow_pickle=True))
clean_ids = [str(x) for x in clean["ids"]]; q_ids = [str(x) for x in q_all["ids"]]
cmap = json.loads((HOME / "clusters_flat.json").read_text())
clean_cl = np.array([cmap.get(s, s) for s in clean_ids])
applied = {json.loads(l)["safe"]: json.loads(l).get("applied") for l in (HOME / "pert/manifest_A3.jsonl").open(encoding="utf-8")}
pos = {s: j for j, s in enumerate(clean_ids)}
rows = [i for i, s in enumerate(q_ids) if s in pos]
qsub = {k: (v[rows] if hasattr(v, "shape") and v.shape and v.shape[0] == len(q_ids) else v) for k, v in q_all.items()}
kept = [q_ids[i] for i in rows]; own = np.array([pos[s] for s in kept]); cl_q = clean_cl[own]
is_applied = np.array([bool(applied.get(s)) for s in kept])
nq, ng = len(kept), len(clean_ids)
same = clean_cl[None, :] == cl_q[:, None]
G = clean["px_gray256"]
keys = np.array([hash(G[i].tobytes()) for i in range(ng)])
X = G.reshape(ng, -1).astype(np.float64); sq = (X * X).sum(1)
rmse = np.sqrt(np.maximum(sq[:, None] + sq[None, :] - 2 * X @ X.T, 0) / X.shape[1])
mats = rung_matrices(qsub, clean, config.WEIGHTS)

def run(ok, label):
    cand = (same & ok).any(1); sel = is_applied & cand
    out = {}
    for rn in RUNGS:
        Mm = np.where(ok, mats[rn], -np.inf); near = Mm >= Mm.max(1, keepdims=True) - EPS
        h = (near & same & ok).any(1); out[rn] = (h, near)
    print(f"  [{label}] n={int(sel.sum())} k={len(set(cl_q[sel]))}  " + "  ".join(
        f"{rn.split('_')[-1]}={cluster_stats(out[rn][0][sel], cl_q[sel])['cluster_mean']:.3f}" for rn in RUNGS))
    return sel, out

ok_dup = ~(keys[None, :] == keys[own][:, None])
print("== dup_excl (exact bytes) ==")
sel, out = run(ok_dup, "dup_excl bytes")
# leakage: among selected queries, does a column with RMSE<=1 to own render remain?
near_own = (rmse[own, :] <= 1.0) & ok_dup
n_leak = int((near_own.any(1) & sel).sum())
print(f"  applied&kept queries with a remaining gallery render within RMSE<=1 of own clean render: {n_leak}/{int(sel.sum())} "
      f"(clusters {len(set(cl_q[sel & near_own.any(1)]))})")
for rn in RUNGS:
    h, near = out[rn]
    hit_sel = h & sel
    # a hit is 'leaky' if every same-cluster column within eps of rowmax is within RMSE<=1 of own render
    tied_same = near & same & ok_dup
    leaky = np.array([tied_same[i].any() and np.all(rmse[own[i], np.flatnonzero(tied_same[i])] <= 1.0) for i in range(nq)])
    print(f"    {rn:14s} hits {int(hit_sel.sum()):3d}  of which only via near-identical (RMSE<=1) renders: {int((leaky & sel).sum()):3d}")

print("== dup_excl with pixel-RMSE dedup (mask cols within RMSE<=t of own render) ==")
for t in [0.5, 1.0, 2.0, 5.0]:
    ok = ~(rmse[own, :] <= t)
    run(ok, f"rmse<={t}")
