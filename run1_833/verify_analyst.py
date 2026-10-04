#!/usr/bin/env python
"""Independent re-derivation of the A3 rank-1 retention under three galleries
(full / self_excl / dup_excl).  Only rung_matrices, cluster_stats (retention2)
and config (pixel/config.py) are imported; masking, hit rule, dedup and the
dHash near-duplicate check are written here from scratch."""
import sys, json, itertools, collections
from pathlib import Path
import numpy as np

HOME = Path(__file__).resolve().parent
sys.path.insert(0, str(HOME))
sys.path.insert(0, str(HOME / "pixel"))
import config                                   # noqa: E402
from retention2 import rung_matrices, cluster_stats   # noqa: E402

EPS = 1e-4
RUNGS = ["P_color", "P5_triv8", "P_ssim_global", "P3_hash", "P_lbp",
         "P0_fusion", "E1_clip", "P4_hog"]

def sec(t):
    print("\n" + "=" * 8 + " " + t + " " + "=" * 8)

# ---------------------------------------------------------------- (1) load
sec("1. load / shapes")
clean = dict(np.load(HOME / "feat_A0.npz", allow_pickle=True))
q_all = dict(np.load(HOME / "feat_A3.npz", allow_pickle=True))
for nm, z in [("A0", clean), ("A3", q_all)]:
    for k, v in z.items():
        print(f"  {nm}.{k:14s} {str(v.shape):18s} {v.dtype}")
clean_ids = [str(x) for x in clean["ids"]]
q_ids = [str(x) for x in q_all["ids"]]
assert len(set(clean_ids)) == len(clean_ids) == 833
cmap = json.loads((HOME / "clusters_flat.json").read_text())
clean_cl = np.array([cmap.get(s, s) for s in clean_ids])
print("  gallery pages", len(clean_ids), "clusters in gallery", len(set(clean_cl)),
      "gallery ids absent from map", sum(s not in cmap for s in clean_ids))

# manifest
mf_rows = [json.loads(l) for l in (HOME / "pert/manifest_A3.jsonl").open(encoding="utf-8")]
print("  manifest rows", len(mf_rows), "distinct safe", len({r["safe"] for r in mf_rows}))
print("  applied value types", collections.Counter(type(r.get("applied")).__name__ for r in mf_rows))
print("  applied value counts", collections.Counter(repr(r.get("applied")) for r in mf_rows))
print("  ok value counts", collections.Counter(repr(r.get("ok")) for r in mf_rows))
applied_map = {r["safe"]: r.get("applied") for r in mf_rows}

# ---------------------------------------------------------------- (2) duplicates
sec("2. duplicate groups on px_gray256 bytes (gallery)")
G = clean["px_gray256"]
keys = [G[i].tobytes() for i in range(len(clean_ids))]
groups = collections.defaultdict(list)
for i, k in enumerate(keys):
    groups[k].append(i)
sizes = np.array([len(groups[k]) for k in keys])          # size of own group per page
n_distinct = len(groups)
n_twin = int((sizes >= 2).sum())
n_straddle = sum(len(set(clean_cl[m])) > 1 for m in groups.values())
cl_sizes = collections.Counter(clean_cl)
n_single = sum(1 for v in cl_sizes.values() if v == 1)
print(f"  distinct thumbnails {n_distinct}  pages with identical twin {n_twin}  "
      f"dup groups straddling clusters {n_straddle}  clusters {len(cl_sizes)}  singletons {n_single}")
print("  dup-group size histogram", sorted(collections.Counter(len(m) for m in groups.values()).items()))
# is the float32 thumbnail integer-valued? (if so, bytes-key == uint8-key)
print("  gray integer-valued:", bool(np.all(G == np.round(G))), " range", G.min(), G.max())
keys8 = [np.round(G[i]).astype(np.uint8).tobytes() for i in range(len(clean_ids))]
print("  distinct on uint8 key", len(set(keys8)))
# also: are duplicate groups duplicates on every other feature? (sanity)
for f in ["px_hash", "px_color", "clip", "px_triv8", "px_hog"]:
    bad = 0
    for m in groups.values():
        if len(m) > 1:
            X = clean[f][m]
            if not np.all(X == X[0]):
                bad += 1
    print(f"  dup groups whose {f} rows differ: {bad}")

# ---------------------------------------------------------------- (3) queries
sec("3. query set (A3 rows present in gallery)")
pos = {s: j for j, s in enumerate(clean_ids)}
rows = [i for i, s in enumerate(q_ids) if s in pos]
dropped = [s for s in q_ids if s not in pos]
print("  A3 rows", len(q_ids), "kept", len(rows), "dropped (not in gallery)", dropped,
      "cluster of dropped:", [cmap.get(s) for s in dropped],
      "applied of dropped:", [applied_map.get(s) for s in dropped])
qsub = {k: (v[rows] if hasattr(v, "shape") and v.shape and v.shape[0] == len(q_ids) else v)
        for k, v in q_all.items()}
kept = [q_ids[i] for i in rows]
own = np.array([pos[s] for s in kept])                     # column of own clean render
cl_q = np.array([cmap.get(s, s) for s in kept])
assert np.all(cl_q == clean_cl[own])
is_applied = np.array([bool(applied_map.get(s)) for s in kept])
is_applied_strict = np.array([applied_map.get(s) is True for s in kept])
print("  applied (truthy)", int(is_applied.sum()), " applied (is True)", int(is_applied_strict.sum()),
      " missing from manifest", sum(s not in applied_map for s in kept))

mats = rung_matrices(qsub, clean, config.WEIGHTS)
print("  rungs from rung_matrices:", list(mats))
assert set(mats) == set(RUNGS)
for rn in RUNGS:
    M = mats[rn]
    print(f"  {rn:14s} shape {M.shape} dtype {M.dtype} nan {int(np.isnan(M).sum())} "
          f"inf {int(np.isinf(M).sum())} min {M.min():.4f} max {M.max():.4f}")
print("  clip row norms (q) min/max", np.linalg.norm(qsub["clip"], axis=1).min(),
      np.linalg.norm(qsub["clip"], axis=1).max())

# ---------------------------------------------------------------- galleries
nq, ng = len(kept), len(clean_ids)
same = clean_cl[None, :] == cl_q[:, None]                  # (nq, ng)
own_key = [keys[j] for j in own]
allow = {
    "full": np.ones((nq, ng), bool),
    "self_excl": np.ones((nq, ng), bool),
    "dup_excl": np.ones((nq, ng), bool),
}
allow["self_excl"][np.arange(nq), own] = False
key_arr = np.array([hash(k) for k in keys])              # for a vectorised compare
own_key_arr = np.array([hash(k) for k in own_key])
allow["dup_excl"] = ~(key_arr[None, :] == own_key_arr[:, None])
# sanity: hash-based equality must agree with bytes equality
for i in range(nq):
    assert set(np.flatnonzero(~allow["dup_excl"][i])) == set(groups[own_key[i]])
assert np.all(~allow["dup_excl"][np.arange(nq), own])   # own column removed too

def hits_maskinf(M, ok):
    Mm = np.where(ok, M, -np.inf)
    rowmax = Mm.max(1, keepdims=True)
    near = Mm >= rowmax - EPS
    return (near & same & ok).any(1), rowmax[:, 0]

def hits_masknan(M, ok):
    Mm = np.where(ok, M, np.nan)
    rowmax = np.nanmax(Mm, axis=1, keepdims=True)
    near = Mm >= rowmax - EPS                               # NaN >= x is False
    return (near & same).any(1)

def hits_loop(M, ok):
    out = np.zeros(nq, bool)
    for i in range(nq):
        cols = np.flatnonzero(ok[i])
        mx = M[i, cols].max()
        best = cols[M[i, cols] >= mx - EPS]
        out[i] = any(clean_cl[j] == cl_q[i] for j in best)
    return out

def hits_argmax(M, ok):
    Mm = np.where(ok, M, -np.inf)
    j = Mm.argmax(1)
    return clean_cl[j] == cl_q

sec("4. retention per gallery variant (applied-only, cluster mean)")
results = {}
for gname in ["full", "self_excl", "dup_excl"]:
    ok = allow[gname]
    cand = (same & ok).any(1)                               # own cluster has a remaining candidate
    sel = is_applied & cand
    n, k = int(sel.sum()), len(set(cl_q[sel]))
    n_itt, k_itt = int(cand.sum()), len(set(cl_q[cand]))
    print(f"\n  [{gname}] masked cols/query: min {(~ok).sum(1).min()} med {np.median((~ok).sum(1)):.0f} "
          f"max {(~ok).sum(1).max()} | applied&cand n={n} k={k} | itt cand n={n_itt} k={k_itt} "
          f"| applied dropped {int((is_applied & ~cand).sum())}")
    results[gname] = {"n": n, "k": k, "rungs": {}}
    line = []
    for rn in RUNGS:
        M = mats[rn]
        h_inf, rowmax = hits_maskinf(M, ok)
        h_nan = hits_masknan(M, ok)
        h_loop = hits_loop(M, ok)
        h_arg = hits_argmax(M, ok)
        assert np.array_equal(h_inf, h_nan) and np.array_equal(h_inf, h_loop), rn
        st = cluster_stats(h_inf[sel], cl_q[sel])
        st_itt = cluster_stats(h_inf[cand], cl_q[cand])
        st_arg = cluster_stats(h_arg[sel], cl_q[sel])
        results[gname]["rungs"][rn] = {"applied_only": st, "itt_all": st_itt,
                                       "argmax_applied": st_arg["cluster_mean"],
                                       "n_ties_at_max_med": float(np.median(((np.where(ok, M, -np.inf) >= rowmax[:, None] - EPS)).sum(1)))}
        line.append(f"{rn.split('_')[-1]}={st['cluster_mean']:.3f}")
        assert st["n_clusters"] == k
    print("   applied cluster_mean: " + "  ".join(line))
    print("   applied page_mean   : " + "  ".join(
        f"{rn.split('_')[-1]}={results[gname]['rungs'][rn]['applied_only']['page_mean']:.3f}" for rn in RUNGS))
    print("   applied boot95      : " + "  ".join(
        f"{rn.split('_')[-1]}={results[gname]['rungs'][rn]['applied_only']['boot95']}" for rn in RUNGS))
    print("   argmax tie-break    : " + "  ".join(
        f"{rn.split('_')[-1]}={results[gname]['rungs'][rn]['argmax_applied']:.3f}" for rn in RUNGS))
    print("   itt cluster_mean    : " + "  ".join(
        f"{rn.split('_')[-1]}={results[gname]['rungs'][rn]['itt_all']['cluster_mean']:.3f}" for rn in RUNGS))
    print("   median #cols within eps of rowmax: " + "  ".join(
        f"{rn.split('_')[-1]}={results[gname]['rungs'][rn]['n_ties_at_max_med']:.0f}" for rn in RUNGS))

# eps-sensitivity of the masked variants (eps=0 and eps=1e-6)
sec("4b. eps sensitivity (applied-only cluster_mean)")
for eps_alt in [0.0, 1e-6, 1e-3]:
    for gname in ["self_excl", "dup_excl"]:
        ok = allow[gname]; cand = (same & ok).any(1); sel = is_applied & cand
        line = []
        for rn in RUNGS:
            Mm = np.where(ok, mats[rn], -np.inf)
            near = Mm >= Mm.max(1, keepdims=True) - eps_alt
            h = (near & same & ok).any(1)
            line.append(f"{rn.split('_')[-1]}={cluster_stats(h[sel], cl_q[sel])['cluster_mean']:.3f}")
        print(f"  eps={eps_alt:g} [{gname}] " + "  ".join(line))

# ---------------------------------------------------------------- (5) compare full vs primary
sec("5. compare 'full' with results_retention*.json")
for fn in ["results_retention_v2.json", "results_retention.json"]:
    p = HOME / fn
    if not p.exists():
        continue
    R = json.loads(p.read_text())
    print(f"  {fn}: n_gallery_pages={R.get('n_gallery_pages')} distinct={R.get('n_gallery_distinct_images')} eps={R.get('eps')}")
    e = R.get("conditions", {}).get("A3")
    if not e:
        print("   no A3 entry; keys:", list(R)[:10]); continue
    print(f"   A3 n_pages={e.get('n_pages')} n_applied={e.get('n_applied')}")
    for rn in RUNGS:
        r = e["rungs"].get(rn)
        if r is None:
            print(f"   {rn}: MISSING"); continue
        ao = r.get("applied_only") or {}
        mine = results["full"]["rungs"][rn]["applied_only"]
        flag = "OK" if ao.get("cluster_mean") == mine["cluster_mean"] else "DIFF"
        print(f"   {rn:14s} file cm={ao.get('cluster_mean')} boot={ao.get('boot95')} pm={ao.get('page_mean')} k={ao.get('n_clusters')} "
              f"| mine cm={mine['cluster_mean']} boot={mine['boot95']} pm={mine['page_mean']} k={mine['n_clusters']} {flag}")

# ---------------------------------------------------------------- (6) near-duplicates
sec("6. near-duplicate audit: 256-bit dHash and RMSE between distinct-byte pairs")
import cv2
def dhash256(img):
    small = cv2.resize(img.astype(np.float32), (17, 16), interpolation=cv2.INTER_AREA)
    return (small[:, 1:] > small[:, :-1]).ravel()
H = np.array([dhash256(G[i]) for i in range(ng)])          # (833, 256) bool
Hi = H.astype(np.int32)
ham = (Hi[:, None, :] != Hi[None, :, :]).sum(-1)             # (833, 833)
same_bytes = key_arr[:, None] == key_arr[None, :]
iu = np.triu_indices(ng, 1)
hd = ham[iu]; sb = same_bytes[iu]
print("  pairs total", len(hd), " same-bytes pairs", int(sb.sum()))
print("  same-bytes pairs with Hamming 0:", int((sb & (hd == 0)).sum()), "(must equal same-bytes pairs)")
for t in [0, 2, 4, 8, 16]:
    m = (~sb) & (hd <= t)
    ii, jj = iu[0][m], iu[1][m]
    same_cl = sum(clean_cl[a] == clean_cl[b] for a, b in zip(ii, jj))
    print(f"  different-bytes pairs with dHash Hamming <= {t:2d}: {int(m.sum()):5d}   of which same cluster {same_cl}")
# pixel RMSE between all pairs via Gram trick
X = G.reshape(ng, -1).astype(np.float64)
sq = (X * X).sum(1)
d2 = sq[:, None] + sq[None, :] - 2 * X @ X.T
d2 = np.maximum(d2, 0)
rmse = np.sqrt(d2 / X.shape[1])
r = rmse[iu]
for t in [0.5, 1, 2, 5, 10]:
    m = (~sb) & (r <= t)
    ii, jj = iu[0][m], iu[1][m]
    same_cl = sum(clean_cl[a] == clean_cl[b] for a, b in zip(ii, jj))
    print(f"  different-bytes pairs with pixel RMSE <= {t:4}: {int(m.sum()):5d}   of which same cluster {same_cl}")
m = (~sb) & (r <= 2)
ii, jj = iu[0][m], iu[1][m]
if len(ii):
    mad = [np.abs(G[a] - G[b]).max() for a, b in zip(ii, jj)]
    print("  max-abs-diff among RMSE<=2 distinct pairs: min", min(mad), "median", np.median(mad), "max", max(mad))
    print("  examples:", [(clean_ids[a], clean_ids[b], round(float(rmse[a, b]), 3), int(ham[a, b])) for a, b in list(zip(ii, jj))[:8]])
print("  smallest nonzero RMSE among distinct-byte pairs:", float(r[~sb].min()) if (~sb).any() else None)

# how many applied queries would additionally be dropped / changed under a Hamming<=t dedup?
sec("6b. dup_excl sensitivity: mask by dHash Hamming <= t instead of exact bytes")
for t in [0, 4, 8]:
    ok = ~(ham[own, :] <= t)                                  # mask every gallery col near own render
    ok = ok & allow["self_excl"]
    cand = (same & ok).any(1); sel = is_applied & cand
    line = []
    for rn in RUNGS:
        Mm = np.where(ok, mats[rn], -np.inf)
        near = Mm >= Mm.max(1, keepdims=True) - EPS
        h = (near & same & ok).any(1)
        line.append(f"{rn.split('_')[-1]}={cluster_stats(h[sel], cl_q[sel])['cluster_mean']:.3f}")
    print(f"  ham<={t}: n={int(sel.sum())} k={len(set(cl_q[sel]))}  " + "  ".join(line))

json.dump({"results": results, "dup_stats": {"distinct": n_distinct, "twins": n_twin, "straddle": n_straddle,
           "clusters": len(cl_sizes), "singletons": n_single}},
          open(HOME / "verify_analyst_out.json", "w"), indent=1, default=float)
print("\nwrote verify_analyst_out.json")
