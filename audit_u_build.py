#!/usr/bin/env python
"""Build the per-page hit store needed to re-run the four audit claims.

Usage:  audit_u_build.py u      ->  audit_hits_u.npz   (1,206 pages / 248 clusters)
        audit_u_build.py f      ->  audit_hits_f2.npz  (833 pages / 172 clusters, cross-check;
                                    audit_hits.npz of 22 September is left untouched)

For every condition and every configuration the store keeps, per query page:

  old rule  (first-pass retention.py: float32 primitives, plain argmax)
      top1_old, selfsim_old, maxsim_old
  new rule  (corrected retention2.py: float64 colour, eps=1e-4 tie tolerance)
      top1_new, selfsim_new, maxsim_new, nties_new,
      hit_dhash  (own dHash kit attains the row max within eps)
      hit_clip   (own CLIP-rebuilt cluster attains the row max within eps)

plus the CLIP re-clustering labels of the clean gallery (audit claim iv).

_hash_sim is replaced by a row-chunked implementation because the released one
materialises an (nq x nr x 10240) float32 temporary (59 GB at nq=nr=1206).  The
chunked version is verified bit-identical on a 120-query subset before use; the
check is printed and aborts the run if it fails.
"""
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

R = Path("/Users/khomeassist/phish-vlm")
sys.path.insert(0, str(R))
sys.path.insert(0, str(R / "pixel"))

import config                      # noqa: E402
import retention as ret_old        # noqa: E402  first-pass rung_matrices
import retention2 as ret_new       # noqa: E402  corrected rung_matrices
from score import _hash_sim as _hash_sim_released   # noqa: E402

CONDS = ["A0n", "A1", "A3", "A4", "S1", "S3", "S7"]
EPS = 1e-4
RUNGS_NEW = ["P_color", "P5_triv8", "P_ssim_global", "P3_hash", "P_lbp",
             "P0_fusion", "E1_clip", "P4_hog"]
RUNGS_OLD = ["P_color", "P5_triv8", "P_ssim", "P3_hash", "P_lbp",
             "P0_fusion", "E1_clip", "P4_hog", "P1_repaired"]


def hash_sim_chunked(hq, wq, hr, bs=96):
    nq = len(hq)
    out = np.empty((nq, len(hr)), dtype=np.float32)
    wsum = wq.sum(-1)
    for i in range(0, nq, bs):
        j = min(i + bs, nq)
        mism = (hq[i:j, None, :] != hr[None, :, :])
        out[i:j] = 1.0 - (mism * wq[i:j, None, :]).sum(-1) / wsum[i:j, None]
    return out


def clip_linkage(S, t):
    """Single linkage on the CLIP cosine matrix at threshold t (audit4.py section 11)."""
    n = len(S)
    par = list(range(n))

    def f(x):
        while par[x] != x:
            par[x] = par[par[x]]
            x = par[x]
        return x

    I, J = np.where(np.triu(S >= t, 1))
    for i, j in zip(I, J):
        a, b = f(int(i)), f(int(j))
        if a != b:
            par[max(a, b)] = min(a, b)
    return np.array([f(i) for i in range(n)])


def main():
    run = sys.argv[1]
    if run == "u":
        clean_path, cl_path, pert, out = (R / "feat_u_A0.npz", R / "clusters_u_flat.json",
                                          R / "pert_u", R / "audit_hits_u.npz")
        featf = lambda c: R / f"feat_u_{c}.npz"
    elif run == "f":
        clean_path, cl_path, pert, out = (R / "feat_A0.npz", R / "clusters_flat.json",
                                          R / "pert", R / "audit_hits_f2.npz")
        featf = lambda c: R / f"feat_{c}.npz"
    else:
        raise SystemExit("run must be u or f")

    clean = dict(np.load(clean_path, allow_pickle=True))
    clean_ids = [str(x) for x in clean["ids"]]
    ng = len(clean_ids)
    pos = {s: i for i, s in enumerate(clean_ids)}
    cmap = json.loads(cl_path.read_text())
    clean_cl = np.array([cmap.get(s, s) for s in clean_ids])
    nk = len(set(clean_cl.tolist()))
    print(f"[{run}] gallery {ng} pages, {nk} dHash kit clusters", flush=True)

    # ---- bit-identity check of the chunked hash similarity ------------------
    hq, wq = clean["px_hash"][:120], clean["px_hw"][:120]
    A = _hash_sim_released(hq, wq, clean["px_hash"])
    B = hash_sim_chunked(hq, wq, clean["px_hash"])
    same = A.dtype == B.dtype and A.tobytes() == B.tobytes()
    print(f"[{run}] chunked _hash_sim bit-identical on 120x{ng} subset: {same}", flush=True)
    if not same:
        raise SystemExit("chunked hash similarity is not bit-identical -- aborting")
    ret_old._hash_sim = hash_sim_chunked
    ret_new._hash_sim = hash_sim_chunked
    del A, B

    # ---- claim (iv): rebuild the clusters from the CLIP embeddings ----------
    E = clean["clip"]
    nrm = np.linalg.norm(E, axis=1)
    print(f"[{run}] CLIP embedding norms: min={nrm.min():.6f} max={nrm.max():.6f}", flush=True)
    S = E @ E.T
    lo, hi = 0.5, 0.9999
    for _ in range(40):
        t = (lo + hi) / 2
        if len(set(clip_linkage(S, t).tolist())) < nk:
            lo = t
        else:
            hi = t
    thr = (lo + hi) / 2
    lab = clip_linkage(S, thr)
    clip_cl = np.array([f"c{v}" for v in lab])
    cs = Counter(clip_cl.tolist())
    print(f"[{run}] CLIP re-clustering: threshold={thr:.6f} -> {len(cs)} clusters "
          f"(target {nk}); top sizes {sorted(cs.values(), reverse=True)[:8]}; "
          f"singletons {sum(1 for v in cs.values() if v == 1)}", flush=True)
    del S

    store = {"clean_ids": np.array(clean_ids),
             "clean_cl_dhash": clean_cl,
             "clean_cl_clip": clip_cl,
             "clip_threshold": np.array([thr])}

    for c in CONDS:
        p = featf(c)
        if not p.exists():
            print(f"[{run}] {p.name} missing -- condition skipped", flush=True)
            continue
        q = dict(np.load(p, allow_pickle=True))
        qids = [str(x) for x in q["ids"]]
        mf = pert / f"manifest_{c}.jsonl"
        man = {json.loads(l)["safe"]: json.loads(l) for l in mf.open(encoding="utf-8")}
        rows = [i for i, s in enumerate(qids) if s in pos]
        qsub = {k: (v[rows] if hasattr(v, "__len__") and len(v) == len(qids) else v)
                for k, v in q.items()}
        kept = [qids[i] for i in rows]
        truth = np.array([pos[s] for s in kept])
        cl_all = np.array([cmap.get(s, s) for s in kept])
        applied = np.array([bool(man.get(s, {}).get("applied")) for s in kept])
        frac = np.array([float(man.get(s, {}).get("frac_changed", 0.0)) for s in kept])
        same_d = (clean_cl[None, :] == cl_all[:, None])
        clipq = clip_cl[truth]
        same_c = (clip_cl[None, :] == clipq[:, None])

        store[f"{c}__ids"] = np.array(kept)
        store[f"{c}__truth"] = truth.astype(np.int32)
        store[f"{c}__applied"] = applied
        store[f"{c}__frac_changed"] = frac

        mats_old = ret_old.rung_matrices(qsub, clean, config.WEIGHTS)
        for rn in RUNGS_OLD:
            M = mats_old[rn]
            t1 = M.argmax(1).astype(np.int32)
            store[f"{c}__old__top1__{rn}"] = t1
            store[f"{c}__old__selfsim__{rn}"] = M[np.arange(len(kept)), truth].astype(np.float64)
            store[f"{c}__old__maxsim__{rn}"] = M.max(1).astype(np.float64)
        del mats_old

        mats_new = ret_new.rung_matrices(qsub, clean, config.WEIGHTS)
        for rn in RUNGS_NEW:
            M = mats_new[rn]
            rowmax = M.max(1, keepdims=True)
            near = M >= (rowmax - EPS)
            store[f"{c}__new__top1__{rn}"] = M.argmax(1).astype(np.int32)
            store[f"{c}__new__selfsim__{rn}"] = M[np.arange(len(kept)), truth]
            store[f"{c}__new__maxsim__{rn}"] = rowmax[:, 0]
            store[f"{c}__new__nties__{rn}"] = near.sum(1).astype(np.int32)
            store[f"{c}__hit_dhash__{rn}"] = (near & same_d).any(1)
            store[f"{c}__hit_clip__{rn}"] = (near & same_c).any(1)
        del mats_new, same_d, same_c, q, qsub

        print(f"[{run}] {c}: {len(kept)} pages, applied {int(applied.sum())}, "
              f"{len(set(cl_all.tolist()))} kit clusters", flush=True)

    np.savez_compressed(out, **store)
    print(f"[{run}] -> {out}", flush=True)


if __name__ == "__main__":
    main()
