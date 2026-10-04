#!/usr/bin/env python
"""Rank-1 retention when the query's own clean render is not in the gallery.

The primary analysis (retention2.py) scores a perturbed render against all
1,206 clean renders of the measurement set, its own clean counterpart included
(833 in the first run, released under run1_833/).  A reviewer can object that
a hit is then partly "find the image you were made from".  Five galleries are
scored here with exactly the primary hit rule (own cluster attains the row
maximum within eps):

  full         the primary gallery, unchanged (reproduces results_retention_v2)
  self_excl    the query's own clean render is removed from its candidate row
  dup_excl     the query's own clean render AND every clean render that is
               byte-identical to it (256x256 grey thumbnail) are removed
  rmse1_excl   sensitivity: every clean render within RMSE <= 1 grey level of
               the query's own clean render is removed (catches renders that
               differ only in a few pixels, e.g. a blinking caret)
  dhash0_excl  sensitivity: every clean render whose 256-bit difference hash
               (17x16 area-resampled thumbnail, the dedup.py hash recomputed
               on the 256x256 thumbnail) equals the query's own is removed

A query whose cluster has no candidate left after exclusion cannot retain by
construction; such queries are dropped and the surviving population is
reported (n_pages, n_clusters).  Because the surviving population changes
with the exclusion, every variant is also scored against the full gallery on
the same population ("rungs_full_gallery_same_population"), and the dup_excl
population is additionally scored under self_excl, so the three galleries can
be compared on one set of clusters.

Paired contrasts are reported as the mean over clusters of the per-cluster
difference, with the cluster bootstrap of that mean: exclusion minus full
gallery (same population) for every rung, and HOG minus CLIP within each
gallery.  Estimand and bootstrap are the primary ones: mean of per-cluster
retention, cluster bootstrap B=4000, seed 20260922, applied-only cells.
"""
import argparse, json, sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from retention2 import rung_matrices, cluster_stats  # noqa: E402

RUNGS = ["P_color", "P5_triv8", "P_ssim_global", "P3_hash", "P_lbp", "P0_fusion", "E1_clip", "P4_hog"]
VARIANTS = ["full", "self_excl", "dup_excl", "rmse1_excl", "dhash0_excl"]


def paired_diff(d, cl, B=4000, seed=20260922):
    """Mean over clusters of a per-page difference, with the cluster bootstrap."""
    keys = np.array(sorted(set(cl)))
    per = np.array([d[cl == k].mean() for k in keys])
    rng = np.random.default_rng(seed)
    draws = np.array([per[rng.integers(0, len(per), len(per))].mean() for _ in range(B)])
    return {"cluster_mean_diff": round(float(per.mean()), 6),
            "boot95": [round(float(np.percentile(draws, 2.5)), 6),
                       round(float(np.percentile(draws, 97.5)), 6)],
            "n_clusters": int(len(keys)),
            "clusters_negative": int((per < 0).sum()),
            "clusters_positive": int((per > 0).sum())}


def dhash256(img):
    small = cv2.resize(img.astype(np.float32), (17, 16), interpolation=cv2.INTER_AREA)
    return (small[:, 1:] > small[:, :-1]).ravel()


def fmt(stats, key="cluster_mean"):
    return "  ".join(f"{rn.split('_')[-1]}=" + (f"{stats[rn][key]:.3f}" if stats[rn] else "  -  ")
                     for rn in RUNGS)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clean", default="feat_A0.npz")
    ap.add_argument("--conds", nargs="+", default=["A0n", "A1", "A3", "A4", "S1", "S3", "S7"])
    ap.add_argument("--clusters", default="clusters_flat.json")
    ap.add_argument("--applied", default="pert")
    ap.add_argument("--eps", type=float, default=1e-4)
    ap.add_argument("--out", default="results_retention_excl.json")
    a = ap.parse_args()

    sys.path.insert(0, str(Path(__file__).parent / "pixel"))
    import config

    clean = dict(np.load(a.clean, allow_pickle=True))
    clean_ids = [str(x) for x in clean["ids"]]
    pos = {s: i for i, s in enumerate(clean_ids)}
    cmap = json.loads(Path(a.clusters).read_text())
    clean_cl = np.array([cmap.get(s, s) for s in clean_ids])
    ng = len(clean_ids)
    G = clean["px_gray256"]

    # exact-duplicate groups among the clean gallery (byte-identical thumbnails)
    key = [G[i].tobytes() for i in range(ng)]
    groups = {}
    for i, k in enumerate(key):
        groups.setdefault(k, []).append(i)
    dup_of = {i: np.array(g) for g in groups.values() for i in g}
    n_distinct = len(groups)
    n_with_twin = sum(len(g) for g in groups.values() if len(g) > 1)
    straddle = sum(1 for g in groups.values() if len(set(clean_cl[g])) > 1)
    print(f"galeri {ng} sayfa | {n_distinct} farkli goruntu | "
          f"{n_with_twin} sayfanin ozdes esi var | kume sinirini asan es grubu: {straddle}",
          file=sys.stderr)

    # near-duplicate structure for the two sensitivity galleries
    X = G.reshape(ng, -1).astype(np.float64)
    sq = (X * X).sum(1)
    rmse = np.sqrt(np.maximum(sq[:, None] + sq[None, :] - 2 * X @ X.T, 0) / X.shape[1])
    H = np.array([dhash256(G[i]) for i in range(ng)]).astype(np.int32)
    ham = (H[:, None, :] != H[None, :, :]).sum(-1)
    iu = np.triu_indices(ng, 1)
    gid = np.array([{k: n for n, k in enumerate(groups)}[k] for k in key])
    diff_bytes = gid[iu[0]] != gid[iu[1]]
    n_pairs_rmse1 = int((diff_bytes & (rmse[iu] <= 1.0)).sum())
    n_pairs_dhash0 = int((diff_bytes & (ham[iu] == 0)).sum())
    print(f"farkli baytli ciftler: RMSE<=1 olan {n_pairs_rmse1}, dHash uzakligi 0 olan {n_pairs_dhash0}",
          file=sys.stderr)

    csize = Counter(clean_cl.tolist())
    res = {"n_gallery_pages": ng, "n_gallery_distinct_images": n_distinct,
           "n_pages_with_identical_twin": n_with_twin,
           "n_duplicate_groups": sum(1 for g in groups.values() if len(g) > 1),
           "n_duplicate_groups_straddling_clusters": straddle,
           "n_distinct_byte_pairs_rmse_le_1": n_pairs_rmse1,
           "n_distinct_byte_pairs_dhash_0": n_pairs_dhash0,
           "n_clusters": len(csize),
           "n_singleton_clusters": sum(1 for v in csize.values() if v == 1),
           "eps": a.eps, "rung_order": RUNGS, "variant_order": VARIANTS,
           "estimand": "mean over clusters of per-cluster rank-1 retention, applied-only, "
                       "cluster bootstrap B=4000 seed 20260922",
           "conditions": {}}

    for name in a.conds:
        path = f"feat_{name}.npz"
        if not Path(path).exists():
            print(f"{name}: {path} yok, atlandi", file=sys.stderr)
            continue
        q = dict(np.load(path, allow_pickle=True))
        qids = [str(x) for x in q["ids"]]
        mf = Path(a.applied) / f"manifest_{name}.jsonl"
        applied = ({json.loads(l)["safe"]: json.loads(l).get("applied")
                    for l in mf.open(encoding="utf-8")} if mf.exists() else {})
        rows = [i for i, s in enumerate(qids) if s in pos]
        qsub = {k: (v[rows] if hasattr(v, "__len__") and len(v) == len(qids) else v)
                for k, v in q.items()}
        kept = [qids[i] for i in rows]
        no_clean = [s for s in qids if s not in pos]
        self_idx = np.array([pos[s] for s in kept])
        cl_all = np.array([cmap.get(s, s) for s in kept])
        is_applied = np.array([bool(applied.get(s)) for s in kept])
        same = (clean_cl[None, :] == cl_all[:, None])
        nq = len(kept)

        mats = rung_matrices(qsub, clean, config.WEIGHTS)
        entry = {"n_query_rows": len(qids), "n_pages": nq,
                 "n_without_clean_render": len(no_clean),
                 "ids_without_clean_render": no_clean,
                 "n_applied": int(is_applied.sum()), "variants": {}}

        masks = {"full": np.ones_like(same, dtype=bool)}
        m1 = np.ones_like(same, dtype=bool)
        m1[np.arange(nq), self_idx] = False
        masks["self_excl"] = m1
        m2 = m1.copy()
        for r, i in enumerate(self_idx):
            m2[r, dup_of[i]] = False
        masks["dup_excl"] = m2
        masks["rmse1_excl"] = m1 & ~(rmse[self_idx, :] <= 1.0)
        masks["dhash0_excl"] = m1 & ~(ham[self_idx, :] == 0)

        # hit vectors for every gallery, over all kept queries
        hits = {}
        for vname, mask in masks.items():
            hits[vname] = {}
            for rn in RUNGS:
                M = np.where(mask, mats[rn], -np.inf)
                near = M >= (M.max(1, keepdims=True) - a.eps)
                hits[vname][rn] = (near & same & mask).any(1)

        for vname in VARIANTS:
            mask = masks[vname]
            retrievable = (same & mask).any(1)          # cluster still has a candidate
            sel = retrievable & is_applied
            cl = cl_all[sel]
            v = {"n_pages": int(sel.sum()),
                 "n_clusters": int(len(set(cl.tolist()))),
                 "n_dropped_unretrievable_applied": int((is_applied & ~retrievable).sum())}
            if not sel.any():
                v["rungs"] = {rn: None for rn in RUNGS}
                entry["variants"][vname] = v
                print(f"{name:<4} {vname:<11} n=  0  (bos)", file=sys.stderr)
                continue
            v["rungs"] = {rn: cluster_stats(hits[vname][rn][sel], cl) for rn in RUNGS}
            # same surviving population, scored against the full gallery: separates
            # the effect of the exclusion from the effect of the population change
            v["rungs_full_gallery_same_population"] = {
                rn: cluster_stats(hits["full"][rn][sel], cl) for rn in RUNGS}
            if vname == "dup_excl":
                v["rungs_self_excl_same_population"] = {
                    rn: cluster_stats(hits["self_excl"][rn][sel], cl) for rn in RUNGS}
            # paired contrasts on the same population
            if vname != "full":
                v["paired_minus_full"] = {
                    rn: paired_diff(hits[vname][rn][sel].astype(float)
                                    - hits["full"][rn][sel].astype(float), cl) for rn in RUNGS}
            v["paired_hog_minus_clip"] = paired_diff(
                hits[vname]["P4_hog"][sel].astype(float) - hits[vname]["E1_clip"][sel].astype(float), cl)
            v["paired_hog_minus_clip_full_gallery_same_population"] = paired_diff(
                hits["full"]["P4_hog"][sel].astype(float) - hits["full"]["E1_clip"][sel].astype(float), cl)
            entry["variants"][vname] = v

            print(f"{name:<4} {vname:<11} n={v['n_pages']:3d} k={v['n_clusters']:3d} "
                  + fmt(v["rungs"]), file=sys.stderr)
            if vname != "full":
                print(f"     (full gallery, same pop)  " + fmt(v["rungs_full_gallery_same_population"]),
                      file=sys.stderr)
                print(f"     (paired minus full)       "
                      + fmt(v["paired_minus_full"], "cluster_mean_diff"), file=sys.stderr)
            d = v["paired_hog_minus_clip"]
            print(f"     HOG-CLIP paired {d['cluster_mean_diff']:+.3f} "
                  f"[{d['boot95'][0]:+.3f}; {d['boot95'][1]:+.3f}]  "
                  f"({d['clusters_positive']} kume HOG>CLIP, {d['clusters_negative']} tersi)",
                  file=sys.stderr)
        res["conditions"][name] = entry

    Path(a.out).write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(f"-> {a.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
