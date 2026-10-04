#!/usr/bin/env python
"""Paired configuration contrasts on the primary (full) gallery.

For every condition, the per-page rank-1 hit of one configuration minus the
hit of another, averaged per cluster and then over clusters, with the same
cluster bootstrap as the tables (B=4000, seed 20260922), on the applied-only
population.  Same hit definition and estimand as retention2.py /
retention_excl.py; this file only exists so that the paired differences
quoted in the text (HOG - CLIP under A3, LBP - fusion under A4) are traceable
to a released file.

Usage (from the working directory that holds feat_*.npz and pert/):
    python paired_contrasts.py --out results_paired_contrasts.json
"""
import argparse, json, sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from retention2 import rung_matrices, cluster_stats  # noqa: E402
from retention_excl import paired_diff, RUNGS  # noqa: E402

CONTRASTS = [("P4_hog", "E1_clip"), ("P_lbp", "P0_fusion")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clean", default="feat_A0.npz")
    ap.add_argument("--conds", nargs="+", default=["A0n", "A1", "A3", "A4", "S1", "S3", "S7"])
    ap.add_argument("--clusters", default="clusters_flat.json")
    ap.add_argument("--applied", default="pert")
    ap.add_argument("--eps", type=float, default=1e-4)
    ap.add_argument("--out", default="results_paired_contrasts.json")
    a = ap.parse_args()
    sys.path.insert(0, str(Path(__file__).parent / "pixel"))
    import config

    clean = dict(np.load(a.clean, allow_pickle=True))
    clean_ids = [str(x) for x in clean["ids"]]
    pos = {s: i for i, s in enumerate(clean_ids)}
    cmap = json.loads(Path(a.clusters).read_text())
    clean_cl = np.array([cmap.get(s, s) for s in clean_ids])

    res = {"eps": a.eps, "estimand": "mean over clusters of the per-cluster mean of the per-page "
                                     "hit difference, applied-only, full gallery, cluster bootstrap "
                                     "B=4000 seed 20260922",
           "contrasts": [f"{x}_minus_{y}" for x, y in CONTRASTS], "conditions": {}}
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
        cl_all = np.array([cmap.get(s, s) for s in kept])
        is_applied = np.array([bool(applied.get(s)) for s in kept])
        same = (clean_cl[None, :] == cl_all[:, None])
        mats = rung_matrices(qsub, clean, config.WEIGHTS)
        hits = {}
        for rn in RUNGS:
            M = mats[rn]
            near = M >= (M.max(1, keepdims=True) - a.eps)
            hits[rn] = (near & same).any(1)
        sel = is_applied
        cl = cl_all[sel]
        entry = {"n_pages": int(sel.sum()), "n_clusters": int(len(set(cl.tolist())))}
        for x, y in CONTRASTS:
            d = hits[x][sel].astype(float) - hits[y][sel].astype(float)
            entry[f"{x}_minus_{y}"] = paired_diff(d, cl)
            # page-weighted mean of the same difference, cluster-resampled (the
            # estimator of the earlier audit; reported for traceability only)
            keys = np.array(sorted(set(cl)))
            idx = [np.where(cl == k)[0] for k in keys]
            rng = np.random.default_rng(20260922)
            draws = []
            for _ in range(4000):
                pick = rng.integers(0, len(keys), len(keys))
                draws.append(d[np.concatenate([idx[i] for i in pick])].mean())
            entry[f"{x}_minus_{y}_page_weighted"] = {
                "page_mean_diff": round(float(d.mean()), 6),
                "boot95": [round(float(np.percentile(draws, 2.5)), 6),
                           round(float(np.percentile(draws, 97.5)), 6)]}
        res["conditions"][name] = entry
        print(name, json.dumps(entry), file=sys.stderr)
    Path(a.out).write_text(json.dumps(res, indent=1))
    print("->", a.out, file=sys.stderr)


if __name__ == "__main__":
    main()
