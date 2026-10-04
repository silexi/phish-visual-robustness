#!/usr/bin/env python
"""Corrected retention analysis, after the hostile statistics audit.

Four changes, each forced by a verified defect in the first pass:

1. Histogram correlation is accumulated in float64. The first pass used a
   two-pass Pearson on float32 histograms and lost the decision to catastrophic
   cancellation -- self-correlations of 1.000006 appeared, and 32 of the colour
   rung's 33 null-condition misses were on bit-identical renders decided at 1e-6.

2. A hit is "the own cluster attains the row maximum within eps". The gallery
   contains pixel-identical duplicates, so exact ties at the maximum are the
   normal case (median 7 per query) and an argmax tie-break was silently
   deciding them.

3. The primary estimand is the mean of per-cluster retention, not the mean over
   pages. Two kits hold 28% of the corpus; under inversion 72% of the hash's
   hits came from a single 107-page kit, and page-weighted means moved by 0.15
   depending on whether that kit was drawn in a bootstrap replicate.

4. Both estimands are reported for the excluded cells: applied-only (the
   conditional question, "when the attack works, how much does it buy") and
   intent-to-treat over all pages (the corpus question). They differ most for
   inversion, whose 136 no-op cells turned out to be 129 redeployments of one kit.
"""
import argparse, json, sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from score import _cos01, _hash_sim, _hist_inter, _ssim_matrix, wilson  # noqa: E402


def hist_corr64(A, B):
    """cv2.HISTCMP_CORREL in float64, centred once."""
    a = A.astype(np.float64)
    b = B.astype(np.float64)
    a -= a.mean(1, keepdims=True)
    b -= b.mean(1, keepdims=True)
    num = a @ b.T
    den = np.sqrt((a * a).sum(1)[:, None] * (b * b).sum(1)[None, :])
    return np.divide(num, den, out=np.zeros_like(num), where=den > 0)


def rung_matrices(q, r, w, roi_w=0.7):
    h_full = _hash_sim(q["px_hash"], q["px_hw"], r["px_hash"]).astype(np.float64)
    h_roi = _hash_sim(q["px_roi_hash"], q["px_roi_hw"], r["px_roi_hash"]).astype(np.float64)
    c_full = hist_corr64(q["px_color"], r["px_color"])
    c_roi = hist_corr64(q["px_roi_color"], r["px_roi_color"])
    l_full = _hist_inter(q["px_lbp"], r["px_lbp"]).astype(np.float64)
    l_roi = _hist_inter(q["px_roi_lbp"], r["px_roi_lbp"]).astype(np.float64)
    g_full = _cos01(q["px_hog"], r["px_hog"]).astype(np.float64)
    g_roi = _cos01(q["px_roi_hog"], r["px_roi_hog"]).astype(np.float64)
    ssim = _ssim_matrix(q["px_gray256"], r["px_gray256"]).astype(np.float64)
    blend = lambda a, b: roi_w * a + (1 - roi_w) * b
    out = {
        "P0_fusion": (w["hash_weight"] * blend(h_roi, h_full)
                      + w["color_weight"] * blend(c_roi, c_full)
                      + w["structure_weight"] * ssim
                      + w["lbp_weight"] * blend(l_roi, l_full)
                      + w["hog_weight"] * blend(g_roi, g_full)),
        "P3_hash": h_full, "P4_hog": g_full, "P_ssim_global": ssim,
        "P_color": c_full, "P_lbp": l_full,
    }
    d = np.linalg.norm(q["px_triv8"][:, None, :].astype(np.float64)
                       - r["px_triv8"][None, :, :].astype(np.float64), axis=-1)
    out["P5_triv8"] = 1.0 - d / max(d.max(), 1e-9)
    out["E1_clip"] = (q["clip"].astype(np.float64) @ r["clip"].astype(np.float64).T)
    return out


def cluster_stats(hit, cl, B=4000, seed=20260922):
    """Mean of per-cluster retention, with a cluster bootstrap of that estimand."""
    keys = np.array(sorted(set(cl)))
    idx = {k: np.flatnonzero(cl == k) for k in keys}
    per = np.array([hit[idx[k]].mean() for k in keys])
    rng = np.random.default_rng(seed)
    draws = np.array([per[rng.integers(0, len(per), len(per))].mean() for _ in range(B)])
    n_all = int(sum(hit[idx[k]].all() for k in keys))
    lo, hi = wilson(n_all, len(keys))
    return {"cluster_mean": round(float(per.mean()), 6),
            "boot95": [round(float(np.percentile(draws, 2.5)), 6),
                       round(float(np.percentile(draws, 97.5)), 6)],
            "page_mean": round(float(hit.mean()), 6),
            "n_clusters": int(len(keys)),
            "clusters_fully_retained": n_all,
            "wilson95_full": [round(lo, 6), round(hi, 6)]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clean", required=True)
    ap.add_argument("--conds", nargs="+", required=True)
    ap.add_argument("--clusters", required=True)
    ap.add_argument("--applied", required=True)
    ap.add_argument("--eps", type=float, default=1e-4)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    sys.path.insert(0, str(Path(__file__).parent / "pixel"))
    import config

    clean = dict(np.load(a.clean, allow_pickle=True))
    clean_ids = [str(x) for x in clean["ids"]]
    cmap = json.loads(Path(a.clusters).read_text())
    clean_cl = np.array([cmap.get(s, s) for s in clean_ids])

    distinct = len({clean["px_gray256"][i].tobytes() for i in range(len(clean_ids))})
    print(f"galeri: {len(clean_ids)} sayfa, {distinct} farkli goruntu, "
          f"{len(set(clean_cl))} kume", file=sys.stderr)

    res = {"n_gallery_pages": len(clean_ids), "n_gallery_distinct_images": distinct,
           "eps": a.eps, "conditions": {}}

    for spec in a.conds:
        name, path = spec.split("=", 1)
        if not Path(path).exists():
            continue
        q = dict(np.load(path, allow_pickle=True))
        qids = [str(x) for x in q["ids"]]
        mf = Path(a.applied) / f"manifest_{name}.jsonl"
        applied = ({json.loads(l)["safe"]: json.loads(l).get("applied")
                    for l in mf.open(encoding="utf-8")} if mf.exists() else {})

        rows = [i for i, s in enumerate(qids) if s in set(clean_ids)]
        qsub = {k: (v[rows] if hasattr(v, "__len__") and len(v) == len(qids) else v)
                for k, v in q.items()}
        kept = [qids[i] for i in rows]
        cl_all = np.array([cmap.get(s, s) for s in kept])
        is_applied = np.array([bool(applied.get(s)) for s in kept])

        mats = rung_matrices(qsub, clean, config.WEIGHTS)
        entry = {"n_pages": len(kept), "n_applied": int(is_applied.sum()), "rungs": {}}
        for rn, M in mats.items():
            rowmax = M.max(1, keepdims=True)
            near = M >= (rowmax - a.eps)
            same = (clean_cl[None, :] == cl_all[:, None])
            hit = (near & same).any(1)
            entry["rungs"][rn] = {
                "applied_only": cluster_stats(hit[is_applied], cl_all[is_applied])
                if is_applied.any() else None,
                "itt_all": cluster_stats(hit, cl_all),
            }
        res["conditions"][name] = entry
        cw = {k: v["applied_only"]["cluster_mean"] if v["applied_only"] else float("nan")
              for k, v in entry["rungs"].items()}
        print(f"{name} (uygulanan {entry['n_applied']}/{entry['n_pages']}): "
              + "  ".join(f"{k.split('_')[-1]}={v:.3f}" for k, v in cw.items()),
              file=sys.stderr)

    Path(a.out).write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(f"-> {a.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
