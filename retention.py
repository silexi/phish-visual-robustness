#!/usr/bin/env python
"""Rank-1 self-retention: does a perturbed page still match its own clean render?

This is the measurement the thesis actually needs, and it needs no brand labels
and no reference database.  For every page i and condition c we ask whether the
attacked render still retrieves its own clean render as the top-1 nearest
neighbour among all N clean renders in the corpus.  Retention is therefore a
*discrimination* quantity -- it asks whether the representation still separates
this page from 800-odd other login pages -- not a raw similarity, which can stay
deceptively high while ranking collapses.

Reported per rung and per condition, with the null condition A0n giving the
re-render noise floor that every attack must be read against, and with the
bootstrap resampling kit clusters rather than pages because phishing kits repeat
across domains.
"""
import argparse, json, sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from score import (_cos01, _hash_sim, _hist_corr, _hist_inter, _ssim_matrix,  # noqa: E402
                   cluster_bootstrap, wilson)


def rung_matrices(q, r, weights, roi_w=0.7):
    """Similarity of every query row against every reference row, per rung."""
    h_full = _hash_sim(q["px_hash"], q["px_hw"], r["px_hash"])
    h_roi = _hash_sim(q["px_roi_hash"], q["px_roi_hw"], r["px_roi_hash"])
    c_full = _hist_corr(q["px_color"], r["px_color"])
    c_roi = _hist_corr(q["px_roi_color"], r["px_roi_color"])
    l_full = _hist_inter(q["px_lbp"], r["px_lbp"])
    l_roi = _hist_inter(q["px_roi_lbp"], r["px_roi_lbp"])
    g_full = _cos01(q["px_hog"], r["px_hog"])
    g_roi = _cos01(q["px_roi_hog"], r["px_roi_hog"])
    ssim = _ssim_matrix(q["px_gray256"], r["px_gray256"])
    blend = lambda a, b: roi_w * a + (1 - roi_w) * b

    out = {}
    out["P0_fusion"] = (weights["hash_weight"] * blend(h_roi, h_full)
                        + weights["color_weight"] * blend(c_roi, c_full)
                        + weights["structure_weight"] * ssim
                        + weights["lbp_weight"] * blend(l_roi, l_full)
                        + weights["hog_weight"] * blend(g_roi, g_full))
    out["P3_hash"] = h_full
    out["P4_hog"] = g_full
    out["P_ssim"] = ssim
    out["P_color"] = c_full
    out["P_lbp"] = l_full
    d = np.linalg.norm(q["px_triv8"][:, None, :] - r["px_triv8"][None, :, :], axis=-1)
    out["P5_triv8"] = 1.0 - d / max(d.max(), 1e-9)
    z = [(m - m.mean()) / max(m.std(), 1e-9) for m in (h_full, c_full, ssim, l_full, g_full)]
    out["P1_repaired"] = sum(z) / len(z)
    out["E1_clip"] = q["clip"] @ r["clip"].T
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clean", required=True)
    ap.add_argument("--conds", nargs="+", required=True, help="name=feat.npz")
    ap.add_argument("--clusters", required=True)
    ap.add_argument("--applied", default=None,
                    help="dir of perturbation manifests; cells whose condition "
                         "did not actually change the pixels are excluded")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    sys.path.insert(0, str(Path(__file__).parent / "pixel"))
    import config

    clean = dict(np.load(a.clean, allow_pickle=True))
    clean_ids = [str(x) for x in clean["ids"]]
    pos = {s: i for i, s in enumerate(clean_ids)}
    clusters_map = json.loads(Path(a.clusters).read_text())
    print(f"temiz korpus: {len(clean_ids)} sayfa, "
          f"{len({clusters_map.get(s, s) for s in clean_ids})} kume", file=sys.stderr)

    results = {"n_clean": len(clean_ids), "conditions": {}}

    for spec in a.conds:
        name, path = spec.split("=", 1)
        q = dict(np.load(path, allow_pickle=True))
        qids = [str(x) for x in q["ids"]]

        # The null condition is defined by *not* changing the pixels, so the
        # efficacy filter must not be applied to it.
        applied = None
        if a.applied and not name.startswith("A0"):
            mf = Path(a.applied) / f"manifest_{name}.jsonl"
            if mf.exists():
                applied = {json.loads(l)["safe"]: json.loads(l).get("applied")
                           for l in mf.open(encoding="utf-8")}

        keep = [i for i, s in enumerate(qids)
                if s in pos and (applied is None or applied.get(s))]
        if not keep:
            print(f"{name}: degerlendirilebilir hucre yok", file=sys.stderr)
            continue
        qsub = {k: (v[keep] if hasattr(v, "__len__") and len(v) == len(qids) else v)
                for k, v in q.items()}
        kept_ids = [qids[i] for i in keep]
        truth = np.array([pos[s] for s in kept_ids])
        cl = np.array([clusters_map.get(s, s) for s in kept_ids])
        # Kits repeat across domains, so the nearest neighbour of a page is very
        # often a sibling render of the same kit.  Counting that as a retrieval
        # failure would measure corpus redundancy, not representation
        # robustness, so a hit is "the top-1 belongs to the same kit cluster".
        clean_cl = np.array([clusters_map.get(s, s) for s in clean_ids])

        mats = rung_matrices(qsub, clean, config.WEIGHTS)
        entry = {"n_eval": len(kept_ids), "n_dropped_not_applied": len(qids) - len(keep),
                 "n_clusters": int(len(set(cl))), "rungs": {}}

        for rn, M in mats.items():
            top1 = M.argmax(1)
            hit = (clean_cl[top1] == cl)
            hit_exact = (top1 == truth)
            self_sim = M[np.arange(len(truth)), truth]
            # margin: self similarity minus the best page from a *different* kit
            M2 = M.copy()
            M2[clean_cl[None, :] == cl[:, None]] = -np.inf
            margin = self_sim - M2.max(1)
            lo, hi = wilson(int(hit.sum()), len(hit))
            m, blo, bhi = cluster_bootstrap(lambda t: hit[t].mean(), cl, B=2000)
            entry["rungs"][rn] = {
                "rank1_cluster": round(float(hit.mean()), 4),
                "rank1_exact": round(float(hit_exact.mean()), 4),
                "wilson95": [round(lo, 4), round(hi, 4)],
                "boot95": [round(blo, 4), round(bhi, 4)],
                "self_sim_mean": round(float(self_sim.mean()), 4),
                "margin_mean": round(float(margin.mean()), 4),
                "margin_p05": round(float(np.percentile(margin, 5)), 4),
            }
        results["conditions"][name] = entry
        print(f"{name} (n={len(kept_ids)}, {entry['n_clusters']} kume): "
              + "  ".join(f"{k}={v['rank1_cluster']:.3f}" for k, v in entry["rungs"].items()),
              file=sys.stderr)

    Path(a.out).write_text(json.dumps(results, indent=1), encoding="utf-8")
    print(f"-> {a.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
