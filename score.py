#!/usr/bin/env python
"""Score every rung against one shared decision rule and one shared operating point.

Decision rule, identical for all rungs:

    flag(q)  iff  max_b S(q, ref_b) >= tau   and   eTLD+1(q) not in domains(argmax_b)

tau is calibrated per rung on a brand-disjoint benign CALIBRATION split at a
target FPR, so no rung is read at a hand-picked threshold -- the original
paper's 0.70 was 0.0074 above a legitimate page and sat at chance.

The domain conjunct is what makes a legitimate page matching its own reference
a non-event: `office_en` vs `office_tr` scores 0.9465 and is correctly never
flagged, which removes the need for any leave-one-out special case.
"""
import argparse, json, sys
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------- pixel rung


def _hash_sim(hq, wq, hr):
    """Weighted Hamming similarity, reference weights as the published code uses."""
    mism = (hq[:, None, :] != hr[None, :, :])
    return 1.0 - (mism * wq[:, None, :]).sum(-1) / wq.sum(-1)[:, None]


def _hist_corr(A, B):
    """cv2.HISTCMP_CORREL, vectorised: Pearson over the flattened histogram."""
    a = A - A.mean(1, keepdims=True)
    b = B - B.mean(1, keepdims=True)
    num = a @ b.T
    den = np.sqrt((a * a).sum(1)[:, None] * (b * b).sum(1)[None, :])
    return np.divide(num, den, out=np.zeros_like(num), where=den > 0)


def _hist_inter(A, B):
    return np.minimum(A[:, None, :], B[None, :, :]).sum(-1)


def _cos01(A, B):
    """Cosine mapped to [0,1] exactly as the published code does."""
    an = A / np.maximum(np.linalg.norm(A, axis=1, keepdims=True), 1e-12)
    bn = B / np.maximum(np.linalg.norm(B, axis=1, keepdims=True), 1e-12)
    return (an @ bn.T + 1.0) / 2.0


def _ssim_matrix(Gq, Gr):
    """Global SSIM on 256x256 grayscale, one statistic per pair (no windows).

    The published code calls skimage's windowed SSIM; a full windowed matrix over
    millions of pairs is not affordable, so this uses the global-statistics form,
    which is the same estimator at a single scale.  Reported as such.
    """
    C1, C2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    # float64 throughout: in float32 the self-SSIM of a render came out as
    # 0.9995..1.0009 rather than 1, which is larger than the eps=1e-4 tie
    # tolerance of the retention hit rule (retention2.py)
    Gq, Gr = Gq.astype(np.float64), Gr.astype(np.float64)
    mq, mr = Gq.mean((1, 2)), Gr.mean((1, 2))
    vq, vr = Gq.var((1, 2)), Gr.var((1, 2))
    Qf = (Gq - mq[:, None, None]).reshape(len(Gq), -1)
    Rf = (Gr - mr[:, None, None]).reshape(len(Gr), -1)
    cov = (Qf @ Rf.T) / Qf.shape[1]
    num = (2 * mq[:, None] * mr[None, :] + C1) * (2 * cov + C2)
    den = (mq[:, None] ** 2 + mr[None, :] ** 2 + C1) * (vq[:, None] + vr[None, :] + C2)
    return num / den


def pixel_scores(q, r, w, roi_w=0.7, variant="P0"):
    """Return {name: (nq x nr) similarity matrix} for the pixel bank."""
    out = {}
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
    P0 = (w["hash_weight"] * blend(h_roi, h_full)
          + w["color_weight"] * blend(c_roi, c_full)
          + w["structure_weight"] * ssim
          + w["lbp_weight"] * blend(l_roi, l_full)
          + w["hog_weight"] * blend(g_roi, g_full))
    out["P0_fusion"] = P0
    out["P3_hash"] = h_full
    out["P4_hog"] = g_full
    out["P_ssim"] = ssim
    out["P_color"] = c_full

    # P5 TRIV-8: negated L2 on the 8x8 mean-RGB thumbnail, min-max mapped to [0,1]
    d = np.linalg.norm(q["px_triv8"][:, None, :] - r["px_triv8"][None, :, :], axis=-1)
    out["P5_triv8"] = 1.0 - d / max(d.max(), 1e-9)

    # P1 scale-repaired: per-metric z-score on this query set, equal weights, no ROI
    parts = [h_full, c_full, ssim, l_full, g_full]
    z = [(m - m.mean()) / max(m.std(), 1e-9) for m in parts]
    out["P1_repaired"] = sum(z) / len(z)
    return out


# ------------------------------------------------------------ decision rule


def calibrate(scores_cal, target_fpr):
    """tau = the (1 - target_fpr) quantile of benign max-similarity scores."""
    s = np.sort(scores_cal)
    k = int(np.ceil((1 - target_fpr) * len(s))) - 1
    return float(s[min(max(k, 0), len(s) - 1)])


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def cluster_bootstrap(fn, clusters, B=2000, seed=20260922):
    """Resample whole clusters, not rows: kits repeat across domains."""
    rng = np.random.default_rng(seed)
    keys = np.array(sorted(set(clusters)))
    idx = {k: np.flatnonzero(clusters == k) for k in keys}
    vals = []
    for _ in range(B):
        take = np.concatenate([idx[k] for k in rng.choice(keys, len(keys), replace=True)])
        v = fn(take)
        if v is not None and np.isfinite(v):
            vals.append(v)
    if not vals:
        return (np.nan, np.nan, np.nan)
    v = np.array(vals)
    return float(v.mean()), float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True, help="npz of benign reference features")
    ap.add_argument("--cal", required=True, help="npz of benign calibration features")
    ap.add_argument("--queries", nargs="+", required=True, help="cond=path.npz")
    ap.add_argument("--clusters", default=None, help="json: query id -> cluster id")
    ap.add_argument("--fpr", type=float, default=0.01)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    sys.path.insert(0, str(Path(__file__).parent / "pixel"))
    import config

    ref = dict(np.load(a.ref, allow_pickle=True))
    cal = dict(np.load(a.cal, allow_pickle=True))
    conds = {}
    for spec in a.queries:
        name, path = spec.split("=", 1)
        conds[name] = dict(np.load(path, allow_pickle=True))

    clusters_map = json.loads(Path(a.clusters).read_text()) if a.clusters else {}

    # --- calibrate every rung on the benign split -------------------------
    cal_px = pixel_scores(cal, ref, config.WEIGHTS)
    cal_clip = cal["clip"] @ ref["clip"].T
    rung_cal = {k: v.max(1) for k, v in cal_px.items()}
    rung_cal["E1_clip"] = cal_clip.max(1)
    taus = {k: calibrate(v, a.fpr) for k, v in rung_cal.items()}

    results = {"fpr_target": a.fpr, "n_ref": int(len(ref["ids"])),
               "n_cal": int(len(cal["ids"])), "taus": taus, "conditions": {}}

    for cname, q in conds.items():
        px = pixel_scores(q, ref, config.WEIGHTS)
        rung = {k: v.max(1) for k, v in px.items()}
        rung["E1_clip"] = (q["clip"] @ ref["clip"].T).max(1)
        ids = q["ids"]
        cl = np.array([clusters_map.get(str(i), str(i)) for i in ids])

        entry = {"n": int(len(ids)), "rungs": {}}
        for rn, sc in rung.items():
            flag = sc >= taus[rn]
            tpr = float(flag.mean())
            lo, hi = wilson(int(flag.sum()), len(flag))
            m, blo, bhi = cluster_bootstrap(
                lambda t: flag[t].mean(), cl, B=1000)
            entry["rungs"][rn] = {
                "tpr": round(tpr, 4),
                "wilson95": [round(lo, 4), round(hi, 4)],
                "boot95": [round(blo, 4), round(bhi, 4)],
                "score_mean": round(float(sc.mean()), 4),
                "score_p05": round(float(np.percentile(sc, 5)), 4),
                "score_p95": round(float(np.percentile(sc, 95)), 4),
            }
        results["conditions"][cname] = entry
        print(f"{cname}: " + "  ".join(
            f"{k}={v['tpr']:.3f}" for k, v in entry["rungs"].items()), file=sys.stderr)

    Path(a.out).write_text(json.dumps(results, indent=1), encoding="utf-8")
    print(f"-> {a.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
