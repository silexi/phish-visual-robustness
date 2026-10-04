#!/usr/bin/env python
"""How visible is each attack to a human, measured against its own clean render.

Perceptibility has to be measured on a perceptual scale, not an Lp scale, or
"imperceptible" is just an assertion.  Three quantities per cell:

  SSIM            structural agreement (Wang 2004 parameters)
  dE2000 mean     average CIELAB colour difference; the JND is ~1
  f_JND           fraction of pixels whose dE2000 exceeds 1.0

f_JND is the honest headline: a global low-amplitude texture and a small opaque
decoy can share a mean dE, yet one is invisible everywhere and the other is
plainly visible in one corner.  The p99 separates those two geometries.
"""
import argparse, json, sys
from pathlib import Path

import numpy as np
from PIL import Image
from skimage.color import deltaE_ciede2000, rgb2lab
from skimage.metrics import structural_similarity as ssim


def measure(clean_png, att_png):
    a = np.asarray(Image.open(clean_png).convert("RGB"), dtype=np.uint8)
    b = np.asarray(Image.open(att_png).convert("RGB"), dtype=np.uint8)
    if a.shape != b.shape:
        return {"geometry_changed": True}
    s = ssim(a, b, channel_axis=2, data_range=255, gaussian_weights=True,
             sigma=1.5, use_sample_covariance=False)
    dE = deltaE_ciede2000(rgb2lab(a / 255.0), rgb2lab(b / 255.0))
    return {"geometry_changed": False,
            "ssim": round(float(s), 6),
            "dE_mean": round(float(dE.mean()), 6),
            "dE_p99": round(float(np.percentile(dE, 99)), 6),
            "f_JND": round(float((dE > 1.0).mean()), 6)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clean", required=True)
    ap.add_argument("--pert", required=True)
    ap.add_argument("--conds", nargs="+", required=True)
    ap.add_argument("--ids", required=True)
    ap.add_argument("--sample", type=int, default=250)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    ids = [x.strip() for x in Path(a.ids).read_text().split() if x.strip()]
    rng = np.random.default_rng(20260922)
    if a.sample and len(ids) > a.sample:
        ids = list(rng.choice(ids, a.sample, replace=False))

    clean = Path(a.clean)
    out = {}
    for c in a.conds:
        d = Path(a.pert) / c
        rows = []
        for i, sid in enumerate(ids):
            cp, ap_ = clean / f"{sid}.png", d / f"{sid}.png"
            if not (cp.exists() and ap_.exists()):
                continue
            try:
                rows.append(measure(cp, ap_))
            except Exception as e:
                print(f"  {c}/{sid}: {type(e).__name__}", file=sys.stderr)
            if (i + 1) % 100 == 0:
                print(f"  {c} {i+1}/{len(ids)}", file=sys.stderr, flush=True)
        good = [r for r in rows if not r.get("geometry_changed")]
        if not good:
            out[c] = {"n": 0}
            continue
        agg = {"n": len(good)}
        for k in ("ssim", "dE_mean", "dE_p99", "f_JND"):
            v = np.array([r[k] for r in good])
            agg[k] = {"mean": round(float(v.mean()), 6),
                      "median": round(float(np.median(v)), 6),
                      "p95": round(float(np.percentile(v, 95)), 6)}
        out[c] = agg
        print(f"{c:<5} n={agg['n']:<4} SSIM={agg['ssim']['median']:.4f}  "
              f"dE_mean={agg['dE_mean']['median']:.2f}  "
              f"dE_p99={agg['dE_p99']['median']:.2f}  "
              f"f_JND={agg['f_JND']['median']:.4f}", file=sys.stderr)

    Path(a.out).write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(f"-> {a.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
