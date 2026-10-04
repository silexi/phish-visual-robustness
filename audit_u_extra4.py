#!/usr/bin/env python
"""Every pair whose relative order differs between the dHash and the CLIP
clustering, put through the paper's paired test.

Claim (iv) is only interesting if the orderings the paper asserts do not depend
on the clustering.  For each condition the two rankings are compared, every
inverted pair is listed, and the paired test is run on that pair (on the primary
dHash clustering, applied-only) so it can be said whether the swap touches an
ordering the test resolves at all.

Usage: audit_u_extra4.py u|f
"""
import json
import sys
from pathlib import Path

import numpy as np

R = Path("/Users/khomeassist/phish-vlm")
B, SEED = 4000, 20260922
RUNGS = ["P_color", "P5_triv8", "P_ssim_global", "P3_hash", "P_lbp",
         "P0_fusion", "E1_clip", "P4_hog"]
SHORT = {"P_color": "colour", "P5_triv8": "8x8", "P_ssim_global": "SSIM*",
         "P3_hash": "hash", "P_lbp": "LBP", "P0_fusion": "fusion",
         "E1_clip": "CLIP", "P4_hog": "HOG"}
CONDS = ["A0n", "A1", "A3", "A4", "S1", "S3", "S7"]


def cmean(h, cl):
    return float(np.mean([h[cl == k].mean() for k in sorted(set(cl.tolist()))]))


def paired(d, cl):
    keys = np.array(sorted(set(cl.tolist())))
    per = np.array([d[cl == k].mean() for k in keys])
    rng = np.random.default_rng(SEED)
    draws = np.array([per[rng.integers(0, len(per), len(per))].mean() for _ in range(B)])
    lo, hi = float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))
    return float(per.mean()), lo, hi, bool(lo > 0 or hi < 0)


def main():
    run = sys.argv[1]
    Z = dict(np.load(R / ("audit_hits_u.npz" if run == "u" else "audit_hits_f2.npz"),
                     allow_pickle=True))
    ccl, clipcl = Z["clean_cl_dhash"], Z["clean_cl_clip"]
    out = {}
    print(f"RUN {run}: pairs whose order differs between the two clusterings")
    nres = 0
    for c in CONDS:
        m = Z[f"{c}__applied"] if not c.startswith("A0") else np.ones(len(Z[f"{c}__applied"]), bool)
        t = Z[f"{c}__truth"][m]
        cld, clc = ccl[t], clipcl[t]
        hd = {rn: Z[f"{c}__hit_dhash__{rn}"][m].astype(float) for rn in RUNGS}
        hc = {rn: Z[f"{c}__hit_clip__{rn}"][m].astype(float) for rn in RUNGS}
        vd = {rn: cmean(hd[rn], cld) for rn in RUNGS}
        vc = {rn: cmean(hc[rn], clc) for rn in RUNGS}
        inv = []
        for i, a in enumerate(RUNGS):
            for b in RUNGS[i + 1:]:
                sd = np.sign(vd[a] - vd[b])
                sc = np.sign(vc[a] - vc[b])
                if sd != sc:
                    inv.append((a, b))
        if not inv:
            print(f"  {c}: no pair changes order")
            out[c] = []
            continue
        out[c] = []
        for a, b in inv:
            diff, lo, hi, sup = paired(hd[a] - hd[b], cld)
            if sup:
                nres += 1
            print(f"  {c}: {SHORT[a]} vs {SHORT[b]}  dHash {vd[a]:.4f}/{vd[b]:.4f}  "
                  f"CLIP {vc[a]:.4f}/{vc[b]:.4f}  paired(dHash) {diff:+.6f} "
                  f"[{lo:+.6f}, {hi:+.6f}]  -> "
                  f"{'RESOLVED BY THE TEST' if sup else 'not resolved by the test'}")
            out[c].append({"a": SHORT[a], "b": SHORT[b],
                           "dhash": [round(vd[a], 6), round(vd[b], 6)],
                           "clip": [round(vc[a], 6), round(vc[b], 6)],
                           "paired_diff": round(diff, 6), "lo": round(lo, 6),
                           "hi": round(hi, 6), "supported": sup})
    print(f"  -> order-changing pairs that the paired test DOES resolve: {nres}")
    out["n_resolved_but_swapped"] = nres
    (R / f"audit_redo_extra4_{run}.json").write_text(json.dumps(out, indent=1))
    print(f"-> audit_redo_extra4_{run}.json")


if __name__ == "__main__":
    main()
