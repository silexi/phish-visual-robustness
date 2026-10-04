#!/usr/bin/env python
"""Three remaining measurements for the audit re-run.

 1. how far above 1 the float32 and float64 colour self-correlations actually go
 2. whether the ranking of the eight configurations survives the CLIP re-clustering
 3. the claim (iv) maxima restricted to the inversion condition, which is the
    condition the paper's argument rests on

Usage: audit_u_extra.py u|f
"""
import json
import sys
from pathlib import Path

import numpy as np

R = Path("/Users/khomeassist/phish-vlm")
RUNGS = ["P_color", "P5_triv8", "P_ssim_global", "P3_hash", "P_lbp",
         "P0_fusion", "E1_clip", "P4_hog"]
SHORT = {"P_color": "colour", "P5_triv8": "8x8", "P_ssim_global": "SSIM*",
         "P3_hash": "hash", "P_lbp": "LBP", "P0_fusion": "fusion",
         "E1_clip": "CLIP", "P4_hog": "HOG"}
CONDS = ["A0n", "A1", "A3", "A4", "S1", "S3", "S7"]


def cmean(h, cl):
    return float(np.mean([h[cl == k].mean() for k in sorted(set(cl.tolist()))]))


def main():
    run = sys.argv[1]
    Z = dict(np.load(R / ("audit_hits_u.npz" if run == "u" else "audit_hits_f2.npz"),
                     allow_pickle=True))
    ccl, clipcl = Z["clean_cl_dhash"], Z["clean_cl_clip"]
    out = {}
    print("=" * 78)
    print(f"RUN {run}  supplementary measurements")
    print("=" * 78)

    print()
    print("1. colour self-correlation above 1 (a render against itself must give exactly 1)")
    s32 = np.concatenate([Z[f"{c}__old__selfsim__P_color"] for c in CONDS])
    s64 = np.concatenate([Z[f"{c}__new__selfsim__P_color"] for c in CONDS])
    for nm, s in (("float32 two-pass (first pass)", s32), ("float64 centred once", s64)):
        ex = s - 1.0
        print(f"   {nm:<30} rows {len(s)}  >1: {int((ex > 0).sum())}  "
              f"max excess {ex.max():.3e}  min {ex.min():.3e}  "
              f"rows |1-self| > 1e-4: {int((np.abs(ex) > 1e-4).sum())}")
        out[nm] = {"rows": int(len(s)), "gt1": int((ex > 0).sum()),
                   "max_excess": float(ex.max()), "min_excess": float(ex.min()),
                   "rows_off_by_more_than_eps": int((np.abs(ex) > 1e-4).sum())}

    print()
    print("2. does the ranking of the eight configurations survive the CLIP re-clustering?")
    out["ranking"] = {}
    for c in CONDS:
        m = Z[f"{c}__applied"] if not c.startswith("A0") else np.ones(len(Z[f"{c}__applied"]), bool)
        t = Z[f"{c}__truth"][m]
        cld, clc = ccl[t], clipcl[t]
        vd = {rn: cmean(Z[f"{c}__hit_dhash__{rn}"][m].astype(float), cld) for rn in RUNGS}
        vc = {rn: cmean(Z[f"{c}__hit_clip__{rn}"][m].astype(float), clc) for rn in RUNGS}
        od = sorted(RUNGS, key=lambda r: vd[r])
        oc = sorted(RUNGS, key=lambda r: vc[r])
        same = od == oc
        print(f"   {c:<4} dHash: " + " < ".join(SHORT[r] for r in od))
        print(f"   {c:<4} CLIP : " + " < ".join(SHORT[r] for r in oc)
              + ("   IDENTICAL" if same else "   DIFFERS"))
        out["ranking"][c] = {"dhash": [SHORT[r] for r in od], "clip": [SHORT[r] for r in oc],
                             "identical": bool(same)}

    print()
    print("3. claim (iv) maxima restricted to the inversion condition A3")
    m = Z["A3__applied"]
    t = Z["A3__truth"][m]
    cld, clc = ccl[t], clipcl[t]
    mp = (0.0, None)
    mc = (0.0, None)
    for rn in RUNGS:
        hd = Z[f"A3__hit_dhash__{rn}"][m].astype(float)
        hc = Z[f"A3__hit_clip__{rn}"][m].astype(float)
        dp = float(hc.mean() - hd.mean())
        dc = cmean(hc, clc) - cmean(hd, cld)
        if abs(dp) > mp[0]:
            mp = (abs(dp), SHORT[rn])
        if abs(dc) > mc[0]:
            mc = (abs(dc), SHORT[rn])
    print(f"   A3 largest |change| page-weighted    {mp[0]:.6f} ({mp[1]})")
    print(f"   A3 largest |change| cluster-weighted {mc[0]:.6f} ({mc[1]})")
    out["A3_max_page"] = [round(mp[0], 6), mp[1]]
    out["A3_max_cluster"] = [round(mc[0], 6), mc[1]]

    (R / f"audit_redo_extra_{run}.json").write_text(json.dumps(out, indent=1))
    print(f"-> audit_redo_extra_{run}.json")


if __name__ == "__main__":
    main()
