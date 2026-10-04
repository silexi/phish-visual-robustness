#!/usr/bin/env python
"""Claim (iv) under the FIRST-PASS hit rule, i.e. exactly what audit4.py section 11 did.

audit4.py scored the dHash grouping against a CLIP-rebuilt grouping using the
first-pass rule (plain argmax over the float32 similarity matrices) and over the
six conditions A0n, A3, S1, S7, A4, S3.  This reproduces that comparison for
both runs so the published "no retention value changed by more than 0.01" can be
checked against its own estimator before being restated for the enlarged set.

Usage: audit_u_claim4_old.py u|f
"""
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

R = Path("/Users/khomeassist/phish-vlm")
RUNGS_OLD = ["P_color", "P5_triv8", "P_ssim", "P3_hash", "P_lbp",
             "P0_fusion", "E1_clip", "P4_hog", "P1_repaired"]
SHORT = {"P_color": "colour", "P5_triv8": "8x8", "P_ssim": "SSIM*", "P3_hash": "hash",
         "P_lbp": "LBP", "P0_fusion": "fusion", "E1_clip": "CLIP", "P4_hog": "HOG",
         "P1_repaired": "P1rep"}
AUDIT4_CONDS = ["A0n", "A3", "S1", "S7", "A4", "S3"]
ALL_CONDS = ["A0n", "A1", "A3", "A4", "S1", "S3", "S7"]


def cmean(h, cl):
    return float(np.mean([h[cl == k].mean() for k in sorted(set(cl.tolist()))]))


def main():
    run = sys.argv[1]
    Z = dict(np.load(R / ("audit_hits_u.npz" if run == "u" else "audit_hits_f2.npz"),
                     allow_pickle=True))
    ccl = Z["clean_cl_dhash"]
    clipcl = Z["clean_cl_clip"]
    print("=" * 78)
    print(f"RUN {run}: claim (iv) under the first-pass rule (argmax, float32), "
          f"audit4.py section 11")
    print(f"CLIP threshold {float(Z['clip_threshold'][0]):.6f}, "
          f"{len(set(clipcl.tolist()))} clusters vs {len(set(ccl.tolist()))} dHash clusters")
    print("=" * 78)
    print("     cond rung      dHash page  CLIP page   d(page)   dHash grp   CLIP grp    d(grp)")
    res = {}
    worst = {}
    for c in ALL_CONDS:
        if f"{c}__truth" not in Z:
            continue
        # audit4 scored the applied-only subsets ("_filt"), A0n over all pages
        m = Z[f"{c}__applied"] if not c.startswith("A0") else np.ones(len(Z[f"{c}__applied"]), bool)
        t = Z[f"{c}__truth"][m]
        cld, clc = ccl[t], clipcl[t]
        for rn in RUNGS_OLD:
            t1 = Z[f"{c}__old__top1__{rn}"][m]
            hd = (ccl[t1] == cld).astype(float)
            hc = (clipcl[t1] == clc).astype(float)
            pd_, pc_ = float(hd.mean()), float(hc.mean())
            gd_, gc_ = cmean(hd, cld), cmean(hc, clc)
            res[f"{c}/{rn}"] = {"dhash_page": round(pd_, 6), "clip_page": round(pc_, 6),
                                "delta_page": round(pc_ - pd_, 6),
                                "dhash_grp": round(gd_, 6), "clip_grp": round(gc_, 6),
                                "delta_grp": round(gc_ - gd_, 6)}
            print(f"     {c:<4} {SHORT[rn]:<8}  {pd_:10.4f}  {pc_:10.4f}  {pc_-pd_:+8.4f}   "
                  f"{gd_:10.4f}  {gc_:10.4f}  {gc_-gd_:+8.4f}")

    def mx(keys, field):
        best = (0.0, None)
        for k in keys:
            v = abs(res[k][field])
            if v > best[0]:
                best = (v, k)
        return best

    for label, conds in (("audit4 condition set (A0n A3 S1 S7 A4 S3)", AUDIT4_CONDS),
                         ("all seven conditions", ALL_CONDS)):
        keys8 = [f"{c}/{rn}" for c in conds for rn in RUNGS_OLD[:8] if f"{c}/{rn}" in res]
        keys9 = [f"{c}/{rn}" for c in conds for rn in RUNGS_OLD if f"{c}/{rn}" in res]
        for nm, kk in (("eight reported configurations", keys8),
                       ("all nine first-pass configurations", keys9)):
            a = mx(kk, "delta_page")
            b = mx(kk, "delta_grp")
            print(f"  -> {label} / {nm}:")
            print(f"       largest |change| page-weighted    {a[0]:.6f}  ({a[1]})")
            print(f"       largest |change| group-weighted   {b[0]:.6f}  ({b[1]})")
            worst[f"{label}|{nm}"] = {"page": [round(a[0], 6), a[1]],
                                      "grp": [round(b[0], 6), b[1]]}
    (R / f"audit_redo_claim4_old_{run}.json").write_text(
        json.dumps({"cells": res, "worst": worst}, indent=1))
    print(f"-> audit_redo_claim4_old_{run}.json")


if __name__ == "__main__":
    main()
