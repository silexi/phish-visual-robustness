#!/usr/bin/env python
"""Re-run the four claims of the independent statistical audit.

Usage:  audit_u_claims.py u|f   (reads audit_hits_u.npz / audit_hits_f2.npz)

Claims, as printed in the manuscript's audit subsection:
 (i)   page-weighted means gave 28% of the total weight to two kits, and under
       inversion 71.8% of hash's hits came from a single kit; moving to cluster
       weighting narrowed the confidence intervals.
 (ii)  outside inversion the colour column was a numerical artefact: two-pass
       Pearson over float32 histograms, catastrophic cancellation.
 (iii) three ordering claims failed the paired test and were withdrawn.
 (iv)  rebuilding the clusters from CLIP embeddings changed no retention value
       by more than 0.01.

Estimand and bootstrap are the paper's: mean over clusters of the per-cluster
mean, cluster bootstrap B=4000, seed 20260922, applied-only cells, eps=1e-4.
"""
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

R = Path("/Users/khomeassist/phish-vlm")
B = 4000
SEED = 20260922
EPS = 1e-4
RUNGS = ["P_color", "P5_triv8", "P_ssim_global", "P3_hash", "P_lbp",
         "P0_fusion", "E1_clip", "P4_hog"]
SHORT = {"P_color": "colour", "P5_triv8": "8x8", "P_ssim_global": "SSIM*",
         "P3_hash": "hash", "P_lbp": "LBP", "P0_fusion": "fusion",
         "E1_clip": "CLIP", "P4_hog": "HOG"}
ATTACK = ["A1", "A3", "A4", "S1", "S3", "S7"]
CONDS = ["A0n"] + ATTACK


def cluster_mean(h, cl):
    keys = sorted(set(cl.tolist()))
    return float(np.mean([h[cl == k].mean() for k in keys]))


def boot_cluster_weighted(h, cl):
    keys = np.array(sorted(set(cl.tolist())))
    per = np.array([h[cl == k].mean() for k in keys])
    rng = np.random.default_rng(SEED)
    draws = np.array([per[rng.integers(0, len(per), len(per))].mean() for _ in range(B)])
    return float(per.mean()), float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


def boot_page_weighted(h, cl):
    """Page-weighted mean with the same cluster bootstrap (the first-pass estimator)."""
    keys = np.array(sorted(set(cl.tolist())))
    idx = [np.flatnonzero(cl == k) for k in keys]
    rng = np.random.default_rng(SEED)
    draws = np.empty(B)
    for b in range(B):
        pick = rng.integers(0, len(keys), len(keys))
        draws[b] = h[np.concatenate([idx[i] for i in pick])].mean()
    return float(h.mean()), float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


def paired_diff(d, cl):
    """The paper's paired test (retention_excl.paired_diff)."""
    keys = np.array(sorted(set(cl.tolist())))
    per = np.array([d[cl == k].mean() for k in keys])
    rng = np.random.default_rng(SEED)
    draws = np.array([per[rng.integers(0, len(per), len(per))].mean() for _ in range(B)])
    lo, hi = float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))
    return {"diff": float(per.mean()), "lo": lo, "hi": hi,
            "n_clusters": int(len(keys)),
            "pos": int((per > 0).sum()), "neg": int((per < 0).sum()),
            "supported": bool(lo > 0 or hi < 0)}


def main():
    run = sys.argv[1]
    Z = dict(np.load(R / ("audit_hits_u.npz" if run == "u" else "audit_hits_f2.npz"),
                     allow_pickle=True))
    clean_ids = [str(x) for x in Z["clean_ids"]]
    ccl = Z["clean_cl_dhash"]
    clipcl = Z["clean_cl_clip"]
    thr = float(Z["clip_threshold"][0])
    ng = len(clean_ids)
    out = {"run": run, "n_gallery": ng}

    def sel(c, applied_only=True):
        a = Z[f"{c}__applied"]
        return a if applied_only else np.ones(len(a), bool)

    def cl_of(c, m, which="dhash"):
        t = Z[f"{c}__truth"]
        return (ccl if which == "dhash" else clipcl)[t][m]

    def hit_new(c, rn, m, which="dhash"):
        return Z[f"{c}__hit_{which}__{rn}"][m].astype(float)

    def hit_old(c, rn, m):
        t1 = Z[f"{c}__old__top1__{rn}"]
        own = ccl[Z[f"{c}__truth"]]
        return (ccl[t1] == own)[m].astype(float)

    bar = "=" * 78
    print(bar)
    print(f"RUN {run}: {ng} gallery pages, {len(set(ccl.tolist()))} dHash kit clusters")
    print(bar)

    # ================================================================ CLAIM i
    print()
    print("CLAIM (i)  UNIT OF AVERAGING")
    print("-" * 78)
    sz = Counter(ccl.tolist())
    big = sz.most_common(2)
    share2 = sum(v for _, v in big) / ng
    print(f"i-a  two largest kit clusters: {big[0][0]} n={big[0][1]}, {big[1][0]} n={big[1][1]}")
    print(f"     share of total page weight = ({big[0][1]}+{big[1][1]})/{ng} = {share2:.6f} "
          f"= {100*share2:.1f}%")
    out["i_two_largest"] = {"clusters": [big[0][0], big[1][0]],
                            "sizes": [big[0][1], big[1][1]],
                            "n_gallery": ng, "share": round(share2, 6)}
    print(f"     all cluster sizes, descending (top 12): "
          f"{[v for _, v in sz.most_common(12)]}; singletons "
          f"{sum(1 for v in sz.values() if v == 1)}")

    print()
    print("i-b  share of a configuration's A3 (inversion) hits held by ONE kit, applied-only")
    out["i_single_kit_share"] = {}
    for rule, hf in (("first-pass (argmax, float32)", hit_old),
                     ("corrected (eps=1e-4, float64)", hit_new)):
        print(f"     rule = {rule}")
        for rn in ["P3_hash", "P0_fusion", "P_lbp", "P_ssim_global", "E1_clip", "P4_hog"]:
            if rule.startswith("first") and rn == "P_ssim_global":
                rnx = "P_ssim"
            else:
                rnx = rn
            m = sel("A3")
            try:
                h = hf("A3", rnx, m)
            except KeyError:
                continue
            cl = cl_of("A3", m)
            tot = h.sum()
            per = Counter()
            for k, v in zip(cl, h):
                per[k] += v
            top = per.most_common(1)[0]
            s2 = sum(per[k] for k, _ in big) / max(tot, 1e-9)
            print(f"       {SHORT[rn]:<7} hits={tot:7.0f}/{len(h)}  page_mean={h.mean():.4f}  "
                  f"largest single kit {top[0]} = {top[1]:.0f} hits = "
                  f"{100*top[1]/max(tot,1e-9):.1f}%   two largest kits = {100*s2:.1f}%")
            out["i_single_kit_share"].setdefault(rule, {})[SHORT[rn]] = {
                "hits": float(tot), "n": int(len(h)),
                "top_kit": str(top[0]), "top_kit_hits": float(top[1]),
                "top_kit_share": round(float(top[1] / max(tot, 1e-9)), 6),
                "two_largest_share": round(float(s2), 6)}

    print()
    print("i-c  interval width, page weighting vs cluster weighting (corrected rule,")
    print("     applied-only, same cluster bootstrap B=4000 seed 20260922)")
    print("     cond rung      page_mean  page_CI            clus_mean  clus_CI            "
          "w_page  w_clus  ratio")
    widths = []
    out["i_interval_widths"] = {}
    for c in ATTACK:
        m = sel(c)
        cl = cl_of(c, m)
        for rn in RUNGS:
            h = hit_new(c, rn, m)
            pm, plo, phi = boot_page_weighted(h, cl)
            cm, clo, chi = boot_cluster_weighted(h, cl)
            wp, wc = phi - plo, chi - clo
            widths.append((c, rn, wp, wc))
            print(f"     {c:<4} {SHORT[rn]:<8}  {pm:8.4f}  [{plo:.4f},{phi:.4f}]  "
                  f"{cm:8.4f}  [{clo:.4f},{chi:.4f}]  {wp:.4f}  {wc:.4f}  "
                  f"{(wc/wp if wp > 0 else float('nan')):.3f}")
            out["i_interval_widths"][f"{c}/{rn}"] = {
                "page_mean": round(pm, 6), "page_ci": [round(plo, 6), round(phi, 6)],
                "cluster_mean": round(cm, 6), "cluster_ci": [round(clo, 6), round(chi, 6)],
                "width_page": round(wp, 6), "width_cluster": round(wc, 6)}
    nar = sum(1 for _, _, wp, wc in widths if wc < wp)
    rat = np.array([wc / wp for _, _, wp, wc in widths if wp > 0])
    print(f"     -> cluster weighting narrower in {nar}/{len(widths)} cells; "
          f"median width ratio {np.median(rat):.3f}, mean {rat.mean():.3f}, "
          f"range [{rat.min():.3f},{rat.max():.3f}]")
    out["i_narrower"] = {"n_narrower": nar, "n_cells": len(widths),
                         "median_ratio": round(float(np.median(rat)), 6),
                         "min_ratio": round(float(rat.min()), 6),
                         "max_ratio": round(float(rat.max()), 6)}

    # =============================================================== CLAIM ii
    print()
    print("CLAIM (ii)  THE COLOUR COLUMN AS A float32 ARTEFACT")
    print("-" * 78)
    out["ii"] = {}
    worst = (0.0, None)
    n_over = 0
    n_rows = 0
    for c in CONDS:
        ss = Z[f"{c}__old__selfsim__P_color"]
        n_rows += len(ss)
        n_over += int((ss > 1.0).sum())
        if ss.max() > worst[0]:
            worst = (float(ss.max()), c)
    print(f"ii-a float32 two-pass Pearson, self-correlation of a render with itself:")
    print(f"     rows with self-correlation > 1 : {n_over} of {n_rows} over all 7 conditions")
    print(f"     largest self-correlation seen  : {worst[0]:.8f} (condition {worst[1]})")
    ss64 = np.concatenate([Z[f"{c}__new__selfsim__P_color"] for c in CONDS])
    print(f"     float64 counterpart, max self-correlation: {ss64.max():.8f}; "
          f"rows > 1: {int((ss64 > 1.0).sum())}")
    out["ii"]["float32_selfcorr_gt1"] = n_over
    out["ii"]["n_rows_all_conds"] = n_rows
    out["ii"]["float32_max_selfcorr"] = round(worst[0], 8)
    out["ii"]["float32_max_selfcorr_cond"] = worst[1]
    out["ii"]["float64_max_selfcorr"] = round(float(ss64.max()), 8)
    out["ii"]["float64_selfcorr_gt1"] = int((ss64 > 1.0).sum())

    print()
    print("ii-b null condition A0n, colour configuration, first-pass argmax rule:")
    m = np.ones(len(Z["A0n__applied"]), bool)
    hold = hit_old("A0n", "P_color", m)
    frac = Z["A0n__frac_changed"]
    ssA = Z["A0n__old__selfsim__P_color"]
    mxA = Z["A0n__old__maxsim__P_color"]
    miss = hold == 0
    gap = (mxA - ssA)[miss]
    print(f"     misses = {int(miss.sum())} of {len(hold)} pages")
    print(f"     of which on renders the inert style did not change at all "
          f"(frac_changed==0): {int((miss & (frac == 0)).sum())}")
    print(f"     median gap (row max - self) over misses: {np.median(gap):.3e}; "
          f"share of misses decided below 1e-4: {float(np.mean(gap < 1e-4)):.3f}; "
          f"below 1e-6: {float(np.mean(gap < 1e-6)):.3f}")
    hnew = hit_new("A0n", "P_color", m)
    print(f"     same cells under the corrected rule: misses = {int((hnew == 0).sum())}")
    out["ii"]["A0n_colour"] = {
        "n_pages": int(len(hold)), "misses_first_pass": int(miss.sum()),
        "misses_on_unchanged_renders": int((miss & (frac == 0)).sum()),
        "median_gap": float(np.median(gap)) if miss.sum() else None,
        "frac_gap_lt_1e-4": float(np.mean(gap < 1e-4)) if miss.sum() else None,
        "frac_gap_lt_1e-6": float(np.mean(gap < 1e-6)) if miss.sum() else None,
        "misses_corrected": int((hnew == 0).sum())}

    print()
    print("ii-c colour configuration retention, first-pass float32+argmax vs corrected")
    print("     float64+eps (applied-only; A0n over all pages, as the paper reports it)")
    print("     cond   n     first-pass page  first-pass clus   corrected page  corrected clus")
    out["ii"]["colour_by_condition"] = {}
    for c in CONDS:
        m = sel(c, applied_only=(c != "A0n"))
        cl = cl_of(c, m)
        ho, hn = hit_old(c, "P_color", m), hit_new(c, "P_color", m)
        row = (float(ho.mean()), cluster_mean(ho, cl), float(hn.mean()), cluster_mean(hn, cl))
        print(f"     {c:<5} {len(ho):<5} {row[0]:15.4f} {row[1]:17.4f} "
              f"{row[2]:16.4f} {row[3]:15.4f}")
        out["ii"]["colour_by_condition"][c] = {
            "n": int(len(ho)), "first_pass_page": round(row[0], 6),
            "first_pass_cluster": round(row[1], 6),
            "corrected_page": round(row[2], 6), "corrected_cluster": round(row[3], 6)}

    # ============================================================== CLAIM iii
    print()
    print("CLAIM (iii)  ORDERING CLAIMS UNDER THE PAPER'S PAIRED TEST")
    print("-" * 78)
    print("test: mean over clusters of the per-cluster mean of the per-page hit")
    print("difference; cluster bootstrap B=4000 seed 20260922; an ordering is")
    print("SUPPORTED iff the 95% interval excludes 0.")
    out["iii"] = {"adjacent": {}, "a3_all_pairs": {}, "text_claims": {}}

    print()
    print("iii-a the two paired differences the manuscript's text quotes")
    for c, x, y in [("A3", "P4_hog", "E1_clip"), ("A4", "P_lbp", "P0_fusion")]:
        m = sel(c)
        cl = cl_of(c, m)
        d = hit_new(c, x, m) - hit_new(c, y, m)
        r = paired_diff(d, cl)
        print(f"     {c} {SHORT[x]}-{SHORT[y]}: {r['diff']:+.6f} "
              f"[{r['lo']:+.6f}, {r['hi']:+.6f}]  {r['n_clusters']} clusters, "
              f"{r['pos']} ahead, {r['neg']} behind  -> "
              f"{'SUPPORTED' if r['supported'] else 'NOT SUPPORTED'}")
        out["iii"]["text_claims"][f"{c}/{x}_minus_{y}"] = r

    print()
    print("iii-b every adjacent pair of the ranking of the eight configurations,")
    print("      per condition (7 adjacent orderings per condition, applied-only)")
    n_fail = 0
    n_tot = 0
    fails = []
    for c in ATTACK:
        m = sel(c)
        cl = cl_of(c, m)
        hs = {rn: hit_new(c, rn, m) for rn in RUNGS}
        order = sorted(RUNGS, key=lambda rn: cluster_mean(hs[rn], cl))
        vals = {rn: cluster_mean(hs[rn], cl) for rn in RUNGS}
        print(f"     -- {c}  ranking (low to high): "
              + " < ".join(f"{SHORT[rn]}({vals[rn]:.3f})" for rn in order))
        for lo_rn, hi_rn in zip(order, order[1:]):
            d = hs[hi_rn] - hs[lo_rn]
            r = paired_diff(d, cl)
            n_tot += 1
            tagstr = f"{c}: {SHORT[hi_rn]} > {SHORT[lo_rn]}"
            if not r["supported"]:
                n_fail += 1
                fails.append((tagstr, r))
            print(f"        {SHORT[hi_rn]:<7} - {SHORT[lo_rn]:<7} {r['diff']:+.6f} "
                  f"[{r['lo']:+.6f}, {r['hi']:+.6f}] {r['pos']:>4} ahead {r['neg']:>4} behind  "
                  f"{'ok' if r['supported'] else 'FAILS'}")
            out["iii"]["adjacent"][tagstr] = r
    print(f"     -> {n_fail} of {n_tot} adjacent orderings are not supported")
    out["iii"]["n_adjacent_failed"] = n_fail
    out["iii"]["n_adjacent_total"] = n_tot
    out["iii"]["adjacent_failed_list"] = [t for t, _ in fails]

    print()
    print("iii-c all 28 configuration pairs under inversion (A3), the condition that")
    print("      carries the paper's argument")
    m = sel("A3")
    cl = cl_of("A3", m)
    hs = {rn: hit_new("A3", rn, m) for rn in RUNGS}
    vals = {rn: cluster_mean(hs[rn], cl) for rn in RUNGS}
    order = sorted(RUNGS, key=lambda rn: vals[rn])
    nf3 = 0
    for i in range(len(order)):
        for j in range(i + 1, len(order)):
            a, b = order[j], order[i]          # a has the higher retention
            r = paired_diff(hs[a] - hs[b], cl)
            if not r["supported"]:
                nf3 += 1
            print(f"     {SHORT[a]:<7} > {SHORT[b]:<7} {r['diff']:+.6f} "
                  f"[{r['lo']:+.6f}, {r['hi']:+.6f}] {r['pos']:>4}/{r['neg']:<4} "
                  f"{'ok' if r['supported'] else 'FAILS'}")
            out["iii"]["a3_all_pairs"][f"{SHORT[a]}_gt_{SHORT[b]}"] = r
    print(f"     -> under A3, {nf3} of 28 pairwise orderings are not supported")
    out["iii"]["a3_n_failed"] = nf3

    # =============================================================== CLAIM iv
    print()
    print("CLAIM (iv)  CLUSTERS REBUILT FROM THE CLIP EMBEDDINGS")
    print("-" * 78)
    nk_clip = len(set(clipcl.tolist()))
    print(f"single-linkage on CLIP cosine similarity, threshold bisected to match the")
    print(f"kit-cluster count: threshold={thr:.6f} -> {nk_clip} clusters")
    cs = Counter(clipcl.tolist())
    print(f"sizes (top 8) {sorted(cs.values(), reverse=True)[:8]}; "
          f"singletons {sum(1 for v in cs.values() if v == 1)}")
    # agreement between the two partitions, over all page pairs
    _, a = np.unique(ccl, return_inverse=True)
    _, bb = np.unique(clipcl, return_inverse=True)
    iu = np.triu_indices(ng, 1)
    sd = a[iu[0]] == a[iu[1]]
    sc = bb[iu[0]] == bb[iu[1]]
    print(f"pairs of pages in the same kit under dHash: {int(sd.sum())}; under CLIP: "
          f"{int(sc.sum())}; under both: {int((sd & sc).sum())}; "
          f"Jaccard {(sd & sc).sum()/max((sd | sc).sum(),1):.4f}")
    out["iv"] = {"threshold": round(thr, 6), "n_clusters_clip": nk_clip,
                 "n_clusters_dhash": len(set(ccl.tolist())),
                 "pairs_same_dhash": int(sd.sum()), "pairs_same_clip": int(sc.sum()),
                 "pairs_same_both": int((sd & sc).sum()),
                 "jaccard": round(float((sd & sc).sum() / max((sd | sc).sum(), 1)), 6),
                 "cells": {}}
    print()
    print("     retention under the two clusterings (applied-only; A0n over all pages)")
    print("     cond rung      dHash page  CLIP page   d(page)   dHash clus  CLIP clus   d(clus)")
    mx_p = (0.0, None)
    mx_c = (0.0, None)
    for c in CONDS:
        m = sel(c, applied_only=(c != "A0n"))
        cld = cl_of(c, m, "dhash")
        clc = cl_of(c, m, "clip")
        for rn in RUNGS:
            hd = hit_new(c, rn, m, "dhash")
            hc = hit_new(c, rn, m, "clip")
            pd_, pc_ = float(hd.mean()), float(hc.mean())
            cd_, cc_ = cluster_mean(hd, cld), cluster_mean(hc, clc)
            dp, dc = pc_ - pd_, cc_ - cd_
            if abs(dp) > mx_p[0]:
                mx_p = (abs(dp), f"{c}/{SHORT[rn]}")
            if abs(dc) > mx_c[0]:
                mx_c = (abs(dc), f"{c}/{SHORT[rn]}")
            print(f"     {c:<4} {SHORT[rn]:<8}  {pd_:10.4f}  {pc_:10.4f}  {dp:+8.4f}   "
                  f"{cd_:10.4f}  {cc_:10.4f}  {dc:+8.4f}")
            out["iv"]["cells"][f"{c}/{rn}"] = {
                "dhash_page": round(pd_, 6), "clip_page": round(pc_, 6),
                "delta_page": round(dp, 6), "dhash_cluster": round(cd_, 6),
                "clip_cluster": round(cc_, 6), "delta_cluster": round(dc, 6)}
    print(f"     -> largest |change|, page-weighted    : {mx_p[0]:.6f}  ({mx_p[1]})")
    print(f"     -> largest |change|, cluster-weighted : {mx_c[0]:.6f}  ({mx_c[1]})")
    out["iv"]["max_abs_change_page"] = round(mx_p[0], 6)
    out["iv"]["max_abs_change_page_cell"] = mx_p[1]
    out["iv"]["max_abs_change_cluster"] = round(mx_c[0], 6)
    out["iv"]["max_abs_change_cluster_cell"] = mx_c[1]

    (R / f"audit_redo_{run}.json").write_text(json.dumps(out, indent=1))
    print()
    print(f"-> audit_redo_{run}.json")


if __name__ == "__main__":
    main()
