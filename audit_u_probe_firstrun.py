#!/usr/bin/env python
"""Reproduce the two published claim-(i) numbers from the September audit artefact.

audit_hits.npz was written by audit_hits.py on 22 September from the first-pass
retention.py matrices; audit2.py read it to produce the 28% and 71.8% figures.
This reads the same file so that the first run's published numbers are quoted
from their own artefact, not recomputed.
"""
import json
from collections import Counter
from pathlib import Path

import numpy as np

R = Path("/Users/khomeassist/phish-vlm")
Z = dict(np.load(R / "audit_hits.npz", allow_pickle=True))
clean_ids = [str(x) for x in Z["clean_ids"]]
cmap = json.load(open(R / "clusters_f.json"))["clusters"]
ccl = np.array([cmap.get(s, s) for s in clean_ids])
sz = Counter(ccl.tolist())
ng = len(clean_ids)
big2 = [k for k, _ in sz.most_common(2)]

print("source: audit_hits.npz (22 September) + clusters_f.json")
print(f"gallery {ng} pages, {len(sz)} kit clusters")
print(f"two largest kits {big2} sizes {[sz[k] for k in big2]} -> "
      f"share of total page weight = {sum(sz[k] for k in big2)}/{ng} = "
      f"{sum(sz[k] for k in big2)/ng:.6f}")

tag = "A3_filt"            # inversion, applied-only, exactly as audit2.py read it
ids = [str(x) for x in Z[tag + "__ids"]]
qcl = np.array([cmap.get(s, s) for s in ids])
for rn in ["P3_hash", "P0_fusion", "P_ssim", "P_lbp", "E1_clip", "P4_hog"]:
    t1 = Z[f"{tag}__top1__{rn}"]
    h = (ccl[t1] == qcl).astype(float)
    per = Counter()
    for k, v in zip(qcl, h):
        per[k] += v
    tot = h.sum()
    k, v = per.most_common(1)[0]
    print(f"  {rn:<11} n={len(ids)} hits={tot:.0f} page_mean={h.mean():.6f}  "
          f"largest single kit {k} (n={sz[k]}) contributes {v:.0f} hits = "
          f"{v/tot:.6f} of all hits")
