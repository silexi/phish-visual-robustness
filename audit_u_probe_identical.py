#!/usr/bin/env python
"""Two precision checks behind claim (ii).

(a) the perturbation manifests write frac_changed with five decimals, so a cell
    reported as 0.0 is not necessarily bit-identical; how many of the A0n cells
    reported as unchanged really carry an identical 512-bin colour histogram and
    an identical 256x256 grey thumbnail
(b) on exactly those rows whose colour histogram IS byte-identical to the clean
    one, what the colour self-correlation comes out as in the matrix the pipeline
    actually computes (float32, first pass) and in the corrected one (float64)

Usage: audit_u_probe_identical.py u|f
"""
import json
import sys
from pathlib import Path

import numpy as np

R = Path("/Users/khomeassist/phish-vlm")
run = sys.argv[1]
suf, nz, pert = ("_u", "audit_hits_u.npz", "pert_u") if run == "u" else ("", "audit_hits_f2.npz", "pert")

cl = np.load(R / f"feat{suf}_A0.npz", allow_pickle=True)
q = np.load(R / f"feat{suf}_A0n.npz", allow_pickle=True)
ci = [str(x) for x in cl["ids"]]
qi = [str(x) for x in q["ids"]]
pos = {s: i for i, s in enumerate(ci)}
kq = {s: i for i, s in enumerate(qi)}
C, Q = cl["px_color"], q["px_color"]
G, H = cl["px_gray256"], q["px_gray256"]
man = {json.loads(l)["safe"]: json.loads(l) for l in (R / pert / "manifest_A0n.jsonl").open()}

Z = np.load(R / nz, allow_pickle=True)
kept = [str(x) for x in Z["A0n__ids"]]
ss32 = Z["A0n__old__selfsim__P_color"]
ss64 = Z["A0n__new__selfsim__P_color"]
mx32 = Z["A0n__old__maxsim__P_color"]

print(f"RUN {run}: A0n, {len(kept)} cells")
z0 = np.array([float(man.get(s, {}).get("frac_changed", 1.0)) == 0.0 for s in kept])
hid = np.array([Q[kq[s]].tobytes() == C[pos[s]].tobytes() for s in kept])
gid = np.array([H[kq[s]].tobytes() == G[pos[s]].tobytes() for s in kept])
print(f"(a) manifest frac_changed == 0.0            : {int(z0.sum())}")
print(f"    of those, identical colour histogram    : {int((z0 & hid).sum())}")
print(f"    of those, identical grey thumbnail      : {int((z0 & gid).sum())}")
print(f"    identical colour histogram, any cell    : {int(hid.sum())}")
print(f"    -> frac_changed is written to five decimals, so {int((z0 & ~hid).sum())} cells "
      f"reported as unchanged do differ in a few pixels")
print(f"(b) rows with a byte-identical colour histogram: {int(hid.sum())}")
print(f"    float32 (first pass) self-correlation: max {ss32[hid].max():.10f}  "
      f"min {ss32[hid].min():.10f}  not exactly 1 on {int((ss32[hid] != 1.0).sum())} rows")
print(f"    float64 (corrected)  self-correlation: max {ss64[hid].max():.14f}  "
      f"max |self-1| {np.abs(ss64[hid]-1).max():.3e}")
print(f"    rows where another page strictly beats the identical render, float32: "
      f"{int((mx32[hid] > ss32[hid]).sum())}  (largest gap {(mx32[hid]-ss32[hid]).max():.3e}, "
      f"all below the 1e-4 tie tolerance: {bool((mx32[hid]-ss32[hid]).max() < 1e-4)})")
