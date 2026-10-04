#!/usr/bin/env python
"""Verify that a row-chunked _hash_sim is bit-identical to the released one, and time it."""
import sys, time
from pathlib import Path
import numpy as np

R = Path("/Users/khomeassist/phish-vlm")
sys.path.insert(0, str(R))
from score import _hash_sim

q = dict(np.load(R / "feat_u_A3.npz", allow_pickle=True))
r = dict(np.load(R / "feat_u_A0.npz", allow_pickle=True))


def hash_sim_chunked(hq, wq, hr, bs=96):
    nq = len(hq)
    out = np.empty((nq, len(hr)), dtype=np.float32)
    wsum = wq.sum(-1)
    for i in range(0, nq, bs):
        j = min(i + bs, nq)
        mism = (hq[i:j, None, :] != hr[None, :, :])
        out[i:j] = 1.0 - (mism * wq[i:j, None, :]).sum(-1) / wsum[i:j, None]
    return out


n = 120
hq = q["px_hash"][:n]
wq = q["px_hw"][:n]
hr = r["px_hash"]

t = time.time()
A = _hash_sim(hq, wq, hr)
t1 = time.time() - t
t = time.time()
B = hash_sim_chunked(hq, wq, hr)
t2 = time.time() - t
print("subset n=%d  released %.2fs  chunked %.2fs" % (n, t1, t2))
print("dtypes", A.dtype, B.dtype, "bit-identical:", bool((A.tobytes() == B.tobytes())),
      "max|diff| =", float(np.abs(A.astype(np.float64) - B.astype(np.float64)).max()))

t = time.time()
F = hash_sim_chunked(q["px_hash"], q["px_hw"], hr)
print("full chunked shape", F.shape, "in %.1fs" % (time.time() - t))
