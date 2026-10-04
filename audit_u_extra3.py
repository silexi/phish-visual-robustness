#!/usr/bin/env python
"""Airtight version of the colour artefact: rows whose colour histogram is
byte-identical to the clean one, so Pearson(h, h) must be exactly 1.0.

The manifest's frac_changed is written with five decimals, so a cell reported as
0.0 can still differ in a handful of pixels; selecting on byte-equality of the
stored 512-bin histogram removes that ambiguity.

Usage: audit_u_extra3.py u|f
"""
import json
import sys
from pathlib import Path

import numpy as np

R = Path("/Users/khomeassist/phish-vlm")
CONDS = ["A0n", "A1", "A3", "A4", "S1", "S3", "S7"]


def hist_corr32(A, B):
    """score._hist_corr: two-pass Pearson left in the input dtype (float32)."""
    a = A - A.mean(1, keepdims=True)
    b = B - B.mean(1, keepdims=True)
    num = a @ b.T
    den = np.sqrt((a * a).sum(1)[:, None] * (b * b).sum(1)[None, :])
    return np.divide(num, den, out=np.zeros_like(num), where=den > 0)


def hist_corr64(A, B):
    """retention2.hist_corr64: the same formula accumulated in float64."""
    a = A.astype(np.float64)
    b = B.astype(np.float64)
    a -= a.mean(1, keepdims=True)
    b -= b.mean(1, keepdims=True)
    num = a @ b.T
    den = np.sqrt((a * a).sum(1)[:, None] * (b * b).sum(1)[None, :])
    return np.divide(num, den, out=np.zeros_like(num), where=den > 0)


def main():
    run = sys.argv[1]
    suf = "_u" if run == "u" else ""
    clean = np.load(R / f"feat{suf}_A0.npz", allow_pickle=True)
    cids = [str(x) for x in clean["ids"]]
    pos = {s: i for i, s in enumerate(cids)}
    C = clean["px_color"]
    out = {}
    print(f"RUN {run}: gallery {len(cids)} pages")
    tot_id = 0
    tot_bad32 = 0
    worst32 = 0.0
    worst64 = 0.0
    for c in CONDS:
        q = np.load(R / f"feat{suf}_{c}.npz", allow_pickle=True)
        qids = [str(x) for x in q["ids"]]
        Q = q["px_color"]
        idx = [(i, pos[s]) for i, s in enumerate(qids) if s in pos]
        ident = [(i, j) for i, j in idx if Q[i].tobytes() == C[j].tobytes()]
        if not ident:
            print(f"  {c}: no byte-identical colour histogram")
            out[c] = {"n_identical": 0}
            continue
        qi = np.array([i for i, _ in ident])
        ci = np.array([j for _, j in ident])
        s32 = np.array([hist_corr32(Q[i:i + 1], C[j:j + 1])[0, 0] for i, j in ident])
        s64 = np.array([hist_corr64(Q[i:i + 1], C[j:j + 1])[0, 0] for i, j in ident])
        bad32 = int((s32 != 1.0).sum())
        bad64 = int((s64 != 1.0).sum())
        d32 = float(np.abs(s32 - 1.0).max())
        d64 = float(np.abs(s64 - 1.0).max())
        tot_id += len(ident)
        tot_bad32 += bad32
        worst32 = max(worst32, float(s32.max()))
        worst64 = max(worst64, float(s64.max()))
        print(f"  {c}: {len(ident):5d} rows with a byte-identical colour histogram | "
              f"float32 self-corr != 1 on {bad32:5d} rows, max |dev| {d32:.3e}, "
              f"max value {s32.max():.10f} | float64 != 1 on {bad64} rows, "
              f"max |dev| {d64:.3e}")
        out[c] = {"n_identical": len(ident), "float32_not_one": bad32,
                  "float32_max_abs_dev": d32, "float32_max_value": float(s32.max()),
                  "float64_not_one": bad64, "float64_max_abs_dev": d64,
                  "float64_max_value": float(s64.max())}
    print(f"  TOTAL over the seven conditions: {tot_id} rows whose colour histogram is "
          f"byte-identical to the clean one; float32 gives a self-correlation other than "
          f"exactly 1 on {tot_bad32} of them ({100*tot_bad32/max(tot_id,1):.1f}%); "
          f"largest value {worst32:.10f}")
    out["total"] = {"n_identical": tot_id, "float32_not_one": tot_bad32,
                    "float32_max_value": worst32, "float64_max_value": worst64}
    (R / f"audit_redo_extra3_{run}.json").write_text(json.dumps(out, indent=1))
    print(f"-> audit_redo_extra3_{run}.json")


if __name__ == "__main__":
    main()
