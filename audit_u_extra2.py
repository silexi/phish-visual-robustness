#!/usr/bin/env python
"""The cleanest statement of the colour artefact: rows where the two renders are
identical, so the colour self-correlation must be exactly 1.0.

The null condition A0n applies an inert style; on the pages where it changed no
pixel at all (frac_changed == 0) the perturbed render is the clean render, and a
correct Pearson correlation of a histogram with itself is exactly 1.

Usage: audit_u_extra2.py u|f
"""
import json
import sys
from pathlib import Path

import numpy as np

R = Path("/Users/khomeassist/phish-vlm")


def main():
    run = sys.argv[1]
    Z = dict(np.load(R / ("audit_hits_u.npz" if run == "u" else "audit_hits_f2.npz"),
                     allow_pickle=True))
    frac = Z["A0n__frac_changed"]
    m = frac == 0.0
    out = {"n_A0n_pages": int(len(frac)), "n_pixel_identical": int(m.sum())}
    print(f"RUN {run}: A0n has {len(frac)} pages, {int(m.sum())} of them pixel-identical "
          f"to the clean render")
    for nm, key in (("float32 two-pass (first pass)", "old"), ("float64 centred once", "new")):
        ss = Z[f"A0n__{key}__selfsim__P_color"][m]
        mx = Z[f"A0n__{key}__maxsim__P_color"][m]
        gap = mx - ss
        e = ss - 1.0
        print(f"  {nm:<30} self-correlation: exactly 1.0 on {int((ss == 1.0).sum())}/"
              f"{int(m.sum())} rows; max {ss.max():.10f}; min {ss.min():.10f}; "
              f"max |self-1| {np.abs(e).max():.3e}")
        print(f"  {' ':<30} rows where another page beats the identical render "
              f"(gap>0): {int((gap > 0).sum())}; max gap {gap.max():.3e}; "
              f"rows with gap > 1e-4: {int((gap > 1e-4).sum())}")
        out[nm] = {"exactly_one": int((ss == 1.0).sum()), "n": int(m.sum()),
                   "max_self": float(ss.max()), "min_self": float(ss.min()),
                   "max_abs_dev": float(np.abs(e).max()),
                   "rows_beaten": int((gap > 0).sum()), "max_gap": float(gap.max()),
                   "rows_gap_gt_eps": int((gap > 1e-4).sum())}
    (R / f"audit_redo_extra2_{run}.json").write_text(json.dumps(out, indent=1))
    print(f"-> audit_redo_extra2_{run}.json")


if __name__ == "__main__":
    main()
