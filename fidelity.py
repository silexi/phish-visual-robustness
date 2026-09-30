#!/usr/bin/env python
"""Render-fidelity gate: does the page look like a styled web page, or like raw HTML?

An archived kit whose external stylesheet is dead renders as default-UA HTML:
a white canvas, black text, blue links.  Counting distinct colours does not
separate that from a genuinely minimalist login page -- antialiasing and link
blue alone reach ten quantised bins.  What does separate them is AREA: a styled
page paints large mid-tone or saturated regions (headers, cards, buttons), while
an unstyled page's non-white pixels are thin glyph strokes.

`ink_area` is the fraction of pixels that are neither near-white nor part of a
thin dark stroke; `block_area` is the fraction belonging to a solid colour run
at least 24 px wide.  Both are reported so the gate can be justified rather than
asserted, and both are kept as covariates because the surviving pages may be
systematically more self-contained than the ones dropped.
"""
import argparse, json, sys
from pathlib import Path

import numpy as np
from PIL import Image


def fidelity(png, min_run=24):
    im = Image.open(png).convert("RGB")
    a = np.asarray(im, dtype=np.int16)
    mx, mn = a.max(2), a.min(2)
    lum = a.mean(2)

    near_white = (mn >= 238)
    saturated = (mx - mn) >= 28
    midtone = (lum >= 55) & (lum <= 225)
    ink = (~near_white) & (saturated | midtone)
    ink_area = float(ink.mean())

    # solid horizontal runs: a styled page has wide bands of one colour that are
    # not the page background; glyph strokes never form runs this wide.
    q = (a // 16).astype(np.int16)
    same = np.all(q[:, 1:, :] == q[:, :-1, :], axis=2)
    block = np.zeros(a.shape[:2], dtype=bool)
    run = np.zeros(a.shape[0], dtype=np.int32)
    for x in range(same.shape[1]):
        run = np.where(same[:, x], run + 1, 0)
        hit = run >= min_run
        if hit.any():
            idx = np.flatnonzero(hit)
            block[idx, max(0, x - min_run):x + 1] = True
    block &= ~near_white
    block_area = float(block.mean())

    return {"ink_area": round(ink_area, 5), "block_area": round(block_area, 5)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--ids", default=None)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    d = Path(a.dir)
    if a.ids:
        ids = [x.strip() for x in Path(a.ids).read_text().split() if x.strip()]
        paths = [d / f"{i}.png" for i in ids]
    else:
        paths = sorted(d.glob("*.png"))
    paths = [p for p in paths if p.exists()]

    rows = []
    for i, p in enumerate(paths):
        try:
            rows.append({"safe": p.stem, **fidelity(p)})
        except Exception as e:
            rows.append({"safe": p.stem, "error": str(e)[:80]})
        if (i + 1) % 200 == 0:
            print(f"  {i+1}/{len(paths)}", file=sys.stderr, flush=True)

    with open(a.out, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")

    ok = [r for r in rows if "ink_area" in r]
    for key in ("ink_area", "block_area"):
        v = sorted(r[key] for r in ok)
        qs = [v[int(q * (len(v) - 1))] for q in (0, .1, .25, .5, .75, .9, 1)]
        print(f"{key:<11} " + "  ".join(f"{x:.4f}" for x in qs)
              + "   (min p10 p25 med p75 p90 max)", file=sys.stderr)
    for t in (0.02, 0.04, 0.06, 0.08, 0.12):
        n = sum(1 for r in ok if r["block_area"] >= t)
        print(f"  block_area >= {t:.2f}: {n}/{len(ok)}", file=sys.stderr)


if __name__ == "__main__":
    main()
