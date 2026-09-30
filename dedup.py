#!/usr/bin/env python
"""Cluster near-duplicate renders so the effective sample size is honest.

Phishing kits are redeployed across many domains, so a corpus counted by page
over-states its independent sample size and every confidence interval built on
it is too narrow.  Pages are clustered by difference-hash Hamming distance on
the rendered screenshot -- which catches the same kit under a different domain,
different injected text, or a different random token, all of which leave the
layout intact.  One representative per cluster is kept, and cluster identity is
carried through analysis so the bootstrap resamples kits rather than pages.
"""
import argparse, json, sys
from pathlib import Path

import numpy as np
from PIL import Image


def dhash(path, size=16):
    """Row-wise gradient hash: robust to scale and mild colour shifts."""
    im = Image.open(path).convert("L").resize((size + 1, size), Image.Resampling.LANCZOS)
    a = np.asarray(im, dtype=np.int16)
    return np.packbits((a[:, 1:] > a[:, :-1]).ravel())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--ids", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--maxdist", type=int, default=24,
                    help="Hamming distance over 256 bits below which two renders "
                         "are treated as the same kit")
    a = ap.parse_args()

    ids = [x.strip() for x in Path(a.ids).read_text().split() if x.strip()]
    d = Path(a.dir)
    ids = [i for i in ids if (d / f"{i}.png").exists()]
    print(f"{len(ids)} görsel hashleniyor", file=sys.stderr)

    H = np.stack([dhash(d / f"{i}.png") for i in ids])
    bits = np.unpackbits(H, axis=1).astype(np.int8)

    # single-linkage over the Hamming graph, done greedily: kits form tight
    # clusters, so the greedy pass and a full linkage agree in practice.
    n = len(ids)
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    B = 512
    for s in range(0, n, B):
        blk = bits[s:s + B]
        dist = (blk[:, None, :] != bits[None, :, :]).sum(-1)
        for i in range(len(blk)):
            for j in np.flatnonzero(dist[i] <= a.maxdist):
                if j == s + i:
                    continue
                ri, rj = find(s + i), find(int(j))
                if ri != rj:
                    parent[max(ri, rj)] = min(ri, rj)

    clusters = {}
    for i, sid in enumerate(ids):
        clusters.setdefault(find(i), []).append(sid)

    reps = {v[0]: f"k{k}" for k, v in clusters.items()}
    assign = {sid: f"k{k}" for k, v in clusters.items() for sid in v}

    Path(a.out).write_text(json.dumps(
        {"clusters": assign, "representatives": sorted(reps)}, indent=1))
    sizes = sorted((len(v) for v in clusters.values()), reverse=True)
    print(f"{n} sayfa -> {len(clusters)} kit kumesi", file=sys.stderr)
    print(f"  en buyuk kumeler: {sizes[:10]}", file=sys.stderr)
    print(f"  tek uyeli kume: {sum(1 for s in sizes if s == 1)}", file=sys.stderr)
    Path(a.out).with_suffix(".ids.txt").write_text("\n".join(sorted(reps)))
    print(f"-> {a.out} ve {Path(a.out).with_suffix('.ids.txt')}", file=sys.stderr)


if __name__ == "__main__":
    main()
