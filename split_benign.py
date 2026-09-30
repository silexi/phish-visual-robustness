#!/usr/bin/env python
"""Split the replayed benign pages into a brand-disjoint reference / calibration pair.

Brand-disjoint matters: if the same brand appeared in both halves, the threshold
would be calibrated against pages that are near-duplicates of the references it
is then scored against, and the false-positive rate would be optimistic in
exactly the way the audit found in the original work.
"""
import json, random, shutil
from pathlib import Path

REPLAY = Path("benign_replay")
FETCH = Path("benign/fetch_manifest.jsonl")

fetch = {}
for line in FETCH.open(encoding="utf-8"):
    r = json.loads(line)
    fetch[r["safe"]] = r

replay = [json.loads(l) for l in (REPLAY / "manifest_replay.jsonl").open(encoding="utf-8")]

# Keep a page if it replayed, rendered something with structure, and still shows
# a form.  A password field is NOT required: many legitimate providers use an
# email-first flow, and requiring one would make "has a password field" a class
# signal rather than a brand signal.
keep = []
for r in replay:
    if not r.get("ok"):
        continue
    meta = fetch.get(r["safe"], {})
    if not meta.get("ok"):
        continue
    if r.get("n_input", 0) < 1 or r.get("text_len", 0) < 40:
        continue
    if r.get("n_colors", 0) < 4:
        continue
    keep.append({**r, "brand": meta.get("brand", "?"), "url": meta.get("url", ""),
                 "title": meta.get("title", "")})

brands = sorted({k["brand"] for k in keep})
random.seed(20260922)
random.shuffle(brands)
half = len(brands) // 2
ref_brands, cal_brands = set(brands[:half]), set(brands[half:])

out = {"benign_ref": [k for k in keep if k["brand"] in ref_brands],
       "benign_cal": [k for k in keep if k["brand"] in cal_brands]}

for name, rows in out.items():
    d = Path(name)
    if d.exists():
        shutil.rmtree(d)
    d.mkdir()
    for r in rows:
        src = REPLAY / "shots" / f"{r['safe']}.png"
        if src.exists():
            shutil.copy(src, d / f"{r['safe']}.png")
    with open(f"{name}_manifest.json", "w", encoding="utf-8") as fh:
        json.dump(rows, fh, indent=1, ensure_ascii=False)

print(f"replay {len(replay)} -> tutulan {len(keep)}  ({len(brands)} marka)")
print(f"  referans : {len(out['benign_ref']):>3} sayfa / {len(ref_brands)} marka")
print(f"  kalibrasyon: {len(out['benign_cal']):>3} sayfa / {len(cal_brands)} marka")
n = len(out["benign_cal"])
print(f"  olculebilir en dusuk FPR = 1/{n} = {1/max(n,1):.4f}")
