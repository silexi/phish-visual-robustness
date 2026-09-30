#!/usr/bin/env python
"""Tile rendered screenshots into one contact sheet for visual QA."""
import argparse, json, random
from pathlib import Path

from PIL import Image, ImageDraw

ap = argparse.ArgumentParser()
ap.add_argument("--dir", required=True)
ap.add_argument("--out", default="sheet.png")
ap.add_argument("--n", type=int, default=24)
ap.add_argument("--cols", type=int, default=6)
ap.add_argument("--seed", type=int, default=7)
a = ap.parse_args()

root = Path(a.dir)
rows = [json.loads(l) for m in sorted(root.glob("manifest*.jsonl"))
        for l in m.open(encoding="utf-8")]
vis = [r for r in rows if r.get("ok") and r.get("pw_visible") and r.get("text_len", 0) > 60]

seen, uniq = set(), []
for r in sorted(vis, key=lambda r: -r["n_colors"]):
    t = (r.get("title") or "")[:30]
    if t in seen:
        continue
    seen.add(t)
    uniq.append(r)

random.seed(a.seed)
half = a.n // 2
pick = uniq[:half] + (random.sample(uniq[half:], min(half, max(0, len(uniq) - half)))
                      if len(uniq) > half else [])

TW, TH, PAD = 320, 180, 16
rowsn = (len(pick) + a.cols - 1) // a.cols
sheet = Image.new("RGB", (a.cols * TW, rowsn * (TH + PAD)), "#1b1b1b")
d = ImageDraw.Draw(sheet)
for i, r in enumerate(pick):
    im = Image.open(root / "shots" / f"{r['safe']}.png").convert("RGB")
    im.thumbnail((TW, TH))
    x, y = (i % a.cols) * TW, (i // a.cols) * (TH + PAD)
    sheet.paste(im, (x, y))
    label = f"nc{r['n_colors']} se{r['styled_els']} {(r.get('title') or '?')[:26]}"
    d.text((x + 3, y + TH + 3), label, fill="#dddddd")
sheet.save(a.out)
print(f"{a.out}  {sheet.size}  ornek={len(pick)}  benzersiz baslik={len(uniq)}  "
      f"aday havuzu={len(vis)}/{len(rows)}")
