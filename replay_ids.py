#!/usr/bin/env python
"""Re-render a named subset of archived HTML, in parallel, over file://.

Same protocol as every other render in the study; the only reason this exists
separately from render.py is that it takes an explicit id list rather than a
parquet, so the clean baseline can be regenerated without re-reading the corpus.
"""
import argparse, json, os, sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image
from playwright.sync_api import sync_playwright

PROTOCOL = {
    "viewport": {"width": 1366, "height": 768},
    "device_scale_factor": 1,
    "color_scheme": "light",
    "locale": "en-US",
    "timezone": "UTC",
    "wait_until": "domcontentloaded",
    "settle_ms": 1200,
    "block_external": True,
    "serve_over": "file",
    "nav_timeout_ms": 15000,
}


def visual_stats(png):
    im = Image.open(png).convert("RGB")
    a = np.asarray(im).reshape(-1, 3)
    a = a[:: max(1, len(a) // 200_000)]
    _, counts = np.unique(a, axis=0, return_counts=True)
    blank = float(counts.max() / len(a))
    q = (a // 32).astype(np.uint16)
    q = q[:, 0] * 64 + q[:, 1] * 8 + q[:, 2]
    _, qc = np.unique(q, return_counts=True)
    return blank, int((qc / len(a) >= 0.0005).sum())


def worker(job):
    htmldir, outdir, ids, tag = job
    htmldir, outdir = Path(htmldir), Path(outdir)
    rows = []
    with sync_playwright() as pw:
        browser = pw.chromium.launch(args=["--disable-lcd-text", "--hide-scrollbars"])
        ctx = browser.new_context(
            viewport=PROTOCOL["viewport"],
            device_scale_factor=PROTOCOL["device_scale_factor"],
            color_scheme=PROTOCOL["color_scheme"],
            locale=PROTOCOL["locale"], timezone_id=PROTOCOL["timezone"])
        blocked = {"n": 0}

        def route(r):
            if r.request.url.startswith(("http://", "https://")):
                blocked["n"] += 1
                r.abort()
            else:
                r.continue_()

        ctx.route("**/*", route)
        page = ctx.new_page()
        page.set_default_navigation_timeout(PROTOCOL["nav_timeout_ms"])

        for i, sid in enumerate(ids):
            src = htmldir / f"{sid}.html"
            rec = {"safe": sid}
            if not src.exists():
                rec.update(ok=False, error="html yok")
                rows.append(rec)
                continue
            blocked["n"] = 0
            try:
                page.goto(src.resolve().as_uri(), wait_until=PROTOCOL["wait_until"])
                page.wait_for_timeout(PROTOCOL["settle_ms"])
                dst = outdir / "shots" / f"{sid}.png"
                dst.write_bytes(page.screenshot(type="png"))
                blank, ncol = visual_stats(dst)
                pw_loc = page.locator('input[type="password"]')
                rec.update(ok=True, blocked=blocked["n"], blankness=round(blank, 4),
                           n_colors=ncol, n_password=pw_loc.count(),
                           pw_visible=bool(pw_loc.count() and pw_loc.first.is_visible()),
                           n_input=page.locator("input").count(),
                           text_len=len((page.inner_text("body") or "").strip()))
            except Exception as e:
                rec.update(ok=False, error=f"{type(e).__name__}: {str(e)[:120]}")
            rows.append(rec)
            if (i + 1) % 100 == 0:
                print(f"  [{tag}] {i+1}/{len(ids)}", file=sys.stderr, flush=True)
        browser.close()
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--htmldir", required=True)
    ap.add_argument("--ids", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()

    out = Path(a.outdir).resolve()
    (out / "shots").mkdir(parents=True, exist_ok=True)
    ids = [x.strip() for x in Path(a.ids).read_text().split() if x.strip()]
    print(f"{len(ids)} sayfa, {a.workers} isci", file=sys.stderr, flush=True)

    chunks = [ids[i::a.workers] for i in range(a.workers)]
    jobs = [(str(a.htmldir), str(out), c, i) for i, c in enumerate(chunks) if c]
    rows = []
    with ProcessPoolExecutor(max_workers=len(jobs)) as ex:
        for r in ex.map(worker, jobs):
            rows.extend(r)

    with (out / "manifest.jsonl").open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    ok = [r for r in rows if r.get("ok")]
    print(f"render {len(ok)}/{len(rows)}", file=sys.stderr)
    print(f"gorunur parola {sum(1 for r in ok if r['pw_visible'])}/{len(ok)}", file=sys.stderr)


if __name__ == "__main__":
    main()
