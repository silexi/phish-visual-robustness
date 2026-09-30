#!/usr/bin/env python
"""Render a directory of archived HTML under the same protocol as the phishing corpus.

This is the matched-provenance arm.  The benign pages were captured live, frozen
to HTML, and are replayed here with every external subresource blocked -- exactly
what happens to the phishing pages, whose hosts are dead.  Without this arm the
two classes would differ by provenance as well as by label, and provenance alone
could produce the entire result.
"""
import argparse, functools, http.server, json, socketserver, sys, threading
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


def serve(directory):
    h = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(directory))
    h.log_message = lambda *a, **k: None
    # HTTP/1.0 closes the connection after every response, which burns one
    # ephemeral port per subresource; at corpus scale that exhausts the
    # 49152-65535 range and every later connect fails with EADDRNOTAVAIL.
    h.protocol_version = "HTTP/1.1"
    socketserver.TCPServer.allow_reuse_address = True
    httpd = socketserver.TCPServer(("127.0.0.1", 0), h)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return httpd.server_address[1]


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--htmldir", required=True)
    ap.add_argument("--outdir", required=True)
    args = ap.parse_args()

    htmldir = Path(args.htmldir).resolve()
    out = Path(args.outdir).resolve()
    (out / "shots").mkdir(parents=True, exist_ok=True)

    files = sorted(htmldir.glob("*.html"))
    print(f"{len(files)} sayfa replay ediliyor", file=sys.stderr, flush=True)
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
            # everything remote is dead anyway; only the local document loads
            if r.request.url.startswith(("http://", "https://")):
                blocked["n"] += 1
                r.abort()
            else:
                r.continue_()

        ctx.route("**/*", route)
        page = ctx.new_page()
        page.set_default_navigation_timeout(PROTOCOL["nav_timeout_ms"])

        for i, f in enumerate(files):
            sid = f.stem
            blocked["n"] = 0
            rec = {"safe": sid}
            try:
                page.goto(f.resolve().as_uri(), wait_until=PROTOCOL["wait_until"])
                page.wait_for_timeout(PROTOCOL["settle_ms"])
                dst = out / "shots" / f"{sid}.png"
                dst.write_bytes(page.screenshot(type="png"))
                blank, ncol = visual_stats(dst)
                pw_loc = page.locator('input[type="password"]')
                rec.update(ok=True, blocked=blocked["n"], blankness=round(blank, 4),
                           n_colors=ncol,
                           n_password=pw_loc.count(),
                           pw_visible=bool(pw_loc.count() and pw_loc.first.is_visible()),
                           n_input=page.locator("input").count(),
                           text_len=len((page.inner_text("body") or "").strip()))
            except Exception as e:
                rec.update(ok=False, error=f"{type(e).__name__}: {str(e)[:140]}")
            rows.append(rec)
            if (i + 1) % 25 == 0:
                print(f"  {i+1}/{len(files)}", file=sys.stderr, flush=True)
        browser.close()

    with (out / "manifest_replay.jsonl").open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    ok = [r for r in rows if r.get("ok")]
    print(f"replay {len(ok)}/{len(rows)}; gorunur parola "
          f"{sum(1 for r in ok if r['pw_visible'])}", file=sys.stderr)


if __name__ == "__main__":
    main()
