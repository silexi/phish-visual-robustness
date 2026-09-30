#!/usr/bin/env python
"""Render archived phishing HTML under a declared, fixed capture protocol.

The source infrastructure is dead, so every external subresource is blocked
rather than waited on.  Pages are served over a local HTTP origin, not file://,
because many credential kits gate their form on a cookie check that a file URL
cannot satisfy.  What each page lost is recorded per row: the amount of missing
styling is itself a measurement the study has to report.
"""
import argparse, functools, hashlib, http.server, json, socketserver, sys, threading
from pathlib import Path

import pandas as pd
from playwright.sync_api import sync_playwright

# --- capture protocol (CPD) -------------------------------------------------
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


def s_(v, n=200):
    """Parquet nulls arrive as float NaN, which is truthy -- `v or ""` keeps it."""
    if v is None or isinstance(v, float):
        return ""
    return str(v)[:n]


def n_(v):
    """Length of a list-valued column that may be null."""
    try:
        return int(len(v))
    except TypeError:
        return 0


def serve(directory):
    """Background HTTP server over `directory`; returns its port."""
    handler = functools.partial(http.server.SimpleHTTPRequestHandler,
                                directory=str(directory))
    handler.log_message = lambda *a, **k: None
    # HTTP/1.0 closes the connection after every response, which burns one
    # ephemeral port per subresource; at corpus scale that exhausts the
    # 49152-65535 range and every later connect fails with EADDRNOTAVAIL.
    handler.protocol_version = "HTTP/1.1"
    socketserver.TCPServer.allow_reuse_address = True
    httpd = socketserver.TCPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return httpd.server_address[1]


def visual_stats(png_bytes):
    """Blankness plus a styling proxy.

    An archived kit whose external stylesheet is dead renders as raw HTML:
    white ground, black text, one link blue.  Counting quantised colours that
    each cover at least 0.05% of the frame separates those from pages whose
    styling is self-contained, which blankness alone cannot do -- a genuinely
    minimalist login page is also ~99% one colour.
    """
    import io

    import numpy as np
    from PIL import Image

    im = Image.open(io.BytesIO(png_bytes)).convert("RGB")
    a = np.asarray(im).reshape(-1, 3)
    a = a[:: max(1, len(a) // 200_000)]
    _, counts = np.unique(a, axis=0, return_counts=True)
    blank = float(counts.max() / len(a))

    q = (a // 32).astype(np.uint16)
    q = q[:, 0] * 64 + q[:, 1] * 8 + q[:, 2]
    _, qc = np.unique(q, return_counts=True)
    n_colors = int((qc / len(a) >= 0.0005).sum())
    return blank, n_colors


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--limit", type=int, default=20)
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--shard-tag", default="")
    args = ap.parse_args()

    out = Path(args.outdir).resolve()
    (out / "shots").mkdir(parents=True, exist_ok=True)
    (out / "html").mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(args.parquet)
    sub = df[(df.has_password_field) & (df.http_status == 200) & (df.html_length > 3000)]
    sub = sub.iloc[args.offset: args.offset + args.limit]
    print(f"{len(sub)} sayfa render edilecek", file=sys.stderr)

    manifest_path = out / f"manifest{args.shard_tag}.jsonl"
    rows = []

    with sync_playwright() as pw:
        browser = pw.chromium.launch(args=["--disable-lcd-text", "--hide-scrollbars"])
        ctx = browser.new_context(
            viewport=PROTOCOL["viewport"],
            device_scale_factor=PROTOCOL["device_scale_factor"],
            color_scheme=PROTOCOL["color_scheme"],
            locale=PROTOCOL["locale"],
            timezone_id=PROTOCOL["timezone"],
        )

        blocked = {"n": 0}

        def route(r):
            # everything remote pointed at hosts that are long dead
            if r.request.url.startswith(("http://", "https://")):
                blocked["n"] += 1
                r.abort()
            else:
                r.continue_()

        if PROTOCOL["block_external"]:
            ctx.route("**/*", route)

        page = ctx.new_page()
        page.set_default_navigation_timeout(PROTOCOL["nav_timeout_ms"])

        for i, (_, r) in enumerate(sub.iterrows()):
            pid = s_(r["id"], 120) or f"row{i}"
            safe = hashlib.sha256(pid.encode()).hexdigest()[:16]
            (out / "html" / f"{safe}.html").write_text(
                r["html"], encoding="utf-8", errors="replace")

            blocked["n"] = 0
            rec = {
                "pid": pid, "safe": safe, "domain": r["domain"],
                "title": s_(r["title"]),
                "language": s_(r["language"], 16),
                "html_length": int(r["html_length"]),
                "ext_css": n_(r["external_css_urls"]), "ext_js": n_(r["external_script_urls"]),
                "inline_style": int(r["inline_style_count"]),
                "inline_js": int(r["inline_scripts_count"]),
                "input_count": int(r["input_count"]),
                "final_url": s_(r["final_url"], 300),
            }
            try:
                page.goto((out / "html" / f"{safe}.html").resolve().as_uri(),
                          wait_until=PROTOCOL["wait_until"])
                page.wait_for_timeout(PROTOCOL["settle_ms"])
                shot = page.screenshot(type="png")
                (out / "shots" / f"{safe}.png").write_bytes(shot)
                blank, n_colors = visual_stats(shot)
                pw_loc = page.locator('input[type="password"]')
                styled = page.evaluate(
                    "() => {const s=getComputedStyle(document.body);"
                    "return [...document.querySelectorAll('*')].filter(e=>{"
                    "const c=getComputedStyle(e);"
                    "return c.backgroundColor!=='rgba(0, 0, 0, 0)' "
                    "|| c.borderTopWidth!=='0px' || c.borderRadius!=='0px';}).length;}")
                rec.update(
                    ok=True, blocked=blocked["n"], bytes=len(shot),
                    blankness=round(blank, 4), n_colors=n_colors, styled_els=styled,
                    n_password=pw_loc.count(),
                    pw_visible=bool(pw_loc.count() and pw_loc.first.is_visible()),
                    n_input=page.locator("input").count(),
                    n_img=page.locator("img").count(),
                    text_len=len((page.inner_text("body") or "").strip()),
                )
            except Exception as e:
                rec.update(ok=False, error=f"{type(e).__name__}: {str(e)[:160]}")

            rows.append(rec)
            if (i + 1) % 250 == 0:
                print(f"  {i+1}/{len(sub)}", file=sys.stderr, flush=True)

        browser.close()

    with manifest_path.open("w", encoding="utf-8") as fh:
        for rec in rows:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

    ok = [r for r in rows if r.get("ok")]
    good = [r for r in ok if r["blankness"] < 0.90 and r["text_len"] > 80 and r["pw_visible"]]
    print(f"\nrender  {len(ok)}/{len(rows)}", file=sys.stderr)
    print(f"gorunur parola alani  {sum(1 for r in ok if r['pw_visible'])}/{len(ok)}", file=sys.stderr)
    print(f"KULLANILABILIR  {len(good)}/{len(rows)}", file=sys.stderr)


if __name__ == "__main__":
    main()
