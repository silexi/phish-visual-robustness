#!/usr/bin/env python
"""Re-render corpus pages under one perturbation condition.

Conditions are injected as CSS/DOM at render time -- the honest attacker model,
since a phisher edits the page, not the screenshot.  Every condition reports
whether it actually took effect on the rendered pixels; a cell whose
`applied` is False is excluded from analysis rather than silently counted as
a successful evasion.
"""
import argparse, functools, http.server, json, socketserver, sys, threading
from pathlib import Path

import numpy as np
from PIL import Image
from playwright.sync_api import sync_playwright

VIEWPORT = {"width": 1366, "height": 768}

# --- conditions -------------------------------------------------------------
# Each returns (css, js, needs_logo).  `Z` is the overlay z-index.
Z = 2147483647

CONDITIONS = {
    # A0' — inert style injection: the null condition that gives every rung a
    # noise floor, so an attack's effect is read as excess over re-render noise.
    "A0n": lambda: ("#__p{}", None, False),

    # A1 — global high-frequency hairline grating, one CSS declaration.
    "A1": lambda a=0.04, p=3: (
        f"#__p{{position:fixed;inset:0;pointer-events:none;z-index:{Z};"
        f"background:repeating-linear-gradient(0deg,rgba(0,0,0,{a}) 0 1px,"
        f"transparent 1px {p}px)}}", None, False),

    # A3 — full inversion; the canonical dark-mode one-liner.
    "A3": lambda: ("html{filter:invert(1)}", None, False),

    # A4 — ROI decoy: dense texture in the top-left steals a contour-density argmax.
    "A4": lambda w=370, h=180: (
        f"#__p{{position:fixed;left:0;top:0;width:{w}px;height:{h}px;"
        f"pointer-events:none;z-index:{Z};"
        f"background:repeating-linear-gradient(45deg,#000 0 2px,#fff 2px 4px)}}",
        None, False),

    # S1 — logo removed: does the semantic rung survive without the wordmark?
    "S1": lambda: (
        "img,svg,[class*=logo],[id*=logo],[class*=brand],[id*=brand]"
        "{visibility:hidden!important}", None, True),

    # S3 — letter-spacing: the text-obfuscation family, applied as CSS.
    "S3": lambda: ("*{letter-spacing:.3em!important;word-spacing:.35em!important}",
                   None, False),

    # S5 — low contrast text.
    "S5": lambda: ("*{color:#9a9a9a!important}", None, False),

    # S7 — brand-contradicting footer: a white-box attack on the semantic rung.
    "S7": lambda: (
        f"#__p{{position:fixed;left:0;bottom:0;width:100%;z-index:{Z};"
        "background:#fff;color:#111;font:16px/1.4 Arial,sans-serif;padding:10px;"
        "text-align:center}}",
        "el.textContent='\\u00a9 2024 Aventro Corporation \\u2014 Aventro Secure Sign-In';",
        False),

    # B1 — phone viewport: benign environment variation.
    "B1": lambda: (None, None, False),
    # B8 — dark colour scheme: benign, prevalence-conditional.
    "B8": lambda: (None, None, False),
}

CTX_OVERRIDE = {
    "B1": dict(viewport={"width": 390, "height": 844}, device_scale_factor=2,
               is_mobile=True, has_touch=True),
    "B8": dict(color_scheme="dark"),
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


def diff_stats(a_png, b_png):
    """Pixel change between clean and perturbed, plus a perceptual proxy."""
    a = np.asarray(Image.open(a_png).convert("RGB"), dtype=np.int16)
    b = np.asarray(Image.open(b_png).convert("RGB"), dtype=np.int16)
    if a.shape != b.shape:
        return {"geometry_changed": True}
    d = np.abs(a - b)
    changed = float((d.max(axis=2) > 2).mean())
    return {"geometry_changed": False,
            "frac_changed": round(changed, 5),
            "mean_abs": round(float(d.mean()), 3),
            "p99_abs": round(float(np.percentile(d, 99)), 2)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True, help="dir with html/ and manifest*.jsonl")
    ap.add_argument("--cond", required=True, choices=sorted(CONDITIONS))
    ap.add_argument("--ids", required=True, help="newline-separated list of `safe` ids")
    ap.add_argument("--outdir", required=True)
    args = ap.parse_args()

    corpus = Path(args.corpus).resolve()
    out = Path(args.outdir).resolve()
    (out / args.cond).mkdir(parents=True, exist_ok=True)

    ids = [x.strip() for x in Path(args.ids).read_text().split() if x.strip()]
    css, js, needs_logo = CONDITIONS[args.cond]()
    rows = []

    with sync_playwright() as pw:
        browser = pw.chromium.launch(args=["--disable-lcd-text", "--hide-scrollbars"])
        ctx_kw = dict(viewport=VIEWPORT, device_scale_factor=1, color_scheme="light",
                      locale="en-US", timezone_id="UTC")
        ctx_kw.update(CTX_OVERRIDE.get(args.cond, {}))
        ctx = browser.new_context(**ctx_kw)
        ctx.route("**/*", lambda r: r.abort()
                  if r.request.url.startswith(("http://", "https://")) else r.continue_())
        page = ctx.new_page()
        page.set_default_navigation_timeout(15000)

        for i, sid in enumerate(ids):
            rec = {"safe": sid, "cond": args.cond}
            try:
                page.goto((corpus / "html" / f"{sid}.html").resolve().as_uri(),
                          wait_until="domcontentloaded")
                page.wait_for_timeout(1200)
                if needs_logo:
                    rec["logo_els"] = page.evaluate(
                        "()=>document.querySelectorAll('img,svg,[class*=logo],"
                        "[id*=logo],[class*=brand],[id*=brand]').length")
                if css:
                    page.add_style_tag(content=css)
                if "#__p{" in (css or ""):
                    page.evaluate("(j)=>{const el=document.createElement('div');"
                                  "el.id='__p';document.body.appendChild(el);"
                                  "if(j)eval(j);}", js)
                page.wait_for_timeout(250)
                dst = out / args.cond / f"{sid}.png"
                dst.write_bytes(page.screenshot(type="png"))

                clean = corpus / "shots" / f"{sid}.png"
                if clean.exists():
                    rec.update(diff_stats(clean, dst))
                    # efficacy: did the condition actually change the pixels?
                    rec["applied"] = bool(rec.get("geometry_changed")
                                          or rec.get("frac_changed", 0) > 0.001)
                else:
                    rec["applied"] = None
                rec["ok"] = True
            except Exception as e:
                rec.update(ok=False, applied=False, error=f"{type(e).__name__}: {str(e)[:140]}")
            rows.append(rec)
            if (i + 1) % 200 == 0:
                print(f"  {args.cond} {i+1}/{len(ids)}", file=sys.stderr, flush=True)

        browser.close()

    with (out / f"manifest_{args.cond}.jsonl").open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    ok = [r for r in rows if r.get("ok")]
    app = [r for r in ok if r.get("applied")]
    fc = sorted(r.get("frac_changed", 0) for r in app)
    print(f"{args.cond}: render {len(ok)}/{len(rows)}, etkili {len(app)}/{len(ok)}"
          + (f", degisen piksel medyan {fc[len(fc)//2]:.3f}" if fc else ""),
          file=sys.stderr)


if __name__ == "__main__":
    main()
