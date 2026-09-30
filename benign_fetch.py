#!/usr/bin/env python
"""Capture legitimate sign-in pages, then archive them so they can be replayed
through the *same* offline renderer as the phishing corpus.

The phishing pages are archived HTML re-rendered with every subresource blocked.
If the benign side were captured live with styling intact, the two classes would
differ by provenance as well as by label, and provenance alone could produce the
whole result.  So the benign side is frozen to HTML here and rendered by the
identical renderer under the identical protocol.

Politeness: one navigation per site, a declared UA with a contact address,
robots.txt honoured, no form ever submitted, no credential ever entered.
"""
import argparse, hashlib, json, socket, sys, time, urllib.parse, urllib.robotparser
from pathlib import Path

from playwright.sync_api import sync_playwright

# RobotFileParser.read() accepts no timeout and will otherwise hang forever on a
# host that completes the TCP handshake and then never answers.
socket.setdefaulttimeout(8)

UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/153.0.0.0 Safari/537.36 "
      "AcademicPhishingRobustnessStudy/1.0 (+research contact in artifact README)")


def robots_ok(url, cache):
    p = urllib.parse.urlparse(url)
    root = f"{p.scheme}://{p.netloc}"
    if root not in cache:
        rp = urllib.robotparser.RobotFileParser()
        rp.set_url(root + "/robots.txt")
        try:
            rp.read()
        except Exception:
            rp = None
        cache[root] = rp
    rp = cache[root]
    if rp is None:
        return True          # an unreachable robots.txt is not a prohibition
    try:
        return rp.can_fetch(UA, url)
    except Exception:
        return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--urls", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--delay", type=float, default=1.5)
    ap.add_argument("--timeout", type=int, default=20000)
    args = ap.parse_args()

    out = Path(args.outdir).resolve()
    (out / "html").mkdir(parents=True, exist_ok=True)
    manifest = out / "fetch_manifest.jsonl"

    entries = []
    for line in Path(args.urls).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        brand, url = line.split(None, 1)
        entries.append((brand, url.strip()))
    print(f"{len(entries)} mesru sayfa", file=sys.stderr, flush=True)

    rows, rcache = [], {}
    with sync_playwright() as pw:
        browser = pw.chromium.launch()
        ctx = browser.new_context(
            viewport={"width": 1366, "height": 768}, device_scale_factor=1,
            color_scheme="light", locale="en-US", timezone_id="UTC", user_agent=UA)
        page = ctx.new_page()
        page.set_default_navigation_timeout(args.timeout)

        for i, (brand, url) in enumerate(entries):
            safe = "b_" + hashlib.sha256(url.encode()).hexdigest()[:14]
            rec = {"safe": safe, "brand": brand, "url": url}
            if not robots_ok(url, rcache):
                rec.update(ok=False, error="robots.txt disallow")
            else:
                try:
                    resp = page.goto(url, wait_until="domcontentloaded")
                    page.wait_for_timeout(2500)
                    html = page.content()
                    (out / "html" / f"{safe}.html").write_text(
                        html, encoding="utf-8", errors="replace")
                    pw_loc = page.locator('input[type="password"]')
                    rec.update(
                        ok=True,
                        status=resp.status if resp else None,
                        final_url=page.url[:300],
                        title=(page.title() or "")[:200],
                        html_length=len(html),
                        n_password=pw_loc.count(),
                        pw_visible=bool(pw_loc.count() and pw_loc.first.is_visible()),
                        n_input=page.locator("input").count(),
                    )
                except Exception as e:
                    rec.update(ok=False, error=f"{type(e).__name__}: {str(e)[:140]}")

            rows.append(rec)
            with manifest.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            if (i + 1) % 10 == 0:
                print(f"  {i+1}/{len(entries)}", file=sys.stderr, flush=True)
            time.sleep(args.delay)

        browser.close()

    ok = [r for r in rows if r.get("ok")]
    print(f"\ncekilen {len(ok)}/{len(rows)}", file=sys.stderr)
    print(f"parola alani olan {sum(1 for r in ok if r.get('n_password'))}/{len(ok)}",
          file=sys.stderr)
    print(f"gorunur parola alani {sum(1 for r in ok if r.get('pw_visible'))}/{len(ok)}",
          file=sys.stderr)


if __name__ == "__main__":
    main()
