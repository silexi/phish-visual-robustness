#!/usr/bin/env python
"""Privacy scan of the released vision-language model outputs.

`vlm/desc_*.jsonl` holds what the model transcribed from the renders. On a
phishing page that can include an address the page itself had pre-filled, so in
the eight phishing files (`desc_A0`, `desc_A0n`, `desc_A1`, `desc_A3`,
`desc_A4`, `desc_S1`, `desc_S3`, `desc_S7`) every transcribed e-mail address and
URL is replaced by `[email redacted]` / `[url redacted]`. Nothing else is
altered. `desc_BENIGN.jsonl` is deliberately not redacted: what it transcribes
is the brands' own public content, it is the same file in both measurements, and
v1.0.0 released it unredacted for that reason.

This script does not redact. It lists every e-mail- and URL-shaped string in the
files it is given, with file, line, page id and JSON field, and classifies each
one. The only part of the classification that can be mechanical is the
placeholder test: a domain reserved by RFC 2606 or RFC 6761 (example.com /
.net / .org, *.example, *.invalid, *.test, *.localhost) cannot correspond to a
real mailbox or host. Those are marked `[yer tutucu]` (placeholder); everything
else is marked `[KARAR]` and needs a human ruling.

Usage:
    python vlm_gizlilik_tarama.py vlm/desc_A0.jsonl vlm/desc_A0n.jsonl \
        vlm/desc_A1.jsonl vlm/desc_A3.jsonl vlm/desc_A4.jsonl \
        vlm/desc_S1.jsonl vlm/desc_S3.jsonl vlm/desc_S7.jsonl
    python vlm_gizlilik_tarama.py --csv findings.csv vlm/desc_*.jsonl

Both patterns are matched case-insensitively. That is not cosmetic: the model
transcribes a page's own capitalisation, and one page of this sample displays
its webmail address in capitals, so a case-sensitive `https?://` leaves it in
the file while reporting that nothing awaits a ruling.

ACCEPTANCE CONDITION: no string in the eight phishing files awaits a ruling,
i.e. the line `karar bekleyen dizge` reads 0. The released files meet it: 58
redaction markers, 32 matches of 2 distinct strings, 0 awaiting a ruling, the
two survivors being `name@example.com` and `someone@example.com`. The condition
was not invented for this release; it was read off v1.0.0, whose eight published
files also meet it, leaving one placeholder (`someone@example.com`, 5 pages, in
all eight conditions) and 32 redaction markers of their own.

`desc_BENIGN.jsonl` is outside the condition. The scan finds 14 matches of 10
distinct strings there and flags 8 of them for a ruling: `info@flypgs.com`,
`name@company.com`, `name@email.com`, `Example@company.com`,
`name@work-email.com`, `http://www2.hm.com/en_us/login`,
`https://www.turkiye.gov.tr` and an Akamai error host. The remaining two
(`email@example.com`, `name@example.com`) are on a reserved domain. All 10 are
deliberately kept, in both measurements.

Bare brand hostnames carrying no scheme (`booking.com`, `chase.com`,
`www.terra.com.br`, `mail.163.com`) are left alone and are not what this scan
looks for. They are the brand and webmail-provider names the page displays --
the content the semantic rung measures -- not addresses that identify anyone,
and the v1.0.0 files keep the same class of string (`mail.ru`, `fedex.com`,
`53.com`).

The redaction cannot move a published number, and that was measured rather than
argued: replacing every e-mail- and URL-shaped string in all nine files (98
occurrences, the obvious placeholders included) and re-running
`vlm_retention.py`, `misattribution.py` and `f1_recheck.py` leaves all three
result files bit-identical. The scored fields are `brand`, `page_type` and
`asks_for_credentials`; the redaction touches `visible_text`, `input_fields` and
the `raw` text of parse failures.
"""
import argparse, csv, glob, json, re, sys
from collections import Counter, defaultdict

EMAIL = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
URL = re.compile(r"(?i)https?://[^\s\"'\\<>)]+")
REDACTED = ("[email redacted]", "[url redacted]")

# RFC 2606 / RFC 6761: alan adlari belgeleme ve sinama icin ayrilmistir, yani
# gercek bir mektup kutusuna ya da sunucuya karsilik gelemezler.  Yer tutucu
# siniflamasinin mekanik olarak denetlenebilen tek olcutu budur.
RESERVED_SUFFIXES = (".example.com", ".example.net", ".example.org",
                     ".example", ".invalid", ".test", ".localhost")
RESERVED_EXACT = ("example.com", "example.net", "example.org")


def host_of(kind, s):
    """Dizgenin alan adi kismi (e-posta icin @ sonrasi, URL icin konak)."""
    if kind == "email":
        return s.rsplit("@", 1)[-1].lower().rstrip(".")
    h = s.split("://", 1)[-1]
    for sep in ("/", "?", "#", ":"):
        h = h.split(sep, 1)[0]
    return h.lower().rstrip(".")


def is_placeholder(kind, s):
    """True ise dizge ayrilmis bir alan adindadir, yani gercek olamaz."""
    h = host_of(kind, s)
    return h in RESERVED_EXACT or h.endswith(RESERVED_SUFFIXES)


def walk(obj, path=""):
    """Her metin alanini (yol, metin) olarak uretir."""
    if isinstance(obj, str):
        yield path, obj
    elif isinstance(obj, dict):
        for k, v in obj.items():
            yield from walk(v, f"{path}.{k}" if path else k)
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            yield from walk(v, f"{path}[{i}]")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+", help="desc_*.jsonl dosyalari")
    ap.add_argument("--csv", default=None, help="bulgulari bu dosyaya da yaz")
    a = ap.parse_args()

    paths = []
    for pat in a.files:
        paths += sorted(glob.glob(pat)) or [pat]

    rows = []
    per_string = Counter()
    per_string_files = defaultdict(set)
    n_redacted = 0
    for p in paths:
        try:
            fh = open(p, encoding="utf-8")
        except OSError as e:
            print(f"{p}: {e}", file=sys.stderr)
            continue
        with fh:
            for ln, line in enumerate(fh, 1):
                line = line.strip()
                if not line:
                    continue
                n_redacted += sum(line.count(m) for m in REDACTED)
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    rec = {"__unparsable_line__": line}
                image = rec.get("image", "?") if isinstance(rec, dict) else "?"
                for field, text in walk(rec):
                    for kind, rx in (("email", EMAIL), ("url", URL)):
                        for m in rx.findall(text):
                            rows.append((p, ln, image, field, kind, m))
                            per_string[(kind, m)] += 1
                            per_string_files[(kind, m)].add(p)

    n_decide = sum(1 for (kind, s) in per_string if not is_placeholder(kind, s))
    print(f"taranan dosya          {len(paths)}")
    print(f"mevcut karartma izi    {n_redacted} ({' / '.join(REDACTED)})")
    print(f"bulgu                  {len(rows)} esleme, {len(per_string)} farkli dizgi")
    print(f"karar bekleyen dizge   {n_decide} (kalani ayrilmis alan adinda)")
    print()
    print("farkli dizgiler (cokluk, kac dosyada, siniflama):")
    for (kind, s), n in per_string.most_common():
        tag = "[yer tutucu]" if is_placeholder(kind, s) else "[KARAR]     "
        print(f"  {kind:5s} {n:4d}  {len(per_string_files[(kind, s)]):2d} dosya  {tag}  {s}")
    print()
    print("yer bazinda (dosya, satir, alan):")
    for p, ln, image, field, kind, m in rows:
        tag = "yer tutucu" if is_placeholder(kind, m) else "KARAR"
        print(f"  {p}:{ln}  {image}  {field}  [{kind}/{tag}]  {m}")

    if a.csv:
        with open(a.csv, "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["file", "line", "image", "field", "kind", "match", "class"])
            for p, ln, image, field, kind, m in rows:
                w.writerow([p, ln, image, field, kind, m,
                            "placeholder" if is_placeholder(kind, m) else "decide"])
        print(f"\nyazildi: {a.csv}")


if __name__ == "__main__":
    main()
