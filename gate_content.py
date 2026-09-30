#!/usr/bin/env python
"""Content gate: from the per-shard render manifests to `ids_gated_all.txt`.

A render is kept when it succeeded, shows a *visible* password field, has more
than 60 characters of body text and at least 10 quantised colours.  The
password-field and text conditions remove kits whose form never rendered; the
colour condition removes the near-blank canvases of kits whose stylesheet is
dead.

Timing disclosure.  The gate was applied once, by hand, on 2026-09-22 at
03:25 local time, when the render workers of 13 of the 20 shards had
finished (shards 0-7 and 9-13).  Shards 8 and 14-19 completed a few minutes
later and were never gated, so the 1,702 ids in `ids_gated_all.txt` come
from those 13 shards (7,778 manifest rows, 6,966 successful renders).  The
same predicate over all 20 shards would have admitted 2,359 pages; the 657
pages that were left out were left out by timing, not by any property of the
pages.  Every later stage (fidelity gate, clustering, perturbation, VLM) ran
on the 13-shard set.  This script therefore defaults to the 13 shards so
that it reproduces `ids_gated_all.txt` exactly; pass `--shards all` to see
the 20-shard counts.  The per-shard manifests are not part of the public
release (they contain the phishing source domains), so the script is kept
for the record and for anyone who re-renders the corpus.
"""
import argparse, collections, json, re
from pathlib import Path

USED_SHARDS = [0, 1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 13]
PREDICATE = "ok and pw_visible and text_len > 60 and n_colors >= 10"


def keep(r):
    return bool(r.get("ok")) and bool(r.get("pw_visible")) and r.get("text_len", 0) > 60 \
        and r.get("n_colors", 0) >= 10


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="corpus", help="dir with manifest_s*.jsonl")
    ap.add_argument("--shards", nargs="+", default=[str(s) for s in USED_SHARDS],
                    help="shard numbers to gate (default: the 13 shards that were gated "
                         "in the reported run); 'all' for every manifest present")
    ap.add_argument("--out", default="ids_gated_all.txt")
    ap.add_argument("--check", default=None,
                    help="an existing id list to compare with (e.g. the released ids_gated_all.txt)")
    a = ap.parse_args()

    paths = sorted(Path(a.corpus).glob("manifest_s*.jsonl"),
                   key=lambda p: int(p.stem.split("_s")[-1]))
    if a.shards != ["all"]:
        want = {int(s) for s in a.shards}
        paths = [p for p in paths if int(p.stem.split("_s")[-1]) in want]

    rows, per_shard = [], []
    for mf in paths:
        sh = [json.loads(l) for l in mf.read_text(encoding="utf-8").splitlines() if l.strip()]
        rows += sh
        per_shard.append((mf.stem, len(sh), sum(1 for r in sh if r.get("ok")),
                          sum(1 for r in sh if keep(r))))
    ok = [r for r in rows if r.get("ok")]
    err = collections.Counter()
    for r in rows:
        if not r.get("ok"):
            e = str(r.get("error", "")).split("\n")[0]
            e = re.sub(r"127\.0\.0\.1:\d+\S*", "127.0.0.1:<port>", e)   # the local file server
            err[e[:70]] += 1
    kept = [r for r in ok if keep(r)]
    ids = sorted({r["safe"] for r in kept})

    print(f"shards                 {len(paths)}: " + " ".join(p.stem.split('_')[-1] for p in paths))
    for name, n, n_ok, n_keep in per_shard:
        print(f"    {name:<14} rows {n:5d}  ok {n_ok:5d}  gate {n_keep:5d}")
    print(f"manifest rows          {len(rows)}")
    print(f"rendered ok            {len(ok)}")
    print(f"render failures        {len(rows) - len(ok)}")
    for e, n in err.most_common(8):
        print(f"    {n:5d}  {e}")
    print(f"predicate              {PREDICATE}")
    print(f"content gate (rows)    {len(kept)}")
    print(f"content gate (ids)     {len(ids)}")
    if a.check:
        ref = set(Path(a.check).read_text().split())
        print(f"vs {a.check}: {len(ref)} ids, in both {len(ref & set(ids))}, "
              f"only here {len(set(ids) - ref)}, only there {len(ref - set(ids))}")
    Path(a.out).write_text("\n".join(ids) + "\n", encoding="utf-8")
    print("wrote", a.out)


if __name__ == "__main__":
    main()
