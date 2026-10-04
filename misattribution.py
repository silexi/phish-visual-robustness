#!/usr/bin/env python
"""Misattribution column of the semantic-retention table (paper Table 6).

Reads only the released VLM outputs (vlm/desc_A0.jsonl = clean run,
vlm/desc_<cond>.jsonl = perturbed runs) and reports, per condition,

  * unconditional misattribution (the table column): pages whose perturbed
    answer parses and names a brand different from the clean brand, or names
    any brand where the clean answer named none; denominator = pages with a
    parsable clean answer that are present in the condition file (195 of the
    200-page subset; all eight condition files hold 200 rows, so the
    denominator is the same in every condition, unlike the first run, where
    A1 and S3 held 199 rows and the denominator fell to 195 from 196);
  * the split of that count into named-brand pages (a different brand) and
    unnamed pages (an invented brand), as quoted in Section 6.3;
  * the conditional count on the brand-eligible pages only (158), i.e. the
    "wrong brand" figure of the earlier audit.

Brand normalisation is the one used everywhere else in the package:
lower-case, non-alphanumerics removed (vlm_retention.py; run1_833/audit12.py).

Usage (from the package root):
    python misattribution.py [--out results_misattribution.json]
"""
import argparse, json, re
from pathlib import Path

R = Path(__file__).resolve().parent
CONDS = ["A0n", "A1", "A3", "A4", "S1", "S3", "S7"]


def norm(b):
    if b is None:
        return None
    b = re.sub(r"[^a-z0-9]+", "", str(b).lower())
    return b or None


def load(p):
    out = {}
    for line in open(p, encoding="utf-8"):
        r = json.loads(line)
        pa = r.get("parsed")
        sid = Path(r["image"]).stem
        out[sid] = {"pe": True} if not pa else {"pe": False, "brand": norm(pa.get("brand"))}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results_misattribution.json")
    a = ap.parse_args()
    clean = load(R / "vlm/desc_A0.jsonl")
    parsed = [i for i in clean if not clean[i]["pe"]]
    res = {"definition": "perturbed answer parses and names a brand != clean brand "
                         "(any brand counts on clean-unnamed pages); denominator = "
                         "clean-parsed pages present in the condition file",
           "n_clean_rows": len(clean), "n_clean_parsed": len(parsed), "conditions": {}}
    for c in CONDS:
        att = load(R / f"vlm/desc_{c}.jsonl")
        ids = [i for i in parsed if i in att]
        named = [i for i in ids if clean[i]["brand"]]
        unnamed = [i for i in ids if not clean[i]["brand"]]

        def mis(i):
            return (not att[i]["pe"]) and att[i]["brand"] is not None and att[i]["brand"] != clean[i]["brand"]

        m_named = sum(mis(i) for i in named)
        m_unnamed = sum(mis(i) for i in unnamed)
        entry = {"n_rows": len(att), "denominator": len(ids), "n_named": len(named),
                 "n_unnamed": len(unnamed), "misattributed": m_named + m_unnamed,
                 "misattributed_on_named": m_named, "invented_on_unnamed": m_unnamed,
                 "rate": round((m_named + m_unnamed) / len(ids), 6),
                 "conditional_rate_on_named": round(m_named / len(named), 6)}
        res["conditions"][c] = entry
        print(f"{c:<4} {entry['misattributed']:>3}/{entry['denominator']} = {entry['rate']:.3f}  "
              f"(named {m_named}/{len(named)}, unnamed {m_unnamed}/{len(unnamed)})")
    Path(a.out).write_text(json.dumps(res, indent=1))
    print("->", a.out)


if __name__ == "__main__":
    main()
