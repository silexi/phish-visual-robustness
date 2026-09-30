#!/usr/bin/env python
"""Semantic-rung retention: does the description survive the perturbation?

The pixel and embedding rungs are scored by whether the perturbed render still
retrieves its own kit cluster.  The semantic rung has no similarity space, so
retention is defined on the description itself: for each page, does the
perturbed render still yield the same brand, the same page type, and the same
credential-seeking verdict as its own clean render?

Three separate numbers, never one, because they fail differently.  Brand is the
fragile one (it depends on a logo or wordmark surviving); asks_for_credentials
is the robust one (it depends only on a form being visible).  Collapsing them
into a single "accuracy" would hide exactly the effect the study is about.

Abstention (`brand: null`) is reported as its own outcome rather than counted as
a miss -- a detector that says "I cannot identify this brand" is behaving
differently from one that names the wrong brand, and the distinction matters
operationally.
"""
import argparse, json, re, sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from score import cluster_bootstrap, wilson  # noqa: E402


def norm_brand(b):
    if b is None:
        return None
    b = re.sub(r"[^a-z0-9]+", "", str(b).lower())
    return b or None


def load(path):
    out = {}
    for line in Path(path).open(encoding="utf-8"):
        r = json.loads(line)
        p = r.get("parsed")
        if not p:
            out[Path(r["image"]).stem] = {"parse_error": True}
            continue
        out[Path(r["image"]).stem] = {
            "brand": norm_brand(p.get("brand")),
            "page_type": (p.get("page_type") or "").strip().lower() or None,
            "cred": p.get("asks_for_credentials"),
            "logo": p.get("logo_present"),
            "parse_error": False,
            "seconds": r.get("seconds"),
        }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clean", required=True)
    ap.add_argument("--conds", nargs="+", required=True, help="name=desc.jsonl")
    ap.add_argument("--clusters", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    clean = load(a.clean)
    clusters_map = json.loads(Path(a.clusters).read_text())
    print(f"temiz tarif: {len(clean)}", file=sys.stderr)

    named = sum(1 for v in clean.values() if v.get("brand"))
    print(f"  marka adlandirilan: {named}/{len(clean)} "
          f"({100*named/max(1,len(clean)):.0f}%)", file=sys.stderr)

    results = {"n_clean_desc": len(clean),
               "clean_brand_coverage": round(named / max(1, len(clean)), 4),
               "conditions": {}}

    for spec in a.conds:
        name, path = spec.split("=", 1)
        if not Path(path).exists():
            continue
        att = load(path)
        ids = [i for i in att if i in clean and not clean[i]["parse_error"]]
        if not ids:
            continue
        cl = np.array([clusters_map.get(i, i) for i in ids])

        brand_same, type_same, cred_same, abst, parse_fail = [], [], [], [], []
        for i in ids:
            c, x = clean[i], att[i]
            if x["parse_error"]:
                parse_fail.append(1); brand_same.append(0)
                type_same.append(0); cred_same.append(0); abst.append(0)
                continue
            parse_fail.append(0)
            abst.append(1 if (c["brand"] is not None and x["brand"] is None) else 0)
            brand_same.append(1 if (c["brand"] is not None
                                    and x["brand"] == c["brand"]) else 0)
            type_same.append(1 if x["page_type"] == c["page_type"] else 0)
            cred_same.append(1 if x["cred"] == c["cred"] else 0)

        # brand retention is only defined where the clean render named a brand
        eligible = [k for k, i in enumerate(ids) if clean[i]["brand"] is not None]
        entry = {"n": len(ids), "n_clusters": int(len(set(cl))),
                 "n_brand_eligible": len(eligible),
                 "parse_fail_rate": round(float(np.mean(parse_fail)), 4)}

        for key, vec, sub in (("brand", brand_same, eligible),
                              ("abstain", abst, eligible),
                              ("page_type", type_same, None),
                              ("asks_for_credentials", cred_same, None)):
            v = np.array(vec)[sub] if sub is not None else np.array(vec)
            c2 = cl[sub] if sub is not None else cl
            if len(v) == 0:
                continue
            lo, hi = wilson(int(v.sum()), len(v))
            _, blo, bhi = cluster_bootstrap(lambda t: v[t].mean(), c2, B=2000)
            entry[key] = {"rate": round(float(v.mean()), 4),
                          "n": int(len(v)),
                          "wilson95": [round(lo, 4), round(hi, 4)],
                          "boot95": [round(blo, 4), round(bhi, 4)]}
        results["conditions"][name] = entry
        print(f"{name:<5} n={entry['n']:<4} marka={entry.get('brand',{}).get('rate',0):.3f} "
              f"cekimser={entry.get('abstain',{}).get('rate',0):.3f} "
              f"tip={entry['page_type']['rate']:.3f} "
              f"kimlik={entry['asks_for_credentials']['rate']:.3f}", file=sys.stderr)

    Path(a.out).write_text(json.dumps(results, indent=1), encoding="utf-8")
    print(f"-> {a.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
