#!/usr/bin/env python
"""Re-derive the credential-intent confusion matrix from the raw VLM outputs.

Everything the manuscript states about `asks_for_credentials` as a stand-alone
detector is recomputed here from `vlm/desc_A0.jsonl` (clean phishing renders)
and `vlm/desc_BENIGN.jsonl` (replayed legitimate captures), with every
definitional choice made explicit:

  positive class      phishing
  predicted positive  parsed.asks_for_credentials is True
  parse failure       excluded from the denominator (primary); two sensitivity
                      checks count it as a negative prediction, or salvage the
                      field by regular expression from the truncated raw text
  negative set        (a) the legitimate pages the model itself labels
                          login / password_entry / mfa_challenge  (circular:
                          the negative set is defined by the model under test)
                      (b) all parsed legitimate pages
                      (c) legitimate pages whose replayed render shows a
                          visible password field (model-independent; the
                          `pw_visible` flag of benign_replay/manifest_replay.jsonl,
                          i.e. the same render the model described)
                      (c') as (c) but with the flag from the live capture
                          (benign/fetch_manifest.jsonl); the two disagree on
                          three pages whose password field is revealed by a
                          script that does not run in the frozen replay

Besides precision / recall / FPR / F1 the script reports, for every cell, the
F1 of the always-positive baseline (predict phishing for every page), the
Youden difference TPR - FPR, and Wilson 95 % intervals.  Recall is also given
cluster-weighted (mean over kit clusters of the per-cluster recall, cluster
bootstrap B=4000, seed 20260922), which is the estimand used everywhere else
in the paper.
"""
import json, math, re
from collections import Counter
from pathlib import Path
from urllib.parse import urlparse

import numpy as np

LOGIN = {"login", "password_entry", "mfa_challenge"}
AFC_RE = re.compile(r'"asks_for_credentials"\s*:\s*(true|false)')


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (round(c - h, 4), round(c + h, 4))


def load(p):
    return [json.loads(l) for l in open(p, encoding="utf-8")]


def stem(r):
    return Path(r["image"]).stem


def pos(r):
    return bool(r.get("parsed")) and r["parsed"].get("asks_for_credentials") is True


def pos_salvaged(r):
    """Strict parse first; on failure, the field as it appears in the raw text."""
    if r.get("parsed"):
        return r["parsed"].get("asks_for_credentials") is True
    m = AFC_RE.search(r.get("raw") or "")
    return bool(m) and m.group(1) == "true"


def reg_dom(u):
    """Registrable domain, with the common two-level public suffixes handled."""
    h = urlparse(u or "").netloc.lower().split(":")[0]
    parts = h.split(".")
    if len(parts) >= 3 and parts[-2] in ("co", "com", "gov", "edu", "org", "net", "ac") \
            and len(parts[-1]) == 2:
        return ".".join(parts[-3:])
    return ".".join(parts[-2:]) if len(parts) >= 2 else h


def cm(ph, neg, name, parse_fail_as_negative=False, salvage=False):
    pred = pos_salvaged if salvage else pos
    if parse_fail_as_negative:          # every row stays, an unparsed one predicts negative
        keep = lambda r: True
    elif salvage:                       # unparsed rows stay if the field is legible in the raw text
        keep = lambda r: bool(r.get("parsed")) or bool(AFC_RE.search(r.get("raw") or ""))
    else:                               # primary: unparsed rows leave the denominator
        keep = lambda r: bool(r.get("parsed"))
    ph_use = [r for r in ph if keep(r)]
    neg_use = [r for r in neg if keep(r)]
    tp = sum(pred(r) for r in ph_use)
    fn = len(ph_use) - tp
    fp = sum(pred(r) for r in neg_use)
    tn = len(neg_use) - fp
    prec = tp / (tp + fp) if tp + fp else float("nan")
    rec = tp / (tp + fn) if tp + fn else float("nan")
    fpr = fp / (fp + tn) if fp + tn else float("nan")
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else float("nan")
    p_all = len(ph_use) / (len(ph_use) + len(neg_use))
    f1_all = 2 * p_all / (1 + p_all)
    out = {"negative_set": name, "parse_fail_as_negative": parse_fail_as_negative,
           "salvaged_parse": salvage,
           "n_phish": len(ph_use), "n_legit": len(neg_use),
           "TP": tp, "FN": fn, "FP": fp, "TN": tn,
           "precision": round(prec, 4), "recall_TPR": round(rec, 4),
           "FPR": round(fpr, 4), "F1": round(f1, 4),
           "TPR_minus_FPR": round(rec - fpr, 4),
           "F1_always_positive": round(f1_all, 4),
           "TPR_wilson95": wilson(tp, tp + fn), "FPR_wilson95": wilson(fp, fp + tn)}
    print(f"{name:<40} pf_neg={int(parse_fail_as_negative)} salv={int(salvage)}  "
          f"TP={tp:3d} FN={fn:3d} FP={fp:3d} TN={tn:3d}  "
          f"P={prec:.3f} R={rec:.3f} FPR={fpr:.3f} F1={f1:.3f}  "
          f"TPR-FPR={rec - fpr:+.3f}  F1(hep+)={f1_all:.3f}")
    return out


def cluster_recall(ph, cmap, B=4000, seed=20260922):
    rows = [r for r in ph if r.get("parsed")]
    cl = np.array([cmap.get(stem(r), stem(r)) for r in rows])
    hit = np.array([pos(r) for r in rows], dtype=float)
    keys = sorted(set(cl.tolist()))
    per = np.array([hit[cl == k].mean() for k in keys])
    rng = np.random.default_rng(seed)
    draws = np.array([per[rng.integers(0, len(per), len(per))].mean() for _ in range(B)])
    maj = int((per > 0.5).sum())
    return {"n_clusters": len(keys), "cluster_mean_recall": round(float(per.mean()), 4),
            "boot95": [round(float(np.percentile(draws, 2.5)), 4),
                       round(float(np.percentile(draws, 97.5)), 4)],
            "clusters_majority_detected": maj,
            "clusters_majority_rate": round(maj / len(keys), 4),
            "clusters_majority_wilson95": wilson(maj, len(keys)),
            "clusters_none_detected": int((per == 0).sum()),
            "clusters_all_detected": int((per == 1).sum()),
            "misses_by_cluster_top": sorted(
                ((k, int(((cl == k) & (hit == 0)).sum())) for k in keys),
                key=lambda x: -x[1])[:3]}


ph = load("vlm/desc_A0.jsonl")
ben = load("vlm/desc_BENIGN.jsonl")
rep = {json.loads(l)["safe"]: json.loads(l)
       for l in open("benign_replay/manifest_replay.jsonl", encoding="utf-8")}
fetch = {}
for l in open("benign/fetch_manifest.jsonl", encoding="utf-8"):
    r = json.loads(l)
    fetch.setdefault(r["safe"], r)          # first capture row per page
cmap = json.loads(Path("clusters_flat.json").read_text())

print(f"phishing clean descs: {len(ph)}  parse failures: {sum(1 for r in ph if not r.get('parsed'))}")
print(f"legit descs: {len(ben)}  parse failures: {sum(1 for r in ben if not r.get('parsed'))}")
brands = {fetch[stem(r)]["brand"] for r in ben}
dom_url = {reg_dom(fetch[stem(r)].get("url")) for r in ben}
dom_final = {reg_dom(fetch[stem(r)].get("final_url")) for r in ben}
print(f"legit brand labels described: {len(brands)}  registrable domains: "
      f"{len(dom_url)} (capture URL) / {len(dom_final)} (final URL)")
ids_sorted = sorted(rep)
desc_ids = [stem(r) for r in ben]
print("described legit == first 120 of 156 replay ids in filename order:",
      desc_ids == ids_sorted[:120], "| sorted-set equal:", set(desc_ids) == set(ids_sorted[:120]))

pt_ben = Counter((r["parsed"].get("page_type") or "") for r in ben if r.get("parsed"))
pt_ph = Counter((r["parsed"].get("page_type") or "") for r in ph if r.get("parsed"))
print("legit page_type:", dict(pt_ben))
print("phish page_type:", dict(pt_ph))
miss_pt = Counter((r["parsed"].get("page_type") or "") for r in ph if r.get("parsed") and not pos(r))
print("page_type of phishing misses:", dict(miss_pt))

neg_a = [r for r in ben if r.get("parsed") and (r["parsed"].get("page_type") or "") in LOGIN]
neg_b = ben
neg_c = [r for r in ben if rep[stem(r)].get("pw_visible")]
neg_c2 = [r for r in ben if fetch[stem(r)].get("pw_visible")]
npar = lambda rows: sum(1 for r in rows if r.get("parsed"))
print(f"negative sets (rows / parsed): (a) model-labelled login {len(neg_a)}/{npar(neg_a)}  "
      f"(b) all {len(neg_b)}/{npar(neg_b)}  (c) visible password field, replay {len(neg_c)}/{npar(neg_c)}  "
      f"(c') visible password field, capture {len(neg_c2)}/{npar(neg_c2)}")
disagree = sorted(stem(r) for r in ben
                  if bool(rep[stem(r)].get("pw_visible")) != bool(fetch[stem(r)].get("pw_visible")))
print(f"pw_visible replay vs capture disagree on {len(disagree)} pages")
print()
res = {"positive_class": "phishing",
       "predicted_positive": "parsed.asks_for_credentials is True",
       "phish_source": "vlm/desc_A0.jsonl (clean renders of the 200-page VLM subset)",
       "legit_source": "vlm/desc_BENIGN.jsonl (first 120 of 156 replayed legitimate captures, filename order)",
       "pw_visible_source_c": "benign_replay/manifest_replay.jsonl (the replayed render the model saw)",
       "pw_visible_source_c_prime": "benign/fetch_manifest.jsonl (live capture)",
       "pw_visible_disagreement_pages": disagree,
       "n_phish_desc": len(ph), "n_phish_parse_fail": sum(1 for r in ph if not r.get("parsed")),
       "n_legit_desc": len(ben), "n_legit_parse_fail": sum(1 for r in ben if not r.get("parsed")),
       "n_legit_brand_labels": len(brands),
       "n_legit_registrable_domains_url": len(dom_url),
       "n_legit_registrable_domains_final_url": len(dom_final),
       "legit_page_type": dict(pt_ben), "phish_page_type": dict(pt_ph),
       "phish_miss_page_type": dict(miss_pt),
       "variants": []}
res["variants"].append(cm(ph, neg_a, "(a) legit labelled login/pw/mfa by model"))
res["variants"].append(cm(ph, neg_b, "(b) all parsed legit pages"))
res["variants"].append(cm(ph, neg_c, "(c) legit with visible password field"))
res["variants"].append(cm(ph, neg_c2, "(c') visible password field, capture flag"))
res["variants"].append(cm(ph, neg_a, "(a) parse-fail counted negative", True))
res["variants"].append(cm(ph, neg_b, "(b) parse-fail counted negative", True))
res["variants"].append(cm(ph, neg_c, "(c) parse-fail counted negative", True))
res["variants"].append(cm(ph, neg_a, "(a) parse salvaged from raw text", salvage=True))
res["variants"].append(cm(ph, neg_b, "(b) parse salvaged from raw text", salvage=True))
res["variants"].append(cm(ph, neg_c, "(c) parse salvaged from raw text", salvage=True))

res["cluster_weighted_recall"] = cluster_recall(ph, cmap)
cr = res["cluster_weighted_recall"]
print()
print(f"cluster-weighted recall: {cr['cluster_mean_recall']} {cr['boot95']} over {cr['n_clusters']} clusters; "
      f"majority-detected clusters {cr['clusters_majority_detected']}/{cr['n_clusters']} "
      f"= {cr['clusters_majority_rate']} {cr['clusters_majority_wilson95']}; "
      f"none detected {cr['clusters_none_detected']}, all detected {cr['clusters_all_detected']}; "
      f"most misses: {cr['misses_by_cluster_top']}")

# rate of the channel per group, as quoted in the text
def rate(rows):
    rows = [r for r in rows if r.get("parsed")]
    k = sum(pos(r) for r in rows)
    return {"k": k, "n": len(rows), "rate": round(k / len(rows), 4) if rows else None,
            "wilson95": wilson(k, len(rows))}
res["rates"] = {"phish_clean": rate(ph), "legit_all": rate(neg_b),
                "legit_login_by_model": rate(neg_a), "legit_pw_visible": rate(neg_c),
                "legit_pw_visible_capture_flag": rate(neg_c2)}
print()
for k, v in res["rates"].items():
    print(f"{k:<30} {v['k']}/{v['n']} = {v['rate']}  {v['wilson95']}")
Path("results_f1_recheck.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
print("-> results_f1_recheck.json")
