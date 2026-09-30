import json, os, sys
from collections import Counter
from pathlib import Path
import numpy as np

def load(p):
    return [json.loads(l) for l in open(p, encoding="utf-8")]

print("== VLM desc files ==")
for f in ["vlm/desc_A0.jsonl", "vlm/desc_BENIGN.jsonl"]:
    if not Path(f).exists():
        print(f, "MISSING"); continue
    rs = load(f)
    stems = [Path(r["image"]).stem for r in rs]
    parsed = [r for r in rs if r.get("parsed")]
    vals = Counter((type(r["parsed"].get("asks_for_credentials")).__name__,
                    r["parsed"].get("asks_for_credentials")) for r in parsed)
    missing_key = sum(1 for r in parsed if "asks_for_credentials" not in r["parsed"])
    pt = Counter((r["parsed"].get("page_type") or "") for r in parsed)
    print(f, "n=", len(rs), "unique stems=", len(set(stems)), "parsed=", len(parsed),
          "parse_fail=", len(rs) - len(parsed))
    print("   asks_for_credentials (type,value):", dict(vals), "| missing key:", missing_key)
    print("   page_type:", dict(pt))
    print("   top-level keys sample:", sorted(rs[0].keys()))
    print("   parsed keys sample:", sorted(parsed[0]["parsed"].keys()) if parsed else None)
    pf = [r for r in rs if not r.get("parsed")]
    if pf:
        print("   parse-fail example keys:", sorted(pf[0].keys()), "| parsed value:", repr(pf[0].get("parsed"))[:80])

print("\n== benign manifests ==")
rep = {json.loads(l)["safe"]: json.loads(l) for l in open("benign_replay/manifest_replay.jsonl", encoding="utf-8")}
fetch = {json.loads(l)["safe"]: json.loads(l) for l in open("benign/fetch_manifest.jsonl", encoding="utf-8")}
print("replay n=", len(rep), "fetch n=", len(fetch))
print("replay keys sample:", sorted(next(iter(rep.values())).keys()))
print("pw_visible values:", Counter(repr(v.get("pw_visible")) for v in rep.values()))
ben = load("vlm/desc_BENIGN.jsonl")
desc_ids = [Path(r["image"]).stem for r in ben]
ids_sorted = sorted(rep)
print("desc_ids == sorted(rep)[:len(desc)]:", desc_ids == ids_sorted[:len(desc_ids)])
print("all desc ids in rep:", all(s in rep for s in desc_ids), "| all in fetch:", all(s in fetch for s in desc_ids))
print("pw_visible among described:", Counter(repr(rep[s].get("pw_visible")) for s in desc_ids))
print("pw_visible among NOT described:", Counter(repr(rep[s].get("pw_visible")) for s in ids_sorted if s not in set(desc_ids)))
# image filename vs safe id alignment
print("image basenames sample:", [r["image"] for r in ben[:3]])

print("\n== feat files / clusters ==")
clean = np.load("feat_A0.npz", allow_pickle=True)
print("feat_A0 keys:", clean.files)
cid = [str(x) for x in clean["ids"]]
print("clean n=", len(cid), "unique=", len(set(cid)))
g = clean["px_gray256"]
print("px_gray256 shape/dtype:", g.shape, g.dtype)
cmap = json.loads(Path("clusters_flat.json").read_text())
vt = Counter(type(v).__name__ for v in cmap.values())
print("cmap n=", len(cmap), "value types:", dict(vt), "| clean ids missing from cmap:", sum(1 for s in cid if s not in cmap))
cl = np.array([cmap.get(s, s) for s in cid])
print("clean_cl dtype:", cl.dtype, "n clusters:", len(set(cl.tolist())))
key = [g[i].tobytes() for i in range(len(cid))]
groups = {}
for i, k in enumerate(key):
    groups.setdefault(k, []).append(i)
print("distinct thumbs:", len(groups), "| pages with twin:", sum(len(x) for x in groups.values() if len(x) > 1),
      "| straddle groups:", sum(1 for x in groups.values() if len(set(cl[x].tolist())) > 1))
sizes = Counter(cl.tolist())
print("singleton clusters:", sum(1 for v in sizes.values() if v == 1), "| largest:", sizes.most_common(3))

for name in ["A0n", "A1", "A3", "A4", "S1", "S3", "S7"]:
    p = f"feat_{name}.npz"
    mf = Path("pert") / f"manifest_{name}.jsonl"
    if not Path(p).exists():
        print(name, "feat MISSING"); continue
    q = np.load(p, allow_pickle=True)
    qids = [str(x) for x in q["ids"]]
    inset = [s for s in qids if s in set(cid)]
    line = f"{name}: n={len(qids)} unique={len(set(qids))} in_gallery={len(inset)} manifest={'yes' if mf.exists() else 'NO'}"
    if mf.exists():
        ap = [json.loads(l) for l in mf.open(encoding="utf-8")]
        ak = Counter(repr(r.get("applied")) for r in ap)
        line += f" applied_values={dict(ak)} manifest_n={len(ap)} manifest_keys={sorted(ap[0].keys())}"
    print(line)

print("\n== existing result files ==")
for f in sorted(Path(".").glob("results_*.json")):
    print(f, os.path.getsize(f))
if Path("results_retention_excl.json").exists() and Path("results_retention_v2.json").exists():
    ex = json.load(open("results_retention_excl.json"))
    v2 = json.load(open("results_retention_v2.json"))
    print("excl meta:", {k: ex[k] for k in ex if k != "conditions"})
    for c, e in ex["conditions"].items():
        full = e["variants"]["full"]["rungs"]
        p = v2["conditions"].get(c)
        if p is None:
            print(c, "not in v2"); continue
        diffs = []
        for rn in full:
            a = full[rn]["cluster_mean"] if full[rn] else None
            b = p["rungs"][rn]["applied_only"]["cluster_mean"] if p["rungs"][rn]["applied_only"] else None
            if a != b:
                diffs.append((rn, a, b))
        print(c, "full==v2 applied_only:", "OK" if not diffs else diffs,
              "| n_applied", e["n_applied"], p["n_applied"])
        for vn in ["self_excl", "dup_excl"]:
            v = e["variants"][vn]
            print(f"   {vn}: n_pages={v['n_pages']} n_clusters={v['n_clusters']} dropped={v['n_dropped_unretrievable_applied']} "
                  + " ".join(f"{rn.split('_')[-1]}={v['rungs'][rn]['cluster_mean'] if v['rungs'][rn] else None}/{v['rungs_full_gallery_same_population'][rn]['cluster_mean'] if v['rungs_full_gallery_same_population'][rn] else None}" for rn in full))
if Path("results_f1_recheck.json").exists():
    r = json.load(open("results_f1_recheck.json"))
    for v in r["variants"]:
        print(v)
    print(r["rates"])
