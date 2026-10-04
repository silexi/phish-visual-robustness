import json
R = json.load(open("results_retention_excl.json")); M = json.load(open("verify_analyst_out.json"))
print({k: v for k, v in R.items() if k != "conditions"})
print("mine dup_stats", M["dup_stats"])
print("conditions in file:", list(R["conditions"]))
e = R["conditions"]["A3"]; print("A3 n_pages", e["n_pages"], "n_applied", e["n_applied"])
RUNGS = R["rung_order"]
bad = 0
for v in ["full", "self_excl", "dup_excl"]:
    f = e["variants"][v]; m = M["results"][v]
    print(f"[{v}] file n={f['n_pages']} k={f['n_clusters']} dropped={f['n_dropped_unretrievable_applied']} | mine n={m['n']} k={m['k']}")
    for rn in RUNGS:
        a = f["rungs"][rn]; b = m["rungs"][rn]["applied_only"]
        same = all(a[x] == b[x] for x in ["cluster_mean", "boot95", "page_mean", "n_clusters", "clusters_fully_retained", "wilson95_full"])
        bad += (not same)
        print(f"   {rn:14s} file cm={a['cluster_mean']:.4f} boot={a['boot95']} pm={a['page_mean']:.4f} k={a['n_clusters']} full={a['clusters_fully_retained']}"
              f" | mine cm={b['cluster_mean']:.4f} boot={b['boot95']} pm={b['page_mean']:.4f} k={b['n_clusters']} full={b['clusters_fully_retained']} {'OK' if same else 'DIFF'}")
    if v != "full":
        print("   full-gallery-same-pop:", "  ".join(f"{rn.split('_')[-1]}={f['rungs_full_gallery_same_population'][rn]['cluster_mean']:.3f}" for rn in RUNGS))
print("mismatching rung rows:", bad)
# also the other conditions' headline rows for context
for c in R["conditions"]:
    if c == "A3":
        continue
    ee = R["conditions"][c]
    for v in ["full", "self_excl", "dup_excl"]:
        f = ee["variants"][v]
        print(f"{c:4s} {v:10s} n={f['n_pages']:3d} k={f['n_clusters']:3d} " + "  ".join(f"{rn.split('_')[-1]}={f['rungs'][rn]['cluster_mean']:.3f}" for rn in RUNGS))
