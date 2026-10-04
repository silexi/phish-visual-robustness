import numpy as np, json
from pathlib import Path
R=Path(__file__).resolve().parent
Z=dict(np.load(R/"audit_hits.npz",allow_pickle=True))
clean_ids=[str(x) for x in Z["clean_ids"]]
cmap=json.load(open(R/"clusters_f.json"))["clusters"]
ccl=np.array([cmap.get(s,s) for s in clean_ids]); p2={s:i for i,s in enumerate(clean_ids)}
fc={json.loads(l)["safe"]:json.loads(l).get("frac_changed",0) for l in open(R/"pert/manifest_A0n.jsonl")}
ids=[str(x) for x in Z["A0n_all__ids"]]
print("=== 18. A0n colour misses: are the images bit-identical? ===")
t1=Z["A0n_all__top1__P_color"]; qi=np.array([p2[s] for s in ids])
h=(ccl[t1]==ccl[qi]); ss=Z["A0n_all__selfsim__P_color"]; mx=Z["A0n_all__maxsim__P_color"]
mi=np.flatnonzero(~h)
print(f"misses={len(mi)}  bit-identical(frac_changed==0)={sum(1 for i in mi if fc[ids[i]]==0)}")
print("  sample: id, frac_changed, self_corr, max_corr, gap, top1_cluster_size")
from collections import Counter
sz=Counter(ccl.tolist())
for i in mi[:10]:
    print(f"   {ids[i]} fc={fc[ids[i]]:.5f} self={ss[i]:.6f} max={mx[i]:.6f} gap={mx[i]-ss[i]:.2e} top1_clu={ccl[t1[i]]}(n={sz[ccl[t1[i]]]}) own={ccl[qi[i]]}(n={sz[ccl[qi[i]]]})")
print(f"  median gap over misses={np.median(mx[mi]-ss[mi]):.3e}   n_with_gap<1e-9={(mx[mi]-ss[mi]<1e-9).sum()}")
print(f"  A0n colour: self_corr==1 exactly on {(ss>=1-1e-12).mean():.3f} of rows")
