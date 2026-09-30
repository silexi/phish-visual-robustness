import json, os, collections
A=[json.loads(l) for l in open('vlm/desc_A0.jsonl') if l.strip()]
C=json.load(open('clusters_flat.json'))
print("clusters_flat entries:",len(C),"distinct clusters:",len(set(C.values())))
stems=[os.path.splitext(r['image'])[0] for r in A]
inC=[s for s in stems if s in C]
print("A0 stems in clusters_flat:",len(inC))
cl=collections.Counter(C[s] for s in inC)
print("A0: distinct clusters:",len(cl),"cluster size dist among A0:",sorted(cl.values(),reverse=True)[:25])
print("A0 singletons:",sum(1 for v in cl.values() if v==1),"pages in clusters>=2:",sum(v for v in cl.values() if v>=2))
# per-cluster agreement of afc
ok=lambda r: isinstance(r.get('parsed'),dict)
byc=collections.defaultdict(list)
for r in A:
    s=os.path.splitext(r['image'])[0]
    byc[C.get(s,'?')].append(None if not ok(r) else (r['parsed'].get('asks_for_credentials') is True))
mixed=sum(1 for v in byc.values() if len(set(x for x in v if x is not None))>1)
print("clusters with mixed afc:",mixed,"of",len(byc))
# cluster-level recall (majority / any)
maj=[]; anyp=[]
for k,v in byc.items():
    vv=[x for x in v if x is not None]
    if not vv: continue
    maj.append(sum(vv)>=len(vv)/2); anyp.append(any(vv))
print("cluster-level recall (majority):",sum(maj),"/",len(maj),"=",round(sum(maj)/len(maj),4))
# exact-duplicate parsed outputs
sig=collections.Counter(json.dumps(r['parsed'],sort_keys=True) for r in A if ok(r))
print("distinct parsed outputs among 196:",len(sig),"; outputs appearing >1:",sum(1 for v in sig.values() if v>1),"; rows covered by repeated outputs:",sum(v for v in sig.values() if v>1))
# parse-failure cluster
pf=[os.path.splitext(r['image'])[0] for r in A if not ok(r)]
print("parse-fail clusters:",[C.get(s) for s in pf])
# brand dist of phishing side by model
print("A0 model brand top:",collections.Counter((r['parsed'].get('brand') or '').lower() for r in A if ok(r)).most_common(12))
