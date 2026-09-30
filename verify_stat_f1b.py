import json, os, re, collections, glob
def load(p):
    return [json.loads(l) for l in open(p) if l.strip()]
A=load('vlm/desc_A0.jsonl'); B=load('vlm/desc_BENIGN.jsonl')
F=load('benign/fetch_manifest.jsonl'); M=load('benign_replay/manifest_replay.jsonl')
stem=lambda r: os.path.splitext(r['image'])[0]
ben=[stem(r) for r in B]
print("b_143efe9ea64e41 in BEN:", 'b_143efe9ea64e41' in ben, " in replay:", 'b_143efe9ea64e41' in {r['safe'] for r in M})
fm={}
for r in F: fm.setdefault(r['safe'],[]).append(r)
# raw text of parse failures: try to find asks_for_credentials / page_type
for nm,R in (("A0",A),("BEN",B)):
    for r in R:
        if not isinstance(r.get('parsed'),dict):
            raw=r.get('raw') or ''
            m=re.search(r'"asks_for_credentials"\s*:\s*(true|false)',raw)
            m2=re.search(r'"page_type"\s*:\s*"([^"]*)"',raw)
            m3=re.search(r'"brand"\s*:\s*"([^"]*)"',raw)
            print(nm, r['image'], "afc_in_raw=", m.group(1) if m else None, "page_type_in_raw=", m2.group(1) if m2 else None, "brand=", m3.group(1) if m3 else None)
            print("   raw head:", repr(raw[:300]))
            print("   raw tail:", repr(raw[-200:]))
# organization-level brand grouping for the 120
brands=[(s,fm[s][0]['brand'],fm[s][0].get('url'),fm[s][0].get('final_url')) for s in ben]
import urllib.parse
def reg_dom(u):
    h=urllib.parse.urlparse(u or '').netloc.lower()
    parts=h.split('.')
    if len(parts)>=3 and parts[-2] in ('co','com','gov','edu','org','net','ac') and len(parts[-1])==2:
        return '.'.join(parts[-3:])
    return '.'.join(parts[-2:]) if len(parts)>=2 else h
dom=collections.Counter(reg_dom(u) for s,b,u,fu in brands)
print("distinct registrable domains (url) among 120:", len(dom), "dups:", {k:v for k,v in dom.items() if v>1})
fdom=collections.Counter(reg_dom(fu) for s,b,u,fu in brands)
print("distinct registrable domains (final_url) among 120:", len(fdom), "dups:", {k:v for k,v in fdom.items() if v>1})
print("rows with final registrable domain in dups:")
for s,b,u,fu in brands:
    d=reg_dom(fu)
    if fdom[d]>1: print("   ",s,b,u,'->',fu)
print("rows with url registrable domain in dups:")
for s,b,u,fu in brands:
    d=reg_dom(u)
    if dom[d]>1: print("   ",s,b,u)
# microsoft-family in 120
ms=[(s,b) for s,b,u,fu in brands if 'microsoft' in (u or '')+(fu or '') or 'office' in (u or '') or 'live.com' in (u or '')]
print("microsoft-family among 120:",ms)
# 156 replay: distinct registrable domains
r156=sorted(r['safe'] for r in M)
d156=collections.Counter(reg_dom(fm[s][0].get('final_url')) for s in r156)
print("156: distinct final domains:",len(d156),"dups:",{k:v for k,v in d156.items() if v>1})
# ids files
for p in sorted(glob.glob('ids_*.txt')):
    ids=[l.strip() for l in open(p) if l.strip()]
    print(p, "n=",len(ids), "first:",ids[:2])
    a0=set(r['image'] for r in A); a0s=set(os.path.splitext(x)[0] for x in a0)
    print("   overlap with A0 images:", len(set(ids)&a0), " with stems:", len(set(ids)&a0s), " with BEN images:", len(set(ids)&set(r['image'] for r in B)), " BEN stems:", len(set(ids)&set(ben)))
# results_*.json listing
print(sorted(glob.glob('results_*.json')))
# check A0 images vs other desc files (same ids?)
for p in sorted(glob.glob('vlm/desc_*.jsonl')):
    R=load(p); print(p, len(R), "overlap w/ A0 imgs:", len({r['image'] for r in R}&{r['image'] for r in A}), "parse fails:", sum(1 for r in R if not isinstance(r.get('parsed'),dict)))
# does the manifest fetch row of the 120 have pw_visible from fetch vs replay agreeing?
agree=collections.Counter((fm[s][0].get('pw_visible'), next(r for r in M if r['safe']==s).get('pw_visible')) for s in ben)
print("fetch pw_visible vs replay pw_visible among 120:",dict(agree))
# (c) using fetch-manifest pw_visible instead of replay
Bp=[r for r in B if isinstance(r.get('parsed'),dict)]
negc2=[r for r in Bp if fm[stem(r)][0].get('pw_visible') is True]
fp=sum(r['parsed'].get('asks_for_credentials') is True for r in negc2)
print("(c') fetch-manifest pw_visible: n=",len(negc2),"FP=",fp,"TN=",len(negc2)-fp)
