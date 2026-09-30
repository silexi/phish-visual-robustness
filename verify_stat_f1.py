import json, math, os, collections, sys

def load(p):
    rows=[]
    with open(p) as f:
        for i,l in enumerate(f):
            l=l.strip()
            if not l:
                print("BLANK LINE at", p, i); continue
            rows.append(json.loads(l))
    return rows

def wilson(k,n,z=1.959964):
    if n==0: return (float('nan'),float('nan'))
    p=k/n
    d=1+z*z/n
    c=(p+z*z/(2*n))/d
    h=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/d
    return (round(c-h,4),round(c+h,4))

A=load('vlm/desc_A0.jsonl'); B=load('vlm/desc_BENIGN.jsonl')
print("rows A0:",len(A),"rows BENIGN:",len(B))
print("keys A0:", sorted(set(k for r in A for k in r)))
print("keys BEN:", sorted(set(k for r in B for k in r)))

# duplicate images
for nm,R in (("A0",A),("BEN",B)):
    c=collections.Counter(r['image'] for r in R)
    d=[k for k,v in c.items() if v>1]
    print(nm,"distinct images:",len(c),"dups:",d)
print("overlap A0/BEN images:", set(r['image'] for r in A)&set(r['image'] for r in B))
print("A0 names starting b_:", sum(r['image'].startswith('b_') for r in A))
print("BEN names NOT starting b_:", [r['image'] for r in B if not r['image'].startswith('b_')])
print("BEN names not .png:", [r['image'] for r in B if not r['image'].endswith('.png')])

def parsed_ok(r):
    return isinstance(r.get('parsed'),dict)
for nm,R in (("A0",A),("BEN",B)):
    fails=[r for r in R if not parsed_ok(r)]
    print(nm,"parse failures:",len(fails))
    for r in fails:
        print("   ",r['image'],"parsed=",repr(r.get('parsed'))[:80],"parse_error=",repr(r.get('parse_error'))[:160], "raw_len=", len(r.get('raw') or '') if r.get('raw') else None)
    weird=[r for r in R if parsed_ok(r) and r.get('parse_error') not in (None,'')]
    print(nm,"parsed dict but parse_error set:",len(weird))
    miss=[r['image'] for r in R if parsed_ok(r) and ('asks_for_credentials' not in r['parsed'] or 'page_type' not in r['parsed'])]
    print(nm,"parsed missing afc/page_type:",miss)

for nm,R in (("A0",A),("BEN",B)):
    c=collections.Counter((type(r['parsed'].get('asks_for_credentials')).__name__, repr(r['parsed'].get('asks_for_credentials'))) for r in R if parsed_ok(r))
    print(nm,"afc value dist:",dict(c))
    pt=collections.Counter(repr(r['parsed'].get('page_type')) for r in R if parsed_ok(r))
    print(nm,"page_type dist:",dict(pt))

def pred_pos(r):
    return parsed_ok(r) and r['parsed'].get('asks_for_credentials') is True

Ap=[r for r in A if parsed_ok(r)]; Bp=[r for r in B if parsed_ok(r)]
TP=sum(pred_pos(r) for r in Ap); FN=len(Ap)-TP
print("\nPhishing parsed:",len(Ap),"TP=",TP,"FN=",FN)
FP_all=sum(pred_pos(r) for r in Bp); TN_all=len(Bp)-FP_all
print("Legit parsed:",len(Bp),"FP_all=",FP_all,"TN_all=",TN_all)

def metrics(TP,FN,FP,TN,label):
    prec=TP/(TP+FP) if TP+FP else float('nan')
    rec=TP/(TP+FN)
    fpr=FP/(FP+TN) if FP+TN else float('nan')
    f1=2*prec*rec/(prec+rec) if prec+rec else float('nan')
    print(f"[{label}] TP={TP} FN={FN} FP={FP} TN={TN} prec={prec:.4f} rec={rec:.4f} rec_wilson={wilson(TP,TP+FN)} fpr={fpr:.4f} fpr_wilson={wilson(FP,FP+TN)} F1={f1:.4f} prec_wilson={wilson(TP,TP+FP)}")

S={'login','password_entry','mfa_challenge'}
neg_a=[r for r in Bp if r['parsed'].get('page_type') in S]
neg_a_ci=[r for r in Bp if isinstance(r['parsed'].get('page_type'),str) and r['parsed']['page_type'].strip().lower() in S]
print("\n(a) exact-match n=",len(neg_a), collections.Counter(r['parsed']['page_type'] for r in neg_a))
print("(a) case/strip-insensitive n=",len(neg_a_ci))
FP=sum(pred_pos(r) for r in neg_a); TN=len(neg_a)-FP
metrics(TP,FN,FP,TN,"a")
print("(a) TN rows:",[(r['image'],r['parsed'].get('brand'),r['parsed'].get('page_type'),r['parsed'].get('asks_for_credentials')) for r in neg_a if not pred_pos(r)])
metrics(TP,FN,FP_all,TN_all,"b")
print("(b) FP page_types:",collections.Counter(r['parsed']['page_type'] for r in Bp if pred_pos(r)))
print("(b) afc True but page_type not in S:",[(r['image'],r['parsed'].get('brand'),r['parsed'].get('page_type')) for r in Bp if pred_pos(r) and r['parsed'].get('page_type') not in S])

M=load('benign_replay/manifest_replay.jsonl')
print("\nreplay manifest rows:",len(M),"keys:",sorted(set(k for r in M for k in r)))
ms=collections.Counter(r['safe'] for r in M); print("replay dup safe:",[k for k,v in ms.items() if v>1])
mm={r['safe']:r for r in M}
stem=lambda r: os.path.splitext(r['image'])[0]
missing=[stem(r) for r in B if stem(r) not in mm]
print("BEN stems not in replay manifest:",missing)
print("replay ok dist:",collections.Counter(r.get('ok') for r in M))
print("replay pw_visible dist:",collections.Counter(repr(r.get('pw_visible')) for r in M))
neg_c=[r for r in Bp if stem(r) in mm and mm[stem(r)].get('pw_visible') is True]
print("(c) n=",len(neg_c))
FPc=sum(pred_pos(r) for r in neg_c); TNc=len(neg_c)-FPc
metrics(TP,FN,FPc,TNc,"c")
print("(c) TN rows:",[(r['image'],r['parsed'].get('brand'),r['parsed'].get('page_type')) for r in neg_c if not pred_pos(r)])
ct=collections.Counter()
for r in Bp:
    pv=mm.get(stem(r),{}).get('pw_visible')
    ct[(pv, r['parsed'].get('page_type') in S, pred_pos(r))]+=1
print("crosstab (pw_visible, page_type_in_loginset, pred_pos):")
for k in sorted(ct, key=str): print("   ",k,ct[k])
print("pw_visible True & page_type not in S:",[(r['image'],r['parsed'].get('brand'),r['parsed'].get('page_type'),pred_pos(r)) for r in Bp if mm.get(stem(r),{}).get('pw_visible') is True and r['parsed'].get('page_type') not in S])
print("page_type in S & pw_visible not True:",[(r['image'],r['parsed'].get('brand'),r['parsed'].get('page_type'),mm.get(stem(r),{}).get('pw_visible'),pred_pos(r)) for r in Bp if r['parsed'].get('page_type') in S and mm.get(stem(r),{}).get('pw_visible') is not True])
# n_password in replay for the 120
print("replay n_password>0 among BEN:", sum(1 for r in Bp if mm.get(stem(r),{}).get('n_password',0)>0), " pw_visible True among BEN:", sum(1 for r in Bp if mm.get(stem(r),{}).get('pw_visible') is True), " among all 120 rows incl parse-fail:", sum(1 for r in B if mm.get(stem(r),{}).get('pw_visible') is True))

TPn=TP; FNn=len(A)-TP
print("\nparse-fail-as-negative, phishing side: TP=",TPn,"FN=",FNn,"recall=",round(TPn/len(A),4), wilson(TPn,len(A)))
for lab,(fp,tn) in (("a",(FP,TN)),("b",(FP_all,TN_all)),("c",(FPc,TNc))):
    prec=TPn/(TPn+fp); rec=TPn/len(A); f1=2*prec*rec/(prec+rec)
    print(f"  [{lab}] legit parse-fails not added to neg set: prec={prec:.4f} rec={rec:.4f} F1={f1:.4f}")
fpb=FP_all; tnb=TN_all+(len(B)-len(Bp))
print("  [b] legit parse-fail as TN: FP=",fpb,"TN=",tnb,"FPR=",round(fpb/(fpb+tnb),4), wilson(fpb,fpb+tnb))
# legit parse failure: is its stem pw_visible / would it be in (a) or (c)?
for r in B:
    if not parsed_ok(r):
        print("  legit parse-fail row:", r['image'], "replay:", mm.get(stem(r)))

replay_ids=sorted(r['safe'] for r in M)
ben_stems=[stem(r) for r in B]
print("\nBEN stems sorted == first 120 sorted replay ids:", sorted(ben_stems)==replay_ids[:120])
print("BEN file order is sorted:", ben_stems==sorted(ben_stems))
print("BEN file order == replay_ids[:120]:", ben_stems==replay_ids[:120])
shots=sorted(os.listdir('benign_replay/shots'))
print("shots dir n=",len(shots),"ext dist:",collections.Counter(os.path.splitext(s)[1] for s in shots))
shot_pngs=sorted(s for s in shots if s.endswith('.png'))
print("shots png n=",len(shot_pngs),"first120 == BEN images(sorted):", shot_pngs[:120]==sorted(r['image'] for r in B))
print("BEN images not in shots:",[r['image'] for r in B if r['image'] not in set(shots)])
print("replay ids with no shot png:",[s for s in replay_ids if s+'.png' not in set(shots)])
print("shots pngs not in replay manifest:",[s for s in shot_pngs if os.path.splitext(s)[0] not in mm])
print("replay ok!=True ids:",[r['safe'] for r in M if r.get('ok') is not True])
d1=set(ben_stems)-set(replay_ids[:120]); d2=set(replay_ids[:120])-set(ben_stems)
print("in BEN not first120:",sorted(d1)); print("in first120 not BEN:",sorted(d2))
# replay-manifest file order vs sorted
print("replay manifest file order sorted?:", [r['safe'] for r in M]==replay_ids)

F=load('benign/fetch_manifest.jsonl')
print("\nfetch manifest rows:",len(F),"keys:",sorted(set(k for r in F for k in r)))
fc=collections.Counter(r['safe'] for r in F); print("fetch dup safe n:",sum(1 for v in fc.values() if v>1), "examples:",[k for k,v in fc.items() if v>1][:10])
fm={}
for r in F: fm.setdefault(r['safe'],[]).append(r)
nob=[s for s in ben_stems if s not in fm]; print("BEN stems not in fetch manifest:",nob)
brands=[fm[s][0].get('brand') for s in ben_stems if s in fm]
bc=collections.Counter(brands)
print("brands n=",len(brands),"distinct exact=",len(bc),"dups:",{k:v for k,v in bc.items() if v>1})
bcl=collections.Counter((b or '').strip().lower() for b in brands)
print("distinct case-insensitive=",len(bcl),"dups:",{k:v for k,v in bcl.items() if v>1})
brands156=[fm[s][0].get('brand') for s in replay_ids if s in fm]
print("156 replay: brands n=",len(brands156),"distinct=",len(set(brands156)), "missing in fetch:",[s for s in replay_ids if s not in fm])
conf=[(s,[x.get('brand') for x in v]) for s,v in fm.items() if len(v)>1 and len(set(x.get('brand') for x in v))>1]
print("fetch same-safe conflicting brands:",conf[:10])
print("fetch ok dist:",collections.Counter(r.get('ok') for r in F))
print("BEN stems whose fetch ok!=True:",[s for s in ben_stems if s in fm and fm[s][0].get('ok') is not True])
# fetch brand -> how many safes per brand overall (are there multiple fetches per brand?)
fb=collections.Counter(r.get('brand') for r in F)
print("fetch: distinct brands total=",len(fb),"brands with >1 row:",sum(1 for v in fb.values() if v>1))
# url domain vs brand: check any BEN whose url host does not contain brand token
import urllib.parse
odd=[]
for s in ben_stems:
    if s in fm:
        u=fm[s][0].get('url') or ''; fu=fm[s][0].get('final_url') or ''
        h=urllib.parse.urlparse(u).netloc.lower(); fh=urllib.parse.urlparse(fu).netloc.lower()
        b=(fm[s][0].get('brand') or '').lower().replace(' ','')
        if b and b not in h and b not in fh: odd.append((s,fm[s][0].get('brand'),h,fh))
print("BEN brand token not in url host nor final host:",len(odd))
for o in odd: print("   ",o)

print("\n(a) negative-set rows: stem | manifest brand | model brand | page_type | afc | pw_visible | url")
for r in neg_a:
    s=stem(r); print("  ",s,'|', fm.get(s,[{}])[0].get('brand'), '|', r['parsed'].get('brand'), '|', r['parsed'].get('page_type'), '|', r['parsed'].get('asks_for_credentials'), '|', mm.get(s,{}).get('pw_visible'), '|', fm.get(s,[{}])[0].get('url'))
print("\nAll BEN rows with afc True: stem | manifest brand | model brand | page_type | pw_visible")
for r in Bp:
    if pred_pos(r):
        s=stem(r); print("  ",s,'|', fm.get(s,[{}])[0].get('brand'), '|', r['parsed'].get('brand'), '|', r['parsed'].get('page_type'), '|', mm.get(s,{}).get('pw_visible'))
