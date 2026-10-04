#!/bin/zsh
# 657-sayfa duyarlilik analizi: content gate 20 shard'in tamamina uygulandiginda
# eklenen sayfalarla olcumun yeniden yapilmasi.
#
# pipeline3.sh ile AYNI protokol, ayni parametreler. Hicbir mevcut cikti
# uzerine yazilmaz: butun yeni ciktilar _u (union) ekiyle.
#
# Mevcut renderlar yeniden uretilmez: clean_f/shots (1.702 id) ve
# clean_new657/shots (657 id) birlestirilir; bozulma renderlarinda da mevcut
# pert/<kosul> dosyalari baglanir, yalnizca yeni sayfalar render edilir.
set -e
cd ~/phish-vlm
export PATH=/opt/homebrew/bin:$PATH
PY=./venv/bin/python
CONDS=(A0n A1 A3 A4 S1 S3 S7)

echo "=== 0. birlesik temiz galeri (sembolik bag) ==="
rm -rf clean_u && mkdir -p clean_u/shots
for f in clean_f/shots/*.png;        do ln -sf "$PWD/$f" "clean_u/shots/${f:t}"; done
for f in clean_new657/shots/*.png;   do ln -sf "$PWD/$f" "clean_u/shots/${f:t}"; done
echo "birlesik temiz render: $(ls clean_u/shots | wc -l)"

echo "=== 1. sadakat kapisi (birlesik) ==="
$PY fidelity.py --dir clean_u/shots --out fidelity_u.jsonl 2>&1 | tail -6
$PY -c "
import json
rows=[json.loads(l) for l in open('fidelity_u.jsonl') if 'block_area' in json.loads(l)]
keep=[r['safe'] for r in rows if r['block_area']>=0.02]
open('ids_u.txt','w').write('\n'.join(sorted(keep))+'\n')
print('sadakat kapisi:', len(keep), '/', len(rows))
"

echo "=== 2. kit kumeleme (maxdist 16, pipeline3 ile ayni) ==="
$PY dedup.py --dir clean_u/shots --ids ids_u.txt --out clusters_u.json --maxdist 16 2>&1 | tail -6
$PY -c "
import json
json.dump(json.load(open('clusters_u.json'))['clusters'], open('clusters_u_flat.json','w'))
print('kume sayisi:', len(set(json.load(open('clusters_u_flat.json')).values())))
"

echo "=== 3. bozulma kosullari: yalnizca YENI sayfalar render edilir ==="
comm -23 <(sort ids_u.txt) <(sort ids_f.txt) > /tmp/ids_u_new.txt
echo "yeni sayfa: $(wc -l < /tmp/ids_u_new.txt)   mevcut: $(wc -l < ids_f.txt)"
rm -rf pert_u && mkdir -p pert_u
for c in $CONDS; do
  $PY perturb.py --corpus corpus --cond $c --ids /tmp/ids_u_new.txt --outdir pert_u 2>&1 | tail -1 &
done
wait
for c in $CONDS; do
  for f in pert/$c/*.png; do ln -sf "$PWD/$f" "pert_u/$c/${f:t}"; done
  # retention2.py --applied, kosul basina manifest_<c>.jsonl okuyor ve her
  # sayfanin 'applied' bayragina bakiyor. pert_u'daki manifest yalnizca yeni
  # sayfalari tasiyor, bu yuzden eski manifest ile birlestiriliyor.
  cat pert/manifest_$c.jsonl pert_u/manifest_$c.jsonl > /tmp/m_$c.jsonl
  mv /tmp/m_$c.jsonl pert_u/manifest_$c.jsonl
  echo "pert_u/$c: $(ls pert_u/$c | wc -l) png, manifest $(wc -l < pert_u/manifest_$c.jsonl) satir"
done

echo "=== 4. ozellik cikarma ==="
rm -rf clean_sub_u && mkdir -p clean_sub_u
while read -r id; do
  [ -f "clean_u/shots/$id.png" ] && ln -sf "$PWD/clean_u/shots/$id.png" "clean_sub_u/$id.png"
done < ids_u.txt
echo "clean_sub_u: $(ls clean_sub_u | wc -l)"
rm -f feat_u_*.npz
$PY features.py --dir clean_sub_u --out feat_u_A0.npz 2>&1 | tail -1
for c in $CONDS; do
  $PY features.py --dir "pert_u/$c" --out "feat_u_$c.npz" 2>&1 | tail -1
done

echo "=== 5. skorlama ==="
$PY score.py \
  --ref feat_benign_ref.npz --cal feat_benign_cal.npz \
  --clusters clusters_u_flat.json --fpr 0.05 \
  --queries A0=feat_u_A0.npz A0n=feat_u_A0n.npz A1=feat_u_A1.npz A3=feat_u_A3.npz \
            A4=feat_u_A4.npz S1=feat_u_S1.npz S3=feat_u_S3.npz S7=feat_u_S7.npz \
  --out results_pixel_clip_u.json 2>&1 | tail -12

echo "=== 6. korunma orani (retention2, denetim sonrasi surum) ==="
$PY retention2.py --clean feat_u_A0.npz \
  --conds A0n=feat_u_A0n.npz A1=feat_u_A1.npz A3=feat_u_A3.npz A4=feat_u_A4.npz \
          S1=feat_u_S1.npz S3=feat_u_S3.npz S7=feat_u_S7.npz \
  --clusters clusters_u_flat.json --applied pert_u --out results_retention_u.json 2>&1 | tail -20

echo "PIPELINE657 BITTI"
