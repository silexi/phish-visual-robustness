#!/bin/zsh
# ON KOSUL: sonuc dosyalari 4 ondaliga yuvarliyor, makale 3 basamagi TAM
# duyarliktan basiyor. 4. ondaligi tam 5 olan hucrelerde 3 basamakli dize
# JSON'dan belirlenemiyor (34 hucre). Yuvarlamayi 6 basamaga cikarip ayni
# betikleri yeniden kosuyoruz.
#
# Hesap degismiyor: bootstrap tohumu sabit (20260922), yalnizca yazim duyarligi
# artiyor. Betiklerin yedegi .bak6 uzantisiyla aliniyor.
setopt nullglob
set -e
cd ~/phish-vlm
export PATH=/opt/homebrew/bin:$PATH
PY=$PWD/venv/bin/python

echo "=== 1. betiklerde yuvarlama 4 -> 6 ==="
for f in retention2.py retention_excl.py paired_contrasts.py perceptibility.py; do
  [ -f "$f.bak6" ] || cp "$f" "$f.bak6"
  cp "$f.bak6" "$f"
  # yalnizca icinde round( gecen satirlarda ", 4)" ve ", 3)" -> ", 6)"
  perl -pi -e 'if (/round\(/) { s/, 4\)/, 6)/g; s/, 3\)/, 6)/g; }' "$f"
  echo "  $f: $(grep -c 'round(.*, 6)' $f) cagri 6 basamaga alindi"
done

echo "=== 2. korunma (Tablo 4) ==="
$PY retention2.py --clean feat_u_A0.npz \
  --conds A0n=feat_u_A0n.npz A1=feat_u_A1.npz A3=feat_u_A3.npz A4=feat_u_A4.npz \
          S1=feat_u_S1.npz S3=feat_u_S3.npz S7=feat_u_S7.npz \
  --clusters clusters_u_flat.json --applied pert_u --out results_retention_u.json 2>&1 | tail -3

echo "=== 3. dislama (Tablo 5) + eslestirilmis ==="
cd u_run
$PY ../retention_excl.py --clean feat_A0.npz --clusters clusters_flat.json \
    --applied pert --out ../results_retention_excl_u.json 2>&1 | tail -2
$PY ../paired_contrasts.py --clean feat_A0.npz --clusters clusters_flat.json \
    --applied pert --out ../results_paired_contrasts_u.json 2>&1 | tail -2
cd ..

echo "=== 4. algisal (Tablo 7) ==="
$PY perceptibility.py --clean clean_sub_u --pert pert_u \
  --conds A0n A1 A3 A4 S1 S3 S7 --ids ids_u.txt --sample 200 \
  --out results_perceptibility_u.json 2>&1 | grep "n=200"

echo "=== 5. dogrulama: A3 HOG tam duyarlikta ==="
$PY -c "
import json
d=json.load(open('results_retention_u.json'))
r=d['conditions']['A3']['rungs']['P4_hog']['applied_only']
print('  cluster_mean      :', r['cluster_mean'])
print('  tam kesir         :', r['clusters_fully_retained'], '/', r['n_clusters'],
      '=', r['clusters_fully_retained']/r['n_clusters'])
print('  uc basamak        : %.3f' % r['cluster_mean'])
"
echo "DUYARLIK BITTI"
