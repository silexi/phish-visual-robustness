#!/bin/zsh
# pipeline657.sh adim 4-6. Adim 0-3 tamamlandi:
#   clean_u/shots 2359, ids_u.txt 1206, clusters_u_flat.json 248 kume,
#   pert_u/<kosul> 1206 png + 1206 manifest satiri.
# zsh bos glob'da hata verdigi icin nullglob aciliyor.
setopt nullglob
set -e
cd ~/phish-vlm
export PATH=/opt/homebrew/bin:$PATH
PY=./venv/bin/python
CONDS=(A0n A1 A3 A4 S1 S3 S7)

echo "=== 4. ozellik cikarma ==="
rm -f feat_u_*.npz
echo "clean_sub_u: $(ls clean_sub_u | wc -l)"
$PY features.py --dir clean_sub_u --out feat_u_A0.npz 2>&1 | tail -1
for c in $CONDS; do
  $PY features.py --dir "pert_u/$c" --out "feat_u_$c.npz" 2>&1 | tail -1
done
ls -la feat_u_*.npz

echo "=== 5. skorlama ==="
$PY score.py \
  --ref feat_benign_ref.npz --cal feat_benign_cal.npz \
  --clusters clusters_u_flat.json --fpr 0.05 \
  --queries A0=feat_u_A0.npz A0n=feat_u_A0n.npz A1=feat_u_A1.npz A3=feat_u_A3.npz \
            A4=feat_u_A4.npz S1=feat_u_S1.npz S3=feat_u_S3.npz S7=feat_u_S7.npz \
  --out results_pixel_clip_u.json 2>&1 | tail -12

echo "=== 6. korunma orani (retention2) ==="
$PY retention2.py --clean feat_u_A0.npz \
  --conds A0n=feat_u_A0n.npz A1=feat_u_A1.npz A3=feat_u_A3.npz A4=feat_u_A4.npz \
          S1=feat_u_S1.npz S3=feat_u_S3.npz S7=feat_u_S7.npz \
  --clusters clusters_u_flat.json --applied pert_u --out results_retention_u.json 2>&1 | tail -25

echo "PIPELINE657B BITTI"
