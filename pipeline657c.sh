#!/bin/zsh
# Genisletilmis kume (1206 sayfa / 248 kume) icin kalan analizler:
# dislama galerileri (Tablo 5), algisal fark edilebilirlik (Tablo 7, Sekil 3)
# ve eslestirilmis karsilastirmalar (metin ici farklar).
# Hepsi yayimlanan turla ayni parametreler; cikti adlari _u ekli.
setopt nullglob
set -e
cd ~/phish-vlm
export PATH=/opt/homebrew/bin:$PATH
PY=./venv/bin/python

echo "=== A. dislama galerileri (Tablo 5) ==="
$PY retention_excl.py --clean feat_u_A0.npz \
  --conds A0n=feat_u_A0n.npz A1=feat_u_A1.npz A3=feat_u_A3.npz A4=feat_u_A4.npz \
          S1=feat_u_S1.npz S3=feat_u_S3.npz S7=feat_u_S7.npz \
  --clusters clusters_u_flat.json --applied pert_u \
  --out results_retention_excl_u.json 2>&1 | tail -30

echo "=== B. eslestirilmis karsilastirmalar ==="
$PY paired_contrasts.py --clean feat_u_A0.npz \
  --conds A0n=feat_u_A0n.npz A1=feat_u_A1.npz A3=feat_u_A3.npz A4=feat_u_A4.npz \
          S1=feat_u_S1.npz S3=feat_u_S3.npz S7=feat_u_S7.npz \
  --clusters clusters_u_flat.json --applied pert_u \
  --out results_paired_contrasts_u.json 2>&1 | tail -30

echo "=== C. algisal fark edilebilirlik (Tablo 7) ==="
$PY perceptibility.py --clean clean_sub_u --pert pert_u \
  --conds A1 A3 A4 S1 S3 S7 --ids ids_u.txt --sample 250 \
  --out results_perceptibility_u.json 2>&1 | tail -20

echo "PIPELINE657C BITTI"
