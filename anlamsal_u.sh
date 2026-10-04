#!/bin/zsh
# Anlamsal duzey analizleri, genisletilmis kume icin.
#
# misattribution.py yollari KENDI dosya konumuna gore cozuyor (R =
# Path(__file__).resolve().parent), f1_recheck.py ise calisma dizinine gore ve
# hic argumani yok. Bu yuzden her ikisinin bir kopyasi u_vlm/ icine konuyor ve
# oradan calistiriliyor; u_vlm icinde vlm/ -> vlm_u/, clusters_flat.json ->
# clusters_u_flat.json baglari var. Veri kopyalanmiyor.
#
# Yuvarlama, Tablo 4-5-7'de oldugu gibi 6 basamaga cikariliyor: makale uc
# basamagi tam duyarliktan basiyor.
setopt nullglob
set -e
cd ~/phish-vlm
export PATH=/opt/homebrew/bin:$PATH
PY=$PWD/venv/bin/python

echo "=== 0. yuvarlama 4 -> 6 ==="
for f in vlm_retention.py misattribution.py f1_recheck.py; do
  [ -f "$f.bak6" ] || cp "$f" "$f.bak6"
  cp "$f.bak6" "$f"
  perl -pi -e 'if (/round\(/) { s/, 4\)/, 6)/g; s/, 3\)/, 6)/g; }' "$f"
  echo "  $f: $(grep -c 'round(.*, 6)' $f) cagri"
done

echo "=== 1. mesru taraf degismiyor, vlm_u icine baglaniyor ==="
[ -e vlm_u/desc_BENIGN.jsonl ] || ln -s "$PWD/vlm/desc_BENIGN.jsonl" vlm_u/desc_BENIGN.jsonl
ls vlm_u/ | tr '\n' ' '; echo

echo "=== 2. calisma dizini ==="
rm -rf u_vlm && mkdir -p u_vlm
ln -s "$PWD/vlm_u" u_vlm/vlm
ln -s "$PWD/clusters_u_flat.json" u_vlm/clusters_flat.json
ln -s "$PWD/benign_replay" u_vlm/benign_replay
ln -s "$PWD/benign" u_vlm/benign
ln -s "$PWD/ids_u.txt" u_vlm/ids_u.txt
cp misattribution.py f1_recheck.py u_vlm/
ls u_vlm/ | tr '\n' ' '; echo

echo "=== 3. anlamsal korunma (Tablo 6) ==="
$PY vlm_retention.py --clean vlm_u/desc_A0.jsonl \
  --conds A0n=vlm_u/desc_A0n.jsonl A1=vlm_u/desc_A1.jsonl A3=vlm_u/desc_A3.jsonl \
          A4=vlm_u/desc_A4.jsonl S1=vlm_u/desc_S1.jsonl S3=vlm_u/desc_S3.jsonl \
          S7=vlm_u/desc_S7.jsonl \
  --clusters clusters_u_flat.json --out results_vlm_u.json 2>&1 | tail -20

echo "=== 4. yanlis atif (Tablo 6 y.atif sutunu) ==="
cd u_vlm
$PY misattribution.py --out ../results_misattribution_u.json 2>&1 | tail -15

echo "=== 5. niyet dedektoru (Tablo 8) ==="
$PY f1_recheck.py 2>&1 | tail -20
[ -f results_f1_recheck.json ] && mv results_f1_recheck.json ../results_f1_recheck_u.json
cd ..

echo "ANLAMSAL_U BITTI"
ls -la results_vlm_u.json results_misattribution_u.json results_f1_recheck_u.json 2>/dev/null
