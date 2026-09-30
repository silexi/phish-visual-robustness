#!/bin/zsh
# Final chain: finish the perturbation set, rebuild the benign side with the
# targeted reference brands, extract every feature, score all rungs under the
# shared decision rule, then hand off to the slow semantic rung.
cd "$(dirname "$0")"
export PATH=/opt/homebrew/bin:$PATH

echo "=== 1. eksik bozma renderlari bekleniyor ==="
while [ "$(pgrep -f perturb.py | wc -l)" -gt 0 ]; do sleep 30; done

echo "=== 2. birlestirme ==="
for c in A0n A1 A3 A4 S1 S3 S7; do
  if [ -d "pert_fill/$c" ]; then
    cp -n pert_fill/$c/*.png pert/$c/ 2>/dev/null
  fi
  # merge the efficacy manifests too; `applied` gates the analysis
  [ -f "pert_fill/manifest_$c.jsonl" ] && cat "pert_fill/manifest_$c.jsonl" >> "pert/manifest_$c.jsonl"
  echo "  $c: $(ls pert/$c | wc -l | tr -d ' ') gorsel"
done

echo "=== 3. mesru replay (yeni sayfalar dahil) ==="
rm -rf benign_replay
./venv/bin/python render_html.py --htmldir benign/html --outdir benign_replay 2>&1 | tail -2

echo "=== 4. marka-ayrik bolunme ==="
./venv/bin/python split_benign.py

echo "=== 5. ozellik cikarma ==="
rm -f feat_benign_ref.npz feat_benign_cal.npz
for d in benign_ref benign_cal; do
  ./venv/bin/python features.py --dir "$d" --out "feat_$d.npz" 2>&1 | grep -E "piksel |clip |gorsel"
done
for c in A0n A1 A3 A4 S1 S3 S7; do
  rm -f "feat_$c.npz"
  ./venv/bin/python features.py --dir "pert/$c" --out "feat_$c.npz" 2>&1 | grep -E "piksel |clip "
done
# clean baseline restricted to the same id set as the conditions
mkdir -p clean_sub && rm -f clean_sub/*.png
while read -r id; do
  [ -f "corpus/shots/$id.png" ] && ln -sf "$PWD/corpus/shots/$id.png" "clean_sub/$id.png"
done < ids_fid_full.txt
rm -f feat_A0.npz
./venv/bin/python features.py --dir clean_sub --out feat_A0.npz 2>&1 | grep -E "piksel |clip "

echo "=== 6. skorlama (ortak karar kurali, FPR=%5) ==="
./venv/bin/python score.py \
  --ref feat_benign_ref.npz --cal feat_benign_cal.npz \
  --clusters clusters_flat.json --fpr 0.05 \
  --queries A0=feat_A0.npz A0n=feat_A0n.npz A1=feat_A1.npz A3=feat_A3.npz \
            A4=feat_A4.npz S1=feat_S1.npz S3=feat_S3.npz S7=feat_S7.npz \
  --out results_pixel_clip.json 2>&1 | tail -12

echo "=== 7. anlamsal katman ==="
N=200 ./vlm_batch.sh
echo "HEPSI BITTI"
