#!/bin/zsh
# Rest of the run, chained so nothing waits on a human.
#  1. wait for perturbation renders and the benign replay to finish
#  2. build the benign reference / calibration split (brand-disjoint)
#  3. extract pixel+CLIP features for every condition
#  4. score all rungs under the shared decision rule
#  5. hand off to the semantic rung, which is the slow one
cd "$(dirname "$0")"
export PATH=/opt/homebrew/bin:$PATH
set -e

echo "=== 1. bekleniyor ==="
while [ "$(pgrep -f perturb.py | wc -l)" -gt 0 ] || [ "$(pgrep -f render_html.py | wc -l)" -gt 0 ]; do
  sleep 30
done
echo "bozma + replay tamam"

echo "=== 2. mesru bolunme ==="
./venv/bin/python split_benign.py

echo "=== 3. ozellik cikarma ==="
for d in benign_ref benign_cal; do
  [ -f "feat_$d.npz" ] || ./venv/bin/python features.py --dir "$d" --out "feat_$d.npz" 2>&1 | tail -2
done
for c in A0n A1 A3 A4 S1 S3 S7; do
  [ -f "feat_$c.npz" ] || ./venv/bin/python features.py --dir "pert/$c" --out "feat_$c.npz" 2>&1 | tail -2
done
[ -f feat_A0.npz ] || ./venv/bin/python features.py --dir corpus/shots --out feat_A0.npz 2>&1 | tail -2

echo "=== 4. skorlama ==="
./venv/bin/python score.py \
  --ref feat_benign_ref.npz --cal feat_benign_cal.npz \
  --clusters clusters.json --fpr 0.05 \
  --queries A0=feat_A0.npz A0n=feat_A0n.npz A1=feat_A1.npz A3=feat_A3.npz \
            A4=feat_A4.npz S1=feat_S1.npz S3=feat_S3.npz S7=feat_S7.npz \
  --out results_pixel_clip.json

echo "=== 5. anlamsal katman basliyor ==="
N=250 ./vlm_batch.sh
echo "HEPSI BITTI"
