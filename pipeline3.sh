#!/bin/zsh
# Everything re-derived under ONE serving mode (file://), because the loopback
# HTTP path exhausted the machine's ephemeral ports and, more importantly,
# because the two classes must not differ by how their bytes reached the
# renderer.  Both corpora, the clean baseline and every perturbation now take
# the identical path.
cd "$(dirname "$0")"
export PATH=/opt/homebrew/bin:$PATH

echo "=== 1. mesru replay (file://) ==="
rm -rf benign_replay
./venv/bin/python render_html.py --htmldir benign/html --outdir benign_replay 2>&1 | tail -2

echo "=== 2. kimlik avi temiz yeniden render (file://) ==="
rm -rf clean_f
mkdir -p clean_f
./venv/bin/python replay_ids.py --htmldir corpus/html --ids ids_gated_all.txt \
    --outdir clean_f 2>&1 | tail -3

echo "=== 3. sadakat kapisi (yeni renderlar uzerinde) ==="
./venv/bin/python fidelity.py --dir clean_f/shots --out fidelity_f.jsonl 2>&1 | tail -6
./venv/bin/python -c "
import json
rows=[json.loads(l) for l in open('fidelity_f.jsonl') if 'block_area' in json.loads(l)]
keep=[r['safe'] for r in rows if r['block_area']>=0.02]
open('ids_f.txt','w').write('\n'.join(sorted(keep)))
print('sadakat kapisi:', len(keep), '/', len(rows))
"

echo "=== 4. kit kumeleme ==="
./venv/bin/python dedup.py --dir clean_f/shots --ids ids_f.txt \
    --out clusters_f.json --maxdist 16 2>&1 | grep -E "kit kumesi|en buyuk|tek uyeli"
./venv/bin/python -c "
import json
json.dump(json.load(open('clusters_f.json'))['clusters'], open('clusters_flat.json','w'))
"

echo "=== 5. bozma kosullari (file://) ==="
rm -rf pert
for c in A0n A1 A3 A4 S1 S3 S7; do
  ./venv/bin/python perturb.py --corpus corpus --cond $c --ids ids_f.txt --outdir pert 2>&1 \
    | grep -E "^(A|S)" &
done
wait

echo "=== 6. ozellik cikarma ==="
rm -f feat_*.npz
./venv/bin/python split_benign.py
for d in benign_ref benign_cal; do
  ./venv/bin/python features.py --dir "$d" --out "feat_$d.npz" 2>&1 | grep -E "piksel |clip "
done
mkdir -p clean_sub && rm -rf clean_sub && mkdir -p clean_sub
while read -r id; do
  [ -f "clean_f/shots/$id.png" ] && ln -sf "$PWD/clean_f/shots/$id.png" "clean_sub/$id.png"
done < ids_f.txt
./venv/bin/python features.py --dir clean_sub --out feat_A0.npz 2>&1 | grep -E "piksel |clip "
for c in A0n A1 A3 A4 S1 S3 S7; do
  ./venv/bin/python features.py --dir "pert/$c" --out "feat_$c.npz" 2>&1 | grep -E "piksel |clip "
done

echo "=== 7. skorlama ==="
./venv/bin/python score.py \
  --ref feat_benign_ref.npz --cal feat_benign_cal.npz \
  --clusters clusters_flat.json --fpr 0.05 \
  --queries A0=feat_A0.npz A0n=feat_A0n.npz A1=feat_A1.npz A3=feat_A3.npz \
            A4=feat_A4.npz S1=feat_S1.npz S3=feat_S3.npz S7=feat_S7.npz \
  --out results_pixel_clip.json 2>&1 | tail -12

echo "PIPELINE3 BITTI"
