#!/bin/zsh
# A ve B adimlari yeniden: retention_excl.py ve paired_contrasts.py --conds'u
# duz ad olarak aliyor ve dosya adini "feat_<ad>.npz" biciminde CALISMA DIZININE
# gore turetiyor. Bu yuzden dogru adlarla sembolik bag iceren bir calisma dizini
# kurulup betikler oradan calistiriliyor. Veri kopyalanmiyor.
setopt nullglob
set -e
cd ~/phish-vlm
export PATH=/opt/homebrew/bin:$PATH
PY=$PWD/venv/bin/python
CONDS=(A0n A1 A3 A4 S1 S3 S7)

rm -rf u_run && mkdir -p u_run
ln -sf "$PWD/feat_u_A0.npz" u_run/feat_A0.npz
for c in $CONDS; do ln -sf "$PWD/feat_u_$c.npz" "u_run/feat_$c.npz"; done
ln -sf "$PWD/clusters_u_flat.json" u_run/clusters_flat.json
ln -sf "$PWD/pert_u" u_run/pert
echo "u_run icerigi: $(ls u_run | tr '\n' ' ')"

cd u_run

echo "=== A. dislama galerileri (Tablo 5) ==="
$PY ../retention_excl.py --clean feat_A0.npz --clusters clusters_flat.json \
    --applied pert --out ../results_retention_excl_u.json 2>&1 | tail -40

echo "=== B. eslestirilmis karsilastirmalar ==="
$PY ../paired_contrasts.py --clean feat_A0.npz --clusters clusters_flat.json \
    --applied pert --out ../results_paired_contrasts_u.json 2>&1 | tail -40

echo "PIPELINE657D BITTI"
