#!/bin/zsh
cd "$(dirname "$0")"
export PATH=/opt/homebrew/bin:$PATH
while [ "$(pgrep -f describe.py | wc -l)" -gt 0 ]; do sleep 60; done
imgs=()
while read -r id; do
  [ -f "pert/S3/$id.png" ] && imgs+=("pert/S3/$id.png")
done < <(head -200 ids_f.txt)
echo "=== S3: ${#imgs[@]} gorsel ==="
./venv/bin/python describe.py --out vlm/desc_S3.jsonl --images "${imgs[@]}" 2>&1 | tail -2
echo "S3 BITTI"
