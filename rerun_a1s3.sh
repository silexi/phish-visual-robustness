#!/bin/zsh
# A1 and S3 were described on a different 200-id subset than every other
# condition, because two id files held the same ids in different order and
# `head -200` therefore selected different pages. Re-run both on exactly the
# ids the clean baseline used, so every condition shares one population.
cd "$(dirname "$0")"
export PATH=/opt/homebrew/bin:$PATH
./venv/bin/python -c "
import json, os
ids=[os.path.basename(json.loads(l)['image'])[:-4] for l in open('vlm/desc_A0.jsonl')]
open('ids_vlm200.txt','w').write('\n'.join(ids))
print(len(ids),'id yazildi')
"
for c in A1 S3; do
  imgs=()
  while read -r id; do
    [ -f "pert/$c/$id.png" ] && imgs+=("pert/$c/$id.png")
  done < ids_vlm200.txt
  echo "=== $c: ${#imgs[@]} gorsel ==="
  ./venv/bin/python describe.py --out "vlm/desc_$c.jsonl" --images "${imgs[@]}" 2>&1 | tail -2
done
echo "A1+S3 YENIDEN BITTI"
