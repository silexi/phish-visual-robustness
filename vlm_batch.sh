#!/bin/zsh
# Semantic rung: one structured description per image, per condition.
# This is the slow rung (~11 s/image), so it is checkpointed per condition and
# skips any condition whose output already exists.
cd "$(dirname "$0")"
export PATH=/opt/homebrew/bin:$PATH
N=${N:-250}
mkdir -p vlm
head -$N ids_fid_full.txt > /tmp/vlm_ids.txt

run() {  # $1 = condition label, $2 = image directory
  out="vlm/desc_$1.jsonl"
  if [ -f "$out" ]; then
    echo "$1 zaten var, atlandi"
    return
  fi
  imgs=()
  while read -r id; do
    [ -f "$2/$id.png" ] && imgs+=("$2/$id.png")
  done < /tmp/vlm_ids.txt
  echo "=== $1: ${#imgs[@]} gorsel ==="
  if [ ${#imgs[@]} -eq 0 ]; then
    echo "$1: gorsel yok, atlandi"
    return
  fi
  ./venv/bin/python describe.py --out "$out" --images "${imgs[@]}" 2>&1 | tail -2
}

run A0  clean_f/shots
run A3  pert/A3
run S1  pert/S1
run A4  pert/A4
run S7  pert/S7
run A0n pert/A0n
run A1  pert/A1
echo "VLM BITTI"
