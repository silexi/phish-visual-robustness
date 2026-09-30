#!/bin/zsh
cd "$(dirname "$0")"
export PATH=/opt/homebrew/bin:$PATH
while [ "$(pgrep -f describe.py | wc -l)" -gt 0 ]; do sleep 60; done
# original run: the first 120 replayed benign renders in filename order
imgs=($(ls benign_replay/shots/*.png | sort | head -120))
echo "=== BENIGN: ${#imgs[@]} gorsel ==="
./venv/bin/python describe.py --out vlm/desc_BENIGN.jsonl --images "${imgs[@]}" 2>&1 | tail -2
echo "BENIGN VLM BITTI"
