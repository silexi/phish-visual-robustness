#!/bin/zsh
cd "$(dirname "$0")"
export PATH=/opt/homebrew/bin:$PATH
for i in "$@"; do
  f="shard$i.parquet"
  [ -f "$f" ] || continue
  ./venv/bin/python render.py --parquet "$f" --outdir corpus --limit 700 --shard-tag "_s$i" 2>&1 \
    | grep -E "^render |^gorunur|^KULLAN" | sed "s/^/s$i /"
done
