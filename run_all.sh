#!/bin/zsh
cd "$(dirname "$0")"
export PATH=/opt/homebrew/bin:$PATH
for f in shard*.parquet; do
  i="${f:r}"; i="${i#shard}"
  [ -f "corpus/manifest_s$i.jsonl" ] && continue
  echo "=== shard$i ==="
  ./venv/bin/python render.py --parquet "$f" --outdir corpus --limit 1200 --shard-tag "_s$i" 2>&1 \
     | grep -vE "^127.0.0.1|GET /|code 404|Traceback|File \"|  " 
done
echo "TUM RENDER BITTI"
