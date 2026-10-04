#!/bin/zsh
# Anlamsal duzey, genisletilmis kume icin yeniden calistirma.
#
# Orneklem kurali yayimlanan turla ayni mantikta: olcum kumesinin id listesinin
# ILK 200 satiri, rastgele secim yok. Fark, listenin artik 1.206 sayfalik
# birlesik kumeden gelmesi (ids_u.txt, siralanmis).
#
# Model, istem, nicemleme ve kod cozme ayarlari yayimlanan turla birebir ayni:
# mlx-community/Qwen3-VL-8B-Instruct-8bit, sicaklik 0, 900 token, 1024 px.
# Kosul basina kontrol noktasi: var olan cikti atlanir, baglanti kopsa da
# kaldigi yerden devam eder.
setopt nullglob
cd ~/phish-vlm
export PATH=/opt/homebrew/bin:$PATH
N=${N:-200}
mkdir -p vlm_u
head -$N ids_u.txt > /tmp/vlm_ids_u.txt
echo "orneklem: $(wc -l < /tmp/vlm_ids_u.txt) sayfa"

run() {  # $1 = kosul etiketi, $2 = goruntu dizini
  out="vlm_u/desc_$1.jsonl"
  if [ -f "$out" ]; then
    echo "$1 zaten var, atlandi ($(wc -l < $out) satir)"
    return
  fi
  imgs=()
  while read -r id; do
    [ -f "$2/$id.png" ] && imgs+=("$2/$id.png")
  done < /tmp/vlm_ids_u.txt
  echo "=== $1: ${#imgs[@]} gorsel ==="
  if [ ${#imgs[@]} -eq 0 ]; then
    echo "$1: gorsel yok, atlandi"
    return
  fi
  ./venv/bin/python describe.py --out "$out" --images "${imgs[@]}" 2>&1 | tail -2
}

run A0  clean_sub_u
run A0n pert_u/A0n
run A1  pert_u/A1
run A3  pert_u/A3
run A4  pert_u/A4
run S1  pert_u/S1
run S3  pert_u/S3
run S7  pert_u/S7
echo "VLM_U BITTI"
wc -l vlm_u/*.jsonl
