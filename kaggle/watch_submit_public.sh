#!/usr/bin/env bash
# Poll pushed kernels; on COMPLETE submit version 1 once. Usage: watch_submit_public.sh slug...
COMP=biohub-cell-tracking-during-development
declare -A done
while :; do
  left=0
  for s in "$@"; do
    [[ -n "${done[$s]}" ]] && continue
    left=1
    st=$(uvx kaggle kernels status "dimiturnt/$s" 2>&1)
    echo "$(date -u +%H:%MZ) $s :: $st"
    if grep -qi complete <<<"$st"; then
      out=$(uvx kaggle competitions submit $COMP -k "dimiturnt/$s" -v 1 -f submission.csv -m "public fork $s v1 (credited upstream)" 2>&1)
      echo "SUBMIT $s :: $out"; done[$s]=1
    elif grep -qiE "error|cancel" <<<"$st"; then
      uvx kaggle kernels output "dimiturnt/$s" -p "logs/kaggle/$s" >/dev/null 2>&1; echo "FAILED $s"; done[$s]=1
    fi
  done
  [[ $left == 0 ]] && break
  sleep 600
done
