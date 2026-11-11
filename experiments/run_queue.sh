#!/bin/bash
# Run experiments one after another from a queue file, for long unattended sweeps on one GPU.
# Each line: "name|arguments for experiments/tokenizer_lm.py" (JSON in single quotes is fine), or
# "name|experiments/<script>.py arguments" to run another experiment script as is (it must set its own --out).
# Lines are taken from the top, so lines can be added or reordered while it runs; finished lines go to queue.done.
#   bash experiments/run_queue.sh results/sweep/queue.txt
Q=${1:-results/sweep/queue.txt}
OUT=$(dirname "$Q")
PY=${PY:-python3}
touch "$Q"
while true; do
  line=$(head -n1 "$Q")
  if [ -z "$line" ]; then sleep 30; continue; fi
  tail -n +2 "$Q" > "$Q.tmp" && mv "$Q.tmp" "$Q"
  name=${line%%|*}
  args=${line#*|}
  echo "$(date '+%F %T') START $name"
  if [[ "$args" == experiments/* ]]; then
    eval "$PY $args" > "$OUT/$name.log" 2>&1
  else
    eval "$PY experiments/tokenizer_lm.py $args --tag $name --out $OUT" > "$OUT/$name.log" 2>&1
  fi
  echo "$line" >> "$OUT/queue.done"
  echo "$(date '+%F %T') END $name $(tail -n1 "$OUT/$name.log" | cut -c1-200)"
done
