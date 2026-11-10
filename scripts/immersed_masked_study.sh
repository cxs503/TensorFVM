#!/usr/bin/env bash
# Launch the immersed-solid projection grid study: one process per (case, mesh, arm).
# Each process writes runs/immersed/<tag>.json (ledger + metrics) and <tag>_fields.pt.
set -u
cd "$(dirname "$0")/.."
mkdir -p runs/immersed logs
PY=${PYTHON:-python}
export PYTHONPATH=src
launch () {  # name cells arm t_end re threads
  local name=$1 cph=$2 arm=$3 tend=$4 re=$5 th=$6
  setsid nohup "$PY" -m tensorfvm.immersed.study --case "$name" --cells-per-h "$cph" \
      --arm "$arm" --t-end "$tend" --reynolds-dh "$re" --threads "$th" \
      --out-dir runs/immersed --json-out "runs/immersed/${name}_H${cph}_${arm}_Re${re}_t${tend}.json" \
      > "logs/${name}_H${cph}_${arm}_t${tend}.log" 2>&1 < /dev/null &
  echo "launched $name H$cph $arm t=$tend pid $!"
}
launch bfs 40 masked 200 800 8
launch bfs 40 legacy 200 800 8
launch bfs 40 masked 40  800 8
launch bfs 40 legacy 40  800 8
launch bfs 80 masked 30  800 8
launch bfs 80 legacy 30  800 8
launch expansion 40 masked 100 100 8
launch expansion 40 legacy 100 100 8
launch expansion 80 masked 40  100 8
