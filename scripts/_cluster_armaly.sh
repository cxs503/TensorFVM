#!/usr/bin/env bash
# Shard launcher for the Armaly (1983) reproduction matrix on the cluster.
# Usage: bash scripts/_cluster_armaly.sh <i> <n>
# Extended matrix: coarse/medium/fine × 12S/40S × Re=100/389
set -u
i="${1:-1}"
out="${ARMALY_OUT:-runs/armaly1983}"
common="--pressure-iterations 300 --tolerance 1e-6 --threads 8 --out ${out}"
case "$i" in
  1) args="--re 100 --levels coarse --downstream 12 --max-iterations 6000" ;;
  2) args="--re 389 --levels coarse --downstream 12 --max-iterations 9000" ;;
  3) args="--re 100 --levels medium --downstream 12 --max-iterations 6000" ;;
  4) args="--re 389 --levels medium --downstream 12 --max-iterations 9000" ;;
  5) args="--re 100 --levels coarse --downstream 40 --max-iterations 6000" ;;
  6) args="--re 389 --levels coarse --downstream 40 --max-iterations 9000" ;;
  7) args="--re 389 --levels medium --downstream 40 --max-iterations 9000" ;;
  8) args="--re 100 --levels fine --downstream 12 --max-iterations 8000" ;;
  9) args="--re 389 --levels fine --downstream 12 --max-iterations 12000" ;;
  10) args="--re 100 --levels fine --downstream 40 --max-iterations 8000" ;;
  11) args="--re 389 --levels fine --downstream 40 --max-iterations 12000" ;;
  *) echo "unknown shard $i"; exit 2 ;;
esac
export PYTHONPATH="src${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p "$out"
echo "[shard $i/$2] python scripts/repro_armaly1983_bfs.py $args $common"
exec python scripts/repro_armaly1983_bfs.py $args $common
