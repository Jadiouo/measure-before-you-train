#!/bin/bash
# sequential, one training at a time, 4G cap, waits for available RAM >= 8G
cd "$(dirname "$0")/.."
mkdir -p runs/m4br/exp2 .scratch
for seed in 0 1 2; do for cell in ent05 ent10lr team; do
  d=runs/m4br/exp2/${cell}_s${seed}
  [ -f $d/eval.json ] && continue
  waited=0
  while [ $(free -g | awk '/Mem:/{print $7}') -lt 8 ]; do
    sleep 30; waited=$((waited+30))
    if [ $waited -gt 3600 ]; then echo "STOP low mem" >> .scratch/m4br_driver.log; exit 1; fi
  done
  echo "$(date +%T) start $cell s$seed avail=$(free -g | awk '/Mem:/{print $7}')G" >> .scratch/m4br_driver.log
  systemd-run --user --scope -q -p MemoryMax=4G -p MemorySwapMax=0 env PYTHONUNBUFFERED=1 \
    .venv/bin/python scripts/run_m4br_exp2.py --cell $cell --seed $seed > .scratch/m4br_${cell}_s${seed}.log 2>&1
  echo "$(date +%T) done $cell s$seed rc=$?" >> .scratch/m4br_driver.log
done; done
