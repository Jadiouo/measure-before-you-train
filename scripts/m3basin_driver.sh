#!/bin/bash
# Sequential driver: one training at a time, each under systemd-run MemoryMax=4G, wait if available RAM < 8 GB.
cd "$(dirname "$0")/.."
STEPS=${STEPS:-1e7}
mkdir -p runs/m3basin .scratch
run() {
  cell=$1; seed=$2
  d=runs/m3basin/${cell}_s${seed}
  [ -f $d/eval.json ] && return
  waited=0
  while [ $(free -g | awk '/Mem:/{print $7}') -lt 8 ]; do
    sleep 30; waited=$((waited+30))
    if [ $waited -gt 1800 ]; then echo "STOP: low memory for 30 min" >> .scratch/driver.log; exit 1; fi
  done
  echo "$(date +%T) start $cell s$seed avail=$(free -g | awk '/Mem:/{print $7}')G" >> .scratch/driver.log
  systemd-run --user --scope -q -p MemoryMax=4G -p MemorySwapMax=0 env PYTHONUNBUFFERED=1 \
    .venv/bin/python scripts/run_m3basin.py --cell $cell --seed $seed --steps $STEPS > .scratch/run_${cell}_s${seed}.log 2>&1
  echo "$(date +%T) done  $cell s$seed rc=$?" >> .scratch/driver.log
}
for seeds in "0 1 2" "3 4"; do
  for s in $seeds; do for c in A B C1 D1 C2 D2; do run $c $s; done; done
done
for s in 0 1 2 3 4; do run E $s; done
echo "ALL DONE $(date +%T)" >> .scratch/driver.log
