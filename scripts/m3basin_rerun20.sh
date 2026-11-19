#!/bin/bash
# After the main sweep: rerun non-fixed / unstable runs at 20M steps (same seed, own lr schedule) into runs/m3basin_20M.
cd "$(dirname "$0")/.."
while ! grep -q "ALL DONE" .scratch/driver.log; do sleep 20; done
for cs in "B 0" "B 4" "D2 4" "D1 0" "D1 4"; do
  set -- $cs
  [ -f runs/m3basin_20M/$1_s$2/eval.json ] && continue
  while [ $(free -g | awk '/Mem:/{print $7}') -lt 8 ]; do sleep 30; done
  echo "$(date +%T) start20 $1 s$2" >> .scratch/driver.log
  systemd-run --user --scope -q -p MemoryMax=4G -p MemorySwapMax=0 env PYTHONUNBUFFERED=1 \
    .venv/bin/python scripts/run_m3basin.py --cell $1 --seed $2 --steps 2e7 --root runs/m3basin_20M > .scratch/run20_$1_s$2.log 2>&1
  echo "$(date +%T) done20 $1 s$2 rc=$?" >> .scratch/driver.log
done
echo "RERUN DONE $(date +%T)" >> .scratch/driver.log
