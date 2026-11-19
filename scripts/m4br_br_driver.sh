#!/bin/bash
# waits for the exp2 driver to finish (one CPU training at a time), then runs the BR trainings sequentially
cd "$(dirname "$0")/.."
while pgrep -f m4br_exp2_driver.sh > /dev/null; do sleep 20; done
mkdir -p runs/m4br/exp1/br .scratch
CFGS=${CFGS:-"self id4"}; SEEDS=${SEEDS:-"0 1 2 3 4"}
for cfg in $CFGS; do for seed in $SEEDS; do
  d=runs/m4br/exp1/br/${cfg}_s${seed}
  [ -f $d/eval.json ] && continue
  waited=0
  while [ $(free -g | awk '/Mem:/{print $7}') -lt 8 ]; do
    sleep 30; waited=$((waited+30))
    if [ $waited -gt 3600 ]; then echo "STOP low mem" >> .scratch/m4br_br_driver.log; exit 1; fi
  done
  echo "$(date +%T) start $cfg s$seed avail=$(free -g | awk '/Mem:/{print $7}')G" >> .scratch/m4br_br_driver.log
  systemd-run --user --scope -q -p MemoryMax=4G -p MemorySwapMax=0 env PYTHONUNBUFFERED=1 \
    .venv/bin/python scripts/run_m4br_br.py --cfg $cfg --seed $seed --steps 1e7 > .scratch/m4br_br_${cfg}_s${seed}.log 2>&1
  echo "$(date +%T) done $cfg s$seed rc=$?" >> .scratch/m4br_br_driver.log
done; done
