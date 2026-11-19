#!/bin/bash
# extras after everything else: (1) BR vs grim identity reciprocators (3 seeds) (2) team reward at kappa 8 (3 seeds)
cd "$(dirname "$0")/.."
while pgrep -f "m4br_exp2_driver.sh|m4br_br_driver.sh" > /dev/null; do sleep 20; done
CFGS="id1000" SEEDS="0 1 2" scripts/m4br_br_driver.sh
for seed in 0 1 2; do
  d=runs/m4br/exp2x/team_k8x/team_s${seed}
  [ -f $d/eval.json ] && continue
  while [ $(free -g | awk '/Mem:/{print $7}') -lt 8 ]; do sleep 30; done
  echo "$(date +%T) start team_k8 s$seed" >> .scratch/m4br_extra.log
  systemd-run --user --scope -q -p MemoryMax=4G -p MemorySwapMax=0 env PYTHONUNBUFFERED=1 \
    .venv/bin/python scripts/run_m4br_exp2.py --cell team --kappa 8 --seed $seed --root runs/m4br/exp2x/team_k8x > .scratch/m4br_teamk8_s${seed}.log 2>&1
  echo "$(date +%T) done team_k8 s$seed rc=$?" >> .scratch/m4br_extra.log
done
