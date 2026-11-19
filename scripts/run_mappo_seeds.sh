#!/bin/bash
# sequential CPU runs per notes/impl/survival_mappo_prereg.md (order A,D,C,B)
cd "$(dirname "$0")/.."
run() { # name steps alpha central seed
  n=$1; [ -f runs/survival_diag/$n/diag.json ] && return
  systemd-run --user --scope -q -p MemoryMax=12G -p MemorySwapMax=0 .venv/bin/python scripts/diag_survival_learnability.py train --name $n --steps $2 --alpha $3 --seed $5 $4 > runs/mappo_$n.out 2>&1
}
for s in 1 2 3; do run mp_A_s$s 1e8 1 --central $s; done
for s in 1 2 3; do run mp_D_s$s 5e7 1 "" $s; done
for s in 1 2 3; do run mp_C_s$s 5e7 0.5 --central $s; done
for s in 1 2 3; do run mp_B_s$s 5e7 0.75 --central $s; done
echo ALLDONE > runs/mappo_alldone
