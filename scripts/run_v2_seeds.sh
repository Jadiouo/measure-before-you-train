#!/bin/bash
# sequential CPU runs per notes/impl/survival_v2_prereg.md (order A2,B2,C2,D2; seeds 21,22,23; 5e7 steps)
cd "$(dirname "$0")/.."
run() { # name alpha central seed
  n=$1; [ -f runs/survival_diag/$n/diag.json ] && return
  systemd-run --user --scope -q -p MemoryMax=12G -p MemorySwapMax=0 .venv/bin/python scripts/diag_survival_learnability.py train --v2 --name $n --steps 5e7 --alpha $2 --seed $4 $3 > runs/v2_$n.out 2>&1
}
for s in 21 22 23; do run v2_A_s$s 1 --central $s; done
for s in 21 22 23; do run v2_B_s$s 1 "" $s; done
for s in 21 22 23; do run v2_C_s$s 0 "" $s; done
for s in 21 22 23; do run v2_D_s$s 0 --central $s; done
echo ALLDONE > runs/v2_alldone
