#!/bin/bash
# sequential CPU runs per notes/impl/survival_ablation_prereg.md (condition E)
cd "$(dirname "$0")/.."
for s in 1 2 3; do n=ab_E_s$s; [ -f runs/survival_diag/$n/diag.json ] && continue
  systemd-run --user --scope -q -p MemoryMax=12G -p MemorySwapMax=0 .venv/bin/python scripts/diag_survival_learnability.py train --name $n --steps 5e7 --alpha 1 --seed $s --separate_critic > runs/ablation_$n.out 2>&1
done
echo ALLDONE > runs/ablation_alldone
