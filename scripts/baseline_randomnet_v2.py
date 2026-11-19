"""Random-initialised actor reward in the v2 env (4 init keys, same 512 evaluation episodes as baselines_survival_v2.py).
Output calib/out/survival_v2_randomnet.json.   systemd-run --user --scope -q -p MemoryMax=12G -p MemorySwapMax=0 .venv/bin/python scripts/baseline_randomnet_v2.py"""
import dataclasses
import json
import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
import jax
import numpy as np

import animalbench  # noqa
from animalbench.marl import MarlConfig
from animalbench.marl.ippo import init_train, make_rollout

NEP = 512
out = {}
for central in (False, True):
    mc = MarlConfig(survival=True, surv_alpha=1.0, surv_eps=1.02, surv_burst_m=1.0, surv_h=0.05, surv_burst_p=0.05, swap_rule="mutual",
                    random_slots=True, surv_obs_abs=True, surv_pw_spread=0.05, path_T=2.0, central=central)
    ecfg = mc.env_cfg()
    rs = []
    for s in (12345, 1, 2, 3):
        ts0 = init_train(jax.random.PRNGKey(s), ecfg, mc)
        mm = dataclasses.replace(mc, n_envs=NEP)
        data, _, _ = jax.jit(make_rollout(ecfg, mm))(ts0.params, jax.random.PRNGKey(777))
        rew = np.asarray(data["rew"]) if "rew" in data else np.asarray(data["reward"])
        ep = rew.sum(0).mean(-1)
        rs.append(ep)
    m = np.mean(rs, 0)
    out["MAPPO_arch" if central else "IPPO_arch"] = dict(per_key_mean=[float(r.mean()) for r in rs], mean=float(m.mean()),
                                                       ci95_episode=float(1.96 * m.std(ddof=1) / np.sqrt(NEP)))
    print(out, flush=True)
json.dump(out, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "calib", "out", "survival_v2_randomnet.json"), "w"), indent=1)
