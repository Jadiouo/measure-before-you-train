"""Volunteer's-dilemma check (mechanism evidence, inference): everybody plays scripted S2 rotation (mutual, N=3,
random_slots); bird 0 deviates by never volunteering (FWD -> STAY; still retreats when fatigued).  Paired seeds.
Reports bird 0's individual return change and the other birds' change, for several kappa.  -> runs/m3basin/dilemma.json"""
import json, os
os.environ.setdefault("JAX_PLATFORMS", "cpu")
import numpy as np, jax.numpy as jnp
import animalbench  # noqa
from animalbench.scenarios.geese import GeeseConfig, scripted as S
from animalbench.scenarios.geese.hlc import Policy, FWD, STAY
from animalbench.marl.basin import scripted_data

out = {}
for k in (0.0, 1.0, 8.0):
    cfg = GeeseConfig(n_max=3, dtype="float32", log_low=False, log_D=False, use_events=False, swap_rule="mutual",
                      kappa=k, random_slots=True)
    base = S.state_rotation(cfg, 0.94, 0.90)

    def act(ps, obs, st, mask, key=None, base=base):
        a, ps2 = base.act(ps, obs, st, mask, key)
        return a.at[0].set(jnp.where(a[0] == FWD, STAY, a[0])), ps2
    dev = Policy("S2_refuser0", base.init, act)
    r = lambda p: (lambda d: d["rew"].sum(0) * cfg.delta_h)(scripted_data(cfg, p, 128, 0))     # (B,3)
    r0, r1 = r(base), r(dev)
    d0 = r1[:, 0] - r0[:, 0]
    do = (r1[:, 1:].mean(1) - r0[:, 1:].mean(1))
    out[str(k)] = dict(refuser_gain=float(d0.mean()), refuser_gain_se=float(d0.std() / np.sqrt(len(d0))),
                       others_change=float(do.mean()), others_se=float(do.std() / np.sqrt(len(do))))
    print(k, out[str(k)], flush=True)
json.dump(out, open("runs/m3basin/dilemma.json", "w"), indent=1)
