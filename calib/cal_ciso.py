"""#11 c_iso calibration: 2-bird scripted rewards over 60 s (follower / leader / isolated solo).
Writes calib/out/ciso.json.   python -m calib.cal_ciso"""
import json
import os

import jax
import jax.numpy as jnp
import numpy as np

from animalbench.core import compile_schedule
from animalbench.scenarios.geese import GeeseConfig, make_rollout, stay_policy

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")


def rewards(c_iso, T=60.0):
    cfg = GeeseConfig(n_max=2, T=T, init_noise=0.0, c_iso=c_iso, use_events=False, log_D=False, log_low=False)
    f = jax.jit(make_rollout(cfg, stay_policy(cfg)))
    sched = compile_schedule([], cfg.e_max)
    _, r2 = f(jax.random.PRNGKey(0), jnp.int32(2), sched)
    _, r1 = f(jax.random.PRNGKey(0), jnp.int32(1), sched)
    R2 = np.asarray(r2["reward"]).mean(0)
    R1 = np.asarray(r1["reward"]).mean(0)
    return dict(follower=float(R2[1]), leader=float(R2[0]), isolated=float(R1[0]))


def evaluate(c_iso):
    r = rewards(c_iso)
    lead_minus_iso = r["leader"] - r["isolated"]
    lead_minus_fol = r["follower"] - r["leader"]
    r["order_ok"] = bool(r["follower"] > r["leader"] > r["isolated"])
    r["gap_ok"] = bool(lead_minus_iso >= 0.03 and lead_minus_iso <= 0.5 * lead_minus_fol)
    r["lead_minus_iso"] = float(lead_minus_iso)
    r["follow_minus_lead"] = float(lead_minus_fol)
    r["c_iso"] = c_iso
    return r


def run():
    return dict(default=evaluate(0.05), negative_control=evaluate(0.0))


if __name__ == "__main__":
    r = run()
    os.makedirs(OUT, exist_ok=True)
    json.dump(r, open(os.path.join(OUT, "ciso.json"), "w"), indent=1)
    print(json.dumps(r))
