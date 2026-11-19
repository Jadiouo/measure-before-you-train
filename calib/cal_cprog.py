"""#10 c_prog calibration (continuous2d reward). Writes calib/out/cprog.json.   python -m calib.cal_cprog
closed form c_prog = v0 P'(v0)/P_solo; single-bird reward argmax over v in [0.7,1.3] v0 (step 0.005);
N=9 fixed V slowed by 10%: team reward vs v0 (simulated through the real step function)."""
import json
import os

import jax
import jax.numpy as jnp
import numpy as np

from animalbench.core import compile_schedule
from animalbench.scenarios.geese import GeeseConfig, wake
from animalbench.scenarios.geese.env import reset
from animalbench.scenarios.geese.step import step_continuous

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")


def _reward_at(cfg, N, v_factor):
    sched = compile_schedule([], cfg.e_max)
    st = reset(cfg, jax.random.PRNGKey(0), N, sched)
    st = st._replace(v=jnp.where(st.alive, cfg.v0 * v_factor, st.v))
    _, out = step_continuous(cfg, st, jnp.zeros((cfg.n_max, 2)))
    return np.asarray(out["r"])[:N]


def run():
    base = GeeseConfig(n_max=9, init_noise=0.0, use_events=False, log_D=False, log_low=False)
    c_closed = float(base.v0 * wake.dp_dv(base, base.v0) / base.P_solo)
    vs = np.round(np.arange(0.7, 1.3 + 1e-9, 0.005), 6)

    def single_argmax(c):
        P = wake.p_nowake(base, vs * base.v0)
        r = -P / base.P_solo + c * (vs - 1.0)
        return float(vs[np.argmax(r)])

    def team_gap(c):
        cfg = base.replace(c_prog=c)
        r1 = _reward_at(cfg, 9, 1.0).mean()
        r9 = _reward_at(cfg, 9, 0.9).mean()
        return float(r9 - r1)             # must be < 0 for "strictly lower than v0"

    res = dict(c_prog_closed_form=c_closed, single_argmax_closed_form=single_argmax(c_closed),
               team_gap_closed_form=team_gap(c_closed))
    res["single_ok"] = bool(0.99 - 1e-9 <= res["single_argmax_closed_form"] <= 1.01 + 1e-9)
    res["team_slowdown_lower_ok"] = bool(res["team_gap_closed_form"] < 0)
    scan = []
    for c in np.round(np.arange(c_closed - 0.03, c_closed + 0.06, 0.0025), 5):
        a, g = single_argmax(c), team_gap(c)
        scan.append(dict(c_prog=float(c), argmax=a, team_gap=g, both_ok=bool(0.99 - 1e-9 <= a <= 1.01 + 1e-9 and g < 0)))
    res["scan"] = scan
    ok = [s["c_prog"] for s in scan if s["both_ok"]]
    res["c_prog_both_ok_range"] = [min(ok), max(ok)] if ok else None
    return res


if __name__ == "__main__":
    r = run()
    os.makedirs(OUT, exist_ok=True)
    json.dump(r, open(os.path.join(OUT, "cprog.json"), "w"), indent=1)
    print({k: v for k, v in r.items() if k != "scan"})
