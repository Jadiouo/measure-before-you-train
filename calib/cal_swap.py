"""#8 swap cost: steady N=9 no wind, single swap. t_swap (both swappers e_form<0.15b and stay), min separation,
others' max e_form, c_swap (group extra energy over 40 s vs paired no-swap run, in P_solo*s).
Writes calib/out/swap_cost.json.   python -m calib.cal_swap"""
import json
import os

import jax
import jax.numpy as jnp
import numpy as np

from animalbench.core import compile_schedule
from animalbench.scenarios.geese import GeeseConfig, make_rollout, table_policy

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
PAIRS = [(0, 1), (0, 2), (1, 3), (2, 4), (3, 5), (5, 7), (6, 8)]   # (front slot, back slot)
T0 = 15          # decision at which the swap is requested (30 s settle)
H = 20           # decisions of horizon for c_swap (40 s)


def run(n_seeds=100):
    cfg = GeeseConfig(n_max=9, T=100.0, init_mode="v", log_D=False, log_stride=1, use_events=False)
    nd = cfg.n_dec
    sched = compile_schedule([], cfg.e_max)

    def one(k, table):
        return make_rollout(cfg, table_policy(cfg, table), nd)(k, jnp.int32(9), sched)
    g = jax.jit(jax.vmap(one, in_axes=(0, None)))
    keys = jax.random.split(jax.random.PRNGKey(1), n_seeds)
    rows = {}
    for f, bk in PAIRS:
        tab = np.zeros((nd, 9), np.int32)
        if f == 0 and bk == 1:
            tab[T0, 0] = 1                  # apex retreats (ctr=0 -> slot 1)
        elif f == 0:
            tab[T0, bk] = 2                 # slot 2 advances
        else:
            tab[T0, f] = 1
        _, r1 = g(keys, jnp.asarray(tab))
        _, r0 = g(keys, jnp.zeros((nd, 9), jnp.int32))
        low = r1["low"]
        B = n_seeds
        t = np.asarray(low["t"])[0].reshape(-1)
        ef = np.asarray(low["e_form"]).reshape(B, -1, 9) / cfg.b
        x = np.asarray(low["x"]).reshape(B, -1, 9)
        y = np.asarray(low["y"]).reshape(B, -1, 9)
        ts = T0 * cfg.delta_h
        sel = t >= ts
        tt = t[sel] - ts
        pair_err = np.maximum(ef[:, sel, f], ef[:, sel, bk])
        tsw = []
        for b in range(B):
            bad = np.where(pair_err[b] >= 0.15)[0]
            tsw.append(float(tt[bad.max()] + cfg.dt) if len(bad) else 0.0)
        others = [i for i in range(9) if i not in (f, bk)]
        oth = float(ef[:, sel][:, :, others].max())
        dxm = x[:, sel, :, None] - x[:, sel, None, :]
        dym = y[:, sel, :, None] - y[:, sel, None, :]
        dist = np.sqrt(dxm ** 2 + dym ** 2) + np.eye(9) * 1e9
        msep = float(dist.min() / cfg.b)
        E1 = np.asarray(r1["E_used"]); E0 = np.asarray(r0["E_used"])
        c = ((E1[:, T0 + H - 1].sum(1) - E1[:, T0 - 1].sum(1)) - (E0[:, T0 + H - 1].sum(1) - E0[:, T0 - 1].sum(1))) / cfg.P_solo
        rows[f"{f}-{bk}"] = dict(t_swap_max=float(max(tsw)), t_swap_min=float(min(tsw)), min_sep_b=msep,
                                 others_e_form_max_b=oth, c_swap_mean=float(c.mean()), c_swap_min=float(c.min()),
                                 c_swap_max=float(c.max()))
    res = dict(pairs=rows, n_seeds=n_seeds, c_swap_apex=rows["0-1"]["c_swap_mean"],
               c_swap_all_mean=float(np.mean([r["c_swap_mean"] for r in rows.values()])),
               t_swap_apex=rows["0-1"]["t_swap_max"],
               note="seeds differ only in initial noise which decays before the swap; spread across seeds ~1e-7")
    return res


if __name__ == "__main__":
    r = run()
    os.makedirs(OUT, exist_ok=True)
    json.dump(r, open(os.path.join(OUT, "swap_cost.json"), "w"), indent=1)
    print(json.dumps({k: v for k, v in r.items() if k != "pairs"}))
