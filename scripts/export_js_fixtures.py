"""Export JAX reference trajectories as JSON fixtures for the JS port (web/test/fixtures/*.json).

Run (memory-capped):  systemd-run --user --scope -q -p MemoryMax=2G -p MemorySwapMax=0 \
    .venv/bin/python scripts/export_js_fixtures.py [--only name1,name2]

Each fixture = initial state + event list + (open-loop actions | rule_blind T_hold draws) + lottery coins + the
per-decision reference trajectory. JS replays the same inputs; JAX-only randomness (T_hold draws, apex lottery) is exported
so the comparison covers the logic, not the PRNG (spec v0.2 sec 1.4). (Regenerated 2026-10 after d_max 12.5 b and path_from frame fixes, notes/impl/physics_fix_decisions.md.)
"""
import argparse
import json
import os
import time

import jax
import jax.numpy as jnp
import numpy as np

from animalbench.core import compile_schedule
from animalbench.core.perturbation import DEFAULTS, PARAM_SLOTS, crosswind, immune_on, injury, relabel, swap_pair_perm
from animalbench.scenarios.geese import GeeseConfig, make_rollout, rule_blind, stay_policy
from animalbench.scenarios.geese import wake
from animalbench.scenarios.geese.env import reset
from animalbench.scenarios.geese.hlc import table_policy

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "web", "test", "fixtures")
K_THOLD = 24


def ev_to_js(ev):
    prm = dict(DEFAULTS[ev.kind]); prm.update(ev.params)
    d = dict(kind=ev.kind, t_on=float(ev.t_on), t_off=None if ev.t_off is None else float(ev.t_off), target=ev.target)
    if ev.kind == "crosswind":
        d.update(speed=prm["speed"], dirRad=float(np.deg2rad(prm["dir_deg"])), ramp=prm["ramp"], ou_sigma=prm["ou_sigma"])
    elif ev.kind == "injury":
        d.update(power_mult=prm["power_mult"], vmax_mult=prm["vmax_mult"])
    elif ev.kind == "relabel":
        d.update(perm=None if ev.perm is None else list(ev.perm), reset_mem=bool(prm["reset_mem"]))
    elif ev.kind == "immune":
        d.update(on=bool(prm["on"] > 0.5))
    return d


def scenarios():
    N9, N14 = 9, 14
    return {
        "stay_N9_calm": dict(N=N9, T=180.0, agent="stay", rule="unilateral", events=[]),
        "table_N9_swaps": dict(N=N9, T=180.0, agent="table", rule="unilateral", events=[], f32=True),
        "table_N9_mutual": dict(N=N9, T=180.0, agent="table", rule="mutual", events=[]),
        "blind_N9_wind_injury": dict(N=N9, T=240.0, agent="rule_blind", rule="unilateral", f32=True,
            events=[crosswind(40.0, duration=80.0, speed=3.0, dir_deg=90.0, ramp=5.0, ou_sigma=0.0),
                    injury(130.0, target="leader")]),
        "blind_N14_mutual_events": dict(N=N14, T=300.0, agent="rule_blind", rule="mutual", f32=True,
            events=[crosswind(30.0, duration=100.0, speed=4.0, dir_deg=45.0, ramp=5.0, ou_sigma=0.0),
                    relabel(100.0, perm=swap_pair_perm(N14, 2, 5), reset_mem=True),
                    immune_on(120.0, "last"),
                    immune_on(160.0, 3), immune_on(220.0, 3, on=False)]),
        "blind_N14_strongwind": dict(N=N14, T=240.0, agent="rule_blind", rule="unilateral",
            events=[crosswind(20.0, duration=150.0, speed=5.0, dir_deg=270.0, ramp=5.0, ou_sigma=0.0)]),
        "blind_N3": dict(N=3, T=180.0, agent="rule_blind", rule="unilateral", events=[]),
        "blind_N5_mutual": dict(N=5, T=180.0, agent="rule_blind", rule="mutual",
            events=[relabel(60.0, perm=[1, 2, 3, 4, 0]), immune_on(90.0, "leader")]),
    }


def make_table(N, n_dec, seed=7):
    rng = np.random.default_rng(seed)
    return rng.choice(3, size=(n_dec, N), p=[0.5, 0.25, 0.25]).astype(np.int32)


def run(name, sc, dtype, init=None, actions=None):
    N = sc["N"]
    cfg = GeeseConfig(n_max=N, T=sc["T"], swap_rule=sc["rule"], dtype=dtype, log_D=False, log_low=False,
                      init_mode="v")
    n_dec = cfg.n_dec
    sched = compile_schedule(list(sc["events"]), cfg.e_max)
    if sc["agent"] == "stay":
        pol = stay_policy(cfg)
    elif sc["agent"] == "table":
        tab = make_table(N, n_dec) if actions is None else actions
        pol = table_policy(cfg, tab)
    else:
        pol = rule_blind(cfg) if actions is None else table_policy(cfg, actions)
    f = jax.jit(make_rollout(cfg, pol))
    key = jax.random.PRNGKey(3)
    io = None
    if init is not None:
        io = dict(mask=jnp.ones((N,), bool), **{k: jnp.asarray(v) for k, v in init.items()})
    st0 = reset(cfg, key, jnp.int32(N), sched, io)
    t0 = time.time()
    st, recs = f(key, jnp.int32(N), sched, io)
    jax.block_until_ready(recs["P_mean"])
    print(f"  {name} {dtype}: {time.time() - t0:.1f}s", flush=True)
    r = {k: np.asarray(v) for k, v in recs.items()}
    return cfg, st0, r


def to_list(a):
    return np.asarray(a).tolist()


def export(name, sc):
    print(name, flush=True)
    N = sc["N"]
    cfg, st0, r64 = run(name, sc, "float64")
    n_dec = cfg.n_dec
    init = dict(x=r64["x"][0], y=r64["y"][0], psi=r64["psi"][0], v=r64["v"][0], slot=r64["slot"][0])
    actions = r64["action_req"]
    # rule_blind T_hold draws and apex lottery coins (the only JAX-PRNG inputs of the logic)
    lo, hi = cfg.t_hold_lo, cfg.t_hold_hi
    kp = st0.key_policy
    thold = [[float(lo + (hi - lo) * jax.random.uniform(jax.random.fold_in(jax.random.fold_in(kp, i), c), (), dtype=cfg.jdtype))
              for c in range(K_THOLD)] for i in range(N)]
    coins = [bool(jax.random.bernoulli(jax.random.fold_in(st0.key_slot, d), 0.5)) for d in range(n_dec)]
    ref_keys = ["x", "y", "psi", "v", "slot", "lock", "leader", "P_mean", "save_mean", "eff_action", "swapped", "blocked",
                "e_form_end", "E_used", "id_owner", "immune", "action_req", "n_swaps"]

    def pack(r):
        return {k: to_list(r[k]) for k in ref_keys}
    fx = dict(name=name, N=N, T=sc["T"], n_dec=n_dec, swap_rule=sc["rule"], agent=sc["agent"], dtype="float64",
              gain=cfg.gain, P_solo=cfg.P_solo, events=[ev_to_js(e) for e in sc["events"]],
              init={k: to_list(v) for k, v in init.items()}, actions=to_list(actions), thold=thold, coins=coins,
              ref=pack(r64))
    json.dump(fx, open(os.path.join(OUT, f"{name}.json"), "w"))
    if sc.get("f32"):
        cfg32, _, r32 = run(name, sc, "float32", init=init, actions=actions)
        fx32 = dict(fx, dtype="float32", ref=pack(r32))
        json.dump(fx32, open(os.path.join(OUT, f"{name}__f32.json"), "w"))


def export_wake():
    cfg = GeeseConfig(n_max=4, T=10.0, log_D=False, log_low=False)
    dx = np.linspace(0.3, 13.1, 41)   # spans past d_max=12.5 (window edge is not a grid point)
    dy = np.linspace(-6, 6, 49)
    vs = [10.0, 14.0, 18.0]
    DX, DY = np.meshgrid(dx, dy, indexing="ij")
    out = dict(gain=cfg.gain, P_solo=cfg.P_solo, dx=dx.tolist(), dy=dy.tolist(), v=vs, w={})
    for v in vs:
        out["w"][str(v)] = np.asarray(wake.wbar(jnp, jnp.asarray(DX), jnp.asarray(DY), jnp.asarray(v), cfg)).tolist()
    cfg32 = GeeseConfig(n_max=4, T=10.0, dtype="float32", log_D=False, log_low=False)
    out["w32"] = {str(v): np.asarray(wake.wbar(jnp, jnp.asarray(DX, jnp.float32), jnp.asarray(DY, jnp.float32),
                                              jnp.asarray(v, jnp.float32), cfg32)).tolist() for v in vs}
    json.dump(out, open(os.path.join(OUT, "wake_grid.json"), "w"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    sel = [s for s in a.only.split(",") if s]
    if not sel or "wake_grid" in sel:
        export_wake()
    for name, sc in scenarios().items():
        if not sel or name in sel:
            export(name, sc)
