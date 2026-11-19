"""reset / rollout (scan over decisions) / trajectory assembly."""
from __future__ import annotations

import functools
import hashlib
import json

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax

from ...core.perturbation import BIG, PerturbSchedule, compile_schedule
from ...core.trajectory import Trajectory
from .config import GeeseConfig
from .dynamics import wind_global
from .events import apply_events
from .hlc import Policy
from .obs import build_obs
from .state import State, occ_from_slot, slot_offset
from .step import decision_step
from .swap import action_mask


def reset(cfg: GeeseConfig, key, N, sched: PerturbSchedule, init_override=None) -> State:
    n, E = cfg.n_max, cfg.e_max
    dt = cfg.jdtype
    k_init, k_wind, k_noise, k_pert, k_pol, k_slot = jax.random.split(key, 6)
    idx = jnp.arange(n)

    def per_bird(k, shape, kind="uniform"):
        f = jax.random.uniform if kind == "uniform" else jax.random.normal
        return jax.vmap(lambda i: f(jax.random.fold_in(k, i), shape, dtype=dt))(idx)

    u = per_bird(jax.random.fold_in(k_init, 0), (3,))
    nz = per_bird(jax.random.fold_in(k_init, 1), (2,), "normal")
    ez = per_bird(jax.random.fold_in(k_init, 2), (), "normal")
    idc = per_bird(jax.random.fold_in(k_init, 3), (3,), "normal")
    alive = idx < N
    b = cfg.b
    if cfg.init_mode == "v":
        if cfg.random_slots:               # random bird->slot permutation over the N alive birds (dead keys sort last)
            rkey = jnp.where(alive, jax.random.uniform(jax.random.fold_in(k_init, 4), (n,), dtype=dt), 2.0 + idx)
            slot = jnp.where(alive, jnp.argsort(jnp.argsort(rkey, stable=True), stable=True), -1).astype(jnp.int32)
        else:
            slot = jnp.where(alive, idx, -1).astype(jnp.int32)
        off = slot_offset(cfg, slot)
        x = off[:, 0] + cfg.init_noise * b * nz[:, 0]
        y = off[:, 1] + cfg.init_noise * b * nz[:, 1]
        psi = jnp.zeros((n,), dt)
    else:
        offs = slot_offset(cfg, idx.astype(jnp.int32))
        cen = jnp.sum(jnp.where(alive[:, None], offs, 0.0), 0) / jnp.maximum(jnp.sum(alive), 1)
        x = cen[0] + (2 * u[:, 0] - 1) * cfg.init_box * b
        y = cen[1] + (2 * u[:, 1] - 1) * cfg.init_box * b
        psi = (2 * u[:, 2] - 1) * jnp.deg2rad(cfg.init_psi_deg)
        order = jnp.argsort(jnp.where(alive, -x, BIG), stable=True)
        rank = jnp.argsort(order, stable=True).astype(jnp.int32)
        slot = jnp.where(alive, rank, -1).astype(jnp.int32)
    v = jnp.full((n,), cfg.v0, dt)
    if cfg.E0_coef is not None:
        E0 = cfg.E0_coef * cfg.P_solo * cfg.T * (1 + cfg.E0_sigma * ez)
    else:
        E0 = jnp.full((n,), BIG, dt)
    if cfg.survival and cfg.surv_h != 0.0:                 # the strong bird (bird 0) carries a larger energy cap
        E0 = E0.at[0].multiply(1.0 + cfg.surv_h)
    pmult = jnp.ones((n,), dt)
    if cfg.survival and cfg.surv_pw_spread != 0.0:         # individual energy-use coefficient (own PRNG branch: other streams untouched)
        kpw = jax.random.PRNGKey(20261031) if cfg.surv_pw_fixed else jax.random.fold_in(k_init, 5)
        pmult = (1.0 + cfg.surv_pw_spread * (2.0 * jax.random.uniform(kpw, (n,), dtype=dt) - 1.0)).astype(dt)
    if init_override:                      # {"mask": (n,) bool, "x"/"y"/"psi"/"v"/"slot"/"alive": (n,)} applied where mask
        o = init_override
        m = jnp.asarray(o["mask"], bool)
        x = jnp.where(m, jnp.asarray(o.get("x", x), dt), x)
        y = jnp.where(m, jnp.asarray(o.get("y", y), dt), y)
        psi = jnp.where(m, jnp.asarray(o.get("psi", psi), dt), psi)
        v = jnp.where(m, jnp.asarray(o.get("v", v), dt), v)
        alive = jnp.where(m, jnp.asarray(o.get("alive", alive), bool), alive)
        slot = jnp.where(m, jnp.asarray(o.get("slot", slot), jnp.int32), slot)
    # event randomness resolved once from the 'pert' stream (agent independent)
    ue = jax.random.uniform(jax.random.fold_in(k_pert, 0), (E,), dtype=dt)
    ud = jax.random.uniform(jax.random.fold_in(k_pert, 1), (E,), dtype=dt)
    p1 = jnp.asarray(sched.p[:, 1], dt)
    evt_dir = jnp.where(jnp.isnan(p1), 2 * jnp.pi * ud, jnp.deg2rad(jnp.where(jnp.isnan(p1), 0.0, p1)))
    s = PerturbSchedule(jnp.asarray(sched.kind_id, jnp.int32), jnp.asarray(sched.t_on, dt),
                        jnp.asarray(sched.t_off, dt), jnp.asarray(sched.target, jnp.int32),
                        jnp.asarray(sched.p, dt), jnp.asarray(sched.perm, jnp.int32))
    zf = jnp.zeros((n,), dt)
    st0 = State(
        t=jnp.zeros((), dt), step=jnp.zeros((), jnp.int32), dec=jnp.zeros((), jnp.int32),
        R=jnp.zeros((2,), dt), x=x, y=y, psi=psi, v=v, alive=alive, slot=slot,
        power_mult=pmult, vmax_mult=jnp.ones((n,), dt),
        E_used=zf, E_rem=E0, E0=E0, phi=jnp.ones((n,), dt),
        lock=jnp.zeros((n,), jnp.int32), blocked=jnp.zeros((n,), bool), last_action=jnp.zeros((n,), jnp.int32),
        P_prev=jnp.where(alive, 1.0, 0.0).astype(dt), a_prev=jnp.zeros((n, 2), dt),
        path_t0=jnp.full((n,), -BIG, dt), path_from=jnp.zeros((n, 2), dt), path_lane=zf,
        apex_ctr=jnp.zeros((), jnp.int32), immune=jnp.zeros((n,), bool),
        died=jnp.zeros((n,), bool), id_owner=jnp.arange(n, dtype=jnp.int32), relabel_n=jnp.zeros((), jnp.int32),
        mem_reset=jnp.zeros((), bool),
        ou=jnp.zeros((2,), dt), wind_f=jnp.zeros((2,), dt), evt_done=jnp.zeros((E,), bool), evt_dir=evt_dir.astype(dt), evt_u=ue,
        id_code=idc,
        cmd_valid=jnp.zeros((), bool), cmd_leader=jnp.full((), -1, jnp.int32),
        cmd_rank=jnp.full((n,), -1, jnp.int32), cmd_t=jnp.zeros((), dt), cmd_exec_left=jnp.zeros((), jnp.int32),
        sched=s, key_wind=k_wind, key_noise=k_noise, key_slot=k_slot, key_policy=k_pol)
    return st0._replace(wind_f=wind_global(cfg, st0).astype(dt))


def make_rollout(cfg: GeeseConfig, policy: Policy, n_dec: int | None = None, record_obs: bool = False):
    """Returns f(key, N, sched, init_override=None, immune=None) -> (final_state, recs); pure, jit/vmap-able.
    `f.trace_count` counts python-level traces (for the fixed-shape tests)."""
    n_dec = cfg.n_dec if n_dec is None else n_dec
    n = cfg.n_max

    def rollout(key, N, sched, init_override=None, immune=None):
        rollout.trace_count[0] += 1
        st = reset(cfg, key, N, sched, init_override)
        if immune is not None:
            st = st._replace(immune=jnp.asarray(immune, bool))
        ps = policy.init(st.key_policy, cfg)

        def body(carry, _):
            st, ps = carry
            obs = build_obs(cfg, st)
            mask = action_mask(cfg, st)
            act, ps = policy.act(ps, obs, st, mask, jax.random.fold_in(st.key_policy, st.dec))
            occ = occ_from_slot(cfg, st.slot, st.alive)
            pre = dict(x=st.x, y=st.y, psi=st.psi, v=st.v, alive=st.alive, slot=st.slot, lock=st.lock,
                       leader=occ[0], n_alive=jnp.sum(st.alive.astype(jnp.int32)), t=st.t,
                       action_req=act, P_prev=st.P_prev, E_used_start=st.E_used)
            if record_obs:
                pre["obs"] = obs
            st, rec = decision_step(cfg, st, act)
            rec.update(pre)
            rec["E_used"] = st.E_used
            rec["phi"] = st.phi
            rec["wind_ou"] = st.ou
            rec["id_owner"] = st.id_owner          # relabel log: who owns the code each bird carries
            rec["relabel_n"] = st.relabel_n
            rec["mem_reset"] = st.mem_reset
            rec["immune"] = st.immune
            return (st, ps), rec

        (st, ps), recs = lax.scan(body, (st, ps), None, length=n_dec)
        return st, recs
    rollout.trace_count = [0]
    return rollout




def run_episode(cfg: GeeseConfig, policy: Policy, seed: int, N: int, events=(), n_dec=None,
                immune=None, init_override=None, jit=True, record_obs=False):
    """Convenience: run one episode, return (final_state, Trajectory)."""
    sched = compile_schedule(list(events), cfg.e_max)
    f = make_rollout(cfg, policy, n_dec, record_obs)
    key = jax.random.PRNGKey(seed)
    g = jax.jit(f, static_argnames=()) if jit else f
    st, recs = g(key, jnp.int32(N), sched, init_override, immune)
    meta = dict(scenario="geese", agent=policy.name, seed=seed, N=N, n_max=cfg.n_max, dt=cfg.dt,
                delta_h=cfg.delta_h, stride=cfg.log_stride, gain=cfg.gain, P_solo=cfg.P_solo, b=cfg.b,
                swap_rule=cfg.swap_rule, action_mode=cfg.action_mode, dtype=cfg.dtype,
                cfg_hash=hashlib.sha1(repr(cfg).encode()).hexdigest()[:12],
                jax=jax.__version__, units=dict(D="P_solo*s", reward="mean(-P_eff/P_solo)", E_used="J",
                                              e_form="m"))
    return st, to_trajectory(cfg, recs, meta, sched)


def to_trajectory(cfg, recs, meta, sched=None) -> Trajectory:
    r = jax.tree_util.tree_map(lambda a: np.asarray(a), recs)
    low = r.pop("low", {})
    D = r.pop("D", None)
    glob = {}
    for k in ("leader", "n_alive", "t", "wind_ou", "n_swaps", "relabel_n", "mem_reset"):
        if k in r:
            glob[k] = r.pop(k)
    if D is not None:
        glob["D"] = D
    agent = dict(r)
    lowf = {}
    for k, v in low.items():
        v = np.asarray(v)
        lowf[k] = v.reshape((-1,) + v.shape[2:])
    events = {}
    if sched is not None:
        events = dict(kind_id=np.asarray(sched.kind_id), t_on=np.asarray(sched.t_on), t_off=np.asarray(sched.t_off),
                      target=np.asarray(sched.target), p=np.asarray(sched.p), perm=np.asarray(sched.perm))
    return Trajectory(agent, glob, lowf, events, [], meta)
