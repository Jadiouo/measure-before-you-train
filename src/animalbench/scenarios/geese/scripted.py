"""Scripted HLC agents for the known-answer test (spec v0.3 #19) and the null-model agents (N-fix/N-rnd/N-cyc).

All of them are ordinary `Policy(name, init, act)` objects run through the real environment (`run_episode` /
`make_rollout`).  Information used (declared per agent, see notes/impl/M1a_stage2_decisions.md):
  treadmill (S1)       : L0 obs only (is_apex, lock)
  state_rotation (S2)  : L0 obs only (own P_prev -> own fatigue EMA, lock); needs swap_rule="mutual"
  reciprocity (S3)     : privileged identity: the occupant ids of slots 0/1/2 (stands for the 'nbr' identity code)
  N-fix                : nothing
  N-rnd / N-cyc        : privileged slot map (queue agents: a designated next leader climbs the V)
"""
from __future__ import annotations

import jax
import jax.numpy as jnp

from .hlc import BACK, FWD, STAY, Policy, stay_policy
from .obs import layout
from .state import occ_from_slot


def _legal(a, mask):
    return jnp.where(jnp.take_along_axis(mask, a[:, None], 1)[:, 0], a, STAY).astype(jnp.int32)


# ----------------------------------------------------------------------------------------------- S1
def treadmill(cfg) -> Policy:
    """S1 retreating treadmill: the apex retreats as soon as its lock is released; never advances."""
    lay = layout(cfg)

    def init(key, cfg_=None):
        return dict()

    def act(ps, obs, st, mask, key=None):
        is_apex = obs[:, lay["is_apex"][0]] > 0.5
        lock0 = obs[:, lay["lock"][0]] <= 0.0
        a = jnp.where(is_apex & lock0, BACK, STAY)
        return _legal(a, mask), ps
    return Policy("S1_treadmill", init, act)


# ----------------------------------------------------------------------------------------------- S2
def state_rotation(cfg, phi_hi=0.94, phi_lo=0.90) -> Policy:
    """S2 state-driven rotation (use with swap_rule='mutual'): every bird keeps its own fatigue EMA
    (same tau_f as the environment, built from its own P_prev), retreats when phi >= phi_hi and volunteers to
    advance when phi <= phi_lo.  A swap needs both sides (mutual), so the heavy bird sinks and the rested one rises.
    Anonymous: no identity, no clock."""
    lay = layout(cfg)
    n = cfg.n_max
    alpha = 1.0 - jnp.exp(-cfg.delta_h / cfg.tau_f)

    def init(key, cfg_=None):
        return dict(phi=jnp.ones((n,), cfg.jdtype))

    def act(ps, obs, st, mask, key=None):
        P_prev = obs[:, lay["P_prev"][0]]
        lock0 = obs[:, lay["lock"][0]] <= 0.0
        phi = ps["phi"] + alpha * (P_prev - ps["phi"])
        a = jnp.where(lock0 & (phi >= phi_hi), BACK, jnp.where(lock0 & (phi <= phi_lo), FWD, STAY))
        return _legal(a, mask), dict(phi=phi)
    return Policy("S2_state_rotation", init, act)


# ----------------------------------------------------------------------------------------------- hold timer
def _hold_update(cfg, ps, apex_id, apex_unlocked, lo, hi, fixed=None):
    """Scalar hold timer of the current apex (counts only unlocked decisions), re-drawn on every handover."""
    changed = apex_id != ps["apex"]
    cnt = ps["cnt"] + changed.astype(jnp.int32)
    k = jax.random.fold_in(ps["key"], cnt)
    if fixed is None:
        thold_new = lo + (hi - lo) * jax.random.uniform(k, (), dtype=cfg.jdtype)
    else:
        thold_new = jnp.asarray(fixed, cfg.jdtype)
    thold = jnp.where(changed, thold_new, ps["thold"])
    held = jnp.where(changed, 0.0, ps["held"])
    return changed, cnt, thold, held


# ----------------------------------------------------------------------------------------------- S3
def reciprocity(cfg, t_hold=(25.0, 35.0)) -> Policy:
    """S3 identity reciprocity with a binary ledger R[i][j] (1: i owes j), unilateral swaps.
    * Handover a -> b (b took the apex from a): R[a][b]=1 (a was relieved by b) and R[b][a]=0 (repayment, ledger flips).
    * The apex yields (BACK) after its hold time only if nobody in slots 1/2 owes it; if a debtor of the apex sits in
      slot 1/2 (unlocked), the debtor advances (FWD) once the apex has served its hold time, and the apex stays.
    The debtor therefore relieves the very bird that relieved it earlier: direct, partner-specific reciprocity."""
    n = cfg.n_max
    lo, hi = t_hold

    def init(key, cfg_=None):
        z = jnp.zeros((), cfg.jdtype)
        return dict(R=jnp.zeros((n, n), jnp.int32), apex=jnp.full((), -1, jnp.int32), held=z, thold=z + lo,
                    cnt=jnp.zeros((), jnp.int32), key=key)

    def act(ps, obs, st, mask, key=None):
        occ = occ_from_slot(cfg, st.slot, st.alive)
        a_id = occ[0]
        ac = jnp.clip(a_id, 0, n - 1)
        a_unl = (st.lock[ac] == 0) & (a_id >= 0)
        changed, cnt, thold, held = _hold_update(cfg, ps, a_id, a_unl, lo, hi)
        prev = jnp.clip(ps["apex"], 0, n - 1)
        valid = changed & (ps["apex"] >= 0) & (a_id >= 0)
        R = ps["R"]
        R = jnp.where(valid, R.at[prev, ac].set(1).at[ac, prev].set(0), R)
        ready = a_unl & (held >= thold)
        c = jnp.stack([occ[1], occ[2]])
        cc = jnp.clip(c, 0, n - 1)
        owes = (c >= 0) & (R[cc, ac] == 1) & (st.lock[cc] == 0)
        any_owes = jnp.any(owes)
        act_ = jnp.zeros((n,), jnp.int32)
        which = jnp.where(owes[0], cc[0], cc[1])
        act_ = jnp.where(jnp.arange(n) == which, jnp.where(ready & any_owes, FWD, STAY), act_)
        # a debtor in slot 1/2 that is still locked blocks the apex from yielding to somebody else
        debtor_exists = jnp.any((c >= 0) & (R[cc, ac] == 1))
        act_ = jnp.where((jnp.arange(n) == ac) & ready & ~debtor_exists, BACK, act_)
        held_next = jnp.where(a_unl, held + cfg.delta_h, held)
        return _legal(act_, mask), dict(R=R, apex=a_id, held=held_next, thold=thold, cnt=cnt, key=ps["key"])
    return Policy("S3_reciprocity", init, act)


# ----------------------------------------------------------------------------------------------- N-rnd / N-cyc
def queue_agent(cfg, order="cyc", t_hold=(25.0, 35.0), name=None, phi_hi=0.94) -> Policy:
    """Null-model agents in the real environment.  A 'designated next leader' d is chosen at every handover
    (cyc: the bird with the next id; rnd: uniform among the other alive birds), climbs the V by FWD (unilateral
    swaps) and takes the apex when the incumbent has served its hold time.  t_hold = (lo, hi) uniform (rnd) or a
    constant if lo == hi (cyc).  Sampling of d uses the 'policy' substream only.
    order='phi' (S2f, fair state-driven rotation, privileged): d = the alive non-apex bird with the lowest fatigue
    phi (true state), and the incumbent is displaced when ITS OWN phi >= phi_hi (no clock, no id order)."""
    n = cfg.n_max
    lo, hi = t_hold
    fixed = lo if lo == hi else None

    def init(key, cfg_=None):
        z = jnp.zeros((), cfg.jdtype)
        return dict(apex=jnp.full((), -1, jnp.int32), held=z, thold=z + lo, cnt=jnp.zeros((), jnp.int32),
                    d=jnp.full((), -1, jnp.int32), key=key)

    def pick(ps, a_id, alive, cnt, phi):
        ids = jnp.arange(n)
        if order == "phi":
            return jnp.argmin(jnp.where(alive & (ids != a_id), phi, 9.0)).astype(jnp.int32)
        if order == "cyc":
            nxt = (a_id + 1) % jnp.maximum(jnp.sum(alive.astype(jnp.int32)), 1)
            return nxt.astype(jnp.int32)
        k = jax.random.fold_in(jax.random.fold_in(ps["key"], 7919), cnt)
        w = jnp.where(alive & (ids != a_id), 1.0, 0.0)
        return jax.random.choice(k, n, p=w / jnp.sum(w)).astype(jnp.int32)

    def act(ps, obs, st, mask, key=None):
        occ = occ_from_slot(cfg, st.slot, st.alive)
        a_id = occ[0]
        ac = jnp.clip(a_id, 0, n - 1)
        a_unl = (st.lock[ac] == 0) & (a_id >= 0)
        changed, cnt, thold, held = _hold_update(cfg, ps, a_id, a_unl, lo, hi, fixed)
        d_new = pick(ps, a_id, st.alive, cnt, st.phi)
        d = jnp.where(changed | (ps["d"] < 0), d_new, ps["d"])
        dc = jnp.clip(d, 0, n - 1)
        k = st.slot[dc]
        front_slot = jnp.where(k <= 2, 0, k - 2)
        front_b = jnp.clip(occ[jnp.clip(front_slot, 0, n - 1)], 0, n - 1)
        front_unl = st.lock[front_b] == 0
        ready = a_unl & ((st.phi[ac] >= phi_hi) if order == "phi" else (held >= thold))
        go = (k >= 1) & (st.lock[dc] == 0) & front_unl & jnp.where(k <= 2, ready, True) & (d != a_id)
        act_ = jnp.where((jnp.arange(n) == dc) & go, FWD, STAY)
        held_next = jnp.where(a_unl, held + cfg.delta_h, held)
        return _legal(act_, mask), dict(apex=a_id, held=held_next, thold=thold, cnt=cnt, d=d, key=ps["key"])
    return Policy(name or f"N_{order}", init, act)


def null_cyc(cfg, t_hold=30.0) -> Policy:
    return queue_agent(cfg, "cyc", (t_hold, t_hold), "N_cyc")


def null_rnd(cfg, t_hold=(25.0, 35.0)) -> Policy:
    return queue_agent(cfg, "rnd", t_hold, "N_rnd")


def state_rotation_fair(cfg, phi_hi=0.94) -> Policy:
    return queue_agent(cfg, "phi", (0.0, 0.0), "S2f_state_rotation_fair", phi_hi)


def null_fix(cfg) -> Policy:
    p = stay_policy(cfg)
    return Policy("N_fix", p.init, p.act)


REGISTRY = {
    "S1_treadmill": (treadmill, "unilateral"),
    "S2_state_rotation": (state_rotation, "mutual"),
    "S2f_state_rotation_fair": (state_rotation_fair, "unilateral"),
    "S3_reciprocity": (reciprocity, "unilateral"),
    "N_cyc": (null_cyc, "unilateral"),
    "N_rnd": (null_rnd, "unilateral"),
    "N_fix": (null_fix, "unilateral"),
}
