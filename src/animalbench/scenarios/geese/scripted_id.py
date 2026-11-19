"""Scripted agents for the INTERVENTION known-answer test (spec v0.4 3.3), added next to scripted.py (which is unchanged).

reciprocity_code (S3c) : S3 with the binary ledger keyed by IDENTITY CODE (st.id_owner[body], i.e. the code a bird currently
                         carries), not by body index.  After an I1 relabel the ledger entries therefore point at the wrong
                         birds.  The ledger is the policy memory: st.mem_reset (I1-M / I1-RM) wipes it.
reciprocity_code(punish=True) (S3p): S3c plus a privileged "withdrawal" rule used only as the positive control of I2:
                         once any alive bird is immune (a bird that will never reciprocate) the apex stops yielding and no
                         debtor advances (collective punishment / exit).  Privileged detector (reads st.immune); it is an
                         extra row, not part of the 4-agent diagonal of 3.3.
queue_agent_ia          : queue_agent (cyc / rnd / phi) whose designated next leader is never an immune bird (the original
                         queue_agent keeps designating the immune free-rider and then deadlocks because the env refuses every
                         swap that involves it).  Reported next to the literal agents.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp

from .hlc import FWD, BACK, STAY, Policy
from .scripted import _hold_update, _legal
from .state import occ_from_slot


def reciprocity_code(cfg, t_hold=(25.0, 35.0), punish=False) -> Policy:
    n = cfg.n_max
    lo, hi = t_hold

    def init(key, cfg_=None):
        z = jnp.zeros((), cfg.jdtype)
        return dict(R=jnp.zeros((n, n), jnp.int32), apex=jnp.full((), -1, jnp.int32), held=z, thold=z + lo,
                    cnt=jnp.zeros((), jnp.int32), key=key)

    def act(ps, obs, st, mask, key=None):
        # --- policy memory reset (I1-M / I1-RM): the env only raises the flag
        wipe = st.mem_reset
        R0 = jnp.where(wipe, jnp.zeros_like(ps["R"]), ps["R"])
        apex_prev = jnp.where(wipe, -1, ps["apex"])
        own = st.id_owner                                         # code carried by body i
        occ = occ_from_slot(cfg, st.slot, st.alive)
        a_body = occ[0]
        ab = jnp.clip(a_body, 0, n - 1)
        a_code = own[ab]
        a_unl = (st.lock[ab] == 0) & (a_body >= 0)
        # apex change is tracked in CODE space (that is all the policy can see)
        ps_h = dict(ps, apex=apex_prev)
        changed, cnt, thold, held = _hold_update(cfg, ps_h, a_code, a_unl, lo, hi)
        prev = jnp.clip(apex_prev, 0, n - 1)
        valid = changed & (apex_prev >= 0) & (a_body >= 0)
        ac = jnp.clip(a_code, 0, n - 1)
        R = jnp.where(valid, R0.at[prev, ac].set(1).at[ac, prev].set(0), R0)
        ready = a_unl & (held >= thold)
        c_body = jnp.stack([occ[1], occ[2]])
        cb = jnp.clip(c_body, 0, n - 1)
        c_code = own[cb]
        owes = (c_body >= 0) & (R[c_code, ac] == 1) & (st.lock[cb] == 0)
        any_owes = jnp.any(owes)
        which = jnp.where(owes[0], cb[0], cb[1])
        act_ = jnp.zeros((n,), jnp.int32)
        go_fwd = ready & any_owes
        debtor_exists = jnp.any((c_body >= 0) & (R[c_code, ac] == 1))
        go_back = ready & ~debtor_exists
        if punish:
            ex = jnp.any(st.alive & st.immune)
            go_fwd = go_fwd & ~ex
            go_back = go_back & ~ex
        act_ = jnp.where(jnp.arange(n) == which, jnp.where(go_fwd, FWD, STAY), act_)
        act_ = jnp.where(jnp.arange(n) == ab, jnp.where(go_back, BACK, act_[ab]), act_)
        held_next = jnp.where(a_unl, held + cfg.delta_h, held)
        return _legal(act_, mask), dict(R=R, apex=a_code, held=held_next, thold=thold, cnt=cnt, key=ps["key"])
    return Policy("S3p_reciprocity_punish" if punish else "S3c_reciprocity_code", init, act)


def queue_agent_ia(cfg, order="cyc", t_hold=(25.0, 35.0), name=None, phi_hi=0.94) -> Policy:
    """queue_agent with immune birds excluded from the candidate set (identical when nobody is immune)."""
    n = cfg.n_max
    lo, hi = t_hold
    fixed = lo if lo == hi else None

    def init(key, cfg_=None):
        z = jnp.zeros((), cfg.jdtype)
        return dict(apex=jnp.full((), -1, jnp.int32), held=z, thold=z + lo, cnt=jnp.zeros((), jnp.int32),
                    d=jnp.full((), -1, jnp.int32), key=key)

    def pick(ps, a_id, cand, cnt, phi):
        ids = jnp.arange(n)
        if order == "phi":
            return jnp.argmin(jnp.where(cand & (ids != a_id), phi, 9.0)).astype(jnp.int32)
        if order == "cyc":
            # walk ids a_id+1, a_id+2, ... (mod n) and take the first candidate (= (a_id+1) % n_alive when all are alive)
            dist = jnp.mod(ids - a_id - 1, n)
            return jnp.argmin(jnp.where(cand & (ids != a_id), dist, n + 1)).astype(jnp.int32)
        k = jax.random.fold_in(jax.random.fold_in(ps["key"], 7919), cnt)
        w = jnp.where(cand & (ids != a_id), 1.0, 0.0)
        return jax.random.choice(k, n, p=w / jnp.maximum(jnp.sum(w), 1e-9)).astype(jnp.int32)

    def act(ps, obs, st, mask, key=None):
        occ = occ_from_slot(cfg, st.slot, st.alive)
        a_id = occ[0]
        ac = jnp.clip(a_id, 0, n - 1)
        a_unl = (st.lock[ac] == 0) & (a_id >= 0)
        changed, cnt, thold, held = _hold_update(cfg, ps, a_id, a_unl, lo, hi, fixed)
        cand = st.alive & ~st.immune
        d_new = pick(ps, a_id, cand, cnt, st.phi)
        stale = (ps["d"] < 0) | st.immune[jnp.clip(ps["d"], 0, n - 1)]       # designated bird became immune -> re-pick
        d = jnp.where(changed | stale, d_new, ps["d"])
        dc = jnp.clip(d, 0, n - 1)
        k = st.slot[dc]
        front_slot = jnp.where(k <= 2, 0, k - 2)
        front_b = jnp.clip(occ[jnp.clip(front_slot, 0, n - 1)], 0, n - 1)
        front_unl = (st.lock[front_b] == 0) & ~st.immune[front_b]
        ready = a_unl & ((st.phi[ac] >= phi_hi) if order == "phi" else (held >= thold))
        go = (k >= 1) & (st.lock[dc] == 0) & front_unl & jnp.where(k <= 2, ready, True) & (d != a_id)
        act_ = jnp.where((jnp.arange(n) == dc) & go, FWD, STAY)
        held_next = jnp.where(a_unl, held + cfg.delta_h, held)
        return _legal(act_, mask), dict(apex=a_id, held=held_next, thold=thold, cnt=cnt, d=d, key=ps["key"])
    return Policy(name or f"N_{order}_ia", init, act)


def null_cyc_ia(cfg, t_hold=30.0):
    return queue_agent_ia(cfg, "cyc", (t_hold, t_hold), "N_cyc_ia")


def null_rnd_ia(cfg, t_hold=(25.0, 35.0)):
    return queue_agent_ia(cfg, "rnd", t_hold, "N_rnd_ia")


def state_rotation_fair_ia(cfg, phi_hi=0.94):
    return queue_agent_ia(cfg, "phi", (0.0, 0.0), "S2f_ia", phi_hi)


# ------------------------------------------------------------------ false-positive agents for the I1 classifier (added 2026-10-30)
def pair_agent(cfg, partner="body", code_noise=False, t_hold=30.0, name=None) -> Policy:
    """Stateless fixed-partner ping-pong (no memory, no ledger): the apex hands over to its partner after a fixed hold time.
    partner="body": partner(b) = b ^ 1 in BODY space (pure state; identity codes are never read).
    partner="code": partner(c) = c ^ 1 in CODE space, looked up through st.id_owner (stateless lookup table that reads the code;
                    after an I1 relabel the pairs change).  Expected I1 truth: identity-dependent (lookup, not memory).
    code_noise=True: the code is read ONLY to jitter the hold time (deterministic hash of apex code and handover count,
                    25..35 s): reads the code, does not reciprocate by it, partner stays body-space => expected truth: NOT identity.
    Immune birds are never designated (same convention as queue_agent_ia)."""
    n = cfg.n_max

    def init(key, cfg_=None):
        z = jnp.zeros((), cfg.jdtype)
        return dict(apex=jnp.full((), -1, jnp.int32), held=z, thold=z + t_hold, cnt=jnp.zeros((), jnp.int32),
                    d=jnp.full((), -1, jnp.int32), key=key)

    def partner_of(x):
        p = jnp.bitwise_xor(x, 1)
        return jnp.where(p < cfg.n_max, p, 0)

    def act(ps, obs, st, mask, key=None):
        occ = occ_from_slot(cfg, st.slot, st.alive)
        a_id = occ[0]
        ac = jnp.clip(a_id, 0, n - 1)
        a_unl = (st.lock[ac] == 0) & (a_id >= 0)
        changed, cnt, thold, held = _hold_update(cfg, ps, a_id, a_unl, t_hold, t_hold, t_hold)
        own = st.id_owner
        if code_noise:
            jit = jnp.mod(own[ac] * 7 + cnt * 3, 10).astype(cfg.jdtype) / 9.0
            thold = jnp.where(changed, 25.0 + 10.0 * jit, ps["thold"])
        if partner == "code":
            pc = partner_of(own[ac])
            d_new = jnp.argmax(st.alive & (own == pc)).astype(jnp.int32)
        else:
            d_new = partner_of(ac).astype(jnp.int32)
        stale = (ps["d"] < 0) | st.immune[jnp.clip(ps["d"], 0, n - 1)]
        d = jnp.where(changed | stale, d_new, ps["d"])
        dc = jnp.clip(d, 0, n - 1)
        k = st.slot[dc]
        front_slot = jnp.where(k <= 2, 0, k - 2)
        front_b = jnp.clip(occ[jnp.clip(front_slot, 0, n - 1)], 0, n - 1)
        front_unl = (st.lock[front_b] == 0) & ~st.immune[front_b]
        ready = a_unl & (held >= thold)
        go = (k >= 1) & (st.lock[dc] == 0) & front_unl & jnp.where(k <= 2, ready, True) & (d != a_id) & ~st.immune[dc]
        act_ = jnp.where((jnp.arange(n) == dc) & go, FWD, STAY)
        held_next = jnp.where(a_unl, held + cfg.delta_h, held)
        return _legal(act_, mask), dict(apex=a_id, held=held_next, thold=thold, cnt=cnt, d=d, key=ps["key"])
    return Policy(name or f"P_{partner}{'_noise' if code_noise else ''}", init, act)


def pair_body(cfg):
    return pair_agent(cfg, "body", False, name="P_body")


def pair_body_codenoise(cfg):
    return pair_agent(cfg, "body", True, name="P_body_codenoise")


def pair_code_lookup(cfg):
    return pair_agent(cfg, "code", False, name="P_code_lookup")


# name -> (maker, swap_rule); same convention as scripted.REGISTRY
REGISTRY_ID = {
    "S3c_reciprocity_code": (reciprocity_code, "unilateral"),
    "S3p_reciprocity_punish": (lambda cfg: reciprocity_code(cfg, punish=True), "unilateral"),
    "N_cyc_ia": (null_cyc_ia, "unilateral"),
    "N_rnd_ia": (null_rnd_ia, "unilateral"),
    "S2f_ia": (state_rotation_fair_ia, "unilateral"),
    "P_body": (pair_body, "unilateral"),
    "P_body_codenoise": (pair_body_codenoise, "unilateral"),
    "P_code_lookup": (pair_code_lookup, "unilateral"),
}
