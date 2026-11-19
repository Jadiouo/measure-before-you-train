"""M4br: scripted LEDGER RECIPROCATORS for swap_rule="mutual" (new file, nothing existing is changed).

Why a new family: M3 showed that individual-reward RL never learns rotation and that, with a scripted rotation as
the other birds' behaviour, "never step forward" pays (dilemma.json).  These policies are the *executable norm*
against which an RL bird's best response is measured (experiment 1, notes/impl/M4br_decisions.md).

Norm (all birds run it; T = decision counter, N = number of birds, theta = base tenure in decisions, 1 decision = 2 s; default 20 = 40 s, chosen so that at kappa=1 the rotation beats fixed leadership as the scripted S2 does, see M4br_decisions.md)
  lead_i   : decisions bird i has spent in the apex slot (own ledger; counted from the bird's own is_apex observable).
  fair     : T / N.
  apex a retreats (asks BACK) when unlocked and its tenure (consecutive apex decisions) >= theta + pend_a.
  a follower i (slot 1/2) volunteers (asks FWD) when unlocked and lead_i <= fair  ("I am not ahead of my fair share").
  Under swap_rule="mutual" a hand-over happens only if the apex asks BACK *and* its target asks FWD.
Versions
  "self"     : uses only the bird's own L0 observables (is_apex, lock, rank) and its own decision counter.
               No bird tracks anybody else, so a refusal cannot be punished; lam is ignored.  (= lambda 0)
  "identity" : additionally remembers WHO refused a hand-over (true-state occupant ids, standing for the 'nbr'
               identity code) and keeps the public fine ledger pend_i.
Refusal penalty lam (ledger_game.py: "a bird that REFUSES a ready hand-over gets u -= lam", i.e. has to lead lam more
decisions): a hand-over is READY when the (reciprocator) apex asks BACK, its target e is unlocked and lead_e <= fair
(= e would volunteer under the norm).  If the apex is still the apex at the next decision, e refused:
pend_e += lam.  A bird with pend_e > 0 who becomes apex needs tenure >= theta + pend_e before the reciprocators accept
its retreat, i.e. it must lead pend_e extra decisions (pend is cleared when it finally hands over).
Design reason: the only thing a reciprocator can withhold is its own FWD (mutual rule), so "no longer accept its
retreat" is the sole punishment channel that exists in the environment; it only bites a refuser who is (or becomes) apex.
Deviation codes (per bird, `kind`; 0 = reciprocator, -1 = externally controlled e.g. the RL bird; its output is ignored)
  1 never_fwd  2 never_back  3 grabber(FWD always, never BACK)  4 stay(never acts)  5 always_fwd (ignores fair share)
  6 short (retreats at theta/2)  7 long (retreats at 2 theta)  8 half_refuse (volunteers with prob .5)
"""
from __future__ import annotations

import jax
import jax.numpy as jnp

from .hlc import BACK, FWD, STAY, Policy
from .obs import layout
from .state import apex_target_slot, occ_from_slot

DEV_NAMES = {1: "never_fwd", 2: "never_back", 3: "grabber", 4: "stay", 5: "always_fwd", 6: "short", 7: "long",
             8: "half_refuse"}


def ledger_reciprocator(cfg, version="self", lam=0, theta=20, dev_id=-1, dev_kind=0, name=None) -> Policy:
    assert version in ("self", "identity")
    n = cfg.n_max
    lay = layout(cfg)
    ident = version == "identity"
    lamf = float(lam) if ident else 0.0
    kind = [0] * n
    if dev_id >= 0:
        kind[dev_id] = dev_kind
    kind = jnp.asarray(kind, jnp.int32)
    ids = jnp.arange(n)

    def init(key, cfg_=None):
        z = jnp.zeros((n,), cfg.jdtype)
        return dict(lead=z, ten=z, pend=z, t=jnp.zeros((), cfg.jdtype), prev_ready=jnp.zeros((), bool),
                    prev_tgt=jnp.zeros((), jnp.int32), prev_apex=jnp.full((), -1, jnp.int32),
                    ready_edge=jnp.zeros((n,), bool), due_apex=jnp.zeros((n,), bool))

    def act(ps, obs, st, mask, key=None):
        is_apex = obs[:, lay["is_apex"][0]] > 0.5
        unl = obs[:, lay["lock"][0]] <= 0.0
        alive = st.alive
        slot = jnp.round(obs[:, lay["rank"][0]] * (n - 1)).astype(jnp.int32)
        fol = alive & ~is_apex & (slot >= 1) & (slot <= 2)
        lead = ps["lead"] + (is_apex & alive).astype(cfg.jdtype)
        ten = jnp.where(is_apex, ps["ten"] + 1.0, 0.0)
        t = ps["t"] + 1.0
        nal = jnp.maximum(jnp.sum(alive.astype(cfg.jdtype)), 1.0)
        fair = t / nal
        pend = ps["pend"]
        occ = occ_from_slot(cfg, st.slot, alive)
        a_id = occ[0]
        ac = jnp.clip(a_id, 0, n - 1)
        if ident:                                           # clear the fine of a bird that has handed over; log refusals
            pend = jnp.where(ids == ps["prev_apex"], jnp.where(a_id != ps["prev_apex"], 0.0, pend), pend)
            refused = ps["prev_ready"] & (a_id == ps["prev_apex"])
            pend = jnp.where((ids == ps["prev_tgt"]) & refused, pend + lamf, pend)
        thr = theta + pend
        thr = jnp.where(kind == 6, theta / 2.0, jnp.where(kind == 7, 2.0 * theta, thr))
        want_back = is_apex & unl & (ten >= thr)
        want_fwd = fol & unl & (lead <= fair)
        u = jax.random.uniform(key if key is not None else jax.random.PRNGKey(0), (n,))
        want_fwd = jnp.where(kind == 5, fol & unl, want_fwd)
        want_fwd = jnp.where(kind == 8, want_fwd & (u < 0.5), want_fwd)
        want_fwd = jnp.where(kind == 3, fol & unl, want_fwd)
        want_fwd = jnp.where((kind == 1) | (kind == 4), False, want_fwd)
        want_back = jnp.where((kind == 2) | (kind == 3) | (kind == 4), False, want_back)
        a = jnp.where(want_back, BACK, jnp.where(want_fwd, FWD, STAY))
        a = jnp.where(jnp.take_along_axis(mask, a[:, None], 1)[:, 0], a, STAY).astype(jnp.int32)
        # diagnostics: is the hand-over from a reciprocator apex to bird e "ready" under the norm?
        tgt_slot = apex_target_slot(cfg, occ, st.apex_ctr)
        tgt_b = jnp.clip(occ[jnp.clip(tgt_slot, 0, n - 1)], 0, n - 1)
        apex_asks = (a[ac] == BACK) & (kind[ac] == 0) & (tgt_slot >= 0)
        ready_tgt = apex_asks & unl[tgt_b] & (lead[tgt_b] <= fair)
        ready_edge = (ids == tgt_b) & ready_tgt
        due_apex = is_apex & unl & (ten >= thr)
        ps2 = dict(lead=lead, ten=ten, pend=pend, t=t, prev_ready=ready_tgt, prev_tgt=tgt_b.astype(jnp.int32),
                   prev_apex=a_id.astype(jnp.int32), ready_edge=ready_edge, due_apex=due_apex)
        return a, ps2
    nm = name or (f"LR_{version}" + (f"_lam{lam}" if ident else "") + (f"_dev{DEV_NAMES[dev_kind]}{dev_id}" if dev_id >= 0 and dev_kind > 0 else ""))
    return Policy(nm, init, act)
