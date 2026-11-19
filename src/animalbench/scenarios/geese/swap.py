"""High-level action semantics: masks, swap resolution (edge conflicts, lock, immunity, lottery), command executor."""
from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import lax

from .dynamics import target_offset
from .state import (State, apex_target_slot, arm_sign, occ_from_slot, slot_from_occ, slot_offset)

STAY, BACK, FWD = 0, 1, 2


class SwapInfo(NamedTuple):
    swapped: jnp.ndarray      # (n,) bool, per bird
    blocked: jnp.ndarray      # (n,) bool, request not executed
    eff_action: jnp.ndarray   # (n,) int32 action after mask/lock/immune enforcement
    edge_acc: jnp.ndarray     # (n,) bool, per back-slot edge accepted
    n_swaps: jnp.ndarray


def action_mask(cfg, st: State):
    """(n,3) bool. STAY always legal; BACK/FWD need a target, no lock, bird alive and not immune."""
    n = cfg.n_max
    occ = occ_from_slot(cfg, st.slot, st.alive)
    k = jnp.maximum(st.slot, 0)
    bt = jnp.where(k == 0, apex_target_slot(cfg, occ, st.apex_ctr), jnp.where(k + 2 < n, k + 2, -1))
    has_back = (bt >= 0) & (occ[jnp.clip(bt, 0, n - 1)] >= 0)
    front = jnp.where(k == 0, -1, jnp.where(k <= 2, 0, k - 2))
    has_fwd = (front >= 0) & (occ[jnp.clip(front, 0, n - 1)] >= 0)
    free = st.alive & (st.lock == 0) & ~st.immune
    return jnp.stack([jnp.ones((n,), bool), has_back & free, has_fwd & free], -1)


def resolve_swaps(cfg, st: State, actions):
    """Resolve one decision's requests. actions: (n,) int in {0,1,2}. Returns (state, SwapInfo)."""
    n = cfg.n_max
    mutual = cfg.swap_rule == "mutual"
    alive = st.alive
    occ = occ_from_slot(cfg, st.slot, alive)
    s_idx = jnp.arange(n, dtype=jnp.int32)
    mask = action_mask(cfg, st)
    raw = jnp.where(alive, actions.astype(jnp.int32), 0)
    # per-slot views
    b_s = occ                                         # bird in slot s (-1 none)
    bs_c = jnp.clip(b_s, 0, n - 1)
    occupied = b_s >= 0
    # target existence ignoring lock/immune (for blocked-flag semantics)
    k = jnp.maximum(st.slot, 0)
    bt = jnp.where(k == 0, apex_target_slot(cfg, occ, st.apex_ctr), jnp.where(k + 2 < n, k + 2, -1))
    has_back = (bt >= 0) & (occ[jnp.clip(bt, 0, n - 1)] >= 0)
    fr = jnp.where(k <= 2, 0, k - 2)
    has_fwd = (k > 0) & (occ[fr] >= 0)
    tgt_exists = ((raw == BACK) & has_back) | ((raw == FWD) & has_fwd)
    # enforcement: locked / immune / dead -> STAY; no target -> STAY
    allowed = jnp.take_along_axis(mask, raw[:, None], 1)[:, 0]
    eff = jnp.where(allowed, raw, 0).astype(jnp.int32)
    forced_stay = (raw != 0) & ((st.lock > 0) | st.immune)           # lock/immune override => blocked
    act_s = jnp.where(occupied, eff[bs_c], 0)         # effective action per slot
    # back target per slot
    bt_s = jnp.where(s_idx == 0, apex_target_slot(cfg, occ, st.apex_ctr), jnp.where(s_idx + 2 < n, s_idx + 2, -1))
    front_s = jnp.where(s_idx <= 2, 0, s_idx - 2)     # front slot of edge indexed by back slot s (s>=1)
    # edge e (indexed by back slot s>=1): (front_s[s], s)
    pf = (act_s[front_s] == BACK) & (bt_s[front_s] == s_idx)
    pb = act_s == FWD
    exist = occupied & occupied[front_s] & (s_idx >= 1)
    cand = exist & ((pf & pb) if mutual else (pf | pb))
    lock_s = jnp.where(occupied, st.lock[bs_c] > 0, False)
    imm_s = jnp.where(occupied, st.immune[bs_c], False)
    ok = cand & ~lock_s & ~lock_s[front_s] & ~imm_s & ~imm_s[front_s]
    coin = jax.random.bernoulli(jax.random.fold_in(st.key_slot, st.dec), 0.5)
    order = [jnp.where(coin, 1, 2), jnp.where(coin, 2, 1)] + list(range(3, n))
    used = jnp.zeros((n,), bool)
    acc = jnp.zeros((n,), bool)
    for e in order:
        if isinstance(e, int):
            if e >= n:
                continue
        elif n < 3:
            continue
        f = front_s[e]
        take = ok[e] & ~used[f] & ~used[e]
        acc = acc.at[e].set(take)
        used = used.at[f].set(used[f] | take)
        used = used.at[e].set(used[e] | take)
    # execute: new occupancy
    new_occ = occ
    f_all = front_s
    for_back = jnp.where(acc, occ, -1)                 # bird currently at back slot of accepted edge
    for_front = jnp.where(acc, occ[f_all], -1)         # bird currently at front slot
    # scatter swaps (indices are disjoint because accepted edges share no slot)
    idx_back = jnp.where(acc, s_idx, n)
    idx_front = jnp.where(acc, f_all, n)
    new_occ = new_occ.at[idx_back].set(for_front, mode="drop")
    new_occ = new_occ.at[idx_front].set(for_back, mode="drop")
    swapped_bird = jnp.zeros((n,), bool)
    swapped_bird = swapped_bird.at[jnp.where(acc, for_back, n)].set(True, mode="drop")
    swapped_bird = swapped_bird.at[jnp.where(acc, for_front, n)].set(True, mode="drop")
    new_slot = jnp.where(alive, slot_from_occ(cfg, new_occ), -1)
    # path bookkeeping: retreating bird (was front) lane = +side(back slot); advancing = -side
    side = arm_sign(s_idx).astype(cfg.jdtype)
    lane = jnp.zeros((n,), cfg.jdtype)
    lane = lane.at[jnp.where(acc, for_front, n)].set(side, mode="drop")
    lane = lane.at[jnp.where(acc, for_back, n)].set(-side, mode="drop")
    cur_off, _ = target_offset(cfg, st, rotate=False)   # formation frame (see target_offset)
    path_from = jnp.where(swapped_bird[:, None], cur_off, st.path_from)
    path_t0 = jnp.where(swapped_bird, st.t, st.path_t0)
    path_lane = jnp.where(swapped_bird, lane, st.path_lane)
    lock = jnp.where(swapped_bird, cfg.lock_dec, st.lock)
    # apex counter: next apex retreat goes to the other arm
    apex_edge = acc[1] | acc[2] if n > 2 else (acc[1] if n > 1 else jnp.zeros((), bool))
    ctr_new = jnp.where(acc[1] if n > 1 else False, 1, 0)
    apex_ctr = jnp.where(apex_edge, ctr_new, st.apex_ctr).astype(jnp.int32)
    blocked = ((raw != 0) & ~swapped_bird & tgt_exists) | forced_stay & ~swapped_bird
    blocked = blocked & alive
    st2 = st._replace(slot=new_slot, path_from=path_from, path_t0=path_t0, path_lane=path_lane,
                      lock=lock.astype(jnp.int32), apex_ctr=apex_ctr)
    return st2, SwapInfo(swapped_bird, blocked, eff, acc, jnp.sum(acc.astype(jnp.int32)))


def apply_command(cfg, st: State, next_leader=-1, order_hint=None):
    """Write a Command {next_leader, order_hint} into the state (python-side call)."""
    n = cfg.n_max
    rank = -jnp.ones((n,), jnp.int32)
    if order_hint is not None:
        for r, i in enumerate(order_hint):
            rank = rank.at[int(i)].set(r)
    return st._replace(cmd_valid=jnp.ones((), bool), cmd_leader=jnp.asarray(next_leader, jnp.int32),
                       cmd_rank=rank, cmd_t=st.t, cmd_exec_left=jnp.asarray(cfg.cmd_exec_max, jnp.int32))


def executor_override(cfg, st: State, actions):
    """CommandExecutor: for <= cmd_exec_max decisions force next_leader FWD and the bird in front BACK (lock-aware)."""
    n = cfg.n_max
    active = st.cmd_valid & (st.cmd_exec_left > 0) & (st.cmd_leader >= 0)
    L = jnp.clip(st.cmd_leader, 0, n - 1)
    occ = occ_from_slot(cfg, st.slot, st.alive)
    kL = jnp.maximum(st.slot[L], 0)
    mask = action_mask(cfg, st)
    can = active & st.alive[L] & (kL > 0) & mask[L, FWD]
    front_slot = jnp.where(kL <= 2, 0, kL - 2)
    fb = occ[front_slot]
    fb_c = jnp.clip(fb, 0, n - 1)
    acts = actions
    acts = acts.at[L].set(jnp.where(can, FWD, acts[L]))
    # make the front bird retreat towards L (set apex counter when the front bird is the apex)
    st2 = st
    want_ctr = jnp.where(kL == 1, 0, 1)
    st2 = st2._replace(apex_ctr=jnp.where(can & (front_slot == 0), want_ctr, st.apex_ctr).astype(jnp.int32))
    mask2 = action_mask(cfg, st2)
    can_f = can & (fb >= 0) & mask2[fb_c, BACK]
    acts = acts.at[fb_c].set(jnp.where(can_f, BACK, acts[fb_c]))
    return st2, acts
