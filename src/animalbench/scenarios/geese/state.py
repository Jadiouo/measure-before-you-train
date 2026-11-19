"""State pytree + slot geometry helpers (pure jax)."""
from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import Array

from ...core.perturbation import PerturbSchedule


class State(NamedTuple):
    # time
    t: Array          # s (= step*dt)
    step: Array       # int32 low-level step index
    dec: Array        # int32 decision index
    R: Array          # (2,) virtual reference point (ground frame)
    # bird kinematics (n_max,)
    x: Array
    y: Array
    psi: Array
    v: Array
    alive: Array      # bool
    slot: Array       # int32 V-slot id, -1 for dead
    power_mult: Array
    vmax_mult: Array
    E_used: Array     # J (integral of P)
    E_rem: Array      # J remaining (BIG if no cap)
    E0: Array
    phi: Array        # fatigue EMA of P/P_solo
    # high-level bookkeeping
    lock: Array       # int32 decisions
    blocked: Array    # bool: last request was not executed
    last_action: Array  # int32 effective last action
    P_prev: Array     # interval mean P/P_solo of previous decision
    a_prev: Array     # (n,2)
    path_t0: Array
    path_from: Array  # (n,2) offset at path start
    path_lane: Array  # signed lane direction (+1 = +y)
    apex_ctr: Array   # int32 0/1: apex retreat target = slot 1+ctr
    immune: Array     # bool
    died: Array       # bool: died of the E0 energy cap (gets the death penalty afterwards)
    id_owner: Array   # (n,) int32: original owner of the identity code the bird currently carries (relabel log)
    relabel_n: Array  # int32 count of relabel events so far
    mem_reset: Array  # bool: a relabel with reset_mem=1 happened during the previous decision (policy-side signal)
    # exogenous
    ou: Array         # (2,) unit OU wind
    wind_f: Array     # (2,) low-pass filtered global wind estimate (LLC formation frame)
    evt_done: Array   # (E,) bool
    evt_dir: Array    # (E,) rad (resolved at reset from 'pert' stream)
    evt_u: Array      # (E,) uniform (random target)
    id_code: Array    # (n,3)
    # command
    cmd_valid: Array
    cmd_leader: Array     # int32 bird id or -1
    cmd_rank: Array       # (n,) int32 target slot per bird (-1 none)
    cmd_t: Array
    cmd_exec_left: Array  # int32
    sched: PerturbSchedule
    key_wind: Array
    key_noise: Array
    key_slot: Array
    key_policy: Array


def slot_offset(cfg, k):
    """Ground-frame offset (..., 2) of V-slot k (k=0 apex). Arm rule: odd=left(+y), even=right(-y)."""
    dt = cfg.jdtype
    kk = jnp.maximum(k, 0)
    a = ((kk + 1) // 2).astype(dt)
    side = jnp.where(kk % 2 == 1, 1.0, -1.0).astype(dt)
    ox = -cfg.slot_dx * cfg.b * a
    oy = jnp.where(kk > 0, side * cfg.slot_dy * cfg.b * a, 0.0).astype(dt)
    return jnp.stack([ox.astype(dt), oy], axis=-1)


def arm_sign(k):
    """+1 left, -1 right, 0 apex."""
    return jnp.where(k == 0, 0, jnp.where(k % 2 == 1, 1, -1))


def occ_from_slot(cfg, slot, alive):
    n = cfg.n_max
    idx = jnp.where(alive, slot, n)
    return jnp.full((n,), -1, jnp.int32).at[idx].set(jnp.arange(n, dtype=jnp.int32), mode="drop")


def slot_from_occ(cfg, occ):
    n = cfg.n_max
    idx = jnp.where(occ >= 0, occ, n)
    return jnp.full((n,), -1, jnp.int32).at[idx].set(jnp.arange(n, dtype=jnp.int32), mode="drop")


def neighbor_slots(cfg, k):
    """(front, backA, backB, opposite) slot ids for slot k; -1 if none / outside n_max."""
    n = cfg.n_max
    k = k.astype(jnp.int32)
    front = jnp.where(k == 0, -1, jnp.where(k <= 2, 0, k - 2))
    backA = jnp.where(k == 0, 1, k + 2)
    backB = jnp.where(k == 0, 2, jnp.where(k % 2 == 1, k + 3, k + 1))
    opp = jnp.where(k == 0, -1, jnp.where(k % 2 == 1, k + 1, k - 1))
    out = []
    for s in (front, backA, backB, opp):
        out.append(jnp.where((s >= 0) & (s < n), s, -1))
    return out


def apex_target_slot(cfg, occ, apex_ctr):
    """Slot the apex would retreat to (alternating 1,2; falls back to the occupied one). -1 if none."""
    n = cfg.n_max
    s_pref = 1 + apex_ctr
    s_alt = 2 - apex_ctr
    ok_pref = (s_pref < n) & (occ[jnp.minimum(s_pref, n - 1)] >= 0)
    ok_alt = (s_alt < n) & (occ[jnp.minimum(s_alt, n - 1)] >= 0)
    return jnp.where(ok_pref, s_pref, jnp.where(ok_alt, s_alt, -1))


def seqsum(terms):
    """Left-to-right sum of a python list of arrays (fixed association => shape-independent bits)."""
    acc = terms[0]
    for t in terms[1:]:
        acc = acc + t
    return acc


def col_sum(M):
    """Sequential sum over the last axis of (..., n) (unrolled)."""
    n = M.shape[-1]
    return seqsum([M[..., j] for j in range(n)])
