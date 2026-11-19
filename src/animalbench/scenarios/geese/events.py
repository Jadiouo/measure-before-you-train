"""Discrete events inside the step: kill (death/leave), join, injury. Fixed shapes, jit-safe."""
from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import lax

from ...core.perturbation import BIG
from .dynamics import target_offset
from .state import State, occ_from_slot, slot_from_occ, slot_offset


def _compact(cfg, occ, k):
    """Remove the occupant of slot k from occupancy: followers on the arm each advance one place.
    Apex (k=0): slot 1 moves up to the apex and the left arm advances."""
    n = cfg.n_max
    s = jnp.arange(n, dtype=jnp.int32)
    aff = (s >= k) & ((s - k) % 2 == 0)
    src = jnp.where(aff, s + 2, s)
    src = jnp.where(k == 0, jnp.where(s == 0, 1, jnp.where(s % 2 == 1, s + 2, s)), src)
    new = jnp.where(src < n, occ[jnp.clip(src, 0, n - 1)], -1)
    return new


def kill_bird(cfg, st: State, i, cond=True):
    """Set bird i dead (if `cond` and alive) and compact the V behind it."""
    n = cfg.n_max
    do = jnp.asarray(cond) & st.alive[i]
    occ = occ_from_slot(cfg, st.slot, st.alive)
    k = jnp.maximum(st.slot[i], 0)
    occ2 = _compact(cfg, occ, k)
    alive2 = st.alive.at[i].set(False)
    slot2 = jnp.where(alive2, slot_from_occ(cfg, occ2), -1)
    changed = do & alive2 & (slot2 != st.slot)
    cur_off, _ = target_offset(cfg, st, rotate=False)   # formation frame (see target_offset)
    st2 = st._replace(
        alive=jnp.where(do, alive2, st.alive),
        slot=jnp.where(do, slot2, st.slot),
        path_from=jnp.where(changed[:, None], cur_off, st.path_from),
        path_t0=jnp.where(changed, st.t, st.path_t0),
        path_lane=jnp.where(changed, 0.0, st.path_lane),
    )
    return st2


def spawn_bird(cfg, st: State, cond, jidx=0, nspawn=1):
    """Activate the first dormant (dead) bird slot behind the V tail; takes the smallest empty V slot."""
    n = cfg.n_max
    dt = cfg.jdtype
    dormant = ~st.alive
    has = jnp.any(dormant)
    i = jnp.argmax(dormant)
    do = jnp.asarray(cond) & has
    occ = occ_from_slot(cfg, st.slot, st.alive)
    empty = occ < 0
    new_slot = jnp.argmax(empty).astype(jnp.int32)
    # tail x-offset over occupied slots
    offs = slot_offset(cfg, jnp.arange(n, dtype=jnp.int32))
    tail_x = jnp.min(jnp.where(~empty, offs[:, 0], 0.0))
    y_off = (jidx - (nspawn - 1) / 2.0) * 1.5 * cfg.b
    x_new = st.R[0] + tail_x - 10.0 * cfg.b
    y_new = st.R[1] + y_off
    E0i = st.E0[i]
    def set_(arr, val):
        return arr.at[i].set(jnp.where(do, val, arr[i]))
    return st._replace(
        x=set_(st.x, x_new), y=set_(st.y, y_new), psi=set_(st.psi, 0.0), v=set_(st.v, cfg.v0),
        alive=set_(st.alive, True), slot=set_(st.slot, new_slot),
        power_mult=set_(st.power_mult, 1.0), vmax_mult=set_(st.vmax_mult, 1.0),
        E_used=set_(st.E_used, 0.0), E_rem=set_(st.E_rem, E0i), phi=set_(st.phi, 1.0),
        lock=set_(st.lock, 0), blocked=set_(st.blocked, False), last_action=set_(st.last_action, 0),
        P_prev=set_(st.P_prev, 1.0), path_t0=set_(st.path_t0, -BIG), path_lane=set_(st.path_lane, 0.0),
        path_from=st.path_from.at[i].set(jnp.where(do, jnp.zeros(2, dt), st.path_from[i])),
        a_prev=st.a_prev.at[i].set(jnp.where(do, jnp.zeros(2, dt), st.a_prev[i])),
        immune=set_(st.immune, False), died=set_(st.died, False),
    )


def _resolve_target(cfg, st, e):
    n = cfg.n_max
    tgt = st.sched.target[e]
    occ = occ_from_slot(cfg, st.slot, st.alive)
    leader = jnp.maximum(occ[0], 0)
    n_alive = jnp.sum(st.alive.astype(jnp.int32))
    kth = jnp.minimum((st.evt_u[e] * n_alive).astype(jnp.int32), jnp.maximum(n_alive - 1, 0))
    cs = jnp.cumsum(st.alive.astype(jnp.int32))
    rnd = jnp.argmax(st.alive & (cs == kth + 1))
    last = jnp.argmax(jnp.where(st.alive, st.slot, -1))
    return jnp.where(tgt == -1, leader, jnp.where(tgt == -2, rnd, jnp.where(tgt == -4, last, jnp.clip(tgt, 0, n - 1))))


def _relabel_src(cfg, st, e):
    """Source index per bird for relabel event e: new_code[i] = old_code[src[i]].  Explicit permutation from the
    schedule, or (row[0] == -1) a random cyclic shift by s in [1, N-1] of the alive birds in slot-independent index
    order (a derangement: no bird keeps its code).  Dead birds map to themselves."""
    n = cfg.n_max
    row = st.sched.perm[e][:n]
    idx = jnp.arange(n, dtype=jnp.int32)
    alive = st.alive
    na = jnp.sum(alive.astype(jnp.int32))
    rank = jnp.cumsum(alive.astype(jnp.int32)) - 1
    sh = 1 + jnp.minimum((st.evt_u[e] * jnp.maximum(na - 1, 1)).astype(jnp.int32), jnp.maximum(na - 2, 0))
    new_rank = jnp.mod(rank + sh, jnp.maximum(na, 1))
    match = alive[None, :] & (rank[None, :] == new_rank[:, None])
    cyc = jnp.where(alive, jnp.argmax(match, axis=1).astype(jnp.int32), idx)
    return jnp.where(st.sched.perm[e][0] < 0, cyc, jnp.where(alive, row, idx).astype(jnp.int32))


def apply_events(cfg, st: State):
    s = st.sched
    due = (s.kind_id >= 3) & (st.t >= s.t_on - 1e-9) & ~st.evt_done

    def run(st):
        def body(e, st):
            de = due[e]
            kind = s.kind_id[e]
            tgt = _resolve_target(cfg, st, e)
            p = s.p[e]
            # injury
            inj = de & (kind == 3) & st.alive[tgt]
            pm = st.power_mult.at[tgt].set(jnp.where(inj, p[0], st.power_mult[tgt]))
            vm = st.vmax_mult.at[tgt].set(jnp.where(inj, p[1], st.vmax_mult[tgt]))
            st = st._replace(power_mult=pm, vmax_mult=vm)
            if cfg.E0_coef is not None:
                er = st.E_rem.at[tgt].set(jnp.where(inj, st.E_rem[tgt] - p[2] * st.E0[tgt], st.E_rem[tgt]))
                st = st._replace(E_rem=er)
            # leave
            st = kill_bird(cfg, st, tgt, de & (kind == 5))
            # identity relabel (I1): codes move between birds; GRU memory is the policy's business (flag mem_reset)
            rl = de & (kind == 6)
            src = jnp.where(rl, _relabel_src(cfg, st, e), jnp.arange(cfg.n_max, dtype=jnp.int32))
            st = st._replace(id_code=st.id_code[src], id_owner=st.id_owner[src],
                             relabel_n=st.relabel_n + rl.astype(jnp.int32),
                             mem_reset=st.mem_reset | (rl & (p[0] > 0.5)))
            # immune toggle (I2): from now on the bird is never swapped (free-rider that never leads)
            im = de & (kind == 7)
            st = st._replace(immune=st.immune.at[tgt].set(jnp.where(im, p[0] > 0.5, st.immune[tgt])))
            # join (up to 3)
            nj = jnp.round(p[0]).astype(jnp.int32)
            for j in range(3):
                st = spawn_bird(cfg, st, de & (kind == 4) & (j < nj), j, jnp.maximum(nj, 1))
            return st._replace(evt_done=st.evt_done.at[e].set(st.evt_done[e] | de))
        return lax.fori_loop(0, cfg.e_max, body, st)

    return lax.cond(jnp.any(due), run, lambda s_: s_, st)
