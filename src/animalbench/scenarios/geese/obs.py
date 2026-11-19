"""Observation builders (hier 25(+15) / continuous2d 31(+15)) and layout map."""
from __future__ import annotations

import jax
import jax.numpy as jnp

from .dynamics import e_ref, safe, wind_at
from .llc import wrap
from .state import State, neighbor_slots, occ_from_slot


def layout(cfg):
    """Return dict name -> (start, stop) and total dims. L0 = everything before 'command'."""
    lay = {}
    o = 0

    def add(name, d):
        nonlocal o
        lay[name] = (o, o + d)
        o += d
    if cfg.action_mode == "hier":
        add("P_prev", 1); add("is_apex", 1); add("rank", 1); add("arm", 1); add("lock", 1); add("blocked", 1)
        add("last_action", 3); add("e_form", 1); add("n_alive", 1)
        if cfg.survival:
            add("e_self", 1); add("e_min_other", 1)
        if cfg.obs_clock:
            add("clock", 1)
        add("wind", 2)
        add("nbr", 8)
    else:
        add("v", 1); add("P_prev", 1); add("a_prev", 2); add("psi", 1)
        if cfg.obs_clock:
            add("clock", 1)
        add("wind", 2)
        add("nbr", 20)
    lay["L0_end"] = (o, o)
    add("command", 4)
    add("identity", 15)
    lay["dim"] = (o, o)
    lay["dim_no_identity"] = (lay["identity"][0], lay["identity"][0])
    return lay


def obs_dim(cfg, identity=True):
    lay = layout(cfg)
    return lay["dim"][0] if identity else lay["dim_no_identity"][0]


def _per_bird_normal(key, n, shape, dtype):
    keys = jax.vmap(lambda i: jax.random.fold_in(key, i))(jnp.arange(n))
    return jax.vmap(lambda k: jax.random.normal(k, shape, dtype=dtype))(keys)


def build_obs(cfg, st: State):
    """(n_max, D) float obs for every bird slot; dead birds are all-zero. Always includes the 15 identity dims
    (zero when cfg.identity == 'none')."""
    n = cfg.n_max
    dt = cfg.jdtype
    alive = st.alive
    nk = jax.random.fold_in(st.key_noise, st.dec)
    wind = wind_at(cfg, st)
    wn = _per_bird_normal(jax.random.fold_in(nk, 0), n, (2,), dt)
    wind_est = wind + cfg.wind_sigma * wn
    cols = []
    lay = layout(cfg)
    ef = e_ref(cfg, st)
    occ = occ_from_slot(cfg, st.slot, alive)
    k = jnp.maximum(st.slot, 0)
    nbs = neighbor_slots(cfg, k)
    nb_id = [jnp.where(s >= 0, occ[jnp.clip(s, 0, n - 1)], -1) for s in nbs]
    nb_ex = [(i >= 0) for i in nb_id]
    nb_lock = [jnp.where(e, st.lock[jnp.clip(i, 0, n - 1)] > 0, False) for i, e in zip(nb_id, nb_ex)]
    n_alive = jnp.sum(alive.astype(jnp.int32)).astype(dt)
    if cfg.action_mode == "hier":
        arm = jnp.where(k == 0, 0, jnp.where(k % 2 == 1, 1, -1)).astype(dt)
        onehot = jax.nn.one_hot(st.last_action, 3, dtype=dt)
        cols += [st.P_prev[:, None], (k == 0).astype(dt)[:, None], (k.astype(dt) / (n - 1))[:, None], arm[:, None],
                 (st.lock.astype(dt) / cfg.lock_dec)[:, None], st.blocked.astype(dt)[:, None], onehot,
                 (ef / cfg.b)[:, None], jnp.full((n, 1), n_alive / n, dt)]
        if cfg.survival:                                   # own energy fraction + lowest energy fraction among the others
            den = (cfg.E0_coef * cfg.P_solo * cfg.T) if cfg.surv_obs_abs else st.E0     # common constant => h / E0 spread visible
            ef_ = jnp.clip(st.E_rem, 0.0, None) / den
            oth = alive[None, :] & ~jnp.eye(n, dtype=bool)
            mo = jnp.min(jnp.where(oth, ef_[None, :], 2.0), axis=1)
            cols += [ef_[:, None], jnp.where(mo > 1.5, 1.0, mo)[:, None]]
        if cfg.obs_clock:
            cols.append(jnp.full((n, 1), st.t / cfg.T, dt))
        cols.append(wind_est)
        nbf = []
        for e, l in zip(nb_ex, nb_lock):
            nbf += [e.astype(dt)[:, None], l.astype(dt)[:, None]]
        cols.append(jnp.concatenate(nbf, -1))
    else:
        x, y, psi, v = safe(st)
        cols += [(v / cfg.v0)[:, None], st.P_prev[:, None], st.a_prev, wrap(psi)[:, None]]
        if cfg.obs_clock:
            cols.append(jnp.full((n, 1), st.t / cfg.T, dt))
        cols.append(wind_est)
        # nearest 4 visible neighbours
        dxw = x[None, :] - x[:, None]
        dyw = y[None, :] - y[:, None]
        c, s_ = jnp.cos(psi)[:, None], jnp.sin(psi)[:, None]
        bx = c * dxw + s_ * dyw
        by = -s_ * dxw + c * dyw
        dist = jnp.sqrt(dxw ** 2 + dyw ** 2)
        ang = jnp.arctan2(by, bx)
        vis = (dist < cfg.sense_radius * cfg.b) & (jnp.abs(ang) < jnp.deg2rad(cfg.fov_deg)) \
            & alive[:, None] & alive[None, :] & ~jnp.eye(n, dtype=bool)
        key = jnp.where(vis, dist, 1e9)
        order = jnp.argsort(key, axis=1, stable=True)[:, :4]
        kk = jnp.take_along_axis(key, order, 1)
        valid = kk < 1e8
        noise = _per_bird_normal(jax.random.fold_in(nk, 1), n, (4, 2), dt) * cfg.nb_sigma
        gx = jnp.take_along_axis(bx, order, 1) / cfg.b + noise[..., 0]
        gy = jnp.take_along_axis(by, order, 1) / cfg.b + noise[..., 1]
        dpsi = wrap(psi[order] - psi[:, None])
        dv = (v[order] - v[:, None]) / cfg.v0
        f = jnp.stack([gx, gy, dpsi, dv, valid.astype(dt)], -1)
        f = jnp.where(valid[..., None], f, 0.0)
        cols.append(f.reshape(n, 20))
    # command slice
    cv = st.cmd_valid
    tr = st.cmd_rank
    has_r = cv & (tr >= 0)
    c0 = jnp.where(has_r, (tr - st.slot).astype(dt) / n, 0.0)
    is_next = (cv & (st.cmd_leader == jnp.arange(n))).astype(dt)
    age = jnp.where(cv, jnp.clip((st.t - st.cmd_t) / cfg.cmd_age_scale, 0.0, 1.0), 0.0)
    cols.append(jnp.stack([c0, is_next, jnp.full((n,), cv, dt), jnp.broadcast_to(age, (n,)).astype(dt)], -1))
    # identity
    idc = st.id_code
    if cfg.identity == "none":
        cols.append(jnp.zeros((n, 15), dt))
    else:
        selfc = idc if cfg.identity == "nbr+self" else jnp.zeros_like(idc)
        parts = [selfc]
        for i, e in zip(nb_id, nb_ex):
            parts.append(jnp.where(e[:, None], idc[jnp.clip(i, 0, n - 1)], 0.0))
        cols.append(jnp.concatenate(parts, -1))
    obs = jnp.concatenate(cols, -1)
    return jnp.where(alive[:, None], obs, 0.0)
