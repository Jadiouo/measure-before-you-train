"""Low-level controller (LLC): pure function state -> (a_turn, a_v). Shared by all three agent generations."""
from __future__ import annotations

import jax.numpy as jnp

from .dynamics import safe, target_offset, wind_at
from .state import State, seqsum


def wrap(a):
    return (a + jnp.pi) % (2 * jnp.pi) - jnp.pi


def separation(cfg, st: State, x, y):
    """Repulsive ground-velocity term (n,2); zero outside sep_radius (never active inside the nominal V)."""
    n = cfg.n_max
    r = cfg.sep_radius * cfg.b
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    dist = jnp.sqrt(dx ** 2 + dy ** 2)
    mask = st.alive[:, None] & st.alive[None, :] & ~jnp.eye(n, dtype=bool)
    mag = cfg.sep_gain * cfg.v0 * jnp.maximum(0.0, (r - dist) / r)
    tiny = dist < 1e-9
    ux = jnp.where(tiny, 0.0, dx / jnp.where(tiny, 1.0, dist))
    uy = jnp.where(tiny, 1.0, dy / jnp.where(tiny, 1.0, dist))
    mx = jnp.where(mask, mag * ux, 0.0)
    my = jnp.where(mask, mag * uy, 0.0)
    sx = seqsum([mx[:, j] for j in range(n)])
    sy = seqsum([my[:, j] for j in range(n)])
    return jnp.stack([sx, sy], -1)


def llc(cfg, st: State, wind=None):
    """Returns a (n,2) in [-1,1]: (a_turn, a_v). Dead birds get 0."""
    dt = cfg.jdtype
    x, y, psi, v = safe(st)
    off, voff = target_offset(cfg, st)
    if wind is None:
        wind = wind_at(cfg, st)
    tx = st.R[0] + off[:, 0]
    ty = st.R[1] + off[:, 1]
    dev = jnp.stack([cfg.Kp * (tx - x) + voff[:, 0], cfg.Kp * (ty - y) + voff[:, 1]], -1)
    cap = cfg.delta_v_max * cfg.v0
    nrm = jnp.sqrt(jnp.sum(dev ** 2, -1) + 1e-18)
    dev = dev * jnp.minimum(1.0, cap / nrm)[:, None]
    sep = separation(cfg, st, x, y)
    vg = jnp.stack([cfg.v0 + dev[:, 0] + sep[:, 0], dev[:, 1] + sep[:, 1]], -1)
    va = vg - wind
    v_cmd = jnp.sqrt(jnp.sum(va ** 2, -1) + 1e-18)
    psi_cmd = jnp.arctan2(va[:, 1], va[:, 0])
    vmax = cfg.vmax_frac * cfg.v0 * st.vmax_mult
    v_cmd = jnp.clip(v_cmd, cfg.vmin_frac * cfg.v0, vmax)
    a_turn = jnp.clip(cfg.k_psi * wrap(psi_cmd - psi), -1.0, 1.0)
    a_v = jnp.clip(cfg.k_v * (v_cmd - v), -1.0, 1.0)
    a = jnp.stack([a_turn, a_v], -1).astype(dt)
    return jnp.where(st.alive[:, None], a, 0.0)
