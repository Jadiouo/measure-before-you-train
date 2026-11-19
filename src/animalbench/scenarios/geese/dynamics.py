"""Wind, wake matrix, target path, formation error. All pure jax; dead birds are sanitised."""
from __future__ import annotations

import jax.numpy as jnp
import numpy as np
from jax import lax

from . import wake
from .state import State, slot_offset, col_sum, seqsum


def safe(st: State):
    """Dead birds -> harmless finite values so nothing can leak into live birds."""
    a = st.alive
    z = jnp.zeros_like(st.x)
    return (jnp.where(a, st.x, z), jnp.where(a, st.y, z), jnp.where(a, st.psi, z),
            jnp.where(a, st.v, jnp.ones_like(st.v)))


def _crosswind_det(cfg, st: State, t):
    """Deterministic part of the global wind (crosswind events) at time t -> (wg (2,), ou_scale)."""
    s = st.sched
    dt = cfg.jdtype
    wg = jnp.zeros((2,), dt)
    ou_scale = jnp.zeros((), dt)
    for e in range(cfg.e_max):
        is_cw = s.kind_id[e] == 1
        speed, ramp, ous = s.p[e, 0], jnp.maximum(s.p[e, 2], 1e-9), s.p[e, 3]
        env = (jnp.clip((t - s.t_on[e]) / ramp, 0.0, 1.0)
               * jnp.clip((s.t_off[e] + ramp - t) / ramp, 0.0, 1.0))
        env = jnp.where(is_cw, env, 0.0)
        d = st.evt_dir[e]
        wg = wg + env * speed * jnp.stack([jnp.cos(d), jnp.sin(d)])
        ou_scale = ou_scale + env * ous
    return wg, ou_scale


def wind_global(cfg, st: State):
    """(2,) uniform wind (crosswind events + OU), without gusts."""
    wg, ou_scale = _crosswind_det(cfg, st, st.t)
    return wg + ou_scale * st.ou


def wind_at(cfg, st: State):
    """(n,2) ground-frame wind at each bird: global wind + gust events."""
    s = st.sched
    dt = cfg.jdtype
    t = st.t
    n = cfg.n_max
    wb = jnp.zeros((n, 2), dt)
    for e in range(cfg.e_max):
        d = st.evt_dir[e]
        is_g = s.kind_id[e] == 2
        amp, front = s.p[e, 0], s.p[e, 2]
        dur = jnp.where(is_g, jnp.maximum(s.t_off[e] - s.t_on[e], 1e-9), 1.0)
        delay = jnp.where(front > 0.5, (st.R[0] - st.x) / cfg.gust_speed, 0.0)
        ph = (t - s.t_on[e] - delay) / dur
        win = jnp.where((ph > 0) & (ph < 1), 0.5 * (1 - jnp.cos(2 * jnp.pi * ph)), 0.0)
        win = jnp.where(is_g, win, 0.0)
        wb = wb + (amp * win)[:, None] * jnp.stack([jnp.cos(d), jnp.sin(d)])[None, :]
    return wb + wind_global(cfg, st)[None, :]


def formation_heading(cfg, st: State):
    """(psi_f, dpsi_f/dt) used by the LLC: air-relative heading of the formation reference computed from the
    low-pass filtered wind estimate (tau_wind_est), so the V is built in a frame aligned with the wake axes.
    A sudden wind change therefore leaves a transient mismatch (the physical effect of a crosswind)."""
    wf = st.wind_f
    w = wind_global(cfg, st)
    wdot = (w - wf) / cfg.tau_wind_est
    X, Y = cfg.v0 - wf[0], -wf[1]
    psi_f = jnp.arctan2(Y, X)
    dpsi = (X * (-wdot[1]) - Y * (-wdot[0])) / (X ** 2 + Y ** 2)
    return psi_f, dpsi


def ideal_heading(cfg, st: State):
    """Heading of the ideal formation frame: true mean (deterministic) wind, no OU turbulence, no lag."""
    w, _ = _crosswind_det(cfg, st, st.t)
    return jnp.arctan2(-w[1], cfg.v0 - w[0])


def _rot(psi_f, o):
    c, s_ = jnp.cos(psi_f), jnp.sin(psi_f)
    return jnp.stack([c * o[..., 0] - s_ * o[..., 1], s_ * o[..., 0] + c * o[..., 1]], -1)


def target_offset(cfg, st: State, rotate=True):
    """Per-bird commanded offset from R and its time derivative (smoothstep path + swap lane).
    rotate=True: ground-frame (formation frame rotated by psi_f), used by the LLC.
    rotate=False: formation-frame offset (what `path_from` must store).  path_from lives in the formation frame
    because target_offset rotates the interpolated path; storing the already-rotated offset (old behaviour)
    rotated it twice under wind, so a swap path started at the wrong point."""
    dt = cfg.jdtype
    to = slot_offset(cfg, st.slot)
    s_raw = (st.t - st.path_t0) / cfg.path_T
    s = jnp.clip(s_raw, 0.0, 1.0)
    h = s * s * (3 - 2 * s)
    inside = (s_raw > 0) & (s_raw < 1)
    hd = jnp.where(inside, 6 * s * (1 - s) / cfg.path_T, 0.0)
    lane = st.path_lane * cfg.swap_lane * cfg.b
    d = to - st.path_from
    off_x = st.path_from[:, 0] + d[:, 0] * h
    off_y = st.path_from[:, 1] + d[:, 1] * h + lane * jnp.sin(jnp.pi * s)
    vx = d[:, 0] * hd
    vy = d[:, 1] * hd + jnp.where(inside, lane * jnp.pi * jnp.cos(jnp.pi * s) / cfg.path_T, 0.0)
    off = jnp.stack([off_x, off_y], -1)
    vel = jnp.stack([vx, vy], -1)
    if not rotate:
        return off.astype(dt), vel.astype(dt)
    psi_f, dpsi_f = formation_heading(cfg, st)
    off_r = _rot(psi_f, off)
    vel_r = _rot(psi_f, vel) + dpsi_f * jnp.stack([-off_r[:, 1], off_r[:, 0]], -1)
    return off_r.astype(dt), vel_r.astype(dt)


def _slot_err_vec(cfg, st: State, frame="ideal"):
    psi_f = ideal_heading(cfg, st) if frame == "ideal" else formation_heading(cfg, st)[0]
    to = _rot(psi_f, slot_offset(cfg, st.slot))
    tx = st.R[0] + to[:, 0]
    ty = st.R[1] + to[:, 1]
    x, y, _, _ = safe(st)
    return jnp.where(st.alive, x - tx, 0.0), jnp.where(st.alive, y - ty, 0.0)


def e_ref(cfg, st: State):
    """Per-bird distance (m) to its LLC slot target (LLC frame) measured against the virtual reference R."""
    ex, ey = _slot_err_vec(cfg, st, "llc")
    return jnp.where(st.alive, jnp.sqrt(ex ** 2 + ey ** 2), 0.0)


def e_form(cfg, st: State):
    """Per-bird formation error (m) with the rigid common translation of the flock removed
    (spec v0.1 sec 3.5): |err_i - mean_alive(err)|, against the ideal wake-aligned V. 0 for dead birds."""
    ex, ey = _slot_err_vec(cfg, st)
    na = jnp.maximum(seqsum([st.alive[j] * 1.0 for j in range(cfg.n_max)]), 1.0)
    mx = seqsum([ex[j] for j in range(cfg.n_max)]) / na
    my = seqsum([ey[j] for j in range(cfg.n_max)]) / na
    d = jnp.sqrt((ex - mx) ** 2 + (ey - my) ** 2)
    return jnp.where(st.alive, d, 0.0)


def wake_matrix(cfg, st: State):
    """w[i, j]: mean upwash (m/s) on follower i from leader j (alive-masked, diag 0)."""
    x, y, psi, v = safe(st)
    dxw = x[None, :] - x[:, None]       # leader j - follower i
    dyw = y[None, :] - y[:, None]
    cj, sj = jnp.cos(psi)[None, :], jnp.sin(psi)[None, :]
    d_x = dxw * cj + dyw * sj
    d_y = dxw * sj - dyw * cj
    w = wake.wbar(jnp, d_x / cfg.b, d_y / cfg.b, v[None, :], cfg)
    n = cfg.n_max
    mask = st.alive[:, None] & st.alive[None, :] & ~jnp.eye(n, dtype=bool)
    return jnp.where(mask, w, 0.0)


def powers(cfg, st: State):
    """Returns dict: P, P_nowake, save_frac (of P_solo), W (clipped), wm_eff (n,n) clipped-scaled upwash."""
    wm = lax.optimization_barrier(wake_matrix(cfg, st))
    Wsum = col_sum(wm)
    floor = -0.2 * cfg.v0
    W = jnp.maximum(Wsum, floor)
    scale = jnp.where(Wsum < floor, floor / jnp.where(Wsum < floor, Wsum, -1.0), 1.0)
    _, _, _, v = safe(st)
    pn = wake.p_nowake(cfg, v)
    mult = st.power_mult
    P = mult * (pn - cfg.gain * cfg.m * wake.G * W)
    P_nw = mult * pn
    alive = st.alive
    P = jnp.where(alive, P, 0.0)
    P_nw = jnp.where(alive, P_nw, 0.0)
    P, P_nw, W = lax.optimization_barrier((P, P_nw, W))
    return dict(P=P, P_nowake=P_nw, W=W, wm_eff=wm * scale[:, None], mult=mult)
