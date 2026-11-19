"""Physics step (continuous2d interface) and the hierarchical decision step."""
from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import lax

from . import wake
from .dynamics import e_form, e_ref, powers, safe, wind_at, wind_global
from .events import apply_events, kill_bird
from . import survival as surv
from .llc import llc
from .state import State, seqsum
from .swap import executor_override, resolve_swaps

BIGD = 1e9


def step_continuous(cfg, st: State, a):
    """Advance one low-level step (dt) with per-bird action a=(a_turn,a_v) in [-1,1].
    Returns (state, out) where out holds pre-step quantities and this step's P / reward."""
    dt = cfg.dt
    Ps = cfg.P_solo
    if cfg.use_events:
        st = apply_events(cfg, st)
    alive = st.alive
    a = jnp.where(alive[:, None], jnp.clip(a, -1.0, 1.0), 0.0)
    a = lax.optimization_barrier(a)
    wind = wind_at(cfg, st)
    pw = powers(cfg, st)
    P, Pnw = pw["P"], pw["P_nowake"]
    x, y, psi, v = safe(st)
    # fatigue / extra power terms (python-static branches keep kappa=0 bitwise == P)
    phi = st.phi
    P_eff = P
    if cfg.kappa != 0.0 or cfg.tau_f is not None:
        phi = jnp.where(alive, st.phi + (1 - jnp.exp(-dt / cfg.tau_f)) * (P / Ps - st.phi), st.phi)
    if cfg.kappa != 0.0:
        P_eff = P_eff * (1 + cfg.kappa * jnp.maximum(0.0, phi - cfg.phi0))
    if cfg.c_nav != 0.0:
        P_eff = P_eff + cfg.c_nav * Ps * (st.slot == 0)
    # reward terms
    n = cfg.n_max
    r = -P_eff / Ps
    if cfg.c_iso != 0.0 or cfg.c_col != 0.0:
        ddx = x[:, None] - x[None, :]
        ddy = y[:, None] - y[None, :]
        dist = jnp.sqrt(ddx ** 2 + ddy ** 2)
        mk = alive[:, None] & alive[None, :] & ~jnp.eye(n, dtype=bool)
        dmin = jnp.min(jnp.where(mk, dist, BIGD), axis=1)
        r = r - cfg.c_iso * (dmin > cfg.R_c * cfg.b) - cfg.c_col * (dmin < 0.3 * cfg.b)
    if cfg.c_prog != 0.0:
        vx = v * jnp.cos(psi) + wind[:, 0]
        r = r + cfg.c_prog * (vx - cfg.v0) / cfg.v0
    r = jnp.where(alive, r, 0.0)
    if cfg.reward_mode != "individual":
        na = jnp.maximum(seqsum([alive[j] * 1.0 for j in range(n)]), 1.0)
        rt = seqsum([r[j] for j in range(n)]) / na
        al = 1.0 if cfg.reward_mode == "team" else cfg.mix_alpha
        r = jnp.where(alive, (1 - al) * r + al * rt, 0.0)
    if cfg.E0_coef is not None and cfg.e0_death_penalty and not cfg.survival:
        r = jnp.where(st.died & ~alive, -cfg.e0_death_cost, r)      # remaining time x solo cost (spec v0.4)
    out = dict(P=P, P_nowake=Pnw, save=(Pnw - P) / Ps, e_form=e_form(cfg, st), e_ref=e_ref(cfg, st), r=r)
    if cfg.log_D:
        out["D"] = (cfg.gain * cfg.m * wake.G * pw["mult"][:, None] * pw["wm_eff"] * dt / Ps).T
    if cfg.log_low:
        out.update(x=st.x, y=st.y, psi=st.psi, v=st.v, alive=alive, wind=wind, t=st.t, a=a)
    # integrate
    psi2 = st.psi + cfg.turn_rate_max * a[:, 0] * dt
    vmax = cfg.vmax_frac * cfg.v0 * st.vmax_mult
    v2 = jnp.clip(st.v + cfg.vdot_max * a[:, 1] * dt, cfg.vmin_frac * cfg.v0, vmax)
    x2 = st.x + (v2 * jnp.cos(psi2) + wind[:, 0]) * dt
    y2 = st.y + (v2 * jnp.sin(psi2) + wind[:, 1]) * dt
    w = lambda new, old: jnp.where(alive, new, old)
    step2 = st.step + 1
    ou_k = jax.random.fold_in(st.key_wind, st.step)
    xi = jax.random.normal(ou_k, (2,), dtype=cfg.jdtype)
    ou2 = st.ou * jnp.exp(-dt / cfg.ou_tau) + jnp.sqrt(1 - jnp.exp(-2 * dt / cfg.ou_tau)) * xi
    st2 = st._replace(
        x=w(x2, st.x), y=w(y2, st.y), psi=w(psi2, st.psi), v=w(v2, st.v),
        E_used=st.E_used + P * dt, E_rem=st.E_rem - P * dt, phi=phi,
        a_prev=a, step=step2, t=step2.astype(cfg.jdtype) * dt, R=st.R + jnp.array([cfg.v0 * dt, 0.0], cfg.jdtype),
        ou=ou2.astype(cfg.jdtype),
        wind_f=(st.wind_f + (1 - jnp.exp(-dt / cfg.tau_wind_est)) * (wind_global(cfg, st) - st.wind_f)).astype(cfg.jdtype))
    if cfg.E0_coef is not None:
        dying = st2.alive & (st2.E_rem <= 0.0)
        st2 = st2._replace(died=st2.died | dying)

        def kill_all(s):
            return lax.fori_loop(0, n, lambda i, s_: kill_bird(cfg, s_, i, dying[i]), s)
        st2 = lax.cond(jnp.any(dying), kill_all, lambda s: s, st2)
    return st2, out


def decision_step(cfg, st: State, actions):
    """One high-level decision: executor -> swap resolution -> n_sub LLC+physics steps.
    Returns (state, rec) with per-interval aggregates."""
    n = cfg.n_max
    st = st._replace(mem_reset=jnp.zeros((), bool))      # visible to the policy for exactly one decision
    if cfg.survival:                                      # survival mode only (python-static; off => untouched)
        R0_before = st.R[0]
        E_pre = jnp.clip(st.E_rem, 0.0, None)
        st = surv.apply_burst(cfg, st)
    st, actions = executor_override(cfg, st, actions)
    st, sw = resolve_swaps(cfg, st, actions)
    st = st._replace(last_action=sw.eff_action, blocked=sw.blocked)
    z = jnp.zeros((n,), cfg.jdtype)
    acc0 = dict(P=z, save=z, r=z)
    if cfg.log_D:
        acc0["D"] = jnp.zeros((n, n), cfg.jdtype)

    def body(carry, _):
        s, acc = carry
        if cfg.use_events:
            s = apply_events(cfg, s)          # events first so the LLC already sees joiners/leavers (idempotent)
        a = llc(cfg, s)
        s, out = step_continuous(cfg, s, a)
        acc = dict(acc)
        acc["P"] = acc["P"] + out["P"]
        acc["save"] = acc["save"] + out["save"]
        acc["r"] = acc["r"] + out["r"]
        if cfg.log_D:
            acc["D"] = acc["D"] + out["D"]
        low = {k: out[k] for k in out if k not in ("D",)} if cfg.log_low else {}
        return (s, acc), low

    (st, acc), low = lax.scan(body, (st, acc0), None, length=cfg.n_sub)
    ns = cfg.n_sub
    Ps = cfg.P_solo
    P_mean = acc["P"] / ns / Ps
    st = st._replace(
        P_prev=jnp.where(st.alive, P_mean, 0.0),
        lock=jnp.maximum(st.lock - 1, 0).astype(jnp.int32),
        dec=st.dec + 1,
        cmd_exec_left=jnp.maximum(st.cmd_exec_left - 1, 0).astype(jnp.int32))
    rec = dict(reward=acc["r"] / ns, P_mean=P_mean, save_mean=acc["save"] / ns,
               swapped=sw.swapped, blocked=sw.blocked, eff_action=sw.eff_action, n_swaps=sw.n_swaps,
               e_form_end=e_form(cfg, st), e_ref_end=e_ref(cfg, st))
    if cfg.survival:
        sr = surv.reward(cfg, R0_before, E_pre, st)
        rec["reward"] = sr["reward"]
        rec["arrived"] = sr["arrived"]
        rec["cross"] = sr["cross"]
        rec["e_frac_end"] = surv.energy_frac(cfg, st)
    if cfg.log_D:
        rec["D"] = acc["D"]
    if cfg.log_low:
        s0 = cfg.log_stride
        rec["low"] = jax.tree_util.tree_map(lambda a_: a_[::s0], low)
    return st, rec
