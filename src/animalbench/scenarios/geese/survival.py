"""Survival-migration mode (cfg.survival=True; default OFF).  Fixed route D, random bursts, energy death, mixed reward.

Spec: notes/impl/survival_ippo_prereg.md (+ small game notes/impl/survival_game.md).
  * route: the formation reference point R advances v0 per second, so the route is finished at a fixed decision; a bird
    ARRIVES iff, at the decision where R[0] first reaches D, it is alive, E_rem>0 and not further than surv_lag_tol behind
    its slot (otherwise: timeout = not arrived).  After that decision the episode reward is 0 (done).
  * bursts: every decision each alive bird, with prob surv_burst_p, loses surv_burst_m*delta_h*Exp(1)*P_solo of energy.
  * death: E_rem <= 0 -> kill_bird (events.py), the V closes up behind it (leader dies -> slot 1 becomes apex).
  * reward (paid once, at the crossing decision):  indiv_i = arrived_i * (1 + beta*E_i/E0_i)
        team = mean over participants (alive or died) of indiv_j ;  r_i = (1-alpha)*indiv_i + alpha*team
    The energy bonus is only paid to arrived birds (dead: 0; alive-but-late: 0).
  * optional dense shaping surv_shaping="energy": per decision  -beta*(clip(E_before)-clip(E_after))/E0 (before-burst to
    after-physics, so bursts count), mixed with alpha the same way, and the terminal energy term is replaced by
    beta*[participant] - beta*[alive & not arrived]*E/E0.  Summed over the episode this equals the sparse return EXACTLY
    (telescoping) at gamma = 1, for every outcome (arrived / died / late).  For gamma < 1 it re-weights WHEN energy is
    spent (early spending is penalised more, relative to the terminal reward), so the optimal policy can change.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import lax

from .events import kill_bird
from .state import slot_offset


def apply_burst(cfg, st):
    """Random energy bursts at the start of a decision, then death check (reuses kill_bird)."""
    n = cfg.n_max
    k = jax.random.fold_in(jax.random.fold_in(st.key_noise, 1000003), st.dec)
    ku, ke = jax.random.split(k)
    hit = (jax.random.uniform(ku, (n,), dtype=cfg.jdtype) < cfg.surv_burst_p) & st.alive
    extra = cfg.surv_burst_m * cfg.delta_h * cfg.P_solo * jax.random.exponential(ke, (n,), dtype=cfg.jdtype)
    dE = jnp.where(hit, extra, 0.0).astype(cfg.jdtype)
    st = st._replace(E_rem=st.E_rem - dE, E_used=st.E_used + dE)
    return kill_dying(cfg, st)


def kill_dying(cfg, st):
    dying = st.alive & (st.E_rem <= 0.0)
    st = st._replace(died=st.died | dying)
    return lax.cond(jnp.any(dying),
                    lambda s: lax.fori_loop(0, cfg.n_max, lambda i, s_: kill_bird(cfg, s_, i, dying[i]), s),
                    lambda s: s, st)


def energy_frac(cfg, st):
    return jnp.clip(st.E_rem, 0.0, None) / st.E0


def reward(cfg, R0_before, E_pre, st):
    """st = state after the physics of this decision.  E_pre = clip(E_rem, 0) before bursts (n,).
    Returns dict(reward (n,), arrived (n,) bool, cross () bool)."""
    D, beta, al = cfg.D_route, cfg.surv_beta, cfg.surv_alpha
    dt = cfg.jdtype
    live = R0_before < D                                   # episode not finished before this decision
    cross = live & (st.R[0] >= D)
    part = (st.alive | st.died).astype(dt)
    lag = st.R[0] + slot_offset(cfg, st.slot)[:, 0] - st.x
    arrived = cross & st.alive & (st.E_rem > 0.0) & (lag <= cfg.surv_lag_tol)
    arr = arrived.astype(dt)
    ef = energy_frac(cfg, st)
    npart = jnp.maximum(part.sum(), 1.0)
    if cfg.surv_shaping == "energy":
        term = arr + beta * part - beta * (st.alive & ~arrived).astype(dt) * ef
        shp = -beta * (E_pre - jnp.clip(st.E_rem, 0.0, None)) / st.E0
        shp = jnp.where(live, shp, 0.0) * part
        indiv = jnp.where(cross, term, 0.0) * part + shp
    else:
        indiv = jnp.where(cross, arr * (1.0 + beta * ef), 0.0)
    team = indiv.sum() / npart      # NB: dead participants have indiv = 0 (sparse) / their shaping+constant (dense)
    r = ((1.0 - al) * indiv + al * team) * part
    return dict(reward=r.astype(dt), arrived=arrived, cross=cross)
