"""JAX port of the L3 drone-bucket simulator (scripts/drone_bucket_l23.py) + MAPPO environment interface.

The dynamics (event, per-step consumption, EMA burn estimate, death/fractional-distance accounting) are a line-for-line
port of `drone_bucket_l23.simulate`.  Position tables, noise, events and instances are NOT re-implemented: they come from
the original module (L1.ratio_table / power_mult, OLD.sample, L23.gen_noise / gen_event), so cells are identical.
Equivalence with the numpy simulator is checked in tests/test_drone_l3.py (rules rot and local, float64).

Decisions happen every DT=5 simulation steps (t % 5 == 0, t > 0), the finest grid of the local rule (tau in {5,10,20}) and
a divisor of every rot tau.  A decision block = decision (applied at the first step of the block, that step is the swap
step) + DT simulation steps.

Actions (per plane, L3 information only; masks in `masks`):
  0 none
  1 swap_rear   : plane at position p asks to swap with the plane at p+1 (valid if p < N-1 and m_p, m_{p+1} differ >= EPS_M)
  2 rotate      : only the plane at position 0 (the leader); every plane moves to p-1 (leader to N-1) = the `rot` rule
  3 gate_swap   : (positive-control mode only) leader swaps with the plane having the largest rd (global gate action)
Conflicts: rotate > gate_swap > pair swaps; pair requests accepted greedily from position 0 (a plane already swapped
with its front neighbour cannot also be swapped with its rear neighbour).
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import NamedTuple

import numpy as np
import jax
import jax.numpy as jnp
from jax import lax

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import drone_bucket_l1 as L1          # noqa: E402
import drone_bucket as OLD            # noqa: E402
import drone_bucket_l23 as L23        # noqa: E402

B0 = L23.B0
DT = 5
EPS_M = L23.EPS_M
EVENT_GAIN = L23.EVENT_GAIN
ALPHA_ENV = 0.05          # EMA burn estimate carried by the env (= alpha of the best local rules; rules use 0 or 0.05)
NB = 300                  # decision blocks per episode cap (1500 sim steps); checked: all episodes end earlier
NEG = -1e9


def fdt():
    return jnp.asarray(0.0).dtype


# ---------------------------------------------------------------------------------------------------------------------
# cell definition (identical construction to L23.run_job)
# ---------------------------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class Cell:
    N: int
    kind: str
    L: int
    sigma: float
    p_event: float
    c: int
    h: float

    @property
    def ni(self):
        return [tuple(x) for x in L23.NOISES].index((self.kind, int(self.L), float(self.sigma)))

    @property
    def name(self):
        return f"N{self.N}_{self.kind}{self.L}_s{self.sigma}_p{self.p_event}_c{self.c}_h{self.h}"


def cell_from_json(c):
    return Cell(int(c["N"]), c["noise"], int(c["L"]), float(c["sigma"]), float(c["p_event"]), int(c["c"]), float(c["h"]))


def cell_params(cell: Cell):
    r, _, _ = L1.ratio_table(cell.N, "V", "actual")
    r2 = L1.two_plane_best()
    scale = 0.40 / (1 - r2)
    mvec = np.asarray(L1.power_mult(r, L23.F_IND, scale), dtype=float)
    s_ref = float(1 - mvec[1:].mean())
    return mvec, s_ref, cell.c * s_ref


def eval_instances(cell: Cell, n_inst=200):
    """Exactly the instances / noise / events that L23.run_job used for this cell (numpy)."""
    ni = cell.ni
    kind, Lc, sigma = L23.NOISES[ni]
    trivial = (kind == "none" and cell.p_event == 0.0)
    n = 1 if (cell.h == 0 and trivial) else n_inst
    k, B = OLD.sample(1000 * cell.N + int(round(cell.h * 100)), n, cell.N, cell.h)
    seed = 10 ** 6 + 1000 * cell.N + 10 * ni + int(round(cell.h * 100))
    w = L23.gen_noise(seed, n, cell.N, kind, Lc, sigma)
    ev = L23.gen_event(seed + 7, n, cell.N, cell.p_event) if cell.p_event > 0 else None
    return k, B, w, ev


# ---------------------------------------------------------------------------------------------------------------------
# state
# ---------------------------------------------------------------------------------------------------------------------
class St(NamedTuple):
    rem: jnp.ndarray        # (B,N)
    pos: jnp.ndarray        # (B,N) int  position index of each plane
    kmul: jnp.ndarray       # (B,N)
    rho: jnp.ndarray        # (B,N) EMA of observed burn multiplier
    last_part: jnp.ndarray  # (B,N) int  last step the plane took part in a swap (-1e6 never)
    last_swap: jnp.ndarray  # (B,)  int  last step any swap happened
    dist: jnp.ndarray       # (B,)
    alive: jnp.ndarray      # (B,) bool
    eps: jnp.ndarray        # (B,N) AR(1) state (generated-noise mode)
    nswap: jnp.ndarray      # (B,)  number of decision blocks with a swap


class Inst(NamedTuple):
    k: jnp.ndarray          # (B,N)
    Bcap: jnp.ndarray       # (B,N)
    ev_on: jnp.ndarray      # (B,) bool
    ev_plane: jnp.ndarray   # (B,) int
    ev_te: jnp.ndarray      # (B,) int


def inst_from_numpy(k, B, ev):
    n = k.shape[0]
    if ev is None:
        ev = (np.zeros(n, bool), np.zeros(n, int), np.full(n, -1))
    return Inst(jnp.asarray(k, fdt()), jnp.asarray(B, fdt()), jnp.asarray(ev[0]), jnp.asarray(ev[1], jnp.int32),
                jnp.asarray(ev[2], jnp.int32))


class Env:
    """Static cell parameters + pure functions (all jit-able; closures over numpy constants)."""

    def __init__(self, cell: Cell, control: bool = False):
        self.cell = cell
        self.N = cell.N
        self.control = control            # positive-control mode: global gate action + global obs features
        mvec, s_ref, ceff = cell_params(cell)
        self.mvec_np = mvec
        self.ceff = float(ceff)
        self.kind, self.Lc, self.sigma = L23.NOISES[cell.ni]
        self.nact = 4 if control else 3
        # which adjacent position pairs are swappable (classes differ)
        self.pair_ok_np = np.abs(mvec[:-1] - mvec[1:]) >= EPS_M
        N = cell.N
        self.obs_dim = N + 26 + (2 if control else 0)
        self.cobs_dim = 5 * N + 3

    # ------------------------------------------------------------------ init
    def init_state(self, inst: Inst, init_pos=None, key=None):
        B, N = inst.k.shape
        pos = jnp.tile(jnp.arange(N, dtype=jnp.int32), (B, 1)) if init_pos is None else jnp.asarray(init_pos, jnp.int32)
        eps = jnp.zeros((B, N), fdt()) if key is None else jax.random.normal(key, (B, N), fdt())
        return St(inst.Bcap, pos, jnp.ones((B, N), fdt()), jnp.ones((B, N), fdt()),
                  jnp.full((B, N), -10 ** 6, jnp.int32), jnp.full((B,), -10 ** 6, jnp.int32),
                  jnp.zeros((B,), fdt()), jnp.ones((B,), bool), eps, jnp.zeros((B,), fdt()))

    # ------------------------------------------------------------------ noise
    def wt_from_eps(self, eps):
        if self.kind == "none":
            return jnp.ones_like(eps)
        return jnp.maximum(1 + self.sigma * eps, 0.1)

    def eps_next(self, eps, key):
        z = jax.random.normal(key, eps.shape, eps.dtype)
        if self.kind == "ar":
            phi = float(np.exp(-1.0 / self.Lc))
            return phi * eps + float(np.sqrt(1 - phi * phi)) * z
        return z

    # ------------------------------------------------------------------ one simulation step (port of L23.simulate)
    def micro(self, st: St, inst: Inst, t, wt, part):
        N = self.N
        mvec = jnp.asarray(self.mvec_np, fdt())
        trig = inst.ev_on & (inst.ev_te == t)
        hit = trig[:, None] & (jnp.arange(N)[None, :] == inst.ev_plane[:, None])
        kmul = jnp.where(hit, EVENT_GAIN, st.kmul)
        mpl = mvec[st.pos]
        f = jnp.where(part, 1.0, mpl)
        cons = inst.k * kmul * wt * f + self.ceff * inst.k * part
        rho = jnp.where(part, st.rho, (1 - ALPHA_ENV) * st.rho + ALPHA_ENV * (kmul * wt))
        new = st.rem - cons
        dead = st.alive & (new <= 0).any(axis=1)
        frac = jnp.where(cons > 0, st.rem / jnp.where(cons > 0, cons, 1.0), jnp.inf).min(axis=1)
        inc = jnp.where(st.alive, jnp.where(dead, frac, 1.0), 0.0)
        rem = jnp.where(st.alive[:, None], new, st.rem)
        return st._replace(rem=rem, kmul=kmul, rho=rho, dist=st.dist + inc, alive=st.alive & ~dead), inc

    # ------------------------------------------------------------------ decisions
    def occ(self, st):
        return jnp.argsort(st.pos, axis=1)

    def rd(self, st, inst, alpha_est):
        return st.rem / (inst.k * (st.rho if alpha_est > 0 else 1.0))

    def apply_decision(self, st: St, t, rotate, swap_req, gate_swap=None, gate_target=None):
        """rotate:(B,) bool; swap_req:(B,N-1) bool indexed by FRONT position; gate_swap:(B,) bool, gate_target:(B,) plane id."""
        B, N = st.pos.shape
        ar = jnp.arange(B)
        pos = st.pos
        # pair swaps with greedy conflict resolution
        acc = []
        prev = jnp.zeros((B,), bool)
        for p in range(N - 1):
            a = swap_req[:, p] & ~prev & bool(self.pair_ok_np[p])
            acc.append(a)
            prev = a
        acc = jnp.stack(acc, 1) if N > 1 else jnp.zeros((B, 0), bool)
        zf = jnp.zeros((B, 1), bool)
        acc_front = jnp.concatenate([acc, zf], 1)       # (B,N) by position
        acc_rear = jnp.concatenate([zf, acc], 1)
        fr = acc_front[ar[:, None], pos]
        re = acc_rear[ar[:, None], pos]
        pos_pair = pos + fr.astype(pos.dtype) - re.astype(pos.dtype)
        part_pair = fr | re
        new_pos, part = pos_pair, part_pair
        if gate_swap is not None:
            leader = jnp.argmax(pos == 0, axis=1)
            tgt = gate_target
            pl = pos[ar, leader]
            pt = pos[ar, tgt]
            pos_g = pos.at[ar, leader].set(pt).at[ar, tgt].set(pl)
            hit = (jnp.arange(N)[None, :] == leader[:, None]) | (jnp.arange(N)[None, :] == tgt[:, None])
            new_pos = jnp.where(gate_swap[:, None], pos_g, new_pos)
            part = jnp.where(gate_swap[:, None], hit, part)
        pos_r = (pos - 1) % N
        new_pos = jnp.where(rotate[:, None], pos_r, new_pos)
        part = jnp.where(rotate[:, None], True, part)
        part = jnp.where(rotate[:, None], jnp.ones_like(part), part & st.alive[:, None])
        did = part.any(axis=1)
        t32 = jnp.asarray(t, jnp.int32)
        last_part = jnp.where(part, t32, st.last_part)
        last_swap = jnp.where(did, t32, st.last_swap)
        return st._replace(pos=new_pos, last_part=last_part, last_swap=last_swap, nswap=st.nswap + did), part

    # ------------------------------------------------------------------ rule policies (for tests / in-JAX baselines)
    def rule_decision(self, st, inst, t, rule, p1, p2=0.0, p3=0.0):
        B, N = st.pos.shape
        ar = jnp.arange(B)
        zero_r = jnp.zeros((B,), bool)
        zero_s = jnp.zeros((B, max(N - 1, 0)), bool)
        if rule == "fixed":
            return zero_r, zero_s
        if rule == "rot":
            return jnp.broadcast_to((t % p1 == 0) & (t > 0), (B,)), zero_s
        if rule == "local":
            rdv = self.rd(st, inst, p3)
            occ = self.occ(st)
            rdp = jnp.take_along_axis(rdv, occ, 1)               # by position
            mv = jnp.asarray(self.mvec_np, fdt())
            mp, mq = mv[:-1], mv[1:]
            fr, rr = rdp[:, :-1], rdp[:, 1:]
            go = jnp.where((mp > mq)[None, :], fr + p2 < rr, rr + p2 < fr)
            parity = (t // p1) % 2
            on = (jnp.arange(N - 1) % 2 == parity) & ((t % p1 == 0) & (t > 0))
            go = go & on[None, :] & st.alive[:, None]
            return zero_r, go
        raise ValueError(rule)

    # ------------------------------------------------------------------ observations
    def feats(self, st, inst):
        """Per-plane scalar features (B,N,5): rem/B0, rd_nom/B0, rd_est/B0, (k-1)*5, dwell/1000 (own)."""
        return None

    def obs(self, st: St, inst: Inst, t):
        """Actor observation (B,N,D): L3-allowed local information only (own + chain neighbours p-1,p+1)."""
        B, N = st.pos.shape
        ar = jnp.arange(B)
        pos = st.pos
        occ = self.occ(st)
        rem = st.rem / B0
        rdn = st.rem / inst.k / B0
        rde = st.rem / (inst.k * st.rho) / B0
        kk = (inst.k - 1.0) * 5.0
        dw = jnp.clip((t - st.last_part).astype(fdt()) / 1000.0, 0.0, 1.5)
        base = jnp.stack([rem, rdn, rde, kk, dw], -1)                              # (B,N,5)
        mv = jnp.asarray(self.mvec_np, fdt())
        mpad = jnp.concatenate([mv[:1], mv, mv[-1:]])                                # m at p-1.. p+1 (edge-padded)
        feats = [jax.nn.one_hot(pos, N, dtype=fdt()), mv[pos][..., None]]
        feats.append((mpad[pos + 2] - mv[pos])[..., None])                           # m_{p+1} - m_p
        feats.append((mv[pos] - mpad[pos])[..., None])                               # m_p - m_{p-1}
        parity = (t // DT) % 2
        feats.append(jnp.broadcast_to(jnp.asarray(parity, fdt()), (B, N, 1)))
        # self + neighbours
        selfb = base
        for off in (-1, 1):
            q = pos + off
            ok = (q >= 0) & (q <= N - 1)
            qc = jnp.clip(q, 0, N - 1)
            nb_plane = jnp.take_along_axis(occ, qc, 1)
            nbf = jnp.take_along_axis(base, nb_plane[..., None], 1)
            nbf = jnp.where(ok[..., None], nbf, 0.0)
            feats.append(ok[..., None].astype(fdt()))
            feats.append(nbf)
            # relative remaining distance (the quantity the local rule compares), scaled so delta=2..20 -> 0.04..0.4
            feats.append(jnp.where(ok, (nbf[..., 2] - selfb[..., 2]) * 20.0, 0.0)[..., None])
            feats.append(jnp.where(ok, (nbf[..., 1] - selfb[..., 1]) * 20.0, 0.0)[..., None])
        feats.append(selfb)
        clock = jnp.broadcast_to(jnp.asarray(t / 1000.0, fdt()), (B, N, 1))
        feats.append(clock)
        if self.control:
            rdl = jnp.where(pos == 0, -jnp.inf, rde)
            mx = rdl.max(axis=1)
            lead = jnp.where(pos == 0, rde, -jnp.inf).max(axis=1)
            feats.append(jnp.broadcast_to(((mx - lead) * 20.0)[:, None, None], (B, N, 1)))
            feats.append(jnp.broadcast_to((jnp.clip((t - st.last_swap).astype(fdt()), 0, 1500) / 1000.0)[:, None, None], (B, N, 1)))
        o = jnp.concatenate(feats, -1)
        return o

    def cobs(self, st: St, inst: Inst, t):
        """Centralised critic input: occupants listed by position (rem, rd_est, rho, k-1, dwell), clock, last-swap dwell, parity."""
        B, N = st.pos.shape
        occ = self.occ(st)
        rem = st.rem / B0
        rde = st.rem / (inst.k * st.rho) / B0
        dw = jnp.clip((t - st.last_part).astype(fdt()) / 1000.0, 0.0, 1.5)
        base = jnp.stack([rem, rde, st.rho, (inst.k - 1.0) * 5.0, dw], -1)
        bp = jnp.take_along_axis(base, occ[..., None], 1).reshape(B, N * 5)
        extra = jnp.stack([jnp.broadcast_to(jnp.asarray(t / 1000.0, fdt()), (B,)),
                           jnp.clip((t - st.last_swap).astype(fdt()), 0, 1500) / 1000.0,
                           jnp.broadcast_to(jnp.asarray((t // DT) % 2, fdt()), (B,))], -1)
        return jnp.concatenate([bp, extra], -1)

    def masks(self, st: St, t):
        """(B,N,nact) bool valid actions."""
        B, N = st.pos.shape
        pos = st.pos
        okp = jnp.asarray(np.concatenate([self.pair_ok_np, [False]]))[pos]            # swap_rear valid
        dec = (t > 0)
        m_none = jnp.ones((B, N), bool)
        m_swap = okp & st.alive[:, None] & dec
        m_rot = (pos == 0) & st.alive[:, None] & dec
        ms = [m_none, m_swap, m_rot]
        if self.control:
            ms.append((pos == 0) & st.alive[:, None] & dec)
        return jnp.stack(ms, -1)

    def decode(self, st, inst, act, t):
        """act:(B,N) int per plane -> (rotate, swap_req_by_front_position, gate_swap, gate_target)."""
        B, N = st.pos.shape
        ar = jnp.arange(B)
        occ = self.occ(st)
        act_pos = jnp.take_along_axis(act, occ, 1)                  # action of the plane at each position
        rotate = act_pos[:, 0] == 2
        swap_req = (act_pos[:, :-1] == 1)
        if self.control:
            gs = act_pos[:, 0] == 3
            rde = self.rd(st, inst, ALPHA_ENV)
            tgt = jnp.argmax(jnp.where(st.pos == 0, -jnp.inf, rde), axis=1)
            return rotate, swap_req, gs, tgt
        return rotate, swap_req, None, None

    # ------------------------------------------------------------------ one decision block
    def block(self, st, inst, t0, decision, noise_fn, key):
        """decision=(rotate, swap_req, gate_swap, gate_target) applied at step t0 (no-op at t0 == 0)."""
        rotate, swap_req, gs, tgt = decision
        st, part = self.apply_decision(st, t0, rotate, swap_req, gs, tgt)
        r = jnp.zeros_like(st.dist)

        def body(c, j):
            st_, r_, key_ = c
            key_, sub = jax.random.split(key_)
            t = t0 + j
            wt, st_ = noise_fn(st_, t, sub)
            p = jnp.where(j == 0, part, jnp.zeros_like(part))
            st_, inc = self.micro(st_, inst, t, wt, p)
            return (st_, r_ + inc, key_), None

        (st, r, key), _ = lax.scan(body, (st, r, key), jnp.arange(DT))
        return st, r, key


def table_noise(w):
    """noise from a precomputed table w:(TMAX,B,N) (or None -> ones)."""
    def fn(st, t, key):
        if w is None:
            return jnp.ones_like(st.rem), st
        return w[t], st
    return fn


def gen_noise_fn(env: Env):
    def fn(st, t, key):
        wt = env.wt_from_eps(st.eps)
        return wt, st._replace(eps=env.eps_next(st.eps, key))
    return fn


def run_rule(env: Env, inst: Inst, rule, p1, p2=0.0, p3=0.0, w=None, init="identity", nb=NB):
    """Run a rule-as-policy to completion; returns (range/B0 (B,), final state)."""
    st = env.init_state(inst)
    if init == "sorted":
        raise NotImplementedError
    nf = table_noise(None if w is None else jnp.asarray(w, fdt()))
    key = jax.random.PRNGKey(0)

    def body(c, b):
        st, key = c
        t0 = b * DT
        rot, req = env.rule_decision(st, inst, t0, rule, p1, p2, p3)
        gs = jnp.zeros_like(rot) if env.control else None
        tg = jnp.zeros(rot.shape, jnp.int32) if env.control else None
        st, r, key = env.block(st, inst, t0, (rot, req, gs, tg), nf, key)
        return (st, key), None

    (st, _), _ = lax.scan(body, (st, key), jnp.arange(nb))
    return st.dist / B0, st
