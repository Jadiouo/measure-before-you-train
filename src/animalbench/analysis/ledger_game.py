"""Ledger-state small game (spec v0.4 §4.3 #15c): is a ledger-based ROTATION an equilibrium under swap_rule="mutual"?

NEW FILE; reuses analysis.small_game.Game only for its swap-resolution tables (RT_*), slot powers p and swap costs.

Game (mutual rule, kappa = 0, one decision = delta_h = 2 s, bird-indexed state)
  state = (slot of every bird, lock of every bird, ctr, u)
    u_i : public ledger = lead decisions of bird i (+ fines), normalised to min_i u_i = 0 and clipped to [0, Lc].
  reward of bird i : -p[new slot]*dh - cost_i(swap)           (identical to small_game)
  ledger update    : the bird in slot 0 AFTER the swap gets u += 1 (no credit while it overstays a READY hand-over,
                     unless credit_overstay=True -- with credit the exchange is exactly 1:1 and only an
                     indifference equilibrium);  a bird that REFUSES a ready hand-over gets u -= lam
                     (then normalise).  Refusal = the bird in front asked BACK towards a ready edge, both unlocked,
                     and the bird behind did not ask FWD.  (lam is the 'fine'; lam = 0 is the pure time ledger.)
Ledger strategy sigma (stationary, public information only)
  an edge (front f, back e) is READY iff u[bird f] - u[bird e] >= theta (the front bird is ahead by theta decisions) and
  both birds are unlocked and (if f is the apex) e is the apex's current retreat target (depends on ctr).
  front bird of a ready edge asks BACK, back bird asks FWD ("the one who is ahead steps back, the one who is behind
  takes over"); everybody else stays.  If a bird is asked both ways, FWD wins.
Solution concept
  * state space = all states reachable from the fresh start under ARBITRARY joint actions (these are exactly the
    subgames of the game started at the fresh state; BFS over all 3^N action profiles x coin).
  * V_i^sigma solved exactly (discount gamma) on that set; one-shot-deviation gap
        gap(s, i) = max_a Q_i(s, a | others follow sigma, then sigma) - V_i(s).
    sigma is a subgame-perfect equilibrium of the discounted game iff max gap <= 0 (one-shot deviation principle,
    continuation payoff bounded).  We report eps = max gap (P_solo*s) over (a) all subgames, (b) on-path states only.
  * average-reward version: gamma = 1 - 5e-4 (gap reported in V units, i.e. an immediate loss/gain).
"""
from __future__ import annotations

import itertools
import time

import numpy as np
import scipy.sparse as sp
import scipy.sparse.csgraph as csg
import scipy.sparse.linalg as spla

from .small_game import Game, front

S_, B_, F_ = 0, 1, 2


def _scaled(cost, m):
    return {e: np.asarray(c) * m for e, c in cost.items()}


class LedgerGame:
    def __init__(self, N, p, cost, lock_dec=4, theta=6, lam=2, Lc=None, dh=2.0, cmult=1.0, credit_overstay=False,
                 apex_only=False):
        self.N, self.theta, self.lam, self.dh = N, int(theta), int(lam), dh
        self.credit_overstay = credit_overstay
        self.apex_only = apex_only        # ledger rule only on the apex edge (deep slots never rearranged)
        self.Lc = int(Lc if Lc is not None else theta + lam + 2)
        self.g = Game(N, p, _scaled(cost, cmult), "mutual", dh=dh, lock_dec=lock_dec)
        self.p = np.asarray(p, float)
        self.nL = lock_dec
        self.pw = self.nL ** np.arange(N)
        self.uw = (self.Lc + 1) ** np.arange(N)
        self.sw_ = N ** np.arange(N)
        self.ar = np.arange(N)
        # states are stored as int8 arrays (slot, lock, u) + int8 ctr

    # ------------------------------------------------------------------ encoding
    def encode(self, slot, lock, ctr, u):
        sc = (slot.astype(np.int64) * self.sw_).sum(1)
        lc = (lock.astype(np.int64) * self.pw).sum(1)
        uc = (u.astype(np.int64) * self.uw).sum(1)
        nLc, nU = self.nL ** self.N, (self.Lc + 1) ** self.N
        return ((sc * nLc + lc) * 2 + ctr.astype(np.int64)) * nU + uc

    def decode(self, code):
        N = self.N
        nLc, nU = self.nL ** N, (self.Lc + 1) ** N
        uc = code % nU
        r = code // nU
        ctr = r % 2
        r = r // 2
        lc = r % nLc
        sc = r // nLc
        dig = lambda x, base: np.stack([(x // base ** i) % base for i in range(N)], 1).astype(np.int8)
        return dig(sc, N), dig(lc, self.nL), ctr.astype(np.int8), dig(uc, self.Lc + 1)

    def fresh(self):
        N = self.N
        return (np.arange(N, dtype=np.int8)[None], np.zeros((1, N), np.int8), np.zeros(1, np.int8),
                np.zeros((1, N), np.int8))

    # ------------------------------------------------------------------ occupancy helpers
    def _occ(self, slot):
        occ = np.empty_like(slot)
        np.put_along_axis(occ, slot.astype(np.int64), np.broadcast_to(self.ar.astype(slot.dtype), slot.shape), 1)
        return occ

    def apex_target(self, ctr):
        pref = 1 + ctr.astype(np.int64)
        return np.where(pref < self.N, pref, 2 - ctr.astype(np.int64))

    # ------------------------------------------------------------------ the strategy
    def ready_edges(self, slot, lock, ctr, u, occ=None):
        """list over edges e=1..N-1 of (bf, be, ready(B,))."""
        occ = self._occ(slot) if occ is None else occ
        tgt = self.apex_target(ctr)
        out = []
        for e in range(1, self.N):
            f = front(e)
            if self.apex_only and f != 0:
                continue
            bf, be = occ[:, f].astype(np.int64), occ[:, e].astype(np.int64)
            rows = np.arange(len(bf))
            ok = (u[rows, bf].astype(int) - u[rows, be].astype(int) >= self.theta) & (lock[rows, bf] == 0) & \
                 (lock[rows, be] == 0)
            if f == 0:
                ok &= tgt == e
            out.append((e, bf, be, ok))
        return out

    def sigma(self, slot, lock, ctr, u):
        B = len(slot)
        act = np.zeros((B, self.N), np.int8)
        rows = np.arange(B)
        occ = self._occ(slot)
        ed = self.ready_edges(slot, lock, ctr, u, occ)
        for e, bf, be, ok in ed:
            cur = act[rows, bf]
            act[rows, bf] = np.where(ok & (cur == S_), B_, cur)
        for e, bf, be, ok in ed:
            act[rows, be] = np.where(ok, F_, act[rows, be])
        return act

    # ------------------------------------------------------------------ one decision (batch)
    def step(self, slot, lock, ctr, u, act, coin):
        """act (B,N) per-bird requested actions, coin (B,) 0/1.  Returns slot2, lock2, ctr2, u2, reward (B,N)."""
        g, N = self.g, self.N
        rows = np.arange(len(slot))[:, None]
        occ = self._occ(slot)
        act_slot = np.take_along_axis(act, occ.astype(np.int64), 1).astype(np.int64)
        lock_slot = np.take_along_axis(lock > 0, occ.astype(np.int64), 1)
        ac = (act_slot * 3 ** self.ar).sum(1)
        lm = (lock_slot * 2 ** self.ar).sum(1)
        ctr_i, coin_i = ctr.astype(np.int64), coin.astype(np.int64)
        ns = g.RT_ns[ac, lm, ctr_i, coin_i]
        sw = g.RT_sw[ac, lm, ctr_i, coin_i]
        cs = g.RT_cost[ac, lm, ctr_i, coin_i]
        ctr2 = g.RT_ctr[ac, lm, ctr_i, coin_i]
        sl = slot.astype(np.int64)
        slot2 = np.take_along_axis(ns, sl, 1)
        swb = np.take_along_axis(sw, sl, 1)
        csb = np.take_along_axis(cs, sl, 1)
        lock2 = np.where(swb, self.nL - 1, np.maximum(lock - 1, 0))
        r = -self.p[slot2] * self.dh - csb
        # ledger: refusal fines (computed from pre-state + actual actions) and lead increment
        u2 = u.astype(np.int16).copy()
        rr = np.arange(len(slot))
        apex1 = occ[:, 0].astype(np.int64)
        overstay = np.zeros(len(slot), bool)
        for e, bf, be, ok in self.ready_edges(slot, lock, ctr, u, occ):
            refuse = ok & (act[rr, bf] == B_) & (act[rr, be] != F_)
            u2[rr, be] -= self.lam * refuse
            if e >= 1 and front(e) == 0:
                overstay |= ok
        apex2 = np.argmax(slot2 == 0, axis=1)
        credit = 1 - ((overstay & (apex2 == apex1)) if not self.credit_overstay else 0)
        u2[rr, apex2] += credit
        if self.apex_only and self.N > 3:
            # birds in slots >= 3 never lead (apex-edge-only rule): keep them out of the normalisation
            elig = slot2 <= 2
            u2 = np.where(elig, u2, 10_000)
            u2 -= u2.min(1, keepdims=True)
            u2 = np.where(elig, u2, 0)
        else:
            u2 -= u2.min(1, keepdims=True)
        u2 = np.clip(u2, 0, self.Lc)
        return slot2.astype(np.int8), lock2.astype(np.int8), ctr2.astype(np.int8), u2.astype(np.int8), r

    # ------------------------------------------------------------------ state space: BFS under arbitrary joint actions
    def enumerate_states(self, max_states=3_000_000, verbose=False):
        N = self.N
        s0 = self.fresh()
        codes = self.encode(*s0)
        known = np.sort(codes)
        frontier = codes
        profiles = np.array(list(itertools.product(range(3), repeat=N)), np.int8)
        t0 = time.time()
        while len(frontier):
            sl, lk, ct, uu = self.decode(frontier)
            new = []
            for prof in profiles:
                act = np.broadcast_to(prof, (len(frontier), N))
                for coin in (0, 1):
                    s2 = self.step(sl, lk, ct, uu, act, np.full(len(frontier), coin))
                    new.append(self.encode(*s2[:4]))
            new = np.unique(np.concatenate(new))
            new = new[~np.isin(new, known, assume_unique=True)]
            known = np.union1d(known, new)
            frontier = new
            if verbose:
                print(f"  bfs: {len(known)} states, frontier {len(frontier)}, {time.time() - t0:.0f}s", flush=True)
            if len(known) > max_states:
                raise MemoryError(f"state space > {max_states}")
        self.codes = known
        self.nS = len(known)
        self.state = self.decode(known)
        return self.nS

    def index_of(self, code):
        i = np.searchsorted(self.codes, code)
        i = np.minimum(i, self.nS - 1)
        assert np.all(self.codes[i] == code), "successor outside the enumerated state space"
        return i

    # ------------------------------------------------------------------ chain under sigma
    def sigma_chain(self, chunk=400_000):
        """P_sigma (csr, 0.5 per coin), r_sigma (nS, N), sigma actions (nS,N)."""
        sl, lk, ct, uu = self.state
        nS, N = self.nS, self.N
        act = np.zeros((nS, N), np.int8)
        rews = np.zeros((nS, N))
        rows, cols = [], []
        for c0 in range(0, nS, chunk):
            sel = slice(c0, min(nS, c0 + chunk))
            a = self.sigma(sl[sel], lk[sel], ct[sel], uu[sel])
            act[sel] = a
            for coin in (0, 1):
                s2 = self.step(sl[sel], lk[sel], ct[sel], uu[sel], a, np.full(len(a), coin))
                rews[sel] += 0.5 * s2[4]
                rows.append(np.arange(nS)[sel])
                cols.append(self.index_of(self.encode(*s2[:4])))
        P = sp.coo_matrix((np.full(2 * nS, 0.5), (np.concatenate(rows), np.concatenate(cols))), shape=(nS, nS)).tocsr()
        self.act_sigma, self.P, self.r = act, P, rews
        return P, rews, act

    def solve_values(self, gamma, tol=1e-11):
        M = (sp.identity(self.nS, format="csr") - gamma * self.P).tocsc()
        lu = spla.splu(M)
        V = lu.solve(self.r)
        res = np.abs(V - self.r - gamma * (self.P @ V)).max()
        return V, float(res)

    def one_shot_gaps(self, V, gamma, chunk=400_000):
        """gap[s, i] = max_a Q_i(s,a) - V_i(s)  (others follow sigma).  Also returns the argmax action."""
        sl, lk, ct, uu = self.state
        nS, N = self.nS, self.N
        gap = np.full((nS, N), -np.inf)
        best = np.zeros((nS, N), np.int8)
        for c0 in range(0, nS, chunk):
            sel = slice(c0, min(nS, c0 + chunk))
            a0 = self.act_sigma[sel]
            for i in range(N):
                for a in range(3):
                    act = a0.copy()
                    act[:, i] = a
                    q = 0.0
                    for coin in (0, 1):
                        s2 = self.step(sl[sel], lk[sel], ct[sel], uu[sel], act, np.full(len(act), coin))
                        q = q + 0.5 * (s2[4][:, i] + gamma * V[self.index_of(self.encode(*s2[:4])), i])
                    d = q - V[sel, i]
                    better = d > gap[sel, i]
                    gap[sel, i] = np.where(better, d, gap[sel, i])
                    best[sel, i] = np.where(better, a, best[sel, i])
        return gap, best

    def on_path(self):
        """boolean mask of states reachable from the fresh start under sigma."""
        s = int(self.index_of(self.encode(*self.fresh()))[0])
        order = csg.breadth_first_order(self.P, s, directed=True, return_predecessors=False)
        m = np.zeros(self.nS, bool)
        m[order] = True
        return m

    def avg_cost(self, mask, n=20000, burn=2000):
        """Cesaro mean of -reward per bird per decision from the fresh start (on-path sub-chain)."""
        idx = np.flatnonzero(mask)
        Pm = self.P[idx][:, idx].tocsr()
        r = self.r[idx]
        d = np.zeros(len(idx))
        d[np.searchsorted(idx, int(self.index_of(self.encode(*self.fresh()))[0]))] = 1.0
        PT = Pm.T.tocsr()
        tot = np.zeros(self.N)
        for t in range(burn + n):
            if t >= burn:
                tot += d @ r
            d = PT @ d
        return -tot / n

    # ------------------------------------------------------------------ one-call analysis
    def analyse(self, gamma, tol=1e-7, verbose=False):
        if not hasattr(self, "P"):
            self.enumerate_states(verbose=verbose)
            self.sigma_chain()
        V, res = self.solve_values(gamma)
        gap, best = self.one_shot_gaps(V, gamma)
        gmax = gap.max(1)
        op = self.on_path()
        # 'unsaturated' subgames: ledger not at the clip Lc (at the clip a fine cannot be recorded = finite-memory
        # artefact of the abstraction, reached only after a long stall)
        unsat = self.state[3].max(1) < self.Lc

        def worst(mask):
            idx = np.flatnonzero(mask)
            w = int(idx[np.argmax(gmax[idx])])
            b = int(np.argmax(gap[w]))
            return dict(slot=self.state[0][w].tolist(), lock=self.state[1][w].tolist(), ctr=int(self.state[2][w]),
                        u=self.state[3][w].tolist(), bird=b, action=int(best[w, b]), sigma=self.act_sigma[w].tolist())
        eps_all, eps_path, eps_unsat = float(gmax.max()), float(gmax[op].max()), float(gmax[unsat].max())
        return dict(gamma=gamma, n_states=int(self.nS), n_on_path=int(op.sum()), residual=res,
                    eps_all=eps_all, eps_unsat=eps_unsat, eps_on_path=eps_path,
                    spe=bool(eps_all <= tol), spe_unsat=bool(eps_unsat <= tol), spe_on_path=bool(eps_path <= tol),
                    frac_states_violating=float((gmax > tol).mean()),
                    worst=worst(np.ones(self.nS, bool)), worst_unsat=worst(unsat), worst_path=worst(op))


# ---------------------------------------------------------------------------------------------- cost comparisons
def fixed_leader_cost(p, dh=2.0):
    """Per-bird cost per decision of the mutual fixed-leader profile (no swaps): mean over birds, leader, follower."""
    p = np.asarray(p, float)
    return dict(mean=float(p.mean() * dh), leader=float(p[0] * dh), follower=float(p[1:].mean() * dh))


def simulate_kappa(lg: LedgerGame, kappa, steps=6000, burn=1500, seeds=4, phi0=0.94, tau_f=60.0):
    """Fatigue-extended COST of the ledger profile (not re-solved): per-bird phi = EMA of power (p + swap transit),
    cost = power*(1+kappa*max(0,phi-phi0))*dh (same as scripts/sweep_poa.py).  Returns mean per-bird cost per decision."""
    N, dh = lg.N, lg.dh
    alpha = 1 - np.exp(-dh / tau_f)
    tot = []
    for sd in range(seeds):
        rng = np.random.default_rng(sd)
        slot, lock, ctr, u = (x.copy() for x in lg.fresh())
        phi = lg.p[slot[0]].copy()
        acc = cnt = 0
        for t in range(steps):
            a = lg.sigma(slot, lock, ctr, u)
            s2 = lg.step(slot, lock, ctr, u, a, np.array([rng.integers(2)]))
            slot, lock, ctr, u, r = s2
            pw = -r[0] / dh
            phi = phi + alpha * (pw - phi)
            c = pw * (1 + kappa * np.maximum(0.0, phi - phi0)) * dh
            if t >= burn:
                acc += c.mean(); cnt += 1
        tot.append(acc / cnt)
    return float(np.mean(tot))


def fixed_leader_kappa_cost(p, kappa, dh=2.0, phi0=0.94):
    p = np.asarray(p, float)
    return float((p * (1 + kappa * np.maximum(0.0, p - phi0))).mean() * dh)
