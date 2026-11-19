"""Analytically tractable small game (spec v0.3 #15): N birds on abstract V slots, actions {stay, back, fwd},
unilateral or mutual swap rule -- the null prediction for learning agents.

Model (everything but the abstraction is copied from the environment: scenarios/geese/swap.py)
  * slot s in 0..N-1; front(s)=0 for s<=2 else s-2; BACK of slot s targets slot s+2 (apex: slot 1+ctr, ctr alternates
    between the arms after every apex swap); FWD of slot s targets front(s).  Illegal / locked requests = stay.
  * edge e=(front(e), e) swaps if (unilateral) either side asks or (mutual) both ask, neither is locked; edges are
    processed in order (coin ? 1,2 : 2,1, 3, 4, ...) and an edge is dropped if one of its slots is already used.
  * a swapped bird gets lock = lock_dec(4), every lock is decremented at the end of the decision, so a decision sees
    locks 0..lock_dec-1 (0..3); locked birds are forced to stay.
  * reward of the bird that ends in slot s' : -p[s'] * delta_h  -  sum over accepted edges of cost[edge][s_before]
    with p, cost MEASURED in the real environment (analysis/slot_params.py -> calib/out/small_game_params.json).
  * state = (lock of the occupant of every slot, ctr); a "deviator" MDP additionally marks one slot (the bird that
    optimises); the others play a symmetric policy pi(a | slot) at lock 0 (identical birds, identity irrelevant).
Solution method
  * deviator MDP: policy iteration (exact sparse linear solves) with discount gamma = 1 - 5e-4 (Abel approximation of the
    average-reward criterion, error O(1-gamma)); all reported gains are per decision in P_solo*s per (1-gamma) units.
  * pure symmetric profiles are enumerated; a profile is an equilibrium if the best full-information deviation does not
    gain more than `tol` at any state reachable from a fresh start (all locks 0).
  * mixed/other equilibria: damped (0.5) symmetric best-response iteration on pi(a|slot); the best response is the
    argmax of the occupancy-weighted one-step Q of the self-play value function.
"""
from __future__ import annotations

import itertools
import json
import os
from functools import lru_cache

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

S_, B_, F_ = 0, 1, 2
PARAMS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "calib", "out",
                           "small_game_params.json")


def front(s):
    return 0 if s <= 2 else s - 2


def load_params(N, path=PARAMS_PATH):
    d = json.load(open(path))[str(N)]
    cost = {int(k): np.array(e["cost"], float) for k, e in d["edges"].items()}
    return np.array(d["p"], float), cost


class Game:
    def __init__(self, N, p, cost, rule="unilateral", dh=2.0, lock_dec=4, gamma=1 - 5e-4):
        assert rule in ("unilateral", "mutual")
        self.N, self.p, self.cost, self.rule, self.dh = N, np.asarray(p, float), cost, rule, dh
        self.lock_dec, self.nL, self.gamma = lock_dec, lock_dec, gamma
        self.legal = np.zeros((N, 3), bool)
        for s in range(N):
            self.legal[s] = [True, (s == 0 and N >= 2) or (s >= 1 and s + 2 < N), s >= 1]
        self.nLc = self.nL ** N
        self.nS = self.nLc * 2 * N
        self.Larr = np.stack(np.unravel_index(np.arange(self.nLc), (self.nL,) * N), 1)[:, ::-1]   # lockcode digit s = L[s]
        self._build_resolution()
        self._build_structure()

    # ------------------------------------------------------------------ resolution (mirror of resolve_swaps)
    def apex_target(self, ctr):
        pref = 1 + ctr
        return pref if pref < self.N else 2 - ctr

    def resolve(self, act, locked, ctr, coin):
        """act: (N,) requested actions, locked: (N,) bool.  Returns ns (new slot of the bird that was in slot s),
        swapped (N,), new ctr, cost per pre-swap slot."""
        N = self.N
        eff = np.array([a if (not locked[s] and self.legal[s, a]) else S_ for s, a in enumerate(act)])
        ok = np.zeros(N, bool)
        for e in range(1, N):
            f = front(e)
            pf = eff[f] == B_ and (self.apex_target(ctr) if f == 0 else f + 2) == e
            pb = eff[e] == F_
            cand = (pf and pb) if self.rule == "mutual" else (pf or pb)
            ok[e] = cand and not locked[e] and not locked[f]
        order = ([1, 2] if coin else [2, 1]) if N >= 3 else [1]
        order = order + list(range(3, N))
        used = np.zeros(N, bool)
        acc = []
        for e in order:
            if e >= N:
                continue
            f = front(e)
            if ok[e] and not used[f] and not used[e]:
                acc.append(e)
                used[f] = used[e] = True
        ns = np.arange(N)
        sw = np.zeros(N, bool)
        cost = np.zeros(N)
        for e in acc:
            f = front(e)
            ns[f], ns[e] = e, f
            sw[f] = sw[e] = True
            cost += self.cost[e]
        ctr_new = ctr
        if any(e in (1, 2) for e in acc):
            ctr_new = 1 if 1 in acc else 0
        return ns, sw, ctr_new, cost

    def _build_resolution(self):
        N = self.N
        nA, nM = 3 ** N, 2 ** N
        self.RT_ns = np.zeros((nA, nM, 2, 2, N), np.int8)
        self.RT_sw = np.zeros((nA, nM, 2, 2, N), bool)
        self.RT_ctr = np.zeros((nA, nM, 2, 2), np.int8)
        self.RT_cost = np.zeros((nA, nM, 2, 2, N))
        for ac in range(nA):
            act = [(ac // 3 ** s) % 3 for s in range(N)]
            for lm in range(nM):
                locked = np.array([(lm >> s) & 1 for s in range(N)], bool)
                for ctr in range(2):
                    for coin in range(2):
                        ns, sw, cn, c = self.resolve(act, locked, ctr, coin)
                        self.RT_ns[ac, lm, ctr, coin], self.RT_sw[ac, lm, ctr, coin] = ns, sw
                        self.RT_ctr[ac, lm, ctr, coin], self.RT_cost[ac, lm, ctr, coin] = cn, c

    # ------------------------------------------------------------------ transition structure (policy independent)
    def _build_structure(self):
        N, nL, nLc = self.N, self.nL, self.nLc
        T = 3 ** (N - 1)
        self.T = T
        self.others = [[s for s in range(N) if s != m] for m in range(N)]
        self.digits = np.stack([(np.arange(T) // 3 ** j) % 3 for j in range(N - 1)], 1)       # (T, N-1)
        lockmask = ((self.Larr > 0) * (1 << np.arange(N))).sum(1)                              # (nLc,)
        self.lockmask = lockmask
        rows, cols, rew, tup, lmk, mm, aa = [], [], [], [], [], [], []
        for m in range(N):
            for a in range(3):
                # effective deviator action per lock state
                a_eff = np.where((self.Larr[:, m] == 0) & self.legal[m, a], a, S_)            # (nLc,)
                # action code of every (lockcode, tuple)
                code = np.zeros((nLc, T), np.int64)
                for j, s in enumerate(self.others[m]):
                    code += self.digits[None, :, j] * 3 ** s
                code += a_eff[:, None] * 3 ** m
                for ctr in range(2):
                    for coin in range(2):
                        ns = self.RT_ns[code, lockmask[:, None], ctr, coin].astype(np.int64)   # (nLc, T, N)
                        sw = self.RT_sw[code, lockmask[:, None], ctr, coin]
                        cn = self.RT_ctr[code, lockmask[:, None], ctr, coin].astype(np.int64)
                        cs = self.RT_cost[code, lockmask[:, None], ctr, coin]
                        newL = np.zeros((nLc, T, N), np.int64)
                        base = np.maximum(self.Larr[:, None, :] - 1, 0)
                        val = np.where(sw, nL - 1, base)
                        np.put_along_axis(newL, ns, val, axis=2)
                        ncode = (newL * (nL ** np.arange(N))).sum(2)
                        m2 = ns[:, :, m]
                        nxt = ncode + nLc * (cn + 2 * m2)
                        r = -self.p[m2] * self.dh - cs[:, :, m]
                        cur = np.arange(nLc)[:, None] + nLc * (ctr + 2 * m)
                        rows.append(np.broadcast_to(cur, nxt.shape).ravel().astype(np.int32))
                        cols.append(nxt.ravel().astype(np.int32))
                        rew.append(r.ravel())
                        tup.append(np.broadcast_to(np.arange(T)[None, :], nxt.shape).ravel().astype(np.int32))
                        lmk.append(np.broadcast_to(lockmask[:, None], nxt.shape).ravel().astype(np.int32))
                        mm.append(np.full(nxt.size, m, np.int8))
                        aa.append(np.full(nxt.size, a, np.int8))
        self.s_row = np.concatenate(rows); self.s_col = np.concatenate(cols); self.s_rew = np.concatenate(rew)
        self.s_tup = np.concatenate(tup); self.s_lm = np.concatenate(lmk)
        self.s_m = np.concatenate(mm); self.s_a = np.concatenate(aa)
        # the coin makes every entry appear with weight 1/2 (2 coin values); entries are duplicated, weights 0.5

    # ------------------------------------------------------------------ policy dependent pieces
    def _tuple_probs(self, pi):
        """PT[lockmask, m, tuple] = prod over other slots of q_s(a_s) (locked -> stay)."""
        N, T = self.N, self.T
        PT = np.ones((2 ** N, N, T))
        for lm in range(2 ** N):
            q = np.where(((lm >> np.arange(N)) & 1)[:, None] == 1, np.array([1.0, 0, 0])[None, :], pi)
            for m in range(N):
                pr = np.ones(T)
                for j, s in enumerate(self.others[m]):
                    pr *= q[s][self.digits[:, j]]
                PT[lm, m] = pr
        return PT

    def matrices(self, pi):
        """Per deviator action a: (P_a (nS,nS) csr, r_a (nS,)) of the deviator MDP when the others play pi."""
        PT = self._tuple_probs(pi)
        w = 0.5 * PT[self.s_lm, self.s_m, self.s_tup]
        P, R = [], []
        for a in range(3):
            sel = self.s_a == a
            Pa = sp.coo_matrix((w[sel], (self.s_row[sel], self.s_col[sel])), shape=(self.nS, self.nS)).tocsr()
            ra = np.bincount(self.s_row[sel], weights=w[sel] * self.s_rew[sel], minlength=self.nS)
            P.append(Pa); R.append(ra)
        return P, R

    def solve(self, pi, sigma=None, V0=None, max_iter=100):
        """sigma None -> optimal deviator (policy iteration); sigma (nS,) int -> evaluate that policy.
        Returns V, sigma, residual (Bellman, max-norm)."""
        P, R = self.matrices(pi)
        g = self.gamma
        I = sp.identity(self.nS, format="csc")
        stack = lambda V: np.stack([R[a] + g * (P[a] @ V) for a in range(3)])

        Pcsr = [P[a].tocsr() for a in range(3)]

        def evaluate(sig):
            M = sp.csr_matrix((self.nS, self.nS))
            r = np.zeros(self.nS)
            for a in range(3):
                M = M + sp.diags((sig == a).astype(float)) @ Pcsr[a]
                r += (sig == a) * R[a]
            return spla.spsolve((I - g * M).tocsc(), r)

        if sigma is not None:
            V = evaluate(sigma)
            return V, sigma, float(np.abs(stack(V).max(0) - V).max())
        sig = np.zeros(self.nS, np.int64) if V0 is None else np.argmax(stack(V0), 0)
        V = evaluate(sig)
        for _ in range(max_iter):
            Q = stack(V)
            new = np.where(Q.max(0) > Q[sig, np.arange(self.nS)] + 1e-7, np.argmax(Q, 0), sig)
            if np.array_equal(new, sig):
                break
            sig = new
            V = evaluate(sig)
        return V, sig, float(np.abs(stack(V).max(0) - V).max())

    def self_play_sigma(self, pi_det):
        """sigma(state) for a pure symmetric profile: the marked bird plays pi_det[slot] at lock 0 (else stay)."""
        sigma = np.zeros(self.nS, np.int64)
        for m in range(self.N):
            for ctr in range(2):
                sl = slice(self.nLc * (ctr + 2 * m), self.nLc * (ctr + 2 * m + 1))
                sigma[sl] = np.where(self.Larr[:, m] == 0, pi_det[m], S_)
        return sigma

    def fresh_states(self):
        return np.array([c * self.nLc + 0 for c in range(2 * self.N)])          # lockcode 0 = all unlocked

    def reachable(self, pi, sigma, start):
        P, _ = self.matrices(pi)
        M = sp.csr_matrix((self.nS, self.nS))
        for a in range(3):
            M = M + sp.diags((sigma == a).astype(float)) @ P[a].tocsr()
        seen = np.zeros(self.nS, bool)
        seen[start] = True
        frontier = start
        while len(frontier):
            nxt = np.unique(M[frontier].indices)
            nxt = nxt[~seen[nxt]]
            seen[nxt] = True
            frontier = nxt
        return seen

    def pure_pi(self, pi_det):
        pi = np.zeros((self.N, 3))
        pi[np.arange(self.N), pi_det] = 1.0
        return pi

    def check_pure(self, pi_det):
        """Deviation gap (average-reward units per decision) of a pure symmetric profile over reachable states."""
        pi = self.pure_pi(pi_det)
        sig = self.self_play_sigma(pi_det)
        Vp, _, res1 = self.solve(pi, sigma=sig)
        Vs, sstar, res2 = self.solve(pi, V0=Vp)
        reach = self.reachable(pi, sig, self.fresh_states())
        gap = (Vs - Vp)[reach]
        return dict(gap=float(gap.max() * (1 - self.gamma)), residual=res2, n_reach=int(reach.sum()),
                    Vp=Vp, Vs=Vs, sigma_star=sstar, reach=reach)

    def pure_profiles(self):
        opts = [[a for a in range(3) if self.legal[s, a]] for s in range(self.N)]
        return list(itertools.product(*opts))

    def equilibria(self, tol=1e-9):
        eq = []
        for prof in self.pure_profiles():
            r = self.check_pure(np.array(prof))
            if r["gap"] <= tol:
                eq.append((prof, r["gap"], r["residual"]))
        return eq

    def average_reward(self, pi_det, start_state=None, n=6000):
        """Long-run mean reward per decision of the marked bird when everybody plays the pure profile (Cesaro mean
        of the self-play chain started from a fresh state; error ~ cycle_length/n)."""
        pi = self.pure_pi(np.asarray(pi_det))
        sig = self.self_play_sigma(np.asarray(pi_det))
        P, R = self.matrices(pi)
        M = sp.csr_matrix((self.nS, self.nS)); r = np.zeros(self.nS)
        for a in range(3):
            M = M + sp.diags((sig == a).astype(float)) @ P[a].tocsr(); r += (sig == a) * R[a]
        d = np.zeros(self.nS)
        d[self.nLc * 0 if start_state is None else start_state] = 1.0
        tot = 0.0
        MT = M.T.tocsr()
        for _ in range(n):
            tot += d @ r
            d = MT @ d
        return tot / n

    def apex_retreat_dominance(self):
        """For every pure profile (apex slot included) check whether BACK is an optimal action of the unlocked apex
        at the fresh state (full-information continuation).  Returns (n_profiles, n_B_optimal, n_B_strictly_better)."""
        nB = nS = 0
        profs = self.pure_profiles()
        for prof in profs:
            pi = self.pure_pi(np.array(prof))
            V, sig, _ = self.solve(pi)
            P, R = self.matrices(pi)
            s0 = 0            # slot 0, ctr 0, all unlocked
            Q = np.array([R[a][s0] + self.gamma * (P[a][s0] @ V) for a in range(3)])
            if Q[B_] >= Q.max() - 1e-7:
                nB += 1
            if Q[B_] > Q[S_] + 1e-7:
                nS += 1
        return len(profs), nB, nS

    # ------------------------------------------------------------------ damped best-response iteration
    def occupancy(self, pi, sigma, start_dist, iters=4000):
        P, _ = self.matrices(pi)
        M = sp.csr_matrix((self.nS, self.nS))
        for a in range(3):
            M = M + sp.diags((sigma == a).astype(float)) @ P[a].tocsr()
        d = start_dist.copy()
        occ = np.zeros(self.nS)
        for _ in range(iters):
            occ += d
            d = M.T @ d
        return occ / iters

    def br_iteration(self, pi0=None, damp=0.5, iters=200, tol=1e-9, rng=None):
        N = self.N
        pi = np.where(self.legal, 1.0, 0.0)
        pi = pi / pi.sum(1, keepdims=True) if pi0 is None else pi0.copy()
        start = np.zeros(self.nS)
        start[self.fresh_states()] = 1.0 / len(self.fresh_states())
        hist = []
        for it in range(iters):
            # self-play value of the current mixed pi: the deviator follows pi (stochastic policy) -> evaluate by
            # mixing the three action matrices with weights pi[slot]
            P, R = self.matrices(pi)
            g = self.gamma
            Pm = sp.csr_matrix((self.nS, self.nS)); r = np.zeros(self.nS)
            wa = np.zeros((3, self.nS))
            for m in range(N):
                for ctr in range(2):
                    sl = slice(self.nLc * (ctr + 2 * m), self.nLc * (ctr + 2 * m + 1))
                    unl = self.Larr[:, m] == 0
                    for a in range(3):
                        wa[a, sl] = np.where(unl, pi[m, a], 1.0 if a == S_ else 0.0)
            for a in range(3):
                Pm = Pm + sp.diags(wa[a]) @ P[a].tocsr(); r += wa[a] * R[a]
            V = spla.spsolve((sp.identity(self.nS, format="csc") - g * Pm).tocsc(), r)
            Q = np.stack([R[a] + g * (P[a] @ V) for a in range(3)])
            # occupancy of the self-play chain
            d = start.copy(); occ = np.zeros(self.nS)
            for _ in range(3000):
                occ += d; d = Pm.T @ d
            occ /= 3000
            br = np.zeros((N, 3))
            for m in range(N):
                num = np.zeros(3); den = 0.0
                for ctr in range(2):
                    sl = slice(self.nLc * (ctr + 2 * m), self.nLc * (ctr + 2 * m + 1))
                    wsel = occ[sl] * (self.Larr[:, m] == 0)
                    num += (Q[:, sl] * wsel).sum(1); den += wsel.sum()
                if den <= 1e-12:      # slot never visited unlocked: uniform over fresh states
                    qv = np.array([Q[a, [self.nLc * (c + 2 * m) for c in range(2)]].mean() for a in range(3)])
                else:
                    qv = num / den
                qv = np.where(self.legal[m], qv, -np.inf)
                best = np.flatnonzero(qv >= qv.max() - 1e-9 * max(1.0, abs(qv.max())))
                br[m, best[0] if pi[m, best[0]] >= pi[m, best].max() else best[np.argmax(pi[m, best])]] = 1.0
                # ties: keep the currently most probable among the tied optimal actions
            new = (1 - damp) * pi + damp * br
            resid = float(np.abs(new - pi).max())
            hist.append(resid)
            pi = new
            if resid < tol:
                break
        return pi, hist


# ---------------------------------------------------------------------------------------------- finite horizon
def finite_horizon_advantage(game: Game, others_pi_det, who_slot, first_action, H_dec, ctr=0):
    """Return(first_action then stay) - Return(stay) of the bird in slot `who_slot` over H_dec decisions, all locks 0,
    the other birds playing the pure symmetric profile `others_pi_det` (the deviator plays stay after the first
    decision).  Exact (expectation over the coin)."""
    pi = game.pure_pi(np.asarray(others_pi_det))
    P, R = game.matrices(pi)
    s0 = game.nLc * (ctr + 2 * who_slot) + 0
    stay = np.zeros(game.nS, np.int64)
    # open-loop: deviator always stays after the first decision (locked birds are forced to stay anyway)
    def run(a0):
        d = np.zeros(game.nS); d[s0] = 1.0
        tot = 0.0
        for t in range(H_dec):
            a = a0 if t == 0 else S_
            tot += d @ R[a]
            d = P[a].T @ d
        return tot
    return run(first_action) - run(S_)
