"""Leadership / reciprocity metrics on a leader sequence (spec v0.3 sec.3, v0.2 sec.3.2).  Pure numpy.

Conventions
  leader : (T,) int array, id of the apex occupant at every decision (one entry per delta_h seconds)
  ind    : (T, N) {0,1} lead-indicator matrix (ind[t, i] = 1 iff i leads at decision t)
"""
from __future__ import annotations

import numpy as np


def leader_to_ind(leader, N):
    leader = np.asarray(leader)
    ind = np.zeros((len(leader), N), np.int8)
    ind[np.arange(len(leader)), leader] = 1
    return ind


def spells(leader):
    """(ids, start, length) of maximal runs of the same leader."""
    leader = np.asarray(leader)
    if len(leader) == 0:
        return np.zeros(0, int), np.zeros(0, int), np.zeros(0, int)
    ch = np.flatnonzero(leader[1:] != leader[:-1]) + 1
    start = np.concatenate([[0], ch])
    end = np.concatenate([ch, [len(leader)]])
    return leader[start], start, end - start


def n_handovers(leader):
    leader = np.asarray(leader)
    return int((leader[1:] != leader[:-1]).sum())


def handover_rate(leader, dh):
    """Leader changes per second."""
    return n_handovers(leader) / (len(leader) * dh)


def tenures(leader, dh, interior=True):
    """Seconds of each leader spell; interior=True drops the first and last (censored) spells when possible."""
    _, _, ln = spells(leader)
    t = ln * dh
    if interior and len(t) > 2:
        t = t[1:-1]
    return t


def lead_share(leader, N):
    return np.bincount(np.asarray(leader), minlength=N) / len(leader)


def lead_support(leader, N, thr=0.05):
    """Number of birds that lead for >= thr of the time."""
    return int((lead_share(leader, N) >= thr).sum())


def free_rider_index(leader, N):
    """Largest lead share relative to the fair share 1/N (1 = fair, N = one bird leads all the time)."""
    return float(lead_share(leader, N).max() * N)


def rls(leader, N, return_rank=False):
    """Longest-idle succession rate: fraction of handovers where the new leader is (one of) the bird(s) that has gone
    longest without leading.  Birds that never led tie for 'longest' (idle = infinity).  The previous leader has
    idle 0.  Returns nan if there is no handover.  return_rank also gives the mean normalised idle rank (1 = longest)."""
    leader = np.asarray(leader)
    last = np.full(N, -np.inf)
    hits, ranks, M = 0, [], 0
    cur = leader[0]
    last[cur] = 0
    for t in range(1, len(leader)):
        b = leader[t]
        if b != cur:
            last[cur] = t - 1                      # old leader led up to t-1
            idle = (t - 1) - last                  # inf for never-led
            idle[cur] = 0.0
            M += 1
            if idle[b] >= idle.max():
                hits += 1
            others = np.delete(idle, cur)
            ranks.append((np.sum(others < idle[b]) + 0.5 * (np.sum(others == idle[b]) - 1)) / max(len(others) - 1, 1))
            cur = b
    if M == 0:
        return (np.nan, np.nan) if return_rank else np.nan
    return (hits / M, float(np.mean(ranks))) if return_rank else hits / M


def psi_matrix(ind, window, min_spells=2):
    """Pairwise exchange index PSI_ij = P(i leads within `window` decisions after a spell of j ended)
    - P(i leads within a random window of the same length).  `ind` (T,N).  NaN where j has < min_spells usable spells
    or i == j."""
    ind = np.asarray(ind, np.int8)
    T, N = ind.shape
    w = int(max(1, window))
    cs = np.vstack([np.zeros((1, N), np.int64), np.cumsum(ind, 0)])       # cs[t] = sum_{s<t}
    def lead_in(a, b):                                                     # i leads in decisions [a, b)  -> (N,)
        return (cs[b] - cs[a]) > 0
    starts = np.arange(0, T - w + 1)
    uncond = ((cs[starts + w] - cs[starts]) > 0).mean(0)                  # (N,)
    P = np.full((N, N), np.nan)
    for j in range(N):
        x = ind[:, j]
        if x.sum() == 0:
            continue
        ends = np.flatnonzero((x[:-1] == 1) & (x[1:] == 0)) + 1          # first decision after a spell of j
        ends = ends[ends + w <= T]
        if len(ends) < min_spells:
            continue
        cond = np.mean([lead_in(e, e + w) for e in ends], axis=0)
        P[:, j] = cond - uncond                                           # P[i, j]
        P[j, j] = np.nan
    return P


def psi_sym(ind, window, min_spells=2):
    """Strongest *mutual* exchange: max over unordered pairs of min(PSI_ij, PSI_ji).  0 if no pair qualifies
    (a one-directional chain such as a cycle gives min ~ 0 or negative).  Also returns the best one-directional PSI."""
    P = psi_matrix(ind, window, min_spells)
    N = P.shape[0]
    best, best_dir = -np.inf, -np.inf
    for i in range(N):
        for j in range(i + 1, N):
            a, b = P[i, j], P[j, i]
            if np.isnan(a) or np.isnan(b):
                continue
            best = max(best, min(a, b))
        row = P[i][~np.isnan(P[i])]
        if len(row):
            best_dir = max(best_dir, row.max())
    return (0.0 if best == -np.inf else float(best)), (0.0 if best_dir == -np.inf else float(best_dir))


def default_window(leader, dh):
    """Half the median interior tenure (in decisions): wide enough to contain the successor's spell start, narrow
    enough that the unconditional probability is not saturated."""
    ten = tenures(leader, dh) / dh
    if len(ten) == 0:
        return 1
    return int(max(1, round(0.5 * np.median(ten))))


def r_lf(lead_time, wake_time):
    """Pearson correlation across birds of lead time and time spent in a wake (diagnostic only, spec v0.2 3.2)."""
    a = np.asarray(lead_time, float)
    b = np.asarray(wake_time, float)
    if a.std() == 0 or b.std() == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def return_rate(leader):
    """Immediate-return (A-B-A) rate: fraction of spells whose second successor is the spell's own leader,
    P(s[m+2] == s[m]).  Uniform random succession over N birds gives 1/(N-1); a two-bird ping-pong gives 1; a cycle
    over >= 3 birds gives 0.  nan with fewer than 3 spells.  (Exploratory complement of PSI, see stage-2 notes.)"""
    ids, _, _ = spells(leader)
    if len(ids) < 3:
        return float("nan")
    return float(np.mean(ids[2:] == ids[:-2]))
