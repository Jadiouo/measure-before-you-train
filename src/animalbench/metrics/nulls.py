"""Null models for the leadership metrics (spec v0.3 sec.3, #20).  All act on the leader series of the tested run.

N-fix : one leader forever (no handover).
N-rnd : same number of handovers at uniformly random times, new leader uniform among the other birds.
N-perm: each bird's lead-indicator series is circularly shifted by an independent random offset (keeps every bird's
        marginal lead share and spell-length distribution exactly, breaks the time alignment).  The result is an
        indicator matrix, not a single-leader sequence (several/no birds may lead at a decision), so it is used
        for the pairwise exchange index only.
N-cyc : deterministic cycle leader0, leader0+1, ... with the tested run's number of handovers at evenly spaced times
        (dwell time = the tested mean, same handover rate).
"""
from __future__ import annotations

import numpy as np

from .leadership import leader_to_ind, n_handovers


def _from_changes(T, changes, ids):
    out = np.empty(T, int)
    bounds = np.concatenate([[0], changes, [T]]).astype(int)
    for k in range(len(bounds) - 1):
        out[bounds[k]:bounds[k + 1]] = ids[k]
    return out


def n_fix(leader, N=None, rng=None):
    leader = np.asarray(leader)
    return np.full(len(leader), leader[0], int)


def n_rnd(leader, N, rng):
    leader = np.asarray(leader)
    T, M = len(leader), n_handovers(leader)
    if M == 0:
        return n_fix(leader)
    ch = np.sort(rng.choice(np.arange(1, T), size=M, replace=False))
    ids = [leader[0]]
    for _ in range(M):
        ids.append(rng.choice([k for k in range(N) if k != ids[-1]]))
    return _from_changes(T, ch, ids)


def n_cyc(leader, N, rng=None):
    leader = np.asarray(leader)
    T, M = len(leader), n_handovers(leader)
    if M == 0:
        return n_fix(leader)
    ch = np.round(np.arange(1, M + 1) * T / (M + 1)).astype(int)
    ids = [(int(leader[0]) + k) % N for k in range(M + 1)]
    return _from_changes(T, ch, ids)


def n_perm_ind(leader, N, rng):
    ind = leader_to_ind(leader, N)
    out = np.empty_like(ind)
    T = ind.shape[0]
    for i in range(N):
        out[:, i] = np.roll(ind[:, i], int(rng.integers(0, T)))
    return out


NULLS = dict(N_fix=n_fix, N_rnd=n_rnd, N_cyc=n_cyc)
