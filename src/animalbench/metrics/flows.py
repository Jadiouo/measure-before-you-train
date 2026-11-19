"""Dyadic wake-flow metrics from D (T, N, N): D[t, i, j] = saving that i's wake gives to j in decision t (P_solo*s)."""
from __future__ import annotations

import numpy as np


def received(D):
    """R_j = sum_t sum_i D[t,i,j]  (N,)"""
    return np.asarray(D).sum((0, 1))


def provided(D):
    """sum_t sum_j D[t,i,j]  (N,)"""
    return np.asarray(D).sum((0, 2))


def net_balance(D):
    """B_i = R_i - provided_i  (sums to 0)."""
    return received(D) - provided(D)


def var_B_windows(D, win):
    """Variance across birds of the windowed net balance B_i(w); returns (n_windows,). win in decisions."""
    D = np.asarray(D)
    T = D.shape[0]
    out = []
    for a in range(0, T - win + 1, win):
        Dw = D[a:a + win]
        B = Dw.sum((0, 1)) - Dw.sum((0, 2))
        out.append(B.var())
    return np.asarray(out)


def rho_D(D, alive=None):
    """Symmetry of the time-integrated flow: Pearson corr(D_ij, D_ji) over ordered pairs i != j (alive birds only)."""
    S = np.asarray(D).sum(0)
    N = S.shape[0]
    idx = np.arange(N) if alive is None else np.flatnonzero(alive)
    S = S[np.ix_(idx, idx)]
    m = ~np.eye(len(idx), dtype=bool)
    x, y = S[m], S.T[m]
    if x.std() < 1e-12 or y.std() < 1e-12:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def time_in_wake(D, dh, thr=0.02):
    """Fraction of decisions in which bird j saved more than `thr` (fraction of P_solo): sum_i D[t,i,j] > thr*dh."""
    sav = np.asarray(D).sum(1)                       # (T, N)
    return (sav > thr * dh).mean(0)
