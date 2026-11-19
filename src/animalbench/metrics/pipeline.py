"""Metric pipeline: one run (leader series + optional D) -> metric dict; plus null-model distributions."""
from __future__ import annotations

import numpy as np

from . import flows, leadership as L, nulls

METRICS = ("rate", "tenure_med", "support", "rls", "rls_rank", "psi", "psi_dir", "free_rider", "rho_D", "varB", "r_lf")


def compute_metrics(leader, dh, N, D=None, alive=None, window=None, win_B=30):
    """Scalar metrics of one run.  `window` (decisions) of PSI defaults to half the median tenure of this run."""
    leader = np.asarray(leader)
    ten = L.tenures(leader, dh)
    w = L.default_window(leader, dh) if window is None else window
    ind = L.leader_to_ind(leader, N)
    r, rr = L.rls(leader, N, return_rank=True)
    ps, pd = L.psi_sym(ind, w)
    out = dict(rate=L.handover_rate(leader, dh), tenure_med=float(np.median(ten)) if len(ten) else float("nan"),
               support=L.lead_support(leader, N), rls=r, rls_rank=rr, psi=ps, psi_dir=pd,
               free_rider=L.free_rider_index(leader, N), return_rate=L.return_rate(leader), window=w)
    if D is not None:
        D = np.asarray(D)
        out["rho_D"] = flows.rho_D(D, alive)
        out["varB"] = float(flows.var_B_windows(D, win_B).mean())
        lead_t = L.lead_share(leader, N)
        out["r_lf"] = L.r_lf(lead_t, flows.time_in_wake(D, dh))
    return out


def null_distribution(leader, dh, N, K=200, seed=0, window=None):
    """Distributions (K draws) of rate / rls / psi under N-rnd, N-perm, (N-cyc, N-fix deterministic) for one run.
    Returns {null: {metric: array(K)}}."""
    rng = np.random.default_rng(seed)
    leader = np.asarray(leader)
    w = L.default_window(leader, dh) if window is None else window
    res = {}
    for name in ("N_rnd", "N_perm"):
        rows = []
        for _ in range(K):
            if name == "N_perm":
                ind = nulls.n_perm_ind(leader, N, rng)
                ps, pd = L.psi_sym(ind, w)
                rows.append(dict(rate=L.handover_rate(leader, dh), rls=np.nan, psi=ps, psi_dir=pd))
            else:
                s = nulls.n_rnd(leader, N, rng)
                ps, pd = L.psi_sym(L.leader_to_ind(s, N), w)
                rows.append(dict(rate=L.handover_rate(s, dh), rls=L.rls(s, N), psi=ps, psi_dir=pd))
        res[name] = {k: np.array([r[k] for r in rows], float) for k in rows[0]}
    for name in ("N_cyc", "N_fix"):
        s = nulls.NULLS[name](leader, N, rng)
        ps, pd = L.psi_sym(L.leader_to_ind(s, N), w)
        res[name] = {k: np.array([v], float) for k, v in
                     dict(rate=L.handover_rate(s, dh), rls=L.rls(s, N), psi=ps, psi_dir=pd).items()}
    return res


def summarize_nulls(nd, qs=(0.05, 0.5, 0.99)):
    return {n: {m: {f"q{int(q * 100):02d}": float(np.nanquantile(v, q)) if np.isfinite(v).any() else float("nan")
                    for q in qs} for m, v in d.items()} for n, d in nd.items()}


def report_with_nulls(leader, dh, N, D=None, K=200, seed=0):
    """Observed metrics plus summary of the null distributions (the form in which results are reported)."""
    return dict(obs=compute_metrics(leader, dh, N, D), nulls=summarize_nulls(null_distribution(leader, dh, N, K, seed)))
