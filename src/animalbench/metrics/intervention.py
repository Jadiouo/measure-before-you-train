"""Interventional classifier for I1 (identity relabel) and I2 (free-rider), spec v0.4 sec.3.  Pure numpy.

Everything is computed on the leader series (apex occupant per decision, body ids) and on D (T,N,N) of ONE episode and
its paired intervention runs (same seed => identical history up to the event).  All thresholds are the spec's and
module constants, not arguments of the classifiers.  Timeline and what is post-hoc: notes/impl/M2gate_intervention.md.

Windows (decisions, delta_h = 2 s): pre [60,300) s, post [360,600) s, event at 300 s.
Q_W  : fraction of the handovers in a window that are a RETURN: handover n (b -> a) such that an earlier handover m
       (a -> b) exists with 0 < t_n - t_m <= W   (body space; credits may originate before the window).
W    : 4 x median interior tenure in t < 300 s, floor 40 s.
Null : N-perm (each bird's lead indicator circularly shifted inside the window, spells re-read in start order) and N-rnd
       (same number of handovers at random times, random new leader), K draws, computed on the PRE window of the episode.
"""
from __future__ import annotations

import numpy as np

# ------------------------------------------------------------------ constants (numbers from spec v0.4 3.1/3.2; the constants below were
# NOT frozen before the first 50-episode run -- the frozen version is the one hashed in notes/impl/M2gate_prereg.md)
T_EVENT = 300.0
PRE = (60.0, 300.0)
POST = (360.0, 600.0)
W_MULT = 4.0
W_FLOOR = 40.0
P_PREMISE = 0.99         # pre-window Q must exceed this N-perm percentile
P_RETURN_IN = 0.95       # post-window Q "falls back inside the null interval" = <= this percentile
DELTA_ID = 0.50          # relative drop => identity dependence
DELTA_SHAM = 0.20        # sham / identity-free tolerance
MEM_CLOSE = 0.20         # |delta_M - delta_R| <= this => ledger lives in memory
I2_DROP = 0.50           # y drop => punish
I2_TOL = 0.20            # tolerate band
K_NULL = 1000
LEAD_THR = 0.05          # lead_support share threshold

I1_LABELS = ["no-signal", "identity", "state-driven", "mixed"]
I2_LABELS = ["no-signal", "punish", "tolerate", "mixed"]


# ------------------------------------------------------------------ handovers and Q_W
def handovers(leader):
    """-> (frm, to, t) arrays; t = decision index at which `to` is first seen as apex."""
    leader = np.asarray(leader)
    t = np.flatnonzero(leader[1:] != leader[:-1]) + 1
    return leader[t - 1], leader[t], t


def q_return(leader, a, b, W):
    """Q_W of the handovers in decisions [a, b); W in decisions.  NaN when the window has no handover."""
    frm, to, t = handovers(leader)
    inwin = np.flatnonzero((t >= a) & (t < b))
    if len(inwin) == 0:
        return float("nan")
    ret = 0
    for n in inwin:
        prev = np.flatnonzero((t < t[n]) & (t[n] - t <= W) & (to == frm[n]) & (frm == to[n]))
        ret += len(prev) > 0
    return ret / len(inwin)


def window_W(leader, dh, t_event=T_EVENT):
    """W in decisions: 4 x median interior tenure of the pre-event series, floor 40 s (value fixed from pre-event data only)."""
    seg = np.asarray(leader)[: int(round(t_event / dh))]
    ch = np.flatnonzero(seg[1:] != seg[:-1]) + 1
    bounds = np.concatenate([[0], ch, [len(seg)]])
    ten = np.diff(bounds) * dh
    ten = ten[1:-1] if len(ten) > 2 else ten
    med = float(np.median(ten)) if len(ten) else W_FLOOR
    return int(np.ceil(max(W_MULT * med, W_FLOOR) / dh))


def _seq_q(ids, t, W):
    """Q of an explicit handover sequence (ids[k] = leader after k-th handover, t[k] its time); first element = initial leader."""
    n = len(ids) - 1
    if n <= 0:
        return float("nan")
    frm, to, tt = ids[:-1], ids[1:], t[1:]
    ret = 0
    for k in range(n):
        m = np.flatnonzero((tt[:k] > tt[k] - W - 1e-9) & (to[:k] == frm[k]) & (frm[:k] == to[k]) & (tt[:k] < tt[k]))
        ret += len(m) > 0
    return ret / n


def null_q(leader, a, b, W, N, rng, K=K_NULL):
    """Null Q distributions for the window [a,b) of this episode: dict(perm=array(K), rnd=array(K))."""
    seg = np.asarray(leader)[a:b]
    T = len(seg)
    ind = np.zeros((T, N), bool)
    ind[np.arange(T), seg] = True
    ch = np.flatnonzero(seg[1:] != seg[:-1]) + 1
    M = len(ch)
    perm, rnd = np.full(K, np.nan), np.full(K, np.nan)
    if M == 0:
        return dict(perm=perm, rnd=rnd)
    for k in range(K):
        # N-perm: independent circular shift of every bird's indicator; spells sorted by start
        starts, ids = [], []
        for i in range(N):
            x = np.roll(ind[:, i], int(rng.integers(0, T)))
            if not x.any():
                continue
            s = np.flatnonzero(x & ~np.roll(x, 1))
            starts += s.tolist(); ids += [i] * len(s)
        o = np.argsort(starts, kind="stable")
        ids_a, st_a = np.asarray(ids)[o], np.asarray(starts)[o]
        keep = np.r_[True, ids_a[1:] != ids_a[:-1]] if len(ids_a) else np.array([], bool)
        perm[k] = _seq_q(ids_a[keep], st_a[keep], W) if len(ids_a) else np.nan
        # N-rnd: M handovers at random decisions, new leader uniform among the others
        tt = np.sort(rng.choice(np.arange(1, T), size=M, replace=False))
        seq = [seg[0]]
        for _ in range(M):
            seq.append(rng.choice([j for j in range(N) if j != seq[-1]]))
        rnd[k] = _seq_q(np.asarray(seq), np.r_[0, tt], W)
    return dict(perm=perm, rnd=rnd)


def _q(x, p):
    x = x[np.isfinite(x)]
    return float(np.quantile(x, p)) if len(x) else float("nan")


def delta(q_pre, q_post, q_null_med):
    den = q_pre - q_null_med
    if not np.isfinite(den) or den <= 1e-12 or not np.isfinite(q_post):
        return float("nan")
    return float((q_pre - q_post) / den)


# ------------------------------------------------------------------ I1
def i1_features(leaders, dh, N, rng, K=K_NULL):
    """leaders: dict cond -> (T,) leader series for conds R, S, M, RM (same seed).  Returns per-condition features."""
    pre = (int(PRE[0] / dh), int(PRE[1] / dh))
    post = (int(POST[0] / dh), int(POST[1] / dh))
    W = window_W(leaders["S"], dh)
    q_pre = q_return(leaders["S"], pre[0], pre[1], W)
    nd = null_q(leaders["S"], pre[0], pre[1], W, N, rng, K)
    n_med = _q(nd["perm"], 0.5)
    f = dict(W=W * dh, q_pre=q_pre, null99=_q(nd["perm"], P_PREMISE), null95=_q(nd["perm"], P_RETURN_IN), null_med=n_med,
             rnd99=_q(nd["rnd"], P_PREMISE))
    for c in ("R", "S", "M", "RM"):
        qp = q_return(leaders[c], post[0], post[1], W)
        f[f"q_post_{c}"] = qp
        f[f"delta_{c}"] = delta(q_pre, qp, n_med)
    f["n_ho_pre"] = len(handovers(leaders["S"])[2][(handovers(leaders["S"])[2] >= pre[0]) & (handovers(leaders["S"])[2] < pre[1])])
    return f


def i1_label(f):
    """Per-episode I1 label from the features (spec 3.1 rules 1,3,5; (c) I1-X is only available for learned policies)."""
    if not (np.isfinite(f["q_pre"]) and np.isfinite(f["null99"]) and f["q_pre"] > f["null99"]):
        return "no-signal"
    dR, dS = f["delta_R"], f["delta_S"]
    if not (np.isfinite(dR) and np.isfinite(dS)):
        return "mixed"
    a = (f["q_post_R"] <= f["null95"]) or (dR > DELTA_ID)
    if a and dS <= DELTA_SHAM:
        return "identity"
    if dR <= DELTA_SHAM:
        return "state-driven"
    return "mixed"


def paired_divergence(lead_R, lead_S, dh, t_event=T_EVENT, t_end=POST[1]):
    """AMENDMENT V2 (not in spec v0.4).  POST-HOC: the V2 rule was proposed only after the spec-literal 50-episode result had been
    seen (rollout files runs/m2gate/cpu/*.npz were written 2026-10-25 05:59-06:31; this file was last modified 06:45, i.e. after
    the results; see notes/impl/M2gate_intervention.md section 'Timeline').  Only the held-out re-run frozen in
    notes/impl/M2gate_prereg.md counts as evidence for V2.
    Same seed => identical history up to the event.  True iff the relabelled run's leader series differs from the sham run's
    anywhere in [t_event, t_end).  Bitwise False for any policy that does not read the codes."""
    a, b = int(round(t_event / dh)), int(round(t_end / dh))
    return bool(np.any(np.asarray(lead_R)[a:b] != np.asarray(lead_S)[a:b]))


def i1_label_v2(f, diverged):
    """(post-hoc, see paired_divergence) V2 = spec rules 1 (premise), 3(b) (sham) and 5 with rule 3(a) 'post-window Q falls back / delta_R > 50%' replaced by
    'the relabelled run diverges from its paired sham run' (Q_W over a 60 s-excluded window cannot see a ledger that re-forms
    within one or two handovers).  identity: premise, diverged, delta_S <= 20%.  state-driven: premise, not diverged."""
    if not (np.isfinite(f["q_pre"]) and np.isfinite(f["null99"]) and f["q_pre"] > f["null99"]):
        return "no-signal"
    if not diverged:
        return "state-driven"
    dS = f["delta_S"]
    if np.isfinite(dS) and dS <= DELTA_SHAM:
        return "identity"
    return "mixed"


def lead_set(leader, a, b, N, thr=LEAD_THR):
    seg = np.asarray(leader)[a:b]
    return frozenset(np.flatnonzero(np.bincount(seg, minlength=N) / len(seg) >= thr).tolist())


def set_changed(lead_R, lead_S, dh, N):
    """V3 (PREREG, designed on development seeds 2000-2019 after the internal review; notes/impl/M2gate_prereg.md).
    True iff the set of birds with lead share >= LEAD_THR in the post window [360,600) s differs between the relabelled run and
    its paired sham.  Separates 'WHO leads changed' (ledger / lookup keyed by the code) from 'WHEN they hand over changed'
    (code used only as timing noise)."""
    a, b = int(round(POST[0] / dh)), int(round(POST[1] / dh))
    return lead_set(lead_R, a, b, N) != lead_set(lead_S, a, b, N)


def i1_label_v3(f, set_ch):
    """V3 = V2 premise and sham condition, with the post-event criterion 'who leads changed' (set_changed) instead of 'any
    divergence'.  identity: premise, set_ch, delta_S <= 0.2.  state-driven: premise, not set_ch (also when only the timing moved)."""
    if not (np.isfinite(f["q_pre"]) and np.isfinite(f["null99"]) and f["q_pre"] > f["null99"]):
        return "no-signal"
    if not set_ch:
        return "state-driven"
    dS = f["delta_S"]
    if np.isfinite(dS) and dS <= DELTA_SHAM:
        return "identity"
    return "mixed"


def i1_memory_sublabel(f):
    """Only meaningful for episodes labelled identity (spec 3.1 rule 4)."""
    dR, dM = f["delta_R"], f["delta_M"]
    if not (np.isfinite(dR) and np.isfinite(dM)):
        return "na"
    if abs(dM - dR) <= MEM_CLOSE:
        return "memory"
    if dM <= DELTA_SHAM and dR > dM:
        return "lookup"
    return "other"


# ------------------------------------------------------------------ I2
def lead_support(leader, a, b, N, thr=LEAD_THR):
    seg = np.asarray(leader)[a:b]
    return int((np.bincount(seg, minlength=N) / len(seg) >= thr).sum())


def y_rate(leader, a, b, dh, exclude=None):
    """Handovers per minute inside [a,b) whose outgoing leader is not `exclude` (the free-rider)."""
    frm, to, t = handovers(leader)
    sel = (t >= a) & (t < b)
    if exclude is not None:
        sel &= frm != exclude
    return float(sel.sum() / ((b - a) * dh / 60.0))


def bstar_check(D, rider, a, b, rng, n_boot=500):
    """Operation check: free-rider's net contribution B* = provided - received per decision over [a,b); True when the
    bootstrap 95% CI of its mean is entirely below 0 (it only takes)."""
    D = np.asarray(D)[a:b]
    inc = D[:, rider, :].sum(1) - D[:, :, rider].sum(1)
    boots = rng.choice(inc, size=(n_boot, len(inc)), replace=True).mean(1)
    return bool(np.quantile(boots, 0.975) < 0), float(inc.mean())


def i2_features(F, S, D_F, rider, dh, N, rng):
    """F, S: leader series of the treated / sham run (same seed); rider: body id of the free-rider."""
    pre = (int(PRE[0] / dh), int(PRE[1] / dh))
    post = (int(POST[0] / dh), int(POST[1] / dh))
    y_pre = y_rate(S, *pre, dh, rider)
    ls_pre = lead_support(S, *pre, N)
    f = dict(rider=int(rider), y_pre=y_pre, ls_pre=ls_pre, y_post_F=y_rate(F, *post, dh, rider),
             y_post_S=y_rate(S, *post, dh, rider), ls_post_F=lead_support(F, *post, N))
    ok, inc = bstar_check(D_F, rider, post[0], post[1], rng)
    f["bstar_ok"], f["bstar_mean"] = ok, inc
    if y_pre > 0:
        f["dy_F"] = (f["y_post_F"] - y_pre) / y_pre
        f["dy_S"] = (f["y_post_S"] - y_pre) / y_pre
    else:
        f["dy_F"] = f["dy_S"] = float("nan")
    f["dls_F"] = (f["ls_post_F"] - ls_pre) / ls_pre if ls_pre > 0 else float("nan")
    f["ls_post_S"] = lead_support(S, *post, N)
    f["dls_S"] = (f["ls_post_S"] - ls_pre) / ls_pre if ls_pre > 0 else float("nan")
    return f


def i2_label(f):
    if not (f["y_pre"] > 0 and f["ls_pre"] >= 2):
        return "no-signal"
    dyF, dyS, dls = f["dy_F"], f["dy_S"], f["dls_F"]
    if dyF <= -I2_DROP and f["bstar_ok"] and abs(dyS) < I2_TOL:
        return "punish"
    if abs(dls) < I2_TOL and abs(dyF) < I2_TOL and abs(dyF - dyS) < I2_TOL:
        return "tolerate"
    return "mixed"


def i2_label_v2(f):
    """AMENDMENT V2 (post-hoc, introduced after the first 50-episode run showed the sham itself drifts by -17..-29% in y and by
    discrete lead_support steps of 17%; see notes/impl/M2gate_intervention.md).  Spec 3.2 says the sham exists to 'control the
    natural pre->post drift', but its thresholds are applied to the raw pre->post change of the treated run.  V2 applies the SAME
    thresholds (0.5 / 0.2) to the sham-corrected effect (difference in differences), nothing else changes:
      punish  : (dy_F - dy_S) <= -0.5 and operation check
      tolerate: |dy_F - dy_S| < 0.2 and |dls_F - dls_S| < 0.2"""
    if not (f["y_pre"] > 0 and f["ls_pre"] >= 2):
        return "no-signal"
    ey, el = f["dy_F"] - f["dy_S"], f["dls_F"] - f["dls_S"]
    if ey <= -I2_DROP and f["bstar_ok"]:
        return "punish"
    if abs(ey) < I2_TOL and abs(el) < I2_TOL:
        return "tolerate"
    return "mixed"


BLOCK_SIZE = 10          # PREREG (notes/impl/M2gate_prereg.md): I2 is decided on blocks of 10 consecutive seeds
BLOCK_VALID_FRAC = 0.5   # a block with fewer valid-premise episodes than this fraction is 'no-signal'


def i2_label_block(fs):
    """PREREG I2 rule (block level).  fs: list of i2_features dicts of one block of episodes.  Same thresholds as i2_label_v2
    (0.5 / 0.2) but applied to block means of the sham-corrected effect, because a single episode contains only ~6 handovers
    per window (episode-level count noise, see notes/impl/M2gate_prereg.md section 'bootstrap')."""
    ok = [f for f in fs if f["y_pre"] > 0 and f["ls_pre"] >= 2]
    if len(ok) < BLOCK_VALID_FRAC * len(fs) or not ok:
        return "no-signal"
    m = lambda k: float(np.mean([f[k] for f in ok]))
    ey, el = m("dy_F") - m("dy_S"), m("dls_F") - m("dls_S")
    if ey <= -I2_DROP and np.mean([bool(f["bstar_ok"]) for f in ok]) >= 0.5:
        return "punish"
    if abs(ey) < I2_TOL and abs(el) < I2_TOL:
        return "tolerate"
    return "mixed"


# ------------------------------------------------------------------ confusion bookkeeping
def confusion(labels_by_agent, columns):
    """-> {agent: [count per column]} (rows in insertion order)."""
    return {a: [int(sum(l == c for l in ls)) for c in columns] for a, ls in labels_by_agent.items()}
