"""Cooperative soaring headroom test (numpy only).

Measures (a) single-agent: omniscient-position reference vs tuned rule,
(b) multi-agent: sharing rule vs non-sharing rule, (c) reference vs sharing rule.
See notes/impl/soaring_headroom.md.  Run:
  systemd-run --user --scope -q -p MemoryMax=12G -p MemorySwapMax=0 \
     .venv/bin/python scripts/soaring_headroom.py [--workers 4] [--quick]
"""
import argparse
import itertools
import json
import os
import sys
import time

import numpy as np

# ------------------------------------------------------------------ constants
A = 5000.0          # torus side (m)
DT = 1.0
T_STEPS = 2400      # mission length (s)
V = 12.0            # airspeed (m/s)
WS = 1.0            # still-air sink rate (m/s)
OM = 0.4            # max turn rate (rad/s) -> min turn radius 30 m
KB = 0.2            # extra sink in a turn: WS*(1+KB*(om/OM)^2)
R0 = 150.0          # nominal thermal radius (m)
H0, HMAX, HRES = 800.0, 1600.0, 100.0
DRIFT = 1.0         # thermal drift relative to the air the gliders fly in (m/s)
SPAN = 3.0          # wing-sensor baseline (m)
LOOK = 90.0         # slot pre-activation lookahead (s), strength is 0 before birth
ALPHA = 0.3         # EMA coefficient of the vario filter
LV = np.array([-1, -.5, -.25, -.125, 0, .125, .25, .5, 1.0]) * OM
COMM_MID = 2000.0

DENS = {"sparse": 0.3, "mid": 1.0, "dense": 2.5}          # active thermals / km^2
LIFE = {"short": 300.0, "long": 1800.0}
SIG = {"low": (0.15, 0.05), "high": (0.5, 0.20)}          # (vario sigma, wing-diff sigma)


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def quant(wc):
    return LV[np.abs(wc[..., None] - LV).argmin(-1)]


def env(u):
    return np.sqrt(np.sin(np.pi * np.clip(u, 0.0, 1.0)))


def sink(om):
    return WS * (1.0 + KB * (om / OM) ** 2)


# ------------------------------------------------------------------ world
def gen_tables(seeds, rho, Lbar, Wk):
    """Pre-generate the whole thermal schedule of each world (Poisson births)."""
    n_act = rho * (A / 1000.0) ** 2
    rate = n_act / Lbar
    rows = []
    for s in seeds:
        rng = np.random.default_rng(int(s))
        n0 = rng.poisson(n_act)
        life0 = Lbar * rng.uniform(.5, 1.5, n0)
        b0 = -rng.uniform(0, 1, n0) * life0
        na = rng.poisson(rate * T_STEPS)
        ba = np.sort(rng.uniform(0, T_STEPS, na))
        lifea = Lbar * rng.uniform(.5, 1.5, na)
        birth = np.concatenate([b0, ba])
        life = np.concatenate([life0, lifea])
        n = len(birth)
        x0 = rng.uniform(0, A, n)
        y0 = rng.uniform(0, A, n)
        W = Wk * rng.uniform(.75, 1.25, n)
        R = R0 * rng.uniform(.8, 1.2, n)
        phi = rng.uniform(0, 2 * np.pi)
        o = np.argsort(birth)
        rows.append(dict(birth=birth[o], life=life[o], x0=x0[o], y0=y0[o], W=W[o], R=R[o],
                         vx=DRIFT * np.cos(phi), vy=DRIFT * np.sin(phi)))
    M = max(len(r["birth"]) for r in rows) + 2
    B = len(rows)
    tab = {k: np.zeros((B, M)) for k in ("birth", "life", "x0", "y0", "W", "R")}
    tab["birth"][:] = 1e12
    tab["life"][:] = 1.0
    tab["R"][:] = 1.0
    K = 0
    for b, r in enumerate(rows):
        n = len(r["birth"])
        for k in tab:
            tab[k][b, :n] = r[k]
        ev = [(bi - LOOK, 1) for bi in r["birth"] if bi + 0 < T_STEPS + LOOK]
        ev += [(bi + li, -1) for bi, li in zip(r["birth"], r["life"])]
        ev.sort(key=lambda e: (e[0], e[1]))
        c = 0
        for tt, d in ev:
            c += d
            K = max(K, c)
    tab["vx"] = np.array([r["vx"] for r in rows])
    tab["vy"] = np.array([r["vy"] for r in rows])
    tab["K"] = K + 3
    tab["B"] = B
    return tab


# ------------------------------------------------------------------ simulation
DEF_RULE = dict(th_leave=0.3, shgain=1.0, lv=1.0, use_roll=0, th_det=0.6, kappa=0.0, Uexp=300.0,
                mc=0.0, th_ref=0.2, kth=1.0)


def simulate(tab, N, kind, P, comm, sig, sigr, Lbar, seed, share=False, trace=None):
    """kind in rule | ref_now | ref_fut.  Returns per-seed arrays.
    comm: communication radius (m); 0 = none.  share: use broadcast beliefs (rule only)."""
    P = {**DEF_RULE, **P}
    B, K = tab["B"], tab["K"]
    rng = np.random.default_rng(seed)
    ar = np.arange(B)
    isref = kind != "rule"
    fut = kind == "ref_fut"
    vx, vy = tab["vx"][:, None], tab["vy"][:, None]
    # agents
    px = rng.uniform(0, A, (B, N))
    py = rng.uniform(0, A, (B, N))
    th = rng.uniform(0, 2 * np.pi, (B, 1)) + 2 * np.pi * np.arange(N)[None, :] / N
    thc = th.copy()
    h = np.full((B, N), H0)
    alive = np.ones((B, N), bool)
    land_t = np.full((B, N), float(T_STEPS))
    energy = np.zeros((B, N))
    om = np.zeros((B, N))
    mode = np.zeros((B, N), np.int8)       # 0 search 1 circle 2 shift 3 goto
    tgt = np.zeros((B, N), int)
    cur = np.zeros((B, N), int)
    cool = np.zeros((B, N), int)
    wf = np.zeros((B, N))
    shl = np.zeros((B, N), int)
    ncirc = np.zeros((B, N), int)
    shh = np.zeros((B, N))
    z = lambda: np.zeros((B, N))
    acc_n, acc_w, acc_px, acc_py, acc_wpx, acc_wpy, acc_gx, acc_gy, acc_ang = (z() for _ in range(9))
    # slots
    f = {k: np.zeros((B, K)) for k in ("birth", "life", "x0", "y0", "W", "R")}
    f["R"][:] = 1.0
    f["life"][:] = 1.0
    f["birth"][:] = 1e12
    act = np.zeros((B, K), bool)
    ptr = np.zeros(B, int)
    # beliefs
    share_on = share and kind == "rule"
    if share_on:
        bt = np.full((B, N, K), -1e9)
        bx = np.zeros((B, N, K)); by = np.zeros((B, N, K)); bm = np.zeros((B, N, K))
        bv = np.zeros((B, N, K), bool)
    best = np.zeros((B, N), int)
    best_ok = np.zeros((B, N), bool)
    alt_rate = np.zeros((B, N))
    lvv = P["lv"]
    om_circ = lvv * OM
    r_circ = V / om_circ
    sink_circ = sink(om_circ)
    kk = np.arange(K)[None, None, :]
    inf = np.inf

    for step in range(T_STEPS):
        t = float(step)
        # ---- thermals
        act &= (t < f["birth"] + f["life"])
        while True:
            nb = tab["birth"][ar, ptr]
            due = nb <= t + LOOK
            if not due.any():
                break
            idx = np.nonzero(due)[0]
            free = np.argmin(act[idx], axis=1)
            for k in f:
                f[k][idx, free] = tab[k][idx, ptr[idx]]
            act[idx, free] = True
            ptr[idx] += 1
        age = t - f["birth"]
        s = f["W"] * env(age / f["life"]) * act
        cx = f["x0"] + vx * age
        cy = f["y0"] + vy * age
        R2 = f["R"] ** 2
        dx = px[:, :, None] - cx[:, None, :]
        dx -= A * np.round(dx / A)
        dy = py[:, :, None] - cy[:, None, :]
        dy -= A * np.round(dy / A)
        g = s[:, None, :] * np.exp(-(dx * dx + dy * dy) / R2[:, None, :])
        w_air = g.sum(2)
        nxv, nyv = -np.sin(th), np.cos(th)
        # ---- sensing
        w_obs = w_air + sig * rng.standard_normal((B, N))
        if not isref:
            gn = (g * (-2.0 / R2[:, None, :]) * (dx * nxv[:, :, None] + dy * nyv[:, :, None])).sum(2)
            roll = SPAN * gn + sigr * rng.standard_normal((B, N))
        wf += ALPHA * (w_obs - wf)

        # ---- communication
        if share_on and comm > 0 and N > 1 and step % 10 == 0:
            ddx = px[:, :, None] - px[:, None, :]
            ddx -= A * np.round(ddx / A)
            ddy = py[:, :, None] - py[:, None, :]
            ddy -= A * np.round(ddy / A)
            ok = (np.hypot(ddx, ddy) <= comm) & alive[:, None, :] & alive[:, :, None]
            ok |= np.eye(N, dtype=bool)[None]
            ct = np.where(ok[..., None], bt[:, None, :, :], -2e9)
            jb = ct.argmax(2)
            bestt = np.take_along_axis(ct, jb[:, :, None, :], 2)[:, :, 0, :]
            upd = bestt > bt
            if upd.any():
                gat = lambda a: a[ar[:, None, None], jb, kk]
                bx = np.where(upd, gat(bx), bx); by = np.where(upd, gat(by), by)
                bm = np.where(upd, gat(bm), bm); bv = np.where(upd, gat(bv), bv)
                bt = np.where(upd, bestt, bt)

        # ---- accumulate circle statistics (rule)
        m1 = (mode == 1) & alive
        if not isref:
            acc_n += m1
            acc_w += np.where(m1, w_obs, 0.0)
            acc_px += np.where(m1, px, 0.0); acc_py += np.where(m1, py, 0.0)
            acc_wpx += np.where(m1, w_obs * px, 0.0); acc_wpy += np.where(m1, w_obs * py, 0.0)
            if P["use_roll"]:
                acc_gx += np.where(m1, roll / SPAN * nxv, 0.0)
                acc_gy += np.where(m1, roll / SPAN * nyv, 0.0)
            acc_ang += np.where(m1, np.abs(om) * DT, 0.0)

        cool = np.maximum(cool - 1, 0)

        # ---- replan (share rule / reference) every 5 steps
        replan = (step % 5 == 0) and (share_on or isref)
        if replan:
            if isref:
                dxc = cx[:, None, :] - px[:, :, None]
                dxc -= A * np.round(dxc / A)
                dyc = cy[:, None, :] - py[:, :, None]
                dyc -= A * np.round(dyc / A)
                d0 = np.hypot(dxc, dyc)
                tau0 = d0 / V
                ex = dxc + vx[:, :, None] * tau0
                ey = dyc + vy[:, :, None] * tau0
                d = np.hypot(ex, ey)
                tau = d / V
                h_arr = h[:, :, None] - WS * tau
                rcr = r_circ
                gfac = np.exp(-rcr * rcr / R2)[:, None, :]
                if fut:
                    t_arr = t + tau
                    age_a = np.maximum(t_arr - f["birth"][:, None, :], 3.0)
                    valid = act[:, None, :] & (t_arr < (f["birth"] + f["life"])[:, None, :]) & \
                        (f["birth"][:, None, :] - t_arr < 10.0)
                    rem = (f["birth"] + f["life"])[:, None, :] - t_arr
                    sa = f["W"][:, None, :] * env(age_a / f["life"][:, None, :])
                    m0 = sa * gfac - sink_circ
                    Tc0 = (HMAX - h_arr) / np.maximum(m0, 0.05)
                    U0 = np.maximum(np.minimum(rem, Tc0), 0.0)
                    smid = f["W"][:, None, :] * env((age_a + U0 / 2) / f["life"][:, None, :])
                    m = smid * gfac - sink_circ
                    U = np.minimum(U0, (HMAX - h_arr) / np.maximum(m, 0.05))
                else:
                    valid = act[:, None, :] & (age[:, None, :] >= 0) & (s[:, None, :] > 0.05)
                    m = s[:, None, :] * gfac - sink_circ
                    Tc = (HMAX - h_arr) / np.maximum(m, 0.05)
                    U = np.minimum(P["Uexp"], Tc)
                U = np.maximum(U, 0.0)
                G = np.maximum(m, 0.0) * U
                sc = G / (tau + U + 1.0)
                feas = valid & (h_arr >= HRES) & (m > P["th_ref"])
                rate_k = m
            else:
                bage = t - bt
                bxp = bx + vx[:, :, None] * bage
                byp = by + vy[:, :, None] * bage
                ex = bxp - px[:, :, None]
                ex -= A * np.round(ex / A)
                ey = byp - py[:, :, None]
                ey -= A * np.round(ey / A)
                d = np.hypot(ex, ey)
                tau = d / V
                h_arr = h[:, :, None] - WS * tau
                p = np.exp(-bage / Lbar)
                Tc = (HMAX - h_arr) / np.maximum(bm, 0.05)
                U = np.maximum(np.minimum(P["Uexp"], Tc), 0.0)
                G = np.maximum(bm, 0.0) * U * p
                sc = G / (tau + U + 1.0)
                feas = bv & (h_arr >= HRES) & (bm > P["th_leave"])
                rate_k = bm * p
            feas &= (h < 0.85 * HMAX)[:, :, None]
            # own current slot excluded for circling agents
            curmask = ((mode == 1) | (mode == 2))[:, :, None] & (kk == cur[:, :, None])
            feas &= ~curmask
            if P["kappa"] > 0:
                occ_ = alive[:, :, None] & ((((mode == 3)[:, :, None] & (kk == tgt[:, :, None])) |
                                             (((mode == 1) | (mode == 2))[:, :, None] & (kk == cur[:, :, None]))))
                cnt = occ_.sum(1, keepdims=True) - occ_
                sc = sc / (1.0 + P["kappa"] * cnt)
            m3 = (mode == 3)[:, :, None] & (kk == tgt[:, :, None])
            sc = np.where(m3, sc * 1.25, sc)
            sc = np.where(feas, sc, -inf)
            best = sc.argmax(2)
            bsc = np.take_along_axis(sc, best[:, :, None], 2)[:, :, 0]
            best_ok = np.isfinite(bsc) & (bsc > 0)
            alt_rate = np.take_along_axis(rate_k, best[:, :, None], 2)[:, :, 0]
            # search / goto agents
            sg = ((mode == 0) | (mode == 3)) & alive
            newgo = sg & best_ok
            tgt = np.where(newgo, best, tgt)
            mode = np.where(newgo, 3, np.where(sg & ~best_ok, 0, mode)).astype(np.int8)
            if isref:
                # circling reference agents: stay / leave decision with true state
                m1r = (mode == 1) & alive
                sc_cur = np.take_along_axis(s, cur, 1)
                cg = np.take_along_axis(np.broadcast_to(np.exp(-r_circ ** 2 / R2), (B, K)), cur, 1)
                m_now = sc_cur * cg - sink_circ
                leave = m1r & ((m_now < P["th_ref"]) | (h >= HMAX - 10.0) |
                               (best_ok & (m_now < P["mc"] * alt_rate)))
                mode = np.where(leave, np.where(best_ok, 3, 0), mode).astype(np.int8)
                tgt = np.where(leave & best_ok, best, tgt)
                cool = np.where(leave, 3, cool)

        # ---- mode transitions (rule)
        if not isref:
            # detection while searching / going
            det = ((mode == 0) | (mode == 3)) & alive & (cool == 0) & (wf > P["th_det"])
            if share_on:
                arr = (mode == 3) & alive
                if arr.any():
                    tb_ = np.take_along_axis(bt, tgt[:, :, None], 2)[:, :, 0]
                    tx = np.take_along_axis(bx, tgt[:, :, None], 2)[:, :, 0] + vx * (t - tb_)
                    ty = np.take_along_axis(by, tgt[:, :, None], 2)[:, :, 0] + vy * (t - tb_)
                    ddx_ = tx - px; ddx_ -= A * np.round(ddx_ / A)
                    ddy_ = ty - py; ddy_ -= A * np.round(ddy_ / A)
                    arrived = arr & (np.hypot(ddx_, ddy_) < 60.0)
                else:
                    arrived = np.zeros((B, N), bool)
            else:
                arrived = np.zeros((B, N), bool)
            enter = det | arrived
            if enter.any():
                domk = g.argmax(2)
                cur = np.where(arrived & ~det, tgt, domk)
                thc = np.where(enter & (mode == 0), th, thc)
                for a_ in (acc_n, acc_w, acc_px, acc_py, acc_wpx, acc_wpy, acc_gx, acc_gy, acc_ang):
                    a_[enter] = 0.0
                ncirc[enter] = 0
                mode = np.where(enter, 1, mode).astype(np.int8)
            # shift countdown
            m2 = (mode == 2) & alive
            shl = np.where(m2, shl - 1, shl)
            endsh = m2 & (shl <= 0)
            if endsh.any():
                for a_ in (acc_n, acc_w, acc_px, acc_py, acc_wpx, acc_wpy, acc_gx, acc_gy, acc_ang):
                    a_[endsh] = 0.0
                mode = np.where(endsh, 1, mode).astype(np.int8)
            # circle completion
            fin = (mode == 1) & alive & (acc_ang >= 2 * np.pi - 1e-9) & (acc_n > 3)
            if fin.any():
                n_ = np.maximum(acc_n, 1)
                m_circ = acc_w / n_ - sink_circ
                top = h >= HMAX - 20.0
                ncirc += fin
                meanw = acc_w / n_
                weak = (m_circ < P["th_leave"]) & ((ncirc >= 2) | (meanw < 0.3))
                cxm, cym = acc_px / n_, acc_py / n_
                mcleave = np.zeros((B, N), bool)
                if share_on and P["mc"] > 0:
                    mcleave = best_ok & (m_circ < P["mc"] * alt_rate)
                leave = fin & (weak | top | mcleave)
                if share_on:
                    bi, ni = np.nonzero(fin)
                    ks = cur[bi, ni]
                    valid_ = ~weak[bi, ni]
                    bt[bi, ni, ks] = t
                    bx[bi, ni, ks] = cxm[bi, ni]; by[bi, ni, ks] = cym[bi, ni]
                    bm[bi, ni, ks] = np.maximum(m_circ[bi, ni], 0.0)
                    bv[bi, ni, ks] = valid_
                stay = fin & ~leave
                # centering shift
                if P["shgain"] > 0:
                    Sx = acc_wpx - acc_w * acc_px / n_
                    Sy = acc_wpy - acc_w * acc_py / n_
                    zs = 1.0 / (max(sig, 0.02) * r_circ * np.sqrt(n_ / 2.0))
                    Dx, Dy = Sx * zs, Sy * zs
                    if P["use_roll"]:
                        zg = 1.0 / (max(sigr, 0.01) / SPAN * np.sqrt(n_ / 2.0))
                        Dx = Dx + acc_gx * zg
                        Dy = Dy + acc_gy * zg
                    gmag = 2.0 * np.hypot(Sx, Sy) / (n_ * r_circ ** 2)
                    d_est = np.clip(gmag * R0 ** 2 / (2.0 * np.maximum(meanw, 0.1)), 0.0, 2.0 * R0)
                    steps = np.minimum(np.round(P["shgain"] * d_est / V), 14).astype(int)
                    doshift = stay & (np.hypot(Dx, Dy) > P["kth"]) & (steps >= 2)
                    shh = np.where(doshift, np.arctan2(Dy, Dx), shh)
                    shl = np.where(doshift, steps, shl)
                    mode = np.where(doshift, 2, mode).astype(np.int8)
                    reset = stay & ~doshift
                else:
                    reset = stay
                for a_ in (acc_n, acc_w, acc_px, acc_py, acc_wpx, acc_wpy, acc_gx, acc_gy, acc_ang):
                    a_[fin] = 0.0
                if leave.any():
                    gomc = leave & mcleave & ~weak & ~top
                    mode = np.where(leave, np.where(gomc, 3, 0), mode).astype(np.int8)
                    tgt = np.where(gomc, best, tgt)
                    cool = np.where(leave, 25, cool)

        # ---- controls
        if isref:
            # steering targets
            om_new = np.zeros((B, N))
            # goto
            dxc = np.take_along_axis(cx, tgt, 1) - px
            dxc -= A * np.round(dxc / A)
            dyc = np.take_along_axis(cy, tgt, 1) - py
            dyc -= A * np.round(dyc / A)
            dd = np.hypot(dxc, dyc)
            tau_ = dd / V
            ix_ = dxc + vx * tau_; iy_ = dyc + vy * tau_
            bearing = np.arctan2(iy_, ix_)
            goto = (mode == 3) & alive
            enterc = goto & (dd < r_circ + 15.0)
            if enterc.any():
                cur = np.where(enterc, tgt, cur)
                mode = np.where(enterc, 1, mode).astype(np.int8)
            # circle control about current centre
            cdx = px - np.take_along_axis(cx, cur, 1)
            cdx -= A * np.round(cdx / A)
            cdy = py - np.take_along_axis(cy, cur, 1)
            cdy -= A * np.round(cdy / A)
            rho = np.hypot(cdx, cdy)
            des = np.arctan2(cdy, cdx) + np.pi / 2 + np.clip(0.9 * (rho - r_circ) / r_circ, -1.0, 1.0)
            ff = np.minimum(V / np.maximum(rho, r_circ), OM)
            c_circ = ff + 1.5 * wrap(des - th)
            c_goto = wrap(bearing - th) / 2.0
            c_srch = wrap(thc - th) / 2.0
            om_cmd = np.where(mode == 1, c_circ, np.where(mode == 3, c_goto, c_srch))
            om = quant(np.clip(om_cmd, -OM, OM))
        else:
            c_goto = np.zeros((B, N))
            if share_on:
                tb_ = np.take_along_axis(bt, tgt[:, :, None], 2)[:, :, 0]
                tx = np.take_along_axis(bx, tgt[:, :, None], 2)[:, :, 0] + vx * (t - tb_)
                ty = np.take_along_axis(by, tgt[:, :, None], 2)[:, :, 0] + vy * (t - tb_)
                ddx_ = tx - px; ddx_ -= A * np.round(ddx_ / A)
                ddy_ = ty - py; ddy_ -= A * np.round(ddy_ / A)
                dd_ = np.hypot(ddx_, ddy_)
                ix_ = ddx_ + vx * dd_ / V; iy_ = ddy_ + vy * dd_ / V
                c_goto = wrap(np.arctan2(iy_, ix_) - th) / 2.0
            c_srch = wrap(thc - th) / 2.0
            c_shift = wrap(shh - th) / 2.0
            om_cmd = np.where(mode == 1, om_circ, np.where(mode == 2, c_shift,
                              np.where(mode == 3, c_goto, c_srch)))
            om = quant(np.clip(om_cmd, -OM, OM))

        # ---- integrate
        th = th + om * DT
        px = np.where(alive, px + V * np.cos(th) * DT, px)
        py = np.where(alive, py + V * np.sin(th) * DT, py)
        h = np.where(alive, np.minimum(h + (w_air - sink(om)) * DT, HMAX), h)
        energy += np.where(alive, h / HMAX, 0.0)
        if trace is not None:
            trace(step, mode, h, w_air, wf, px, py, cur, cx, cy, s)
        newly = alive & (h <= 0.0)
        land_t = np.where(newly, t, land_t)
        alive &= ~newly
        if not alive.any():
            break

    aloft = land_t / T_STEPS
    return dict(aloft=aloft, energy=energy / T_STEPS, landed=~alive)


def summarise(r):
    return dict(Amin=r["aloft"].min(1), Amean=r["aloft"].mean(1), Emin=r["energy"].min(1),
                Emean=r["energy"].mean(1), fail=r["landed"].any(1).astype(float))


def objective(sm):
    return float(np.mean(sm["Amin"] + 0.3 * sm["Emin"]))


# ------------------------------------------------------------------ tuning
GRID_RULE = [dict(th_leave=a, shgain=b, lv=c, use_roll=d, th_det=e)
             for a, b, c, d, e in itertools.product([-0.8, -0.5, -0.3, 0.0, 0.1], [0.0, 0.7, 1.0, 1.4], [1.0, 0.5], [0, 1], [0.4, 0.6, 1.0])]
GRID_SHARE = [dict(kappa=a, Uexp=b, mc=c) for a, b, c in
              itertools.product([0.0, 0.5], [150.0, 500.0, 1200.0], [0.0, 0.6])]
GRID_REF = [dict(lv=a, th_ref=b, mc=c) for a, b, c in itertools.product([1.0, 0.5], [0.1, 0.4], [0.0, 0.7])]
GRID_REF_NOW = [dict(g, Uexp=u) for g in GRID_REF for u in (200.0, 800.0)]


def tune(tab, N, kind, base, grid, comm, sig, sigr, Lbar, share, seed):
    best, bj = None, -1
    for gp in grid:
        P = {**base, **gp}
        j = objective(summarise(simulate(tab, N, kind, P, comm, sig, sigr, Lbar, seed, share)))
        if j > bj:
            bj, best = j, P
    return best, bj


def boot(x, rng, n=2000):
    x = np.asarray(x, float)
    idx = rng.integers(0, len(x), (n, len(x)))
    m = x[idx].mean(1)
    return [float(x.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]


def run_condition(cond, Ns=(1, 5, 10), n_eval=60, n_tune=24, quick=False):
    name, rho, Lbar, Wk, sig, sigr = cond
    t0 = time.time()
    tune_seeds = np.arange(1000, 1000 + n_tune)
    eval_seeds = np.arange(0, n_eval)
    tt = gen_tables(tune_seeds, rho, Lbar, Wk)
    te = gen_tables(eval_seeds, rho, Lbar, Wk)
    rng = np.random.default_rng(12345)
    grid_rule = GRID_RULE[::12] if quick else GRID_RULE
    out = dict(name=name, rho=rho, Lbar=Lbar, Wk=Wk, sig=sig, sigr=sigr, tuned={}, cells={})
    # 1. single-agent rule tuned on N=1
    P1, j1 = tune(tt, 1, "rule", {}, grid_rule, 0.0, sig, sigr, Lbar, False, 7)
    out["tuned"]["rule1"] = P1
    per = {}                                    # (N, label) -> summary on eval seeds

    def ev(N, kind, P, comm, share, label):
        sm = summarise(simulate(te, N, kind, P, comm, sig, sigr, Lbar, 99, share))
        per[(N, label)] = sm
        return sm

    for N in Ns:
        ev(N, "rule", P1, 0.0, False, "noshare")
        jn = objective(summarise(simulate(tt, N, "rule", P1, 0.0, sig, sigr, Lbar, 7, False)))
        # reference tuned per N; the better of ref_now / ref_fut (by tuning-seed objective) is "ref"
        jr = {}
        for kind in ("ref_now", "ref_fut"):
            gr = GRID_REF if kind == "ref_fut" else GRID_REF_NOW
            Pr, jr[kind] = tune(tt, N, kind, {"Uexp": 600.0, "kappa": 0.0}, gr[::3] if quick else gr,
                                0.0, sig, sigr, Lbar, False, 7)
            out["tuned"][f"{kind}_N{N}"] = Pr
            ev(N, kind, Pr, 0.0, False, kind)
        kb = max(jr, key=jr.get)
        per[(N, "ref")] = per[(N, kb)]
        out["tuned"][f"ref_choice_N{N}"] = kb
        if N > 1:
            for cn, cr in (("inf", 1e9), ("mid", COMM_MID)):
                Ps, js = tune(tt, N, "rule", P1, GRID_SHARE[::3] if quick else GRID_SHARE, cr, sig, sigr, Lbar, True, 7)
                out["tuned"][f"share_{cn}_N{N}"] = Ps
                if jn >= js:       # sharing does not help on the tuning seeds -> best rule is "do not use broadcasts"
                    out["tuned"][f"share_{cn}_N{N}"] = dict(P1, share_off=True)
                    per[(N, f"share_{cn}")] = per[(N, "noshare")]
                else:
                    ev(N, "rule", Ps, cr, True, f"share_{cn}")
    # gaps
    def gap(a, b, key):           # a - b in percentage points, paired by seed
        return boot(100.0 * (a[key] - b[key]), rng)

    for N in Ns:
        cell = {}
        for lab in [l for (n, l) in per if n == N]:
            sm = per[(N, lab)]
            cell[lab] = {k: float(v.mean()) for k, v in sm.items()}
            cell[lab]["Amin_seeds"] = [float(x) for x in sm["Amin"]]
        for key in ("Amin", "Emin"):
            g = {}
            g["ref_vs_noshare"] = gap(per[(N, "ref")], per[(N, "noshare")], key)
            g["fut_vs_now"] = gap(per[(N, "ref_fut")], per[(N, "ref_now")], key)
            if N > 1:
                for cn in ("inf", "mid"):
                    g[f"b_{cn}"] = gap(per[(N, f"share_{cn}")], per[(N, "noshare")], key)
                    g[f"c_{cn}"] = gap(per[(N, "ref")], per[(N, f"share_{cn}")], key)
                    g[f"cnow_{cn}"] = gap(per[(N, "ref_now")], per[(N, f"share_{cn}")], key)
                    g[f"cfut_{cn}"] = gap(per[(N, "ref_fut")], per[(N, f"share_{cn}")], key)
            else:
                g["a"] = g["ref_vs_noshare"]
                g["anow"] = gap(per[(N, "ref_now")], per[(N, "noshare")], key)
                g["afut"] = gap(per[(N, "ref_fut")], per[(N, "noshare")], key)
            cell["gaps_" + key] = g
        out["cells"][str(N)] = cell
    out["secs"] = time.time() - t0
    return out


def conditions(sig_override=None):
    cs = []
    for dn, rho in DENS.items():
        for ln, L in LIFE.items():
            for Wk in (1.5, 3.0):
                for sn, (sg, sr) in SIG.items():
                    cs.append((f"{dn}|{ln}|W{Wk}|{sn}", rho, L, Wk, sg, sr))
    return cs


def _worker(args):
    cond, Ns, quick = args
    return run_condition(cond, Ns, quick=quick)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--only", default="")
    ap.add_argument("--out", default="calib/out/soaring_headroom.json")
    ap.add_argument("--no-anchor", action="store_true")
    a = ap.parse_args()
    import multiprocessing as mp
    ctx = mp.get_context("spawn")
    cs = conditions()
    if a.only:
        cs = [c for c in cs if a.only in c[0]]
    jobs = [(c, (1, 5, 10), a.quick) for c in cs]
    anchors = []
    if not a.no_anchor:
        anchors = [(("anchor|dense|long|W3|sig0", DENS["dense"], LIFE["long"], 3.0, 0.0, 0.0), (1, 5), a.quick),
                   (("anchor|dense|long|W3|sig_low", DENS["dense"], LIFE["long"], 3.0, 0.15, 0.05), (1, 5), a.quick),
                   (("anchor|dense|long|W1.5|sig0", DENS["dense"], LIFE["long"], 1.5, 0.0, 0.0), (1, 5), a.quick)]
    jobs = anchors + jobs
    res = []
    t0 = time.time()
    with ctx.Pool(a.workers) as pool:
        for i, r in enumerate(pool.imap_unordered(_worker, jobs)):
            res.append(r)
            print(f"[{i + 1}/{len(jobs)}] {r['name']} {r['secs']:.0f}s elapsed {time.time() - t0:.0f}s", flush=True)
            os.makedirs(os.path.dirname(a.out), exist_ok=True)
            json.dump(dict(consts=dict(A=A, T=T_STEPS, V=V, WS=WS, OM=OM, KB=KB, R0=R0, H0=H0, HMAX=HMAX,
                                       DRIFT=DRIFT, SPAN=SPAN, COMM_MID=COMM_MID), results=res),
                      open(a.out, "w"))


if __name__ == "__main__":
    main()
