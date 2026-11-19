"""Cooperative soaring headroom test, round 2: information-matched comparison.
Imports world model/rule/helpers from soaring_headroom.py (unchanged); defines simulate2 with
information levels L2/L3, stronger search baselines and a posterior-sampling planner.
Altitude doubled (H0=1600, HMAX=3200) to remove the landing floor; gate metric = Emin.
See notes/impl/soaring_headroom2.md.  Run:
  systemd-run --user --scope -q -p MemoryMax=12G -p MemorySwapMax=0 \
     .venv/bin/python scripts/soaring_headroom2.py --workers 3
"""
import argparse
import itertools
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from soaring_headroom import (A, DT, T_STEPS, V, WS, OM, KB, R0, HRES, DRIFT, SPAN, LOOK, ALPHA, LV,  # noqa: E402
                              COMM_MID, DENS, LIFE, SIG, wrap, quant, env, sink, gen_tables, boot,
                              DEF_RULE, GRID_RULE, GRID_SHARE, GRID_REF, GRID_REF_NOW)

H0, HMAX = 1600.0, 3200.0     # doubled altitude (round 1: 800 / 1600)
GC = 40                       # coverage grid cells per side
CS = A / GC
DEF2 = dict(lvl=1, srch="line", H=200.0, gam=0.9, nst=12, Ks=6, rdet=150.0)

def wt(a):
    return a - A * np.round(a / A)


def ray_heading(px, py, vx, vy, t, lastv, P, Lp, nd=16):
    """Heading maximising discounted 'unknown-thermal factor' along a ray (thermal frame)."""
    B, N = px.shape
    nst, gam = int(P["nst"]), P["gam"]
    f = 1.0 - np.exp(-(t - lastv) / Lp)
    phis = 2 * np.pi * np.arange(nd) / nd
    ux = V * np.cos(phis)[None, :] - vx          # (B,nd)
    uy = V * np.sin(phis)[None, :] - vy
    j = np.arange(1, nst + 1)
    tau = CS * j / V
    qx = (px - vx * t)[:, :, None, None]
    qy = (py - vy * t)[:, :, None, None]
    X = (qx + ux[:, None, :, None] * tau) % A
    Y = (qy + uy[:, None, :, None] * tau) % A
    ix = (X / CS).astype(int) % GC
    iy = (Y / CS).astype(int) % GC
    val = f[np.arange(B)[:, None, None, None], ix, iy]
    sc = (val * (gam ** j)).sum(3)               # (B,N,nd)
    return phis[sc.argmax(2)]


def plan_search(rng, px, py, h, sgm, vx, vy, t, lastv, P, prior, r_circ, sink_circ, hcur):
    """Posterior-sampling rollout planner for searching agents.  Draws Ks samples of the unknown
    thermals (prior density x not-recently-searched factor, prior strength / lifetime), then for each
    agent (sequentially; thermals reachable by an earlier agent's chosen heading are claimed) scores D
    straight-line headings over horizon H by the expected best (gain/time) of a thermal met on the way.
    Returns (value (B,N), heading (B,N))."""
    B, N = px.shape
    Ks, M, D = int(P["Ks"]), 80, 12
    H, rdet, Uexp = P["H"], P["rdet"], P["Uexp"]
    Lp, rho_p, Wp = prior["Lbar"], prior["rho"], prior["Wk"]
    ar = np.arange(B)
    f = 1.0 - np.exp(-(t - lastv) / Lp)
    qj = rng.uniform(0, A, (B, Ks, M, 2))
    ix = (qj[..., 0] / CS).astype(int) % GC
    iy = (qj[..., 1] / CS).astype(int) % GC
    pacc = np.minimum(1.0, rho_p * (A / 1000.0) ** 2 / M * f[ar[:, None, None], ix, iy])
    u = rng.uniform(0.05, 0.95, (B, Ks, M))
    life = Lp * rng.uniform(0.5, 1.5, (B, Ks, M))
    sj = Wp * rng.uniform(0.75, 1.25, (B, Ks, M)) * env(u)
    m = sj * np.exp(-r_circ ** 2 / R0 ** 2) - sink_circ
    rem = life * (1.0 - u)
    ok = (rng.uniform(size=(B, Ks, M)) < pacc) & (m > 0.1)
    claimed = np.zeros_like(ok)
    phis = 2 * np.pi * np.arange(D) / D
    vs = np.zeros((B, N))
    hd = np.zeros((B, N))
    for i in range(N):
        if not sgm[:, i].any():
            continue
        ph = np.concatenate([np.broadcast_to(phis, (B, D)), hcur[:, i:i + 1]], 1)     # + "keep heading"
        ux = V * np.cos(ph) - vx
        uy = V * np.sin(ph) - vy
        u2 = (ux ** 2 + uy ** 2)[:, None, None, :]
        uxe, uye = ux[:, None, None, :], uy[:, None, None, :]
        rx = wt(qj[..., 0] - (px[:, i] - vx[:, 0] * t)[:, None, None])[..., None]
        ry = wt(qj[..., 1] - (py[:, i] - vy[:, 0] * t)[:, None, None])[..., None]
        tau = (rx * uxe + ry * uye) / u2
        perp = np.hypot(rx - uxe * tau, ry - uye * tau)
        h_arr = h[:, i][:, None, None, None] - WS * tau
        reach = (ok & ~claimed)[..., None] & (tau > 5.0) & (tau <= H) & (perp < rdet) & (h_arr >= HRES)
        Tc = (HMAX - h_arr) / np.maximum(m[..., None], 0.05)
        U = np.maximum(np.minimum(np.minimum(Uexp, Tc), rem[..., None] - tau), 0.0)
        rate = np.where(reach, np.maximum(m, 0.0)[..., None] * U / (tau + U + 1.0), 0.0)
        val = rate.max(2).mean(1)                # (B,D)
        d = val[:, :D].argmax(1)
        keep = val[:, D] * 1.2 >= val[ar, d]                # hysteresis: keep heading unless clearly better
        d = np.where(keep, D, d)
        vs[:, i] = val[ar, d]
        hd[:, i] = ph[ar, d]
        sel = reach[ar, :, :, d] & (val[ar, d] > 0)[:, None, None] & sgm[:, i][:, None, None]
        claimed |= sel
    return vs, hd


def simulate2(tab, N, kind, P, comm, sig, sigr, prior, seed, share=False, trace=None):
    """kind in rule | ref_now | ref_fut.  Returns per-seed arrays.
    comm: communication radius (m); 0 = none.  share: use broadcast beliefs (rule only).
    prior: dict(rho, Lbar, Wk) = what the decision rules believe (not necessarily the truth).
    P['lvl']: 1 = shared rule beliefs, 2 = ref_seen (true state of team-detected thermals),
              3 = ref_seen + perfect centring.  P['srch']: line | zone | prior | plan."""
    P = {**DEF_RULE, **DEF2, **P}
    Lbar = prior["Lbar"]
    lvl, srch = int(P["lvl"]), P["srch"]
    trk = srch in ("prior", "plan")
    rngp = np.random.default_rng(seed + 7919)
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
    seen = np.zeros((B, K), bool)
    lastv = np.full((B, GC, GC), -1e9)
    thz = np.zeros((B, N))
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
            seen[idx, free] = False
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

        # ---- coverage map (thermal frame) for prior / plan search
        if trk:
            cix = ((px - vx * t) % A / CS).astype(int) % GC
            ciy = ((py - vy * t) % A / CS).astype(int) % GC
            bi2 = np.broadcast_to(ar[:, None], (B, N))
            for dxo in (-1, 0, 1):
                for dyo in (-1, 0, 1):
                    lastv[bi2[alive], ((cix + dxo) % GC)[alive], ((ciy + dyo) % GC)[alive]] = t

        # ---- communication
        if share_on and lvl < 2 and comm > 0 and N > 1 and step % 10 == 0:
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
        replan = (step % (10 if srch == "plan" else 5) == 0) and (share_on or isref)
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
                if lvl >= 2:
                    # team-detected thermals: true centre / strength / drift (perfect memory and association)
                    mt_ = np.maximum(s * np.exp(-r_circ ** 2 / R2) - sink_circ, 0.0)
                    sn_ = seen & act & (s > 0.05)
                    bt = np.full((B, N, K), t)
                    bx = np.broadcast_to(cx[:, None, :], (B, N, K)).copy()
                    by = np.broadcast_to(cy[:, None, :], (B, N, K)).copy()
                    bm = np.broadcast_to(mt_[:, None, :], (B, N, K)).copy()
                    bv = np.broadcast_to(sn_[:, None, :], (B, N, K)).copy()
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
            if srch == "plan" and not isref and sg.any():
                vsb, hdg = plan_search(rngp, px, py, h, sg, vx, vy, t, lastv, P, prior, r_circ, sink_circ,
                                       np.where(mode == 3, th, thc))
                usearch = sg & (vsb > np.where(best_ok, bsc, 0.0))
                newgo = sg & best_ok & ~usearch
                thc = np.where(sg & ~newgo, np.where(vsb > 0, hdg, np.where(mode == 3, th, thc)), thc)
            tgt = np.where(newgo, best, tgt)
            mode = np.where(newgo, 3, np.where(sg & ~newgo, 0, mode)).astype(np.int8)
            if srch == "prior" and not isref:
                m0_ = (mode == 0) & alive
                if m0_.any():
                    thc = np.where(m0_, ray_heading(px, py, vx, vy, t, lastv, P, prior["Lbar"]), thc)
            if lvl >= 2 and not isref:
                m1r = (mode == 1) & alive
                sc_cur = np.take_along_axis(s, cur, 1)
                cg = np.take_along_axis(np.broadcast_to(np.exp(-r_circ ** 2 / R2), (B, K)), cur, 1)
                m_now = sc_cur * cg - sink_circ
                leave = m1r & ((m_now < P["th_ref"]) | (h >= HMAX - 10.0) |
                               (best_ok & (m_now < P["mc"] * alt_rate)))
                mode = np.where(leave, np.where(best_ok, 3, 0), mode).astype(np.int8)
                tgt = np.where(leave & best_ok, best, tgt)
                cool = np.where(leave, 25, cool)
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
                bi_, ni_ = np.nonzero(enter)
                seen[bi_, cur[bi_, ni_]] = True
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
                leave = fin & (top if lvl >= 2 else (weak | top | mcleave))
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
                    doshift = stay & (np.hypot(Dx, Dy) > P["kth"]) & (steps >= 2) & (lvl < 3)
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
            if srch == "zone":
                ycen = ((np.arange(N) + 0.5) / N * A)[None, :]
                dyz = ycen - (py - vy * t)
                dyz -= A * np.round(dyz / A)
                hxs = np.where(np.arange(N) % 2 == 0, 1.0, -1.0)[None, :] * np.ones((B, 1))
                c_srch = wrap(np.arctan2(np.clip(dyz / 300.0, -1.0, 1.0), hxs) - th) / 2.0
            c_shift = wrap(shh - th) / 2.0
            c_m1 = om_circ
            if lvl >= 3:
                cdx = px - np.take_along_axis(cx, cur, 1)
                cdx -= A * np.round(cdx / A)
                cdy = py - np.take_along_axis(cy, cur, 1)
                cdy -= A * np.round(cdy / A)
                rho_ = np.hypot(cdx, cdy)
                des = np.arctan2(cdy, cdx) + np.pi / 2 + np.clip(0.9 * (rho_ - r_circ) / r_circ, -1.0, 1.0)
                c_m1 = np.minimum(V / np.maximum(rho_, r_circ), OM) + 1.5 * wrap(des - th)
            om_cmd = np.where(mode == 1, c_m1, np.where(mode == 2, c_shift,
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


# ------------------------------------------------------------------ evaluation
def summarise(r):
    return dict(Amin=r["aloft"].min(1), Amean=r["aloft"].mean(1), Emin=r["energy"].min(1),
                Emean=r["energy"].mean(1), fail=r["landed"].any(1).astype(float))


def objective(sm):                      # gate is Emin; Amin as a small tie-breaker
    return float(np.mean(sm["Emin"] + 0.3 * sm["Amin"]))


GRID_PRIOR = [dict(gam=a, nst=b, kappa=c) for a, b, c in itertools.product([0.8, 0.9, 0.95], [8, 16], [0.0, 0.5])]
GRID_PLAN = [dict(H=a, Uexp=b, kappa=c) for a, b, c in itertools.product([120.0, 200.0], [150.0, 500.0, 1200.0], [0.0, 0.5])]
INF = 1e9


def tune(tab, N, kind, base, grid, comm, sig, sigr, prior, share, seed):
    best, bj = None, -1e9
    for gp in grid:
        P = {**base, **gp}
        j = objective(summarise(simulate2(tab, N, kind, P, comm, sig, sigr, prior, seed, share)))
        if j > bj:
            bj, best = j, P
    return best, bj


def run_condition(cond, Ns, n_eval=60, n_tune=24, quick=False):
    name, rho, Lbar, Wk, sig, sigr = cond
    t0 = time.time()
    tt = gen_tables(np.arange(1000, 1000 + n_tune), rho, Lbar, Wk)
    te = gen_tables(np.arange(0, n_eval), rho, Lbar, Wk)
    rng = np.random.default_rng(12345)
    pri = dict(rho=rho, Lbar=Lbar, Wk=Wk)
    sub = (lambda g: g[::4]) if quick else (lambda g: g)
    out = dict(name=name, rho=rho, Lbar=Lbar, Wk=Wk, sig=sig, sigr=sigr, tuned={}, jtune={}, cells={}, secs_stage={})
    P1, j1 = tune(tt, 1, "rule", {}, GRID_RULE[::12] if quick else GRID_RULE, 0.0, sig, sigr, pri, False, 7)
    out["tuned"]["rule1"] = P1
    out["secs_stage"]["rule1"] = time.time() - t0

    for N in Ns:
        per, J, PP = {}, {}, {}
        ts = time.time()

        def ev(label, kind, P, comm, share, prior=pri):
            per[label] = summarise(simulate2(te, N, kind, P, comm, sig, sigr, prior, 99, share))

        def tn(label, kind, base, grid, comm, share):
            P, j = tune(tt, N, kind, base, sub(grid), comm, sig, sigr, pri, share, 7)
            PP[label], J[label] = P, j
            return P

        # L0 : rule, no sharing
        ev("L0", "rule", P1, 0.0, False)
        J["L0"] = objective(summarise(simulate2(tt, N, "rule", P1, 0.0, sig, sigr, pri, 7, False)))
        PP["L0"] = P1
        # L1 : shared rule (retuned); falls back to L0 if sharing does not help on tune seeds
        Ps = tn("L1", "rule", {**P1, "lvl": 1, "srch": "line"}, GRID_SHARE, INF, True)
        if J["L0"] >= J["L1"]:
            out["tuned"][f"L1_share_off_N{N}"] = True
            PP["L1"], J["L1"] = P1, J["L0"]
            per["L1"] = per["L0"]
            base = {**P1, "lvl": 1, "srch": "line"}
        else:
            ev("L1", "rule", Ps, INF, True)
            base = {**Ps, "lvl": 1, "srch": "line"}
        out["secs_stage"][f"L0L1_N{N}"] = time.time() - ts
        for lab, extra, grid in (("L2", dict(lvl=2), GRID_SHARE), ("L3", dict(lvl=3), GRID_SHARE),
                                 ("zone", dict(srch="zone"), GRID_SHARE), ("prior", dict(srch="prior"), GRID_PRIOR),
                                 ("plan", dict(srch="plan"), GRID_PLAN)):
            ts = time.time()
            P = tn(lab, "rule", {**base, **extra}, grid, INF, True)
            ev(lab, "rule", P, INF, True)
            out["secs_stage"][f"{lab}_N{N}"] = time.time() - ts
        # map-omniscient reference (upper reference only)
        ts = time.time()
        jr = {}
        for kind, gr in (("ref_fut", GRID_REF[::2]), ("ref_now", GRID_REF_NOW[::4])):
            Pr, jr[kind] = tune(tt, N, kind, {"Uexp": 600.0, "kappa": 0.0}, gr[::2] if quick else gr, 0.0, sig, sigr, pri, False, 7)
            ev(kind, kind, Pr, 0.0, False)
            PP[kind] = Pr
        kb = max(jr, key=jr.get)
        per["ref"] = per[kb]
        out["tuned"][f"ref_choice_N{N}"] = kb
        out["secs_stage"][f"ref_N{N}"] = time.time() - ts

        def gap(a, b, key):
            return boot(100.0 * (per[a][key] - per[b][key]), rng)

        cell = {lab: {k: float(v.mean()) for k, v in sm.items()} for lab, sm in per.items()}
        for lab, sm in per.items():
            cell[lab]["Emin_seeds"] = [float(x) for x in sm["Emin"]]
        for key in ("Emin", "Amin"):
            cell["gap_vs_L1_" + key] = {lab: gap(lab, "L1", key) for lab in per if lab != "L1"}
        cell["plan_vs"] = {key: {lab: gap("plan", lab, key) for lab in ("L1", "L2", "L3", "zone", "prior")} for key in ("Emin", "Amin")}
        cand = [l for l in ("L1", "zone", "prior")]
        bb = max(cand, key=lambda l: J[l])
        cell["best_baseline"] = bb
        cell["jtune"] = J
        out["tuned"][f"N{N}"] = PP
        # prior perturbation (only if the planner looks ahead of L1 on the eval seeds)
        if cell["gap_vs_L1_Emin"]["plan"][0] > 1.0 and not quick:
            ts = time.time()
            pert = {}
            for pn, (rs, ls) in {"rho_x2": (2.0, 1.0), "rho_x0.5": (0.5, 1.0), "life_x2": (1.0, 2.0), "life_x0.5": (1.0, 0.5)}.items():
                pp = dict(rho=rho * rs, Lbar=Lbar * ls, Wk=Wk)
                r = {}
                for lab in {"plan", "L1", bb}:
                    P = PP[lab]
                    r[lab] = summarise(simulate2(te, N, "rule", P, INF if not (lab == "L1" and J["L0"] >= J["L1"]) else 0.0,
                                                 sig, sigr, pp, 99, not (lab == "L1" and J["L0"] >= J["L1"])))
                pert[pn] = dict(
                    plan=float(r["plan"]["Emin"].mean()), L1=float(r["L1"]["Emin"].mean()), bb=float(r[bb]["Emin"].mean()),
                    plan_vs_L1=boot(100 * (r["plan"]["Emin"] - r["L1"]["Emin"]), rng),
                    plan_vs_bb=boot(100 * (r["plan"]["Emin"] - r[bb]["Emin"]), rng))
            cell["perturb"] = pert
            out["secs_stage"][f"perturb_N{N}"] = time.time() - ts
        out["cells"][str(N)] = cell
    out["secs"] = time.time() - t0
    return out


def cells():
    cs = []
    for dn in ("sparse", "mid"):
        for ln in ("short", "long"):
            for Wk in (1.5, 3.0):
                sg, sr = SIG["low"]
                cs.append((f"{dn}|{ln}|W{Wk}|low", DENS[dn], LIFE[ln], Wk, sg, sr))
    return cs


N10 = {"mid|long|W3.0|low", "sparse|long|W1.5|low", "mid|short|W1.5|low"}


def _worker(args):
    cond, Ns, quick = args
    return run_condition(cond, Ns, quick=quick)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--only", default="")
    ap.add_argument("--Ns", default="")
    ap.add_argument("--out", default="calib/out/soaring_headroom2.json")
    a = ap.parse_args()
    import multiprocessing as mp
    ctx = mp.get_context("spawn")
    cs = [c for c in cells() if a.only in c[0]]
    jobs = []
    for c in cs:
        Ns = tuple(int(x) for x in a.Ns.split(",")) if a.Ns else ((5, 10) if c[0] in N10 else (5,))
        jobs.append((c, Ns, a.quick))
    jobs.sort(key=lambda j: -len(j[1]))
    res = []
    t0 = time.time()
    with ctx.Pool(a.workers) as pool:
        for i, r in enumerate(pool.imap_unordered(_worker, jobs)):
            res.append(r)
            print(f"[{i + 1}/{len(jobs)}] {r['name']} {r['secs']:.0f}s elapsed {time.time() - t0:.0f}s", flush=True)
            os.makedirs(os.path.dirname(a.out), exist_ok=True)
            json.dump(dict(consts=dict(A=A, T=T_STEPS, V=V, WS=WS, OM=OM, KB=KB, R0=R0, H0=H0, HMAX=HMAX,
                                       DRIFT=DRIFT, SPAN=SPAN), results=res), open(a.out, "w"))


if __name__ == "__main__":
    main()
