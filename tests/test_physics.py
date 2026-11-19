"""M1a #1-#5: single bird, P(v), two-bird shape, robustness / gain range, analytic cancellation; PoC agreement."""
import json
import os

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from animalbench.scenarios.geese import GeeseConfig, wake
from animalbench.scenarios.geese.env import reset
from animalbench.scenarios.geese.step import step_continuous
from animalbench.core import compile_schedule

CFG = GeeseConfig()


def test_matches_poc(poc):
    p = dict(poc.BASE)
    gw, dys = poc.calibrate(p)
    assert abs(CFG.gain - gw) < 1e-12
    assert abs(CFG.P_solo - poc.powers(p)[3]) < 1e-12
    dx = np.linspace(0.55, 11.95, 60)
    dy = np.linspace(-3, 3, 61)
    DX, DY = np.meshgrid(dx, dy, indexing="ij")
    a = poc.wbar(DX, DY, p)
    b = wake.wbar(np, DX, DY, CFG.v0, CFG)
    assert np.abs(a - b).max() < 1e-12
    # jax float64 path equals numpy path
    bj = np.asarray(wake.wbar(jnp, jnp.asarray(DX), jnp.asarray(DY), CFG.v0, CFG))
    assert np.abs(bj - b).max() < 1e-12
    assert np.abs(poc.save_frac(DX, DY, p, gw) - wake.save_frac_np(CFG, DX, DY, gw)).max() < 1e-12


def test_1_single_bird_straight_100s():
    cfg = GeeseConfig(n_max=1, log_D=False, log_low=False, use_events=False)
    st = reset(cfg, jax.random.PRNGKey(0), 1, compile_schedule([], cfg.e_max))
    st = st._replace(x=jnp.zeros(1), y=jnp.zeros(1))

    def body(s, _):
        s, out = step_continuous(cfg, s, jnp.zeros((1, 2)))
        return s, None
    st, _ = jax.jit(lambda s: jax.lax.scan(body, s, None, length=1000))(st)
    E = float(st.E_used[0])
    assert abs(E - cfg.P_solo * 100) / (cfg.P_solo * 100) <= 1e-3
    assert abs(float(st.t) - 100.0) < 1e-9
    assert abs(float(st.v[0]) - cfg.v0) < 1e-12


def test_2_power_u_shape():
    c = CFG
    A = 2 * (c.m * wake.G) ** 2 / (np.pi * c.rho * c.b ** 2)
    B = 0.5 * c.rho * c.SbCD
    v_star = (A / (3 * B)) ** 0.25
    vs = np.linspace(0.7 * c.v0 * 0.3, 1.3 * c.v0 * 1.5, 200001)
    P = wake.p_nowake(c, vs)
    v_num = vs[P.argmin()]
    assert abs(v_num - v_star) / v_star <= 0.01
    k = P.argmin()
    assert np.all(np.diff(P[:k]) < 0) and np.all(np.diff(P[k:]) > 0)  # U shape
    # derivative helper consistent
    h = 1e-5
    assert abs(wake.dp_dv(c, 14.0) - (wake.p_nowake(c, 14 + h) - wake.p_nowake(c, 14 - h)) / (2 * h)) < 1e-6


DXS = np.linspace(0.55, 11.95, 229)
DYS = np.linspace(-3, 3, 241)
DYF = np.linspace(-8, 8, 321)


def shape_report(c):
    gw = c.gain
    DX, DY = np.meshgrid(DXS, DYS, indexing="ij")
    S = wake.save_frac_np(c, DX, DY, gw)
    behind = S[:, np.argmin(np.abs(DYS))]
    best = S.max(1)
    DXf, DYf = np.meshgrid(DXS, DYF, indexing="ij")
    Sf = wake.save_frac_np(c, DXf, DYf, gw)
    far = np.abs(Sf[:, np.abs(DYF) > 4.5]).max()           # v0.4 #3: far field starts at |d_y| > 4.5b
    far_4b = np.abs(Sf[:, np.abs(DYF) > 4]).max()          # v0.3 threshold, kept for the record
    thr = None
    for dyv in np.linspace(3.0, 8.0, 1001):          # smallest d_y beyond which |save|<0.5% for every d_x
        if np.abs(wake.save_frac_np(c, DXS[:, None], np.array([dyv, dyv + 0.5, dyv + 1, 8.0])[None, :], gw)).max() < 0.005:
            thr = float(dyv); break
    return dict(far_thr_half_pct=thr,
        behind_neg=bool(np.all(behind < 0)),
        best_monotone=bool(np.all(np.diff(best) <= 1e-12)),
        mirror_err=float(np.abs(S - S[:, ::-1]).max()),
        far_max=float(far), far_max_4b=float(far_4b),
        peak_dx2=float(wake.save_frac_np(c, 2.0, np.linspace(0, 3, 601), gw).max()),
    )


def test_3_two_bird_shape():
    r = shape_report(CFG)
    assert r["behind_neg"]
    assert r["best_monotone"]
    assert r["mirror_err"] < 1e-9
    assert r["far_max"] < 0.005, r                      # v0.4 #3: |d_y| > 4.5b -> |save| < 0.5 %
    assert r["far_thr_half_pct"] <= 4.15, r             # <0.5% holds from d_y >= 4.1b for the base wake
    assert abs(r["peak_dx2"] - 0.125) < 0.003


def test_3_v03_threshold_4b_was_too_tight():
    """Documents why v0.4 moved the threshold: at |d_y| > 4b the base wake still reaches 0.526 % (d_x = 0.55b)."""
    r = shape_report(CFG)
    assert 0.005 < r["far_max_4b"] < 0.0055, r


def test_4_robustness_27_cells_and_gain_range():
    rows = []
    for x_d in (20.0, 40.0, 80.0):
        for rc in (0.05, 0.1, 0.2):
            for kc in (0.01, 0.02, 0.04):
                c = GeeseConfig(x_d=x_d, r_c0=rc, k_c=kc)
                r = shape_report(c)
                r.update(x_d=x_d, r_c0=rc, k_c=kc, gain=c.gain)
                rows.append(r)
                assert r["behind_neg"] and r["best_monotone"] and r["mirror_err"] < 1e-9 and r["far_max"] < 0.011, r
    from calib import gain_range
    out = gain_range.run()
    n_strict = sum(1 for r in rows if r["far_max"] < 0.005)
    json.dump(dict(grid27=rows, n_cells_strict_far_field=n_strict, gain_range=out), open(os.path.join(os.path.dirname(gain_range.__file__), "out", "gain_range.json"), "w"), indent=1)
    assert abs(out["gain_min"] - 0.69) <= 0.05 and abs(out["gain_max"] - 1.82) <= 0.05, out
    assert out["max_curve_diff"] < 1e-6, out


def test_5_upwash_downwash_cancel():
    c = CFG
    y = np.linspace(-400, 400, 800001)   # units of b
    for dx in (0.6, 2.0, 6.0, 11.0):
        w = wake.wbar(np, dx, y, c.v0, c)
        net = np.trapezoid(w, y)
        tot = np.trapezoid(np.abs(w), y)
        assert abs(net) / tot < 0.02, (dx, net, tot)
        assert abs(w[0]) < 1e-3 * np.abs(w).max() and abs(w[-1]) < 1e-3 * np.abs(w).max()
