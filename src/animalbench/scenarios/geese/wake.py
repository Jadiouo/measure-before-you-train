"""Wake (upwash) + power model. Backend-agnostic (numpy for calibration, jax.numpy for sim).

Numerics follow poc/wake_rotation_poc.py (wbar / powers / save_frac / calibrate) term by term.
"""
from __future__ import annotations

import functools

import numpy as np

G = 9.81


def quad_nodes(nq: int, b: float):
    th = (np.arange(nq) + 0.5) * np.pi / nq
    sk = -(b / 2) * np.cos(th)
    om = np.sin(th) ** 2
    om = om / om.sum()
    return sk, om


def wbar(xp, dx_b, dy_b, v, c):
    """Mean upwash (m/s) on a follower d_x behind / d_y left of leader (units of b), leader airspeed v.

    c: object with m, b, rho, x_d, r_c0, k_c, nq, d_min, d_max.  gain NOT applied.
    d_x window (d_min, d_max) exclusive, as in the PoC.
    """
    m, b, rho = c.m, c.b, c.rho
    dxc = xp.clip(dx_b, c.d_min, c.d_max)            # keep everything finite outside the window
    dx = dxc * b
    dy = dy_b * b
    b0 = np.pi / 4 * b
    G0 = m * G / (rho * v * b0)
    Gam = G0 * xp.exp(-dx / (c.x_d * b))
    rc = c.r_c0 * b + c.k_c * dx
    sk, om = quad_nodes(c.nq, b)
    out = 0.0
    for s_, w_ in zip(sk, om):
        y = dy + float(s_)
        wv = 0.0
        for s in (+1, -1):
            yy = y - s * b0 / 2
            wv = wv + s * Gam / (2 * np.pi) * yy / (yy ** 2 + rc ** 2)
        out = out + float(w_) * wv
    return xp.where((dx_b > c.d_min) & (dx_b < c.d_max), out, 0.0)


def p_ind(c, v):
    return 2 * (c.m * G) ** 2 / (np.pi * c.rho * v * c.b ** 2)


def p_par(c, v):
    return 0.5 * c.rho * v ** 3 * c.SbCD


def p_nowake(c, v):
    return p_ind(c, v) + p_par(c, v) + c.P_pro


def dp_dv(c, v):
    """Analytic dP/dv (P_pro constant)."""
    return -p_ind(c, v) / v + 3 * p_par(c, v) / v


def p_solo(c) -> float:
    return float(p_nowake(c, c.v0))


def save_frac_np(c, dx_b, dy_b, gw=1.0):
    """Fractional power saving relative to solo (positive = saving), gw*m*g*W/P_solo (PoC save_frac)."""
    W = np.maximum(wbar(np, dx_b, dy_b, c.v0, c), -0.2 * c.v0)
    return gw * c.m * G * W / p_solo(c)


@functools.lru_cache(maxsize=None)
def _calibrate_cached(key):
    m, b, v0, rho, SbCD, P_pro, x_d, r_c0, k_c, nq, d_min, d_max, target, dxref = key

    class C:  # light duck-typed namespace
        pass
    c = C()
    c.m, c.b, c.v0, c.rho, c.SbCD, c.P_pro = m, b, v0, rho, SbCD, P_pro
    c.x_d, c.r_c0, c.k_c, c.nq, c.d_min, c.d_max = x_d, r_c0, k_c, nq, d_min, d_max
    dys = np.linspace(0, 3, 601)
    s = save_frac_np(c, dxref, dys, 1.0)
    return float(target / s.max()), float(dys[s.argmax()])


def calibrate(c, target=0.125, dxref=2.0):
    """(gain, dy*): max over d_y of saving at d_x=dxref equals `target` (PoC calibrate())."""
    return _calibrate_cached((c.m, c.b, c.v0, c.rho, c.SbCD, c.P_pro, c.x_d, c.r_c0, c.k_c, c.nq,
                              c.d_min, c.d_max, target, dxref))
