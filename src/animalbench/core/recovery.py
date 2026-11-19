"""Scenario-agnostic recovery time (framework function, spec v0.2 sec 1.2 #3)."""
from __future__ import annotations

import numpy as np


def recovery_time(dev, t, t_on, base_window=20.0, tol_rel=1.2, tol_abs=0.05, hold=5.0):
    """First time after t_on at which dev <= tol_rel*dev_base + tol_abs holds for `hold` seconds.

    dev: (T,) deviation series; t: (T,) times. dev_base = mean over [t_on-base_window, t_on).
    Returns (t_rec, censored): t_rec measured from t_on; censored -> t_rec = t[-1]-t_on.
    """
    dev = np.asarray(dev, float); t = np.asarray(t, float)
    base_sel = (t >= t_on - base_window) & (t < t_on)
    base = dev[base_sel].mean() if base_sel.any() else dev[t < t_on].mean()
    thr = tol_rel * base + tol_abs
    ok = dev <= thr
    dt = float(np.median(np.diff(t)))
    need = max(1, int(round(hold / dt)))
    idx = np.where(t >= t_on)[0]
    run = 0
    for k in idx:
        run = run + 1 if ok[k] else 0
        if run >= need:
            return float(t[k - need + 1] - t_on), False
    return float(t[-1] - t_on), True
