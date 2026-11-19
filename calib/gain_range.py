"""Reproduce the wake-gain range under +-30% of m, b, P_pro (PoC: 0.69-1.82) and check the normalised curve is
unchanged after re-solving gain. Writes calib/out/gain_table.json.   Run: python -m calib.gain_range"""
import json
import os

import numpy as np

from animalbench.scenarios.geese import GeeseConfig, wake

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")


def run():
    base = GeeseConfig()
    dxs = np.linspace(0.55, 11.95, 80)
    dys = np.linspace(-3, 3, 121)
    DX, DY = np.meshgrid(dxs, dys, indexing="ij")

    def norm_curve(c):
        s = wake.save_frac_np(c, DX, DY, c.gain)
        return s / wake.save_frac_np(c, 2.0, np.linspace(0, 3, 601), c.gain).max()
    ref = norm_curve(base)
    rows, gains, diffs = [], [], []
    for name in ("m", "b", "P_pro"):
        for f in (0.7, 1.0, 1.3):
            c = base.replace(**{name: getattr(base, name) * f})
            nc = norm_curve(c)
            d = float(np.abs(nc - ref).max())
            rows.append(dict(param=name, factor=f, gain=c.gain, P_solo=c.P_solo, curve_diff=d))
            gains.append(c.gain); diffs.append(d)
    res = dict(table=rows, gain_min=float(min(gains)), gain_max=float(max(gains)), max_curve_diff=float(max(diffs)),
               reference_range=[0.69, 1.82])
    return res


if __name__ == "__main__":
    r = run()
    os.makedirs(OUT, exist_ok=True)
    json.dump(r, open(os.path.join(OUT, "gain_table.json"), "w"), indent=1)
    print(json.dumps({k: v for k, v in r.items() if k != "table"}))
