"""POST-HOC supplementary analysis (not in prereg): observable-energy hand-over rule `lowest_gate` vs gate30s / rot30 on the SAME episode keys.
Writes calib/out/survival_v2_obsgate.json.
  systemd-run --user --scope -q -p MemoryMax=12G -p MemorySwapMax=0 .venv/bin/python scripts/obsgate_survival_v2.py"""
import dataclasses, json, os, sys
os.environ.setdefault("JAX_PLATFORMS", "cpu")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import jax, jax.numpy as jnp
import baselines_survival_v2 as B
from animalbench.scenarios.geese.hlc import rule_blind

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "calib", "out", "survival_v2_obsgate.json")


def equivalence_check(ecfg):
    rec = []
    pol = B.lowest_gate_policy(ecfg, 15 // 2 * 0 + 8, check=lambda *a: rec.append([float(np.asarray(x)) for x in a]))
    old = B.NEP; B.NEP = 8
    B.run(ecfg, pol); B.NEP = old
    jax.effects_barrier()
    r = np.array(rec)
    return dict(n_calls=len(r), max_abs_diff_e_self=float(np.abs(r[:, 0] - r[:, 2]).max()), max_abs_diff_e_min_other=float(np.abs(r[:, 1] - r[:, 3]).max()),
                gate_mismatch_obs_vs_stfrac=int((r[:, 4] != r[:, 5]).sum()), gate_mismatch_obs_vs_absolute_E_rem=int((r[:, 4] != r[:, 6]).sum()),
                frac_gate_true=float(r[:, 4].mean()))


def main():
    ecfg = B.v2_env(2.0)
    eq = equivalence_check(ecfg)
    print("equiv", eq, flush=True)
    pols = {"rot30": (dataclasses.replace(ecfg, t_hold_lo=30.0, t_hold_hi=30.0), rule_blind(dataclasses.replace(ecfg, t_hold_lo=30.0, t_hold_hi=30.0))),
            "gate30s": (ecfg, B.gate_policy(ecfg, 15)),
            "lowest_gate15s": (ecfg, B.lowest_gate_policy(ecfg, 8)),
            "lowest_gate30s": (ecfg, B.lowest_gate_policy(ecfg, 15))}
    res, eps = {}, {}
    for name, (e, pol) in pols.items():
        d = B.run(e, pol)
        r, ep = B.analyse(d, ecfg)
        r["arrival"] = float(d["arrived"].any(0).mean())
        r["incoming_is_max"] = r.pop("incoming_is_global_max")
        r.pop("incoming_is_higher_energy")
        eps[name] = ep; res[name] = r
        print(name, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items() if k != "lead_share_by_body"}, flush=True)
    ci = lambda x: float(1.96 * x.std(ddof=1) / np.sqrt(len(x)))
    for a in ("lowest_gate15s", "lowest_gate30s"):
        for b in ("rot30", "gate30s"):
            df = eps[a] - eps[b]
            res[a][f"minus_{b}_mean"] = float(df.mean()); res[a][f"minus_{b}_ci95_episode"] = ci(df)
    json.dump(dict(posthoc=True, n_ep=B.NEP, env="pathT2.0 edge eps=1.02 m_g=1 h=0.05 p=0.05 N=3 mutual random_slots surv_obs_abs surv_pw_spread=0.05 redrawn; same keys PRNGKey(777) as survival_v2_baselines.json",
                   equivalence=eq, results=res), open(OUT, "w"), indent=1)


if __name__ == "__main__":
    main()
