"""Post-hoc analysis of v2 runs per notes/impl/survival_v2_prereg.md. Writes calib/out/survival_v2_seeds.json and notes/impl/survival_v2_results.md.
usage: .venv/bin/python scripts/analyze_survival_v2.py   (CPU)"""
import os, sys, json, dataclasses, glob
os.environ.setdefault("JAX_PLATFORMS", "cpu")
import jax, numpy as np
from scipy import stats as st
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import animalbench  # noqa
from animalbench.marl import load_ckpt
from animalbench.marl.ippo import init_train, make_rollout
from animalbench.marl.survival_eval import survival_metrics, classify_survival
import baselines_survival_v2 as B
from animalbench.scenarios.geese.hlc import rule_blind

NEP, EPS = 512, 1e-6
RAND_BASE, ROT30, GATE30 = 0.9622, 0.9875, 1.0284
COND = {"A2": ("MAPPO", 1, "v2_A"), "B2": ("IPPO", 1, "v2_B"), "C2": ("IPPO", 0, "v2_C"), "D2": ("MAPPO", 0, "v2_D")}
PRED = {"A2": "维持", "B2": "塌缩", "C2": "塌缩", "D2": "不维持"}
tci = lambda x: (float(np.mean(x)), float(st.t.ppf(0.975, len(x) - 1) * np.std(x, ddof=1) / np.sqrt(len(x)))) if len(x) > 1 else (float(np.mean(x)), None)


def energy_metrics(slot, ef, ecfg):
    T, Bn, A = slot.shape
    E0 = np.full(A, ecfg.E0_coef * ecfg.P_solo * ecfg.T); E0[0] *= (1 + ecfg.surv_h)
    efp = np.concatenate([np.ones((1, Bn, A)), ef[:-1]], 0); Ea = efp * E0
    lead = np.argmax(slot == 0, -1)
    ti, bi = np.where(lead[1:] != lead[:-1])
    n = gm = ol = 0
    for t, b in zip(ti, bi):
        o, nw = lead[t, b], lead[t + 1, b]
        if not (ef[t, b, o] > EPS and ef[t, b, nw] > EPS) or (efp[t, b] > EPS).sum() < 3: continue
        e = Ea[t, b]; n += 1; gm += float(e[nw] == e.max()); ol += float(e[o] == e.min())
    m3 = (efp > EPS).sum(-1) == 3
    el = np.take_along_axis(Ea, lead[..., None], -1)[..., 0]
    top_lead = float(((el >= Ea.max(-1) - 1e-12) & m3).sum() / max(m3.sum(), 1))   # leader currently holds the most absolute energy
    return dict(n_swaps_3alive=n, incoming_is_max=(gm / n if n else None), outgoing_is_lowest=(ol / n if n else None),
                lead_share_body0=float((slot == 0)[..., :].mean((0, 1))[0]) if False else float((lead == 0).mean()), leader_has_max_energy_share=top_lead)


def run_net(mc, params):
    ecfg = mc.env_cfg(); mm = dataclasses.replace(mc, n_envs=NEP)
    data, _, _ = jax.jit(make_rollout(ecfg, mm))(params, jax.random.PRNGKey(777))
    return {k: np.asarray(v) for k, v in data.items()}, ecfg


def main():
    res = {}
    # references (same 512 reset keys as scripted baselines; marl rollout uses PRNGKey(777) -> pairing checked below via rot30 reward in baselines json)
    ecfg0 = B.v2_env(2.0)
    e30 = dataclasses.replace(ecfg0, t_hold_lo=30.0, t_hold_hi=30.0)
    d30 = B.run(e30, rule_blind(e30)); _, ep30 = B.analyse(d30, ecfg0)
    g = B.run(ecfg0, B.gate_policy(ecfg0, 15)); _, epg = B.analyse(g, ecfg0)
    res["_refs"] = dict(rot30=float(ep30.mean()), gate30s=float(epg.mean()))
    rnd_eps = None
    for cname, (algo, al, pre) in COND.items():
        rows = []
        for s in (21, 22, 23):
            p = f"runs/survival_diag/{pre}_s{s}"
            if not os.path.exists(p + "/diag.json"): continue
            dj = json.load(open(p + "/diag.json")); ts, mc, _ = load_ckpt(p + "/final.ckpt")
            d, ecfg = run_net(mc, ts.params)
            if rnd_eps is None:
                eps_ = []
                for k in [12345, 1, 2, 3]:
                    t0 = init_train(jax.random.PRNGKey(k), ecfg, mc); dd, _ = run_net(mc, t0.params); eps_.append(dd["rew"].sum(0).mean(-1))
                rnd_eps = np.mean(eps_, 0); res["_refs"]["random_net_mean_this_script"] = float(rnd_eps.mean())
            ep = d["rew"].sum(0).mean(-1); ci = float(1.96 * ep.std(ddof=1) / np.sqrt(len(ep)))
            m = survival_metrics(d, ecfg); died = (d["e_frac"] <= EPS)[-1].any(-1)
            sw = dj["swap_by_decile"]; last = sw[-1]
            beat = float(ep.mean()) >= RAND_BASE + 2 * ci
            verdict = "维持" if (last >= 0.01 and beat) else ("塌缩" if last < 0.005 else "中间")
            em = energy_metrics(d["slot"], d["e_frac"], ecfg)
            if verdict == "维持":
                if em["n_swaps_3alive"] < 100: aw = "无法判定"
                else: aw = "能量感知" if em["incoming_is_max"] >= 0.5 else ("能量盲" if em["incoming_is_max"] <= 0.4 else "中间")
            else: aw = "n/a"
            pd30, pdg, pdr = ep - ep30, ep - epg, ep - rnd_eps
            rows.append(dict(seed=s, verdict=verdict, awareness=aw, reward=float(ep.mean()), ci95=ci, arrival=m["arrival_rate"], all_arrive=m["all_arrive"],
                             death_any=float(died.mean()), cls=classify_survival(m), swap_rate_eval=m["swap_rate"], last_decile_swap=last, beat_random_2ci=bool(beat),
                             ret_by_decile=[x / 2 for x in dj["ret_by_decile"]], ent_by_decile=dj["ent_by_decile"], swap_by_decile=sw, wall_s=dj["wall_s"],
                             energy=em, minus_rot30=float(pd30.mean()), minus_gate30=float(pdg.mean()), minus_random=float(pdr.mean())))
        r = dict(algo=algo, alpha=al, runs=rows)
        if rows:
            kv = sum(1 for x in rows if x["verdict"] == "维持"); kc = sum(1 for x in rows if x["verdict"] == "塌缩")
            tgt = PRED[cname]
            k = kv if tgt == "维持" else kc if tgt == "塌缩" else sum(1 for x in rows if x["verdict"] != "维持")
            r["n_sustain"], r["n_collapse"], r["pred"], r["k_match"] = kv, kc, tgt, k
            r["pred_result"] = ("成立" if k == 3 else "多数" if k == 2 else "不成立") + f" ({k}/{len(rows)})"
            for key in ("reward", "minus_rot30", "minus_gate30", "minus_random"): r[key + "_seedlevel"] = tci([x[key] for x in rows])
            sus = [x for x in rows if x["verdict"] == "维持"]
            if sus:
                nav = sum(1 for x in sus if x["awareness"] == "能量感知")
                im = [x["energy"]["incoming_is_max"] for x in sus if x["energy"]["incoming_is_max"] is not None]
                mean, ci_ = tci(im) if len(im) > 1 else (float(np.mean(im)) if im else None, None)
                r["aware_runs"] = f"{nav}/{len(sus)}"; r["incoming_max_seedlevel"] = (mean, ci_)
                lb = (mean - ci_) if ci_ is not None else None
                r["aware_condition"] = ("显著" if (nav / len(sus) >= 2 / 3 and lb is not None and lb > 0.345) else ("倾向" if mean and mean > 0.40 else "未达"))
        res[cname] = r
    json.dump(res, open("calib/out/survival_v2_seeds.json", "w"), indent=1)
    print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk != "runs"} for k, v in res.items() if k != "_refs"}, indent=1, ensure_ascii=False))
    for c, v in res.items():
        if c == "_refs": continue
        for x in v["runs"]: print(c, x["seed"], x["verdict"], x["awareness"], f"R={x['reward']:.4f}±{x['ci95']:.4f}", x["cls"], "inc_max", x["energy"]["incoming_is_max"], "n", x["energy"]["n_swaps_3alive"], "lastsw", round(x["last_decile_swap"], 4))


if __name__ == "__main__":
    main()
