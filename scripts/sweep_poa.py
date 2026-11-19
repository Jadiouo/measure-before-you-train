"""Sensitivity sweep of the "price of anarchy" (PoA) of the small game  (notes/impl/poa_sweep_decisions.md).

PoA := (mean per-bird cost of the unilateral hot-potato profile) / (mean per-bird cost of mutual fixed leader) - 1,
cost = -reward per decision, units P_solo*s (dh = 2 s per decision).  Costs are long-run (Cesaro) means of the exact
self-play chain of the pure symmetric profile; the equilibrium status of a profile is a separate flag.

Grid: N in {3,4,5} (exact slot-abstraction game; N=9 is not enumerable -> closed-form only, see below),
      t_lock in {4,6,8,12,16} s (lock_dec = t_lock/2 decisions), c_swap multiplier in {0.5,1,1.5,2},
      kappa in {0,0.5,1,2} (fatigue; evaluated on the kappa=0 profiles, NOT re-solved).
Solution concepts (per cell):
  * 'treadmill' = pure profile (BACK at the apex, STAY elsewhere) = the hot-potato rule of scripted agent S1.
    It is checked for the exact deviation gap (full-information deviator MDP) under two criteria: average reward
    (gamma=1-5e-4) and discounted gamma=0.99 (= PPO gamma).  gap<=1e-9 -> pure Nash equilibrium.
  * all pure symmetric equilibria are enumerated when the state space is small (nS<=5000, or N=5 at mult 1); else only treadmill.
  * N=4 has no pure symmetric equilibrium at baseline: the PoA reported there is the cost of the treadmill profile,
    which is an eps-equilibrium (eps = reported gap) -- NOT an equilibrium.
Closed form (inference, verified numerically below): swaps only come from the apex hot potato, one per lock period,
  so  PoA_formula = mean_{e in apex edges} total_swap_cost(e) * mult / (t_lock * sum_j p_j)
(the permutation of birds over slots leaves sum_j p_j dh unchanged; the PoA is the swap energy rate / flock power).
usage: python scripts/sweep_poa.py [--env 1]   (CPU only; run with a MemoryMax)"""
import argparse
import json
import os
import time

os.environ.setdefault("JAX_PLATFORMS", "cpu")
import numpy as np

from animalbench.analysis import small_game as sg

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
S, B, F = 0, 1, 2
DH = 2.0
PHI0, TAU_F = 0.94, 60.0
GAMMAS = {"avg": 1 - 5e-4, "g099": 0.99}


def scaled(cost, m):
    return {e: c * m for e, c in cost.items()}


def chain_cost(game, prof, burn=2000, n=18000):
    """Long-run mean cost per bird per decision of the pure symmetric profile (uniform over fresh states)."""
    pi = game.pure_pi(np.asarray(prof))
    sig = game.self_play_sigma(np.asarray(prof))
    P, R = game.matrices(pi)
    M = sp_zero(game.nS); r = np.zeros(game.nS)
    for a in range(3):
        import scipy.sparse as sp
        M = M + sp.diags((sig == a).astype(float)) @ P[a].tocsr(); r += (sig == a) * R[a]
    M.eliminate_zeros()
    MT = M.T.tocsr()
    d = np.zeros(game.nS); fs = game.fresh_states(); d[fs] = 1.0 / len(fs)
    tot = 0.0
    for t in range(burn + n):
        if t >= burn:
            tot += float(np.sum(d * r))      # (BLAS dot is pathologically slow here)
        d = MT @ d
        if t % 50 == 0:
            d[d < 1e-14] = 0.0          # avoid denormals (very slow)
    return -tot / n


def sp_zero(n):
    import scipy.sparse as sp
    return sp.csr_matrix((n, n))


def sim_profile(game, prof, kappa, steps=6000, burn=1500, seeds=4):
    """Monte-Carlo of the self-play chain with per-bird fatigue phi (EMA of p, tau=60 s, as the env), cost
    p*(1+kappa*max(0,phi-phi0))*dh + swap cost.  Returns mean per-bird cost per decision."""
    N, p = game.N, game.p
    alpha = 1 - np.exp(-DH / TAU_F)
    tot = []
    for sd in range(seeds):
        rng = np.random.default_rng(sd)
        slot = np.arange(N)                 # slot of each bird
        lock = np.zeros(N, int)
        phi = p.copy()
        ctr = 0
        acc = 0.0; cnt = 0
        for t in range(steps):
            occ = np.empty(N, int); occ[slot] = np.arange(N)        # bird in each slot
            act = np.array([prof[s] if lock[occ[s]] == 0 else S for s in range(N)])
            locked = lock[occ] > 0
            ns, sw, ctr, cs = game.resolve(act, locked, ctr, int(rng.integers(2)))
            new_slot = np.empty(N, int)
            for s in range(N):
                new_slot[occ[s]] = ns[s]
            swc = np.zeros(N)
            for s in range(N):
                swc[occ[s]] = cs[s]
            swapped = np.zeros(N, bool)
            for s in range(N):
                swapped[occ[s]] = sw[s]
            pw = p[new_slot] + swc / DH          # swap transit energy also counts as power (as in the env)
            phi = phi + alpha * (pw - phi)
            c = pw * (1 + kappa * np.maximum(0.0, phi - PHI0)) * DH
            lock = np.where(swapped, game.lock_dec - 1, np.maximum(lock - 1, 0))
            slot = new_slot
            if t >= burn:
                acc += c.mean(); cnt += 1
        tot.append(acc / cnt)
    return float(np.mean(tot))


def analyse_cell(N, t_lock, mult, p, cost, enum_all):
    L = int(round(t_lock / DH))
    out = dict(N=N, t_lock=t_lock, mult=mult, lock_dec=L)
    tread = np.array([B] + [S] * (N - 1))
    g_avg = sg.Game(N, p, scaled(cost, mult), "unilateral", lock_dec=L, gamma=GAMMAS["avg"])
    out["nS"] = int(g_avg.nS)
    c_uni = chain_cost(g_avg, tread)
    c_mut = float(np.mean(p) * DH)
    out.update(cost_treadmill=c_uni, cost_mutual_fixed=c_mut, poa_treadmill=c_uni / c_mut - 1)
    edges = [e for e in (1, 2)]
    tot = np.mean([cost[e].sum() * mult for e in edges])
    out["poa_formula"] = float(tot / (t_lock * np.sum(p)))
    for key, gam in GAMMAS.items():
        g = g_avg if key == "avg" else sg.Game(N, p, scaled(cost, mult), "unilateral", lock_dec=L, gamma=gam)
        r = g.check_pure(tread)
        out[f"treadmill_gap_{key}"] = r["gap"]
        out[f"treadmill_is_eq_{key}"] = bool(r["gap"] <= 1e-9)
        if enum_all:
            eqs = g.equilibria()
            out[f"n_pure_eq_{key}"] = len(eqs)
            costs = [chain_cost(g, np.array(e[0])) for e in eqs]
            out[f"eq_profiles_{key}"] = [[int(x) for x in e[0]] for e in eqs]
            if costs:
                out[f"poa_eq_min_{key}"] = min(costs) / c_mut - 1
                out[f"poa_eq_max_{key}"] = max(costs) / c_mut - 1
        else:
            out[f"n_pure_eq_{key}"] = None
    # mutual: fixed leader is an equilibrium?
    gm = sg.Game(N, p, scaled(cost, mult), "mutual", lock_dec=L, gamma=GAMMAS["g099"])
    out["mutual_fixed_is_eq_g099"] = bool(gm.check_pure(np.zeros(N, int))["gap"] <= 1e-9)
    # kappa (profile cost re-evaluation, no re-solve)
    gk = {}
    for kappa in ((0.0, 0.5, 1.0, 2.0) if (t_lock == 8 and mult == 1.0) else ()):
        cu = sim_profile(g_avg, tread, kappa)
        cm = sim_profile(g_avg, np.zeros(N, int), kappa)
        gk[str(kappa)] = dict(cost_treadmill=cu, cost_fixed=cm, poa=cu / cm - 1)
    out["kappa"] = gk
    return out


def env_check(points, seeds=(0, 1, 2)):
    import jax
    import jax.numpy as jnp
    from animalbench.core import compile_schedule
    from animalbench.scenarios.geese import GeeseConfig, make_rollout, stay_policy, treadmill
    res = []
    for (N, t_lock, kappa) in points:
        cfg = GeeseConfig(n_max=N, T=600.0, swap_rule="unilateral", t_lock=float(t_lock), kappa=float(kappa))
        sched = compile_schedule([], cfg.e_max)
        tot = {}
        for name, pol in (("treadmill", treadmill(cfg)), ("fixed", stay_policy(cfg))):
            f = jax.jit(make_rollout(cfg, pol))
            v = []
            for sd in seeds:
                st, r = f(jax.random.PRNGKey(sd), jnp.int32(N), sched)
                v.append(-float(r["reward"].sum(0).mean()))
            tot[name] = float(np.mean(v)); tot[name + "_sd"] = float(np.std(v))
        res.append(dict(N=N, t_lock=t_lock, kappa=kappa, cost_treadmill=tot["treadmill"], cost_fixed=tot["fixed"],
                        poa_env=tot["treadmill"] / tot["fixed"] - 1, sd=(tot["treadmill_sd"], tot["fixed_sd"])))
        print("env", res[-1], flush=True)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", type=int, default=1)
    a = ap.parse_args()
    params = json.load(open(sg.PARAMS_PATH))
    cells = []
    t0 = time.time()
    for N in (3, 4, 5):
        p, cost = sg.load_params(N)
        for t_lock in (4, 6, 8, 12, 16):
            L = t_lock // 2
            if N == 5 and L > 4:
                continue                      # nL^5*10 states x 81 tuples: too heavy for the memory budget
            for mult in (0.5, 1.0, 1.5, 2.0):
                nS = L ** N * 2 * N
                enum_all = nS <= 5000 or (N == 5 and mult == 1.0)
                cells.append(analyse_cell(N, t_lock, mult, p, cost, enum_all))
                c = cells[-1]
                print(f"N={N} lock={t_lock} m={mult} poa={c['poa_treadmill']:.4f} formula={c['poa_formula']:.4f} "
                      f"eq_avg={c['treadmill_is_eq_avg']} eq_099={c['treadmill_is_eq_g099']} "
                      f"[{time.time()-t0:.0f}s]", flush=True)
    # N=9: closed form only
    p9, cost9 = sg.load_params(9)
    n9 = []
    for t_lock in (4, 6, 8, 12, 16):
        for mult in (0.5, 1.0, 1.5, 2.0):
            tot = np.mean([cost9[e].sum() * mult for e in (1, 2)])
            n9.append(dict(N=9, t_lock=t_lock, mult=mult, poa_formula=float(tot / (t_lock * np.sum(p9)))))
    env = env_check([(3, 8, 0), (3, 4, 0), (3, 16, 0), (4, 8, 0), (5, 8, 0), (9, 8, 0), (3, 8, 1.0)]) if a.env else []
    out = dict(dh=DH, gammas=GAMMAS, cells=cells, n9_formula_only=n9, env_check=env,
               notes="poa_treadmill = exact chain cost ratio of profile (BACK at apex, STAY elsewhere); "
                     "poa_formula = swap-energy-rate / flock-power closed form; kappa block re-evaluates the kappa=0 "
                     "profiles under fatigue (not re-solved); N=9 closed form only (inference).")
    os.makedirs(os.path.join(ROOT, "calib", "out"), exist_ok=True)
    json.dump(out, open(os.path.join(ROOT, "calib", "out", "poa_sweep.json"), "w"), indent=1, default=float)
    plot(out)


def plot(out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cells = out["cells"]
    fig, ax = plt.subplots(2, 2, figsize=(11, 8))
    cols = {0.5: "#9ecae1", 1.0: "#3182bd", 1.5: "#e6550d", 2.0: "#a63603"}
    # (a) N=3: PoA vs t_lock, per mult
    a = ax[0, 0]
    for m, c in cols.items():
        pts = sorted([x for x in cells if x["N"] == 3 and x["mult"] == m], key=lambda x: x["t_lock"])
        a.plot([x["t_lock"] for x in pts], [100 * x["poa_treadmill"] for x in pts], "-", color=c, label=f"c_swap x{m}")
        ok = [x for x in pts if x["treadmill_is_eq_g099"] and x["treadmill_is_eq_avg"]]
        a.plot([x["t_lock"] for x in ok], [100 * x["poa_treadmill"] for x in ok], "o", color=c)
        bad = [x for x in pts if not (x["treadmill_is_eq_g099"] and x["treadmill_is_eq_avg"])]
        a.plot([x["t_lock"] for x in bad], [100 * x["poa_treadmill"] for x in bad], "x", color=c, ms=9)
    for e in out["env_check"]:
        if e["N"] == 3 and e["kappa"] == 0:
            a.plot(e["t_lock"], 100 * e["poa_env"], "k*", ms=12, label="env S1 vs fixed" if e["t_lock"] == 4 else None)
    a.axhline(4, ls=":", c="gray"); a.set_xlabel("t_lock (s)"); a.set_ylabel("PoA (%)")
    a.set_title("N=3: PoA vs lock (o: eq under both criteria, x: not eq)"); a.legend(fontsize=7)
    # (b) PoA vs mult at lock 8
    b = ax[0, 1]
    for N, c in ((3, "#3182bd"), (4, "#31a354"), (5, "#756bb1")):
        pts = sorted([x for x in cells if x["N"] == N and x["t_lock"] == 8], key=lambda x: x["mult"])
        b.plot([x["mult"] for x in pts], [100 * x["poa_treadmill"] for x in pts], "o-", color=c, label=f"N={N}")
    pts9 = sorted([x for x in out["n9_formula_only"] if x["t_lock"] == 8], key=lambda x: x["mult"])
    b.plot([x["mult"] for x in pts9], [100 * x["poa_formula"] for x in pts9], "s--", color="gray", label="N=9 closed form")
    b.set_xlabel("c_swap multiplier"); b.set_ylabel("PoA (%)"); b.set_title("lock=8 s: PoA vs c_swap"); b.legend(fontsize=8)
    # (c) log-log vs lock, mult 1, all N + env
    c_ = ax[1, 0]
    for N, c in ((3, "#3182bd"), (4, "#31a354"), (5, "#756bb1")):
        pts = sorted([x for x in cells if x["N"] == N and x["mult"] == 1.0], key=lambda x: x["t_lock"])
        c_.plot([x["t_lock"] for x in pts], [100 * x["poa_treadmill"] for x in pts], "o-", color=c, label=f"N={N}")
    p9 = sorted([x for x in out["n9_formula_only"] if x["mult"] == 1.0], key=lambda x: x["t_lock"])
    c_.plot([x["t_lock"] for x in p9], [100 * x["poa_formula"] for x in p9], "s--", color="gray", label="N=9 closed form")
    for e in out["env_check"]:
        if e["kappa"] == 0:
            c_.plot(e["t_lock"], 100 * e["poa_env"], "k*", ms=11)
    c_.set_xscale("log"); c_.set_yscale("log"); c_.set_xlabel("t_lock (s)"); c_.set_ylabel("PoA (%)")
    c_.set_title("c_swap x1: PoA ~ 1/t_lock (stars: real env)"); c_.legend(fontsize=8)
    # (d) kappa
    d = ax[1, 1]
    for N, c in ((3, "#3182bd"), (4, "#31a354"), (5, "#756bb1")):
        x = [x for x in cells if x["N"] == N and x["t_lock"] == 8 and x["mult"] == 1.0][0]
        ks = sorted(x["kappa"], key=float)
        d.plot([float(k) for k in ks], [100 * x["kappa"][k]["poa"] for k in ks], "o-", color=c, label=f"N={N}")
    for e in out["env_check"]:
        if e["N"] == 3 and e["t_lock"] == 8:
            d.plot(e["kappa"], 100 * e["poa_env"], "k*", ms=11)
    d.axhline(0, c="gray", lw=0.8); d.set_xlabel("kappa"); d.set_ylabel("PoA of kappa=0 profiles (%)")
    d.set_title("fatigue kappa (profile cost, not re-solved)"); d.legend(fontsize=8)
    fig.tight_layout()
    os.makedirs(os.path.join(ROOT, "notes", "impl", "poa_figs"), exist_ok=True)
    fig.savefig(os.path.join(ROOT, "notes", "impl", "poa_figs", "poa_sweep.png"), dpi=130)


if __name__ == "__main__":
    main()
