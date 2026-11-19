"""M4br exp1b: RL best response against two scripted ledger reciprocators (kappa=1, N=3, mutual).
usage: run_m4br_br.py --cfg self|id2|id4|id1000 --seed 0 [--steps 5e6]   (RL bird = bird 0)"""
import argparse, json, os
os.environ.setdefault("JAX_PLATFORMS", "cpu")
import jax, jax.numpy as jnp, numpy as np
import animalbench  # noqa
from animalbench.marl import MarlConfig
from animalbench.marl.br import make_rollout_br, train_br
from animalbench.marl.basin import episode_metrics, classify
from animalbench.scenarios.geese.scripted_ledger import ledger_reciprocator

CFG = {"self": ("self", 0), "id2": ("identity", 2), "id4": ("identity", 4), "id1000": ("identity", 1000)}
ap = argparse.ArgumentParser()
ap.add_argument("--cfg", required=True); ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--steps", type=float, default=5e6); ap.add_argument("--n_envs", type=int, default=128)
ap.add_argument("--root", default="runs/m4br/exp1/br")
a = ap.parse_args()
ver, lam = CFG[a.cfg]
out = os.path.join(a.root, f"{a.cfg}_s{a.seed}"); os.makedirs(out, exist_ok=True)
m1 = MarlConfig(n_agents=1, n_envs=a.n_envs, swap_rule="mutual", kappa=1.0, shared=True, memory="gru",
                random_slots=True, total_agent_steps=int(a.steps))
ecfg = m1.env_cfg().replace(n_max=3)
pol = ledger_reciprocator(ecfg, ver, lam, dev_id=0, dev_kind=-1)
ts, ecfg, _ = train_br(m1, pol, 100 * a.seed + 7, out)


def streak_success(ready, act):          # ready, act: (T,B). A streak = maximal run of ready decisions; success if its last decision was FWD
    T, B = ready.shape
    n = s = 0
    for b in range(B):
        t = 0
        while t < T:
            if ready[t, b]:
                u = t
                while u + 1 < T and ready[u + 1, b]:
                    u += 1
                n += 1; s += int(act[u, b] == 2); t = u + 1
            else:
                t += 1
    return (s / n if n else float("nan")), n


def evaluate(greedy):
    mm = m1.__dict__ | {"n_envs": 128}
    d, _, _ = jax.jit(make_rollout_br(ecfg, MarlConfig(**mm), pol, 0, 128, greedy))(ts.params, jax.random.PRNGKey(777))
    d = {k: np.asarray(v) for k, v in d.items()}
    r = episode_metrics(d["slot3"], d["n_swaps"], d["rew3"], ecfg.delta_h, ecfg.T)
    r["class"] = classify(r); r["greedy"] = greedy
    ready, due, act = d["ready"], d["due"], d["a_rl_"]
    r["takeover_per_decision"] = float(((act == 2) & ready).sum() / max(ready.sum(), 1))
    r["takeover_per_streak"], r["n_streaks"] = streak_success(ready, act)
    r["n_ready_per_ep"] = float(ready.sum() / ready.shape[1])
    r["retreat_when_due"] = float(((act == 1) & due).sum() / max(due.sum(), 1))
    r["n_due_per_ep"] = float(due.sum() / due.shape[1])
    r["rl_lead_share"] = float((d["slot3"][..., 0] == 0).mean())
    r["rl_ret"] = float(d["rew3"][..., 0].sum(0).mean() * ecfg.delta_h)
    r["others_ret"] = float(d["rew3"][..., 1:].sum(0).mean() * ecfg.delta_h)
    r["rl_swaps_in"] = float(((d["slot3"][..., 0] >= 1) & (act == 2) & ready).sum() / ready.shape[1])
    return r


res = dict(cfg=a.cfg, version=ver, lam=lam, seed=a.seed, steps=a.steps, sampled=evaluate(False), greedy=evaluate(True))
json.dump(res, open(os.path.join(out, "eval.json"), "w"), indent=1)
s = res["sampled"]
print("FINAL", {k: (round(v, 3) if isinstance(v, float) else v) for k, v in s.items()}, flush=True)
