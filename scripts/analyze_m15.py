"""Aggregate runs/m15/*: learning curves, final action distributions, M1.5 criteria table (writes runs/m15/summary.json)."""
import glob
import json
import os
import shutil

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
RUNS = os.path.join(ROOT, "runs", "m15")
FIGS = os.path.join(RUNS, "figs")
KEEP = os.path.join(ROOT, "notes", "impl", "m15_figs")
os.makedirs(FIGS, exist_ok=True)
os.makedirs(KEEP, exist_ok=True)


def load(name):
    rows = [json.loads(l) for l in open(os.path.join(RUNS, name, "log.jsonl"))]
    return {k: np.array([r[k] for r in rows]) for k in rows[0]}


def smooth(x, k=8):
    return np.convolve(np.pad(x, (k - 1, 0), mode="edge"), np.ones(k) / k, mode="valid")


names = sorted(d for d in os.listdir(RUNS) if os.path.exists(os.path.join(RUNS, d, "eval.json")))
logs = {n: load(n) for n in names}
ev = {n: json.load(open(os.path.join(RUNS, n, "eval.json"))) for n in names}
col = {"s0": "C0", "s1": "C1", "s2": "C2"}

# ---------------------------------------------------------------- criteria
summary = {}
for n in names:
    L, E = logs[n], ev[n]
    rule = E["rule"]
    k = max(5, len(L["ret"]) // 10)
    last, prev = L["ret"][-k:].mean(), L["ret"][-2 * k:-k].mean()
    finite = bool(L["finite"].min() == 1.0 and np.all(np.isfinite(L["ret"])))
    stable = abs(last - prev) <= 0.001 * abs(last)
    s = E["sampled"]
    cf = E["counterfactual"]
    row = dict(rule=rule, shared=E["shared"],
               a_finite=finite, a_stable=bool(stable), ret_last=float(last), ret_delta_last2=float(last - prev),
               b_p_back_apex=s["p_back_apex"], b_p_back_non=s["p_back_non"],
               b_gap=s["p_back_apex"] - s["p_back_non"], c_tv=s["tv"], c_tv_greedy=E["greedy"]["tv"],
               swap_rate=s["swap_rate"], leader_changes=s["leadership"]["leader_changes_per_ep"],
               max_lead_share=s["leadership"]["max_lead_share"], p_fwd_fol=s["p_fwd_fol"],
               d_policy=cf["policy"]["adv_disc"], d_policy_ci=cf["policy"]["ci_disc"],
               d_oneshot=cf["oneshot"]["adv_H"], d_oneshot_ci=cf["oneshot"]["ci_H"],
               wall=float(L["wall"][-1]), steps=float(L["agent_steps"][-1]))
    row["pass_a"] = row["a_finite"] and row["a_stable"]
    row["pass_b"] = row["b_gap"] >= 0.3
    row["pass_c"] = row["c_tv"] <= 0.3
    row["pass_d"] = row["d_policy_ci"][0] > 0 and row["d_oneshot_ci"][0] > 0
    row["pass_all"] = bool(row["pass_a"] and row["pass_b"] and row["pass_c"] and row["pass_d"])
    row["fixed_leader"] = bool(row["leader_changes"] <= 1.0 and row["max_lead_share"] >= 0.9)
    summary[n] = row
json.dump(summary, open(os.path.join(RUNS, "summary.json"), "w"), indent=1, default=float)

# ---------------------------------------------------------------- learning curves
for rule in ("unilateral", "mutual"):
    fig, ax = plt.subplots(1, 4, figsize=(18, 3.8))
    for n in names:
        if not n.startswith(rule):
            continue
        L = logs[n]
        sd = n.split("_")[-1]
        ls = "-" if "shared" in n else "--"
        x = L["agent_steps"] / 1e6
        for a, key, lab in zip(ax, ("ret", "swap_rate", "p_back_apex", "ent"),
                               ("return / bird (P_solo*s per 600 s)", "swap rate (1/s)", "P(back | apex, unlocked)", "policy entropy")):
            a.plot(x, smooth(L[key]), ls, color=col[sd], label=n.replace(rule + "_", ""))
            a.set_xlabel("agent-steps (M)")
            a.set_title(lab, fontsize=9)
    ax[2].axhline(1.0 if rule == "unilateral" else 0.0, color="k", lw=0.6, ls=":")
    if rule == "unilateral":
        ax[1].axhline(0.125, color="gray", lw=0.6, ls=":")      # max swap rate: 1 per 8 s lock
    ax[0].legend(fontsize=7)
    fig.suptitle(f"M1.5 N=3 IPPO, {rule} (solid=shared params, dashed=independent)")
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, f"curves_{rule}.png"), dpi=110)
    plt.close(fig)

# ---------------------------------------------------------------- final action distributions
fig, axes = plt.subplots(2, 2, figsize=(11, 6.5), sharey=True)
for i, rule in enumerate(("unilateral", "mutual")):
    for j, tag in enumerate(("shared", "indep")):
        a = axes[i, j]
        ns = [n for n in names if n.startswith(rule) and tag in n]
        w = 0.8 / max(len(ns), 1)
        for q, n in enumerate(ns):
            mp = ev[n]["sampled"]["stats"]["mean_probs"]
            for s in range(3):
                bottom = 0
                for act, c in zip(range(3), ("#999999", "#d62728", "#1f77b4")):
                    v = mp[str(s)][act]
                    a.bar(s + q * w - 0.4 + w / 2, v, w * 0.9, bottom=bottom, color=c,
                          label=("stay", "back", "fwd")[act] if (s == 0 and q == 0) else None)
                    bottom += v
        a.set_xticks(range(3)); a.set_xticklabels(["apex (slot 0)", "slot 1", "slot 2"])
        a.set_title(f"{rule}, {tag}: learned P(a | slot, unlocked), one bar per seed", fontsize=9)
axes[0, 0].legend(fontsize=7, loc="center right")
fig.tight_layout()
fig.savefig(os.path.join(FIGS, "action_dist.png"), dpi=110)
plt.close(fig)

for f in ("curves_unilateral.png", "curves_mutual.png", "action_dist.png"):
    shutil.copy(os.path.join(FIGS, f), os.path.join(KEEP, f))

# ---------------------------------------------------------------- print
for n, r in summary.items():
    print(n, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()})
