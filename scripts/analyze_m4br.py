"""Aggregate M4br: exp2 (PPO-bias controls) counts + exp1 (deviation table, RL best response) + figures."""
import glob, json, os, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
R = os.path.join(ROOT, "runs", "m4br"); F = os.path.join(ROOT, "notes", "impl", "m4br_figs"); os.makedirs(F, exist_ok=True)
L = lambda d: {k: np.array([r[k] for r in (json.loads(l) for l in open(os.path.join(d, "log.jsonl")))]) for k in json.loads(open(os.path.join(d, "log.jsonl")).readline())}
summary = {}

# ---------------- exp2
for root, cells, tag in [(os.path.join(R, "exp2"), ["ent05", "ent10lr", "team"], "exp2"),
                         (os.path.join(R, "exp2x", "team_k8x"), ["team"], "exp2x_team_kappa8")]:
    cnt = {}
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
    for c in cells:
        runs = sorted(glob.glob(os.path.join(root, f"{c}_s*")))
        rows = []
        for d in runs:
            if not os.path.exists(os.path.join(d, "eval.json")): continue
            e = json.load(open(os.path.join(d, "eval.json")))["sampled"]
            lg = L(d); n = len(lg["ret"]); w = slice(int(.9 * n), n)
            rows.append(dict(run=os.path.basename(d), cls=e["class"], min_share=e["min_lead_share"], max_share=e["max_lead_share"],
                             swap=e["swap_rate"], ent=e["lead_entropy"], ret=e["ret_bird"],
                             train_last10_min=float(lg["min_lead_share"][w].mean()), train_last10_max=float(lg["max_lead_share"][w].mean())))
            ax[0].plot(lg["agent_steps"] / 1e6, lg["max_lead_share"], label=os.path.basename(d), lw=1)
            ax[1].plot(lg["agent_steps"] / 1e6, lg["min_lead_share"], lw=1)
        cnt[c] = dict(n=len(rows), fixed=sum(r["cls"] == "fixed" for r in rows), rotation=sum(r["cls"] == "rotation" for r in rows),
                      other=sum(r["cls"] == "other" for r in rows), min_share_gt_0p133=sum(r["min_share"] > 0.4 / 3 for r in rows), runs=rows)
    ax[0].set_title("max_lead_share (train)"); ax[1].set_title("min_lead_share (train)"); ax[0].set_xlabel("agent steps (M)"); ax[1].axhline(0.2, ls=":", c="k")
    ax[0].legend(fontsize=6); fig.tight_layout(); fig.savefig(os.path.join(F, f"{tag}_curves.png"), dpi=110); plt.close(fig)
    summary[tag] = cnt

# ---------------- exp1a deviations
dp = os.path.join(R, "exp1", "deviations_kappa1.json")
if os.path.exists(dp):
    D = json.load(open(dp)); summary["exp1_deviations"] = D
    names = [k for k in D["self_lam0"] if k != "baseline"]
    fig, ax = plt.subplots(figsize=(9, 3.8)); wdt = 0.2
    for i, cfg in enumerate(D):
        g = [D[cfg][n]["dev_gain"] for n in names]; se = [D[cfg][n]["dev_gain_se"] for n in names]
        ax.bar(np.arange(len(names)) + i * wdt, g, wdt, yerr=se, label=cfg)
    ax.axhline(0, c="k", lw=.6); ax.set_xticks(np.arange(len(names)) + 1.5 * wdt); ax.set_xticklabels(names, rotation=20)
    ax.set_ylabel("deviator gain vs complying (P_solo*s / 600 s)"); ax.legend(fontsize=7); fig.tight_layout()
    fig.savefig(os.path.join(F, "exp1_deviation_gain.png"), dpi=110); plt.close(fig)

# ---------------- exp1b RL best response
br = {}
fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
for cfg in ["self", "id2", "id4", "id1000"]:
    runs = sorted(glob.glob(os.path.join(R, "exp1", "br", f"{cfg}_s*")))
    rows = []
    for d in runs:
        if not os.path.exists(os.path.join(d, "eval.json")): continue
        E = json.load(open(os.path.join(d, "eval.json"))); s, g = E["sampled"], E["greedy"]
        lg = L(d)
        rows.append(dict(run=os.path.basename(d), takeover=s["takeover_per_decision"], takeover_streak=s["takeover_per_streak"],
                         takeover_greedy=g["takeover_per_decision"], retreat=s["retreat_when_due"], rl_ret=s["rl_ret"], others_ret=s["others_ret"],
                         rl_lead_share=s["rl_lead_share"], swap=s["swap_rate"], cls=s["class"], n_ready=s["n_ready_per_ep"],
                         verdict=("obey" if s["takeover_per_decision"] >= .8 else "disobey" if s["takeover_per_decision"] <= .2 else "between")))
        ax[0].plot(lg["agent_steps"] / 1e6, lg["takeover"], lw=1, label=os.path.basename(d)); ax[1].plot(lg["agent_steps"] / 1e6, lg["ret_rl"], lw=1)
    if rows:
        br[cfg] = dict(n=len(rows), mean_takeover=float(np.mean([r["takeover"] for r in rows])), obey=sum(r["verdict"] == "obey" for r in rows),
                       disobey=sum(r["verdict"] == "disobey" for r in rows), between=sum(r["verdict"] == "between" for r in rows), runs=rows)
ax[0].set_title("RL take-over rate (train, per decision)"); ax[0].axhline(.8, ls=":", c="g"); ax[0].axhline(.2, ls=":", c="r")
ax[1].set_title("RL bird return"); ax[1].axhline(-557.2, ls=":", c="g"); ax[1].axhline(-528.6, ls=":", c="r"); ax[0].legend(fontsize=5)
for a in ax: a.set_xlabel("RL steps (M)")
fig.tight_layout(); fig.savefig(os.path.join(F, "exp1_br_curves.png"), dpi=110); plt.close(fig)
summary["exp1_br"] = br
json.dump(summary, open(os.path.join(R, "summary.json"), "w"), indent=1)
for k, v in summary.items():
    if k.startswith("exp2"):
        for c, x in v.items(): print(k, c, {a: b for a, b in x.items() if a != "runs"})
for c, x in br.items():
    print("BR", c, {a: b for a, b in x.items() if a != "runs"})
    for r in x["runs"]: print("   ", {a: (round(b, 3) if isinstance(b, float) else b) for a, b in r.items()})
