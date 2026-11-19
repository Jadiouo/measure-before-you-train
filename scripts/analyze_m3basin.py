"""Aggregate runs/m3basin/*: classification table (summary.json), learning-curve and classification figures."""
import glob, json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from animalbench.marl.basin import classify, THR

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
RUNS = os.path.join(ROOT, "runs", "m3basin")
FIGS = os.path.join(ROOT, "notes", "impl", "m3basin_figs")
os.makedirs(FIGS, exist_ok=True)
KT = json.load(open(os.path.join(RUNS, "kappa_threshold.json")))
CELLS = ["A", "B", "C1", "C2", "D1", "D2", "E"]
KAPPA = {"A": 0.0, "B": 0.0, "C1": 1.0, "C2": 8.0, "D1": 1.0, "D2": 8.0, "E": 8.0}
DESC = {"A": "k=0 random", "B": "k=0 BC-warm", "C1": "k=1 random", "C2": "k=8 random", "D1": "k=1 BC-warm",
        "D2": "k=8 BC-warm", "E": "unilateral k=8 random"}


def load_log(d):
    rows = [json.loads(l) for l in open(os.path.join(d, "log.jsonl"))]
    return {k: np.array([r[k] for r in rows]) for k in rows[0]}


def window_class(L, lo, hi):
    n = len(L["ret"]); sl = slice(int(lo * n), int(hi * n))
    m = {k: float(L[k][sl].mean()) for k in ("swap_rate", "max_lead_share", "min_lead_share", "lead_entropy")}
    return classify(m)


rows = {}
for d in sorted(glob.glob(os.path.join(RUNS, "*_s[0-9]"))):
    n = os.path.basename(d)
    if not os.path.exists(os.path.join(d, "eval.json")):
        continue
    ev = json.load(open(os.path.join(d, "eval.json")))
    L = load_log(d)
    s = ev["sampled"]
    k = str(KAPPA[ev["cell"]])
    kt = KT.get(k)
    row = dict(cell=ev["cell"], seed=ev["seed"], kappa=ev["kappa"], rule=ev["rule"], warm=ev["warm"],
               cls=s["class"], cls_greedy=ev["greedy"]["class"],
               cls_log_last10=window_class(L, 0.9, 1.0), cls_log_80_90=window_class(L, 0.8, 0.9),
               swap_rate=s["swap_rate"], max_lead_share=s["max_lead_share"], min_lead_share=s["min_lead_share"],
               lead_entropy=s["lead_entropy"], median_tenure_s=s["median_tenure_s"], ret_bird=s["ret_bird"],
               ret_worst_bird=s["ret_worst_bird"], steps=float(L["agent_steps"][-1]))
    row["treadmill"] = bool(row["cls"] == "rotation" and row["swap_rate"] >= 0.1)     # hot potato at the lock-limited max rate
    row["stable"] = row["cls_log_last10"] == row["cls_log_80_90"] == row["cls"]
    if kt and ev["rule"] == "mutual":
        row["ref_fixed"] = kt["fixed"]
        row["ref_rot_S2"] = kt["rot"]["0.94/0.9"]["ret"]
        row["vs_fixed"] = row["ret_bird"] - kt["fixed"]
        row["vs_scripted_rot"] = row["ret_bird"] - kt["rot"]["0.94/0.9"]["ret"]
    bcp = os.path.join(d, "bc_eval.json")
    if os.path.exists(bcp):
        b = json.load(open(bcp))
        row["bc_cls"] = b["sampled"]["class"]; row["bc_swap_rate"] = b["sampled"]["swap_rate"]
        row["bc_min_lead_share"] = b["sampled"]["min_lead_share"]; row["bc_ret"] = b["sampled"]["ret_bird"]
        row["bc_acc"] = b["final_acc"]
    rows[n] = row

cnt = {}
for c in CELLS:
    rs = [r for r in rows.values() if r["cell"] == c]
    if rs:
        cnt[c] = dict(n=len(rs), fixed=sum(r["cls"] == "fixed" for r in rs), rotation=sum(r["cls"] == "rotation" for r in rs), of_which_treadmill=sum(r["treadmill"] for r in rs),
                      other=sum(r["cls"] == "other" for r in rs), unstable=sum(not r["stable"] for r in rs),
                      mean_ret_bird=float(np.mean([r["ret_bird"] for r in rs])),
                      mean_swap_rate=float(np.mean([r["swap_rate"] for r in rs])))
rer = {}
for d in sorted(glob.glob(os.path.join(ROOT, "runs", "m3basin_20M", "*_s[0-9]"))):
    if os.path.exists(os.path.join(d, "eval.json")):
        ev = json.load(open(os.path.join(d, "eval.json"))); s_ = ev["sampled"]; L = load_log(d)
        rer[os.path.basename(d)] = dict(cls=s_["class"], swap_rate=s_["swap_rate"], max_lead_share=s_["max_lead_share"],
                                         min_lead_share=s_["min_lead_share"], ret_bird=s_["ret_bird"],
                                         cls_log_last10=window_class(L, 0.9, 1.0), cls_log_80_90=window_class(L, 0.8, 0.9),
                                         cls_10M=rows[os.path.basename(d)]["cls"])
json.dump(dict(thresholds=THR, counts=cnt, runs=rows, rerun_20M=rer), open(os.path.join(RUNS, "summary.json"), "w"), indent=1)
for c, v in cnt.items():
    print(c, DESC[c], v)
for n, v in rer.items():
    print('20M', n, v)

# ---- figures
col = plt.cm.tab10
for key, ylabel, fn in (("ret", "return per bird (P_solo*s / 600 s)", "curves_return.png"),
                        ("swap_rate", "leader swap rate (1/s)", "curves_swap_rate.png"),
                        ("min_lead_share", "min lead share per episode", "curves_min_lead_share.png")):
    fig, axs = plt.subplots(2, 4, figsize=(17, 7), sharex=True)
    for ax, c in zip(axs.ravel(), CELLS):
        for n, r in rows.items():
            if r["cell"] != c:
                continue
            L = load_log(os.path.join(RUNS, n))
            y = np.convolve(np.pad(L[key], (7, 0), mode="edge"), np.ones(8) / 8, mode="valid")
            ax.plot(L["agent_steps"] / 1e6, y, color=col(r["seed"]), lw=1, label=f"s{r['seed']} {r['cls']}")
        if key == "ret" and c in ("A", "B", "C1", "C2", "D1", "D2"):
            kt = KT[str(KAPPA[c])]
            ax.axhline(kt["fixed"], color="k", ls="--", lw=0.8); ax.axhline(kt["rot"]["0.94/0.9"]["ret"], color="g", ls=":", lw=0.8)
        ax.set_title(f"{c}: {DESC[c]}", fontsize=9); ax.legend(fontsize=6); ax.set_xlabel("agent steps (M)")
        ax.set_ylabel(ylabel, fontsize=8)
    axs.ravel()[-1].axis("off")
    fig.suptitle(key + ("  (dashed: scripted fixed leader; dotted: scripted S2 rotation)" if key == "ret" else ""))
    fig.tight_layout(); fig.savefig(os.path.join(FIGS, fn), dpi=110); plt.close(fig)

fig, ax = plt.subplots(figsize=(9, 3.6))
cs = [c for c in CELLS if c in cnt]
bot = np.zeros(len(cs))
for k, cc in (("fixed", "#c44"), ("rotation", "#4a4"), ("other", "#999")):
    v = np.array([cnt[c][k] for c in cs]); ax.bar(range(len(cs)), v, bottom=bot, color=cc, label=k)
    for i, (b, x) in enumerate(zip(bot, v)):
        if x:
            ax.text(i, b + x / 2, str(x), ha="center", va="center", color="w")
    bot += v
ax.set_xticks(range(len(cs))); ax.set_xticklabels([f"{c}\n{DESC[c]}" for c in cs], fontsize=7)
ax.set_ylabel("runs"); ax.legend(); ax.set_title("final classification (pre-registered thresholds)")
fig.tight_layout(); fig.savefig(os.path.join(FIGS, "classification_table.png"), dpi=110)
