"""aggregate runs/survival_diag/mp_* -> calib/out/survival_mappo_seeds.json + notes/impl/survival_mappo_results.md (verdicts per prereg)."""
import json, glob, os
BASE = {1.0: (0.987, 0.010), 0.5: (0.987, 0.010)}
b75 = "calib/out/survival_mappo_baseline_a075.json"
if os.path.exists(b75):
    j = json.load(open(b75)); BASE[0.75] = (j["mean"], j["ci95_mean_of_keys"])
COND = {"A": ("MAPPO", 1.0, "1e8"), "D": ("IPPO", 1.0, "5e7"), "C": ("MAPPO", 0.5, "5e7"), "B": ("MAPPO", 0.75, "5e7")}
f3 = lambda xs: " ".join(f"{x:.3f}" for x in xs)
f2 = lambda xs: " ".join(f"{x:.2f}" for x in xs)
res, md = {}, []
for c in "ADCB":
    algo, al, st = COND[c]; rows = []
    for s in (1, 2, 3):
        p = f"runs/survival_diag/mp_{c}_s{s}/diag.json"
        if not os.path.exists(p): continue
        d = json.load(open(p)); e = d["eval"]; sw = d["swap_by_decile"]; last = sw[-1]
        base, _ = BASE.get(al, (None, None))
        beat = base is not None and e["reward"] >= base + 2 * e["ci95"]
        v = "維持" if (last >= 0.01 and beat) else ("塌縮" if last < 0.005 else "中間")
        a5 = all(x >= 0.005 for x in sw[4:]) if c == "A" else None
        rows.append(dict(seed=s, ret_by_decile=[x / 2 for x in d["ret_by_decile"]], verdict=v, last_swap=last, beats_random_2ci=beat, a_no_collapse_5e7_to_end=a5, **{k: d[k] for k in ["ent_by_decile", "swap_by_decile", "wall_s"]}, eval=e))
    res[c] = dict(algo=algo, alpha=al, steps=st, baseline=BASE.get(al), runs=rows)
    md.append(f"## 條件 {c}：{algo} α={al}，{st} 步（隨機基準 {BASE.get(al)}）")
    md.append("| seed | 報酬±CI | 到達 | 換位/s(評估) | 任期s | 結構 | 最後十分位換位 | 判定 |\n|---|---|---|---|---|---|---|---|")
    for r in rows:
        e = r["eval"]
        md.append(f"| {r['seed']} | {e['reward']:.4f}±{e['ci95']:.4f} | {e['arrival']:.3f} | {e['swap_rate']:.3f} | {e['tenure_s']} | {e['cls']} | {r['last_swap']:.4f} | {r['verdict']} |")
    for r in rows:
        md.append(f"- seed {r['seed']} 十等分報酬 {f3(r['ret_by_decile'])}；熵 {f2(r['ent_by_decile'])}；換位 {f3(r['swap_by_decile'])}；wall {r['wall_s']:.0f}s")
    n = len(rows); vs = [r["verdict"] for r in rows]
    if n == 0:
        verdict = "未執行／未完成"
    elif c == "B":
        verdict = f"無預測（探索）：{vs}"
    else:
        tgt = "維持" if c == "A" else "塌縮"
        k = sum(1 for r in rows if r["verdict"] == tgt and (c != "A" or r["a_no_collapse_5e7_to_end"]))
        verdict = f"預測「{tgt}」：{k}/{n} 符合 → " + ("成立" if k == 3 else "多數" if k == 2 else "不成立")
    res[c]["verdict"] = verdict; md.append(f"\n**判定**：{verdict}\n")
json.dump(res, open("calib/out/survival_mappo_seeds.json", "w"), indent=1)
open("notes/impl/survival_mappo_results.md", "w").write("# 生存遷徙 MAPPO 多 seed 結果（對照 survival_mappo_prereg.md）\n邊緣格、隨機起點、評估 512 回合（種子 777）。報酬為每鳥平均 α 報酬。\n\n" + "\n".join(md) + "\n")
print("\n".join(md))
