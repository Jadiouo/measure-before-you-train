"""M2 gate: classify the intervention rollouts of scripts/m2gate_rollout.py (I1 and I2), build the confusion matrices,
evaluate the gate (every gate row: hit rate >= 0.90; identity-free rows: P(identity) <= 0.10) and write
calib/out/intervention_confusion.json.  CPU/numpy only.   python scripts/m2gate_classify.py --dir runs/m2gate/cpu"""
import argparse
import json
import os
import time

import numpy as np

from animalbench.metrics import intervention as iv

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# (agent, I1 allowed labels, I2 allowed labels, role)
ROWS = [
    ("S1_treadmill", {"state-driven", "no-signal"}, {"tolerate"}, "gate"),
    ("S2_state_rotation", {"state-driven", "no-signal"}, {"tolerate"}, "gate"),
    ("S2f_state_rotation_fair", {"state-driven", "no-signal"}, {"tolerate"}, "gate"),
    ("S3c_reciprocity_code", {"identity"}, {"tolerate"}, "gate"),
    ("N_cyc", {"state-driven", "no-signal"}, {"tolerate"}, "gate"),
    ("N_rnd", {"state-driven", "no-signal"}, {"tolerate"}, "gate"),
    ("N_fix", {"state-driven", "no-signal"}, {"no-signal", "tolerate"}, "gate"),
    ("S3p_reciprocity_punish", {"identity"}, {"punish"}, "extra"),
    ("S2f_ia", {"state-driven", "no-signal"}, {"tolerate"}, "agent-fix"),
    ("N_cyc_ia", {"state-driven", "no-signal"}, {"tolerate"}, "agent-fix"),
    ("N_rnd_ia", {"state-driven", "no-signal"}, {"tolerate"}, "agent-fix"),
    ("M15_unilateral_shared_s0", {"state-driven", "no-signal"}, None, "real-policy-null"),
]
IDENTITY_FREE = {"S1_treadmill", "S2_state_rotation", "S2f_state_rotation_fair", "N_cyc", "N_rnd", "N_fix", "S2f_ia",
                 "N_cyc_ia", "N_rnd_ia", "M15_unilateral_shared_s0"}


def classify_agent(z, rng, K):
    conds = [str(c) for c in z["conds"]]
    ci = {c: i for i, c in enumerate(conds)}
    L, S_, D = z["leader"], z["slot"], z["D"]
    dh = float(z["dh"])
    N = int(L.max() + 1) if False else int(z["alive"].shape[-1])
    ns = L.shape[1]
    i1, i2, f1s, f2s = [], [], [], []
    t_ev = int(round(iv.T_EVENT / dh))
    for s in range(ns):
        lead = {c: L[ci["I1_" + c], s] for c in ("R", "S", "M", "RM")}
        f1 = iv.i1_features(lead, dh, N, rng, K)
        lab1 = iv.i1_label(f1)
        f1["label"], f1["mem_sub"] = lab1, iv.i1_memory_sublabel(f1)
        f1["diverged_R"] = iv.paired_divergence(lead["R"], lead["S"], dh)
        f1["diverged_M"] = iv.paired_divergence(lead["M"], lead["S"], dh)
        f1["label_v2"] = iv.i1_label_v2(f1, f1["diverged_R"])
        f1["set_changed_R"] = iv.set_changed(lead["R"], lead["S"], dh, N)
        f1["set_changed_M"] = iv.set_changed(lead["M"], lead["S"], dh, N)
        f1["label_v3"] = iv.i1_label_v3(f1, f1["set_changed_R"])
        rider = int(np.argmax(np.where(z["alive"][ci["I2_F"], s, t_ev], S_[ci["I2_F"], s, t_ev], -1)))
        f2 = iv.i2_features(L[ci["I2_F"], s], L[ci["I2_S"], s], D[ci["I2_F"], s], rider, dh, N, rng)
        f2["label"] = iv.i2_label(f2)
        f2["label_v2"] = iv.i2_label_v2(f2)
        i1.append(lab1); i2.append(f2["label"]); f1s.append(f1); f2s.append(f2)
    return i1, i2, f1s, f2s


def agg_label_i1(f1s):
    """Secondary, policy-level reading of spec 3.1: premise = >=50% of seeds above the N-perm 99th pct; deltas from the medians."""
    prem = np.mean([(np.isfinite(f["q_pre"]) and f["q_pre"] > f["null99"]) for f in f1s])
    if prem < 0.5:
        return "no-signal", float(prem)
    med = lambda k: float(np.nanmedian([f[k] for f in f1s]))
    f = dict(q_pre=med("q_pre"), null99=-1.0, null95=med("null95"), null_med=med("null_med"))
    for c in ("R", "S", "M", "RM"):
        f[f"q_post_{c}"] = med(f"q_post_{c}")
        f[f"delta_{c}"] = iv.delta(f["q_pre"], f[f"q_post_{c}"], f["null_med"])
    return iv.i1_label(f), float(prem)


# ------------------------------------------------------------------------------------------- FROZEN HOLD-OUT PROTOCOL
# (notes/impl/M2gate_prereg.md).  agent -> (I1 allowed labels (V3), I2 expected block label set or None, role)
HOLD_ROWS = [
    ("S1_treadmill", {"state-driven", "no-signal"}, {"tolerate"}, "gate"),
    ("S2_state_rotation", {"state-driven", "no-signal"}, {"tolerate"}, "gate"),
    ("S2f_ia", {"state-driven", "no-signal"}, {"tolerate"}, "gate"),
    ("N_cyc_ia", {"state-driven", "no-signal"}, {"tolerate"}, "gate"),
    ("N_rnd_ia", {"state-driven", "no-signal"}, {"tolerate"}, "gate"),
    ("N_fix", {"state-driven", "no-signal"}, {"no-signal", "tolerate"}, "gate"),
    ("S3c_reciprocity_code", {"identity"}, {"tolerate"}, "gate"),
    ("P_body", {"state-driven", "no-signal"}, None, "false-positive"),
    ("P_body_codenoise", {"state-driven", "no-signal"}, None, "false-positive"),
    ("S3p_reciprocity_punish", {"identity"}, {"punish"}, "positive-control-I2"),
    ("P_code_lookup", {"identity"}, None, "informational-lookup"),
    ("S2f_state_rotation_fair", {"state-driven", "no-signal"}, {"tolerate"}, "informational-literal"),
    ("N_cyc", {"state-driven", "no-signal"}, {"tolerate"}, "informational-literal"),
    ("N_rnd", {"state-driven", "no-signal"}, {"tolerate"}, "informational-literal"),
    ("M15_unilateral_shared_s0", {"state-driven", "no-signal"}, None, "informational-N3"),
]
HOLD_IDENTITY_FREE = {"S1_treadmill", "S2_state_rotation", "S2f_ia", "N_cyc_ia", "N_rnd_ia", "N_fix", "P_body", "P_body_codenoise"}


def main_holdout(a):
    res = dict(source=a.dir, K=a.K, protocol="notes/impl/M2gate_prereg.md", block_size=iv.BLOCK_SIZE,
               thresholds={k: getattr(iv, k) for k in dir(iv) if k.isupper() and not k.startswith("_")}, agents={})
    for ag, ok1, ok2, role in HOLD_ROWS:
        p = os.path.join(a.dir, f"{ag}.npz")
        if not os.path.exists(p):
            res["agents"][ag] = dict(role=role, missing=True)
            continue
        z = np.load(p)
        rng = np.random.default_rng(12345)
        l1, l2, f1, f2 = classify_agent(z, rng, a.K)
        n = len(l1)
        l3 = [f["label_v3"] for f in f1]
        blocks = [iv.i2_label_block(f2[i:i + iv.BLOCK_SIZE]) for i in range(0, n, iv.BLOCK_SIZE)]
        d = dict(role=role, n=n, i1_allowed=sorted(ok1), i2_allowed=None if ok2 is None else sorted(ok2),
                 i1v3_counts={c: l3.count(c) for c in iv.I1_LABELS}, i1v3_hit=float(np.mean([l in ok1 for l in l3])),
                 i1v3_identity_frac=l3.count("identity") / n,
                 i1v2_counts={c: [f["label_v2"] for f in f1].count(c) for c in iv.I1_LABELS},
                 i1_literal_counts={c: l1.count(c) for c in iv.I1_LABELS},
                 set_changed_R_frac=float(np.mean([f["set_changed_R"] for f in f1])),
                 set_changed_M_frac=float(np.mean([f["set_changed_M"] for f in f1])),
                 diverged_R_frac=float(np.mean([f["diverged_R"] for f in f1])), diverged_M_frac=float(np.mean([f["diverged_M"] for f in f1])),
                 premise_frac=float(np.mean([(np.isfinite(f["q_pre"]) and f["q_pre"] > f["null99"]) for f in f1])),
                 i2_block_labels=blocks, i2_block_counts={c: blocks.count(c) for c in iv.I2_LABELS},
                 i2_block_hit=None if ok2 is None else float(np.mean([b in ok2 for b in blocks])),
                 i2_episode_v2_counts={c: [f["label_v2"] for f in f2].count(c) for c in iv.I2_LABELS},
                 i2_episode_v2_hit=None if ok2 is None else float(np.mean([f["label_v2"] in ok2 for f in f2])),
                 i2_ey_mean=float(np.nanmean([f["dy_F"] - f["dy_S"] for f in f2])), i2_el_mean=float(np.nanmean([f["dls_F"] - f["dls_S"] for f in f2])),
                 per_seed=dict(i1=[{k: (bool(v) if isinstance(v, np.bool_) else v) for k, v in f.items()} for f in f1],
                               i2=[{k: (bool(v) if isinstance(v, np.bool_) else v) for k, v in f.items()} for f in f2]))
        res["agents"][ag] = d
        print(f"{ag:26s} [{role}] I1v3 {d['i1v3_counts']} hit {d['i1v3_hit']:.2f} | I2 blocks {d['i2_block_counts']} hit {d['i2_block_hit']}", flush=True)
    A = res["agents"]
    def ok(name, key, thr=0.90):
        return key in A.get(name, {}) and A[name].get(key) is not None and A[name][key] >= thr
    g_i1 = {n: ok(n, "i1v3_hit") for n, _, _, r in HOLD_ROWS if r == "gate"}
    g_fp = {n: (n in A and not A[n].get("missing") and A[n]["i1v3_identity_frac"] <= 0.10) for n in sorted(HOLD_IDENTITY_FREE)}
    g_i2 = {n: ok(n, "i2_block_hit") for n, _, _, r in HOLD_ROWS if r == "gate"}
    pc = ok("S3p_reciprocity_punish", "i2_block_hit")
    lk = A.get("P_code_lookup", {})
    lookup_info = None if lk.get("missing") else dict(identity_frac=lk["i1v3_identity_frac"], diverged_M_frac=lk["diverged_M_frac"],
                                                       meets_prereg_info_criteria=bool(lk["i1v3_identity_frac"] >= 0.6 and lk["diverged_M_frac"] <= 0.1))
    res["verdict"] = dict(I1_gate_rows=g_i1, identity_free_rows_le_10pct=g_fp, I2_gate_rows=g_i2, I2_positive_control_S3p=pc, lookup_informational=lookup_info,
                          I1_pass=bool(all(g_i1.values())), false_positive_pass=bool(all(g_fp.values())), I2_pass=bool(all(g_i2.values()) and pc))
    res["verdict"]["gate_pass"] = bool(res["verdict"]["I1_pass"] and res["verdict"]["false_positive_pass"] and res["verdict"]["I2_pass"])
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=1, default=float)
    print("VERDICT", {k: v for k, v in res["verdict"].items() if k in ("I1_pass", "false_positive_pass", "I2_pass", "gate_pass")})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--out", default=os.path.join(ROOT, "calib", "out", "intervention_confusion.json"))
    ap.add_argument("--K", type=int, default=iv.K_NULL)
    ap.add_argument("--tag", default="")
    ap.add_argument("--holdout", action="store_true", help="frozen hold-out protocol (notes/impl/M2gate_prereg.md)")
    a = ap.parse_args()
    if a.holdout:
        return main_holdout(a)
    res = dict(source=a.dir, K=a.K, thresholds={k: getattr(iv, k) for k in dir(iv) if k.isupper() and not k.startswith("_")},
               i1=dict(columns=iv.I1_LABELS), i2=dict(columns=iv.I2_LABELS), agents={})
    for ag, ok1, ok2, role in ROWS:
        p = os.path.join(a.dir, f"{ag}.npz")
        if not os.path.exists(p):
            continue
        t = time.time()
        z = np.load(p)
        rng = np.random.default_rng(12345)
        l1, l2, f1, f2 = classify_agent(z, rng, a.K)
        n = len(l1)
        hit1 = float(np.mean([l in ok1 for l in l1]))
        l1v2 = [f["label_v2"] for f in f1]
        hit1v2 = float(np.mean([l in ok1 for l in l1v2]))
        hit2 = None if ok2 is None else float(np.mean([l in ok2 for l in l2]))
        l2v2 = [f["label_v2"] for f in f2]
        hit2v2 = None if ok2 is None else float(np.mean([l in ok2 for l in l2v2]))
        agg, prem = agg_label_i1(f1)
        res["agents"][ag] = dict(
            role=role, n=n, i1v2_counts={c: l1v2.count(c) for c in iv.I1_LABELS}, i1v2_hit=hit1v2,
            diverged_R_frac=float(np.mean([f["diverged_R"] for f in f1])), diverged_M_frac=float(np.mean([f["diverged_M"] for f in f1])),
            i1v2_identity_frac=l1v2.count("identity") / n, i1_counts={c: l1.count(c) for c in iv.I1_LABELS}, i2_counts={c: l2.count(c) for c in iv.I2_LABELS},
            i2v2_counts={c: l2v2.count(c) for c in iv.I2_LABELS}, i2v2_hit=hit2v2,
            i1_hit=hit1, i2_hit=hit2, i1_allowed=sorted(ok1), i2_allowed=None if ok2 is None else sorted(ok2),
            i1_identity_frac=l1.count("identity") / n, i1_mem_sub_among_identity={
                k: sum(f["mem_sub"] == k and f["label"] == "identity" for f in f1) for k in ("memory", "lookup", "other", "na")},
            i1_aggregate_label=agg, i1_premise_frac=prem,
            medians={k: float(np.nanmedian([f[k] for f in f1])) for k in ("W", "q_pre", "null99", "null95", "null_med", "q_post_R", "q_post_S",
                                                                     "q_post_M", "q_post_RM", "delta_R", "delta_S", "delta_M", "delta_RM")},
            i2_medians={k: float(np.nanmedian([f[k] for f in f2])) for k in ("y_pre", "y_post_F", "y_post_S", "dy_F", "dy_S", "dls_F", "bstar_mean")},
            i2_bstar_ok_frac=float(np.mean([f["bstar_ok"] for f in f2])),
            per_seed=dict(i1=[{k: (v if not isinstance(v, (np.floating, np.integer)) else float(v)) for k, v in f.items()} for f in f1],
                          i2=[{k: (v if not isinstance(v, (np.floating, np.integer, np.bool_)) else (bool(v) if isinstance(v, np.bool_) else float(v))) for k, v in f.items()} for f in f2]))
        res["agents"][ag]["per_seed"]["i1"] = [{k: (bool(v) if isinstance(v, np.bool_) else v) for k, v in f.items()} for f in res["agents"][ag]["per_seed"]["i1"]]
        print(f"{ag:28s} I1v2 hit {hit1v2:.2f} (div_R {res['agents'][ag]['diverged_R_frac']:.2f})")
        print(f"{ag:28s} I1 hit {hit1:.2f} {res['agents'][ag]['i1_counts']}  I2 hit {hit2} {res['agents'][ag]['i2_counts']}  | I2v2 hit {hit2v2} {res['agents'][ag]['i2v2_counts']}  ({time.time() - t:.0f}s)", flush=True)
    A = res["agents"]
    gate = [g for g, _, _, r in ROWS if r == "gate" and g in A]
    fix = {"S2f_state_rotation_fair": "S2f_ia", "N_cyc": "N_cyc_ia", "N_rnd": "N_rnd_ia"}

    def verdict(use_fix):
        rows = []
        for g in gate:
            name = fix.get(g, g) if use_fix and fix.get(g) in A else g
            rows.append((g, name))
        out = {}
        for g, name in rows:
            a_ = A[name]
            out[g] = dict(agent_used=name, i1_hit=a_["i1_hit"], i1v2_hit=a_["i1v2_hit"], i2_hit=a_["i2_hit"], i2v2_hit=a_["i2v2_hit"],
                          identity_frac_v2=a_["i1v2_identity_frac"] if g in IDENTITY_FREE else None,
                          identity_frac=a_["i1_identity_frac"] if g in IDENTITY_FREE else None)
        ok1 = all(v["i1_hit"] >= 0.90 for v in out.values())
        ok2 = all(v["i2_hit"] >= 0.90 for v in out.values())
        ok2v2 = all(v["i2v2_hit"] >= 0.90 for v in out.values())
        ok1v2 = all(v["i1v2_hit"] >= 0.90 for v in out.values())
        okidv2 = all(v["identity_frac_v2"] <= 0.10 for v in out.values() if v["identity_frac_v2"] is not None)
        okid = all(v["identity_frac"] <= 0.10 for v in out.values() if v["identity_frac"] is not None)
        return dict(rows=out, I1_pass=bool(ok1), I2_pass=bool(ok2), identity_free_le_10=bool(okid), gate_pass=bool(ok1 and ok2 and okid),
                    I1_v2_pass=bool(ok1v2), identity_free_v2_le_10=bool(okidv2), I2_v2_pass=bool(ok2v2), gate_pass_v2=bool(ok1v2 and ok2v2 and okidv2))
    res["gate_literal_agents"] = verdict(False)
    res["gate_agent_fixed"] = verdict(True)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=1, default=float)
    print("GATE literal :", {k: res["gate_literal_agents"][k] for k in ("I1_pass", "I2_pass", "identity_free_le_10", "gate_pass", "I1_v2_pass", "I2_v2_pass", "gate_pass_v2")})
    print("GATE agentfix:", {k: res["gate_agent_fixed"][k] for k in ("I1_pass", "I2_pass", "identity_free_le_10", "gate_pass", "I1_v2_pass", "I2_v2_pass", "gate_pass_v2")})


if __name__ == "__main__":
    main()
