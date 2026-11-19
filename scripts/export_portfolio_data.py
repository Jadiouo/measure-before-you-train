"""Export compact JSON for the portfolio page (web/portfolio/data/) from calib/out/*.json and git log.

Every number on the page must come from a file written here (or from scripts/record_soaring_replay.py);
the mapping page-number -> source file/field is in notes/impl/portfolio_numbers.md.
Run: .venv/bin/python scripts/export_portfolio_data.py   (python3 works too; numpy not needed)
"""
import json
import os
import statistics
import subprocess

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "calib/out")
DST = os.path.join(ROOT, "web/portfolio/data")
os.makedirs(DST, exist_ok=True)


def load(name):
    return json.load(open(os.path.join(OUT, name + ".json")))


def dump(name, obj):
    p = os.path.join(DST, name + ".json")
    s = json.dumps(obj, ensure_ascii=False, separators=(",", ":"))
    open(p, "w").write(s)
    sz = os.path.getsize(p)
    assert sz < 300_000, (name, sz)
    print(f"{name}.json  {sz} bytes")


def r(x, n=4):
    return None if x is None else round(float(x), n)


# ------------------------------------------------------------------ horseshoe (section 3)
def horseshoe():
    d = load("horseshoe_formation")
    items = []
    for key, v in d["sweep"].items():
        if v["form"] not in ("V", "echelon", "line"):
            continue
        items.append(dict(
            N=v["N"], form=v["form"], setting=v["setting"], kernel=f'{v["kind"]}_rc{v["rc"]}',
            o=v["overlap_o"], dx=v["dx_over_b"],
            mean=r(v["mean_ratio"]), mean_ahead=r(v["mean_ratio_ahead_only"]), mean_nearest=r(v["mean_ratio_nearest_only"]),
            pos=[[p["label"], r(p["x"], 3), r(p["y"], 3), r(p["ratio_all"]), r(p["ratio_ahead"]), r(p["ratio_nearest"])]
                 for p in v["per_position"]] if "per_position" in v else None))   # per-position only stored for rc=0.075
    cal = d["calibration"]
    mv = {(i["N"], i["form"], i["setting"], i["kernel"]): i["mean"] for i in items}
    vdiff = max(abs(mv[(n, "V", st, k)] - mv[(n, "echelon", st, k)]) for (n, f, st, k) in mv if f == "V")
    dump("horseshoe", dict(v_vs_echelon_max_abs_diff=vdiff, dump_note="mean = fleet mean of D_i/D_solo (all-wings superposition)",
        meta=d["meta"], items=items,
        calib=dict(maskew_V3=cal["maskew_3wing_V_x3_y0.89"],
                   eq8=cal["mean_vs_maskew_eq8_V_x3_y0.89"],
                   two_wing_opt=cal["two_wing_global_opt"])))


# ------------------------------------------------------------------ bucket (section 4)
BINS = [0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 1e9]   # percent


def hist(vals, flags):
    """vals in percent; flags True = LP upper bound timed out (dual bound used)."""
    ok = [0] * (len(BINS) - 1)
    to = [0] * (len(BINS) - 1)
    for v, f in zip(vals, flags):
        v = max(v, 0.0)
        for i in range(len(BINS) - 1):
            if BINS[i] <= v < BINS[i + 1]:
                (to if f else ok)[i] += 1
                break
    return dict(ok=ok, timeout=to, n=len(vals), gt2=sum(1 for v in vals if v > 2.0),
                gt2_ok=sum(1 for v, f in zip(vals, flags) if v > 2.0 and not f),
                median=r(statistics.median(vals), 3), max=r(max(vals), 3))


def timed_out(c):
    """LP bound not proven optimal: solver reported timeouts, or the dual bound differs from the incumbent."""
    return c["opt_timeouts"] > 0 or abs(c["opt"] - c["opt_incumbent"]) > 1e-9 * max(1.0, abs(c["opt"]))


def bucket():
    b0 = load("drone_bucket")["cells"]
    L0 = dict(
        tail=hist([100 * c["gap_tail"] for c in b0], [False] * len(b0)),
        gate=hist([100 * c["gap_gate"] for c in b0], [False] * len(b0)),
        n_hetero=sum(1 for c in b0 if c["h"] > 0),
        tail_gt2_hetero=sum(1 for c in b0 if c["h"] > 0 and c["gap_tail"] > 0.02),
        tail_gt2_homog=sum(1 for c in b0 if c["h"] == 0 and c["gap_tail"] > 0.02),
        tail_max=r(100 * max(c["gap_tail"] for c in b0), 2))
    b1 = load("drone_bucket_l1")["jobs"]
    cells1 = [c for j in b1 for c in j["cells"]]
    L1 = dict(best=hist([100 * c["best_gap"] for c in cells1], [timed_out(c) for c in cells1]),
              best_rule_counts={k: sum(1 for c in cells1 if c["best_rule"] == k) for k in ("fixed", "rot", "gate", "sort", "pair")},
              by_h={str(h): r(100 * statistics.mean(c["best_gap"] for c in cells1 if c["h"] == h), 3)
                    for h in sorted({c["h"] for c in cells1})})
    b23 = load("drone_bucket_l23")["jobs"]
    cells23 = [c for j in b23 for c in j["cells"]]
    fl = [timed_out(c) for c in cells23]
    L23 = dict(
        global_rule=hist([100 * c["best_global_gap"] for c in cells23], fl),
        local_bubble=hist([100 * c["rules"]["local"]["gap"] for c in cells23], fl),
        rot=hist([100 * c["rules"]["rot"]["gap"] for c in cells23], fl),
        n_timeout_cells=sum(fl),
        n_cells=len(cells23))
    # L3 summary (round 1 internal-review fix): trusted (bound not timed out) cells where rot > 2%.
    # best-local = min(rot, local) gap; global = best_global_gap (gate in all of them); diff in pp.
    ok23 = [c for c in cells23 if not timed_out(c)]
    bad = [c for c in ok23 if c["rules"]["rot"]["gap"] > 0.02]
    bl = [100 * min(c["rules"]["rot"]["gap"], c["rules"]["local"]["gap"]) for c in bad]
    gl = [100 * c["best_global_gap"] for c in bad]
    df = [b - g for b, g in zip(bl, gl)]
    L23["L3"] = dict(
        n_ok=len(ok23), n_rot_gt2_ok=len(bad),
        n_n5ar=sum(1 for c in bad if c["N"] == 5 and c["noise"] == "ar" and c["h"] == 0),
        n_bestlocal_gt2=sum(1 for x in bl if x > 2), n_global_gt2=sum(1 for x in gl if x > 2),
        local_min=r(min(bl), 2), local_max=r(max(bl), 2), glob_min=r(min(gl), 2), glob_max=r(max(gl), 2),
        diff_min=r(min(df), 2), diff_max=r(max(df), 2), diff_median=r(statistics.median(df), 2))
    dump("bucket", dict(bins=BINS[:-1] + [None], L0=L0, L1=L1, L23=L23,
                        anchors=dict(L1_theory_max_rel_err_lp=load("drone_bucket_l1")["anchor"]["theory_max_rel_err_lp"],
                                     L1_theory_max_rel_err_sim=load("drone_bucket_l1")["anchor"]["theory_max_rel_err_sim"])))


# ------------------------------------------------------------------ wake (section 5)
def wake():
    d = load("wake_headroom")
    by = {}
    for c in d["cells"]:
        by.setdefault(c["drift"], []).append(c)
    rows = []
    for dr, cs in by.items():
        def mm(key):
            v = [x[key] for x in cs]
            return [r(min(v), 2), r(statistics.median(v), 2), r(max(v), 2)]
        best_trad = [max(x["sav"].values()) for x in cs]
        rows.append(dict(
            drift=dr, n=len(cs),
            fixed=r(statistics.mean(x["sav"]["fixed"] for x in cs), 2),
            kf=r(statistics.mean(x["sav"]["kf"] for x in cs), 2),
            esc=r(statistics.mean(x["sav"]["esc"] for x in cs), 2),
            priv=r(statistics.mean(x["priv"] for x in cs), 2),
            priv_gps=r(statistics.mean(x["priv_gps"] for x in cs), 2),
            best_trad=r(statistics.mean(best_trad), 2),
            gap=mm("gap"), gap_gps=mm("gap_gpsprivileged"),
            ci_lo_gt2=sum(1 for x in cs if x["gap_ci"][0] > 2.0),
            ci_lo_gt2_gps=sum(1 for x in cs if x["gap_gpsprivileged_ci"][0] > 2.0),
            best_kf=sum(1 for x in cs if x["best"] == "kf"), best_fixed=sum(1 for x in cs if x["best"] == "fixed"),
            best_esc=sum(1 for x in cs if x["best"] == "esc")))
    allc = d["cells"]
    dump("wake", dict(rows=rows, n_cells=len(allc),
                      max_gap_gps_all=r(max(c["gap_gpsprivileged"] for c in allc), 2),
                      max_gap_random_drift=r(max(c["gap"] for c in allc if c["drift"].startswith("amp")), 2),
                      n_gps_lt2=sum(1 for c in allc if c["gap_gpsprivileged"] < 2.0),
                      anchors=d["anchors"],
                      best_counts={k: sum(1 for c in allc if c["best"] == k) for k in ("kf", "fixed", "esc")}))


# ------------------------------------------------------------------ soaring (section 6)
def soaring():
    d1 = load("soaring_headroom")
    n_cond = [x for x in d1["results"] if "|" in x["name"] and x["name"].count("|") == 3 and "anchor" not in x["name"]]
    summ = {}
    for N in ("5", "10"):
        for key, label in (("b_inf", "share_vs_noshare"), ("c_inf", "ref_vs_share")):
            tot = lo = pt = 0
            for x in n_cond:
                c = x["cells"].get(N)
                if not c:
                    continue
                g = c["gaps_Amin"][key]
                tot += 1
                lo += g[1] > 2.0
                pt += g[0] > 2.0
            summ[f"N{N}_{label}"] = dict(n=tot, lo_gt2=lo, pt_gt2=pt)
    # single-agent (a): N=1
    tot = lo = pt = 0
    for x in n_cond:
        c = x["cells"].get("1")
        if not c:
            continue
        g = c["gaps_Amin"]["ref_vs_noshare"]
        tot += 1
        lo += g[1] > 2.0
        pt += g[0] > 2.0
    summ["N1_ref_vs_rule"] = dict(n=tot, lo_gt2=lo, pt_gt2=pt)
    d3 = load("soaring_headroom3")
    rows = []
    for x in d3["results"]:
        for N, c in x["cells"].items():
            g = c["gaps"]["Emin"]
            rows.append(dict(
                cell=x["name"].rsplit("|", 1)[0], N=int(N), floor=bool(c["floor_cell"]), landed=r(c["landed_L1"], 2),
                emin={k: r(100 * c[k]["Emin"], 1) for k in ("L0", "L1", "L15a", "L15b", "L2")},
                amin={k: r(100 * c[k]["Amin"], 1) for k in ("L1", "L15a", "L15b", "L2")},
                gap={k: [r(v, 1) for v in g[k]] for k in ("L15a-L1", "L15b-L1", "L2-L1", "L2-max(L15a,L15b)")},
                cmp=g["cmp_eval"]))
    # share of L2-L1 recovered by L1.5b
    for row in rows:
        l2 = row["gap"]["L2-L1"][0]
        row["recov_b"] = r(100 * row["gap"]["L15b-L1"][0] / l2, 0) if abs(l2) > 1e-9 else None
    d2 = load("soaring_headroom2")
    rows2 = []
    for x in d2["results"]:
        for N, c in x["cells"].items():
            g = c["gap_vs_L1_Emin"]
            rows2.append(dict(cell=x["name"].rsplit("|", 1)[0], N=int(N),
                              L1_emin=r(100 * c["L1"]["Emin"], 1),
                              gap={k: [r(v, 1) for v in g[k]] for k in ("L0", "L2", "L3", "zone", "prior", "plan", "ref")}))
    r2s = {}
    for k in ("L0", "L2", "L3", "zone", "prior", "plan", "ref"):
        pts = [q["gap"][k][0] for q in rows2 if q["N"] == 5]
        r2s[k] = dict(min=min(pts), max=max(pts), lo_gt2=sum(1 for q in rows2 if q["N"] == 5 and q["gap"][k][1] > 2.0),
                      n=len(pts))
    r3s = dict(
        l2max_lo_gt2_nonfloor_N5=sum(1 for q in rows if q["N"] == 5 and not q["floor"] and q["gap"]["L2-max(L15a,L15b)"][1] > 2.0),
        n_nonfloor_N5=sum(1 for q in rows if q["N"] == 5 and not q["floor"]),
        l2max_lo_gt2_any=sum(1 for q in rows if q["gap"]["L2-max(L15a,L15b)"][1] > 2.0),
        n_cells=len(rows),
        l2max_min=min(q["gap"]["L2-max(L15a,L15b)"][0] for q in rows),
        l2max_max=max(q["gap"]["L2-max(L15a,L15b)"][0] for q in rows),
        l15a_min=min(q["gap"]["L15a-L1"][0] for q in rows), l15a_max=max(q["gap"]["L15a-L1"][0] for q in rows),
        l15a_wins=sum(1 for q in rows if q["gap"]["L15a-L1"][0] > 0))
    dump("soaring", dict(round1=summ, round2=rows2, round3=rows, round2_summary=r2s, round3_summary=r3s,
                         consts=d3["consts"],
                         n_floor_cells_N5=sum(1 for q in rows if q["N"] == 5 and q["floor"])))


# ------------------------------------------------------------------ birds (section 2)
def birds():
    d = load("survival_v2_seeds")
    ob = load("survival_v2_obsgate")
    out = dict(refs=d["_refs"], conds={}, obsgate={})
    for c in ("A2", "B2", "C2", "D2"):
        x = d[c]
        out["conds"][c] = dict(
            algo=x["algo"], alpha=x["alpha"], n_sustain=x["n_sustain"], n_collapse=x["n_collapse"],
            reward_seedlevel=[r(v) for v in x["reward_seedlevel"]],
            runs=[dict(seed=q["seed"], verdict=q["verdict"].replace("维持", "維持").replace("塌缩", "塌縮").replace("中间", "中間"),
                       reward=r(q["reward"]), ci95=r(q["ci95"]), last_decile_swap=r(q["last_decile_swap"], 4),
                       incoming_is_max=r(q["energy"]["incoming_is_max"], 3) if q["energy"]["incoming_is_max"] is not None else None)
                  for q in x["runs"]])
    out["conds"]["A2"]["minus_rot30"] = [r(v) for v in d["A2"]["minus_rot30_seedlevel"]]
    out["conds"]["A2"]["minus_gate30"] = [r(v) for v in d["A2"]["minus_gate30_seedlevel"]]
    out["conds"]["A2"]["incoming_max_seedlevel"] = [r(v) for v in d["A2"]["incoming_max_seedlevel"]]
    for k, v in ob["results"].items():
        out["obsgate"][k] = dict(reward=r(v["reward"]), ci95=r(v["reward_ci95_episode"]),
                                 minus_rot30=[r(v["minus_rot30_mean"]), r(v["minus_rot30_ci95_episode"])] if "minus_rot30_mean" in v else None,
                                 incoming_is_max=r(v.get("incoming_is_max"), 3) if v.get("incoming_is_max") is not None else None)
    bl = load("survival_v2_baselines")["results"]
    out["fixed_leader"] = r(bl["pathT2.0/fixed"]["reward"])
    dump("birds", out)


# ------------------------------------------------------------------ LLM commander (section 8)
def commander():
    base = os.path.join(ROOT, ".scratch/llm_commander")
    ins = json.load(open(os.path.join(base, "instructions.json")))
    out = dict(instructions=[], models={})
    for m in ("haiku", "sonnet"):
        res = {x["instruction_id"]: x for x in json.load(open(os.path.join(base, "results", m + ".json")))}
        raw = {}
        for line in open(os.path.join(base, "raw", m + ".txt"), encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            iid, js = line.split(" ", 1)
            raw[iid] = js                      # verbatim model output (JSON text incl. notes)
        out["models"][m] = {}
        for q in ins:
            x, j = res[q["id"]], json.loads(raw[q["id"]])
            out["models"][m][q["id"]] = dict(
                raw=raw[q["id"]], notes=j.get("notes"), valid=x["valid"], constraint_ok=x["constraint_ok"],
                matched=x["matched_readings"], range=r(x["team_range_ratio"]), oracle=r(x["oracle_ratio"]),
                regret_pct=r(100 * x["regret"], 2), conflict_note=x["notes_mentions_conflict"],
                nswap_max=x["nswap_max"])
    for q in ins:
        out["instructions"].append(dict(id=q["id"], category=q["category"], text=q["text"], scoring=q["scoring"],
                                        readings=[rd["name"] for rd in q["readings"]]))
    mx = {m: max(v["regret_pct"] for v in out["models"][m].values()) for m in out["models"]}
    out["summary"] = dict(n=len(ins), max_regret_pct=mx,
                          valid={m: sum(v["valid"] for v in out["models"][m].values()) for m in out["models"]},
                          ok={m: sum(v["constraint_ok"] for v in out["models"][m].values()) for m in out["models"]},
                          median_regret_pct={m: r(statistics.median(v["regret_pct"] for v in out["models"][m].values()), 2) for m in out["models"]})
    dump("commander", out)


# ------------------------------------------------------------------ timeline (section 7)
# (label, prediction commit, results commit). Hashes were read from `git log` on 2026-11-02;
# subjects/times are re-read from git below and the order is asserted.
PAIRS = [
    ("鳥類 v2（MARL 能量感知）", "2838e68", "98937d9", "survival v2 prereg"),
    ("輪換 L0（能量桶）", "a0619ab", "a7e76aa", "drone bucket prediction"),
    ("輪換 L1（位置差異）", "79b4dbc", "3c774e7", "drone L1 prediction"),
    ("輪換 L2／L3（擾動、局部資訊）", "e8d6059", "ae62fc8", "drone L2/L3 prediction"),
    ("貼尾流控制", "f49e6e8", "3437930", "wake headroom prediction"),
    ("找熱氣流 第一輪", "071574f", "a477e3b", "soaring headroom prediction"),
    ("找熱氣流 第二輪", "4eacb94", "3219473", "soaring headroom 2 prediction"),
    ("找熱氣流 第三輪", "02dc53f", "7c215c2", "soaring headroom 3 prediction"),
    ("LLM 指揮官", "5c53a88", "f1c439d", "llm commander prediction"),
]


def gitinfo(h):
    o = subprocess.check_output(["git", "-C", ROOT, "show", "-s", "--format=%H|%aI|%s", h], text=True).strip()
    full, iso, subj = o.split("|", 2)
    return dict(hash=full[:7], iso=iso, subject=subj)


def timeline():
    rows = []
    for label, p, q, _ in PAIRS:
        gp, gq = gitinfo(p), gitinfo(q)
        assert gp["iso"] < gq["iso"], (label, gp, gq)       # ISO with same +08:00 offset sorts lexicographically
        rows.append(dict(label=label, pred=gp, result=gq))
    dump("timeline", dict(rows=rows))


if __name__ == "__main__":
    horseshoe()
    bucket()
    wake()
    soaring()
    birds()
    commander()
    timeline()
