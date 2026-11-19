"""LLM commander evaluation harness (no LLM calls here).

A commander (Claude) turns a Chinese instruction into rule parameters (JSON); this module scores it.
Environment (fixed): drone_bucket L1, N=7, V formation, 'optimal' position setting, f_ind=0.4, c_swap=3,
individual spread +-15%, 64 fixed evaluation instances (seed 7007).

  evaluate(instruction_id, params) -> {valid, constraint_ok, team_range_ratio, oracle_ratio, regret, ...}
  python scripts/llm_commander.py build-oracle      # grid search, writes .scratch/llm_commander/oracle.json
  python scripts/llm_commander.py score <model>      # reads outputs/<model>/<id>.json, writes results/<model>.json
Run with: systemd-run --user --scope -q -p MemoryMax=12G -p MemorySwapMax=0 .venv/bin/python scripts/llm_commander.py ...
"""
import itertools, json, os, re, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import drone_bucket_l1 as L1
import drone_bucket as OLD

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
DIR = os.path.join(ROOT, ".scratch", "llm_commander")
N, FORM, SETTING, F_IND, C_SWAP, H, N_INST, SEED = 7, "V", "optimal", 0.4, 3, 0.15, 64, 7007
ROTATIONS = ["fixed", "time", "gate", "rot"]
KEYS = {"rotation", "tenure", "gate_margin", "no_lead", "protect", "max_swaps", "notes"}

# ---------------------------------------------------------------- environment
_ENV = {}


def env():
    if not _ENV:
        r2 = L1.two_plane_best()
        scale = 0.40 / (1 - r2)
        r, _, _ = L1.ratio_table(N, FORM, SETTING)
        m = L1.power_mult(r, F_IND, scale)
        s_ref = float(1 - m[1:].mean())
        k, B = OLD.sample(SEED, N_INST, N, H)
        _ENV.update(m=m, ce=np.full(N_INST, C_SWAP * s_ref), k=k, B=B)
    return _ENV


# ---------------------------------------------------------------- schema
def validate(p):
    """Return (ok, normalized params or error string). Strict: types, ranges, no unknown keys."""
    if not isinstance(p, dict):
        return False, "not an object"
    if set(p) - KEYS:
        return False, f"unknown keys {sorted(set(p) - KEYS)}"
    q = {}
    rot = p.get("rotation")
    if rot not in ROTATIONS:
        return False, "rotation must be one of fixed/time/gate/rot"
    q["rotation"] = rot
    isint = lambda v: isinstance(v, int) and not isinstance(v, bool)
    isnum = lambda v: isinstance(v, (int, float)) and not isinstance(v, bool) and np.isfinite(v)
    ten = p.get("tenure", 40)
    if not isint(ten) or not 1 <= ten <= 2000:
        return False, "tenure must be an integer 1..2000"
    q["tenure"] = ten
    gm = p.get("gate_margin", 10)
    if not isnum(gm) or not 0 <= gm <= 500:
        return False, "gate_margin must be a number 0..500"
    q["gate_margin"] = float(gm)
    for key in ("no_lead", "protect"):
        v = p.get(key, [])
        if (not isinstance(v, list) or not all(isint(x) and 1 <= x <= N for x in v) or len(set(v)) != len(v)):
            return False, f"{key} must be a list of distinct plane numbers 1..{N}"
        q[key] = sorted(v)
    ms = p.get("max_swaps")
    if ms is not None and (not isint(ms) or not 0 <= ms <= 10000):
        return False, "max_swaps must be null or an integer >= 0"
    q["max_swaps"] = ms
    if not isinstance(p.get("notes", ""), str):
        return False, "notes must be a string"
    # feasibility: protected planes take the deepest follower slots, at least one other plane must be able to lead
    forb = set(q["no_lead"]) | set(q["protect"])
    if len(q["protect"]) > N - 2:
        return False, "protect leaves fewer than 2 mobile planes"
    if len(forb) >= N:
        return False, "no plane left that may lead"
    return True, q


# ---------------------------------------------------------------- simulator (extends L1 with per-plane constraints)
def simulate(q, k=None, B=None):
    """Vectorised over instances. Returns dict of per-instance arrays:
    ratio (range/B0), nswap, lead (steps as leader per plane), steps, reserve (remaining fraction per plane at end)."""
    E = env()
    m = E["m"]
    k = E["k"] if k is None else k
    B = E["B"] if B is None else B
    ce = E["ce"]
    I = len(k)
    ar = np.arange(I)
    rule, tenure, margin = q["rotation"], q["tenure"], q["gate_margin"]
    forb = np.zeros(N, bool)
    forb[[x - 1 for x in q["no_lead"]]] = True
    prot = np.zeros(N, bool)
    prot[[x - 1 for x in q["protect"]]] = True
    forb |= prot
    cap = q["max_swaps"]
    # slots: protected planes lock the deepest (smallest m) positions, never position 0
    order_deep = [p for p in np.argsort(m, kind="stable") if p != 0]
    locked = set(order_deep[:int(prot.sum())])
    P = np.array([p for p in range(N) if p not in locked])     # mobile positions (sorted, includes 0)
    idxmap = np.full(N, -1); idxmap[P] = np.arange(len(P))
    Mn = len(P)
    # initial assignment
    rem = B.copy()
    rd = rem / k
    pos = np.zeros((I, N), int)
    mob = ~prot
    lead_score = np.where(mob & ~forb, rd, -np.inf)
    leader = lead_score.argmax(1)
    pos[ar, leader] = 0
    rest_pos = sorted(P[1:], key=lambda p: -m[p])               # least-saving mobile slots first
    rd2 = np.where(mob, rd, -np.inf)
    rd2[ar, leader] = -np.inf
    rank = np.argsort(-rd2, axis=1, kind="stable")              # mobile planes by rd desc (leader/protected last)
    for j, p in enumerate(rest_pos):
        pos[ar, rank[:, j]] = p
    pl = [p for p in range(N) if p in locked]
    prot_idx = np.where(prot)[0]
    for j, pi in enumerate(prot_idx):
        pos[:, pi] = pl[j] if j < len(pl) else 0
    dist = np.zeros(I)
    alive = np.ones(I, bool)
    nswap = np.zeros(I, int)
    lead = np.zeros((I, N))
    steps = np.zeros(I)
    t, tmax = 0, int(8 * L1.B0)
    while alive.any() and t < tmax:
        part = np.zeros((I, N), bool)
        can = alive & ((nswap < cap) if cap is not None else True)
        if rule != "fixed" and t > 0 and t % tenure == 0 and can.any() and Mn >= 2:
            ld = (pos == 0).argmax(1)
            if rule == "rot":
                idx = idxmap[pos]                      # index among mobile slots, -1 if locked
                plane_at = np.full((I, Mn), -1)
                for i_ in range(N):
                    mi = idx[:, i_] >= 0
                    plane_at[ar[mi], idx[mi, i_]] = i_
                s = np.zeros(I, int)
                for cand in range(Mn - 1, 0, -1):      # smallest shift whose new leader may lead
                    ok = ~forb[plane_at[ar, cand]]
                    s = np.where(ok, cand, s)
                go = can & (s > 0)
                newidx = np.where(idx >= 0, (idx - s[:, None]) % Mn, -1)
                newpos = np.where(newidx >= 0, P[np.clip(newidx, 0, Mn - 1)], pos)
                pos = np.where(go[:, None], newpos, pos)
                part = (idx >= 0) & go[:, None]
            else:
                rdn = rem / k
                elig = mob & ~forb
                elig = np.tile(elig, (I, 1)) & (np.arange(N)[None, :] != ld[:, None])
                if rule == "time":      # leader hands over to the allowed mobile plane at the deepest slot
                    key = np.where(elig, pos, -1)
                    partner = key.argmax(1)
                    go = can & (key[ar, partner] >= 0)
                else:                   # gate: richest allowed plane takes over if it beats the leader by > margin
                    key = np.where(elig, rdn, -np.inf)
                    partner = key.argmax(1)
                    go = can & ((rdn[ar, partner] - rdn[ar, ld]) > margin) & np.isfinite(key[ar, partner])
                if go.any():
                    ig = ar[go]; a = ld[go]; b = partner[go]
                    pa = pos[ig, a].copy(); pos[ig, a] = pos[ig, b]; pos[ig, b] = pa
                    part[ig, a] = True; part[ig, b] = True
        nswap += part.any(1)
        ldn = (pos == 0).argmax(1)
        lead[ar, ldn] += alive
        steps += alive
        f = np.where(part, 1.0, m[pos])
        cons = k * f + ce[:, None] * k * part
        new = rem - cons
        dead = alive & (new <= 0).any(axis=1)
        frac = np.where(cons > 0, rem / cons, np.inf).min(axis=1)
        dist += np.where(alive, np.where(dead, frac, 1.0), 0.0)
        rem = np.where(alive[:, None], new, rem)
        alive &= ~dead
        t += 1
    return dict(ratio=dist / L1.B0, nswap=nswap, lead=lead, steps=steps, reserve=rem / B)


def stats(sim):
    return dict(ratio=float(sim["ratio"].mean()), nswap_max=int(sim["nswap"].max()),
                lead_share_max=(sim["lead"] / sim["steps"][:, None]).max(0).tolist(),
                led_all=bool(((sim["lead"] > 0).all(1)).all()),
                reserve_mean=sim["reserve"].mean(0).tolist())


# ---------------------------------------------------------------- constraints
def check_reading(c, st, best_ratio=None):
    """Return dict constraint-name -> bool for one reading."""
    out = {}
    if "never_lead" in c:
        out["never_lead"] = all(st["lead_share_max"][i - 1] == 0 for i in c["never_lead"])
    if "max_swaps" in c:
        out["max_swaps"] = st["nswap_max"] <= c["max_swaps"]
    if c.get("all_lead"):
        out["all_lead"] = st["led_all"]
    if "lead_share_max" in c:
        pl, sh = c["lead_share_max"]["plane"], c["lead_share_max"]["share"]
        v = max(st["lead_share_max"]) if pl == "all" else st["lead_share_max"][pl - 1]
        out["lead_share_max"] = v <= sh + 1e-12
    if "reserve_above_mean" in c:
        rm = st["reserve_mean"]
        out["reserve_above_mean"] = rm[c["reserve_above_mean"]["plane"] - 1] >= np.mean(rm) + c["reserve_above_mean"]["margin"]
    if "range_ge_frac_of_best" in c:
        out["range_ge_frac_of_best"] = best_ratio is not None and st["ratio"] >= c["range_ge_frac_of_best"] * best_ratio
    return out


def instructions():
    return {x["id"]: x for x in json.load(open(os.path.join(DIR, "instructions.json")))}


def load_oracle():
    return json.load(open(os.path.join(DIR, "oracle.json")))


CONFLICT_WORDS = ["衝突", "矛盾", "無法同時", "不可能同時", "互相抵觸", "兩者", "取捨", "conflict", "contradict"]


def evaluate(instruction_id, params, oracle=None):
    ins = instructions()[instruction_id]
    oracle = oracle or load_oracle()
    ok, q = validate(params)
    if not ok:
        return dict(instruction_id=instruction_id, valid=False, error=q, constraint_ok=None,
                    team_range_ratio=None, oracle_ratio=None, regret=None)
    st = stats(simulate(q))
    best = oracle["g1"]["main"]["ratio"]
    readings = []
    for r in ins["readings"]:
        chk = check_reading(r["constraints"], st, best)
        readings.append(dict(name=r["name"], satisfied=all(chk.values()) if chk else True, checks=chk,
                             oracle_ratio=oracle[instruction_id][r["name"]]["ratio"]))
    sat = [r for r in readings if r["satisfied"]]
    pool = sat if sat else readings
    orr = max(r["oracle_ratio"] for r in pool) if not sat else min(r["oracle_ratio"] for r in sat)
    note = str(params.get("notes", ""))
    return dict(instruction_id=instruction_id, valid=True, constraint_ok=bool(sat),
                matched_readings=[r["name"] for r in sat], team_range_ratio=st["ratio"],
                oracle_ratio=orr, regret=(orr - st["ratio"]) / orr,
                regret_if_constraint_ok=((orr - st["ratio"]) / orr if sat else None),
                notes_mentions_conflict=any(w in note for w in CONFLICT_WORDS) if ins["needs_conflict_note"] else None,
                nswap_max=st["nswap_max"], lead_share_max=[round(x, 3) for x in st["lead_share_max"]],
                reserve_mean=[round(x, 3) for x in st["reserve_mean"]], readings=readings, params=q)


# ---------------------------------------------------------------- oracle (grid search)
TEN_TR = [10, 20, 40, 80, 160, 320, 640]
GATE_TEN = [1, 10, 40]
GATE_MAR = [1, 5, 10, 20, 40, 80]


def grid(no_lead_c, protect_c, caps):
    for nl, pr in itertools.product(no_lead_c, protect_c):
        yield dict(rotation="fixed", tenure=1, gate_margin=0, no_lead=nl, protect=pr, max_swaps=None)
        for cap in caps:
            for ten in TEN_TR:
                for rot in ("time", "rot"):
                    yield dict(rotation=rot, tenure=ten, gate_margin=0, no_lead=nl, protect=pr, max_swaps=cap)
            for ten, mar in itertools.product(GATE_TEN, GATE_MAR):
                yield dict(rotation="gate", tenure=ten, gate_margin=mar, no_lead=nl, protect=pr, max_swaps=cap)


def key_of(q):
    return json.dumps(q, sort_keys=True)


def build_oracle():
    ins = instructions()
    cache = {}
    out = {}

    def run(q):
        kq = key_of(q)
        if kq not in cache:
            cache[kq] = stats(simulate(q))
        return cache[kq]

    # g1 first (needed as 'best' for range_ge_frac_of_best)
    order = ["g1"] + [i for i in ins if i != "g1"]
    for iid in order:
        out[iid] = {}
        best_ratio = out["g1"]["main"]["ratio"] if iid != "g1" else None
        for r in ins[iid]["readings"]:
            c = r["constraints"]
            nl = [[]]; pr = [[]]
            if "never_lead" in c: nl.append(sorted(c["never_lead"]))
            if "lead_share_max" in c and c["lead_share_max"]["plane"] != "all": nl.append([c["lead_share_max"]["plane"]])
            if "reserve_above_mean" in c: pr.append([c["reserve_above_mean"]["plane"]])
            caps = [None]
            if "max_swaps" in c:
                L = c["max_swaps"]
                caps = sorted({L, max(L - 1, 0), L // 2}) if L > 0 else [0]
            elif c.get("all_lead"):
                caps = [None, 6, 8, 12]
            best = None
            for q in grid(nl, pr, caps):
                ok, qq = validate(q)
                if not ok:
                    continue
                st = run(qq)
                chk = check_reading(c, st, best_ratio)
                if all(chk.values()) and (best is None or st["ratio"] > best[0]):
                    best = (st["ratio"], qq)
            out[iid][r["name"]] = dict(ratio=best[0] if best else None, params=best[1] if best else None,
                                       feasible=best is not None)
            print(iid, r["name"], out[iid][r["name"]], flush=True)
    json.dump(out, open(os.path.join(DIR, "oracle.json"), "w"), ensure_ascii=False, indent=1)
    return out


# ---------------------------------------------------------------- scoring CLI
def extract_json(text):
    t = text.strip()
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", t, re.S)
    if m:
        t = m.group(1)
    else:
        a, b = t.find("{"), t.rfind("}")
        t = t[a:b + 1] if a >= 0 and b > a else t
    return json.loads(t)


def score(model):
    ins = instructions()
    orc = load_oracle()
    res = []
    for iid in ins:
        fp = os.path.join(DIR, "outputs", model, f"{iid}.json")
        if not os.path.exists(fp):
            continue
        try:
            params = extract_json(open(fp).read())
        except Exception as e:
            res.append(dict(instruction_id=iid, valid=False, error=f"unparseable JSON: {e}", constraint_ok=None,
                            team_range_ratio=None, oracle_ratio=None, regret=None))
            continue
        res.append(evaluate(iid, params, orc))
    os.makedirs(os.path.join(DIR, "results"), exist_ok=True)
    json.dump(res, open(os.path.join(DIR, "results", f"{model}.json"), "w"), ensure_ascii=False, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    for r in res:
        print(r["instruction_id"], "valid" if r["valid"] else "INVALID", "ok" if r.get("constraint_ok") else "VIOL",
              r.get("team_range_ratio") and round(r["team_range_ratio"], 4), r.get("oracle_ratio") and round(r["oracle_ratio"], 4),
              r.get("regret") is not None and round(r["regret"], 4), r.get("notes_mentions_conflict"))
    v = [r for r in res if r["valid"]]
    print(f"n={len(res)} valid={len(v)} constraint_ok={sum(1 for r in v if r['constraint_ok'])}")
    return res


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "build-oracle":
        build_oracle()
    elif cmd == "score":
        score(sys.argv[2])
    else:
        print(__doc__)
