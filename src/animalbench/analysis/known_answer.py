"""Known-answer test (spec v0.3 #19): scripted agents run in the real environment -> metric pipeline -> classifier ->
confusion matrix.  Thresholds are fixed BEFORE looking at the scripted agents' labels and are derived from null models
only (99th percentile), using the calibration half of the null-agent seeds; the confusion matrix is reported on the
other half (and on the calibration half as an in-sample check).

Decision order (spec #19, written down before seeing results):
  1. rate >= 1/20 s and lead_support <= N/2                         -> treadmill
  2. no metric above its null 99th percentile                       -> none
  3. (PSI or rho_D above null) and RLS not above null               -> reciprocity
  4. RLS above null and PSI not above null                          -> state_rotation
  5. otherwise (RLS and PSI both above null)                        -> ambiguous
null 99th percentiles
  RLS : 99th pct over K N-rnd draws of the run's own leader series (same number of handovers)
  PSI : max( 99th pct over K N-perm draws of the run's own series, PSI of the run's N-cyc surrogate )
  rho_D: 99th pct of rho_D over the calibration half of the N-cyc and N-rnd null agents (needs physics, so it comes
         from the real null agents rather than an abstract surrogate)
"""
from __future__ import annotations

import json
import os

import numpy as np

from ..core import crosswind
from ..metrics import leadership as L
from ..metrics import nulls
from ..metrics.pipeline import compute_metrics
from ..scenarios.geese import SCRIPTED, GeeseConfig
from .batch import run_batch

# v0.4: N-cyc has ONE role: the formal-level "fair rotation" reference class (true label state_rotation).  The null classes
# for "none" are N-rnd and N-fix only.  N-cyc is not a "none" agent; that a clockwork fair rotation is not reciprocity is
# what the PSI / rho_D / Q thresholds (set above the N-cyc values) express.
TRUE_LABEL = {"S1_treadmill": "treadmill", "S2_state_rotation": "state_rotation",
              "S2f_state_rotation_fair": "state_rotation", "S3_reciprocity": "reciprocity",
              "N_cyc": "state_rotation", "N_rnd": "none", "N_fix": "none"}
PRED_LABELS = ["none", "treadmill", "state_rotation", "reciprocity", "ambiguous"]
CLASSES6 = ["S1_treadmill", "S2_state_rotation", "S3_reciprocity", "N_cyc", "N_rnd", "N_fix"]
CLASSES7 = CLASSES6[:2] + ["S2f_state_rotation_fair"] + CLASSES6[2:]
RATE_THR = 1.0 / 20.0
NULL_AGENTS = ("N_cyc", "N_rnd", "N_fix")
# every seed gets a different scattered start (leader identity differs) and OU turbulence
EVENTS = [crosswind(0.0, 1e4, speed=0.0, ramp=1.0, ou_sigma=0.5)]


def make_cfg(name, N=9, T=600.0, init_mode="scatter", **kw):
    mk, rule = SCRIPTED[name]
    cfg = GeeseConfig(n_max=N, T=T, swap_rule=rule, log_low=False, log_D=True, init_mode=init_mode, **kw)
    return cfg, mk(cfg)


def run_class(name, seeds, N=9, T=600.0, events=EVENTS, init_mode="scatter", chunk=25):
    cfg, pol = make_cfg(name, N, T, init_mode)
    return cfg, run_batch(cfg, pol, seeds, N, events=events, chunk=chunk)


def run_features(cfg, recs, N, K=100, seed0=0):
    """Per-seed metrics + own-null thresholds."""
    dh = cfg.delta_h
    rows = []
    for s in range(recs["leader"].shape[0]):
        lead = recs["leader"][s]
        m = compute_metrics(lead, dh, N, recs["D"][s])
        rng = np.random.default_rng(seed0 + s)
        w = m["window"]
        psi_p = [L.psi_sym(nulls.n_perm_ind(lead, N, rng), w)[0] for _ in range(K)]
        rnd = [nulls.n_rnd(lead, N, rng) for _ in range(K)]
        rls_r = [L.rls(x, N) for x in rnd]
        ret_r = [L.return_rate(x) for x in rnd]
        m["ret_thr"] = float(np.nanquantile(ret_r, 0.99)) if np.isfinite(ret_r).any() else 1.0
        cyc = nulls.n_cyc(lead, N)
        m["psi_thr"] = float(max(np.quantile(psi_p, 0.99), L.psi_sym(L.leader_to_ind(cyc, N), w)[0]))
        m["rls_thr"] = float(np.nanquantile(rls_r, 0.99)) if np.isfinite(rls_r).any() else 1.0
        ten = L.tenures(lead, dh)
        m["tenure_cv"] = float(ten.std() / ten.mean()) if len(ten) > 1 and ten.mean() > 0 else 0.0
        rows.append(m)
    return rows


def fit_rho_threshold(feats_calib):
    """99th pct of rho_D over the calibration half of the N-cyc and N-rnd null agents."""
    v = [f["rho_D"] for c in ("N_cyc", "N_rnd") for f in feats_calib[c] if np.isfinite(f["rho_D"])]
    return float(np.quantile(v, 0.99))


def classify(f, N, rho_thr, rate_thr=RATE_THR):
    if f["rate"] >= rate_thr and f["support"] <= N / 2:
        return "treadmill"
    rls_hi = bool(np.isfinite(f["rls"]) and f["rls"] > f["rls_thr"])
    psi_hi = bool(f["psi"] > f["psi_thr"])
    rho_hi = bool(np.isfinite(f["rho_D"]) and f["rho_D"] > rho_thr)
    if not (rls_hi or psi_hi or rho_hi):
        return "none"
    if (psi_hi or rho_hi) and not rls_hi:
        return "reciprocity"
    if rls_hi and not psi_hi:
        return "state_rotation"
    return "ambiguous"


def classify_v2(f, N, rho_thr, cv_thr=0.10, rate_thr=RATE_THR):
    """EXPLORATORY post-hoc variant (NOT the spec rule, written after seeing the spec-rule confusion matrix):
    (a) the immediate-return rate (A-B-A) above its N-rnd 99th percentile counts as an exchange signal next to PSI,
    rho_D; (b) a 'state_rotation' with a constant dwell (CV of tenure < cv_thr) is a clockwork cycle (N-cyc)
    and is relabelled none."""
    if f["rate"] >= rate_thr and f["support"] <= N / 2:
        return "treadmill"
    rls_hi = bool(np.isfinite(f["rls"]) and f["rls"] > f["rls_thr"])
    ex_hi = bool(f["psi"] > f["psi_thr"] or (np.isfinite(f["rho_D"]) and f["rho_D"] > rho_thr)
                 or (np.isfinite(f["return_rate"]) and f["return_rate"] > f["ret_thr"]))
    if not (rls_hi or ex_hi):
        return "none"
    if ex_hi and not rls_hi:
        return "reciprocity"
    if rls_hi and not ex_hi:
        return "none" if f["tenure_cv"] < cv_thr else "state_rotation"
    return "ambiguous"


def confusion(preds_by_class, classes):
    M = np.zeros((len(classes), len(PRED_LABELS)), int)
    for i, c in enumerate(classes):
        for p in preds_by_class[c]:
            M[i, PRED_LABELS.index(p)] += 1
    return M


def acceptance(M, classes):
    """Spec #19: diagonal >= 90 % per class (null classes: predicted 'none'), null agents judged non-none <= 10 %."""
    out = {}
    for i, c in enumerate(classes):
        n = M[i].sum()
        ok = M[i, PRED_LABELS.index(TRUE_LABEL[c])] / n
        out[c] = float(ok)
    return out


def evaluate(n_seeds=100, N=9, K=100, classes=CLASSES7, split=2, feats=None):
    """Full pipeline.  Calibration seeds = even indices, test seeds = odd indices (split=2).
    Returns res with res['spec'] (pre-registered rule) and res['v2'] (exploratory), each {test, calib: {M, diag}}."""
    if feats is None:
        feats = {}
        for c in classes:
            cfg, recs = run_class(c, range(n_seeds), N)
            feats[c] = run_features(cfg, recs, N, K)
    calib = {c: feats[c][0::split] for c in classes}
    test = {c: feats[c][1::split] for c in classes}
    rho_thr = fit_rho_threshold(calib)
    res = dict(rho_thr=rho_thr, N=N, n_seeds=n_seeds, K=K, feats=feats)
    for rule, fn in (("spec", classify), ("v2", classify_v2)):
        res[rule] = {}
        for tag, part in (("test", test), ("calib", calib)):
            preds = {c: [fn(f, N, rho_thr) for f in part[c]] for c in classes}
            M = confusion(preds, classes)
            res[rule][tag] = dict(M=M.tolist(), diag=acceptance(M, classes))
    return res


def write_report(res, classes, path_json, path_md):
    os.makedirs(os.path.dirname(path_json), exist_ok=True)
    json.dump({k: v for k, v in res.items() if k != "feats"}, open(path_json, "w"), indent=1)
    lines = [f"# Known-answer confusion matrices, N={res['N']}, seeds={res['n_seeds']} "
             f"(calibration = even seeds, test = odd seeds), rho_D threshold={res['rho_thr']:.3f}", ""]
    for rule, title in (("spec", "pre-registered spec rule (#19)"),
                        ("v2", "EXPLORATORY post-hoc variant v2 (return-rate signal + constant-dwell relabel)")):
        for tag in ("test", "calib"):
            M = np.array(res[rule][tag]["M"])
            lines += [f"## {title} -- {tag} half (rows = true class, columns = predicted)", "",
                      "| true \\ pred | " + " | ".join(PRED_LABELS) + " | diag |", "|---" * (len(PRED_LABELS) + 2) + "|"]
            for i, c in enumerate(classes):
                lines.append(f"| {c} | " + " | ".join(str(x) for x in M[i]) + f" | {res[rule][tag]['diag'][c]:.2f} |")
            lines.append("")
    open(path_md, "w").write("\n".join(lines))
