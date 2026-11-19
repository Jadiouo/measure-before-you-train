"""M1a stage 2: #17 metric unit tests on artificial sequences, #18 (metric side) on hand-made D, #20 null models."""
import numpy as np
import pytest

from animalbench.metrics import (flows, free_rider_index, handover_rate, lead_share, lead_support, leader_to_ind,
                                 n_cyc, n_fix, n_perm_ind, n_rnd, psi_sym, r_lf, return_rate, rls, spells, tenures)
from animalbench.metrics.pipeline import compute_metrics, null_distribution

N, T, DH = 9, 300, 2.0


def seq_fixed():
    return np.zeros(T, int)


def seq_cycle(dwell=20):
    return (np.arange(T) // dwell) % N


def seq_random(seed, dwell=20, n=N):
    rng = np.random.default_rng(seed)
    out, cur = [], 0
    for _ in range(T // dwell + 1):
        cur = rng.choice([k for k in range(n) if k != cur])
        out += [cur] * dwell
    return np.array(out[:T])


def seq_pingpong(dwell=20):
    return (np.arange(T) // dwell) % 2


# ------------------------------------------------------------------------------------------------ #17
def test_17_fixed_leader():
    s = seq_fixed()
    assert handover_rate(s, DH) == 0.0
    assert lead_support(s, N) == 1
    assert free_rider_index(s, N) == pytest.approx(N)
    assert np.isnan(rls(s, N))
    assert psi_sym(leader_to_ind(s, N), 5)[0] == 0.0


def test_17_cycle_is_fair_rotation():
    s = seq_cycle()
    assert rls(s, N) == 1.0                                   # RLS(N-cyc) = 1
    assert handover_rate(s, DH) == pytest.approx((T // 20 - 1) / (T * DH))
    assert lead_support(s, N) == N
    assert tenures(s, DH).tolist() == [40.0] * (len(tenures(s, DH)))
    assert free_rider_index(s, N) == pytest.approx(1.0, abs=0.2)
    ps, pd = psi_sym(leader_to_ind(s, N), 10)
    assert pd > 0.7 and ps < 0.0                              # one-directional chain: strong PSI_ij but no mutual pair
    assert return_rate(s) == 0.0


def test_17_random_succession_rls_at_chance():
    # long random sequence: chance level is ~1/(N-1) once every bird has led (the 'never led' ties inflate RLS only
    # in the first handovers; that finite-sample inflation is why RLS is always compared with N-rnd of equal length)
    rng = np.random.default_rng(0)
    ids, cur = [], 0
    for _ in range(4000):
        cur = rng.choice([k for k in range(N) if k != cur]); ids.append(cur)
    assert rls(np.array(ids), N) == pytest.approx(1 / (N - 1), abs=0.03)
    short = [rls(seq_random(k), N) for k in range(200)]
    assert 0.3 < np.mean(short) < 0.6 and np.mean(short) < 1.0           # 14 handovers: inflated but clearly < N-cyc
    assert np.mean([return_rate(seq_random(k)) for k in range(200)]) == pytest.approx(1 / (N - 1), abs=0.04)


def test_17_pingpong():
    s = seq_pingpong()
    assert lead_support(s, N) == 2                            # spec: ping-pong lead_support = 2
    assert return_rate(s) == 1.0
    ps, _ = psi_sym(leader_to_ind(s, N), 10)
    assert ps > 0.2                                           # strong *mutual* exchange
    assert rls(s, N) < 0.5


def test_17_lead_support_threshold_and_spells():
    s = np.array([0] * 90 + [1] * 6 + [2] * 4)                  # bird 1: 6 %, bird 2: 4 % (< 5 %)
    assert lead_support(s, 3) == 2
    ids, st, ln = spells(s)
    assert ids.tolist() == [0, 1, 2] and ln.tolist() == [90, 6, 4]


def test_17_r_lf_and_diagnostics():
    # fixed leader: the leader spends no time in a wake, the followers do -> r_LF = -1 (PoC observation)
    lead_t = np.array([1.0] + [0.0] * 8)
    wake_t = np.array([0.0] + [1.0] * 8)
    assert r_lf(lead_t, wake_t) == pytest.approx(-1.0)
    assert np.isnan(r_lf(np.ones(5), np.arange(5)))           # undefined when one variable is constant


def test_17_var_B_rotation_lower_than_fixed():
    """D built from a toy wake rule: the leader (apex) gives 1 unit to every follower per decision."""
    def toy_D(leader):
        D = np.zeros((len(leader), N, N))
        for t, l in enumerate(leader):
            D[t, l, :] = 1.0
            D[t, l, l] = 0.0
        return D
    vf = flows.var_B_windows(toy_D(seq_fixed()), 180).mean()
    vc = flows.var_B_windows(toy_D(seq_cycle(20)), 180).mean()      # window = one full cycle -> perfectly balanced
    assert vf > 100 * vc and vc < 1e-9
    assert np.allclose(flows.net_balance(toy_D(seq_cycle())).sum(), 0.0, atol=1e-12)


# ------------------------------------------------------------------------------------------------ #18 (metrics)
def test_18_flow_metrics_hand_example():
    D = np.zeros((2, 3, 3))
    D[0, 0, 1] = 2.0          # 0 gives 2 to 1
    D[0, 0, 2] = 1.0
    D[1, 1, 2] = 0.5
    R = flows.received(D); Pv = flows.provided(D); B = flows.net_balance(D)
    assert R.tolist() == [0.0, 2.0, 1.5]
    assert Pv.tolist() == [3.0, 0.5, 0.0]
    assert B.tolist() == [-3.0, 1.5, 1.5] and abs(B.sum()) < 1e-12
    S = np.zeros((1, 3, 3)); S[0] = [[0, 1, 2], [1, 0, 3], [2, 3, 0]]            # symmetric
    assert flows.rho_D(S) == pytest.approx(1.0)
    A = np.zeros((1, 4, 4)); A[0, 0, 1:] = [1, 2, 3]; A[0, 1, 2:] = [1, 2]; A[0, 2, 3] = 1   # strictly one-directional
    assert flows.rho_D(A) < 0


# ------------------------------------------------------------------------------------------------ #20
@pytest.mark.parametrize("src", ["cycle", "random", "pingpong"])
def test_20_null_models(src):
    s = {"cycle": seq_cycle(), "random": seq_random(3, dwell=15), "pingpong": seq_pingpong(25)}[src]
    rng = np.random.default_rng(0)
    r0 = handover_rate(s, DH)
    # N-cyc: rate within 5 % of the tested run, RLS = 1 (for >= N spells)
    c = n_cyc(s, N)
    assert handover_rate(c, DH) == pytest.approx(r0, rel=0.05)
    if len(spells(c)[0]) >= N:
        assert rls(c, N) == 1.0
    # N-rnd: same number of handovers
    for _ in range(20):
        r = n_rnd(s, N, rng)
        assert handover_rate(r, DH) == pytest.approx(r0, rel=0.05)
    # N-fix: no handovers
    assert handover_rate(n_fix(s), DH) == 0.0
    # N-perm: marginal lead share exact, spell-length distribution KS < 0.05
    ind = leader_to_ind(s, N)
    p = n_perm_ind(s, N, rng)
    assert np.allclose(p.sum(0), ind.sum(0))
    def spell_lengths(M):                       # circular series: a run crossing the wrap point is one spell
        out = []
        for i in range(N):
            x = M[:, i].astype(int)
            if x.sum() == 0:
                continue
            if x.min() == 1:
                out.append(len(x)); continue
            x = np.roll(x, -int(np.flatnonzero(x == 0)[0]))
            d = np.diff(np.concatenate([[0], x, [0]]))
            out += (np.flatnonzero(d == -1) - np.flatnonzero(d == 1)).tolist()
        return np.sort(out)
    a, b = spell_lengths(ind), spell_lengths(p)
    grid = np.arange(0, T + 1)
    ks = np.abs(np.searchsorted(a, grid, side="right") / len(a) - np.searchsorted(b, grid, side="right") / len(b)).max()
    assert ks < 0.05


def test_20_null_distribution_report_has_all_nulls():
    s = seq_random(1)
    nd = null_distribution(s, DH, N, K=20)
    assert set(nd) == {"N_rnd", "N_perm", "N_cyc", "N_fix"}
    m = compute_metrics(s, DH, N)
    assert {"rate", "rls", "psi", "support", "tenure_med", "free_rider"} <= set(m)
    assert nd["N_cyc"]["rls"][0] == 1.0
    assert nd["N_rnd"]["rate"].std() < 1e-12                                                # rate identical to tested
