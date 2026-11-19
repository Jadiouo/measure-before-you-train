"""M3 basin tooling tests: classification thresholds, episode metrics, value-only warm-up freeze, tiny DAgger BC."""
import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
import jax
import numpy as np

import animalbench  # noqa: F401
from animalbench.marl import MarlConfig
from animalbench.marl.basin import classify, episode_metrics


def _slots(T, B, lead_fn):
    s = np.ones((T, B, 3), int)
    for t in range(T):
        for b in range(B):
            s[t, b, lead_fn(t, b)] = 0
    return s


def test_classify_fixed_rotation_other():
    T = 300
    fixed = episode_metrics(_slots(T, 4, lambda t, b: b % 3), np.zeros((T, 4)), -np.ones((T, 4, 3)))
    assert classify(fixed) == "fixed" and fixed["max_lead_share"] == 1.0
    sw = np.zeros((T, 4)); sw[::25] = 1                     # 12 swaps / 600 s = 0.02 /s
    rot = episode_metrics(_slots(T, 4, lambda t, b: (t // 25) % 3), sw, -np.ones((T, 4, 3)))
    assert classify(rot) == "rotation" and rot["min_lead_share"] >= 0.3
    sw2 = np.zeros((T, 4)); sw2[100] = 1                    # one swap: neither fixed-rate nor rotation
    oth = episode_metrics(_slots(T, 4, lambda t, b: 0 if t < 100 else 1), sw2, -np.ones((T, 4, 3)))
    assert classify(oth) == "other"


def test_value_warmup_freezes_policy_and_trunk():
    from animalbench.marl.ippo import init_train, make_iteration
    m = MarlConfig(n_envs=2, total_agent_steps=2 * 3 * 300 * 2, vf_warmup_iters=1, random_slots=True, n_mb=2)
    ecfg = m.env_cfg()
    assert ecfg.random_slots
    it = jax.jit(make_iteration(ecfg, m))
    ts = init_train(jax.random.PRNGKey(0), ecfg, m)
    ts1, _ = it(ts, jax.random.PRNGKey(1))                  # warm-up iteration: only wv/bv may move
    for k in ts.params:
        moved = float(np.abs(np.asarray(ts1.params[k]) - np.asarray(ts.params[k])).max()) > 0
        assert moved == (k in ("wv", "bv")), k
    ts2, _ = it(ts1, jax.random.PRNGKey(2))                 # afterwards the policy trains
    assert float(np.abs(np.asarray(ts2.params["wpi"]) - np.asarray(ts1.params["wpi"])).max()) > 0


def test_dagger_bc_runs_and_labels_agree():
    from animalbench.marl.warmstart import dagger_bc
    from animalbench.scenarios.geese import scripted as S
    m = MarlConfig(swap_rule="mutual", random_slots=True)
    ecfg = m.env_cfg()
    params, hist = dagger_bc(ecfg, m, S.state_rotation(ecfg, 0.94, 0.90), iters=3, n_envs=4, beta0_iters=2,
                             log=lambda *_: None)
    assert np.isfinite(hist[-1][1]) and 0.0 <= hist[-1][2] <= 1.0
    assert params["wpi"].shape[0] == 1
