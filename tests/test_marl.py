"""M1.5 IPPO unit tests (small, seconds each; no long training)."""
import json
import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import animalbench  # noqa: F401  (x64 on: the learner must stay float32 anyway)
from animalbench.marl import MarlConfig, gae, init_train, load_ckpt, make_rollout, ppo_update, save_ckpt, train
from animalbench.marl.ippo import group, init_params, net_seq, policy_probs, seq_group, ungroup


@pytest.mark.parametrize("memory", ["mlp", "gru"])
def test_ppo_learns_trivial_contextual_bandit(memory):
    """3 contexts (one-hot obs), reward 1 iff action == context.  PPO update alone must find the mapping."""
    m = MarlConfig(memory=memory, hidden=32, n_mb=2, epochs=4, lr=3e-3, ent_coef=0.0, shared=True)
    D, S, T = 4, 64, 6
    params = init_params(jax.random.PRNGKey(0), D, m)
    from animalbench.marl.ippo import adam_init
    opt = adam_init(params)
    rng = np.random.default_rng(0)
    seq = jax.vmap(jax.vmap(lambda p, o, r: net_seq(p, o, r, memory, m.hidden), in_axes=(None, 0, 0)), in_axes=(0, 0, 0))
    mask = jnp.ones((1, S, T, 3), bool)
    rflag = jnp.zeros((1, S, T), bool)
    p_correct = []
    for it in range(40):
        ctx = rng.integers(0, 3, (1, S, T))
        obs = jnp.asarray(np.eye(4, dtype=np.float32)[ctx])
        lg, v = seq(params, obs, rflag)
        a = jax.random.categorical(jax.random.PRNGKey(it), lg, axis=-1)
        logp = jnp.take_along_axis(jax.nn.log_softmax(lg, -1), a[..., None], -1)[..., 0]
        rew = (a == jnp.asarray(ctx)).astype(jnp.float32)
        p_correct.append(float(rew.mean()))
        adv = rew - rew.mean()
        sdata = dict(obs=obs, mask=mask, act=a, logp=logp, adv=adv, ret_n=rew, reset=rflag)
        params, opt, _ = ppo_update(params, opt, sdata, m.lr, jax.random.PRNGKey(1000 + it), m)
    assert np.mean(p_correct[:3]) < 0.5
    assert np.mean(p_correct[-3:]) > 0.9


def test_gae_matches_python_reference():
    rng = np.random.default_rng(1)
    T = 12
    r, v = rng.normal(size=(T, 2)).astype(np.float32), rng.normal(size=(T, 2)).astype(np.float32)
    lv = rng.normal(size=(2,)).astype(np.float32)
    g, l = 0.97, 0.9
    adv_ref = np.zeros((T, 2))
    last = np.zeros(2)
    for t in reversed(range(T)):
        nv = lv if t == T - 1 else v[t + 1]
        delta = r[t] + g * nv - v[t]
        last = delta + g * l * last
        adv_ref[t] = last
    adv, ret = gae(jnp.asarray(r), jnp.asarray(v), jnp.asarray(lv), g, l)
    np.testing.assert_allclose(np.asarray(adv), adv_ref, atol=1e-5)
    np.testing.assert_allclose(np.asarray(ret), adv_ref + v, atol=1e-5)


@pytest.mark.parametrize("shared", [True, False])
def test_group_roundtrip(shared):
    x = jnp.arange(4 * 3 * 5).reshape(4, 3, 5)
    np.testing.assert_array_equal(np.asarray(ungroup(group(x, shared), 4, 3, shared)), np.asarray(x))
    xs = jnp.arange(7 * 4 * 3).reshape(7, 4, 3)
    g = seq_group(xs, shared)
    assert g.shape == ((1, 12, 7) if shared else (3, 4, 7))


def test_checkpoint_roundtrip_same_policy(tmp_path):
    m = MarlConfig(n_envs=4, shared=False, memory="gru")
    ecfg = m.env_cfg()
    ts = init_train(jax.random.PRNGKey(3), ecfg, m)
    p = str(tmp_path / "c.ckpt")
    save_ckpt(p, ts, ecfg, m, dict(it=7))
    ts2, m2, extra = load_ckpt(p)
    assert m2 == m and extra["it"] == 7
    for a, b in zip(jax.tree_util.tree_leaves(ts), jax.tree_util.tree_leaves(ts2)):
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b))
    f = jax.jit(make_rollout(ecfg, m, n_dec=6, greedy=True), static_argnums=(2, 3))
    d1, _, _ = f(ts.params, jax.random.PRNGKey(0), -1, 0)
    d2, _, _ = f(ts2.params, jax.random.PRNGKey(0), -1, 0)
    np.testing.assert_array_equal(np.asarray(d1["logits"]), np.asarray(d2["logits"]))
    np.testing.assert_array_equal(np.asarray(d1["act"]), np.asarray(d2["act"]))


def test_rollout_shapes_finite_and_mask_respected():
    m = MarlConfig(n_envs=4, shared=True)
    ecfg = m.env_cfg()
    ts = init_train(jax.random.PRNGKey(0), ecfg, m)
    f = jax.jit(make_rollout(ecfg, m, n_dec=10), static_argnums=(2, 3))
    d, last_v, _ = f(ts.params, jax.random.PRNGKey(1), -1, 0)
    assert d["obs"].shape == (10, 4, 3, 21) and d["act"].shape == (10, 4, 3) and last_v.shape == (4, 3)
    assert all(bool(jnp.all(jnp.isfinite(v.astype(jnp.float32)))) for v in d.values())
    chosen_ok = jnp.take_along_axis(d["mask"], d["act"][..., None], -1)[..., 0]
    assert bool(chosen_ok.all())
    # N=3: the slot-1/2 birds have no slot behind them, so BACK is always masked for them
    assert not bool(d["mask"][..., 1][d["slot"] >= 1].any())
    pr = policy_probs(d["logits"], d["mask"])
    assert float(jnp.abs(pr.sum(-1) - 1).max()) < 1e-5


def test_dev_mode_forces_first_action():
    m = MarlConfig(n_envs=4)
    ecfg = m.env_cfg()
    ts = init_train(jax.random.PRNGKey(0), ecfg, m)
    f = jax.jit(make_rollout(ecfg, m, n_dec=6), static_argnums=(2, 3))
    d, _, _ = f(ts.params, jax.random.PRNGKey(1), 1, 2)
    assert bool((d["act"][0, :, 0] == 1).all()) and bool((d["act"][1:, :, 0] == 0).all())


def test_train_smoke_two_iterations(tmp_path):
    m = MarlConfig(n_envs=8, shared=True, memory="gru", total_agent_steps=2 * 8 * 3 * 300)
    ts, _ = train(m, 0, str(tmp_path), ckpt_every=0, verbose=False)
    rows = [json.loads(l) for l in open(tmp_path / "log.jsonl")]
    assert len(rows) == 2 and all(r["finite"] == 1.0 for r in rows)
    assert os.path.exists(tmp_path / "final.ckpt")
    assert -700 < rows[0]["ret"] < -450      # per-bird return of a 600 s episode in P_solo*s is ~ -0.9 * 600


def _params_hash(m, tmp_path):
    import hashlib
    ts, _ = train(m, 0, str(tmp_path), ckpt_every=0, verbose=False)
    h = hashlib.sha256()
    for x in jax.tree_util.tree_leaves(ts.params):
        h.update(np.asarray(x).tobytes())
    return h.hexdigest(), ts


def test_central_off_is_bitwise_identical_to_legacy(tmp_path):
    """central=False (default) must reproduce the pre-MAPPO params exactly (hash recorded from the old code)."""
    m = MarlConfig(n_envs=8, shared=True, memory="gru", total_agent_steps=2 * 8 * 3 * 300)
    hx, ts = _params_hash(m, tmp_path)
    assert "c" not in ts.params
    assert hx == "c7cdb43c57620f4924c03209f86a114b13420651b539f55c89ffe164695183fb"


def test_central_critic_short_training_runs(tmp_path):
    m = MarlConfig(n_envs=8, shared=True, memory="gru", central=True, total_agent_steps=2 * 8 * 3 * 300)
    _, ts = _params_hash(m, tmp_path)
    rows = [json.loads(l) for l in open(tmp_path / "log.jsonl")]
    assert len(rows) == 2 and all(r["finite"] == 1.0 for r in rows)
    assert ts.params["c"]["w1"].shape[-1] == m.hidden and ts.params["c"]["w1"].shape[-2] == 3 * ts.params["w1"].shape[-2]


def test_separate_critic_off_is_bitwise_identical_to_legacy(tmp_path):
    m = MarlConfig(n_envs=8, shared=True, memory="gru", separate_critic=False, total_agent_steps=2 * 8 * 3 * 300)
    hx, ts = _params_hash(m, tmp_path)
    assert "c" not in ts.params
    assert hx == "c7cdb43c57620f4924c03209f86a114b13420651b539f55c89ffe164695183fb"


def test_separate_critic_short_training_runs(tmp_path):
    m = MarlConfig(n_envs=8, shared=True, memory="gru", separate_critic=True, total_agent_steps=2 * 8 * 3 * 300)
    _, ts = _params_hash(m, tmp_path)
    rows = [json.loads(l) for l in open(tmp_path / "log.jsonl")]
    assert len(rows) == 2 and all(r["finite"] == 1.0 for r in rows)
    assert ts.params["c"]["w1"].shape[-2] == 3 * ts.params["w1"].shape[-2]
    # same critic parameter count as MAPPO
    mc = MarlConfig(n_envs=8, shared=True, memory="gru", central=True, total_agent_steps=2 * 8 * 3 * 300)
    from animalbench.marl.ippo import init_params
    import jax
    D = ts.params["w1"].shape[-2]
    n = lambda p: sum(x.size for x in jax.tree_util.tree_leaves(p["c"]))
    assert n(init_params(jax.random.PRNGKey(0), D, m)) == n(init_params(jax.random.PRNGKey(0), D, mc))
