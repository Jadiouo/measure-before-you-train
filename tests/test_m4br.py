import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")
import numpy as np
import jax
import animalbench  # noqa
from animalbench.marl import MarlConfig
from animalbench.marl.basin import scripted_data, episode_metrics, classify
from animalbench.marl.br import train_br
from animalbench.scenarios.geese import GeeseConfig
from animalbench.scenarios.geese.scripted_ledger import ledger_reciprocator

CFG = GeeseConfig(n_max=3, dtype="float32", log_low=False, log_D=False, use_events=False, swap_rule="mutual",
                  kappa=1.0, random_slots=True)


def _ret(pol, n=32):
    d = scripted_data(CFG, pol, n_envs=n, seed=3)
    return d, episode_metrics(d["slot"], d["n_swaps"], d["rew"], CFG.delta_h, CFG.T)


def test_reciprocators_rotate_fairly():
    for ver, lam in (("self", 0), ("identity", 4)):
        _, m = _ret(ledger_reciprocator(CFG, ver, lam))
        assert classify(m) == "rotation" and m["min_lead_share"] > 0.3 and m["lead_entropy"] > 0.99


def test_free_rider_gains_and_apex_grabber_loses():
    d0, _ = _ret(ledger_reciprocator(CFG, "identity", 4))
    base = d0["rew"].sum(0)[:, 0].mean() * CFG.delta_h
    d1, _ = _ret(ledger_reciprocator(CFG, "identity", 4, dev_id=0, dev_kind=1))
    d2, _ = _ret(ledger_reciprocator(CFG, "identity", 4, dev_id=0, dev_kind=2))
    assert d1["rew"].sum(0)[:, 0].mean() * CFG.delta_h > base + 10
    assert d2["rew"].sum(0)[:, 0].mean() * CFG.delta_h < base - 10


def test_marl_defaults_unchanged_and_br_smoke(tmp_path):
    m = MarlConfig()
    assert m.reward_mode == "individual" and m.env_cfg().reward_mode == "individual"
    m1 = MarlConfig(n_agents=1, n_envs=8, swap_rule="mutual", kappa=1.0, random_slots=True, total_agent_steps=8 * 300 * 2)
    pol = ledger_reciprocator(m1.env_cfg().replace(n_max=3), "identity", 4, dev_id=0, dev_kind=-1)
    ts, ecfg, it = train_br(m1, pol, 0, str(tmp_path), verbose=False)
    leaves = jax.tree_util.tree_leaves(ts.params)
    assert all(bool(np.isfinite(np.asarray(x)).all()) for x in leaves)
