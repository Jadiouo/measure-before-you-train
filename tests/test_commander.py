"""M2-pre (A): commander_view, Command, AnnounceExecutor, scripted_commander (scenarios/geese/commander.py)."""
import json

import jax
import numpy as np
import pytest

from animalbench.core import compile_schedule
from animalbench.core.perturbation import leave
from animalbench.scenarios.geese import GeeseConfig, stay_policy
from animalbench.scenarios.geese.commander import (Command, ViewTracker, announce_stats, commander_view, run_announce_episode,
                                                   scripted_commander, view_to_text)
from animalbench.scenarios.geese.env import reset


def mk(N, T=300.0):
    return GeeseConfig(n_max=N, swap_rule="mutual", T=T, log_low=False, log_D=False)


def test_view_deterministic_and_capped():
    cfg = mk(9)
    st = reset(cfg, jax.random.PRNGKey(0), 9, compile_schedule([], cfg.e_max))
    tr = ViewTracker(cfg)
    tr.observe(st)
    v1, v2 = commander_view(cfg, st, tr), commander_view(cfg, st, tr)
    s1, s2 = view_to_text(v1), view_to_text(v2)
    assert s1 == s2
    d = json.loads(s1)
    assert d["n_alive"] == 9 and d["swap_rule"] == "mutual" and len(d["birds"]) == 9
    assert set(d["birds"][0]) >= {"id", "slot", "lead_share", "E_rel", "lock", "immune"}
    assert "id_code" not in s1 and "logit" not in s1             # no private information
    assert len(s1) <= 1800
    short = view_to_text(v1, max_chars=400)                       # deterministic truncation
    assert len(short) <= 400 and short == view_to_text(v2, max_chars=400)


def test_command_json_roundtrip():
    c = Command.from_json('{"next_leader": 2, "order_hint": [2, 0, 1], "announce": true, "period": 24, "extra": 1}')
    assert c == Command(2, (2, 0, 1), True, 24.0)


def test_announce_vs_none_n3():
    cfg = mk(3, 300.0)
    _, tr0, _ = run_announce_episode(cfg, stay_policy(cfg), None, seed=0, N=3)
    _, tr1, info = run_announce_episode(cfg, stay_policy(cfg), scripted_commander(30.0), seed=0, N=3)
    s0, s1 = announce_stats(tr0, cfg, 3), announce_stats(tr1, cfg, 3)
    assert s0["handovers"] == 0 and s0["lead_share"] == [1.0, 0.0, 0.0]          # mutual without announcement: fixed leader
    assert s1["handover_rate_per_s"] > 0.02                                       # ~1/30 s
    assert s1["max_share_dev"] < 0.1                                              # lead share ~ 1/N
    assert len(info["calls"]) == 1                                                # one commander call at t=0
    assert s1["cost_mean"] > s0["cost_mean"] * 0.99                               # rotation is not free (kappa=0)


def test_text_interface_equals_dict_interface_and_event_replan():
    cfg = mk(5, 240.0)
    ev = [leave(100.0, "random")]
    _, tr_a, ia = run_announce_episode(cfg, stay_policy(cfg), scripted_commander(30.0), seed=1, N=5, events=ev)
    _, tr_b, ib = run_announce_episode(cfg, stay_policy(cfg), scripted_commander(30.0), seed=1, N=5, events=ev,
                                       text_interface=True)
    assert len(ia["calls"]) == 2 and len(ia["calls"]) <= 3                       # start + leave event
    assert [c["order"] for c in ia["calls"]] == [c["order"] for c in ib["calls"]]
    assert np.array_equal(tr_a.global_fields["leader"], tr_b.global_fields["leader"])
