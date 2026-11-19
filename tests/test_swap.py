"""M1a #7: swap semantics from hand-made request tables."""
import jax
import jax.numpy as jnp
import numpy as np

from animalbench.core import compile_schedule
from animalbench.scenarios.geese import GeeseConfig, action_mask, resolve_swaps
from animalbench.scenarios.geese.env import reset
from animalbench.scenarios.geese.state import slot_offset

BACK, FWD = 1, 2


def mk(rule="unilateral", N=9, n_max=9):
    cfg = GeeseConfig(n_max=n_max, swap_rule=rule)
    st = reset(cfg, jax.random.PRNGKey(0), N, compile_schedule([], cfg.e_max))
    return cfg, st


def acts(cfg, **kw):
    a = np.zeros(cfg.n_max, np.int32)
    for k, v in kw.items():
        a[int(k[1:])] = v
    return jnp.asarray(a)


def slots(st):
    return np.asarray(st.slot).tolist()


def test_a_targets_and_masks():
    cfg, st = mk()
    # retreat: slot 3 <-> 5 ; advance: slot 4 -> 2 ; slot 1/2 forward -> apex
    for kw, pair in [(dict(b3=BACK), (3, 5)), (dict(b5=FWD), (3, 5)), (dict(b4=FWD), (2, 4)), (dict(b1=FWD), (0, 1)),
                     (dict(b2=FWD), (0, 2)), (dict(b6=BACK), (6, 8))]:
        s2, info = resolve_swaps(cfg, st, acts(cfg, **kw))
        exp = list(range(9)); exp[pair[0]], exp[pair[1]] = exp[pair[1]], exp[pair[0]]
        assert slots(s2) == exp, (kw, slots(s2))
        assert int(info.n_swaps) == 1
        assert int(s2.lock[pair[0]]) == cfg.lock_dec and int(s2.lock[pair[1]]) == cfg.lock_dec
    # apex rotation alternates between slot 1 and slot 2
    s = st
    seen = []
    for _ in range(3):
        s, info = resolve_swaps(cfg, s, acts(cfg, **{f"b{int(np.asarray(s.slot).tolist().index(0))}": BACK}))
        occ0 = np.asarray(s.slot).tolist().index(0)
        seen.append(occ0)
        s = s._replace(lock=jnp.zeros_like(s.lock), dec=s.dec + 1)
    # first retreat -> bird1 becomes apex (slot1), then bird0(slot1) ... second retreat goes to slot 2 occupant
    assert seen[0] == 1 and seen[1] == 2
    # masks: no target -> BACK masked (deep slot, apex alone), FWD masked at apex
    m = np.asarray(action_mask(cfg, st))
    assert not m[7, BACK] and not m[8, BACK] and not m[0, FWD]
    assert m[0, BACK] and m[3, BACK] and m[5, FWD] and m[:, 0].all()
    s2, info = resolve_swaps(cfg, st, acts(cfg, b7=BACK, b0=FWD))
    assert slots(s2) == list(range(9)) and not np.asarray(info.blocked).any()      # masked = stay, not blocked
    # N=2: only slot 1 exists, apex target falls back to the occupied arm
    cfg2, st2 = mk(N=2)
    s3, _ = resolve_swaps(cfg2, st2._replace(apex_ctr=jnp.int32(1)), acts(cfg2, b0=BACK))
    assert slots(s3)[:2] == [1, 0]


def test_b_double_proposal_swaps_once():
    cfg, st = mk()
    s2, info = resolve_swaps(cfg, st, acts(cfg, b3=BACK, b5=FWD))
    exp = list(range(9)); exp[3], exp[5] = 5, 3
    assert slots(s2) == exp and int(info.n_swaps) == 1
    assert not np.asarray(info.blocked).any()


def test_c_front_priority_and_blocked_flag():
    cfg, st = mk()
    s2, info = resolve_swaps(cfg, st, acts(cfg, b1=BACK, b3=BACK))   # edges (1,3) and (3,5): front end first
    exp = list(range(9)); exp[1], exp[3] = 3, 1
    assert slots(s2) == exp
    bl = np.asarray(info.blocked)
    assert bl[3] == False and bl[1] == False                         # bird 3 was moved by edge (1,3): request "satisfied"
    # request that is really cancelled: bird 5 BACK conflicts with edge (3,5) proposed by bird 3
    s2, info = resolve_swaps(cfg, st, acts(cfg, b3=BACK, b5=BACK))
    exp = list(range(9)); exp[3], exp[5] = 5, 3
    assert slots(s2) == exp
    assert np.asarray(info.blocked)[5] == False                       # 5 was swapped (by 3's edge)
    s2, info = resolve_swaps(cfg, st, acts(cfg, b1=BACK, b3=FWD, b5=BACK, b4=FWD))
    # (1,3) accepted, (3,5) cancelled -> bird 5 requested BACK to slot 7: edge (5,7) free -> accepted; bird 4 FWD (2,4) ok
    assert slots(s2)[1] == 3 and slots(s2)[3] == 1 and slots(s2)[4] == 2 and slots(s2)[2] == 4
    cfg_, st_ = mk()
    # genuine blocked: bird 3 BACK (edge 3-5) loses to bird 1 BACK? no (1,3) takes slot 3; bird 3 already swapped.
    # Use: bird 5 BACK (5,7) and bird 3 BACK (3,5): (3,5) first -> slot 5 used, (5,7) cancelled; 5 was swapped by (3,5) anyway.
    # Cancelled-and-unswapped case: bird 7 FWD (5,7) vs bird 3 BACK (3,5): (3,5) accepted, (5,7) cancelled, bird 7 blocked.
    s2, info = resolve_swaps(cfg, st, acts(cfg, b3=BACK, b7=FWD))
    exp = list(range(9)); exp[3], exp[5] = 5, 3
    assert slots(s2) == exp
    bl = np.asarray(info.blocked)
    assert bl[7] and bl.sum() == 1


def test_d_lock_forces_stay():
    cfg, st = mk()
    lock = jnp.zeros(9, jnp.int32).at[3].set(2)
    stl = st._replace(lock=lock)
    s2, info = resolve_swaps(cfg, stl, acts(cfg, b1=BACK))          # partner locked
    assert slots(s2) == list(range(9)) and np.asarray(info.blocked)[1]
    s2, info = resolve_swaps(cfg, stl, acts(cfg, b3=BACK))          # own request while locked
    assert slots(s2) == list(range(9)) and np.asarray(info.blocked)[3]
    assert not np.asarray(action_mask(cfg, stl))[3, 1:].any()
    # lock expires: decision_step decrements; exact semantics checked in test_timing


def test_e_slot0_lottery_fair():
    cfg, st = mk()
    a = acts(cfg, b1=FWD, b2=FWD)
    f = jax.jit(jax.vmap(lambda d: resolve_swaps(cfg, st._replace(dec=d), a)[1].edge_acc))
    acc = np.asarray(f(jnp.arange(10000, dtype=jnp.int32)))
    assert (acc[:, 1] ^ acc[:, 2]).all()                               # exactly one of the two edges
    assert abs(acc[:, 1].mean() - 0.5) <= 0.02


def test_f_mutual_needs_handshake():
    cfg, st = mk("mutual")
    s2, _ = resolve_swaps(cfg, st, acts(cfg, b3=BACK))
    assert slots(s2) == list(range(9))
    s2, info = resolve_swaps(cfg, st, acts(cfg, b5=FWD))
    assert slots(s2) == list(range(9)) and np.asarray(info.blocked)[5]  # unmatched request is reported as not executed
    s2, _ = resolve_swaps(cfg, st, acts(cfg, b3=BACK, b5=FWD))
    exp = list(range(9)); exp[3], exp[5] = 5, 3
    assert slots(s2) == exp
    s2, _ = resolve_swaps(cfg, st, acts(cfg, b0=BACK, b1=FWD))
    assert slots(s2)[0] == 1 and slots(s2)[1] == 0
    s2, _ = resolve_swaps(cfg, st, acts(cfg, b0=BACK, b2=FWD))       # apex targets slot 1 first -> 2 does not match
    assert slots(s2) == list(range(9))


def test_g_immune_never_swapped():
    cfg, st = mk()
    sti = st._replace(immune=jnp.zeros(9, bool).at[5].set(True))
    s2, _ = resolve_swaps(cfg, sti, acts(cfg, b3=BACK))
    assert slots(s2) == list(range(9))
    s2, _ = resolve_swaps(cfg, sti, acts(cfg, b5=FWD))
    assert slots(s2) == list(range(9))
    s2, _ = resolve_swaps(cfg, sti, acts(cfg, b7=FWD, b4=BACK))        # others still swap (7<->5 blocked; 4<->6 ok)
    assert slots(s2)[5] == 5 and slots(s2)[4] == 6 and slots(s2)[6] == 4


# ---- physics-fix regressions (notes/impl/physics_fix_decisions.md)
def _crosswind_state(cfg, st, wy=3.0):
    """State in a steady crosswind: formation frame rotated by psi_f = atan2(-wy, v0) ~ -12 deg (tolerates f64 only)."""
    return st._replace(wind_f=jnp.asarray([0.0, wy], cfg.jdtype))


def test_swap_path_starts_at_pre_swap_position_under_crosswind():
    """path_from is stored in the formation frame: right after a swap (s=0) the commanded position of every swapped
    bird equals its commanded position just before the swap.  Tolerance 1e-9 m (f64).  The pre-fix code (stored the
    already-rotated offset, rotated again) was off by ~|off|*2*sin(psi_f/2) ~ 1-3 m here."""
    from animalbench.scenarios.geese.dynamics import formation_heading, target_offset
    cfg, st = mk()
    st = _crosswind_state(cfg, st)
    assert abs(float(formation_heading(cfg, st)[0])) > 0.1                      # test is non-vacuous (psi_f ~ -0.21)
    st = st._replace(t=jnp.asarray(5.0, cfg.jdtype))
    before, _ = target_offset(cfg, st)
    s2, info = resolve_swaps(cfg, st, acts(cfg, b3=BACK))
    assert int(info.n_swaps) == 1 and np.asarray(info.swapped).sum() == 2
    after, _ = target_offset(cfg, s2)
    err = np.abs(np.asarray(after) - np.asarray(before)).max()
    assert err < 1e-9, err
    # mid-flight: a second swap involving a bird still on its first transition path (smoothstep h in (0,1))
    s3 = s2._replace(t=s2.t + 1.2, lock=jnp.zeros_like(s2.lock))
    before3, _ = target_offset(cfg, s3)
    occ = np.asarray(s3.slot).tolist()
    b_back = occ.index(5)                                   # bird now in slot 5 (retreated earlier) goes FWD again
    s4, info4 = resolve_swaps(cfg, s3, acts(cfg, **{f"b{b_back}": FWD}))
    assert int(info4.n_swaps) == 1
    after4, _ = target_offset(cfg, s4)
    assert np.abs(np.asarray(after4) - np.asarray(before3)).max() < 1e-9


def test_kill_compaction_path_starts_at_pre_kill_position_under_crosswind():
    from animalbench.scenarios.geese.dynamics import target_offset
    from animalbench.scenarios.geese.events import kill_bird
    cfg, st = mk()
    st = _crosswind_state(cfg, st)._replace(t=jnp.asarray(5.0, cfg.jdtype))
    before, _ = target_offset(cfg, st)
    s2 = kill_bird(cfg, st, 3)
    after, _ = target_offset(cfg, s2)
    alive = np.asarray(s2.alive)
    assert np.abs(np.asarray(after)[alive] - np.asarray(before)[alive]).max() < 1e-9


def test_dmax_not_on_slot_lattice():
    """d_max must keep >= 0.25 b from every slot-pair d_x (= 2*a*b, a = 0..7 covers N <= 16)."""
    cfg = GeeseConfig(n_max=14)
    lattice = 2.0 * np.arange(0, 8)
    assert np.abs(lattice - cfg.d_max).min() >= 0.25


def test_random_slots_reset():
    base = GeeseConfig(n_max=9)
    cfg = GeeseConfig(n_max=9, random_slots=True)
    sched = compile_schedule([], cfg.e_max)
    st0 = reset(base, jax.random.PRNGKey(0), 9, sched)
    assert slots(st0) == list(range(9))                                         # default unchanged
    perms = set()
    for sd in range(6):
        st = reset(cfg, jax.random.PRNGKey(sd), 9, sched)
        sl = slots(st)
        assert sorted(sl) == list(range(9))
        perms.add(tuple(sl))
        # birds start near their own slot (formation consistent with the permutation)
        off = np.asarray(slot_offset(cfg, st.slot))
        assert np.abs(np.stack([st.x, st.y], -1) - off).max() < 6 * cfg.init_noise * cfg.b
    assert len(perms) > 1
    # N < n_max: dead birds keep slot -1, alive occupy 0..N-1
    st = reset(GeeseConfig(n_max=12, random_slots=True), jax.random.PRNGKey(1), 7, sched)
    sl = slots(st)
    assert sorted(s for s in sl if s >= 0) == list(range(7)) and all(s == -1 for s in sl[7:])
