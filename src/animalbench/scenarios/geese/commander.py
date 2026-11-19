"""Announced rotation (spec v0.4 §2.2): commander_view, Command, AnnounceExecutor, scripted_commander and a python-driven
episode runner.  NEW FILE -- nothing in the existing modules is modified.

Design (see notes/impl/M2pre_commander_ledger.md):
  * `commander_view(...)`  : public, deterministic, length-capped summary of the flock (what an LLM would read).
  * `Command`              : {next_leader, order_hint, announce, period}   (spec §2.2 schema).
  * a commander is ANY callable  view(dict) -> Command | None.  `scripted_commander` is the stand-in for the LLM
    (same input, same output type, so an LLM wrapper only has to parse JSON into `Command`).
  * `AnnounceExecutor`     : turns the plan into per-decision `apply_command` writes; the existing
    `swap.executor_override` (called inside `decision_step`) then makes the designated next leader request FWD and the
    bird in front request BACK in the SAME decision, i.e. under swap_rule="mutual" both sides consent.  The birds stay
    self-interested: the base policy (e.g. stay_policy = the learned 'fixed leader' behaviour) is unchanged; a
    `comply_p < 1` knob drops a fraction of the execution ticks (partial obedience).
  * `run_announce_episode` : python loop (jit'ed act/decision_step) -- the commander is called host-side, at most
    `max_calls` (=3, spec §4.0) times per episode, on start / event onset / n_alive change.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Callable, Optional

import jax
import jax.numpy as jnp
import numpy as np

from ...core.perturbation import KIND, compile_schedule
from .config import GeeseConfig
from .env import reset, to_trajectory
from .hlc import Policy
from .obs import build_obs
from .state import occ_from_slot
from .step import decision_step
from .summary import episode_stats
from .swap import action_mask, apply_command

KIND_NAME = {v: k for k, v in KIND.items()}
TARGET_NAME = {-1: "leader", -2: "random", -3: "none", -4: "last"}


@dataclass(frozen=True)
class Command:
    next_leader: Optional[int] = None
    order_hint: tuple = ()
    announce: bool = False
    period: float = 30.0

    @staticmethod
    def from_json(s) -> "Command":
        """Parse an LLM reply (JSON object with the Command fields); unknown fields are ignored."""
        d = json.loads(s) if isinstance(s, str) else dict(s)
        nl = d.get("next_leader")
        return Command(None if nl is None else int(nl), tuple(int(i) for i in d.get("order_hint", ())),
                       bool(d.get("announce", False)), float(d.get("period", 30.0)))


# ------------------------------------------------------------------------------------------------ view
class ViewTracker:
    """Host-side bookkeeping of what is public: cumulative lead seconds, tenures of the apex."""

    def __init__(self, cfg: GeeseConfig):
        self.cfg = cfg
        self.lead_s = np.zeros(cfg.n_max)
        self.apex = -1
        self.apex_since = 0.0
        self.tenures = []                   # completed tenures [(bird, seconds), ...]
        self.last_tenure = np.zeros(cfg.n_max)

    def observe(self, st):
        """Call once per decision BEFORE the commander/executor look at the state (state at decision start)."""
        cfg = self.cfg
        occ = np.asarray(occ_from_slot(cfg, st.slot, st.alive))
        a = int(occ[0])
        t = float(st.t)
        if a != self.apex:
            if self.apex >= 0:
                dur = t - self.apex_since
                self.tenures.append((self.apex, round(dur, 1)))
                self.last_tenure[self.apex] = dur
            self.apex, self.apex_since = a, t
        return a

    def accrue(self):
        """Call after the decision was executed: the apex of the decision start led for delta_h seconds."""
        if self.apex >= 0:
            self.lead_s[self.apex] += self.cfg.delta_h


def commander_view(cfg: GeeseConfig, st, tracker: ViewTracker, n_events: int = 3) -> dict:
    """Public view (spec §2.2).  Contains: t, n_alive, swap_rule, per-alive-bird rows
    {id, slot, lead_share, E_rel, lock, immune, last_tenure}, last tenures, last `n_events` perturbation events.
    NOT contained: identity codes, GRU states, logits, other birds' observations."""
    t = float(st.t)
    alive = np.asarray(st.alive)
    slot = np.asarray(st.slot)
    lock = np.asarray(st.lock)
    imm = np.asarray(st.immune)
    E = np.where(alive, np.asarray(st.E_used), 0.0) / cfg.P_solo
    na = max(int(alive.sum()), 1)
    Erel = np.where(alive, E - E.sum() / na, 0.0)
    share = tracker.lead_s / max(t, cfg.delta_h)
    birds = []
    for i in range(cfg.n_max):
        if not alive[i]:
            continue
        birds.append(dict(id=i, slot=int(slot[i]), lead_share=round(float(share[i]), 3), E_rel=round(float(Erel[i]), 2),
                          lock=int(lock[i]), immune=int(imm[i]), last_tenure=round(float(tracker.last_tenure[i]), 1)))
    birds.sort(key=lambda r: r["id"])
    sc = st.sched
    kid, ton, tgt = np.asarray(sc.kind_id), np.asarray(sc.t_on), np.asarray(sc.target)
    evs = [(float(ton[e]), e) for e in range(len(kid)) if kid[e] != 0 and ton[e] <= t]
    evs.sort()
    events = [dict(kind=KIND_NAME[int(kid[e])], t=round(te, 1),
                   target=(TARGET_NAME.get(int(tgt[e]), int(tgt[e])))) for te, e in evs[-n_events:]]
    return dict(t=round(t, 1), n_alive=int(alive.sum()), swap_rule=cfg.swap_rule, birds=birds,
                tenures=[list(x) for x in tracker.tenures[-3:]], events=events)


def view_to_text(view: dict, max_chars: int = 1800) -> str:
    """Deterministic compact JSON (sorted keys, no spaces).  If longer than `max_chars`, fields are dropped in a fixed
    order (per-bird last_tenure, then E_rel, then tenures, then events) until it fits; if it still does not fit the
    string is hard-cut (marked with a trailing '~')."""
    def dump(v):
        return json.dumps(v, sort_keys=True, separators=(",", ":"))
    s = dump(view)
    v = json.loads(s)
    for drop in ("last_tenure", "E_rel"):
        if len(s) <= max_chars:
            break
        for b in v["birds"]:
            b.pop(drop, None)
        s = dump(v)
    for drop in ("tenures", "events"):
        if len(s) <= max_chars:
            break
        v.pop(drop, None)
        s = dump(v)
    return s if len(s) <= max_chars else s[:max_chars - 1] + "~"


# ------------------------------------------------------------------------------------------------ commanders
def scripted_commander(period: float = 30.0, fair: bool = True) -> Callable[[dict], Command]:
    """Stand-in for the LLM.  Reads only the view.  Plan: ring = alive non-immune birds sorted by lead_share ascending
    (ties: slot, id); `announce=True`, `period` seconds per tenure (default = middle of the rule_blind T_hold range).
    fair=False gives a 'bad' plan (ring in id order ignoring the ledger) for ablations."""
    def commander(view: dict) -> Command:
        rows = [b for b in view["birds"] if not b["immune"]]
        if fair:
            rows.sort(key=lambda b: (b["lead_share"], b["slot"], b["id"]))
        else:
            rows.sort(key=lambda b: b["id"])
        order = tuple(b["id"] for b in rows)
        apex = [b["id"] for b in view["birds"] if b["slot"] == 0]
        nxt = next((i for i in order if not apex or i != apex[0]), None)
        return Command(next_leader=nxt, order_hint=order, announce=True, period=float(period))
    return commander


def silent_commander(view: dict) -> Command:
    return Command(announce=False)


# ------------------------------------------------------------------------------------------------ executor
class AnnounceExecutor:
    """Plan -> `apply_command` writes (python side, one per decision at most).

    * ring successor: the next leader is the ring successor of the CURRENT apex among alive, non-immune birds
      (ring = plan.order_hint; if the apex is not in the ring the first eligible bird is taken).
    * the designee climbs the V whenever it is deeper than slot 2 (one place per decision; executor_override makes the
      front bird retreat -> a mutual swap), and takes the apex when the apex has served `period` seconds and is unlocked.
    * comply_p: probability that an execution tick is actually carried out (1.0 = everybody obeys)."""

    def __init__(self, cfg: GeeseConfig, seed: int = 0, comply_p: float = 1.0):
        self.cfg, self.rng, self.comply_p = cfg, np.random.default_rng(10_000 + seed), comply_p
        self.plan: Optional[Command] = None
        self.n_ticks = 0

    def set_plan(self, cmd: Command):
        self.plan = cmd if cmd.announce else None

    def designee(self, st, apex: int) -> int:
        plan, alive, imm = self.plan, np.asarray(st.alive), np.asarray(st.immune)
        ring = [i for i in plan.order_hint if alive[i] and not imm[i]]
        if not ring:
            return -1
        if apex in ring:
            k = ring.index(apex)
            ring = ring[k + 1:] + ring[:k]
        return ring[0] if ring else -1

    def tick(self, st, tracker: ViewTracker):
        if self.plan is None:
            return st
        cfg = self.cfg
        occ = np.asarray(occ_from_slot(cfg, st.slot, st.alive))
        apex = int(occ[0])
        d = self.designee(st, apex)
        if d < 0 or d == apex:
            return st
        k = int(np.asarray(st.slot)[d])
        due = (float(st.t) - tracker.apex_since) >= self.plan.period - 1e-9
        if k > 2 or due:
            if self.comply_p >= 1.0 or self.rng.random() < self.comply_p:
                self.n_ticks += 1
                return apply_command(cfg, st, next_leader=d, order_hint=list(self.plan.order_hint))
        return st


# ------------------------------------------------------------------------------------------------ runner
def run_announce_episode(cfg: GeeseConfig, base_policy: Policy, commander: Optional[Callable] = None, seed: int = 0,
                         N: int = 3, events=(), comply_p: float = 1.0, max_calls: int = 3, n_dec: int | None = None,
                         view_chars: int = 1800, text_interface: bool = False):
    """Run one episode.  commander=None -> no announcement (base policy alone).  With text_interface=True the commander
    receives the parsed `view_to_text` JSON instead of the raw dict (exactly what an LLM gets).
    Returns (final_state, Trajectory, info) ; info has the commander call log and executor tick count."""
    n_dec = cfg.n_dec if n_dec is None else n_dec
    sched = compile_schedule(list(events), cfg.e_max)
    key = jax.random.PRNGKey(seed)
    st = reset(cfg, key, jnp.int32(N), sched)
    ps = base_policy.init(st.key_policy, cfg)

    @jax.jit
    def act_fn(st, ps):
        obs = build_obs(cfg, st)
        mask = action_mask(cfg, st)
        act, ps2 = base_policy.act(ps, obs, st, mask, jax.random.fold_in(st.key_policy, st.dec))
        occ = occ_from_slot(cfg, st.slot, st.alive)
        pre = dict(x=st.x, y=st.y, psi=st.psi, v=st.v, alive=st.alive, slot=st.slot, lock=st.lock, leader=occ[0],
                   n_alive=jnp.sum(st.alive.astype(jnp.int32)), t=st.t, action_req=act, P_prev=st.P_prev,
                   E_used_start=st.E_used)
        return act, ps2, pre

    @jax.jit
    def step_fn(st, act):
        st2, rec = decision_step(cfg, st, act)
        return st2, rec, st2.E_used, st2.phi, st2.id_owner, st2.relabel_n, st2.mem_reset, st2.immune

    tracker, ex = ViewTracker(cfg), AnnounceExecutor(cfg, seed, comply_p)
    calls, recs = [], []
    seen_events, seen_n = 0, int(N)
    for _ in range(n_dec):
        tracker.observe(st)
        if commander is not None and len(calls) < max_calls:
            n_ev = int(np.sum((np.asarray(st.sched.kind_id) != 0) & (np.asarray(st.sched.t_on) <= float(st.t))))
            n_al = int(np.asarray(st.alive).sum())
            # an event's own consequence (n_alive changing 1-2 decisions after its onset) is not a second trigger
            recent = bool(calls) and (float(st.t) - calls[-1]["t"]) <= 2 * cfg.delta_h + 1e-9
            if len(calls) == 0 or n_ev != seen_events or (n_al != seen_n and not recent):
                view = commander_view(cfg, st, tracker)
                arg = json.loads(view_to_text(view, view_chars)) if text_interface else view
                cmd = commander(arg)
                if cmd is not None:
                    ex.set_plan(cmd)
                calls.append(dict(t=float(st.t), order=None if cmd is None else list(cmd.order_hint),
                                  period=None if cmd is None else cmd.period, announce=bool(cmd and cmd.announce)))
                seen_events, seen_n = n_ev, n_al
            elif n_al != seen_n:
                seen_n = n_al
        act, ps, pre = act_fn(st, ps)
        st = ex.tick(st, tracker)
        st, rec, E_used, phi, id_owner, relabel_n, mem_reset, immune = step_fn(st, act)
        tracker.accrue()
        rec = dict(rec)
        rec.update(pre)
        rec.update(E_used=E_used, phi=phi, id_owner=id_owner, relabel_n=relabel_n, mem_reset=mem_reset, immune=immune)
        recs.append(jax.tree_util.tree_map(np.asarray, rec))
    stacked = jax.tree_util.tree_map(lambda *a: np.stack(a), *recs)
    meta = dict(scenario="geese", agent=base_policy.name + ("+announce" if commander else ""), seed=seed, N=N,
                n_max=cfg.n_max, delta_h=cfg.delta_h, swap_rule=cfg.swap_rule, P_solo=cfg.P_solo)
    traj = to_trajectory(cfg, stacked, meta, sched)
    info = dict(calls=calls, ticks=ex.n_ticks, lead_share=(tracker.lead_s / max(n_dec * cfg.delta_h, 1)).tolist())
    return st, traj, info


def announce_stats(traj, cfg, N: int) -> dict:
    """Summary used in validation: episode_stats + lead share per bird + cost per bird (P_solo*s, positive = cost)."""
    s = episode_stats(traj, cfg)
    lead = np.asarray(traj.global_fields["leader"])
    share = np.array([(lead == i).mean() for i in range(N)])
    a = traj.agent_fields
    cost = (np.asarray(a["P_mean"]) * np.asarray(a["alive"]).astype(float)).sum(0)[:N] * cfg.delta_h
    return dict(handover_rate_per_s=s["handover_rate_per_s"], handovers=s["handovers"], n_swaps=s["n_swaps_total"],
                lead_share=share.round(3).tolist(), max_share_dev=float(np.abs(share - 1.0 / N).max()),
                cost_per_bird=cost.round(2).tolist(), cost_mean=float(cost.mean()), S=s["S"], E_group=s["E_group"])
