"""Scenario-independent perturbation description + fixed-shape compiled schedule (E_max events)."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, NamedTuple, Sequence

import numpy as np

BIG = 1e12  # "never" time (finite, avoids inf-inf NaNs inside jit)
KIND = dict(none=0, crosswind=1, gust=2, injury=3, join=4, leave=5, relabel=6, immune=7)
PMAX = 32  # max n_max supported by the (E, PMAX) relabel permutation table
# parameter slots per kind (p has NP columns)
NP = 6
PARAM_SLOTS = {
    "crosswind": ("speed", "dir_deg", "ramp", "ou_sigma"),
    "gust": ("amp", "dir_deg", "spatial"),           # spatial 0=uniform, 1=front sweep (c_g)
    "injury": ("power_mult", "vmax_mult", "E_loss"),
    "join": ("n",),
    "leave": (),
    "relabel": ("reset_mem",),           # 1 = also ask the policy to reset its memory (policy side)
    "immune": ("on",),                  # 1 = become a never-leads free-rider, 0 = clear
}
DEFAULTS = {
    "crosswind": dict(speed=3.0, dir_deg=90.0, ramp=5.0, ou_sigma=0.5),
    "gust": dict(amp=5.0, dir_deg=float("nan"), spatial=0.0),   # nan dir -> random from 'pert' stream
    "injury": dict(power_mult=1.5, vmax_mult=0.85, E_loss=0.10),
    "join": dict(n=1.0),
    "leave": dict(),
    "relabel": dict(reset_mem=0.0),
    "immune": dict(on=1.0),
}
TARGET_LEADER, TARGET_RANDOM, TARGET_NONE, TARGET_LAST = -1, -2, -3, -4


@dataclass(frozen=True)
class Perturbation:
    kind: str
    t_on: float
    t_off: float | None = None
    target: int | str | None = None
    params: Mapping[str, float] = field(default_factory=dict)
    perm: tuple | None = None   # relabel only: new_code[i] = old_code[perm[i]]; None -> random cyclic shift of alive birds


class PerturbSchedule(NamedTuple):
    kind_id: np.ndarray   # (E,) int32
    t_on: np.ndarray      # (E,)
    t_off: np.ndarray     # (E,)
    target: np.ndarray    # (E,) int32  (>=0 bird id, -1 leader, -2 random alive, -3 none)
    p: np.ndarray         # (E, NP)
    perm: np.ndarray      # (E, PMAX) int32 relabel permutation; row[0] == -1 -> random cyclic shift among alive birds


def _target_code(t) -> int:
    if t is None:
        return TARGET_NONE
    if isinstance(t, str):
        return {"leader": TARGET_LEADER, "random": TARGET_RANDOM, "none": TARGET_NONE, "last": TARGET_LAST}[t]
    return int(t)


def compile_schedule(events: Sequence[Perturbation], e_max: int = 8) -> PerturbSchedule:
    if len(events) > e_max:
        raise ValueError(f"{len(events)} events > e_max={e_max}")
    kind = np.zeros(e_max, np.int32)
    t_on = np.full(e_max, BIG)
    t_off = np.full(e_max, BIG)
    target = np.full(e_max, TARGET_NONE, np.int32)
    p = np.zeros((e_max, NP))
    perm = np.tile(np.arange(PMAX, dtype=np.int32), (e_max, 1))
    for e, ev in enumerate(events):
        kind[e] = KIND[ev.kind]
        t_on[e] = ev.t_on
        t_off[e] = BIG if ev.t_off is None else ev.t_off
        target[e] = _target_code(ev.target)
        prm = dict(DEFAULTS[ev.kind]); prm.update(ev.params)
        for j, name in enumerate(PARAM_SLOTS[ev.kind]):
            p[e, j] = prm[name]
        if ev.kind == "relabel":
            if ev.perm is None:
                perm[e, 0] = -1
            else:
                q = np.asarray(ev.perm, np.int32)
                if len(q) > PMAX or sorted(q.tolist()) != list(range(len(q))):
                    raise ValueError("perm must be a permutation of 0..len-1 (len <= PMAX)")
                perm[e, :len(q)] = q
    return PerturbSchedule(kind, t_on, t_off, target, p, perm)


# convenience constructors -------------------------------------------------
def crosswind(t_on, duration=40.0, speed=3.0, dir_deg=90.0, ramp=5.0, ou_sigma=0.5):
    return Perturbation("crosswind", t_on, t_on + duration, None,
                        dict(speed=speed, dir_deg=dir_deg, ramp=ramp, ou_sigma=ou_sigma))


def gust(t_on, duration=3.0, amp=5.0, dir_deg=float("nan"), spatial=0.0):
    return Perturbation("gust", t_on, t_on + duration, None, dict(amp=amp, dir_deg=dir_deg, spatial=spatial))


def injury(t_on, target="leader", power_mult=1.5, vmax_mult=0.85, E_loss=0.10):
    return Perturbation("injury", t_on, None, target,
                        dict(power_mult=power_mult, vmax_mult=vmax_mult, E_loss=E_loss))


def join(t_on, n=1):
    return Perturbation("join", t_on, None, None, dict(n=float(n)))


def leave(t_on, target="random"):
    return Perturbation("leave", t_on, None, target, {})


def relabel(t_on, perm=None, reset_mem=False):
    """Identity-code relabelling at t_on.  perm=None: random cyclic shift among alive birds (no fixed point, drawn from the
    'pert' stream); perm=list: explicit permutation (bird i then carries the code bird perm[i] carried; identity list =
    sham).  reset_mem only sets the env flag `mem_reset` for the policy; the env never touches policy memory."""
    return Perturbation("relabel", t_on, None, None, dict(reset_mem=float(reset_mem)),
                        None if perm is None else tuple(int(i) for i in perm))


def swap_pair_perm(N, i, j, n=None):
    q = list(range(N if n is None else n)); q[i], q[j] = q[j], q[i]
    return q


def immune_on(t_on, target, on=True):
    """Turn bird `target` (id, or 'leader'/'random'/'last' = last slot) into a never-leads free-rider (swap-immune)."""
    return Perturbation("immune", t_on, None, target, dict(on=1.0 if on else 0.0))
