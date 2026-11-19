"""GeeseConfig: every tunable lives here (spec v0.2 2.10 + v0.3 2.10). Frozen => hashable => jit-static."""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Optional

import jax.numpy as jnp

from . import wake


@dataclass(frozen=True)
class GeeseConfig:
    # ---- shape / mode
    n_max: int = 20
    e_max: int = 8
    action_mode: str = "hier"          # "hier" | "continuous2d"
    dtype: str = "float64"
    T: float = 600.0
    dt: float = 0.1
    delta_h: float = 2.0               # high-level period (s)
    # ---- species (placeholders, see spec)
    m: float = 1.3
    b: float = 1.3
    v0: float = 14.0
    rho: float = 1.2
    SbCD: float = 0.006 * 0.4
    P_pro: float = 4.0
    # ---- wake
    x_d: float = 40.0
    r_c0: float = 0.1
    k_c: float = 0.02
    nq: int = 32
    d_min: float = 0.5
    # d_max = 12.5 b (was 12.0): V slots sit at d_x = 2*a*b (a = arm index) from each other, so for N >= 13 (slots
    # 12, 13 at a = 6) a pair sat EXACTLY on the exclusive window edge 12 b and the wake flickered on/off with any
    # position noise.  12.5 b is >= 0.5 b from the lattice values 12 b and 14 b => no N <= 14 pair is within 0.25 b
    # of the edge (N = 15/16 would put slots 14/15 at 14 b, still 1.5 b away).  Nominal d_x of N <= 9 is <= 8 b.
    d_max: float = 12.5
    gain: Optional[float] = None       # None -> calibrated (max saving at d_x=2b is 12.5%)
    gain_target: float = 0.125
    # ---- dynamics
    turn_rate_max: float = 0.5
    vdot_max: float = 1.0
    vmin_frac: float = 0.7
    vmax_frac: float = 1.3
    # ---- slots / LLC
    slot_dx: float = 2.0               # b units; V arm step
    slot_dy: float = 0.875
    Kp: float = 0.5
    delta_v_max: float = 0.15
    k_psi: float = 8.0
    k_v: float = 4.0
    sep_radius: float = 0.8            # b units
    sep_gain: float = 0.15             # x v0 at full overlap
    swap_lane: float = 0.8             # b units
    path_T: float = 3.0                # s, duration of the swap transition path
    # ---- swap / HLC
    swap_rule: str = "unilateral"      # "unilateral" | "mutual"
    t_lock: float = 8.0
    t_hold_lo: float = 25.0
    t_hold_hi: float = 35.0
    cmd_age_scale: float = 60.0
    cmd_exec_max: int = 3
    oracle_debt: float = 3.0           # P_solo*s
    shared_params: bool = True         # (MARL; unused in M1a)
    # ---- observation
    obs_clock: bool = False
    identity: str = "none"             # none | nbr | nbr+self
    wind_sigma: float = 0.3
    nb_sigma: float = 0.05
    fov_deg: float = 135.0
    sense_radius: float = 8.0          # b units
    llc_wind_noise: float = 0.0
    tau_wind_est: float = 3.0          # s, lag of the formation-frame wind estimate
    # ---- reward
    reward_mode: str = "individual"    # individual | team | mixed
    mix_alpha: float = 0.5
    kappa: float = 0.0
    phi0: float = 0.94
    tau_f: float = 60.0
    c_iso: float = 0.0
    R_c: float = 6.0
    c_prog: float = 0.0
    c_nav: float = 0.0
    c_col: float = 0.0
    E0_coef: Optional[float] = None
    E0_sigma: float = 0.05
    e0_death_penalty: bool = True      # v0.4: after an E0 death every remaining low step pays e0_death_cost (default = solo cost)
    e0_death_cost: float = 1.0         # in P_solo units per step (= remaining time x solo cost)
    # ---- survival-migration mode (default OFF => every existing behaviour/test bit-identical; notes/impl/survival_ippo_prereg.md)
    survival: bool = False
    surv_D: Optional[float] = None     # fixed route length (m); None -> v0*(T - 0.25*delta_h) (reached at the last decision)
    surv_alpha: float = 0.0            # r_i = (1-a)*individual + a*team   (a = kinship)
    surv_beta: float = 0.5             # weight of remaining-energy fraction
    surv_burst_p: float = 0.05         # per decision per bird probability of a burst (gust/injury)
    surv_burst_m: float = 0.0          # m_g: extra energy = m_g * delta_h * Exp(1) * P_solo
    surv_h: float = 0.0                # bird 0 energy cap x (1+h)  (the strong bird)
    surv_shaping: str = "none"         # "none" | "energy": dense per-decision energy-use penalty (telescopes to the terminal energy term at gamma=1)
    surv_lag_tol: float = 13.0         # m: a bird further behind its slot than this at the end has NOT arrived (timeout)
    # v2 model switches (default OFF => bit-identical; notes/impl/survival_v2_prereg.md, survival_env_notes.md)
    surv_obs_abs: bool = False         # obs e_self / e_min_other = E_rem / E_ref with ONE common constant E_ref = E0_coef*P_solo*T (not own E0)
    surv_pw_spread: float = 0.0        # per-bird power multiplier ~ U(1-s, 1+s) (individual energy-use coefficient); 0 = off
    surv_pw_fixed: bool = False        # False: redrawn every episode (from the episode key); True: the same 3 multipliers every episode
    surv_K: int = 1                    # TODO(reserved): number of trips; only 1 implemented
    surv_unknown_end: bool = False     # TODO(reserved): unknown end point; not implemented
    # ---- init
    init_mode: str = "v"               # "v" (perturbed V) | "scatter"
    # random_slots: reset assigns birds to V slots by a random permutation (init_mode "v" only) instead of slot = bird
    # index.  A config field (jit-static, recorded with the run, no change to reset()/make_rollout signatures);
    # default False keeps existing tests and the M1.5 results bit-identical (no extra PRNG draw when False).
    random_slots: bool = False
    init_noise: float = 0.3            # b
    init_box: float = 6.0              # b
    init_psi_deg: float = 30.0
    # ---- logging / switches
    log_D: bool = True
    log_low: bool = True
    log_stride: int = 5
    use_events: bool = True
    ou_tau: float = 5.0
    gust_speed: float = 14.0           # c_g

    def __post_init__(self):
        if self.survival:
            if self.E0_coef is None:
                raise ValueError("survival mode needs E0_coef (energy cap)")
            if self.surv_K != 1 or self.surv_unknown_end:
                raise NotImplementedError("multi-trip K>1 / unknown end point are reserved (TODO), not implemented")
            if self.surv_shaping not in ("none", "energy"):
                raise ValueError("surv_shaping must be 'none' or 'energy'")
            if self.action_mode != "hier":
                raise ValueError("survival mode is implemented for action_mode='hier' only")
        if self.gain is None:
            g, _ = wake.calibrate(self, self.gain_target)
            object.__setattr__(self, "gain", g)

    # derived
    @property
    def jdtype(self):
        return jnp.float64 if self.dtype == "float64" else jnp.float32

    @property
    def n_sub(self) -> int:
        return int(round(self.delta_h / self.dt))

    @property
    def n_dec(self) -> int:
        return int(round(self.T / self.delta_h))

    @property
    def lock_dec(self) -> int:
        return int(round(self.t_lock / self.delta_h))

    @property
    def D_route(self) -> float:
        return float(self.surv_D) if self.surv_D is not None else self.v0 * (self.T - 0.25 * self.delta_h)

    @property
    def P_solo(self) -> float:
        return wake.p_solo(self)

    def replace(self, **kw) -> "GeeseConfig":
        d = {f.name: getattr(self, f.name) for f in dataclasses.fields(self)}
        # re-calibrate gain if a physics/wake parameter changed and gain not given
        if "gain" not in kw and any(k in kw for k in ("m", "b", "v0", "rho", "SbCD", "P_pro", "x_d", "r_c0",
                                                       "k_c", "nq", "d_min", "d_max", "gain_target")):
            d["gain"] = None
        d.update(kw)
        return GeeseConfig(**d)
