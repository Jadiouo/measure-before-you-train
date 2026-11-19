// GeeseConfig defaults (config.py). Only fields used by the JS port are kept; names identical to Python.
import { calibrate, pNowake, quadNodes } from './wake.js';

export const DEFAULTS = {
  dt: 0.1, delta_h: 2.0,
  m: 1.3, b: 1.3, v0: 14.0, rho: 1.2, SbCD: 0.006 * 0.4, P_pro: 4.0,
  x_d: 40.0, r_c0: 0.1, k_c: 0.02, nq: 32, d_min: 0.5, d_max: 12.5,  /* not on the 2b slot lattice (see config.py) */ gain: null, gain_target: 0.125,
  turn_rate_max: 0.5, vdot_max: 1.0, vmin_frac: 0.7, vmax_frac: 1.3,
  slot_dx: 2.0, slot_dy: 0.875, Kp: 0.5, delta_v_max: 0.15, k_psi: 8.0, k_v: 4.0,
  sep_radius: 0.8, sep_gain: 0.15, swap_lane: 0.8, path_T: 3.0,
  swap_rule: 'unilateral', t_lock: 8.0, t_hold_lo: 25.0, t_hold_hi: 35.0,
  wind_sigma: 0.3, tau_wind_est: 3.0, ou_tau: 5.0, init_noise: 0.3,
};

export function makeConfig(over = {}) {
  const c = { ...DEFAULTS, ...over };
  c.nodes = quadNodes(c.nq, c.b);
  if (c.gain == null) c.gain = calibrate(c, c.gain_target);
  c.n_sub = Math.round(c.delta_h / c.dt);
  c.lock_dec = Math.round(c.t_lock / c.delta_h);
  c.P_solo = pNowake(c, c.v0);
  return c;
}
