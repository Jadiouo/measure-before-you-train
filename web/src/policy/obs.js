// Observation builder for the hier action mode (port of obs.py build_obs, L0 columns only: no command / identity slices,
// which the M1.5 policies do not see) + the network-input rescaling of ippo.prep_obs.  No DOM.
// Column layout (obs.py layout(), 21 dims):
//   0 P_prev | 1 is_apex | 2 rank | 3 arm | 4 lock/lock_dec | 5 blocked | 6..8 last_action one-hot | 9 e_ref/b | 10 n_alive/n
//   | 11,12 wind estimate (ground frame, noisy) | 13..20 (exists, locked) of the neighbours front, backA, backB, opposite.
import { formationHeading, windGlobal } from '../sim/physics.js';
import { slotOffset, occupancy } from '../sim/state.js';

export const OBS_DIM = 21;
export const COL = { P_prev: 0, e_form: 9 };

// slots of (front, backA, backB, opposite) for slot k, -1 if none / outside N (state.py neighbor_slots)
function neighborSlots(N, k, out) {
  const front = k === 0 ? -1 : (k <= 2 ? 0 : k - 2);
  const backA = k === 0 ? 1 : k + 2;
  const backB = k === 0 ? 2 : (k % 2 === 1 ? k + 3 : k + 1);
  const opp = k === 0 ? -1 : (k % 2 === 1 ? k + 1 : k - 1);
  out[0] = front; out[1] = backA; out[2] = backB; out[3] = opp;
  for (let j = 0; j < 4; j++) if (!(out[j] >= 0 && out[j] < N)) out[j] = -1;
}

const tmp2 = new Float64Array(2), nbs = new Int32Array(4);

/**
 * buildObs(cfg, S, windNoise?, out?) -> Float64Array(N * OBS_DIM) raw (un-rescaled) observation, row i = bird i.
 * windNoise: Float64Array(2N) of N(0,1) draws; the per-bird wind estimate is windGlobal + cfg.wind_sigma * noise
 * (JAX uses per-bird PRNG draws, JS cannot reproduce them -> tests inject the JAX draws).  null -> noiseless.
 */
export function buildObs(cfg, S, windNoise = null, out = null) {
  const N = S.N;
  const o = out || new Float64Array(N * OBS_DIM);
  const occ = occupancy(S);
  const [wx, wy] = windGlobal(S);
  const sigma = cfg.wind_sigma ?? 0.3;
  const [psif] = formationHeading(cfg, S);
  const c = Math.cos(psif), s_ = Math.sin(psif);
  const lockDec = cfg.lock_dec;
  for (let i = 0; i < N; i++) {
    const k = Math.max(S.slot[i], 0), b = i * OBS_DIM;
    o.fill(0, b, b + OBS_DIM);
    o[b] = S.P_prev[i];
    o[b + 1] = k === 0 ? 1 : 0;
    o[b + 2] = k / (N - 1);
    o[b + 3] = k === 0 ? 0 : (k % 2 === 1 ? 1 : -1);
    o[b + 4] = S.lock[i] / lockDec;
    o[b + 5] = S.blocked[i] ? 1 : 0;
    o[b + 6 + S.last_action[i]] = 1;
    // e_ref: distance to the LLC slot target (formation frame rotated by the filtered-wind heading)
    slotOffset(cfg, k, tmp2);
    const tx = S.Rx + c * tmp2[0] - s_ * tmp2[1], ty = S.Ry + s_ * tmp2[0] + c * tmp2[1];
    o[b + 9] = Math.hypot(S.x[i] - tx, S.y[i] - ty) / cfg.b;
    o[b + 10] = 1;                                   // n_alive / n (no deaths in the browser sim)
    o[b + 11] = wx + (windNoise ? sigma * windNoise[2 * i] : 0);
    o[b + 12] = wy + (windNoise ? sigma * windNoise[2 * i + 1] : 0);
    neighborSlots(N, k, nbs);
    for (let j = 0; j < 4; j++) {
      const id = nbs[j] >= 0 ? occ[nbs[j]] : -1;
      o[b + 13 + 2 * j] = id >= 0 ? 1 : 0;
      o[b + 14 + 2 * j] = id >= 0 && S.lock[id] > 0 ? 1 : 0;
    }
  }
  return o;
}

/** network input = ippo.prep_obs: P_prev -> (P_prev - 0.9) * 10, e_ref/b clipped to [-3, 3] (returns a new array). */
export function prepObs(raw, N, out = null) {
  const o = out || new Float64Array(raw.length);
  o.set(raw);
  for (let i = 0; i < N; i++) {
    const b = i * OBS_DIM;
    o[b] = (raw[b] - 0.9) * 10;
    o[b + 9] = Math.min(Math.max(raw[b + 9], -3), 3);
  }
  return o;
}
