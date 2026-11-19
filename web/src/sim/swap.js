// Swap resolution (swap.py): action mask, edge conflicts, lock, immunity, lottery for the apex edges.
import { armSign, occupancy } from './state.js';
import { targetOffset } from './physics.js';

export const STAY = 0, BACK = 1, FWD = 2;

// slot the apex would retreat to (alternating 1,2; falls back to the occupied one). -1 if none.
export function apexTargetSlot(N, apex_ctr) {
  const pref = 1 + apex_ctr, alt = 2 - apex_ctr;
  if (pref < N) return pref;       // every slot < N is occupied (no deaths in the JS port)
  if (alt < N) return alt;
  return -1;
}
const backTarget = (N, k, apex_ctr) => (k === 0 ? apexTargetSlot(N, apex_ctr) : (k + 2 < N ? k + 2 : -1));
const frontSlot = (k) => (k === 0 ? -1 : (k <= 2 ? 0 : k - 2));

// mask[i*3 + a]
export function actionMask(S, out) {
  const N = S.N;
  const m = out || new Uint8Array(N * 3);
  for (let i = 0; i < N; i++) {
    const k = S.slot[i];
    const free = S.lock[i] === 0 && !S.immune[i];
    const hasBack = backTarget(N, k, S.apex_ctr) >= 0;
    const hasFwd = frontSlot(k) >= 0;
    m[i * 3] = 1;
    m[i * 3 + 1] = hasBack && free ? 1 : 0;
    m[i * 3 + 2] = hasFwd && free ? 1 : 0;
  }
  return m;
}

// actions: Int32Array(N) in {0,1,2}; coin: boolean (the bernoulli draw deciding which apex edge goes first).
// Mutates S (slot, path_*, lock, apex_ctr). Returns {swapped, blocked, eff, nSwaps}.
export function resolveSwaps(cfg, S, actions, coin) {
  const N = S.N;
  const mutual = cfg.swap_rule === 'mutual';
  const occ = occupancy(S);
  const mask = actionMask(S);
  const raw = Int32Array.from(actions);
  const eff = new Int32Array(N);
  const forcedStay = new Uint8Array(N);
  const tgtExists = new Uint8Array(N);
  for (let i = 0; i < N; i++) {
    const k = S.slot[i];
    eff[i] = mask[i * 3 + raw[i]] ? raw[i] : 0;
    forcedStay[i] = raw[i] !== 0 && (S.lock[i] > 0 || S.immune[i]) ? 1 : 0;
    const hasBack = backTarget(N, k, S.apex_ctr) >= 0;
    const hasFwd = k > 0 && occ[k <= 2 ? 0 : k - 2] >= 0;
    tgtExists[i] = (raw[i] === BACK && hasBack) || (raw[i] === FWD && hasFwd) ? 1 : 0;
  }
  const actS = new Int32Array(N), btS = new Int32Array(N);
  for (let s = 0; s < N; s++) { actS[s] = eff[occ[s]]; btS[s] = backTarget(N, s, S.apex_ctr); }
  const ok = new Uint8Array(N);
  for (let s = 1; s < N; s++) {
    const f = s <= 2 ? 0 : s - 2;
    const pf = actS[f] === BACK && btS[f] === s;
    const pb = actS[s] === FWD;
    const cand = mutual ? pf && pb : pf || pb;
    const lk = S.lock[occ[s]] > 0 || S.lock[occ[f]] > 0;
    const im = S.immune[occ[s]] || S.immune[occ[f]];
    ok[s] = cand && !lk && !im ? 1 : 0;
  }
  const order = [coin ? 1 : 2, coin ? 2 : 1];
  for (let e = 3; e < N; e++) order.push(e);
  const used = new Uint8Array(N), acc = new Uint8Array(N);
  for (const e of order) {
    if (e >= N) continue;
    const f = e <= 2 ? 0 : e - 2;
    const take = ok[e] && !used[f] && !used[e];
    acc[e] = take ? 1 : 0;
    if (take) { used[f] = 1; used[e] = 1; }
  }
  // execute
  const swapped = new Uint8Array(N);
  const lane = new Float64Array(N);
  const newSlot = Int32Array.from(S.slot);
  let nSwaps = 0;
  for (let e = 1; e < N; e++) {
    if (!acc[e]) continue;
    nSwaps++;
    const f = e <= 2 ? 0 : e - 2;
    const bBack = occ[e], bFront = occ[f];   // bird now in back slot / front slot
    newSlot[bBack] = f; newSlot[bFront] = e;
    swapped[bBack] = 1; swapped[bFront] = 1;
    const side = armSign(e);
    lane[bFront] = side;     // retreating bird goes out on the side of the back slot
    lane[bBack] = -side;
  }
  // path bookkeeping starts from the CURRENT commanded offset (continuity); targetOffset fills S.ufx/ufy
  targetOffset(cfg, S);
  for (let i = 0; i < N; i++) {
    if (!swapped[i]) continue;
    S.path_fx[i] = S.ufx[i]; S.path_fy[i] = S.ufy[i];   // formation frame (not the rotated offx/offy)
    S.path_t0[i] = S.t; S.path_lane[i] = lane[i];
    S.lock[i] = cfg.lock_dec;
  }
  S.slot.set(newSlot);
  if (N > 2 && (acc[1] || acc[2])) S.apex_ctr = acc[1] ? 1 : 0;
  const blocked = new Uint8Array(N);
  for (let i = 0; i < N; i++) {
    blocked[i] = !swapped[i] && ((raw[i] !== 0 && tgtExists[i]) || forcedStay[i]) ? 1 : 0;
  }
  return { swapped, blocked, eff, nSwaps };
}
