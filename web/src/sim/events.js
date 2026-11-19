// Discrete events applied inside the step (events.py): injury, relabel (I1), immune (I2). crosswind is handled in physics.js.
// Event: {kind, t_on, target, ...params, done}. target: bird id | 'leader' | 'last' | 'random'.
import { occupancy } from './state.js';

function resolveTarget(S, e, rand) {
  const t = e.target;
  if (typeof t === 'number') return Math.min(Math.max(t, 0), S.N - 1);
  const occ = occupancy(S);
  if (t === 'leader') return Math.max(occ[0], 0);
  if (t === 'last') return occ[S.N - 1];     // highest slot of the alive birds
  if (t === 'random') return Math.min(Math.floor((rand ? rand.uniform() : 0.5) * S.N), S.N - 1);
  return 0;
}

export function applyEvents(S, rand) {
  for (const e of S.events) {
    if (e.kind === 'crosswind' || e.done) continue;
    if (!(S.t >= e.t_on - 1e-9)) continue;
    const N = S.N;
    if (e.kind === 'injury') {
      const tg = resolveTarget(S, e, rand);
      S.power_mult[tg] = e.power_mult; S.vmax_mult[tg] = e.vmax_mult;
    } else if (e.kind === 'relabel') {
      // new_code[i] = old_code[src[i]]
      let src;
      if (e.perm) src = Int32Array.from({ length: N }, (_, i) => (i < e.perm.length ? e.perm[i] : i));
      else {
        const sh = 1 + Math.min(Math.floor((rand ? rand.uniform() : 0.5) * Math.max(N - 1, 1)), Math.max(N - 2, 0));
        src = Int32Array.from({ length: N }, (_, i) => (i + sh) % N);   // cyclic shift, no fixed point
      }
      const oldCode = Int32Array.from(S.id_code), oldOwn = Int32Array.from(S.id_owner);
      for (let i = 0; i < N; i++) { S.id_code[i] = oldCode[src[i]]; S.id_owner[i] = oldOwn[src[i]]; }
      S.relabel_n += 1;
      if (e.reset_mem) S.mem_reset = true;
      e.srcUsed = Array.from(src);
    } else if (e.kind === 'immune') {
      const tg = resolveTarget(S, e, rand);
      S.immune[tg] = e.on ? 1 : 0;
    }
    e.done = true;
  }
}
