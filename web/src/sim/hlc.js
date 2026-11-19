// Rule-based high-level controllers (hlc.py): rule_blind (L0), stay. Interface shared with a future MARL policy:
//   policy.act(S, mask, ctx) -> Int32Array(N) of {STAY,BACK,FWD}
import { STAY, BACK, FWD } from './swap.js';
import { hashUniform } from './rng.js';

// drawThold(i, cnt) -> hold time (s). Default: seeded hash (not bit-equal to JAX); tests inject the JAX draws.
export function ruleBlind(cfg, { seed = 0, drawThold } = {}) {
  const lo = cfg.t_hold_lo, hi = cfg.t_hold_hi;
  const draw = drawThold || ((i, c) => lo + (hi - lo) * hashUniform(seed, i, c));
  const mutual = cfg.swap_rule === 'mutual';
  let ps = null;
  const init = (N) => {
    ps = { held: new Float64Array(N), thold: new Float64Array(N).fill(lo), wasApex: new Uint8Array(N), cnt: new Int32Array(N) };
  };
  return {
    name: 'rule_blind',
    reset: init,
    act(S, mask) {
      const N = S.N;
      if (!ps || ps.held.length !== N) init(N);
      const a = new Int32Array(N);
      for (let i = 0; i < N; i++) {
        const isApex = S.slot[i] === 0;
        const lock0 = S.lock[i] === 0;
        const newApex = isApex && !ps.wasApex[i];
        const thold = newApex ? draw(i, ps.cnt[i]) : ps.thold[i];
        const held = newApex ? 0 : ps.held[i];
        if (newApex) ps.cnt[i] += 1;
        const retreat = isApex && lock0 && held >= thold;
        let act = retreat ? BACK : STAY;
        if (mutual && !isApex && lock0 && S.slot[i] >= 1 && S.slot[i] <= 2) act = FWD;
        if (!mask[i * 3 + act]) act = STAY;
        a[i] = act;
        ps.held[i] = isApex ? (lock0 ? held + cfg.delta_h : held) : 0;
        ps.thold[i] = thold;
        ps.wasApex[i] = isApex ? 1 : 0;
      }
      return a;
    },
    state: () => ps,
  };
}

export function stayPolicy() {
  return { name: 'stay', reset() {}, act: (S) => new Int32Array(S.N) };
}

// open-loop table: table[dec][i]; masked to STAY when illegal (respect_mask=True in Python)
export function tablePolicy(table) {
  return {
    name: 'table', reset() {},
    act(S, mask) {
      const row = table[Math.min(S.dec, table.length - 1)];
      return Int32Array.from({ length: S.N }, (_, i) => (mask[i * 3 + row[i]] ? row[i] : STAY));
    },
  };
}
