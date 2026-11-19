// MARL policy for the browser sim: same interface as ruleBlind -> {name, reset(N), act(S, mask, ctx) -> Int32Array(N)}.
// One shared GRU actor, one hidden state per BIRD (body), exactly as in training (ippo.py).  N = 3 only.
import { buildObs, prepObs, OBS_DIM } from './obs.js';
import { GruActor, decodeB64F32, maskedProbs } from './gru.js';
import { mulberry32, makeNormal } from '../sim/rng.js';
import { POLICY_WEIGHTS } from './weights_data.js';

const KEYS = ['w1', 'b1', 'wi', 'wh', 'bi', 'bh', 'wpi', 'bpi'];

export function parseWeightsJSON(obj) {
  const errs = [];
  if (!obj || typeof obj !== 'object') errs.push('not an object');
  else {
    if (!obj.meta || typeof obj.meta !== 'object') errs.push('missing meta');
    else for (const k of ['obs_dim', 'n_actions', 'hidden']) if (!Number.isFinite(obj.meta[k])) errs.push(`meta.${k} missing`);
    if (!obj.actor || typeof obj.actor !== 'object') errs.push('missing actor');
    else for (const k of KEYS) if (!obj.actor[k] || typeof obj.actor[k].b64 !== 'string') errs.push(`actor.${k} missing`);
    if (!errs.length) {
      const { obs_dim: D, hidden: H, n_actions: A } = obj.meta;
      const want = { w1: D * H, b1: H, wi: 3 * H * H, wh: 3 * H * H, bi: 3 * H, bh: 3 * H, wpi: H * A, bpi: A };
      for (const k of KEYS) {
        const n = obj.actor[k].shape ? obj.actor[k].shape.reduce((a, b) => a * b, 1) : -1;
        if (n !== want[k]) errs.push(`actor.${k} shape ${JSON.stringify(obj.actor[k].shape)} != ${want[k]} elements`);
      }
      if (D !== OBS_DIM || A !== 3) errs.push(`unsupported obs_dim/n_actions (${D}/${A}), expected ${OBS_DIM}/3`);
    }
  }
  return { ok: errs.length === 0, errors: errs };
}

/**
 * obj: {meta, actor} as written by scripts/export_policy_js.py.  cfg: sim config (for lock_dec, b, wind_sigma, slot geometry).
 * opts: {mode: 'argmax'|'sample', seed, noise: (S) -> Float64Array(2N) | null (inject wind-estimate noise; default: seeded N(0,1)),
 *        record: fn(info) called per decision with {obs, prep, probs, h} (tests)}.
 */
export function marlPolicyFromWeights(obj, cfg, opts = {}) {
  const v = parseWeightsJSON(obj);
  if (!v.ok) throw new Error('invalid weights JSON: ' + v.errors.join('; '));
  const { obs_dim: D, hidden: H } = obj.meta;
  const w = {};
  for (const k of KEYS) w[k] = decodeB64F32(obj.actor[k].b64);
  const net = new GruActor(w, D, H);
  const mode = opts.mode || 'argmax';
  const urand = mulberry32((opts.seed ?? 1) >>> 0), nrand = makeNormal(mulberry32(((opts.seed ?? 1) * 2654435761 + 1) >>> 0));
  let hs = null, N = 0;
  const logits = new Float64Array(3), pr = new Float64Array(3);
  const reset = (n) => { N = n; hs = Array.from({ length: n }, () => new Float64Array(H)); };
  return {
    name: obj.meta.name || 'marl', meta: obj.meta, mode, reset,
    act(S, mask, ctx) {
      if (S.N !== 3) throw new Error(`MARL policy was trained for N=3 (got N=${S.N})`);
      if (!hs || N !== S.N) reset(S.N);
      let noise = null;
      if (opts.noise) noise = opts.noise(S);
      else if (cfg.wind_sigma !== 0) { noise = new Float64Array(2 * S.N); for (let i = 0; i < noise.length; i++) noise[i] = nrand(); }
      const raw = buildObs(cfg, S, noise);
      const x = prepObs(raw, S.N);
      const a = new Int32Array(S.N);
      for (let i = 0; i < S.N; i++) {
        if (ctx && ctx.memReset) hs[i].fill(0);
        net.step(hs[i], x, i * D, logits);
        maskedProbs(logits, mask, i * 3, pr);
        let act = 0;
        if (mode === 'sample') {
          const u = urand();
          act = u < pr[0] ? 0 : (u < pr[0] + pr[1] ? 1 : 2);
          if (!mask[i * 3 + act]) act = 0;
        } else {
          for (let q = 1; q < 3; q++) if (pr[q] > pr[act]) act = q;
        }
        a[i] = act;
        if (opts.record) opts.record({ d: S.dec, i, raw, x, probs: Float64Array.from(pr), h: Float64Array.from(hs[i]) });
      }
      return a;
    },
  };
}

export const BRAINS = Object.keys(POLICY_WEIGHTS);          // ['unilateral', 'mutual']
export function bundledMarlPolicy(rule, cfg, opts) {
  if (!POLICY_WEIGHTS[rule]) throw new Error('no bundled MARL policy for rule ' + rule);
  return marlPolicyFromWeights(POLICY_WEIGHTS[rule], cfg, opts);
}
