// M1.5 policy export: JS obs builder + GRU actor vs the JAX reference (scripts/export_policy_js.py -> fixtures/policy_N3.json),
// and an end-to-end closed-loop run (JS env + JS policy, N=3, no wind, 600 s).
import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import path from 'node:path';
import { FIX } from './harness.mjs';
import { GeeseSim } from '../src/sim/sim.js';
import { makeConfig } from '../src/sim/config.js';
import { buildObs, prepObs, OBS_DIM } from '../src/policy/obs.js';
import { GruActor, decodeB64F32, maskedProbs } from '../src/policy/gru.js';
import { bundledMarlPolicy, parseWeightsJSON } from '../src/policy/marl.js';
import { POLICY_WEIGHTS } from '../src/policy/weights_data.js';

const fx = JSON.parse(readFileSync(path.join(FIX, 'policy_N3.json'), 'utf8')).scenarios;
const stats = { obs: 0, prep: 0, tv: 0, h: 0, agree: 0, total: 0, maskBad: 0 };

for (const sc of fx) {
  test(`obs builder vs JAX: ${sc.name}`, () => {
    const cfg = makeConfig({ swap_rule: sc.swap_rule });
    let maxRaw = 0, maxPrep = 0, d = 0, maskBad = 0;
    const col = new Float64Array(OBS_DIM);
    const rec = {
      name: 'fixture', reset() {},
      act(S, mask) {
        const noise = Float64Array.from(sc.wn[d].flat());
        const raw = buildObs(cfg, S, noise);
        const x = prepObs(raw, S.N);
        for (let i = 0; i < S.N; i++) for (let j = 0; j < OBS_DIM; j++) {
          const e = Math.abs(raw[i * OBS_DIM + j] - sc.obs[d][i][j]);
          maxRaw = Math.max(maxRaw, e); col[j] = Math.max(col[j], e);
          maxPrep = Math.max(maxPrep, Math.abs(x[i * OBS_DIM + j] - sc.obs_prep[d][i][j]));
        }
        for (let k = 0; k < 9; k++) if (mask[k] !== sc.mask[d].flat()[k]) maskBad++;
        return Int32Array.from(sc.act[d]);
      },
    };
    const sim = new GeeseSim({ N: sc.N, cfgObject: cfg, init: sc.init, policy: rec, coinFn: (k) => sc.coin[k] });
    for (const ev of sc.events) sim.S.events.push({ ...ev, done: false });
    for (d = 0; d < sc.n_dec; d++) sim.runDecision();
    console.log(`  ${sc.name}: obs max err ${maxRaw.toExponential(2)}, prepped ${maxPrep.toExponential(2)}, mask mismatches ${maskBad}`);
    stats.obs = Math.max(stats.obs, maxRaw); stats.prep = Math.max(stats.prep, maxPrep); stats.maskBad += maskBad;
    assert.equal(maskBad, 0);
    // discrete / structural columns are exact; wind estimate = formula only (fixture noise injected, no OU); the two continuous
    // columns carry the f32-JAX vs f64-JS drift of the open-loop replay: P_prev (wake power, 1e-3 level) and e_ref/b
    // (position error in a ground frame growing to ~8 km, f32 ulp ~1 mm)
    for (const j of [1, 2, 3, 4, 5, 6, 7, 8, 10, 13, 14, 15, 16, 17, 18, 19, 20]) assert.ok(col[j] < 1e-9, `col ${j} err ${col[j]}`);
    assert.ok(col[11] < 1e-5 && col[12] < 1e-5, `wind cols ${col[11]} ${col[12]}`);
    assert.ok(col[0] < 1e-2, `P_prev ${col[0]}`);
    assert.ok(col[9] < 2e-2, `e_ref/b ${col[9]}`);
    assert.ok(maxPrep < 0.1, `prepped obs err ${maxPrep}`);   // P_prev is scaled x10
  });

  test(`GRU actor vs JAX probabilities: ${sc.name}`, () => {
    const W = POLICY_WEIGHTS[sc.policy];
    const w = Object.fromEntries(Object.entries(W.actor).map(([k, v]) => [k, decodeB64F32(v.b64)]));
    const net = new GruActor(w, W.meta.obs_dim, W.meta.hidden);
    const hs = [0, 1, 2].map(() => new Float64Array(W.meta.hidden));
    const lg = new Float64Array(3), pr = new Float64Array(3);
    let tvMax = 0, hMax = 0, agree = 0, total = 0;
    for (let d = 0; d < sc.n_dec; d++) {
      for (let i = 0; i < 3; i++) {
        if (sc.mem_reset[d]) hs[i].fill(0);
        net.step(hs[i], Float64Array.from(sc.obs_prep[d][i]), 0, lg);
        maskedProbs(lg, Uint8Array.from(sc.mask[d][i]), 0, pr);
        const ref = sc.probs[d][i];
        let tv = 0, ja = 0, ra = 0;
        for (let a = 0; a < 3; a++) { tv += 0.5 * Math.abs(pr[a] - ref[a]); if (pr[a] > pr[ja]) ja = a; if (ref[a] > ref[ra]) ra = a; }
        tvMax = Math.max(tvMax, tv);
        for (let j = 0; j < hs[i].length; j++) hMax = Math.max(hMax, Math.abs(hs[i][j] - sc.h_out[d][i][j]));
        total++; if (ja === ra) agree++;
      }
    }
    console.log(`  ${sc.name}: TV max ${tvMax.toExponential(2)}, hidden max err ${hMax.toExponential(2)}, argmax agreement ${agree}/${total}`);
    stats.tv = Math.max(stats.tv, tvMax); stats.h = Math.max(stats.h, hMax); stats.agree += agree; stats.total += total;
    assert.ok(tvMax < 1e-3, `TV ${tvMax}`);
    assert.equal(agree, total, 'argmax agreement must be 100%');
  });
}

test('consistency summary', () => {
  console.log(`  SUMMARY obs max err ${stats.obs.toExponential(2)} (prepped ${stats.prep.toExponential(2)}), TV max ${stats.tv.toExponential(2)}, ` +
    `hidden max ${stats.h.toExponential(2)}, argmax agreement ${stats.agree}/${stats.total} = ${(100 * stats.agree / stats.total).toFixed(2)}%`);
});

test('bundled weights parse and carry the expected metadata', () => {
  for (const rule of ['unilateral', 'mutual']) {
    const W = POLICY_WEIGHTS[rule];
    assert.ok(parseWeightsJSON(W).ok);
    assert.equal(W.meta.obs_dim, 21); assert.equal(W.meta.hidden, 64); assert.equal(W.meta.rule, rule);
  }
});

// ---------------------------------------------------------------- end to end
function swapRate(rule, mode, seed, T = 600) {
  const cfg = makeConfig({ swap_rule: rule });
  const pol = bundledMarlPolicy(rule, cfg, { mode, seed });
  const sim = new GeeseSim({ N: 3, seed, cfgObject: cfg, policy: pol });
  let swaps = 0; const lead = [];
  const nd = Math.round(T / cfg.delta_h);
  for (let d = 0; d < nd; d++) { const r = sim.runDecision(); swaps += r.nSwaps; lead.push(r.leader); }
  let changes = 0; for (let d = 1; d < lead.length; d++) if (lead[d] !== lead[d - 1]) changes++;
  return { rate: swaps / T, changes, share: [0, 1, 2].map((b) => lead.filter((x) => x === b).length / lead.length), saving: sim.teamSaving() };
}

for (const mode of ['argmax', 'sample']) {
  test(`end-to-end N=3 no wind 600 s (${mode}): unilateral ~0.125 swaps/s, mutual ~0`, () => {
    for (const seed of [1, 2, 3]) {
      const u = swapRate('unilateral', mode, seed), m = swapRate('mutual', mode, seed);
      console.log(`  ${mode} seed ${seed}: unilateral ${u.rate.toFixed(4)}/s (leader changes ${u.changes}, shares ${u.share.map((x) => x.toFixed(2))}, S ${u.saving.toFixed(3)}); ` +
        `mutual ${m.rate.toFixed(4)}/s (changes ${m.changes}, shares ${m.share.map((x) => x.toFixed(2))}, S ${m.saving.toFixed(3)})`);
      assert.ok(Math.abs(u.rate - 0.125) < 0.01, `unilateral rate ${u.rate}`);
      assert.ok(m.rate < 0.005, `mutual rate ${m.rate}`);
    }
  });
}
