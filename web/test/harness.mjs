// Replays a JAX fixture through the JS sim and returns max errors. Used by consistency.test.mjs and measure_errors.mjs.
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import path from 'node:path';
import { GeeseSim } from '../src/sim/sim.js';
import { ruleBlind, tablePolicy, stayPolicy } from '../src/sim/hlc.js';
import { makeConfig } from '../src/sim/config.js';

export const FIX = path.join(path.dirname(fileURLToPath(import.meta.url)), 'fixtures');
export const loadFixture = (name) => JSON.parse(readFileSync(path.join(FIX, name + '.json'), 'utf8'));

const DISCRETE = ['slot', 'lock', 'leader', 'eff_action', 'swapped', 'blocked', 'id_owner', 'immune'];

export function replay(fx, mode = 'open') {
  const cfg = makeConfig({ swap_rule: fx.swap_rule });
  const policy = mode === 'closed'
    ? ruleBlind(cfg, { drawThold: (i, c) => fx.thold[i][c] })
    : stayPolicy();
  const sim = new GeeseSim({ N: fx.N, cfgObject: cfg, init: fx.init, policy, coinFn: (d) => fx.coins[d] });
  for (const ev of fx.events) sim.S.events.push({ ...ev, done: false });
  const recs = [];
  for (let d = 0; d < fx.n_dec; d++) recs.push(mode === 'open' ? sim.runDecision(fx.actions[d]) : sim.runDecision());
  return { sim, recs, cfg };
}

function maxAbs(recs, ref, field, tf = (a) => a) {
  let m = 0;
  for (let d = 0; d < recs.length; d++) {
    const a = recs[d][field], b = ref[field][d];
    for (let i = 0; i < b.length; i++) m = Math.max(m, Math.abs(tf(a[i]) - tf(b[i])));
  }
  return m;
}

export function compare(fx, recs, cfg) {
  const ref = fx.ref;
  const out = { n_dec: recs.length };
  out.gain_rel = Math.abs(cfg.gain - fx.gain) / fx.gain;
  out.P_solo_rel = Math.abs(cfg.P_solo - fx.P_solo) / fx.P_solo;
  // positions: max over decisions & birds of the euclidean distance (m)
  let dpos = 0, dpsi = 0, dv = 0;
  for (let d = 0; d < recs.length; d++) {
    for (let i = 0; i < fx.N; i++) {
      dpos = Math.max(dpos, Math.hypot(recs[d].x[i] - ref.x[d][i], recs[d].y[i] - ref.y[d][i]));
      dpsi = Math.max(dpsi, Math.abs(recs[d].psi[i] - ref.psi[d][i]));
      dv = Math.max(dv, Math.abs(recs[d].v[i] - ref.v[d][i]));
    }
  }
  out.pos_m = dpos; out.psi_rad = dpsi; out.v_ms = dv;
  // position error up to t = 180 s (the spec's J1 horizon) and formation-relative error (centroid removed)
  let d180 = 0, drel = 0;
  for (let d = 0; d < recs.length; d++) {
    let cx = 0, cy = 0, rx = 0, ry = 0;
    for (let i = 0; i < fx.N; i++) { cx += recs[d].x[i]; cy += recs[d].y[i]; rx += ref.x[d][i]; ry += ref.y[d][i]; }
    cx /= fx.N; cy /= fx.N; rx /= fx.N; ry /= fx.N;
    for (let i = 0; i < fx.N; i++) {
      const e = Math.hypot(recs[d].x[i] - ref.x[d][i], recs[d].y[i] - ref.y[d][i]);
      if (recs[d].t <= 180 + 1e-9) d180 = Math.max(d180, e);
      drel = Math.max(drel, Math.hypot((recs[d].x[i] - cx) - (ref.x[d][i] - rx), (recs[d].y[i] - cy) - (ref.y[d][i] - ry)));
    }
  }
  out.pos_m_t180 = d180; out.pos_rel_m = drel;
  out.P_mean_abs = maxAbs(recs, ref, 'P_mean');          // P_solo units
  out.save_mean_abs = maxAbs(recs, ref, 'save_mean');
  out.e_form_abs_m = maxAbs(recs, ref, 'e_form_end');
  // cumulative energy relative error at the end (per bird), and the max over all decisions
  let eRel = 0;
  for (let d = 0; d < recs.length; d++) for (let i = 0; i < fx.N; i++) {
    const r = ref.E_used[d][i];
    if (r > 0) eRel = Math.max(eRel, Math.abs(recs[d].E_used[i] - r) / r);
  }
  out.E_used_rel = eRel;
  let eFin = 0; const L = recs.length - 1;
  for (let i = 0; i < fx.N; i++) eFin = Math.max(eFin, Math.abs(recs[L].E_used[i] - ref.E_used[L][i]) / ref.E_used[L][i]);
  out.E_final_rel_max = eFin;
  let eTeam = 0;
  for (let d = 0; d < recs.length; d++) {
    let a = 0, b = 0;
    for (let i = 0; i < fx.N; i++) { a += recs[d].E_used[i]; b += ref.E_used[d][i]; }
    eTeam = Math.max(eTeam, Math.abs(a - b) / b);
  }
  out.E_team_rel = eTeam;
  out.discrete_mismatch = {};
  for (const f of DISCRETE) {
    let bad = 0;
    for (let d = 0; d < recs.length; d++) {
      const a = recs[d][f], b = ref[f][d];
      if (Array.isArray(b) || b.length !== undefined) { for (let i = 0; i < b.length; i++) if (Number(a[i]) !== Number(b[i])) bad++; }
      else if (Number(a) !== Number(b)) bad++;
    }
    out.discrete_mismatch[f] = bad;
  }
  return out;
}
