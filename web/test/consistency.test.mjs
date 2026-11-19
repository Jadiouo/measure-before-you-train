// JAX <-> JS consistency (spec v0.3 M1b J1/J2, v0.4 sec 7-5). Fixtures are produced by scripts/export_js_fixtures.py.
// Tolerances were set AFTER measuring (test/measured_errors.json, notes/impl/M1b_decisions.md), about 2-10x the measured max.
import test from 'node:test';
import assert from 'node:assert/strict';
import { readdirSync, readFileSync } from 'node:fs';
import path from 'node:path';
import { FIX, loadFixture, replay, compare } from './harness.mjs';
import { makeConfig } from '../src/sim/config.js';
import { quadNodes, wbar } from '../src/sim/wake.js';

const names = readdirSync(FIX).filter((n) => n.endsWith('.json') && !n.startsWith('wake') && !n.startsWith('policy')).map((n) => n.replace('.json', '')).sort();

// (Until the 2026-10 physics fix N >= 13 had slot pairs exactly at d_x = d_max = 12 b and needed looser P/E tolerances;
// d_max is now 12.5 b, off the 2b slot lattice, so all fixtures use the same tolerances.)

function checkDiscrete(r, label) {
  for (const [k, bad] of Object.entries(r.discrete_mismatch)) assert.equal(bad, 0, `${label}: ${k} mismatches`);
}

for (const name of names) {
  const fx = loadFixture(name);
  const f32 = fx.dtype === 'float32';
  const modes = fx.agent === 'rule_blind' ? ['open', 'closed'] : ['open'];
  for (const mode of modes) {
    test(`${name} [${mode}-loop, ${fx.dtype}, N=${fx.N}, ${fx.swap_rule}]`, () => {
      const { recs, cfg } = replay(fx, mode);
      const r = compare(fx, recs, cfg);
      assert.ok(r.gain_rel < 1e-12 && r.P_solo_rel < 1e-12, 'gain / P_solo calibration');
      // J2: swap decisions, slot assignment, lock, blocked flags, id_owner, immune: exactly equal per decision
      checkDiscrete(r, name);
      if (!f32) {
        assert.ok(r.pos_m < 1e-9, `pos ${r.pos_m}`);                   // spec J1 asks < 1e-6 m
        assert.ok(r.psi_rad < 1e-11 && r.v_ms < 1e-9);
        assert.ok(r.e_form_abs_m < 1e-9);
        assert.ok(r.E_team_rel < 1e-5, `E_team ${r.E_team_rel}`);        // spec: < 0.1 %
        assert.ok(r.P_mean_abs < 1e-10, `P_mean ${r.P_mean_abs}`);
        assert.ok(r.E_final_rel_max < 1e-11, `E_final ${r.E_final_rel_max}`);
      } else {
        assert.ok(r.pos_m_t180 < 0.05, `f32 pos<=180s ${r.pos_m_t180}`);   // spec guess 0.05 m: holds up to 180 s
        assert.ok(r.pos_m < 0.25, `f32 pos(all t) ${r.pos_m}`);            // grows with ground coordinate (x ~ 14 m/s * t)
        assert.ok(r.pos_rel_m < 0.01, `f32 formation-relative pos ${r.pos_rel_m}`);
        assert.ok(r.v_ms < 0.01 && r.psi_rad < 1e-4);
        assert.ok(r.e_form_abs_m < 0.01);
        assert.ok(r.E_team_rel < 1e-3, `E_team ${r.E_team_rel}`);
        assert.ok(r.E_final_rel_max < 1e-3, `E_final ${r.E_final_rel_max}`);
        assert.ok(r.P_mean_abs < 1e-3, `P_mean ${r.P_mean_abs}`);
      }
    });
  }
}

test('wake field wbar: JAX f64 and f32 vs JS on a (d_x, d_y, v) grid', () => {
  const g = JSON.parse(readFileSync(path.join(FIX, 'wake_grid.json'), 'utf8'));
  const cfg = makeConfig();
  assert.ok(Math.abs(cfg.gain - g.gain) / g.gain < 1e-12);
  const nodes = quadNodes(cfg.nq, cfg.b);
  let e64 = 0, e32 = 0;
  const keys = Object.keys(g.w);                     // python formats the speeds as '10.0', '14.0', ...
  for (let q = 0; q < g.v.length; q++) {
    const v = g.v[q], ref = g.w[keys[q]], ref32 = g.w32[keys[q]];
    let scale = 0;
    for (const row of ref) for (const w of row) scale = Math.max(scale, Math.abs(w));
    for (let i = 0; i < g.dx.length; i++) for (let j = 0; j < g.dy.length; j++) {
      const w = wbar(cfg, nodes, g.dx[i], g.dy[j], v);
      e64 = Math.max(e64, Math.abs(w - ref[i][j]) / scale);
      e32 = Math.max(e32, Math.abs(w - ref32[i][j]) / scale);
    }
  }
  console.log(`  wake grid max err / max|w|: f64 ${e64.toExponential(2)}, f32 ${e32.toExponential(2)}`);
  assert.ok(e64 < 1e-12, `f64 ${e64}`);
  assert.ok(e32 < 1e-4, `f32 ${e32}`);                                   // spec: relative 1e-4
});
