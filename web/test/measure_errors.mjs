// node test/measure_errors.mjs  -> prints and writes test/measured_errors.json (used to set the tolerances in consistency.test.mjs)
import { readdirSync, writeFileSync } from 'node:fs';
import { FIX, loadFixture, replay, compare } from './harness.mjs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const out = {};
for (const f of readdirSync(FIX).filter((n) => n.endsWith('.json') && !n.startsWith('wake')).sort()) {
  const name = f.replace('.json', '');
  const fx = loadFixture(name);
  const modes = fx.agent === 'rule_blind' ? ['open', 'closed'] : ['open'];
  for (const mode of modes) {
    const { recs, cfg } = replay(fx, mode);
    out[`${name}:${mode}`] = compare(fx, recs, cfg);
  }
}
const here = path.dirname(fileURLToPath(import.meta.url));
writeFileSync(path.join(here, 'measured_errors.json'), JSON.stringify(out, null, 1));
const fmt = (x) => (typeof x === 'number' ? x.toExponential(2) : x);
console.log('case'.padEnd(46), 'pos_m  pos_t180  pos_rel  v_ms  P_mean  E_final  E_team  eform  discrete');
for (const [k, r] of Object.entries(out)) {
  const bad = Object.values(r.discrete_mismatch).reduce((a, b) => a + b, 0);
  console.log(k.padEnd(46), [r.pos_m, r.pos_m_t180, r.pos_rel_m, r.v_ms, r.P_mean_abs, r.E_final_rel_max, r.E_team_rel, r.e_form_abs_m].map(fmt).join(' '), bad);
}
