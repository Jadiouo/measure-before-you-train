// Behaviour of the M1b interaction layer (free-rider forced retreat, relabel log, N range, determinism).
import test from 'node:test';
import assert from 'node:assert/strict';
import { GeeseSim } from '../src/sim/sim.js';
import { parseWeightsJSON, marlPolicyFromWeights } from '../src/policy/marl.js';

const run = (sim, secs) => { const n = Math.round(secs / sim.cfg.delta_h); const out = []; for (let d = 0; d < n; d++) out.push(sim.runDecision()); return out; };

test('N = 3..14 all run, stay formed and save energy', () => {
  for (let N = 3; N <= 14; N++) {
    const sim = new GeeseSim({ N, seed: 5, policy: 'rule_blind' });
    run(sim, 120);
    const S_ = sim.teamSaving();
    assert.ok(Number.isFinite(S_) && S_ > 0.02, `N=${N} S=${S_}`);
    for (const v of sim.S.x) assert.ok(Number.isFinite(v));
  }
});

test('same seed -> identical trajectory (seeded PRNG)', () => {
  const a = new GeeseSim({ N: 9, seed: 11 }), b = new GeeseSim({ N: 9, seed: 11 });
  run(a, 100); run(b, 100);
  assert.deepEqual(Array.from(a.S.x), Array.from(b.S.x));
  assert.deepEqual(Array.from(a.S.slot), Array.from(b.S.slot));
});

test('clicking the apex bird: forced retreat first, then it becomes immune and never leads again', () => {
  const sim = new GeeseSim({ N: 9, seed: 3, avoidImmuneApex: true });
  run(sim, 20);
  const leader = sim.S.slot.indexOf(0);
  assert.equal(sim.requestImmune(leader, true), 'pending');
  const recs = run(sim, 16);
  assert.ok(!sim.pending.has(leader), 'pending cleared');
  assert.equal(sim.S.immune[leader], 1);
  assert.notEqual(sim.S.slot[leader], 0);
  const tImm = sim.marks.find((m) => m.kind === 'immune').t;
  run(sim, 200);
  // after becoming immune the bird must never be a leader again, and never swap
  for (const h of sim.hist) if (h.t > tImm + 2) assert.notEqual(h.leader, leader);
  assert.ok(sim.hist.filter((h) => h.t > tImm + 2).some((h) => h.nSwaps > 0), 'others keep rotating');
});

test('without the UI helper a free rider sitting on the retreat slot blocks the apex (known Python-exact limitation)', () => {
  const sim = new GeeseSim({ N: 9, seed: 3, avoidImmuneApex: false });
  run(sim, 4);
  const occ = Array.from(sim.S.slot).map((s, i) => [s, i]).sort((a, b) => a[0] - b[0]).map((p) => p[1]);
  const tgt = 1 + sim.S.apex_ctr;
  sim.requestImmune(occ[tgt], true);
  run(sim, 120);
  assert.equal(sim.hist.filter((h) => h.t > 10).every((h) => h.leader === occ[0]), true);
});

test('relabel: id_owner permutes, sham keeps identity, composes over events', () => {
  const sim = new GeeseSim({ N: 6, seed: 2 });
  run(sim, 4);
  sim.relabel({ perm: sim.swapPairPerm(1, 4), resetMem: true });
  const r = run(sim, 4);
  assert.deepEqual(Array.from(sim.S.id_owner), [0, 4, 2, 3, 1, 5]);
  assert.equal(r[0].mem_reset, true);
  assert.equal(r[1].mem_reset, false);
  sim.relabel({ sham: true });
  run(sim, 2);
  assert.deepEqual(Array.from(sim.S.id_owner), [0, 4, 2, 3, 1, 5]);
  sim.relabel({});                                    // random cyclic shift: no fixed point
  run(sim, 2);
  const before = [0, 4, 2, 3, 1, 5];
  for (let i = 0; i < 6; i++) assert.notEqual(sim.S.id_owner[i], before[i]);
  assert.equal(sim.S.relabel_n, 3);
});

test('wind on/off event: formation recovers after the crosswind ends', () => {
  const sim = new GeeseSim({ N: 9, seed: 4, policy: 'stay' });
  run(sim, 30);
  sim.crosswindOn(3, 90, 0.0);
  run(sim, 60);
  sim.crosswindOff();
  run(sim, 60);
  const ef = sim.lastRec.e_form_end;
  assert.ok(Math.max(...ef) < 0.3, `e_form ${Math.max(...ef)}`);
});

test('MARL weights: validates the file shape; bundled policies parse', () => {
  assert.equal(parseWeightsJSON({}).ok, false);
  assert.throws(() => marlPolicyFromWeights({ meta: { obs_dim: 21, n_actions: 3, hidden: 64 }, actor: {} }), /invalid weights/);
});

// Physics-fix regression (notes/impl/physics_fix_decisions.md): path_from lives in the formation frame, so under a
// crosswind the commanded position right after a swap equals the one right before it. Tolerance 1e-9 m (f64).
test('crosswind: swap transition path starts at the pre-swap commanded position (path_from frame)', async () => {
  const { createState } = await import('../src/sim/state.js');
  const { makeConfig } = await import('../src/sim/config.js');
  const { targetOffset, formationHeading } = await import('../src/sim/physics.js');
  const { resolveSwaps, BACK, FWD } = await import('../src/sim/swap.js');
  const cfg = makeConfig();
  const N = 9;
  const init = { x: Array.from({ length: N }, (_, i) => -2 * cfg.b * Math.ceil(i / 2)), y: new Array(N).fill(0) };
  const S = createState(cfg, N, init);
  S.wf_y = 3.0; S.t = 5.0;                                   // steady crosswind estimate: psi_f ~ -12 deg
  assert.ok(Math.abs(formationHeading(cfg, S)[0]) > 0.1, 'non-vacuous: formation frame is rotated');
  const snap = () => { targetOffset(cfg, S); return [Array.from(S.offx), Array.from(S.offy)]; };
  const maxDiff = (a, b) => Math.max(...a[0].map((v, i) => Math.abs(v - b[0][i])), ...a[1].map((v, i) => Math.abs(v - b[1][i])));
  const before = snap();
  const act = new Int32Array(N); act[3] = BACK;
  const r = resolveSwaps(cfg, S, act, true);
  assert.equal(r.nSwaps, 1);
  assert.ok(maxDiff(before, snap()) < 1e-9, `swap start jump ${maxDiff(before, snap())}`);
  // second swap while a bird is mid-path
  S.t += 1.2; S.lock.fill(0);
  const before2 = snap();
  const act2 = new Int32Array(N); act2[S.slot.indexOf(5)] = FWD;
  assert.equal(resolveSwaps(cfg, S, act2, true).nSwaps, 1);
  assert.ok(maxDiff(before2, snap()) < 1e-9, `mid-path swap start jump ${maxDiff(before2, snap())}`);
});
