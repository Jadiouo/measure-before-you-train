// GeeseSim: driver that assembles physics + LLC + HLC into the decision/low-step loop of step.py (decision_step),
// plus the M1b interaction layer (applyEvent API, free-rider forced retreat, history for the viewer).
import { makeConfig } from './config.js';
import { createState, occupancy, slotOffset } from './state.js';
import { llc, stepLow, eForm } from './physics.js';
import { actionMask, resolveSwaps, apexTargetSlot, BACK, FWD } from './swap.js';
import { applyEvents } from './events.js';
import { ruleBlind, stayPolicy } from './hlc.js';
import { mulberry32, makeNormal } from './rng.js';

export function makeRand(seed) {
  const u = mulberry32(seed);
  return { uniform: u, normal: makeNormal(u) };
}

// perturbed-V initial state: slot k at offset(k) + init_noise * b * N(0,1)
export function randomInit(cfg, N, rand) {
  const x = new Float64Array(N), y = new Float64Array(N), o = new Float64Array(2);
  for (let i = 0; i < N; i++) {
    slotOffset(cfg, i, o);
    x[i] = o[0] + cfg.init_noise * cfg.b * rand.normal();
    y[i] = o[1] + cfg.init_noise * cfg.b * rand.normal();
  }
  return { x, y };
}

export class GeeseSim {
  /**
   * opts: {N, seed, config, policy ('rule_blind'|'stay'|policy object), init, coinFn(dec)->bool, drawThold,
   *        avoidImmuneApex (UI helper, default false: Python-exact behaviour)}
   */
  constructor(opts = {}) {
    this.cfg = opts.cfgObject || makeConfig(opts.config || {});
    this.opts = opts;
    this.reset(opts.N ?? 9, opts.seed ?? 1, opts.init);
  }

  reset(N = this.S.N, seed = this.seed, init = null) {
    this.seed = seed;
    this.rand = makeRand(seed);
    const cfg = this.cfg;
    const i0 = init || randomInit(cfg, N, this.rand);
    this.S = createState(cfg, N, i0);
    this.setPolicy(this.opts.policy ?? 'rule_blind');
    this.coinFn = this.opts.coinFn || (() => this.rand.uniform() < 0.5);
    this.avoidImmuneApex = !!this.opts.avoidImmuneApex;
    this.pending = new Map();         // free-rider requests waiting for the apex to retreat: id -> decisions waited
    this.hist = [];                   // per-decision light records for the viewer
    this.marks = [];                  // {t, kind, label, target}
    this.cur = null;
    this.sub = 0;                     // low steps done in the current decision
    this.acc = { P: new Float64Array(N), save: new Float64Array(N) };
    this.prev = { x: Float64Array.from(this.S.x), y: Float64Array.from(this.S.y), psi: Float64Array.from(this.S.psi), Rx: 0 };
    this.lastRec = null;
    this.statusMsg = '';
    // initial wind estimate = global wind at t=0
    const S = this.S;
    this._syncWindEstimate();
    return this;
  }

  _syncWindEstimate() {
    // wf = wind_global at reset; the events array is empty at reset so this is zero (kept explicit)
    const S = this.S;
    S.wf_x = 0; S.wf_y = 0;
  }

  setPolicy(p) {
    if (typeof p === 'string') {
      this.policy = p === 'stay' ? stayPolicy() : ruleBlind(this.cfg, { seed: this.seed, drawThold: this.opts.drawThold });
    } else this.policy = p;
    this.policy.reset?.(this.S.N);
  }

  setSwapRule(rule) { this.cfg.swap_rule = rule; }

  // ------------------------------------------------------------ events (what the UI calls)
  addEvent(ev) {
    const e = { t_on: this.S.t, t_off: null, done: false, ...ev };
    this.S.events.push(e);
    this.marks.push({ t: Math.max(e.t_on, this.S.t), kind: e.kind, label: ev.label || e.kind, target: e.target, on: e.on });
    return e;
  }
  crosswindOn(speed = 3, dirDeg = 90, ouSigma = 0.5, ramp = 5) {
    this.crosswindOff();
    return this.addEvent({ kind: 'crosswind', speed, dirRad: (dirDeg * Math.PI) / 180, ramp, ou_sigma: ouSigma, label: 'wind' });
  }
  crosswindOff() {
    for (const e of this.S.events) if (e.kind === 'crosswind' && e.t_off == null) { e.t_off = this.S.t; this.marks.push({ t: this.S.t, kind: 'crosswind', label: 'wind off' }); }
  }
  injure(i, on = true) {
    return this.addEvent({ kind: 'injury', target: i, power_mult: on ? 1.5 : 1, vmax_mult: on ? 0.85 : 1, label: on ? 'injury' : 'heal' });
  }
  relabel({ perm = null, resetMem = false, sham = false } = {}) {
    const N = this.S.N;
    const p = sham ? Array.from({ length: N }, (_, i) => i) : perm;
    return this.addEvent({ kind: 'relabel', perm: p, reset_mem: resetMem, label: sham ? 'sham' : 'relabel' });
  }
  swapPairPerm(i, j) {
    const q = Array.from({ length: this.S.N }, (_, k) => k); q[i] = j; q[j] = i; return q;
  }
  /** Click-a-bird free rider. on=true: immune (forced retreat first if it is the apex, spec v0.4 sec 7-1). */
  requestImmune(i, on = true) {
    const S = this.S;
    if (!on) { this.pending.delete(i); this.addEvent({ kind: 'immune', target: i, on: false, label: 'immune off' }); return 'off'; }
    if (S.immune[i]) return 'already';
    if (S.slot[i] !== 0) { this.addEvent({ kind: 'immune', target: i, on: true, label: 'immune' }); return 'immediate'; }
    this.pending.set(i, 0);
    this.statusMsg = 'pending-retreat';
    return 'pending';
  }

  // ------------------------------------------------------------ decision machinery
  _preRecord() {
    const S = this.S, occ = occupancy(S);
    return {
      t: S.t, dec: S.dec, leader: occ[0],
      x: Float64Array.from(S.x), y: Float64Array.from(S.y), psi: Float64Array.from(S.psi), v: Float64Array.from(S.v),
      slot: Int32Array.from(S.slot), lock: Int32Array.from(S.lock),
    };
  }

  beginDecision(actionsOverride = null) {
    const S = this.S, cfg = this.cfg, N = S.N;
    const memReset = S.mem_reset;              // policy sees the flag for exactly one decision (cleared right after act())
    const pre = this._preRecord();
    if (this.avoidImmuneApex) this._avoidImmuneApexTarget();
    const mask = actionMask(S);
    let actions = actionsOverride ? Int32Array.from(actionsOverride) : this.policy.act(S, mask, { sim: this, memReset });
    S.mem_reset = false;
    pre.action_req = Int32Array.from(actions);
    if (this.pending.size) actions = this._forceRetreat(actions, mask);
    const coin = this.coinFn(S.dec);
    const sw = resolveSwaps(cfg, S, actions, coin);
    S.last_action.set(sw.eff); S.blocked.set(sw.blocked);          // obs: last_action (effective), blocked
    pre.eff_action = sw.eff; pre.swapped = sw.swapped; pre.blocked = sw.blocked; pre.nSwaps = sw.nSwaps;
    // free riders that just left the apex become immune now (event is applied in this decision's first low step)
    for (const [i, w] of Array.from(this.pending)) {
      if (S.slot[i] !== 0) { this.pending.delete(i); this.addEvent({ kind: 'immune', target: i, on: true, label: 'immune' }); }
      else if (w >= 12) { this.pending.delete(i); this.addEvent({ kind: 'immune', target: i, on: true, label: 'immune (apex)' }); this.statusMsg = 'stuck-at-apex'; }
      else this.pending.set(i, w + 1);
    }
    this.cur = pre;
    this.sub = 0;
    this.acc.P.fill(0); this.acc.save.fill(0);
  }

  _avoidImmuneApexTarget() {
    const S = this.S, N = S.N, occ = occupancy(S);
    const t = apexTargetSlot(N, S.apex_ctr);
    if (t > 0 && S.immune[occ[t]]) {
      const other = t === 1 ? 2 : 1;
      if (other < N && !S.immune[occ[other]]) S.apex_ctr = other - 1;
    }
  }

  _forceRetreat(actions, mask) {
    const S = this.S, N = S.N, occ = occupancy(S);
    const a = Int32Array.from(actions);
    for (const i of this.pending.keys()) {
      if (S.slot[i] !== 0 || S.lock[i] > 0) continue;
      const tgt = apexTargetSlot(N, S.apex_ctr);
      if (tgt < 0 || !mask[i * 3 + BACK]) continue;
      a[i] = BACK;
      const p = occ[tgt];
      if (p >= 0 && mask[p * 3 + FWD]) a[p] = FWD;       // CommandExecutor style: the partner steps up
    }
    return a;
  }

  /** advance one low-level step (dt); starts a new decision when needed. Returns the record when a decision completes. */
  step() {
    const S = this.S, cfg = this.cfg;
    if (this.sub === 0) this.beginDecision();
    const p = this.prev;
    p.x.set(S.x); p.y.set(S.y); p.psi.set(S.psi); p.Rx = S.Rx;
    applyEvents(S, this.rand);
    llc(cfg, S);
    stepLow(cfg, S, this.rand, this.acc);
    this.sub++;
    if (this.sub === cfg.n_sub) return this._endDecision();
    return null;
  }

  _endDecision() {
    const S = this.S, cfg = this.cfg, N = S.N;
    for (let i = 0; i < N; i++) if (S.lock[i] > 0) S.lock[i] -= 1;
    S.dec += 1;
    const rec = this.cur;
    const ns = cfg.n_sub, Ps = cfg.P_solo;
    rec.P_mean = Float64Array.from(this.acc.P, (v) => v / ns / Ps);
    S.P_prev.set(rec.P_mean);
    rec.save_mean = Float64Array.from(this.acc.save, (v) => v / ns);
    rec.E_used = Float64Array.from(S.E_used);
    rec.e_form_end = eForm(cfg, S);
    rec.id_owner = Int32Array.from(S.id_owner);
    rec.immune = Uint8Array.from(S.immune);
    rec.relabel_n = S.relabel_n; rec.mem_reset = S.mem_reset;
    rec.apex_ctr = S.apex_ctr;
    this.lastRec = rec;
    this.hist.push({ t: rec.t, leader: rec.leader, nSwaps: rec.nSwaps });
    if (this.hist.length > 3000) this.hist.shift();
    this.cur = null; this.sub = 0;
    return rec;
  }

  /** one whole decision with optional explicit actions (tests, headless runs) */
  runDecision(actionsOverride = null) {
    this.beginDecision(actionsOverride);
    let rec = null;
    for (let k = 0; k < this.cfg.n_sub; k++) {
      const S = this.S, p = this.prev;
      p.x.set(S.x); p.y.set(S.y); p.psi.set(S.psi); p.Rx = S.Rx;
      applyEvents(S, this.rand);
      llc(this.cfg, S);
      stepLow(this.cfg, S, this.rand, this.acc);
      this.sub++;
    }
    rec = this._endDecision();
    return rec;
  }

  // ------------------------------------------------------------ metrics for the viewer
  teamSaving() {            // S = 1 - E_group / E_solo  (every bird flying solo at P_solo)
    const S = this.S;
    if (S.t <= 0) return 0;
    let e = 0;
    for (let i = 0; i < S.N; i++) e += S.E_used[i];
    return 1 - e / (S.N * this.cfg.P_solo * S.t);
  }
  energyRel() {             // E_used / (P_solo * t) per bird; 1 = solo cost
    const S = this.S, d = this.cfg.P_solo * Math.max(S.t, 1e-9);
    return Float64Array.from(S.E_used, (e) => e / d);
  }
}
