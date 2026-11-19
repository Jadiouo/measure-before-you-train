// Wind, wake matrix, target path, LLC, physics step. Ports dynamics.py / llc.py / step.py (step_continuous).
import { G, wbar, pNowake } from './wake.js';
import { slotOffset } from './state.js';

// ---------------------------------------------------------------- wind
// Event schema (JS): {kind:'crosswind', t_on, t_off|null, dirRad, speed, ramp, ou_sigma}
export function crosswindDet(S, t) {
  let wx = 0, wy = 0, ous = 0;
  for (const e of S.events) {
    if (e.kind !== 'crosswind') continue;
    const ramp = Math.max(e.ramp, 1e-9);
    const t_off = e.t_off == null ? 1e12 : e.t_off;
    const env = Math.min(Math.max((t - e.t_on) / ramp, 0), 1) * Math.min(Math.max((t_off + ramp - t) / ramp, 0), 1);
    wx += env * e.speed * Math.cos(e.dirRad);
    wy += env * e.speed * Math.sin(e.dirRad);
    ous += env * e.ou_sigma;
  }
  return [wx, wy, ous];
}
export function windGlobal(S) {
  const [wx, wy, ous] = crosswindDet(S, S.t);
  return [wx + ous * S.ou_x, wy + ous * S.ou_y];
}
export function idealHeading(cfg, S) {
  const [wx, wy] = crosswindDet(S, S.t);
  return Math.atan2(-wy, cfg.v0 - wx);
}
// (psi_f, dpsi_f/dt) of the LLC formation frame, built from the low-pass filtered wind estimate
export function formationHeading(cfg, S) {
  const [wx, wy] = windGlobal(S);
  const wdx = (wx - S.wf_x) / cfg.tau_wind_est, wdy = (wy - S.wf_y) / cfg.tau_wind_est;
  const X = cfg.v0 - S.wf_x, Y = -S.wf_y;
  return [Math.atan2(Y, X), (X * -wdy - Y * -wdx) / (X * X + Y * Y)];
}

// ---------------------------------------------------------------- target path (offset from R, rotated into the wake frame)
const tmp2 = new Float64Array(2);
export function targetOffset(cfg, S) {
  const [psif, dpsif] = formationHeading(cfg, S);
  const c = Math.cos(psif), s_ = Math.sin(psif);
  for (let i = 0; i < S.N; i++) {
    slotOffset(cfg, S.slot[i], tmp2);
    const sRaw = (S.t - S.path_t0[i]) / cfg.path_T;
    const s = Math.min(Math.max(sRaw, 0), 1);
    const h = s * s * (3 - 2 * s);
    const inside = sRaw > 0 && sRaw < 1;
    const hd = inside ? (6 * s * (1 - s)) / cfg.path_T : 0;
    const lane = S.path_lane[i] * cfg.swap_lane * cfg.b;
    const dx = tmp2[0] - S.path_fx[i], dy = tmp2[1] - S.path_fy[i];
    const offx = S.path_fx[i] + dx * h;
    const offy = S.path_fy[i] + dy * h + lane * Math.sin(Math.PI * s);
    const vx = dx * hd;
    const vy = dy * hd + (inside ? (lane * Math.PI * Math.cos(Math.PI * s)) / cfg.path_T : 0);
    S.ufx[i] = offx; S.ufy[i] = offy;   // unrotated, for swap path_from (py: target_offset(rotate=False))
    const orx = c * offx - s_ * offy, ory = s_ * offx + c * offy;
    S.offx[i] = orx; S.offy[i] = ory;
    S.voffx[i] = c * vx - s_ * vy + dpsif * -ory;
    S.voffy[i] = s_ * vx + c * vy + dpsif * orx;
  }
}

// ---------------------------------------------------------------- formation error (m), rigid translation removed
export function eForm(cfg, S, out) {
  const N = S.N;
  const psif = idealHeading(cfg, S);
  const c = Math.cos(psif), s_ = Math.sin(psif);
  const ex = new Float64Array(N), ey = new Float64Array(N);
  let mx = 0, my = 0;
  for (let i = 0; i < N; i++) {
    slotOffset(cfg, S.slot[i], tmp2);
    const tx = S.Rx + c * tmp2[0] - s_ * tmp2[1];
    const ty = S.Ry + s_ * tmp2[0] + c * tmp2[1];
    ex[i] = S.x[i] - tx; ey[i] = S.y[i] - ty;
    mx += ex[i]; my += ey[i];
  }
  mx /= N; my /= N;
  const o = out || new Float64Array(N);
  for (let i = 0; i < N; i++) o[i] = Math.hypot(ex[i] - mx, ey[i] - my);
  return o;
}

// ---------------------------------------------------------------- wake matrix + powers
// fills S.P, S.Pnw ; optional W (n) and wm (n*n, [i*N+j] = upwash on i from j, clipped-scaled) outputs
export function powers(cfg, S, W_out, wm_out) {
  const N = S.N;
  const floor = -0.2 * cfg.v0;
  const gm = cfg.gain * cfg.m * G;
  for (let i = 0; i < N; i++) {
    let Wsum = 0;
    const xi = S.x[i], yi = S.y[i];
    for (let j = 0; j < N; j++) {
      let w = 0;
      if (j !== i) {
        const dxw = S.x[j] - xi, dyw = S.y[j] - yi;
        const cj = Math.cos(S.psi[j]), sj = Math.sin(S.psi[j]);
        const d_x = dxw * cj + dyw * sj;
        const d_y = dxw * sj - dyw * cj;
        w = wbar(cfg, cfg.nodes, d_x / cfg.b, d_y / cfg.b, S.v[j]);
      }
      if (wm_out) wm_out[i * N + j] = w;
      Wsum += w;
    }
    const W = Math.max(Wsum, floor);
    if (W_out) W_out[i] = W;
    const pn = pNowake(cfg, S.v[i]);
    const mult = S.power_mult[i];
    S.P[i] = mult * (pn - gm * W);
    S.Pnw[i] = mult * pn;
  }
}

// ---------------------------------------------------------------- LLC
function wrap(a) {
  const t = 2 * Math.PI;
  return ((((a + Math.PI) % t) + t) % t) - Math.PI;
}
const clip = (v, lo, hi) => Math.min(Math.max(v, lo), hi);

export function llc(cfg, S) {
  const N = S.N;
  targetOffset(cfg, S);
  const [wx, wy] = windGlobal(S);
  const cap = cfg.delta_v_max * cfg.v0;
  const r = cfg.sep_radius * cfg.b;
  for (let i = 0; i < N; i++) {
    const tx = S.Rx + S.offx[i], ty = S.Ry + S.offy[i];
    let dvx = cfg.Kp * (tx - S.x[i]) + S.voffx[i];
    let dvy = cfg.Kp * (ty - S.y[i]) + S.voffy[i];
    const nrm = Math.sqrt(dvx * dvx + dvy * dvy + 1e-18);
    const k = Math.min(1, cap / nrm);
    dvx *= k; dvy *= k;
    let sx = 0, sy = 0;
    for (let j = 0; j < N; j++) {
      if (j === i) continue;
      const dx = S.x[i] - S.x[j], dy = S.y[i] - S.y[j];
      const dist = Math.sqrt(dx * dx + dy * dy);
      const mag = cfg.sep_gain * cfg.v0 * Math.max(0, (r - dist) / r);
      const tiny = dist < 1e-9;
      const ux = tiny ? 0 : dx / dist, uy = tiny ? 1 : dy / dist;
      sx += mag * ux; sy += mag * uy;
    }
    const vax = cfg.v0 + dvx + sx - wx, vay = dvy + sy - wy;
    let vcmd = Math.sqrt(vax * vax + vay * vay + 1e-18);
    const psicmd = Math.atan2(vay, vax);
    const vmax = cfg.vmax_frac * cfg.v0 * S.vmax_mult[i];
    vcmd = clip(vcmd, cfg.vmin_frac * cfg.v0, vmax);
    S.aTurn[i] = clip(cfg.k_psi * wrap(psicmd - S.psi[i]), -1, 1);
    S.aV[i] = clip(cfg.k_v * (vcmd - S.v[i]), -1, 1);
  }
}

// ---------------------------------------------------------------- one low-level step (dt)
// acc (optional): {P: Float64Array(N), save: Float64Array(N)} accumulators (sum over substeps)
// rand: {normal()} for the OU wind
export function stepLow(cfg, S, rand, acc) {
  const N = S.N, dt = cfg.dt;
  powers(cfg, S);
  const [wx, wy] = windGlobal(S);
  const Ps = cfg.P_solo;
  for (let i = 0; i < N; i++) {
    if (acc) { acc.P[i] += S.P[i]; acc.save[i] += (S.Pnw[i] - S.P[i]) / Ps; }
    const a0 = clip(S.aTurn[i], -1, 1), a1 = clip(S.aV[i], -1, 1);
    const psi2 = S.psi[i] + cfg.turn_rate_max * a0 * dt;
    const vmax = cfg.vmax_frac * cfg.v0 * S.vmax_mult[i];
    const v2 = clip(S.v[i] + cfg.vdot_max * a1 * dt, cfg.vmin_frac * cfg.v0, vmax);
    const x2 = S.x[i] + (v2 * Math.cos(psi2) + wx) * dt;
    const y2 = S.y[i] + (v2 * Math.sin(psi2) + wy) * dt;
    S.x[i] = x2; S.y[i] = y2; S.psi[i] = psi2; S.v[i] = v2;
    S.E_used[i] += S.P[i] * dt;
  }
  // exogenous: wind estimate filter uses the pre-step global wind (as in Python)
  const kf = 1 - Math.exp(-dt / cfg.tau_wind_est);
  S.wf_x += kf * (wx - S.wf_x);
  S.wf_y += kf * (wy - S.wf_y);
  const e1 = Math.exp(-dt / cfg.ou_tau), e2 = Math.sqrt(1 - Math.exp((-2 * dt) / cfg.ou_tau));
  S.ou_x = S.ou_x * e1 + e2 * (rand ? rand.normal() : 0);
  S.ou_y = S.ou_y * e1 + e2 * (rand ? rand.normal() : 0);
  S.step += 1;
  S.t = S.step * dt;
  S.Rx += cfg.v0 * dt;
}
