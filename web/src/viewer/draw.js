// Canvas drawing for the viewer (no physics here). World frame: x forward (right on screen), y left (up on screen).
import { wbar, quadNodes, G } from '../sim/wake.js';
import { slotOffset } from '../sim/state.js';

export const birdColor = (i, light = 62) => `hsl(${(i * 137.508 + 20) % 360} 72% ${light}%)`;

const tmp = new Float64Array(2);

// ------------------------------------------------------------------ camera
export function makeCamera(cfg, N, w, h) {
  const m = Math.max(N, 7);
  const arms = Math.ceil((m - 1) / 2);
  const ex = cfg.slot_dx * cfg.b * arms + 8 * cfg.b;                // metres to show along x
  const ey = 2 * cfg.slot_dy * cfg.b * arms + 9 * cfg.b;
  const ppm = Math.min(w / ex, h / ey);
  // centre on the middle of the ideal V
  let cx = 0;
  for (let k = 0; k < N; k++) { slotOffset(cfg, k, tmp); cx += tmp[0]; }
  cx /= N;
  return { ppm, cx, w, h };
}
const sx = (cam, wx, Rx) => cam.w / 2 + (wx - Rx - cam.cx) * cam.ppm;
const sy = (cam, wy) => cam.h / 2 - wy * cam.ppm;

// ------------------------------------------------------------------ wake heat map (upwash on a hypothetical follower)
let heatCanvas = null, heatCtx = null, heatNodes = null, heatCfg = null;
export const getHeatCanvas = () => heatCanvas;
export function updateHeat(sim, cam, cellPx = 11) {
  const cfg = sim.cfg, S = sim.S, N = S.N;
  if (!heatCanvas) { heatCanvas = document.createElement('canvas'); heatCtx = heatCanvas.getContext('2d'); }
  if (!heatCfg || heatCfg.b !== cfg.b) { heatNodes = quadNodes(8, cfg.b); heatCfg = cfg; }   // 8 nodes: visual only
  const gw = Math.max(8, Math.ceil(cam.w / cellPx)), gh = Math.max(8, Math.ceil(cam.h / cellPx));
  if (heatCanvas.width !== gw || heatCanvas.height !== gh) { heatCanvas.width = gw; heatCanvas.height = gh; }
  const img = heatCtx.createImageData(gw, gh);
  const gm = (cfg.gain * cfg.m * G) / cfg.P_solo, floor = -0.2 * cfg.v0;
  const cs = new Float64Array(N), sn = new Float64Array(N);
  for (let j = 0; j < N; j++) { cs[j] = Math.cos(S.psi[j]); sn[j] = Math.sin(S.psi[j]); }
  const Rx = S.Rx;
  for (let gy = 0; gy < gh; gy++) {
    const wy = ((cam.h / 2 - ((gy + 0.5) * cam.h) / gh) / cam.ppm);
    for (let gx = 0; gx < gw; gx++) {
      const wx = Rx + cam.cx + (((gx + 0.5) * cam.w) / gw - cam.w / 2) / cam.ppm;
      let W = 0;
      for (let j = 0; j < N; j++) {
        const dxw = S.x[j] - wx, dyw = S.y[j] - wy;
        const d_x = dxw * cs[j] + dyw * sn[j];
        if (d_x < 0.5 * cfg.b || d_x > 12 * cfg.b) continue;
        const d_y = dxw * sn[j] - dyw * cs[j];
        W += wbar(cfg, heatNodes, d_x / cfg.b, d_y / cfg.b, S.v[j]);
      }
      W = Math.max(W, floor);
      const s = gm * W;                          // fractional power saving a follower would get here
      const a = Math.min(Math.abs(s) / 0.15, 1);
      const o = (gy * gw + gx) * 4;
      if (s >= 0) { img.data[o] = 60; img.data[o + 1] = 230; img.data[o + 2] = 170; }
      else { img.data[o] = 255; img.data[o + 1] = 90; img.data[o + 2] = 160; }
      img.data[o + 3] = Math.round(125 * Math.pow(a, 0.85));
    }
  }
  heatCtx.putImageData(img, 0, 0);
}

// ------------------------------------------------------------------ bird glyphs
function chevron(ctx, x, y, ang, r) {
  ctx.save();
  ctx.translate(x, y);
  ctx.rotate(ang);
  ctx.beginPath();
  ctx.moveTo(r, 0);
  ctx.lineTo(-r * 0.75, r * 0.72);
  ctx.lineTo(-r * 0.3, 0);
  ctx.lineTo(-r * 0.75, -r * 0.72);
  ctx.closePath();
  ctx.restore();
}
export function drawLock(ctx, x, y, s, color) {
  ctx.save();
  ctx.strokeStyle = color; ctx.fillStyle = color; ctx.lineWidth = Math.max(1.3, s * 0.18);
  ctx.beginPath(); ctx.arc(x, y - s * 0.35, s * 0.32, Math.PI, 0); ctx.stroke();
  ctx.fillRect(x - s * 0.45, y - s * 0.35, s * 0.9, s * 0.7);
  ctx.restore();
}

// ------------------------------------------------------------------ scene
// view: {sim, alpha, cam, opts:{heat,badges,labels}, selected, pair, anim, now}
export function drawScene(ctx, v) {
  const { sim, cam, alpha, opts } = v;
  const S = sim.S, cfg = sim.cfg, N = S.N, P = sim.prev;
  ctx.clearRect(0, 0, cam.w, cam.h);
  const g = ctx.createLinearGradient(0, 0, 0, cam.h);
  g.addColorStop(0, '#0f1d36'); g.addColorStop(1, '#16294a');
  ctx.fillStyle = g; ctx.fillRect(0, 0, cam.w, cam.h);
  const Rx = P.Rx + (S.Rx - P.Rx) * alpha;

  // ground reference dots (world fixed): show forward motion and wind drift
  ctx.fillStyle = 'rgba(190,210,255,0.16)';
  const sp = 8 * cfg.b;
  const x0 = Math.floor((Rx + cam.cx - cam.w / 2 / cam.ppm) / sp) * sp;
  for (let wx = x0; wx < Rx + cam.cx + cam.w / 2 / cam.ppm + sp; wx += sp) {
    for (let wy = -10 * cfg.b; wy <= 10 * cfg.b; wy += sp) {
      ctx.beginPath(); ctx.arc(sx(cam, wx, Rx), sy(cam, wy), 1.6, 0, 6.2832); ctx.fill();
    }
  }
  if (opts.heat && v.heatReady) {
    ctx.imageSmoothingEnabled = true;
    ctx.drawImage(v.heatCanvas, 0, 0, cam.w, cam.h);
  }

  // interpolated body state
  const X = new Float64Array(N), Y = new Float64Array(N), A = new Float64Array(N);
  for (let i = 0; i < N; i++) {
    X[i] = sx(cam, P.x[i] + (S.x[i] - P.x[i]) * alpha, Rx);
    Y[i] = sy(cam, P.y[i] + (S.y[i] - P.y[i]) * alpha);
    A[i] = -(P.psi[i] + (S.psi[i] - P.psi[i]) * alpha);
  }
  const r = Math.max(9, Math.min(17, cfg.b * cam.ppm * 0.62));
  const lead = S.slot.indexOf(0);

  // faint chain lines to the slot in front (shows the V)
  ctx.strokeStyle = 'rgba(180,200,255,0.12)'; ctx.lineWidth = 1;
  const occ = new Int32Array(N); for (let i = 0; i < N; i++) occ[S.slot[i]] = i;
  for (let k = 1; k < N; k++) {
    const f = k <= 2 ? 0 : k - 2;
    ctx.beginPath(); ctx.moveTo(X[occ[k]], Y[occ[k]]); ctx.lineTo(X[occ[f]], Y[occ[f]]); ctx.stroke();
  }

  for (let i = 0; i < N; i++) {
    const col = birdColor(i);
    const imm = S.immune[i] === 1, pend = sim.pending.has(i), inj = S.power_mult[i] > 1;
    if (i === lead) {                                       // leader halo
      ctx.beginPath(); ctx.arc(X[i], Y[i], r + 7, 0, 6.2832);
      ctx.strokeStyle = '#ffd24a'; ctx.lineWidth = 2.5; ctx.stroke();
      ctx.fillStyle = 'rgba(255,210,74,0.14)'; ctx.fill();
    }
    if (v.selected === i || v.pair === i) {
      ctx.beginPath(); ctx.arc(X[i], Y[i], r + 12, 0, 6.2832);
      ctx.strokeStyle = '#ffffff'; ctx.lineWidth = 1.5; ctx.setLineDash([4, 3]); ctx.stroke(); ctx.setLineDash([]);
    }
    chevron(ctx, X[i], Y[i], A[i], r);
    if (imm) { ctx.fillStyle = 'rgba(15,29,54,0.9)'; ctx.fill(); ctx.strokeStyle = col; ctx.lineWidth = 2.4; ctx.stroke(); }
    else { ctx.fillStyle = col; ctx.fill(); ctx.strokeStyle = 'rgba(0,0,0,0.45)'; ctx.lineWidth = 1; ctx.stroke(); }
    if (pend) { ctx.beginPath(); ctx.arc(X[i], Y[i], r + 4, 0, 6.2832); ctx.strokeStyle = '#ffd24a'; ctx.setLineDash([2, 3]); ctx.lineWidth = 1.5; ctx.stroke(); ctx.setLineDash([]); }
    if (imm) drawLock(ctx, X[i], Y[i] + r + 9, 9, '#ffd24a');
    if (inj) {                                              // red cross = injured
      const cx = X[i] + r * 0.9, cy = Y[i] - r * 0.9;
      ctx.strokeStyle = '#ff4d5e'; ctx.lineWidth = 2.4;
      ctx.beginPath(); ctx.moveTo(cx - 4, cy); ctx.lineTo(cx + 4, cy); ctx.moveTo(cx, cy - 4); ctx.lineTo(cx, cy + 4); ctx.stroke();
    }
    if (opts.labels) {
      ctx.fillStyle = '#fff'; ctx.font = '600 11px system-ui, sans-serif'; ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
      ctx.fillText(String(i + 1), X[i] - r * 0.1 - 1, Y[i] + r + (imm ? 21 : 13));
    }
  }
  if (lead >= 0) {
    ctx.fillStyle = '#ffd24a'; ctx.font = '700 11px system-ui, sans-serif'; ctx.textAlign = 'center'; ctx.textBaseline = 'alphabetic';
    ctx.fillText('領頭', X[lead], Y[lead] - r - 11);
  }

  // identity-code badges: body colour = body, badge colour = code owner ("body unchanged, code swapped")
  if (opts.badges) {
    const br = 4.2;
    const anim = v.anim;
    for (let i = 0; i < N; i++) {
      let bx = X[i] + r * 0.2, by = Y[i] - r - 3, owner = S.id_owner[i];
      if (anim && v.now - anim.t0 < anim.dur) {
        const p = Math.min((v.now - anim.t0) / anim.dur, 1), e = p * p * (3 - 2 * p);
        const from = anim.src[i];
        const fx = anim.px[from], fy = anim.py[from];
        bx = fx + (bx - fx) * e; by = fy + (by - fy) * e - Math.sin(Math.PI * e) * 18;
      }
      ctx.beginPath(); ctx.arc(bx, by, br, 0, 6.2832);
      ctx.fillStyle = birdColor(owner, 58); ctx.fill();
      ctx.strokeStyle = '#fff'; ctx.lineWidth = 1.2; ctx.stroke();
    }
  }

  // wind arrow (top-right)
  const [wx_, wy_] = [S.wf_x, S.wf_y];
  const sp2 = Math.hypot(wx_, wy_);
  if (sp2 > 0.15) {
    const ax = cam.w - 46, ay = 42, len = 12 + sp2 * 6, ang = Math.atan2(-wy_, wx_);
    ctx.save(); ctx.translate(ax, ay); ctx.rotate(ang);
    ctx.strokeStyle = '#7fc4ff'; ctx.fillStyle = '#7fc4ff'; ctx.lineWidth = 3;
    ctx.beginPath(); ctx.moveTo(-len / 2, 0); ctx.lineTo(len / 2, 0); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(len / 2 + 6, 0); ctx.lineTo(len / 2 - 4, -5); ctx.lineTo(len / 2 - 4, 5); ctx.closePath(); ctx.fill();
    ctx.restore();
    ctx.fillStyle = '#9fd0ff'; ctx.font = '600 11px system-ui, sans-serif'; ctx.textAlign = 'center';
    ctx.fillText(`風 ${sp2.toFixed(1)} m/s`, ax, ay + 24);
  }
  return { X, Y, r };
}

// pick nearest bird (css px); returns -1 if none within the radius
export function pick(pos, px, py, radius) {
  let best = -1, bd = radius * radius;
  for (let i = 0; i < pos.X.length; i++) {
    const d = (pos.X[i] - px) ** 2 + (pos.Y[i] - py) ** 2;
    if (d < bd) { bd = d; best = i; }
  }
  return best;
}

// ------------------------------------------------------------------ rotation timeline (Gantt: one row per bird, bar while leading)
const EVT_COLOR = { crosswind: '#7fc4ff', injury: '#ff4d5e', immune: '#ffd24a', relabel: '#c58bff' };
export function drawTimeline(ctx, w, h, sim, windowSec = 240) {
  const N = sim.S.N, dh = sim.cfg.delta_h;
  ctx.clearRect(0, 0, w, h);
  ctx.fillStyle = '#101b30'; ctx.fillRect(0, 0, w, h);
  const left = 24, top = 16, bot = 14, rowH = (h - top - bot) / N;
  const tNow = sim.S.t, t0 = Math.max(0, tNow - windowSec), span = Math.max(windowSec, 1);
  const X = (t) => left + ((t - (tNow - span)) / span) * (w - left - 6);
  ctx.font = '10px system-ui, sans-serif'; ctx.textBaseline = 'middle'; ctx.textAlign = 'right';
  for (let i = 0; i < N; i++) {
    const y = top + i * rowH;
    ctx.fillStyle = i % 2 ? 'rgba(255,255,255,0.025)' : 'rgba(255,255,255,0.0)'; ctx.fillRect(left, y, w - left, rowH);
    ctx.fillStyle = birdColor(i); ctx.fillText(String(i + 1), left - 5, y + rowH / 2);
  }
  for (const hrec of sim.hist) {
    if (hrec.t + dh < t0) continue;
    const i = hrec.leader;
    ctx.fillStyle = birdColor(i);
    const xa = Math.max(left, X(hrec.t)), xb = X(hrec.t + dh);
    ctx.fillRect(xa, top + i * rowH + 1, Math.max(xb - xa, 1), Math.max(rowH - 2, 2));
  }
  // events
  ctx.textAlign = 'center'; ctx.textBaseline = 'alphabetic'; ctx.font = '600 10px system-ui, sans-serif';
  for (const m of sim.marks) {
    if (m.t < t0 || m.t > tNow + 0.01) continue;
    const x = X(m.t);
    const c = EVT_COLOR[m.kind] || '#fff';
    ctx.strokeStyle = c; ctx.lineWidth = 1.2; ctx.setLineDash([3, 3]);
    ctx.beginPath(); ctx.moveTo(x, top - 2); ctx.lineTo(x, h - bot); ctx.stroke(); ctx.setLineDash([]);
    ctx.fillStyle = c;
    const lab = { crosswind: m.label === 'wind off' ? '風止' : '風', injury: m.label === 'heal' ? '癒' : '傷', immune: m.on === false ? '解' : '搭', relabel: m.label === 'sham' ? 'sham' : '碼' }[m.kind] || '?';
    ctx.fillText(lab, x, top - 5);
  }
  // time axis
  ctx.fillStyle = '#8fa3c7'; ctx.textAlign = 'left'; ctx.textBaseline = 'alphabetic';
  ctx.fillText(`${Math.max(0, Math.round(tNow - span))}s`, left, h - 3);
  ctx.textAlign = 'right'; ctx.fillText(`${Math.round(tNow)}s`, w - 6, h - 3);
}

export function drawSpark(ctx, w, h, vals, lo, hi, color) {
  ctx.clearRect(0, 0, w, h);
  if (vals.length < 2) return;
  ctx.strokeStyle = color; ctx.lineWidth = 1.6; ctx.beginPath();
  for (let k = 0; k < vals.length; k++) {
    const x = (k / (vals.length - 1)) * (w - 2) + 1;
    const y = h - 2 - ((Math.min(Math.max(vals[k], lo), hi) - lo) / (hi - lo)) * (h - 4);
    k ? ctx.lineTo(x, y) : ctx.moveTo(x, y);
  }
  ctx.stroke();
}
