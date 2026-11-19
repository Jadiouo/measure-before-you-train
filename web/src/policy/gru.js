// Actor network of the M1.5 IPPO policies: tanh(obs W1 + b1) -> GRU(H) -> linear policy head (port of ippo.net_step, memory="gru").
// Weights are row-major float32 (x @ W): w1 (D,H), wi/wh (H,3H) with gate order r,z,n, wpi (H,3).  Compute in float64.
export function decodeB64F32(b64) {
  const bin = atob(b64);
  const buf = new ArrayBuffer(bin.length);
  const u8 = new Uint8Array(buf);
  for (let i = 0; i < bin.length; i++) u8[i] = bin.charCodeAt(i);
  const dv = new DataView(buf), out = new Float64Array(bin.length / 4);
  for (let i = 0; i < out.length; i++) out[i] = dv.getFloat32(4 * i, true);
  return out;
}

export class GruActor {
  /** w: {w1,b1,wi,wh,bi,bh,wpi,bpi} Float64Array each; D obs dim, H hidden. */
  constructor(w, D, H) {
    this.w = w; this.D = D; this.H = H;
    this.x = new Float64Array(H); this.gi = new Float64Array(3 * H); this.gh = new Float64Array(3 * H);
  }
  /** one decision: h (H, updated in place), obs (D, at offset o) -> writes logits (3) */
  step(h, obs, o, logits) {
    const { w1, b1, wi, wh, bi, bh, wpi, bpi } = this.w;
    const { D, H, x, gi, gh } = this;
    for (let j = 0; j < H; j++) x[j] = b1[j];
    for (let d = 0; d < D; d++) { const v = obs[o + d]; if (v !== 0) for (let j = 0; j < H; j++) x[j] += v * w1[d * H + j]; }
    for (let j = 0; j < H; j++) x[j] = Math.tanh(x[j]);
    for (let j = 0; j < 3 * H; j++) { gi[j] = bi[j]; gh[j] = bh[j]; }
    for (let d = 0; d < H; d++) {
      const xv = x[d], hv = h[d], r0 = d * 3 * H;
      for (let j = 0; j < 3 * H; j++) { gi[j] += xv * wi[r0 + j]; gh[j] += hv * wh[r0 + j]; }
    }
    for (let j = 0; j < H; j++) {
      const r = 1 / (1 + Math.exp(-(gi[j] + gh[j])));
      const z = 1 / (1 + Math.exp(-(gi[H + j] + gh[H + j])));
      const n = Math.tanh(gi[2 * H + j] + r * gh[2 * H + j]);
      x[j] = (1 - z) * n + z * h[j];                // reuse x as h'
    }
    h.set(x);
    for (let a = 0; a < 3; a++) {
      let s = bpi[a];
      for (let j = 0; j < H; j++) s += h[j] * wpi[j * 3 + a];
      logits[a] = s;
    }
  }
}

/** masked softmax (policy_probs: illegal logits -> -1e9) */
export function maskedProbs(logits, mask, mo, out) {
  let m = -Infinity;
  for (let a = 0; a < 3; a++) { const l = mask[mo + a] ? logits[a] : -1e9; out[a] = l; if (l > m) m = l; }
  let s = 0;
  for (let a = 0; a < 3; a++) { out[a] = Math.exp(out[a] - m); s += out[a]; }
  for (let a = 0; a < 3; a++) out[a] /= s;
  return out;
}
