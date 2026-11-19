// Wake (upwash) + power model. Port of scenarios/geese/wake.py (term by term).
export const G = 9.81;

export function quadNodes(nq, b) {
  const sk = new Float64Array(nq), om = new Float64Array(nq);
  let s = 0;
  for (let k = 0; k < nq; k++) {
    const th = ((k + 0.5) * Math.PI) / nq;
    sk[k] = -(b / 2) * Math.cos(th);
    om[k] = Math.sin(th) ** 2;
    s += om[k];
  }
  for (let k = 0; k < nq; k++) om[k] /= s;
  return { sk, om };
}

// mean upwash (m/s) d_x behind / d_y left of a leader (units of b), leader airspeed v. gain NOT applied.
// c: {m,b,rho,x_d,r_c0,k_c,d_min,d_max}; nodes from quadNodes. Window (d_min,d_max) exclusive, 0 outside.
export function wbar(c, nodes, dxb, dyb, v) {
  if (!(dxb > c.d_min && dxb < c.d_max)) return 0;
  const { m, b, rho } = c;
  const dx = dxb * b;
  const dy = dyb * b;
  const b0 = (Math.PI / 4) * b;
  const G0 = (m * G) / (rho * v * b0);
  const Gam = G0 * Math.exp(-dx / (c.x_d * b));
  const rc = c.r_c0 * b + c.k_c * dx;
  const rc2 = rc * rc;
  const k = Gam / (2 * Math.PI);
  const { sk, om } = nodes;
  let out = 0;
  for (let q = 0; q < sk.length; q++) {
    const y = dy + sk[q];
    const y1 = y - b0 / 2, y2 = y + b0 / 2;
    // s=+1: yy = y - b0/2 ; s=-1: yy = y + b0/2
    const wv = k * (y1 / (y1 * y1 + rc2)) - k * (y2 / (y2 * y2 + rc2));
    out += om[q] * wv;
  }
  return out;
}

export function pInd(c, v) { return (2 * (c.m * G) ** 2) / (Math.PI * c.rho * v * c.b ** 2); }
export function pPar(c, v) { return 0.5 * c.rho * v ** 3 * c.SbCD; }
export function pNowake(c, v) { return pInd(c, v) + pPar(c, v) + c.P_pro; }

// gain such that the max saving over d_y at d_x = dxref equals `target` (PoC calibrate()); 601-point d_y grid on [0,3]
export function calibrate(c, target = 0.125, dxref = 2.0) {
  const nodes = quadNodes(c.nq, c.b);
  const Ps = pNowake(c, c.v0);
  let best = -Infinity;
  for (let k = 0; k <= 600; k++) {
    const dy = (3 * k) / 600;
    const W = Math.max(wbar(c, nodes, dxref, dy, c.v0), -0.2 * c.v0);
    const s = (c.m * G * W) / Ps;
    if (s > best) best = s;
  }
  return target / best;
}
