// State container + slot geometry (state.py). Everything is a plain object of Float64Array / Int32Array / Uint8Array.
// All N birds are alive (leave/join events are out of M1b scope), so dead-bird masking is dropped.
export const BIG = 1e12;

export function slotOffset(cfg, k, out, o = 0) {
  const kk = Math.max(k, 0);
  const a = Math.floor((kk + 1) / 2);
  const side = kk % 2 === 1 ? 1 : -1;
  out[o] = -cfg.slot_dx * cfg.b * a;
  out[o + 1] = kk > 0 ? side * cfg.slot_dy * cfg.b * a : 0;
}

export function armSign(k) { return k === 0 ? 0 : (k % 2 === 1 ? 1 : -1); }

export function createState(cfg, N, init) {
  // init: {x,y,psi,v,slot} arrays of length N (explicit initial state)
  const f = () => new Float64Array(N);
  const S = {
    N, t: 0, step: 0, dec: 0, Rx: 0, Ry: 0,
    x: Float64Array.from(init.x), y: Float64Array.from(init.y),
    psi: Float64Array.from(init.psi || new Float64Array(N)),
    v: Float64Array.from(init.v || new Float64Array(N).fill(cfg.v0)),
    slot: Int32Array.from(init.slot || Array.from({ length: N }, (_, i) => i)),
    power_mult: new Float64Array(N).fill(1), vmax_mult: new Float64Array(N).fill(1),
    E_used: f(),
    lock: new Int32Array(N),
    P_prev: new Float64Array(N).fill(1), last_action: new Int32Array(N), blocked: new Uint8Array(N),   // policy-visible (obs.py)
    path_t0: new Float64Array(N).fill(-BIG), path_fx: f(), path_fy: f(), path_lane: f(),
    apex_ctr: 0,
    immune: new Uint8Array(N),
    id_owner: Int32Array.from({ length: N }, (_, i) => i),
    id_code: Int32Array.from({ length: N }, (_, i) => i),  // id of the original owner whose code the bird carries (== id_owner; kept for naming)
    relabel_n: 0, mem_reset: false,
    ou_x: 0, ou_y: 0, wf_x: 0, wf_y: 0,
    events: [],
    // scratch
    ufx: f(), ufy: f(),  // formation-frame (unrotated) offset: what path_fx/path_fy must store
    offx: f(), offy: f(), voffx: f(), voffy: f(),
    aTurn: f(), aV: f(), P: f(), Pnw: f(),
  };
  return S;
}

// bird in each slot (-1 none)
export function occupancy(S, out) {
  const occ = out || new Int32Array(S.N);
  occ.fill(-1);
  for (let i = 0; i < S.N; i++) occ[S.slot[i]] = i;
  return occ;
}
