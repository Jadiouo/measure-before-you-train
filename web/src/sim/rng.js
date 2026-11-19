// Seeded PRNG (mulberry32). JS randomness is NOT bit-identical to JAX (spec v0.2 1.4); only used for the
// interactive app (initial noise, OU wind, T_hold draws, random targets). Consistency tests inject JAX draws.
export function mulberry32(seed) {
  let a = seed >>> 0;
  return function () {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export function makeNormal(rand) {
  return function () {
    let u = 0;
    while (u === 0) u = rand();
    const v = rand();
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
  };
}

// stateless hash -> U[0,1): used for T_hold draws keyed by (seed, bird, count), like fold_in in the JAX version
export function hashUniform(seed, i, c) {
  let h = (seed >>> 0) ^ 0x9e3779b9;
  h = Math.imul(h ^ (i + 0x7f4a7c15), 0x85ebca6b);
  h = Math.imul(h ^ (h >>> 13) ^ (c + 0x165667b1), 0xc2b2ae35);
  h ^= h >>> 16;
  return mulberry32(h)();
}
