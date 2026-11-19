// node bench/bench_step.mjs : cost of one low-level step (dt = 0.1 s sim time), whole decisions included.
import { GeeseSim } from '../src/sim/sim.js';
import { performance } from 'node:perf_hooks';

for (const N of [3, 9, 14]) {
  const sim = new GeeseSim({ N, seed: 1 });
  sim.crosswindOn(3, 90, 0.5);
  for (let k = 0; k < 2000; k++) sim.step();               // warm-up (JIT)
  const reps = 5, n = 6000, ts = [];
  for (let r = 0; r < reps; r++) {
    const t0 = performance.now();
    for (let k = 0; k < n; k++) sim.step();
    ts.push(((performance.now() - t0) / n) * 1000);          // microseconds per low step
  }
  ts.sort((a, b) => a - b);
  const med = ts[Math.floor(reps / 2)];
  // a 60 fps frame at real-time speed advances 1/60 s = 1/6 low step; at 8x speed 1.33 steps; budget 16.7 ms
  console.log(`N=${String(N).padStart(2)}  ${med.toFixed(1)} us/low-step (median of ${reps}x${n}, min ${ts[0].toFixed(1)}, max ${ts[reps - 1].toFixed(1)})` +
    `  -> max low steps per 16.7 ms frame at 50% budget: ${Math.floor(8333 / med)}  (= ${(Math.floor(8333 / med) * 6).toFixed(0)}x real time at 60 fps)`);
}
