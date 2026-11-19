// Viewer controller: wires the DOM, runs the fixed-step sim loop at the display rate, draws.
import { GeeseSim } from '../sim/sim.js';
import { makeConfig } from '../sim/config.js';
import { parseWeightsJSON, marlPolicyFromWeights, bundledMarlPolicy } from '../policy/marl.js';
import { drawScene, drawTimeline, drawSpark, updateHeat, makeCamera, pick, birdColor } from './draw.js';

const $ = (id) => document.getElementById(id);
const cfg = makeConfig();
const state = {
  N: 9, seed: 1, speed: 4, paused: false, tool: 'free', agent: 'rule_blind', brain: 'rule', customW: null, pair: null, selected: null,
  opts: { heat: true, badges: true, labels: true },
  sim: null, shadow: null, anim: null, lastRelabel: 0, hint: '', win: [],   // win: per-decision [t, Eteam, Eshadow]
  pos: null, cam: null, heatTick: 0, heatReady: false,
};

// ------------------------------------------------------------------ world
// brain: 'rule' (hand-written, uses state.agent) | 'unilateral' | 'mutual' (bundled MARL, N=3 only) | 'custom' (loaded weights, N=3)
function makePolicy(seed) {
  if (state.brain === 'rule') return state.agent === 'stay' ? 'stay' : 'rule_blind';
  const opts = { mode: $('chk-sample').checked ? 'sample' : 'argmax', seed };
  return state.brain === 'custom' ? marlPolicyFromWeights(state.customW, cfg, opts) : bundledMarlPolicy(state.brain, cfg, opts);
}
const BRAIN_HINT = {
  rule: null,
  unilateral: '學出來的「熱山芋」：領頭者一解鎖（每 8 秒）就自己退位，三隻鳥輪流領頭。這是單方規則下的唯一均衡。',
  mutual: '學出來的「固定領頭」：退位要雙方同意，策略沒有學會輪替，領頭者一直是同一隻。',
};
function syncBrainUI(N = state.N) {
  const marl = state.brain !== 'rule';
  const ok = N === 3;
  for (const b of $('seg-brain').children) {
    b.classList.toggle('on', b.dataset.v === state.brain);
    if (b.dataset.v !== 'rule') { b.disabled = !ok; b.title = ok ? '' : `MARL 策略只在 N=3 可用（目前 N=${N}）`; }
  }
  $('brain-note').textContent = ok ? 'MARL＝強化學習訓練出的策略（3 隻鳥、GRU，權重內嵌在這個網頁、在你的瀏覽器裡推論）。'
    : `MARL 只在 N=3 可用：策略是在 3 隻鳥上訓練的（輸入維度與鄰居結構固定）。目前 N=${N}，把「鳥數 N」拉到 3 就能選。`;
  for (const b of $('seg-rule').children) { b.disabled = marl; b.classList.toggle('on', b.dataset.v === state.rule); }
  for (const b of $('seg-agent').children) b.disabled = marl;
}
function setBrain(v) {
  state.brain = v;
  if (v === 'unilateral' || v === 'mutual') state.rule = v;
  cfg.swap_rule = state.rule;
  state.sim.setPolicy(makePolicy(state.seed));
  syncBrainUI();
  setHint(BRAIN_HINT[v] || (v === 'rule' ? '規則式輪替：領頭者累積 25–35 秒後退位。' : `載入的權重（${state.customW?.meta?.name || 'custom'}）。`));
}

function newWorld(N, seed) {
  const dropped = N !== 3 && state.brain !== 'rule';
  if (dropped) state.brain = 'rule';
  cfg.swap_rule = state.rule || 'unilateral';
  const sim = new GeeseSim({ N, seed, cfgObject: cfg, policy: makePolicy(seed), avoidImmuneApex: true });
  const shadow = new GeeseSim({ N, seed, cfgObject: cfg, policy: 'stay', init: { x: sim.S.x, y: sim.S.y } });
  state.sim = sim; state.shadow = shadow; state.N = N; state.seed = seed;
  state.pair = null; state.selected = null; state.anim = null; state.lastRelabel = 0; state.win = [];
  state.heatReady = false;
  $('rng-wind').value = 0; $('o-wind').textContent = '0.0';
  buildBars();
  resize();
  syncBrainUI();
  setHint(dropped ? `N=${N}：MARL 策略只在 N=3 可用，已改回規則式大腦。` : '點鳥來操作；也可以拉「側風」或按「領頭者受傷」。');
}
const both = (f) => { f(state.sim); f(state.shadow); };

function stepBoth() {
  const { sim, shadow } = state;
  shadow.S.ou_x = sim.S.ou_x; shadow.S.ou_y = sim.S.ou_y;          // same turbulence in the fixed-leader shadow
  shadow.step();
  const rec = sim.step();
  if (rec) {
    state.win.push([sim.S.t, teamE(sim.S), teamE(shadow.S)]);
    if (state.win.length > 400) state.win.shift();
  }
  if (sim.S.relabel_n !== state.lastRelabel) onRelabel();
}
const teamE = (S) => { let e = 0; for (let i = 0; i < S.N; i++) e += S.E_used[i]; return e; };

function onRelabel() {
  const sim = state.sim, S = sim.S;
  state.lastRelabel = S.relabel_n;
  const ev = [...S.events].reverse().find((e) => e.kind === 'relabel' && e.done && e.srcUsed);
  if (!ev || !state.pos) return;
  const src = ev.srcUsed;
  // badge i flies from where the badge of bird src[i] was (screen px of the last drawn frame)
  state.anim = { t0: performance.now(), dur: 1000, src, px: Float64Array.from(state.pos.X, (x) => x + state.pos.r * 0.2), py: Float64Array.from(state.pos.Y, (y) => y - state.pos.r - 3) };
}

// ------------------------------------------------------------------ hint / bars
function setHint(t) { state.hint = t; $('hint').textContent = t; }
function buildBars() {
  const N = state.N, el = $('bars');
  el.innerHTML = '';
  for (let i = 0; i < N; i++) {
    const d = document.createElement('div');
    d.className = 'bar'; d.dataset.i = i;
    d.innerHTML = `<span class="id" style="background:${birdColor(i)}">${i + 1}</span><span class="track"><span class="fill" style="background:${birdColor(i)}"></span><span class="mean"></span></span><span class="num"></span>`;
    d.addEventListener('click', () => { state.selected = state.selected === i ? null : i; });
    el.appendChild(d);
  }
}
function updateBars() {
  const { sim } = state, S = sim.S, N = S.N;
  const er = sim.energyRel();
  let mean = 0; for (let i = 0; i < N; i++) mean += er[i]; mean /= N;
  const lo = 0.7, hi = 1.05, kids = $('bars').children;
  for (let i = 0; i < N && i < kids.length; i++) {
    const k = kids[i];
    const f = Math.min(Math.max((er[i] - lo) / (hi - lo), 0.02), 1);
    k.querySelector('.fill').style.width = (f * 100).toFixed(1) + '%';
    k.querySelector('.mean').style.left = (Math.min(Math.max((mean - lo) / (hi - lo), 0), 1) * 100).toFixed(1) + '%';
    const lead = S.slot[i] === 0;
    k.classList.toggle('lead', lead); k.classList.toggle('sel', state.selected === i);
    const tag = (lead ? '★' : '') + (S.immune[i] ? '鎖' : '') + (S.power_mult[i] > 1 ? '+' : '');
    k.querySelector('.num').innerHTML = `${tag ? `<span class="mark">${tag}</span>` : ''}${S.t > 1 ? er[i].toFixed(2) : '–'}`;
  }
}

// ------------------------------------------------------------------ panel numbers
function updatePanel() {
  const { sim, shadow, win } = state, S = sim.S;
  const pct = (x) => (x * 100).toFixed(1) + '%';
  if (S.t < 6) { $('v-s').textContent = '–'; $('v-ac').textContent = '–'; $('v-s60').textContent = '–'; }
  else {
    $('v-s').textContent = pct(sim.teamSaving());
    const ac = teamE(S) / teamE(shadow.S) - 1;
    $('v-ac').textContent = (ac >= 0 ? '+' : '') + pct(ac);
    const last = win[win.length - 1];
    let first = win[0];
    for (const w of win) if (w[0] >= last[0] - 60) { first = w; break; }
    if (last && last[0] - first[0] >= 10) $('v-s60').textContent = pct(1 - (last[1] - first[1]) / (S.N * cfg.P_solo * (last[0] - first[0])));
  }
  const hist = sim.hist;
  let sw = 0; const tw = Math.min(S.t, 120);
  for (const h of hist) if (h.t >= S.t - tw) sw += h.nSwaps;
  $('v-rate').textContent = tw >= 10 ? ((sw / tw) * 60).toFixed(1) + ' 次/分' : '–';
  let ten = 0;
  for (let k = hist.length - 1; k >= 0 && hist[k].leader === hist[hist.length - 1].leader; k--) ten += cfg.delta_h;
  $('v-tenure').textContent = hist.length ? `${ten.toFixed(0)} s（鳥 ${hist[hist.length - 1].leader + 1}）` : '–';
  // sparkline: windowed S over 30 s
  const sp = [];
  for (let k = 0; k < win.length; k++) {
    let j = k; while (j > 0 && win[k][0] - win[j][0] < 30) j--;
    if (win[k][0] - win[j][0] >= 10) sp.push(1 - (win[k][1] - win[j][1]) / (S.N * cfg.P_solo * (win[k][0] - win[j][0])));
  }
  const c = $('spark'), x = c.getContext('2d');
  drawSpark(x, c.width, c.height, sp.slice(-120), 0.0, 0.2, '#5ab0ff');
}

// ------------------------------------------------------------------ canvas sizing
const dpr = () => Math.min(window.devicePixelRatio || 1, 2);
function sizeCanvas(c) {
  const r = c.getBoundingClientRect(), d = dpr();
  const w = Math.max(100, Math.round(r.width)), h = Math.max(80, Math.round(r.height));
  if (c.width !== Math.round(w * d) || c.height !== Math.round(h * d)) { c.width = Math.round(w * d); c.height = Math.round(h * d); }
  return { w, h, d };
}
let sceneSize = { w: 800, h: 480, d: 1 };
function resize() {
  sceneSize = sizeCanvas($('scene'));
  state.cam = makeCamera(cfg, state.N, sceneSize.w, sceneSize.h);
  state.heatReady = false;
  sizeCanvas($('timeline'));
}
window.addEventListener('resize', resize);

// ------------------------------------------------------------------ interaction
function onPick(i) {
  const { sim, shadow } = state;
  if (i < 0) { state.selected = null; state.pair = null; return; }
  state.selected = i;
  const S = sim.S;
  if (state.tool === 'free') {
    const on = !(S.immune[i] || sim.pending.has(i));
    const res = sim.requestImmune(i, on);
    setHint(res === 'pending' ? `鳥 ${i + 1} 正在領頭：先讓牠退位，退出頂點後才會變成搭便車者（不領頭、不換位）。`
      : res === 'immediate' ? `鳥 ${i + 1} 變成搭便車者：只收尾流、不再被換位。再點一次可解除。`
      : res === 'off' ? `鳥 ${i + 1} 恢復正常輪替。` : '');
  } else if (state.tool === 'injure') {
    const inj = S.power_mult[i] > 1;
    both((s) => s.injure(i, !inj));
    setHint(inj ? `鳥 ${i + 1} 痊癒。` : `鳥 ${i + 1} 受傷：功率 ×1.5、最大速度 ×0.85。`);
  } else if (state.tool === 'pair') {
    if (state.pair == null) { state.pair = i; setHint(`已選鳥 ${i + 1}，再點另一隻鳥來對調兩者的身分碼。`); }
    else if (state.pair === i) { state.pair = null; setHint('取消對調。'); }
    else {
      const a = state.pair; state.pair = null;
      sim.relabel({ perm: sim.swapPairPerm(a, i), resetMem: $('chk-resetmem').checked });
      setHint(`鳥 ${a + 1} 與鳥 ${i + 1} 的碼對調了：身體沒變，碼換了。`);
    }
  }
}
$('scene').addEventListener('pointerdown', (e) => {
  const r = e.currentTarget.getBoundingClientRect();
  if (!state.pos) return;
  const rad = e.pointerType === 'touch' ? 34 : 24;
  onPick(pick(state.pos, e.clientX - r.left, e.clientY - r.top, rad));
});

function seg(id, fn) {
  const el = $(id);
  el.addEventListener('click', (e) => {
    const b = e.target.closest('button'); if (!b || b.disabled) return;
    [...el.children].forEach((c) => c.classList.toggle('on', c === b));
    fn(b.dataset.v);
  });
}
const TOOL_HELP = {
  free: '點一隻鳥讓牠變成搭便車者（永不領頭、不被換位）。領頭者會先退位。再點一次解除。',
  injure: '點一隻鳥讓牠受傷（功率 ×1.5、最大速度 ×0.85）；再點一次痊癒。',
  pair: '依序點兩隻鳥，對調牠們攜帶的身分碼（身體不動）。',
};
seg('seg-speed', (v) => { state.speed = +v; });
seg('seg-tool', (v) => { state.tool = v; state.pair = null; $('tool-help').textContent = TOOL_HELP[v]; });
seg('seg-brain', (v) => setBrain(v));
$('chk-sample').addEventListener('change', () => { if (state.brain !== 'rule') state.sim.setPolicy(makePolicy(state.seed)); });
seg('seg-agent', (v) => { state.agent = v; both((s) => s === state.shadow || s.setPolicy(v)); setHint(v === 'stay' ? '不換位：領頭者永遠領頭，這是對照組。' : '規則式輪替：領頭者累積 25–35 秒後退位。'); });
seg('seg-rule', (v) => { state.rule = v; cfg.swap_rule = v; state.sim.setPolicy(state.agent === 'stay' ? 'stay' : 'rule_blind'); setHint(v === 'mutual' ? '雙方同意：第二、三位會主動上前，退位要雙方握手。' : '單方：領頭者想退位就能換。'); });
$('tool-help').textContent = TOOL_HELP.free;
$('btn-play').addEventListener('click', () => { state.paused = !state.paused; $('btn-play').textContent = state.paused ? '繼續' : '暫停'; });
$('btn-reset').addEventListener('click', () => newWorld(state.N, (state.seed * 7919 + 13) % 100000));
$('rng-n').addEventListener('input', (e) => { $('o-n').textContent = e.target.value; syncBrainUI(+e.target.value); });
$('rng-n').addEventListener('change', (e) => newWorld(+e.target.value, state.seed));
function applyWind() {
  const sp = +$('rng-wind').value, dir = +$('rng-dir').value;
  $('o-wind').textContent = sp.toFixed(1); $('o-dir').textContent = dir + '°';
}
$('rng-wind').addEventListener('input', applyWind);
$('rng-dir').addEventListener('input', applyWind);
function commitWind() {
  const sp = +$('rng-wind').value, dir = +$('rng-dir').value;
  if (sp <= 0) { both((s) => s.crosswindOff()); setHint('風停了：隊形要幾秒才恢復。'); }
  else { both((s) => s.crosswindOn(sp, dir, 0.5)); setHint(`側風 ${sp.toFixed(1)} m/s：看熱圖和隊形怎麼歪、再怎麼重組。`); }
}
$('rng-wind').addEventListener('change', commitWind);
$('rng-dir').addEventListener('change', () => { if (+$('rng-wind').value > 0) commitWind(); });
$('btn-hurt-lead').addEventListener('click', () => {
  const lead = state.sim.S.slot.indexOf(0);
  both((s) => s.injure(lead, true));
  setHint(`領頭者（鳥 ${lead + 1}）受傷：功率 ×1.5、最大速度 ×0.85。`);
});
$('btn-free-lead').addEventListener('click', () => {
  const lead = state.sim.S.slot.indexOf(0);
  onPickFree(lead);
});
function onPickFree(i) { const t = state.tool; state.tool = 'free'; onPick(i); state.tool = t; }
$('btn-relabel-all').addEventListener('click', () => {
  state.sim.relabel({ resetMem: $('chk-resetmem').checked });
  $('chk-badge').checked = true; state.opts.badges = true;
  setHint('全員換碼：每隻鳥現在攜帶另一隻鳥的碼（循環移位，沒有鳥保留自己的碼）。');
});
$('btn-sham').addEventListener('click', () => {
  state.sim.relabel({ sham: true, resetMem: $('chk-resetmem').checked });
  setHint('sham：走一遍換碼程序，但碼沒有變（對照組）。');
});
for (const [id, k] of [['chk-heat', 'heat'], ['chk-badge', 'badges'], ['chk-label', 'labels']]) {
  $(id).addEventListener('change', (e) => { state.opts[k] = e.target.checked; });
}
$('file-w').addEventListener('change', async (e) => {
  const f = e.target.files[0]; if (!f) return;
  try {
    const obj = JSON.parse(await f.text());
    const v = parseWeightsJSON(obj);
    if (!v.ok) throw new Error(v.errors.join('；'));
    marlPolicyFromWeights(obj, cfg);                       // throws if the arrays do not match
    if (state.N !== 3) throw new Error(`權重格式正確，但 MARL 只在 N=3 可用（目前 N=${state.N}）`);
    state.customW = obj;
    if (obj.meta.rule === 'unilateral' || obj.meta.rule === 'mutual') state.rule = obj.meta.rule;
    setBrain('custom');
    $('w-msg').textContent = `已載入「${obj.meta.name || 'custom'}」（obs_dim=${obj.meta.obs_dim}），正在使用。`;
  } catch (err) { $('w-msg').textContent = '權重檔無效：' + err.message; }
  e.target.value = '';
});

// ------------------------------------------------------------------ main loop
let last = performance.now(), acc = 0, frame = 0;
function loop(now) {
  const dtReal = Math.min((now - last) / 1000, 0.1); last = now;
  if (!state.paused) {
    acc += dtReal * state.speed;
    let n = 0;
    while (acc >= cfg.dt && n < 400) { stepBoth(); acc -= cfg.dt; n++; }
    if (n >= 400) acc = 0;
  }
  const alpha = state.paused ? 1 : Math.min(acc / cfg.dt, 1);
  const ctx = $('scene').getContext('2d');
  ctx.setTransform(sceneSize.d, 0, 0, sceneSize.d, 0, 0);
  if (state.opts.heat && (frame % 4 === 0 || !state.heatReady)) { updateHeat(state.sim, state.cam); state.heatReady = true; }
  state.pos = drawScene(ctx, {
    sim: state.sim, cam: state.cam, alpha, opts: state.opts, selected: state.selected, pair: state.pair,
    anim: state.anim, now, heatReady: state.heatReady, heatCanvas: heatCanvasRef(),
  });
  if (frame % 6 === 0) {
    updateBars(); updatePanel();
    const tl = $('timeline'), s = sizeCanvas(tl), tctx = tl.getContext('2d');
    tctx.setTransform(s.d, 0, 0, s.d, 0, 0);
    drawTimeline(tctx, s.w, s.h, state.sim);
  }
  frame++;
  requestAnimationFrame(loop);
}
import { getHeatCanvas } from './draw.js';
const heatCanvasRef = getHeatCanvas;

state.rule = 'unilateral';
newWorld(9, 1);
requestAnimationFrame(loop);
// handy for debugging / headless checks
window.__geese = state;
