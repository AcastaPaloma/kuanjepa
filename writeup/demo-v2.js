// The v2 Results tab's two demos. Nothing here runs a model: V-JEPA 2 and EmbeddingGemma are far too big for a
// browser, so scripts/export_v2_results.py recorded every answer on my Mac and this file plays them back.
//   Reveal      data/reveal.json: for 10 photos, the guess after each of 64 tiles, masked before and after the encoder
//   Arithmetic  data/arith.json + arith.bin: every validation photo's predicted vector and a word list's embeddings
import { escapeHtml } from './md.js';

const $ = id => document.getElementById(id);
const getJson = url => fetch(url).then(r => { if (!r.ok) throw new Error(`${url}: ${r.status}`); return r.json(); });
const status = $('status');
const reduceMotion = matchMedia('(prefers-reduced-motion: reduce)');

// ---- modes ------------------------------------------------------------------------------------------------------
document.querySelectorAll('.mode-tabs button').forEach(b => b.addEventListener('click', () => {
  document.querySelectorAll('.mode-tabs button').forEach(x => x.setAttribute('aria-pressed', String(x === b)));
  $('mode-reveal').hidden = b.dataset.mode !== 'reveal';
  $('mode-arith').hidden = b.dataset.mode !== 'arith';
  if (b.dataset.mode === 'arith') arith.ready.then(() => arith.run());
  else stopAuto();
}));

const [R, A, abin] = await Promise.all([getJson('./data/reveal.json'), getJson('./data/arith.json'), fetch('./data/arith.bin').then(r => r.arrayBuffer())]);
status.textContent = `Recorded on my Mac with the final predictor (epoch ${R.epoch + 1} of 30). V-JEPA 2 and EmbeddingGemma are too big to run in your browser, so every answer here was computed ahead of time.`;

// ================================================================================================ Reveal
const tilesEl = $('tiles'), cover = $('cover'), img = $('reveal-img');
cover.innerHTML = '<i></i>'.repeat(64);
const coverCells = [...cover.children];
let photo = null, t = 0, auto = 0, called = null;
const seen = new Set();

const picks = $('reveal-picks');
picks.innerHTML = R.photos.map((p, i) => `<button type="button" role="option" data-i="${i}" aria-selected="false" aria-label="Photo ${i + 1}"><span>${i + 1}</span><img src="data/${p.image}" alt=""></button>`).join('');
const pickButtons = [...picks.children];
picks.addEventListener('click', e => { const b = e.target.closest('button'); if (b) choose(Number(b.dataset.i)); });

// first tile count at which one of the photo's own captions is v2's top pick among the 5,070 validation captions
const knowsAt = ranks => { const i = ranks.indexOf(1); return i < 0 ? null : i + 1; };

function choose(i) {
  stopAuto();
  photo = R.photos[i];
  pickButtons.forEach((b, j) => b.setAttribute('aria-selected', String(j === i)));
  img.src = `data/${photo.image}`;
  called = null;
  $('verdict').hidden = true;
  $('reveal-know').disabled = false;
  setT(0, false);
}

function setT(n, animate = true) {
  t = n;
  const open = new Set(photo.order.slice(0, t));
  coverCells.forEach((c, k) => { c.classList.toggle('open', open.has(k)); c.style.transitionDuration = animate && !reduceMotion.matches ? '' : '0s'; });
  $('tile-count').textContent = `${t} / 64`;
  $('reveal-one').disabled = t >= 64;
  for (const side of ['before', 'after']) {
    const main = $(`guess-${side}`), rank = $(`rank-${side}`);
    if (!t) { main.textContent = side === 'before' ? 'Nothing to go on yet.' : '–'; rank.textContent = ''; if (side === 'before') $('alt-before').innerHTML = ''; continue; }
    const g = photo[side];
    main.textContent = R.captions[g.guess[t - 1][0]];
    const r = g.rank[t - 1];
    rank.innerHTML = r === 1 ? '<b>Its own caption is the top pick</b> among the 5,070 validation captions.' : `Its own captions rank <b>#${r.toLocaleString()}</b> of 5,070.`;
    if (side === 'before') $('alt-before').innerHTML = g.guess[t - 1].slice(1).map(j => `<li>${escapeHtml(R.captions[j])}</li>`).join('');
  }
  drawRanks();
  if (t >= 64) { stopAuto(); if (!called) finish(null); }
}

// the rank of the photo's own best caption, after every tile: lower is better, 1 is a perfect answer
function drawRanks() {
  const W = 760, H = 170, l = 46, r = 10, top = 12, b = 28;
  const px = n => l + ((n - 1) / 63) * (W - l - r);
  const py = rank => top + (Math.log10(rank) / Math.log10(5070)) * (H - top - b);
  const upto = called ? 64 : t;
  const path = ranks => ranks.slice(0, upto).map((v, i) => `${i ? 'L' : 'M'}${px(i + 1).toFixed(1)} ${py(v).toFixed(1)}`).join('');
  const mark = (ranks, color) => { const k = knowsAt(ranks); return k && k <= upto ? `<circle cx="${px(k)}" cy="${py(1)}" r="4.5" fill="${color}" stroke="#fff" stroke-width="1.5"/>` : ''; };
  const ticks = [1, 10, 100, 1000].map(v => `<line x1="${l}" x2="${W - r}" y1="${py(v)}" y2="${py(v)}" class="grid"/><text x="${l - 6}" y="${py(v) + 4}" text-anchor="end" class="axis">#${v.toLocaleString()}</text>`).join('');
  const xt = [1, 16, 32, 48, 64].map(v => `<text x="${px(v)}" y="${H - 8}" text-anchor="middle" class="axis">${v}</text>`).join('');
  const cursor = t && !called ? `<line x1="${px(t)}" x2="${px(t)}" y1="${top}" y2="${H - b}" stroke="var(--yellow)" stroke-width="2"/>` : '';
  const you = called ? `<line x1="${px(called)}" x2="${px(called)}" y1="${top}" y2="${H - b}" stroke="var(--ink)" stroke-dasharray="3 3"/><text x="${px(called) + 5}" y="${H - b - 6}" class="axis you">you</text>` : '';
  $('rank-chart').innerHTML = `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="Rank of the photo's own caption after each tile, for both versions">${ticks}${xt}${cursor}${you}
    <path d="${path(photo.after.rank)}" fill="none" stroke="var(--grad)" stroke-width="1.8" stroke-dasharray="5 4"/>
    <path d="${path(photo.before.rank)}" fill="none" stroke="var(--accent)" stroke-width="2.4"/>
    ${mark(photo.after.rank, 'var(--grad)')}${mark(photo.before.rank, 'var(--accent)')}</svg>
    <figcaption><span><i class="k-honest"></i>v2</span><span><i class="k-leaky"></i>leaky v2</span><span>rank of its own caption, by tiles uncovered</span></figcaption>`;
}

function finish(at) {
  called = at ?? 64;
  stopAuto();
  $('reveal-know').disabled = true;
  seen.add(R.photos.indexOf(photo));
  pickButtons.forEach((b, j) => b.classList.toggle('seen', seen.has(j)));
  // uncover what is left, quickly
  const rest = photo.order.slice(t);
  rest.forEach((k, j) => setTimeout(() => coverCells[k].classList.add('open'), reduceMotion.matches ? 0 : j * 14));
  const kb = knowsAt(photo.before.rank), ka = knowsAt(photo.after.rank);
  const tiles = n => `${n} tile${n === 1 ? '' : 's'}`;
  let line;
  if (at === null) line = kb ? `v2 put one of its own captions first after <b>${tiles(kb)}</b>.` : `Even with every tile uncovered, v2 never put one of its own captions first.`;
  else if (!kb || kb > at) line = `You called it at <b>${tiles(at)}</b>. ${kb ? `v2 needed <b>${tiles(kb)}</b>.` : 'v2 never got there.'} You win this one.`;
  else if (kb === at) line = `You both got there at <b>${tiles(at)}</b>.`;
  else line = `You called it at <b>${tiles(at)}</b>, but v2 already had it at <b>${tiles(kb)}</b>.`;
  const leak = ka ? `The leaky version “knew” after ${tiles(ka)}${kb && ka < kb ? `, ${kb - ka} sooner: its tokens had already seen the hidden tiles` : ''}.` : '';
  $('verdict').innerHTML = `<p>${line} ${leak}</p><p class="truth">What people wrote: ${photo.captions.slice(0, 3).map(c => `“${escapeHtml(c)}”`).join(' ')}</p>
    <button type="button" id="reveal-next" class="primary">Next photo</button>`;
  $('verdict').hidden = false;
  $('reveal-next').addEventListener('click', () => choose((R.photos.indexOf(photo) + 1) % R.photos.length));
  drawRanks();
}

function stopAuto() { clearInterval(auto); auto = 0; $('reveal-auto').textContent = 'Play'; }
$('reveal-one').addEventListener('click', () => { if (t < 64 && !called) setT(t + 1); });
$('reveal-auto').addEventListener('click', () => {
  if (auto) return stopAuto();
  if (called || t >= 64) choose(R.photos.indexOf(photo));
  $('reveal-auto').textContent = 'Pause';
  auto = setInterval(() => (t < 64 && !called ? setT(t + 1) : stopAuto()), 650);
});
$('reveal-know').addEventListener('click', () => { if (!called) finish(Math.max(t, 1)); });
choose(0);

// ================================================================================================ Arithmetic
const D = A.dim;
const deq = (q, scales, row) => { const out = new Float32Array(D), s = scales[row]; for (let k = 0; k < D; k++) out[k] = q[row * D + k] * s; return out; };
const P = { q: new Int8Array(abin, A.offsets.photos, A.photos * D), s: new Float32Array(abin, A.offsets.photoScales, A.photos) };
const Wd = { q: new Int8Array(abin, A.offsets.words, A.words.length * D), s: new Float32Array(abin, A.offsets.wordScales, A.words.length) };
const photoNorm = Float32Array.from({ length: A.photos }, (_, n) => { let s = 0; for (let k = 0; k < D; k++) s += (P.q[n * D + k] * P.s[n]) ** 2; return Math.sqrt(s); });
const wordIndex = Object.fromEntries(A.words.map((w, i) => [w, i]));

const STARTS = [140, 166, 58, 44, 14, 71, 180, 41, 105, 116];
// checked by hand in export_v2_results.py; dog → cat does not work (Flickr30k has almost no cats)
const PRESETS = [[140, 'beach', 'snow'], [140, 'dog', 'horse'], [58, 'surfing', 'skiing'], [105, 'water', 'snow'], [71, 'climbing', 'swimming'], [116, 'guitar', 'drum'], [44, 'car', 'horse'], [166, 'dog', 'child']];
let start = STARTS[0];
const arithPicks = $('arith-picks');
arithPicks.innerHTML = STARTS.map(n => `<button type="button" role="option" data-n="${n}" aria-selected="false"><img src="data/v2/val/${n}.jpg" alt=""></button>`).join('');
const options = ['<option value="">nothing</option>', ...[...A.words].sort().map(w => `<option value="${w}">${w}</option>`)].join('');
$('arith-minus').innerHTML = options; $('arith-plus').innerHTML = options;
$('arith-presets').innerHTML = '<span>Try</span>' + PRESETS.map(([n, a, b], i) => `<button type="button" data-i="${i}"><img src="data/v2/val/${n}.jpg" alt=""> − ${a} + ${b}</button>`).join('');

const arith = {
  ready: Promise.resolve(),
  run() {
    const minus = $('arith-minus').value, plus = $('arith-plus').value;
    const v = deq(P.q, P.s, start);
    if (minus) { const w = deq(Wd.q, Wd.s, wordIndex[minus]); for (let k = 0; k < D; k++) v[k] -= w[k]; }
    if (plus) { const w = deq(Wd.q, Wd.s, wordIndex[plus]); for (let k = 0; k < D; k++) v[k] += w[k]; }
    let vn = 0;
    for (let k = 0; k < D; k++) vn += v[k] * v[k];
    vn = Math.sqrt(vn);
    const scores = [];
    for (let n = 0; n < A.photos; n++) {
      if (n === start) continue;
      let s = 0;
      const o = n * D, sc = P.s[n];
      for (let k = 0; k < D; k++) s += P.q[o + k] * sc * v[k];
      scores.push([n, s / (photoNorm[n] * vn)]);
    }
    scores.sort((a, b) => b[1] - a[1]);
    $('arith-img').src = `data/v2/val/${start}.jpg`;
    $('arith-cap').textContent = A.firstCaption[start];
    arithPicks.querySelectorAll('button').forEach(b => b.setAttribute('aria-selected', String(Number(b.dataset.n) === start)));
    $('arith-hits').innerHTML = scores.slice(0, 8).map(([n, s]) => `<li><button type="button" data-n="${n}" title="Start from this photo"><img src="data/v2/val/${n}.jpg" alt="" loading="lazy"></button><p><b>${s.toFixed(3)}</b>${escapeHtml(A.firstCaption[n])}</p></li>`).join('');
    $('arith-note').textContent = minus || plus
      ? `The ${A.photos.toLocaleString()} validation photos nearest to photo${minus ? ` − “${minus}”` : ''}${plus ? ` + “${plus}”` : ''}. Click one to start from it.`
      : `The validation photos v2 places nearest this one. Pick words to move it.`;
  }
};
arithPicks.addEventListener('click', e => { const b = e.target.closest('button'); if (b) { start = Number(b.dataset.n); arith.run(); } });
$('arith-hits').addEventListener('click', e => { const b = e.target.closest('button'); if (b) { start = Number(b.dataset.n); arith.run(); $('mode-arith').scrollTo({ top: 0, behavior: reduceMotion.matches ? 'auto' : 'smooth' }); } });
['arith-minus', 'arith-plus'].forEach(id => $(id).addEventListener('change', () => arith.run()));
$('arith-presets').addEventListener('click', e => {
  const b = e.target.closest('button');
  if (!b) return;
  const [n, a, c] = PRESETS[Number(b.dataset.i)];
  start = n; $('arith-minus').value = a; $('arith-plus').value = c;
  arith.run();
});
$('arith-minus').value = 'beach'; $('arith-plus').value = 'snow';
