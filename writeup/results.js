// The Results tab. Right: the training run (content/results-*.md, with its {{charts}} filled in from
// data/metrics.json). Left (v1 only): the trained model running in a Web Worker (worker.js → infer.js), the same
// maths as infer.py: photo → 256 numbers, caption → 256 numbers, score = their dot product.
import { parseSteps, escapeHtml, rich } from './md.js';
import './split.js';

const version = document.body.dataset.version;
const article = document.getElementById('article');
const getJson = url => fetch(url).then(r => { if (!r.ok) throw new Error(`${url}: ${r.status}`); return r.json(); });
const katexReady = new Promise(resolve => {
  if (window.katex) return resolve();
  document.querySelector('script[src*="katex"]').addEventListener('load', resolve, { once: true });
});

// ---- the article ------------------------------------------------------------------------------------------------
const [source, metrics] = await Promise.all([
  fetch(`./content/results-${version}.md`, { cache: 'no-cache' }).then(r => r.text()),
  getJson('./data/metrics.json'), katexReady
]);
parseSteps(source).forEach((step, i) => {
  const section = document.createElement('section');
  section.className = 'reading-block';
  section.id = step.id;
  section.innerHTML = (i === 0 ? `<h2>${rich(step.title)}</h2>` : `<h3>${rich(step.title)}</h3>`) + step.html;
  article.append(section);
});

const NS = 'http://www.w3.org/2000/svg';
function lineChart(series, { yMax, yMin = 0, xMax, xTitle, yTicks, marks = [], h = 260 }) {
  const W = 640, H = h, l = 40, r = 12, t = 12, b = 34;
  const px = x => l + (x / xMax) * (W - l - r), py = y => t + (1 - (y - yMin) / (yMax - yMin)) * (H - t - b);
  const parts = [];
  yTicks.forEach(v => parts.push(`<line class="grid" x1="${l}" x2="${W - r}" y1="${py(v)}" y2="${py(v)}"/><text class="axis" x="${l - 6}" y="${py(v) + 4}" text-anchor="end">${v}</text>`));
  [0, 50, 100, 150, 200].filter(x => x <= xMax).forEach(x => parts.push(`<text class="axis" x="${px(x)}" y="${H - b + 16}" text-anchor="middle">${x}</text>`));
  parts.push(`<text class="axis-title" x="${(l + W - r) / 2}" y="${H - 4}" text-anchor="middle">${xTitle}</text>`);
  marks.forEach(m => parts.push(m(px, py)));
  series.forEach(s => parts.push(`<path d="${s.points.map(([x, y], i) => `${i ? 'L' : 'M'}${px(x).toFixed(1)} ${py(y).toFixed(1)}`).join('')}" fill="none" stroke="${s.color}" stroke-width="${s.width || 2}" stroke-opacity="${s.opacity || 1}" ${s.dash ? `stroke-dasharray="${s.dash}"` : ''}/>`));
  const key = series.filter(s => s.label).map(s => `<span><i style="border-color:${s.color}"></i>${s.label}</span>`).join('');
  return `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="${escapeHtml(series.map(s => s.label).filter(Boolean).join(', '))}">${parts.join('')}</svg><div class="key">${key}</div>`;
}
const E = metrics.epochs, spe = metrics.steps_per_epoch;
const pct = v => `${(v * 100).toFixed(1)}%`;
const widgets = {
  loss: () => lineChart([
    { points: metrics.steps.map(([s, v]) => [s / spe, v]), color: 'var(--accent)', width: 1, opacity: .45, label: 'training loss, per step' },
    { points: E.map(e => [e.epoch, e.val_loss]), color: 'var(--grad)', width: 2.4, label: 'validation loss, per epoch' }
  ], { yMax: 7.2, xMax: 200, xTitle: 'epoch', yTicks: [0, 2, 4, 6], marks: [
    (px, py) => `<line x1="${px(0)}" x2="${px(200)}" y1="${py(metrics.chance)}" y2="${py(metrics.chance)}" stroke="var(--muted)" stroke-dasharray="5 5"/><text class="axis" x="${px(200)}" y="${py(metrics.chance) - 6}" text-anchor="end">guessing: ${metrics.chance.toFixed(2)}</text>`,
    (px, py) => `<line x1="${px(metrics.best_epoch)}" x2="${px(metrics.best_epoch)}" y1="${py(7.2)}" y2="${py(0)}" stroke="var(--yellow)" stroke-width="2"/><text class="axis" x="${px(metrics.best_epoch) + 6}" y="${py(0.4)}">best, epoch ${metrics.best_epoch}</text>`
  ] }),
  recall: () => lineChart([
    { points: E.map(e => [e.epoch, e.r5 * 100]), color: 'var(--accent)', label: 'R@5, photo → caption' },
    { points: E.map(e => [e.epoch, e.r1 * 100]), color: 'var(--neg)', label: 'R@1, photo → caption' }
  ], { yMax: 40, xMax: 200, xTitle: 'epoch (checked each epoch on caption 1 of each photo, 1,014 to choose from)', yTicks: [0, 10, 20, 30, 40], h: 220, marks: [
    (px, py) => `<line x1="${px(metrics.best_epoch)}" x2="${px(metrics.best_epoch)}" y1="${py(40)}" y2="${py(0)}" stroke="var(--yellow)" stroke-width="2"/>`
  ] }),
  temperature: () => lineChart([
    { points: E.map(e => [e.epoch, e.temp]), color: 'var(--ink)', label: 'learned scale e^t' }
  ], { yMin: 10, yMax: 35, xMax: 200, xTitle: 'epoch', yTicks: [10, 20, 30], h: 200, marks: [
    (px, py) => `<line x1="${px(metrics.best_epoch)}" x2="${px(metrics.best_epoch)}" y1="${py(35)}" y2="${py(10)}" stroke="var(--yellow)" stroke-width="2"/>`
  ] }),
  retrieval: () => {
    const F = metrics.final;
    const row = (name, r) => `<tr><td>${name}</td><td class="n">${pct(r['1'])}</td><td class="n">${pct(r['5'])}</td><td class="n">${pct(r['10'])}</td></tr>`;
    return `<table class="recall-table"><thead><tr><th></th><th>R@1</th><th>R@5</th><th>R@10</th></tr></thead><tbody>
      ${row(`photo → caption <span style="color:var(--muted)">(of ${F.captions.toLocaleString()})</span>`, F.i2t)}
      ${row(`caption → photo <span style="color:var(--muted)">(of ${F.images.toLocaleString()})</span>`, F.t2i)}
      <tr class="chance"><td>at random</td><td class="n">0.1%</td><td class="n">0.5%</td><td class="n">1.0%</td></tr></tbody></table>`;
  }
};
article.querySelectorAll('.widget').forEach(fig => {
  const make = widgets[fig.dataset.widget];
  if (make) { fig.classList.add(fig.dataset.widget === 'retrieval' ? 'table' : 'chart'); fig.innerHTML = make(); }
});

// ---- the demo (v1) ----------------------------------------------------------------------------------------------
if (version === 'v1') {
  const $ = id => document.getElementById(id);
  const status = $('status');
  const [refs, trace, refBytes] = await Promise.all([
    getJson('./data/refs.json'), getJson('./data/trace.json'),
    fetch('./data/refs.bin').then(r => r.arrayBuffer())
  ]);
  const refQ = new Int8Array(refBytes, refs.int8Offset, refs.n * refs.dim);
  const refScale = new Float32Array(refBytes, refs.scaleOffset, refs.n);
  // every validation caption against one vector
  const scoreAll = vec => {
    const out = new Float32Array(refs.n);
    for (let c = 0; c < refs.n; c++) {
      let s = 0;
      const o = c * refs.dim;
      for (let k = 0; k < refs.dim; k++) s += refQ[o + k] * vec[k];
      out[c] = s * refScale[c];
    }
    return out;
  };

  // the photos: the write-up's dog first, then 48 more validation photos
  const items = [
    { full: `./data/${trace.image.display}`, input: `./data/${trace.image.input}`, val: trace.image.val, captions: trace.image.captions, caption: trace.text.caption },
    ...refs.gallery.map(g => ({ full: `./data/gallery/${g.file}.jpg`, input: `./data/gallery/${g.file}-128.png`, val: g.val, captions: g.captions, caption: g.captions[0] }))
  ];
  const thumbs = $('thumbs');
  thumbs.innerHTML = items.map((it, i) => `<button type="button" role="option" data-i="${i}" aria-selected="false" aria-label="Validation photo ${i + 1}"><img src="${it.full}" alt="" loading="lazy"></button>`).join('');
  const thumbButtons = [...thumbs.children];

  // the worker
  const worker = new Worker('./worker.js', { type: 'module' });
  let nextId = 0;
  const pending = new Map();
  const ask = msg => new Promise((resolve, reject) => {
    const id = ++nextId;
    pending.set(id, { resolve, reject });
    worker.postMessage({ ...msg, id }, msg.rgba ? [msg.rgba.buffer] : []);
  });
  const ready = new Promise((resolve, reject) => {
    worker.onmessage = ({ data }) => {
      if (data.type === 'ready') return resolve(data);
      if (data.type === 'error' && !data.id) return reject(new Error(data.message));
      const p = pending.get(data.id);
      if (!p) return;
      pending.delete(data.id);
      data.type === 'error' ? p.reject(new Error(data.message)) : p.resolve(data);
    };
  });
  worker.postMessage({ type: 'load', base: new URL('./', location.href).href });

  // 128 × 128 RGBA, the way the training images were prepared: short side to 128, then the centre square
  const canvas = document.createElement('canvas');
  canvas.width = canvas.height = 128;
  const g2d = canvas.getContext('2d', { willReadFrequently: true });
  const pixels = img => {
    const s = Math.min(img.naturalWidth, img.naturalHeight);
    g2d.imageSmoothingQuality = 'high';
    g2d.clearRect(0, 0, 128, 128);
    g2d.drawImage(img, (img.naturalWidth - s) / 2, (img.naturalHeight - s) / 2, s, s, 0, 0, 128, 128);
    return new Uint8ClampedArray(g2d.getImageData(0, 0, 128, 128).data);
  };
  const load = src => new Promise((resolve, reject) => { const im = new Image(); im.onload = () => resolve(im); im.onerror = reject; im.src = src; });

  const photo = $('photo'), photoImg = $('photo-img'), heat = $('heat');
  heat.innerHTML = '<i></i>'.repeat(64);
  const heatCells = [...heat.children];
  $('show-heat').addEventListener('change', e => photo.classList.toggle('show-heat', e.target.checked));
  const caption = $('caption');
  let current = null; // { vec, val, scores }
  let busy = 0;

  async function choose(i, file) {
    const token = ++busy;
    thumbButtons.forEach((b, j) => b.setAttribute('aria-selected', String(j === i)));
    let img, valPos = null;
    if (file) {
      const url = URL.createObjectURL(file);
      photoImg.src = url;
      img = await load(url);
      $('photo-note').textContent = 'your photo';
    } else {
      const it = items[i];
      photoImg.src = it.full;
      img = await load(it.input); // exactly the 128 × 128 the model saw in validation
      valPos = it.val;
      $('photo-note').textContent = `validation photo #${it.val}`;
      if (!caption.dataset.touched) caption.value = it.caption;
    }
    status.textContent = 'Loading the weights (14 MB), then running the Image Encoder and the Predictor…';
    await ready;
    const r = await ask({ type: 'image', rgba: pixels(img) });
    if (token !== busy) return;
    const scores = scoreAll(r.vec);
    current = { vec: r.vec, val: valPos, scores };
    const q = r.qAttn.subarray(64, 128), m = Math.max(...q);
    heatCells.forEach((c, k) => { c.style.opacity = (0.72 * q[k] / m).toFixed(3); });
    const order = [...scores.keys()].sort((a, b) => scores[b] - scores[a]).slice(0, 5);
    $('best').innerHTML = order.map(c => `<li class="${refs.owner[c] === valPos ? 'own' : ''}" data-c="${c}"><span class="s">${scores[c].toFixed(3)}</span><span>${escapeHtml(refs.captions[c])}</span></li>`).join('');
    status.innerHTML = `Photo → 256 numbers in <b>${Math.round(r.ms)} ms</b>, in your browser · scored against 5,070 captions`;
    await scoreCaption();
  }
  $('best').addEventListener('click', e => {
    const li = e.target.closest('li');
    if (!li) return;
    caption.value = refs.captions[li.dataset.c];
    caption.dataset.touched = '1';
    scoreCaption();
  });

  let typing = 0;
  const norm = t => t.toLowerCase().replace(/[^a-z0-9' ]+/g, ' ').replace(/\s+/g, ' ').trim();
  async function scoreCaption() {
    if (!current) return;
    const text = caption.value.trim();
    if (!text) { $('score').textContent = '–'; $('rank').textContent = ''; $('unknown').textContent = ''; $('rank-fill').style.width = '0'; return; }
    const token = ++typing;
    const r = await ask({ type: 'text', text });
    if (token !== typing) return;
    if (!r.vec) { $('score').textContent = '–'; $('rank').textContent = 'None of these words are in the vocabulary.'; $('rank-fill').style.width = '0'; return; }
    let s = 0;
    for (let k = 0; k < 256; k++) s += r.vec[k] * current.vec[k];
    // a caption typed word for word from the pool is not its own competition
    const same = norm(text);
    let above = 0, pool = 0;
    current.scores.forEach((v, c) => { if (norm(refs.captions[c]) === same) return; pool++; if (v > s) above++; });
    $('score').textContent = s.toFixed(3);
    $('rank-fill').style.width = `${(100 * (1 - above / pool)).toFixed(1)}%`;
    $('rank').innerHTML = `Rank <b>${(above + 1).toLocaleString()}</b> against ${pool.toLocaleString()} validation captions. ` +
      (above === 0 ? 'None of them fits this photo better.' : above === 1 ? 'Only one fits this photo better.' : `${above.toLocaleString()} fit this photo better.`);
    $('unknown').textContent = r.unknown.length ? `Not in its 7,441-word vocabulary, read as <unk>: ${r.unknown.join(', ')}` : '';
  }
  caption.addEventListener('input', () => { caption.dataset.touched = '1'; clearTimeout(scoreCaption.t); scoreCaption.t = setTimeout(scoreCaption, 120); });

  thumbs.addEventListener('click', e => { const b = e.target.closest('button'); if (b) choose(Number(b.dataset.i)); });
  $('upload').addEventListener('change', e => { const f = e.target.files[0]; if (f) choose(-1, f); });
  const body = document.querySelector('.demo-body');
  body.addEventListener('dragover', e => { e.preventDefault(); body.classList.add('is-dragging'); });
  body.addEventListener('dragleave', () => body.classList.remove('is-dragging'));
  body.addEventListener('drop', e => { e.preventDefault(); body.classList.remove('is-dragging'); const f = [...e.dataTransfer.files].find(f => f.type.startsWith('image/')); if (f) choose(-1, f); });

  // search: rank the photos by a caption. Their vectors are computed once, in the background, when first needed.
  const vecs = new Array(items.length).fill(null);
  let embedding = null;
  const embedAll = () => embedding ||= (async () => {
    await ready;
    for (let i = 0; i < items.length; i++) {
      const r = await ask({ type: 'image', rgba: pixels(await load(items[i].input)) });
      vecs[i] = r.vec;
      $('search-note').textContent = `Embedding the ${items.length} photos in your browser: ${i + 1} / ${items.length}`;
      if (i % 6 === 5) runSearch();
    }
    runSearch();
  })();
  const query = $('query');
  let searchToken = 0;
  async function runSearch() {
    const text = query.value.trim();
    const done = vecs.filter(Boolean).length;
    if (done === items.length) $('search-note').textContent = `Ranking ${items.length} validation photos by this caption`;
    if (!text) { $('hits').innerHTML = ''; return; }
    const token = ++searchToken;
    const r = await ask({ type: 'text', text });
    if (token !== searchToken || !r.vec) return;
    const ranked = vecs.map((v, i) => v && [i, v.reduce((s, x, k) => s + x * r.vec[k], 0)]).filter(Boolean).sort((a, b) => b[1] - a[1]).slice(0, 8);
    $('hits').innerHTML = ranked.map(([i, s]) => `<li><img src="${items[i].full}" alt=""><p><b>${s.toFixed(3)}</b>${escapeHtml(items[i].captions[0])}</p></li>`).join('');
  }
  query.addEventListener('input', () => { clearTimeout(runSearch.t); runSearch.t = setTimeout(runSearch, 150); });

  document.querySelectorAll('.mode-tabs button').forEach(b => b.addEventListener('click', () => {
    const search = b.dataset.mode === 'search';
    document.querySelectorAll('.mode-tabs button').forEach(x => x.setAttribute('aria-pressed', String(x === b)));
    $('mode-photo').hidden = search; $('mode-search').hidden = !search;
    document.querySelector('.picker').hidden = search;
    if (search) { if (!query.value) query.value = 'a dog running on the beach'; embedAll(); runSearch(); query.focus(); }
  }));

  ready.then(() => { if (!current) status.textContent = 'Weights loaded. Running the Image Encoder and the Predictor…'; }).catch(err => { status.textContent = `Could not load the model: ${err.message}`; });
  choose(0);
}
