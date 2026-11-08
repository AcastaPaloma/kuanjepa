// v2, the JEPA version: planned, not trained. The photo, the caption and the token pictures are v1's real ones,
// standing in for what v2 will compute; anything a trained v2 would produce (its guesses, its losses) is
// illustrative and labelled so.
import { el, txt, group, box, chip, arrow, image, matrixURL, vectorURL, show } from './stage.js';

const C = { pink: 'var(--x-pink)', orange: 'var(--x-orange)', orange2: 'var(--x-orange-2)', purple: 'var(--x-purple)', blue: 'var(--x-blue)', yellow: 'var(--x-yellow)', green: 'var(--x-green)' };
// the four hidden blocks on the 8 × 8 grid: [column, row, width, height]
export const TARGETS = [[1, 1, 3, 3], [5, 0, 3, 3], [0, 5, 3, 3], [4, 5, 3, 3]];
const hidden = new Set(TARGETS.flatMap(([x, y, w, h]) => Array.from({ length: w * h }, (_, i) => (y + Math.floor(i / w)) * 8 + x + (i % w))));
const context = Array.from({ length: 64 }, (_, k) => k).filter(k => !hidden.has(k));
const targetsList = Array.from({ length: 64 }, (_, k) => k).filter(k => hidden.has(k));

// a small seeded random, so the illustrative numbers are the same on every visit
const rng = seed => () => ((seed = (seed * 1664525 + 1013904223) >>> 0) / 4294967296);

export function buildV2(world, data) {
  const { trace: T, sprites: S, metrics: M } = data;
  const IMG = 'data/' + T.image.display, INPUT = 'data/' + T.image.input;
  const L = T.text.words.length;
  const tokVal = (sp, rows, stage, r, k) => (sp.at(k, stage * rows + r) - 127.5) / 127.5;
  const rowsURL = (sp, rows, stage, list, dim = () => false) => matrixURL(list.length, 256, (i, k) => dim(list[i]) ? 0 : tokVal(sp, rows, stage, list[i], k));
  const all64 = Array.from({ length: 64 }, (_, k) => k);
  const scenes = {};

  // the photo with its grid and hidden blocks, reused by several scenes
  function maskedPhoto(parent, x, y, size, { grid = true, blocks = true, ctxTint = false } = {}) {
    const g = group(parent);
    image(g, INPUT, x, y, size, size);
    const cs = size / 8;
    const tint = group(g, 'fade');
    context.forEach(k => el('rect', { x: x + (k % 8) * cs, y: y + Math.floor(k / 8) * cs, width: cs, height: cs, fill: 'var(--accent)', opacity: .22 }, tint));
    show(tint, ctxTint);
    const blockEls = TARGETS.map(([bx, by, w, h], i) => {
      const bg = group(g, 'fade');
      el('rect', { x: x + bx * cs, y: y + by * cs, width: w * cs, height: h * cs, fill: '#4a4a46', opacity: .82 }, bg);
      el('rect', { x: x + bx * cs, y: y + by * cs, width: w * cs, height: h * cs, fill: 'none', stroke: 'var(--x-red)', 'stroke-width': 2.5 }, bg);
      if (size > 200) txt(bg, x + (bx + w / 2) * cs, y + (by + h / 2) * cs + 6, `target ${i + 1}`, { 'text-anchor': 'middle', 'font-size': 15, fill: '#fff' });
      show(bg, blocks);
      return bg;
    });
    if (grid) for (let i = 1; i < 8; i++) {
      el('line', { x1: x + i * cs, y1: y, x2: x + i * cs, y2: y + size, stroke: '#ffffffaa', 'stroke-width': 1 }, g);
      el('line', { x1: x, y1: y + i * cs, x2: x + size, y2: y + i * cs, stroke: '#ffffffaa', 'stroke-width': 1 }, g);
    }
    el('rect', { x, y, width: size, height: size, fill: 'none', class: 'box' }, g);
    return { g, blockEls, tint };
  }

  // ================================================================================================ overview: three games
  {
    const g = group(world);
    const rows = [
      { y: 40, title: 'Game 1 · fill in the photo', from: 'photo with holes', guess: 'hidden patches', answer: 'slow copy of the Image Encoder, whole photo' },
      { y: 210, title: 'Game 2 · fill in the caption', from: 'caption with holes', guess: 'hidden words', answer: 'slow copy of the Text Encoder, whole caption' },
      { y: 380, title: 'Game 3 · the bridge', from: 'photo with holes', guess: 'the caption’s TARGET', answer: 'slow copy of the Text Encoder, whole caption' }
    ];
    const els = rows.map((r, i) => {
      const rg = group(g, 'fade');
      txt(rg, 20, r.y, r.title, { 'font-size': 20 });
      if (i === 1) {
        box(rg, 20, r.y + 20, 150, 90, '#fff');
        txt(rg, 32, r.y + 52, 'a brown ▇▇▇ on', { 'font-size': 14 }); txt(rg, 32, r.y + 74, 'the ▇▇▇▇ with a', { 'font-size': 14 }); txt(rg, 32, r.y + 96, 'yellow ball…', { 'font-size': 14 });
      } else maskedPhoto(rg, 50, r.y + 16, 100, { grid: false });
      arrow(rg, `M178 ${r.y + 65} H206`);
      box(rg, 212, r.y + 30, 150, 70, C.pink); txt(rg, 287, r.y + 70, 'encoder + predictor', { 'text-anchor': 'middle', 'font-size': 14 });
      arrow(rg, `M366 ${r.y + 65} H394`);
      box(rg, 400, r.y + 30, 130, 70, C.yellow); txt(rg, 465, r.y + 62, 'guess:', { 'text-anchor': 'middle', 'font-size': 13, class: 't-dim' }); txt(rg, 465, r.y + 84, r.guess, { 'text-anchor': 'middle', 'font-size': 14 });
      txt(rg, 550, r.y + 72, '≈ ?', { 'font-size': 22 });
      el('rect', { x: 600, y: r.y + 30, width: 200, height: 70, rx: 9, fill: '#fff', stroke: 'var(--ink)', 'stroke-width': 2, 'stroke-dasharray': '7 5' }, rg);
      r.answer.split(', ').forEach((line, j) => txt(rg, 700, r.y + 60 + j * 22, line, { 'text-anchor': 'middle', 'font-size': 13 }));
      return rg;
    });
    const note = txt(g, 410, 540, 'every guess is 256 numbers, graded against 256 numbers', { 'text-anchor': 'middle', 'font-size': 16, class: 't-red fade' });
    scenes['v2-overview'] = {
      g, view: [0, 0, 820, 560],
      beats: ['Game 1: fill in the photo', 'Game 2: fill in the caption', 'Game 3: guess the caption’s meaning from the photo', 'All graded in representation space, never pixels or words'],
      show(b) { els.forEach((e, i) => show(e, b >= i)); show(note, b >= 3); }
    };
  }

  // ================================================================================================ masking
  {
    const g = group(world);
    const top = 700;
    const mp = maskedPhoto(g, 40, top + 30, 360, { blocks: false });
    const labels = group(g, 'fade');
    txt(labels, 430, top + 80, `${targetsList.length} hidden tokens`, { 'font-size': 18, class: 't-red' });
    txt(labels, 430, top + 108, 'in 4 blocks', { 'font-size': 15, class: 't-dim' });
    const ctxLabel = group(g, 'fade');
    txt(ctxLabel, 430, top + 180, `${context.length} context tokens`, { 'font-size': 18, class: 't-accent' });
    txt(ctxLabel, 430, top + 208, 'what the encoder sees', { 'font-size': 15, class: 't-dim' });
    scenes.mask = {
      g, view: [20, top, 620, 420],
      beats: ['The 8 × 8 token grid', 'Target block 1', 'Target block 2', 'Target block 3', 'Target block 4', 'The rest is the context'],
      show(b) {
        mp.blockEls.forEach((bg, i) => show(bg, b >= i + 1));
        show(labels, b >= 4); show(mp.tint, b >= 5); show(ctxLabel, b >= 5);
      }
    };
  }

  // ================================================================================================ context encoder
  {
    const g = group(world);
    const top = 1300;
    const full = image(g, rowsURL(S.imgTokens, 64, 1, all64, k => hidden.has(k)), 40, top + 40, 120, 256);
    el('rect', { x: 40, y: top + 40, width: 120, height: 256, fill: 'none', class: 'thin' }, g);
    txt(g, 100, top + 26, '64 tokens + positions', { 'text-anchor': 'middle', 'font-size': 14 });
    const grey = group(g, 'fade');
    targetsList.forEach(k => el('rect', { x: 40, y: top + 40 + k * 4, width: 120, height: 4, fill: '#4a4a46', opacity: .75 }, grey));
    const drop = group(g, 'fade');
    arrow(drop, `M168 ${top + 168} H206`);
    image(drop, rowsURL(S.imgTokens, 64, 1, context), 214, top + 40 + (256 - context.length * 4) / 2, 120, context.length * 4);
    txt(drop, 274, top + 26 + (256 - context.length * 4) / 2, `keep ${context.length}`, { 'text-anchor': 'middle', 'font-size': 14 });
    const enc = group(g, 'fade');
    box(enc, 360, top + 40, 150, 256, C.pink);
    txt(enc, 435, top + 30, 'Context Encoder', { 'text-anchor': 'middle', 'font-size': 15 });
    for (let i = 0; i < 6; i++) { el('rect', { x: 372, y: top + 52 + i * 38, width: 126, height: 30, rx: 6, fill: C.orange, class: 'box' }, enc); txt(enc, 435, top + 73 + i * 38, `Block ${i + 1}`, { 'text-anchor': 'middle', 'font-size': 14 }); }
    arrow(enc, `M338 ${top + 168} H354`);
    const out = group(g, 'fade');
    arrow(out, `M514 ${top + 168} H540`);
    image(out, rowsURL(S.imgTokens, 64, 8, context), 548, top + 40 + (256 - context.length * 4) / 2, 120, context.length * 4);
    txt(out, 608, top + 26 + (256 - context.length * 4) / 2, 'context tokens', { 'text-anchor': 'middle', 'font-size': 14, class: 't-red' });
    scenes.context = {
      g, view: [20, top, 680, 320],
      beats: ['v1’s tokens, after the Positional Embeddings', 'The hidden ones are dropped', 'Only the context goes through the six blocks', `Out: ${context.length} context tokens`],
      show(b) { show(grey, b >= 1); show(drop, b >= 1); full.style.opacity = b >= 1 ? .35 : 1; show(enc, b >= 2); show(out, b >= 3); }
    };
  }

  // ================================================================================================ the slow copy
  {
    const g = group(world);
    const top = 1900, n = 6, cs = 22;
    const r = rng(7);
    // a few weights of each encoder, over several training steps: θ jumps around, θ̄ follows slowly
    const STEPS = 6, tau = 0.75; // τ exaggerated so the lag is visible
    const theta = [Array.from({ length: n * n }, () => r() * 2 - 1)];
    for (let s = 1; s <= STEPS; s++) theta.push(theta[s - 1].map(v => Math.max(-1, Math.min(1, v + (r() - 0.5) * 0.9))));
    const slow = [theta[0].slice()];
    for (let s = 1; s <= STEPS; s++) slow.push(slow[s - 1].map((v, i) => tau * v + (1 - tau) * theta[s][i]));
    const grid = (x, label, dashed) => {
      const gg = group(g);
      el('rect', { x, y: top + 30, width: 220, height: 230, rx: 9, fill: dashed ? '#fff' : C.pink, stroke: 'var(--ink)', 'stroke-width': 2, 'stroke-dasharray': dashed ? '7 5' : null }, gg);
      txt(gg, x + 110, top + 62, label, { 'text-anchor': 'middle', 'font-size': 17 });
      const cells = Array.from({ length: n * n }, (_, i) => el('rect', { x: x + 44 + (i % n) * cs, y: top + 84 + Math.floor(i / n) * cs, width: cs - 2, height: cs - 2, rx: 2, style: 'transition: fill .7s' }, gg));
      txt(gg, x + 110, top + 248, 'a few of its weights', { 'text-anchor': 'middle', 'font-size': 12, class: 't-muted' });
      return cells;
    };
    const a = grid(30, 'Context Encoder  θ', false), b = grid(380, 'slow copy  θ̄', true);
    const fill = v => { const t = v >= 0 ? [31, 58, 138] : [138, 59, 107], k = Math.min(1, Math.abs(v)); return `rgb(${255 + (t[0] - 255) * k} ${255 + (t[1] - 255) * k} ${255 + (t[2] - 255) * k})`; };
    const drift = group(g, 'fade');
    el('path', { d: `M254 ${top + 145} H372`, class: 'arrow flow', 'marker-end': 'url(#head)' }, drift);
    txt(drift, 313, top + 132, 'drifts toward', { 'text-anchor': 'middle', 'font-size': 13 });
    const formula = group(g, 'fade');
    txt(formula, 315, top + 300, 'θ̄ ← τ·θ̄ + (1 − τ)·θ        τ = 0.996 → 1', { 'text-anchor': 'middle', 'font-size': 18 });
    txt(formula, 315, top + 326, '(drawn with τ = 0.75 so you can see it move)', { 'text-anchor': 'middle', 'font-size': 13, class: 't-muted' });
    const nograd = group(g, 'fade');
    el('path', { d: `M490 ${top + 380} V${top + 268}`, class: 'arrow-red', 'stroke-dasharray': '6 5' }, nograd);
    el('path', { d: `M478 ${top + 316} l24 24 m0 -24 l-24 24`, stroke: 'var(--x-red)', 'stroke-width': 3 }, nograd);
    txt(nograd, 510, top + 372, 'the loss never trains it', { 'font-size': 15, class: 't-red' });
    let timer = 0;
    const paint = step => { a.forEach((c, i) => c.setAttribute('fill', fill(theta[step][i]))); b.forEach((c, i) => c.setAttribute('fill', fill(slow[step][i]))); };
    scenes.ema = {
      g, view: [20, top, 640, 400], beatMs: 2600,
      beats: ['Two copies of the Image Encoder, same weights at the start', 'Training changes θ every step; θ̄ follows slowly', 'The averaging rule', 'No gradient ever reaches the slow copy'],
      show(bt) {
        clearInterval(timer);
        show(drift, bt >= 1); show(formula, bt >= 2); show(nograd, bt >= 3);
        if (bt === 0) { paint(0); return; }
        let s = 0;
        paint(0);
        timer = setInterval(() => { s = s % STEPS + 1; paint(s); if (s === STEPS) s = 0; }, 900);
      },
      leave() { clearInterval(timer); }
    };
  }

  // ================================================================================================ game 1
  {
    const g = group(world);
    const top = 2500;
    const nT = targetsList.length, h = 6;
    image(g, rowsURL(S.imgTokens, 64, 8, context), 30, top + 50, 100, context.length * 5);
    txt(g, 80, top + 36, 'context', { 'text-anchor': 'middle', 'font-size': 14 });
    const masks = group(g, 'fade');
    targetsList.forEach((k, i) => el('rect', { x: 30, y: top + 60 + context.length * 5 + i * 5, width: 100, height: 4, fill: '#bdbcb5' }, masks));
    txt(masks, 80, top + 80 + (context.length + nT) * 5, `+ ${nT} mask tokens`, { 'text-anchor': 'middle', 'font-size': 14 });
    txt(masks, 80, top + 98 + (context.length + nT) * 5, 'position only', { 'text-anchor': 'middle', 'font-size': 12, class: 't-muted' });
    const pred = group(g, 'fade');
    box(pred, 160, top + 120, 120, 150, C.pink); txt(pred, 220, top + 110, 'image predictor', { 'text-anchor': 'middle', 'font-size': 14 });
    [0, 1].forEach(i => { el('rect', { x: 172, y: top + 140 + i * 60, width: 96, height: 46, rx: 6, fill: C.orange, class: 'box' }, pred); txt(pred, 220, top + 168 + i * 60, `Block ${i + 1}`, { 'text-anchor': 'middle', 'font-size': 14 }); });
    arrow(pred, `M136 ${top + 195} H154`);
    // guesses: the slow copy's answer plus noise (illustrative)
    const r = rng(3);
    const answer = (i, k) => tokVal(S.imgTokens, 64, 8, targetsList[i], k);
    const noise = Array.from({ length: nT * 256 }, () => (r() - 0.5) * 1.1);
    const guesses = group(g, 'fade');
    arrow(guesses, `M284 ${top + 195} H306`);
    image(guesses, matrixURL(nT, 256, (i, k) => Math.max(-1, Math.min(1, answer(i, k) + noise[i * 256 + k]))), 312, top + 50, 110, nT * h);
    txt(guesses, 367, top + 36, 'guesses', { 'text-anchor': 'middle', 'font-size': 14 });
    const ans = group(g, 'fade');
    image(ans, matrixURL(nT, 256, answer), 560, top + 50, 110, nT * h);
    txt(ans, 615, top + 36, 'slow copy, same spots', { 'text-anchor': 'middle', 'font-size': 14 });
    const full = maskedPhoto(ans, 580, top + nT * h + 80, 70, { grid: false, blocks: false });
    txt(ans, 615, top + nT * h + 168, 'it sees the whole photo', { 'text-anchor': 'middle', 'font-size': 12, class: 't-muted' });
    const dist = group(g, 'fade');
    const d = Array.from({ length: nT }, (_, i) => noise.slice(i * 256, i * 256 + 256).reduce((s, v) => s + Math.abs(v), 0) / 256);
    d.forEach((v, i) => el('rect', { x: 440, y: top + 50 + i * h + 1, width: v * 180, height: h - 2, fill: 'var(--grad)', opacity: .75 }, dist));
    txt(dist, 490, top + 66 + nT * h, 'SmoothL1, per token', { 'text-anchor': 'middle', 'font-size': 13 });
    txt(dist, 490, top + 84 + nT * h, '(illustrative)', { 'text-anchor': 'middle', 'font-size': 12, class: 't-muted' });
    scenes['game-image'] = {
      g, view: [10, top, 690, Math.max(nT * h + 190, 340)],
      beats: ['The context tokens', 'Plus one mask token per hidden spot', 'Through the image predictor', 'Its guesses for the hidden tokens', 'The answers: the slow copy, looking at the whole photo', 'The loss: how far each guess is'],
      show(b) { show(masks, b >= 1); show(pred, b >= 2); show(guesses, b >= 3); show(ans, b >= 4); show(dist, b >= 5); }
    };
  }

  // ================================================================================================ game 2
  {
    const g = group(world);
    const top = 3200;
    const hide = new Set(['dog', 'beach', 'yellow', 'mouth'].map(w => T.text.words.indexOf(w)).filter(i => i >= 0));
    const chips = [];
    let x = 30, y = top + 50;
    T.text.words.forEach((w, i) => {
      const width = Math.max(w.length, hide.has(i) ? 6 : 0) * 9 + 16;
      if (x + width > 680) { x = 30; y += 46; }
      const cg = group(g);
      const real = group(cg), masked = group(cg, 'fade');
      el('rect', { x, y: y - 19, width, height: 28, rx: 5, fill: '#fff', class: 'chip-rect' }, real);
      txt(real, x + width / 2, y, w, { 'text-anchor': 'middle', 'font-size': 16 });
      el('rect', { x, y: y - 19, width, height: 28, rx: 5, fill: '#4a4a46', stroke: 'var(--x-red)', 'stroke-width': 2 }, masked);
      txt(masked, x + width / 2, y, 'mask', { 'text-anchor': 'middle', 'font-size': 14, fill: '#fff' });
      chips.push({ masked, on: hide.has(i), cx: x + width / 2, cy: y + 12 });
      x += width + 8;
    });
    const bottom = y + 30;
    const enc = group(g, 'fade');
    box(enc, 30, bottom + 30, 200, 60, C.pink); txt(enc, 130, bottom + 66, 'Text Encoder', { 'text-anchor': 'middle', 'font-size': 16 });
    box(enc, 260, bottom + 30, 160, 60, C.pink); txt(enc, 340, bottom + 66, 'text predictor', { 'text-anchor': 'middle', 'font-size': 16 });
    arrow(enc, `M234 ${bottom + 60} H254`);
    const r = rng(11), ids = [...hide].sort((a, b) => a - b);
    const word = (i, k) => tokVal(S.txtTokens, L, 8, i, k);
    const guess = group(g, 'fade');
    arrow(guess, `M424 ${bottom + 60} H446`);
    image(guess, matrixURL(ids.length, 256, (j, k) => Math.max(-1, Math.min(1, word(ids[j], k) + (r() - 0.5) * 1.2))), 452, bottom + 30, 110, ids.length * 15);
    txt(guess, 507, bottom + 22, 'guesses', { 'text-anchor': 'middle', 'font-size': 14 });
    const ans = group(g, 'fade');
    image(ans, matrixURL(ids.length, 256, (j, k) => word(ids[j], k)), 452, bottom + 140, 110, ids.length * 15);
    ids.forEach((i, j) => txt(ans, 570, bottom + 152 + j * 15, T.text.words[i], { 'font-size': 13 }));
    el('rect', { x: 260, y: bottom + 130, width: 160, height: 60, rx: 9, fill: '#fff', stroke: 'var(--ink)', 'stroke-width': 2, 'stroke-dasharray': '7 5' }, ans);
    txt(ans, 340, bottom + 156, 'slow copy', { 'text-anchor': 'middle', 'font-size': 15 }); txt(ans, 340, bottom + 176, 'whole caption', { 'text-anchor': 'middle', 'font-size': 13, class: 't-muted' });
    arrow(ans, `M424 ${bottom + 160} H446`);
    txt(ans, 600, bottom + 100, '≈ ?', { 'font-size': 24 });
    scenes['game-text'] = {
      g, view: [10, top, 690, bottom - top + 220],
      beats: ['The caption', `Hide ${ids.length} of its ${L} words`, 'The Text Encoder reads the rest; a predictor guesses the hidden words', 'A slow copy of the Text Encoder reads the whole caption: the answers'],
      show(b) { chips.forEach(c => show(c.masked, b >= 1 && c.on)); show(enc, b >= 2); show(guess, b >= 2); show(ans, b >= 3); }
    };
  }

  // ================================================================================================ game 3: the bridge
  {
    const g = group(world);
    const top = 3800;
    const ph = maskedPhoto(g, 30, top + 40, 120, { grid: false });
    box(g, 180, top + 50, 150, 100, C.pink); txt(g, 255, top + 92, 'Context Encoder', { 'text-anchor': 'middle', 'font-size': 14 }); txt(g, 255, top + 114, '+ Predictor, with q', { 'text-anchor': 'middle', 'font-size': 14 });
    arrow(g, `M154 ${top + 100} H174`);
    const yhat = group(g, 'fade');
    arrow(yhat, `M334 ${top + 100} H356`);
    image(yhat, vectorURL(T.predictor.vec), 362, top + 88, 180, 22); el('rect', { x: 362, y: top + 88, width: 180, height: 22, fill: 'none', class: 'thin' }, yhat);
    txt(yhat, 452, top + 78, 'ŷ  (guess)', { 'text-anchor': 'middle', 'font-size': 15, class: 't-red' });
    const ybar = group(g, 'fade');
    box(ybar, 30, top + 200, 120, 70, '#fff'); txt(ybar, 90, top + 232, '“A brown dog', { 'text-anchor': 'middle', 'font-size': 12 }); txt(ybar, 90, top + 250, 'on the beach…”', { 'text-anchor': 'middle', 'font-size': 12 });
    el('rect', { x: 180, y: top + 200, width: 150, height: 70, rx: 9, fill: '#fff', stroke: 'var(--ink)', 'stroke-width': 2, 'stroke-dasharray': '7 5' }, ybar);
    txt(ybar, 255, top + 232, 'slow Text Encoder', { 'text-anchor': 'middle', 'font-size': 14 }); txt(ybar, 255, top + 252, 'average → length 1', { 'text-anchor': 'middle', 'font-size': 12, class: 't-muted' });
    arrow(ybar, `M154 ${top + 235} H174`); arrow(ybar, `M334 ${top + 235} H356`);
    image(ybar, vectorURL(T.text.vec), 362, top + 224, 180, 22); el('rect', { x: 362, y: top + 224, width: 180, height: 22, fill: 'none', class: 'thin' }, ybar);
    txt(ybar, 452, top + 274, 'ȳ  (TARGET, from the slow copy)', { 'text-anchor': 'middle', 'font-size': 15, class: 't-red' });
    // the angle between them
    const ang = group(g, 'fade');
    const cx = 600, cy = top + 250, R = 90, th = Math.acos(Math.max(-1, Math.min(1, T.score)));
    el('path', { d: `M${cx} ${cy} L${cx + R} ${cy}`, class: 'arrow', 'marker-end': 'url(#head)' }, ang);
    el('path', { d: `M${cx} ${cy} L${cx + R * Math.cos(th)} ${cy - R * Math.sin(th)}`, class: 'arrow', 'marker-end': 'url(#head)' }, ang);
    el('path', { d: `M${cx + 34} ${cy} A34 34 0 0 0 ${cx + 34 * Math.cos(th)} ${cy - 34 * Math.sin(th)}`, fill: 'none', stroke: 'var(--yellow)', 'stroke-width': 3 }, ang);
    txt(ang, cx + R + 8, cy + 5, 'ȳ', { 'font-size': 16 });
    txt(ang, cx + R * Math.cos(th) + 6, cy - R * Math.sin(th) - 6, 'ŷ', { 'font-size': 16 });
    txt(ang, cx + 20, cy + 34, `1 − cos = ${(1 - T.score).toFixed(2)}`, { 'font-size': 16 });
    txt(ang, cx + 20, cy + 54, '(v1’s angle, for scale)', { 'font-size': 12, class: 't-muted' });
    scenes['game-bridge'] = {
      g, view: [10, top + 20, 720, 330],
      beats: ['The photo, with its holes', 'v1’s Predictor and q turn the context into a guess ŷ', 'The slow Text Encoder turns the whole caption into ȳ', 'Graded by the angle between them'],
      show(b) { show(yhat, b >= 1); show(ybar, b >= 2); show(ang, b >= 3); }
    };
  }

  // ================================================================================================ the total, and collapse
  {
    const g = group(world);
    const top = 4400;
    const parts = [['image', 0.42, 'var(--accent)'], ['text', 0.31, 'var(--neg)'], ['bridge', 0.27, 'var(--grad)']];
    let x = 40;
    const bars = parts.map(([name, w, color], i) => {
      const bg = group(g, 'fade');
      el('rect', { x, y: top + 40, width: w * 420, height: 34, fill: color, opacity: .8 }, bg);
      txt(bg, x + (w * 420) / 2, top + 63, `L_${name}`, { 'text-anchor': 'middle', 'font-size': 15, fill: '#fff' });
      bg.style.transitionDelay = `${i * 0.2}s`;
      x += w * 420 + 2;
      return bg;
    });
    const sum = txt(g, x + 14, top + 64, '= L', { 'font-size': 20, class: 'fade' });
    txt(g, 40, top + 26, 'one photo at a time, no batch table', { 'font-size': 15, class: 't-dim' });
    // a cloud of slow-copy vectors (drawn in 2-D): spread out when healthy, one dot when collapsed
    const r = rng(5);
    const pts = Array.from({ length: 46 }, () => [r() * 2 - 1, r() * 2 - 1]);
    const cloud = (cx, title, spread, colour) => {
      const cg = group(g, 'fade');
      el('rect', { x: cx - 110, y: top + 130, width: 220, height: 200, rx: 9, fill: '#fff', class: 'thin' }, cg);
      txt(cg, cx, top + 120, title, { 'text-anchor': 'middle', 'font-size': 16 });
      const dots = pts.map(([px, py]) => el('circle', { cx: cx + px * 80 * spread, cy: top + 230 + py * 70 * spread, r: 4.5, fill: colour, opacity: .75, style: 'transition: cx 1.2s, cy 1.2s' }, cg));
      return { cg, dots, cx };
    };
    const ok = cloud(150, 'healthy: spread out', 1, 'var(--accent)');
    const bad = cloud(430, 'collapsed: one point', 1, 'var(--grad)');
    const meter = group(g, 'fade');
    txt(meter, 290, top + 360, 'log the spread every epoch; if it heads to 0, stop', { 'text-anchor': 'middle', 'font-size': 15, class: 't-red' });
    scenes['v2-loss'] = {
      g, view: [20, top, 580, 380],
      beats: ['Three losses, added', 'Healthy: different photos, different vectors', 'Collapsed: everything at one point, loss zero, nothing learned', 'The check'],
      show(b) {
        bars.forEach(e => show(e, true)); show(sum, true);
        show(ok.cg, b >= 1); show(bad.cg, b >= 2); show(meter, b >= 3);
        bad.dots.forEach((d, i) => { d.setAttribute('cx', b >= 2 ? bad.cx + 2 : bad.cx + pts[i][0] * 80); d.setAttribute('cy', b >= 2 ? top + 230 : top + 230 + pts[i][1] * 70); });
      }
    };
  }

  // ================================================================================================ the plan
  {
    const g = group(world);
    const top = 5000;
    const F = M.final;
    const rows = [['', 'v1 (trained)', 'v2 (planned)'], ['image → text R@1', `${(F.i2t['1'] * 100).toFixed(1)}%`, '—'], ['image → text R@5', `${(F.i2t['5'] * 100).toFixed(1)}%`, '—'], ['text → image R@1', `${(F.t2i['1'] * 100).toFixed(1)}%`, '—'], ['training photos', '29,000', '29,000'], ['needs a big batch', 'yes', 'no']];
    const table = rows.map((row, i) => {
      const rg = group(g, 'fade');
      row.forEach((cell, j) => txt(rg, [40, 300, 470][j], top + 40 + i * 36, cell, { 'font-size': i ? 17 : 15, class: i === 0 ? 't-muted' : j === 2 ? 't-accent' : '' }));
      if (i) el('line', { x1: 40, y1: top + 50 + i * 36, x2: 600, y2: top + 50 + i * 36, stroke: '#ecebe6' }, rg);
      rg.style.transitionDelay = `${i * 0.08}s`;
      return rg;
    });
    const foot = txt(g, 40, top + 280, 'Same 1,014 validation photos, same 5,070 captions, chance = 0.1%', { 'font-size': 14, class: 't-dim fade' });
    scenes['v2-plan'] = {
      g, view: [20, top, 600, 300],
      beats: ['Same test as v1, side by side'],
      show() { table.forEach(r => show(r, true)); show(foot, true); }
    };
  }

  return scenes;
}
