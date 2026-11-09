// v2's forward pass, laid out like v1's map: the photo lane on the left, the caption lane on the right, the predictor
// and the loss below. Every picture comes from data/v2trace.json, recorded by scripts/export_v2.py from the frozen
// V-JEPA 2, the cached EmbeddingGemma targets and the predictor checkpoint saved during training.
import { el, txt, group, box, chip, arrow, image, matrixURL, vectorURL, show } from './stage.js';

const C = { pink: 'var(--x-pink)', green: 'var(--x-green)', orange: 'var(--x-orange)', orange2: 'var(--x-orange-2)', purple: 'var(--x-purple)', blue: 'var(--x-blue)', yellow: 'var(--x-yellow)', grey: '#e9e8e3' };

// a small padlock: frozen, never trained
function lock(parent, x, y, s = 1) {
  const g = group(parent);
  el('path', { d: `M${x + 4 * s} ${y + 8 * s} v${-3 * s} a${4 * s} ${4 * s} 0 0 1 ${8 * s} 0 v${3 * s}`, fill: 'none', stroke: 'var(--ink)', 'stroke-width': 1.6 }, g);
  el('rect', { x, y: y + 8 * s, width: 16 * s, height: 11 * s, rx: 2, fill: C.blue, stroke: 'var(--ink)', 'stroke-width': 1.4 }, g);
  return g;
}

export function buildV2(world, data) {
  const T = data.v2, S = data.v2sprites, T1 = data.trace;
  const INPUT = 'data/' + T.input;
  const vis = new Set(T.visible_cells), patches = T.visible_patches, k = T.visible_cells.length;
  const tokURL = (sp, rows, cols = 256) => matrixURL(rows, cols, (r, c) => (sp.at(c, r) - 127.5) / 127.5);
  const qMean = T.q_attn.map(block => Array.from({ length: k }, (_, i) => block.reduce((s, head) => s + head[i + 1], 0) / block.length));
  const scenes = {};
  const cellXY = (c, x, y, cs) => [x + (c % 8) * cs, y + Math.floor(c / 8) * cs];

  // the photo with the 8 × 8 cell grid and the hidden cells greyed out
  function maskedPhoto(parent, x, y, size, { grid = true } = {}) {
    const g = group(parent);
    image(g, INPUT, x, y, size, size, { 'image-rendering': 'auto' });
    const cs = size / 8;
    const holes = group(g, 'fade');
    for (let c = 0; c < 64; c++) if (!vis.has(c)) { const [cx, cy] = cellXY(c, x, y, cs); el('rect', { x: cx, y: cy, width: cs, height: cs, fill: '#3d3d39', opacity: .86 }, holes); }
    if (grid) for (let i = 1; i < 8; i++) {
      el('line', { x1: x + i * cs, y1: y, x2: x + i * cs, y2: y + size, stroke: '#ffffffaa', 'stroke-width': 1 }, g);
      el('line', { x1: x, y1: y + i * cs, x2: x + size, y2: y + i * cs, stroke: '#ffffffaa', 'stroke-width': 1 }, g);
    }
    el('rect', { x, y, width: size, height: size, fill: 'none', class: 'box' }, g);
    return { g, holes };
  }

  // ================================================================================================ overview
  {
    const g = group(world);
    const y0 = -600;
    maskedPhoto(g, 30, y0 + 40, 150, { grid: false });
    chip(g, 105, y0 + 22, 'Image, with holes');
    box(g, 220, y0 + 55, 170, 120, C.pink); txt(g, 305, y0 + 108, 'V-JEPA 2', { 'text-anchor': 'middle', 'font-size': 19 }); txt(g, 305, y0 + 132, 'ViT-L · 304M', { 'text-anchor': 'middle', 'font-size': 14, class: 't-dim' });
    lock(g, 365, y0 + 40);
    arrow(g, `M184 ${y0 + 115} H214`);
    const pred = group(g, 'fade');
    arrow(pred, `M394 ${y0 + 115} H424`);
    box(pred, 430, y0 + 55, 150, 120, C.orange2); txt(pred, 505, y0 + 108, 'Predictor', { 'text-anchor': 'middle', 'font-size': 19 }); txt(pred, 505, y0 + 132, 'trained · 11.3M', { 'text-anchor': 'middle', 'font-size': 14, class: 't-accent' });
    arrow(pred, `M584 ${y0 + 115} H614`);
    image(pred, vectorURL(T.pred), 620, y0 + 103, 260, 24); el('rect', { x: 620, y: y0 + 103, width: 260, height: 24, fill: 'none', class: 'thin' }, pred);
    txt(pred, 750, y0 + 94, 'guess ŷ · 768 numbers', { 'text-anchor': 'middle', 'font-size': 15, class: 't-red' });
    const textSide = group(g, 'fade');
    box(textSide, 30, y0 + 260, 150, 110, '#fff');
    ['“The yellow dog walks…”', '“A brown dog on…”', '“A dog runs in the…”', '“A yellow dog…”', '“A tan dog runs…”'].forEach((c, i) => txt(textSide, 42, y0 + 282 + i * 19, c, { 'font-size': 12 }));
    chip(textSide, 105, y0 + 242, '5 captions');
    arrow(textSide, `M184 ${y0 + 315} H214`);
    box(textSide, 220, y0 + 260, 170, 110, C.pink); txt(textSide, 305, y0 + 310, 'EmbeddingGemma', { 'text-anchor': 'middle', 'font-size': 17 }); txt(textSide, 305, y0 + 334, '300M', { 'text-anchor': 'middle', 'font-size': 14, class: 't-dim' });
    lock(textSide, 365, y0 + 245);
    arrow(textSide, `M394 ${y0 + 315} H424`);
    box(textSide, 430, y0 + 285, 150, 60, '#fff'); txt(textSide, 505, y0 + 322, 'average of 5', { 'text-anchor': 'middle', 'font-size': 16 });
    arrow(textSide, `M584 ${y0 + 315} H614`);
    image(textSide, vectorURL(T.target), 620, y0 + 303, 260, 24); el('rect', { x: 620, y: y0 + 303, width: 260, height: 24, fill: 'none', class: 'thin' }, textSide);
    txt(textSide, 750, y0 + 356, 'TARGET y · 768 numbers', { 'text-anchor': 'middle', 'font-size': 15, class: 't-red' });
    const loss = group(g, 'fade');
    el('path', { d: `M750 ${y0 + 132} V${y0 + 182} M750 ${y0 + 298} V${y0 + 250}`, class: 'arrow flow' }, loss);
    el('rect', { x: 655, y: y0 + 188, width: 190, height: 56, rx: 10, fill: C.yellow, class: 'box' }, loss);
    txt(loss, 750, y0 + 223, '‖ŷ − y‖²', { 'text-anchor': 'middle', 'font-size': 22 });
    const key = group(g, 'fade');
    lock(key, 30, y0 + 400, 0.8); txt(key, 50, y0 + 414, 'frozen: never trained', { 'font-size': 14, class: 't-dim' });
    el('rect', { x: 230, y: y0 + 402, width: 14, height: 14, rx: 3, fill: C.orange2, class: 'thin' }, key); txt(key, 252, y0 + 414, 'the only part that learns', { 'font-size': 14, class: 't-dim' });
    scenes['v2-overview'] = {
      g, view: [10, y0, 890, 430],
      beats: ['A photo with holes goes through a frozen V-JEPA 2', 'Only the Predictor learns: it guesses 768 numbers', 'Its five captions go through a frozen EmbeddingGemma, then get averaged', 'The loss is the squared distance between the two'],
      show(b) { show(pred, b >= 1); show(textSide, b >= 2); show(loss, b >= 3); show(key, b >= 1); }
    };
  }

  // ================================================================================================ 256 px, patches, two frames
  {
    const g = group(world);
    const top = 0, P = 320;
    image(g, INPUT, 30, top + 40, P, P, { 'image-rendering': 'auto' });
    el('rect', { x: 30, y: top + 40, width: P, height: P, fill: 'none', class: 'box' }, g);
    chip(g, 190, top + 20, 'Image · 256 × 256');
    const grid = group(g, 'fade');
    for (let i = 1; i < 16; i++) {
      el('line', { x1: 30 + i * P / 16, y1: top + 40, x2: 30 + i * P / 16, y2: top + 40 + P, stroke: '#ffffffb0', 'stroke-width': .8 }, grid);
      el('line', { x1: 30, y1: top + 40 + i * P / 16, x2: 30 + P, y2: top + 40 + i * P / 16, stroke: '#ffffffb0', 'stroke-width': .8 }, grid);
    }
    el('rect', { x: 30 + 7 * P / 16, y: top + 40 + 5 * P / 16, width: P / 16, height: P / 16, class: 'q-cell' }, grid);
    txt(grid, 190, top + 386, '16 × 16 = 256 patches of 16 × 16 px', { 'text-anchor': 'middle', 'font-size': 15 });
    const frames = group(g, 'fade');
    [1, 0].forEach(i => {
      image(frames, INPUT, 400 + i * 22, top + 70 - i * 22, 150, 150, { 'image-rendering': 'auto' });
      el('rect', { x: 400 + i * 22, y: top + 70 - i * 22, width: 150, height: 150, fill: 'none', class: 'thin' }, frames);
    });
    txt(frames, 486, top + 252, 'the same photo, twice:', { 'text-anchor': 'middle', 'font-size': 14 });
    txt(frames, 486, top + 272, 'a 2-frame clip for a video model', { 'text-anchor': 'middle', 'font-size': 14, class: 't-dim' });
    const v1 = group(g, 'fade');
    image(v1, 'data/' + T1.image.input, 420, top + 300, 64, 64);
    el('rect', { x: 420, y: top + 300, width: 64, height: 64, fill: 'none', class: 'thin' }, v1);
    txt(v1, 494, top + 324, 'v1 saw this:', { 'font-size': 13, class: 't-muted' }); txt(v1, 494, top + 342, '128 × 128', { 'font-size': 13, class: 't-muted' });
    scenes['v2-input'] = {
      g, view: [10, top, 600, 400],
      beats: ['The photo at 256 × 256', 'Cut into 256 patches', 'Shown as a two-frame clip', 'Four times v1’s pixels'],
      show(b) { show(grid, b >= 1); show(frames, b >= 2); show(v1, b >= 3); }
    };
  }

  // ================================================================================================ hiding
  {
    const g = group(world);
    const top = 460, P = 320, cs = P / 8;
    image(g, INPUT, 30, top + 30, P, P, { 'image-rendering': 'auto' });
    for (let i = 1; i < 8; i++) {
      el('line', { x1: 30 + i * cs, y1: top + 30, x2: 30 + i * cs, y2: top + 30 + P, stroke: '#ffffffcc', 'stroke-width': 1.2 }, g);
      el('line', { x1: 30, y1: top + 30 + i * cs, x2: 30 + P, y2: top + 30 + i * cs, stroke: '#ffffffcc', 'stroke-width': 1.2 }, g);
    }
    const blocks = T.blocks.map(([bx, by, w, h], i) => {
      const bg = group(g, 'fade');
      el('rect', { x: 30 + bx * cs, y: top + 30 + by * cs, width: w * cs, height: h * cs, fill: '#3d3d39', opacity: .55 }, bg);
      el('rect', { x: 30 + bx * cs + 1.5, y: top + 30 + by * cs + 1.5, width: w * cs - 3, height: h * cs - 3, fill: 'none', stroke: 'var(--x-red)', 'stroke-width': 3 }, bg);
      txt(bg, 30 + (bx + w / 2) * cs, top + 30 + (by + h / 2) * cs + 6, `${i + 1}`, { 'text-anchor': 'middle', 'font-size': 22, fill: '#fff' });
      return bg;
    });
    el('rect', { x: 30, y: top + 30, width: P, height: P, fill: 'none', class: 'box' }, g);
    const union = group(g, 'fade');
    for (let c = 0; c < 64; c++) if (!vis.has(c)) { const [x, y] = cellXY(c, 30, top + 30, cs); el('rect', { x, y, width: cs, height: cs, fill: '#3d3d39', opacity: .9 }, union); }
    const note = group(g, 'fade');
    txt(note, 380, top + 80, `${64 - k} cells hidden`, { 'font-size': 20, class: 't-red' });
    txt(note, 380, top + 108, `${k} cells left`, { 'font-size': 20, class: 't-accent' });
    txt(note, 380, top + 140, 'blocks overlap; each', { 'font-size': 14, class: 't-dim' });
    txt(note, 380, top + 160, 'is 15–20% of the grid', { 'font-size': 14, class: 't-dim' });
    txt(note, 380, top + 200, 'a new draw every step,', { 'font-size': 14, class: 't-dim' });
    txt(note, 380, top + 220, 'for every photo', { 'font-size': 14, class: 't-dim' });
    const cellNote = txt(g, 190, top + 376, '8 × 8 cells, each 2 × 2 patches', { 'text-anchor': 'middle', 'font-size': 15, class: 'fade' });
    scenes['v2-mask'] = {
      g, view: [10, top, 580, 400],
      beats: ['8 × 8 cells, 2 × 2 patches each', 'Block 1', 'Block 2', 'Block 3', 'Block 4', `What is left: ${k} of 64 cells`],
      show(b) { blocks.forEach((bg, i) => show(bg, b >= i + 1 && b < 5)); show(union, b >= 5); show(note, b >= 5); show(cellNote, b === 0); }
    };
  }

  // ================================================================================================ frozen encoder, layer by layer
  {
    const g = group(world);
    const top = 920, P = 320, ps = P / 16;
    image(g, INPUT, 30, top + 40, P, P, { 'image-rendering': 'auto' });
    const hidden = group(g);
    for (let p = 0; p < 256; p++) if (!patches.includes(p)) el('rect', { x: 30 + (p % 16) * ps, y: top + 40 + Math.floor(p / 16) * ps, width: ps, height: ps, fill: '#3d3d39', opacity: .9 }, hidden);
    const heat = patches.map(p => el('rect', { x: 30 + (p % 16) * ps, y: top + 40 + Math.floor(p / 16) * ps, width: ps, height: ps, class: 'cell-hot', opacity: 0, style: 'transition: opacity .5s' }, g));
    el('rect', { x: 30, y: top + 40, width: P, height: P, fill: 'none', class: 'box' }, g);
    txt(g, 190, top + 26, `${patches.length} visible patches go in`, { 'text-anchor': 'middle', 'font-size': 15 });
    // the 24 layers, as one tall frozen box
    box(g, 380, top + 40, 150, 320, C.pink);
    txt(g, 455, top + 28, 'V-JEPA 2 · 24 layers', { 'text-anchor': 'middle', 'font-size': 14 });
    lock(g, 512, top + 6);
    const layers = Array.from({ length: 24 }, (_, i) => el('rect', { x: 392, y: top + 50 + i * 12.9, width: 126, height: 10, rx: 2, fill: C.orange, stroke: 'var(--ink)', 'stroke-width': .8, style: 'transition: fill .3s, opacity .3s' }, g));
    const ln = group(g, 'fade');
    box(ln, 380, top + 372, 150, 28, C.purple); txt(ln, 455, top + 391, 'LayerNorm', { 'text-anchor': 'middle', 'font-size': 14 });
    const layerLabel = txt(g, 190, top + 386, '', { 'text-anchor': 'middle', 'font-size': 15, class: 't-dim' });
    const stops = [0, 4, 8, 12, 16, 20, 24];
    const normAt = (l, i) => S.norms.at(i, l) / 255;
    scenes['v2-encode'] = {
      g, view: [10, top, 560, 420], beatMs: 1500,
      beats: ['Patch embedding: the hidden patches are dropped here', ...stops.slice(1).map(l => `After layer ${l}: how strongly each patch fires`), 'LayerNorm: 132 tokens of 1,024 numbers'],
      show(b) {
        const l = stops[Math.min(b, stops.length - 1)];
        layers.forEach((r, i) => { r.setAttribute('fill', i < l ? (i === l - 1 ? 'var(--yellow)' : C.orange) : '#f3e3cf'); });
        heat.forEach((r, i) => r.setAttribute('opacity', (0.78 * normAt(l, i)).toFixed(3)));
        layerLabel.textContent = l === 0 ? 'before the first layer' : `after layer ${l} of 24`;
        show(ln, b >= stops.length);
      }
    };
  }

  // ================================================================================================ pooling to cells
  {
    const g = group(world);
    const top = 1380, P = 240, cs = P / 8;
    const left = maskedPhoto(g, 30, top + 40, P);
    // 2 × 2 patch squares inside one visible cell, merging
    const focus = T.visible_cells[Math.floor(k / 2)];
    const [fx, fy] = cellXY(focus, 30, top + 40, cs);
    const quad = group(g, 'fade');
    [0, 1, 2, 3].forEach(i => el('rect', { x: fx + (i % 2) * cs / 2, y: fy + Math.floor(i / 2) * cs / 2, width: cs / 2, height: cs / 2, fill: 'none', stroke: 'var(--yellow)', 'stroke-width': 2 }, quad));
    const one = el('rect', { x: fx, y: fy, width: cs, height: cs, class: 'q-cell fade' }, g);
    txt(g, 150, top + 26, '4 patch vectors → 1 cell vector', { 'text-anchor': 'middle', 'font-size': 15 });
    const before = group(g, 'fade');
    image(before, tokURL(S.patchTokens, patches.length), 300, top + 40, 110, 264);
    el('rect', { x: 300, y: top + 40, width: 110, height: 264, fill: 'none', class: 'thin' }, before);
    txt(before, 355, top + 322, `${patches.length} patches`, { 'text-anchor': 'middle', 'font-size': 13 });
    const after = group(g, 'fade');
    arrow(after, `M416 ${top + 172} H436`);
    image(after, tokURL(S.cellTokens, k), 442, top + 106, 110, 132);
    el('rect', { x: 442, y: top + 106, width: 110, height: 132, fill: 'none', class: 'box' }, after);
    txt(after, 497, top + 256, `${k} cell tokens`, { 'text-anchor': 'middle', 'font-size': 14, class: 't-red' });
    txt(after, 497, top + 274, '× 1,024 numbers', { 'text-anchor': 'middle', 'font-size': 12, class: 't-muted' });
    scenes['v2-pool'] = {
      g, view: [10, top, 560, 340],
      beats: ['The encoder’s 132 patch tokens', 'Each cell averages its 4 patches', `${k} cell tokens, one per visible cell`],
      show(b) { show(before, true); show(quad, b === 1); show(one, b >= 2); show(after, b >= 2); left.holes.style.opacity = 1; }
    };
  }

  // ================================================================================================ captions → one target
  {
    const g = group(world);
    const top = 0, x0 = 640;
    box(g, x0, top + 40, 340, 170, '#fff');
    chip(g, x0 + 170, top + 22, 'Captions');
    const lines = T.captions.map((c, i) => {
      const t = txt(g, x0 + 14, top + 72 + i * 30, `${i + 1}. ${c.length > 38 ? c.slice(0, 36) + '…' : c}`, { 'font-size': 14, class: 'fade' });
      t.style.transitionDelay = `${i * .1}s`;
      return t;
    });
    // word pieces of caption 1, as EmbeddingGemma reads it
    const pieces = group(g, 'fade');
    let x = x0, y = top + 250;
    T.pieces[0].forEach(p => {
      const label = p.replace(/^▁/, '·');
      const w = label.length * 8.4 + 12;
      if (x + w > x0 + 340) { x = x0; y += 34; }
      el('rect', { x, y: y - 17, width: w, height: 24, rx: 4, fill: p.startsWith('▁') ? '#fff' : C.yellow, class: 'chip-rect' }, pieces);
      txt(pieces, x + w / 2, y, label, { 'text-anchor': 'middle', 'font-size': 14 });
      x += w + 5;
    });
    txt(pieces, x0, y + 30, 'word pieces; · marks the start of a word', { 'font-size': 12, class: 't-muted' });
    const enc = group(g, 'fade');
    box(enc, x0 + 60, top + 360, 220, 60, C.pink); txt(enc, x0 + 170, top + 396, 'EmbeddingGemma · 300M', { 'text-anchor': 'middle', 'font-size': 15 });
    lock(enc, x0 + 262, top + 344);
    const vecs = group(g, 'fade');
    arrow(vecs, `M${x0 + 170} ${top + 424} V${top + 446}`);
    image(vecs, tokURL(S.captions, 5, 768), x0 + 20, top + 452, 300, 60);
    el('rect', { x: x0 + 20, y: top + 452, width: 300, height: 60, fill: 'none', class: 'thin' }, vecs);
    txt(vecs, x0 + 330, top + 486, '5 × 768', { 'font-size': 13, class: 't-dim' });
    const avg = group(g, 'fade');
    [0, 1, 2, 3, 4].forEach(i => el('path', { d: `M${x0 + 170} ${top + 458 + i * 12} C ${x0 + 120} ${top + 540}, ${x0 + 170} ${top + 540}, ${x0 + 170} ${top + 552}`, fill: 'none', stroke: 'var(--muted)', 'stroke-width': 1 }, avg));
    image(avg, vectorURL(T.target), x0 + 20, top + 560, 300, 22);
    el('rect', { x: x0 + 20, y: top + 560, width: 300, height: 22, fill: 'none', class: 'box' }, avg);
    txt(avg, x0 + 170, top + 606, 'TARGET · their average', { 'text-anchor': 'middle', 'font-size': 17, class: 't-red' });
    scenes['v2-text'] = {
      g, view: [x0 - 20, top, 380, 620],
      beats: ['The photo’s five captions', 'Cut into word pieces', 'Through a frozen EmbeddingGemma', 'Five vectors of 768 numbers', 'Averaged into one TARGET'],
      show(b) { lines.forEach(l => show(l, true)); show(pieces, b === 1 || b === 2); show(enc, b >= 2); show(vecs, b >= 3); show(avg, b >= 4); }
    };
  }

  // ================================================================================================ predictor
  {
    const g = group(world);
    const top = 1800, P = 240, cs = P / 8;
    const cols = group(g);
    image(cols, tokURL(S.cellTokens, k), 30, top + 70, 80, 165);
    el('rect', { x: 30, y: top + 70, width: 80, height: 165, fill: 'none', class: 'thin' }, cols);
    txt(cols, 70, top + 58, `${k} × 1,024`, { 'text-anchor': 'middle', 'font-size': 13 });
    const proj = group(g, 'fade');
    box(proj, 122, top + 70, 34, 165, C.orange); txt(proj, 144, top + 152, 'Linear(1024, 384)', { 'font-size': 12, transform: `rotate(-90 144 ${top + 152})`, 'text-anchor': 'middle' });
    box(proj, 162, top + 70, 22, 165, C.blue); txt(proj, 177, top + 152, '+ 2-D sin-cos', { 'font-size': 11, transform: `rotate(-90 177 ${top + 152})`, 'text-anchor': 'middle' });
    const q = group(g, 'fade');
    el('rect', { x: 30, y: top + 44, width: 80, height: 10, rx: 2, fill: 'var(--yellow)', class: 'thin' }, q);
    txt(q, 70, top + 36, 'q first', { 'text-anchor': 'middle', 'font-size': 14 });
    const stem = group(g, 'fade');
    box(stem, 196, top + 44, 104, 300, C.orange2);
    txt(stem, 248, top + 34, '6 blocks', { 'text-anchor': 'middle', 'font-size': 14 });
    const chips = Array.from({ length: 6 }, (_, i) => {
      const r = el('rect', { x: 206, y: top + 54 + i * 47, width: 84, height: 38, rx: 6, fill: C.orange, class: 'box', style: 'transition: fill .3s' }, stem);
      txt(stem, 248, top + 79 + i * 47, i ? `B${i + 1}` : 'Block 1', { 'text-anchor': 'middle', 'font-size': 14 });
      return r;
    });
    const look = group(g, 'fade');
    const ph = maskedPhoto(look, 320, top + 44, P, { grid: false });
    const heat = T.visible_cells.map(c => { const [x, y] = cellXY(c, 320, top + 44, cs); return el('rect', { x, y, width: cs, height: cs, class: 'cell-hot', opacity: 0, style: 'transition: opacity .5s' }, look); });
    txt(look, 440, top + 34, 'where q looks', { 'text-anchor': 'middle', 'font-size': 14 });
    const out = group(g, 'fade');
    box(out, 196, top + 356, 104, 26, C.purple); txt(out, 248, top + 374, 'LayerNorm', { 'text-anchor': 'middle', 'font-size': 13 });
    box(out, 196, top + 388, 104, 26, C.orange); txt(out, 248, top + 406, 'Linear(384, 768)', { 'text-anchor': 'middle', 'font-size': 12 });
    arrow(out, `M302 ${top + 401} H322`);
    image(out, vectorURL(T.pred), 328, top + 390, 232, 22); el('rect', { x: 328, y: top + 390, width: 232, height: 22, fill: 'none', class: 'box' }, out);
    txt(out, 444, top + 436, 'guess ŷ · 768 numbers', { 'text-anchor': 'middle', 'font-size': 16, class: 't-red' });
    scenes['v2-predictor'] = {
      g, view: [10, top, 570, 450], beatMs: 1600,
      beats: [`The ${k} cell tokens`, 'Down to 384 numbers, plus a code for each cell’s row and column', 'q goes in front', ...Array.from({ length: 6 }, (_, i) => `Block ${i + 1}: where q looks`), 'Keep q → LayerNorm → Linear(384, 768)'],
      show(b) {
        show(proj, b >= 1); show(q, b >= 2); show(stem, b >= 2); show(look, b >= 3); show(out, b >= 9);
        const blk = Math.min(Math.max(b - 3, 0), 5), on = b >= 3 && b <= 8;
        chips.forEach((r, i) => r.setAttribute('fill', on && i === blk ? 'var(--yellow)' : C.orange));
        const m = Math.max(...qMean[blk]);
        heat.forEach((r, i) => r.setAttribute('opacity', b >= 3 ? (0.8 * qMean[blk][i] / m).toFixed(3) : 0));
        ph.holes.style.opacity = 1;
      }
    };
  }

  // ================================================================================================ loss
  {
    const g = group(world);
    const top = 2320, x0 = 30, W = 520;
    const sc = Math.max(...T.pred.map(Math.abs), ...T.target.map(Math.abs));
    const strip = (y, vec, label, cls, scale) => {
      const sg = group(g, 'fade');
      txt(sg, x0, y - 8, label, { 'font-size': 15, class: cls });
      image(sg, vectorURL(vec, scale), x0, y, W, 24); el('rect', { x: x0, y, width: W, height: 24, fill: 'none', class: 'thin' }, sg);
      return sg;
    };
    const a = strip(top + 40, T.pred, 'guess ŷ (predictor, epoch ' + T.epoch + ')', 't-red', sc);
    const b2 = strip(top + 100, T.target, 'TARGET y (frozen EmbeddingGemma, averaged)', 't-red', sc);
    const lockG = group(g, 'fade'); lock(lockG, x0 + W + 10, top + 102, .9);
    const d = group(g, 'fade');
    txt(d, x0, top + 162, '(ŷᵢ − yᵢ)², all 768 of them', { 'font-size': 14, class: 't-dim' });
    const mx = Math.max(...T.sq);
    T.sq.forEach((v, i) => el('rect', { x: x0 + (i / 768) * W, y: top + 230 - 50 * Math.sqrt(v / mx), width: W / 768 + .3, height: 50 * Math.sqrt(v / mx), fill: 'var(--grad)', opacity: .8 }, d));
    el('line', { x1: x0, y1: top + 230, x2: x0 + W, y2: top + 230, class: 'thin' }, d);
    const sum = group(g, 'fade');
    el('rect', { x: x0 + 120, y: top + 252, width: 280, height: 50, rx: 10, fill: C.yellow, class: 'box' }, sum);
    txt(sum, x0 + 260, top + 284, `add them up = ${T.loss.toFixed(1)}`, { 'text-anchor': 'middle', 'font-size': 20 });
    txt(sum, x0 + 260, top + 322, 'this photo, this mask, the epoch-' + T.epoch + ' checkpoint', { 'text-anchor': 'middle', 'font-size': 12, class: 't-muted' });
    const why = group(g, 'fade');
    txt(why, x0, top + 356, 'No other captions, no table: the target is frozen,', { 'font-size': 15, class: 't-accent' });
    txt(why, x0, top + 376, 'so the guess can’t win by dragging it somewhere easy.', { 'font-size': 15, class: 't-accent' });
    scenes['v2-loss'] = {
      g, view: [10, top, 580, 390],
      beats: ['The guess', 'The target, which never moves', 'Squared difference, number by number', 'Summed: the loss', 'Why there are no wrong answers'],
      show(b) { show(a, true); show(b2, b >= 1); show(lockG, b >= 1); show(d, b >= 2); show(sum, b >= 3); show(why, b >= 4); }
    };
  }

  // ================================================================================================ scale
  {
    const g = group(world);
    const top = 2780, x0 = 160, W = 380;
    const logBar = (v, lo, hi) => W * (Math.log10(v) - lo) / (hi - lo);
    const rows = (title, y, items, lo, hi) => {
      const rg = group(g, 'fade');
      txt(rg, 30, y, title, { 'font-size': 17 });
      items.forEach(([name, v, label, color], i) => {
        const yy = y + 22 + i * 34;
        txt(rg, x0 - 12, yy + 17, name, { 'text-anchor': 'end', 'font-size': 15 });
        el('rect', { x: x0, y: yy, width: Math.max(3, logBar(v, lo, hi)), height: 24, rx: 3, fill: color, class: 'thin' }, rg);
        txt(rg, x0 + Math.max(3, logBar(v, lo, hi)) + 8, yy + 17, label, { 'font-size': 14, class: 't-dim' });
      });
      return rg;
    };
    const params = rows('Weights that learn (log scale)', top + 30, [['v1', 14.1e6, '14.1M', '#d9d8d2'], ['v2', 11.3e6, '11.3M', C.orange2], ['VL-JEPA', 790e6, '≈ 790M', C.pink]], 6, 9.2);
    const seen = rows('Pairs seen while training (log scale)', top + 190, [['v1', 5.8e6, '5.8M', '#d9d8d2'], ['v2', 0.87e6, '0.9M', C.orange2], ['VL-JEPA', 3.4e9, '3.3–3.6B', C.pink]], 5.5, 9.9);
    const hw = group(g, 'fade');
    txt(hw, 30, top + 350, 'v1 and v2: one Mac.   VL-JEPA: 192 H200 GPUs for about 4 weeks.', { 'font-size': 15, class: 't-dim' });
    scenes['v2-scale'] = {
      g, view: [10, top, 600, 370],
      beats: ['How many weights learn', 'How much data they see', 'And on what'],
      show(b) { show(params, true); show(seen, b >= 1); show(hw, b >= 2); }
    };
  }

  return scenes;
}
