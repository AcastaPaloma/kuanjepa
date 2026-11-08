// v1's forward pass, laid out as a map. Left lane (x 0–500): the photo through the Image Encoder and the Predictor.
// Right lane (x 500–1000): the caption through the Text Encoder. They meet at the bottom for the score and the loss.
// Every picture of numbers comes from data/trace.json and data/trace/*, recorded from the trained checkpoint by
// scripts/export.py: the same photo, the same caption, the same weights as the Results tab.
import { el, txt, group, box, chip, arrow, image, matrixURL, vectorURL, cropURL, show, toggle } from './stage.js';

const C = {
  pink: 'var(--x-pink)', green: 'var(--x-green)', green2: 'var(--x-green-2)', orange: 'var(--x-orange)',
  orange2: 'var(--x-orange-2)', purple: 'var(--x-purple)', blue: 'var(--x-blue)', yellow: 'var(--x-yellow)'
};
const fmt = (v, d = 2) => (v < 0 ? '−' : '') + Math.abs(v).toFixed(d);

// wrap a caption into lines of at most n characters
const wrap = (s, n) => s.split(' ').reduce((lines, w) => {
  const last = lines[lines.length - 1];
  if (last && (last + ' ' + w).length <= n) lines[lines.length - 1] = last + ' ' + w; else lines.push(w);
  return lines;
}, []);

export function buildV1(world, data) {
  const { trace: T, sprites: S, metrics: M } = data;
  const IMG = 'data/' + T.image.display, INPUT = 'data/' + T.image.input;
  const scenes = {};

  // token pictures, one per recorded stage: 64 rows (image) or L rows (text), 256 columns
  const tokURL = (sp, rows, stage) => matrixURL(rows, 256, (r, k) => (sp.at(k, stage * rows + r) - 127.5) / 127.5);
  const L = T.text.words.length;
  const imgTok = T.tokens.stages.map((_, s) => tokURL(S.imgTokens, 64, s));
  const txtTok = T.text.stages.map((_, s) => tokURL(S.txtTokens, L, s));
  // the learned positional embedding = (stage 1) − (stage 0), recovered from the two recorded stages
  const diffURL = (sp, rows, absmax) => {
    const d = Array.from({ length: rows }, (_, r) => Array.from({ length: 256 }, (_, k) =>
      (sp.at(k, rows + r) - 127.5) / 127.5 * absmax[1] - (sp.at(k, r) - 127.5) / 127.5 * absmax[0]));
    const m = Math.max(...d.flat().map(Math.abs));
    return matrixURL(rows, 256, (r, k) => d[r][k] / m);
  };
  const attnAt = (b, h, q, k) => S.attn.at(h * 64 + k, b * 64 + q) / 255; // each (block, head, query) row scaled to its max
  const qAttn = T.predictor.q_attn.map(block => Array.from({ length: 64 }, (_, k) => block.reduce((s, head) => s + head[k], 0) / block.length));
  // the token q looks at most in the last predictor block: the one the image-side attention follows
  const focus = qAttn[1].indexOf(Math.max(...qAttn[1]));

  // ================================================================================================ overview
  {
    const g = group(world);
    const y0 = -760;
    const photo = image(g, IMG, 40, y0 + 40, 200, 200, { 'image-rendering': 'auto', class: 'fade' });
    chip(g, 140, y0 + 18, 'Image');
    const capBox = group(g, 'fade');
    box(capBox, 40, y0 + 330, 200, 150, '#fff');
    wrap(`“${T.text.caption}”`, 24).forEach((line, i) => txt(capBox, 54, y0 + 362 + i * 24, line, { 'font-size': 16 }));
    chip(g, 140, y0 + 308, 'Caption');
    const enc = group(g, 'fade');
    box(enc, 300, y0 + 70, 170, 140, C.pink); txt(enc, 385, y0 + 146, 'Image Encoder', { 'text-anchor': 'middle', 'font-size': 18 });
    box(enc, 300, y0 + 345, 170, 120, C.pink); txt(enc, 385, y0 + 411, 'Text Encoder', { 'text-anchor': 'middle', 'font-size': 18 });
    const pred = group(g, 'fade');
    box(pred, 520, y0 + 80, 140, 120, C.pink); txt(pred, 590, y0 + 146, 'Predictor', { 'text-anchor': 'middle', 'font-size': 18 });
    const a1 = [arrow(g, `M245 ${y0 + 140} H292`), arrow(g, `M245 ${y0 + 405} H292`), arrow(g, `M475 ${y0 + 140} H512`)];
    const vecs = group(g, 'fade');
    image(vecs, vectorURL(T.predictor.vec), 700, y0 + 128, 260, 24); el('rect', { x: 700, y: y0 + 128, width: 260, height: 24, fill: 'none', class: 'thin' }, vecs);
    image(vecs, vectorURL(T.text.vec), 700, y0 + 393, 260, 24); el('rect', { x: 700, y: y0 + 393, width: 260, height: 24, fill: 'none', class: 'thin' }, vecs);
    txt(vecs, 830, y0 + 118, 'VISUAL TOKENS · 256 numbers', { 'text-anchor': 'middle', 'font-size': 15, class: 't-red' });
    txt(vecs, 830, y0 + 450, 'TARGET · 256 numbers', { 'text-anchor': 'middle', 'font-size': 15, class: 't-red' });
    const a2 = [arrow(g, `M665 ${y0 + 140} H692`), arrow(g, `M475 ${y0 + 405} H692`)];
    const score = group(g, 'fade');
    el('path', { d: `M830 ${y0 + 160} V${y0 + 245} M830 ${y0 + 385} V${y0 + 315}`, class: 'arrow flow' }, score);
    el('circle', { cx: 830, cy: y0 + 280, r: 34, fill: C.yellow, class: 'box' }, score);
    txt(score, 830, y0 + 288, fmt(T.score, 3), { 'text-anchor': 'middle', 'font-size': 22 });
    txt(score, 880, y0 + 286, 'score', { 'font-size': 16, class: 't-dim' });
    scenes.overview = {
      g,
      view: [0, y0, 1000, 500],
      beats: ['A photo and a caption go in', 'Each goes through its own encoder', '…and comes out as 256 numbers', 'One number says how well they match'],
      show(b) {
        show(photo, true); show(capBox, true);
        show(enc, b >= 1); a1.slice(0, 2).forEach(a => show(a, b >= 1));
        show(pred, b >= 1); show(a1[2], b >= 1);
        show(vecs, b >= 2); a2.forEach(a => show(a, b >= 2));
        show(score, b >= 3);
      }
    };
  }

  // ================================================================================================ data
  {
    const g = group(world);
    chip(g, 250, 18, 'Image');
    const photo = image(g, IMG, 100, 42, 300, 300, { 'image-rendering': 'auto' });
    el('rect', { x: 100, y: 42, width: 300, height: 300, fill: 'none', class: 'box' }, g);
    chip(g, 750, 18, 'Captions');
    box(g, 520, 42, 470, 300, '#fff');
    const lines = T.image.captions.map((cap, i) => {
      const row = group(g, 'fade');
      const y = 72 + i * 56;
      const hl = el('rect', { x: 528, y: y - 22, width: 454, height: 52, rx: 6, class: 'hl-rect' }, row);
      el('circle', { cx: 548, cy: y - 4, r: 11, fill: '#fff', class: 'thin' }, row);
      txt(row, 548, y + 1, String(i + 1), { 'text-anchor': 'middle', 'font-size': 13 });
      wrap(`“${cap}”`, 50).slice(0, 2).forEach((line, j) => txt(row, 568, y + j * 20, line, { 'font-size': 15 }));
      row.style.transitionDelay = `${i * 0.12}s`;
      return { row, hl };
    });
    const pick = txt(g, 750, 372, `this step: caption ${T.image.caption_used + 1}, picked at random`, { 'text-anchor': 'middle', 'font-size': 15, class: 't-dim fade' });
    scenes.data = {
      g,
      view: [0, 0, 1000, 380],
      beats: ['One Flickr30k photo', 'Five people described it', 'Each step uses one caption, at random'],
      show(b) {
        lines.forEach(({ row, hl }, i) => { show(row, b >= 1); hl.style.opacity = b >= 2 && i === T.image.caption_used ? 1 : 0; });
        show(pick, b >= 2);
        photo.style.opacity = 1;
      }
    };
  }

  // ================================================================================================ input
  {
    const g = group(world);
    const shrink = arrow(g, 'M250 350 V410');
    const input = group(g, 'fade');
    image(input, INPUT, 150, 420, 200, 200);
    el('rect', { x: 150, y: 420, width: 200, height: 200, fill: 'none', class: 'box' }, input);
    txt(input, 250, 645, '128 × 128 pixels', { 'text-anchor': 'middle', 'font-size': 16 });
    const planes = group(g, 'fade');
    const tints = ['#ff6b6b', '#51cf66', '#4dabf7'];
    tints.forEach((tint, i) => {
      const x = 380 + i * 14, y = 440 + i * 14;
      image(planes, cropURL(S.input, 0, 0, 128, 128, tint), x, y, 100, 100);
      el('rect', { x, y, width: 100, height: 100, fill: 'none', class: 'thin' }, planes);
    });
    txt(planes, 446, 590, '× 3 colours', { 'text-anchor': 'middle', 'font-size': 15 });
    const grid = group(g, 'fade');
    for (let i = 1; i < 16; i++) {
      el('line', { x1: 150 + i * 12.5, y1: 420, x2: 150 + i * 12.5, y2: 620, class: 'grid-line' }, grid);
      el('line', { x1: 150, y1: 420 + i * 12.5, x2: 350, y2: 420 + i * 12.5, class: 'grid-line' }, grid);
    }
    scenes.input = {
      g, with: ['data'],
      view: [0, 0, 560, 660],
      beats: ['The photo', 'Shrunk and cropped to 128 × 128', 'Every pixel is three numbers: red, green, blue'],
      show(b) { show(shrink, b >= 1); show(input, b >= 1); show(grid, b === 1); show(planes, b >= 2); }
    };
  }

  // ================================================================================================ convolution stem
  {
    const g = group(world);
    const top = 690;
    const blocks = T.conv.map((stage, k) => {
      const sizes = [100, 80, 64, 52], xs = [22, 172, 300, 412], size = sizes[k], off = k < 2 ? 5 : 4;
      const sg = group(g, 'fade');
      chip(sg, xs[k] + size / 2 + 14, top + 12, `Block ${k + 1}`, { fill: k === 3 ? C.green2 : C.green, size: 13 });
      const y = top + 70 + (100 - size) / 2;
      stage.picks.slice(0, 8).forEach((_, j) => {
        const d = (7 - j) * off;
        image(sg, cropURL(S.conv[k].img, j * stage.size, 0, stage.size, stage.size), xs[k] + d, y - d, size, size);
        el('rect', { x: xs[k] + d, y: y - d, width: size, height: size, fill: 'none', class: 'thin' }, sg);
      });
      txt(sg, xs[k] + size / 2 + 14, top + 205, `${stage.size}×${stage.size}×${stage.channels}`, { 'text-anchor': 'middle', 'font-size': 15 });
      txt(sg, xs[k] + size / 2 + 14, top + 226, `${stage.channels} channels`, { 'text-anchor': 'middle', 'font-size': 12, class: 't-muted' });
      sg.style.transitionDelay = '.15s';
      return sg;
    });
    const links = [0, 1, 2].map(k => arrow(g, `M${[160, 292, 405][k]} ${top + 120} H${[168, 296, 408][k] + 4}`));
    const feed = arrow(g, `M146 ${top - 80} C 80 ${top - 70}, 70 ${top - 40}, 70 ${top + 30}`);
    // the 3 × 3 window, drawn larger than life so it can be seen, sliding over the photo in steps of 2
    const kernel = el('rect', { x: 150, y: 420, width: 22, height: 22, class: 'kernel fade' }, g);
    const klabel = txt(g, 250, 668, '3 × 3 window, every 2 pixels (drawn bigger)', { 'text-anchor': 'middle', 'font-size': 14, class: 't-dim fade' });
    const cell = el('rect', { x: 150, y: 420, width: 25, height: 25, class: 'q-cell fade' }, g);
    const celllabel = txt(g, 250, 668, 'one final cell ≈ a 16 × 16 patch of the photo', { 'text-anchor': 'middle', 'font-size': 14, class: 't-dim fade' });
    let slide = 0;
    scenes.conv = {
      g, with: ['input'],
      view: [0, 395, 510, 560],
      beats: ['A 3 × 3 window slides over the photo, two pixels at a time', 'Block 1: 64 × 64, 64 channels', 'Block 2: 32 × 32, 128 channels', 'Block 3: 16 × 16, 256 channels', 'Block 4: 8 × 8, 256 channels'],
      show(b) {
        blocks.forEach((sg, k) => show(sg, b >= k + 1));
        links.forEach((a, k) => show(a, b >= k + 2));
        show(feed, b >= 1);
        show(kernel, b === 0); show(klabel, b === 0);
        show(cell, b === 4); show(celllabel, b === 4);
        cancelAnimationFrame(slide);
        if (b === 0) {
          const t0 = performance.now();
          const tick = now => {
            const i = Math.floor((now - t0) / 70) % 64; // 8 × 8 stops, drawn coarse
            kernel.setAttribute('x', 150 + (i % 8) * 22.4); kernel.setAttribute('y', 420 + Math.floor(i / 8) * 22.4);
            slide = requestAnimationFrame(tick);
          };
          slide = requestAnimationFrame(tick);
        }
        if (b === 4) { cell.setAttribute('x', 150 + (focus % 8) * 25); cell.setAttribute('y', 420 + Math.floor(focus / 8) * 25); }
      },
      leave() { cancelAnimationFrame(slide); }
    };
  }

  // ================================================================================================ tokens + positional embeddings
  {
    const g = group(world);
    const top = 1070;
    image(g, INPUT, 20, top + 40, 170, 170);
    const grid = group(g);
    for (let i = 0; i <= 8; i++) {
      el('line', { x1: 20 + i * 21.25, y1: top + 40, x2: 20 + i * 21.25, y2: top + 210, stroke: '#fff', 'stroke-width': 1.2 }, grid);
      el('line', { x1: 20, y1: top + 40 + i * 21.25, x2: 190, y2: top + 40 + i * 21.25, stroke: '#fff', 'stroke-width': 1.2 }, grid);
    }
    const cell = el('rect', { x: 20 + (focus % 8) * 21.25, y: top + 40 + Math.floor(focus / 8) * 21.25, width: 21.25, height: 21.25, class: 'q-cell' }, g);
    txt(g, 105, top + 236, '8 × 8 grid → 64 tokens', { 'text-anchor': 'middle', 'font-size': 14 });
    const col = (x, href, label, fill) => {
      const cg = group(g, 'fade');
      image(cg, href, x, top + 40, 70, 256);
      el('rect', { x, y: top + 40, width: 70, height: 256, fill: 'none', class: 'thin' }, cg);
      if (label) chip(cg, x + 35, top + 18, label, { fill, size: 12 });
      return cg;
    };
    const stem = col(205, imgTok[0], 'tokens', '#fff');
    const posOnly = diffURL(S.imgTokens, 64, T.tokens.absmax);
    const pos = col(298, posOnly, '+ Positional', C.blue);
    const sum = col(392, imgTok[1], '= tokens', '#fff');
    const plus = txt(g, 286, top + 175, '+', { 'text-anchor': 'middle', 'font-size': 22, class: 'fade' });
    const eq = txt(g, 380, top + 175, '=', { 'text-anchor': 'middle', 'font-size': 22, class: 'fade' });
    const row = el('rect', { x: 205, y: top + 40 + focus * 4, width: 70, height: 4, fill: 'none', stroke: 'var(--yellow)', 'stroke-width': 2, class: 'fade' }, g);
    const link = el('path', { d: `M${20 + (focus % 8) * 21.25 + 21} ${top + 50 + Math.floor(focus / 8) * 21.25} L205 ${top + 42 + focus * 4}`, stroke: 'var(--yellow)', 'stroke-width': 2, fill: 'none', class: 'fade' }, g);
    const dims = txt(g, 240, top + 318, '256 numbers →', { 'text-anchor': 'middle', 'font-size': 13, class: 't-muted fade' });
    scenes.tokens = {
      g,
      view: [0, top, 500, 340],
      beats: ['The 8 × 8 × 256 output, read as a grid', 'Each cell becomes a token: one row of 256 numbers', 'Every position has its own learned vector…', '…added on, so each token knows where it was'],
      show(b) {
        show(stem, b >= 1); show(row, b >= 1); show(link, b >= 1); show(dims, b >= 1);
        show(pos, b >= 2); show(plus, b >= 2);
        show(sum, b >= 3); show(eq, b >= 3);
        cell.style.opacity = 1;
      }
    };
  }

  // ================================================================================================ image Transformer Stem
  {
    const g = group(world);
    const top = 1460, N = 300, cs = N / 8;
    image(g, INPUT, 20, top + 30, N, N);
    const heat = Array.from({ length: 64 }, (_, k) =>
      el('rect', { x: 20 + (k % 8) * cs, y: top + 30 + Math.floor(k / 8) * cs, width: cs, height: cs, class: 'cell-hot', opacity: 0, style: 'transition: opacity .5s' }, g));
    el('rect', { x: 20 + (focus % 8) * cs, y: top + 30 + Math.floor(focus / 8) * cs, width: cs, height: cs, class: 'q-cell' }, g);
    el('rect', { x: 20, y: top + 30, width: N, height: N, fill: 'none', class: 'box' }, g);
    txt(g, 170, top + 18, 'where the yellow token looks', { 'text-anchor': 'middle', 'font-size': 14, class: 't-dim' });
    // Transformer Stem: six orange blocks, like the drawing
    box(g, 340, top + 24, 150, 326, C.orange2);
    txt(g, 415, top + 14, 'Transformer Stem', { 'text-anchor': 'middle', 'font-size': 14 });
    const chips = Array.from({ length: 6 }, (_, i) => {
      const r = el('rect', { x: 352, y: top + 36 + i * 50, width: 126, height: 40, rx: 7, fill: C.orange, class: 'box', style: 'transition: fill .3s, stroke .3s, opacity .3s' }, g);
      txt(g, 415, top + 62 + i * 50, i ? `B${i + 1}` : 'Block 1', { 'text-anchor': 'middle', 'font-size': 15 });
      return r;
    });
    // the four heads of the current block, small
    const heads = Array.from({ length: 4 }, (_, h) => {
      const hg = group(g);
      const x = 20 + h * 78, y = top + 350;
      image(hg, INPUT, x, y, 66, 66);
      const cells = Array.from({ length: 64 }, (_, k) => el('rect', { x: x + (k % 8) * 8.25, y: y + Math.floor(k / 8) * 8.25, width: 8.25, height: 8.25, class: 'cell-hot', opacity: 0, style: 'transition: opacity .5s' }, hg));
      el('rect', { x, y, width: 66, height: 66, fill: 'none', class: 'thin' }, hg);
      txt(hg, x + 33, y + 82, `head ${h + 1}`, { 'text-anchor': 'middle', 'font-size': 12, class: 't-muted' });
      return cells;
    });
    // token column: the 64 tokens as they leave each block
    const cols = imgTok.map((href, s) => image(g, href, 340, top + 372, 150, 96, { class: 'fade' }));
    el('rect', { x: 340, y: top + 372, width: 150, height: 96, fill: 'none', class: 'thin' }, g);
    txt(g, 415, top + 488, '64 tokens × 256', { 'text-anchor': 'middle', 'font-size': 12, class: 't-muted' });
    const lnBar = group(g, 'fade');
    box(lnBar, 20, top + 520, 470, 34, C.purple); txt(lnBar, 255, top + 543, 'LayerNorm', { 'text-anchor': 'middle', 'font-size': 16 });
    const out = group(g, 'fade');
    image(out, imgTok[8], 20, top + 580, 470, 120);
    el('rect', { x: 20, y: top + 580, width: 470, height: 120, fill: 'none', class: 'box' }, out);
    txt(out, 255, top + 722, 'visual tokens  [64, 256]', { 'text-anchor': 'middle', 'font-size': 16, class: 't-red' });
    const beats = [...Array.from({ length: 6 }, (_, b) => `Block ${b + 1}: every token looks at every other token`), 'LayerNorm, then out: 64 visual tokens'];
    scenes['image-blocks'] = {
      g,
      view: [0, top - 10, 500, 760],
      beats,
      show(b) {
        const blk = Math.min(b, 5), on = b <= 5;
        chips.forEach((r, i) => { r.setAttribute('fill', i === blk && on ? 'var(--yellow)' : C.orange); r.style.opacity = i <= blk ? 1 : .45; });
        const mean = Array.from({ length: 64 }, (_, k) => [0, 1, 2, 3].reduce((s, h) => s + attnAt(blk, h, focus, k), 0) / 4);
        const top = Math.max(...mean);
        heat.forEach((r, k) => r.setAttribute('opacity', on ? (0.72 * mean[k] / top).toFixed(3) : 0));
        heads.forEach((cells, h) => cells.forEach((r, k) => r.setAttribute('opacity', (0.8 * attnAt(blk, h, focus, k)).toFixed(3))));
        cols.forEach((c, s) => show(c, s === (on ? blk + 2 : 8)));
        show(lnBar, b >= 6); show(out, b >= 6);
      }
    };
  }

  // ================================================================================================ words → vectors
  {
    const g = group(world);
    const top = 400;
    const sentence = txt(g, 750, top + 16, `“${T.text.caption}”`, { 'text-anchor': 'middle', 'font-size': 15, class: 'fade' });
    // word chips, wrapped into rows
    const chips = [];
    let x = 520, y = top + 60;
    T.text.words.forEach((w, i) => {
      const width = w.length * 8.6 + 16;
      if (x + width > 985) { x = 520; y += 58; }
      const cg = group(g, 'fade');
      el('rect', { x, y: y - 18, width, height: 26, rx: 5, fill: '#fff', class: 'chip-rect' }, cg);
      txt(cg, x + width / 2, y, w, { 'text-anchor': 'middle', 'font-size': 15 });
      const id = txt(cg, x + width / 2, y + 24, String(T.text.ids[i]), { 'text-anchor': 'middle', 'font-size': 11, class: 't-mono t-muted fade' });
      cg.style.transitionDelay = `${i * 0.05}s`;
      chips.push({ cg, id, cx: x + width / 2, cy: y + 30 });
      x += width + 8;
    });
    // the Embedding Matrix: one row per word in the vocabulary
    const mtop = top + 210, mx = 560, mw = 240, mh = 300;
    const mat = group(g, 'fade');
    box(mat, mx, mtop, mw, mh, C.yellow);
    txt(mat, mx + mw / 2, mtop - 10, 'Embedding Matrix', { 'text-anchor': 'middle', 'font-size': 16 });
    for (let i = 1; i < 30; i++) el('line', { x1: mx + 6, y1: mtop + i * 10, x2: mx + mw - 6, y2: mtop + i * 10, stroke: '#00000012' }, mat);
    txt(mat, mx - 12, mtop + mh / 2, `${M.vocab.toLocaleString()} words`, { 'font-size': 13, transform: `rotate(-90 ${mx - 12} ${mtop + mh / 2})`, 'text-anchor': 'middle', class: 't-dim' });
    txt(mat, mx + mw / 2, mtop + mh + 18, '256 numbers per row', { 'text-anchor': 'middle', 'font-size': 13, class: 't-dim' });
    // rows sit roughly where their id is (common words have small ids, so they bunch up at the top)
    const rowY = id => mtop + 4 + Math.sqrt(id / M.vocab) * (mh - 8);
    const lookups = T.text.ids.map((id, i) => {
      const lg = group(g, 'fade');
      el('rect', { x: mx + 4, y: rowY(id) - 2, width: mw - 8, height: 4, fill: 'var(--x-red)', opacity: .8 }, lg);
      el('path', { d: `M${chips[i].cx} ${chips[i].cy} C ${chips[i].cx} ${mtop - 30}, ${mx - 20} ${rowY(id)}, ${mx + 4} ${rowY(id)}`, fill: 'none', stroke: 'var(--x-red)', 'stroke-width': 1, opacity: .5 }, lg);
      lg.style.transitionDelay = `${i * 0.06}s`;
      return lg;
    });
    // a close-up of one lookup: the word "dog" and its neighbouring rows
    const pick = Math.max(0, T.text.words.indexOf('dog'));
    const zoom = group(g, 'fade');
    box(zoom, 820, mtop + 40, 160, 150, '#fff');
    txt(zoom, 900, mtop + 30, `row ${T.text.ids[pick]}`, { 'text-anchor': 'middle', 'font-size': 14 });
    T.text.neighbors[pick].forEach(([id, word], j) => {
      const yy = mtop + 66 + j * 26;
      if (id === T.text.ids[pick]) el('rect', { x: 826, y: yy - 17, width: 148, height: 24, rx: 4, class: 'hl-rect' }, zoom);
      txt(zoom, 836, yy, String(id), { 'font-size': 11, class: 't-mono t-muted' });
      txt(zoom, 880, yy, word, { 'font-size': 15 });
    });
    const vec = group(g, 'fade');
    image(vec, txtTok[0], 820, mtop + 220, 160, L * 6);
    el('rect', { x: 820, y: mtop + 220, width: 160, height: L * 6, fill: 'none', class: 'thin' }, vec);
    txt(vec, 900, mtop + 212, `${L} vectors`, { 'text-anchor': 'middle', 'font-size': 14 });
    const out = arrow(g, `M${mx + mw + 4} ${mtop + mh - 40} C ${mx + mw + 30} ${mtop + mh - 40}, 800 ${mtop + 260}, 814 ${mtop + 250}`);
    scenes.embed = {
      g,
      view: [500, top - 10, 500, 570],
      beats: ['The caption', 'Lowercased and split into words', 'Each word becomes its number in the vocabulary', 'The number picks a row of the Embedding Matrix', `${L} words → ${L} vectors of 256`],
      show(b) {
        show(sentence, b === 0);
        chips.forEach(c => { show(c.cg, b >= 1); show(c.id, b >= 2); });
        show(mat, b >= 3); lookups.forEach(l => show(l, b >= 3)); show(zoom, b >= 3);
        show(vec, b >= 4); show(out, b >= 4);
      }
    };
  }

  // ================================================================================================ text Transformer Stem
  {
    const g = group(world);
    const top = 1070;
    const col = (x, href, label, fill) => {
      const cg = group(g, 'fade');
      image(cg, href, x, top + 40, 110, L * 8);
      el('rect', { x, y: top + 40, width: 110, height: L * 8, fill: 'none', class: 'thin' }, cg);
      chip(cg, x + 55, top + 18, label, { fill, size: 12 });
      return cg;
    };
    const posOnly = diffURL(S.txtTokens, L, T.text.absmax);
    const parts = [col(530, txtTok[0], 'word vectors', '#fff'), col(690, posOnly, '+ Positional', C.blue), col(850, txtTok[1], '=', '#fff')];
    const signs = [txt(g, 670, top + 40 + L * 4 + 8, '+', { 'text-anchor': 'middle', 'font-size': 22, class: 'fade' }), txt(g, 830, top + 40 + L * 4 + 8, '=', { 'text-anchor': 'middle', 'font-size': 22, class: 'fade' })];
    // words down the left, attention arcs bulging to the right
    const wtop = top + 220, gap = 34;
    const qi = Math.max(0, T.text.words.indexOf('dog'));
    const words = T.text.words.map((w, i) => {
      const t = txt(g, 640, wtop + i * gap, w, { 'text-anchor': 'end', 'font-size': 16 });
      return t;
    });
    el('rect', { x: 645 - T.text.words[qi].length * 9 - 14, y: wtop + qi * gap - 18, width: T.text.words[qi].length * 9 + 14, height: 26, rx: 5, class: 'hl-rect' }, g);
    const pads = group(g, 'fade');
    [0, 1, 2].forEach(j => {
      const y = wtop + (L + j) * gap;
      el('rect', { x: 576, y: y - 17, width: 66, height: 24, rx: 5, fill: '#f1f0eb', stroke: '#bdbcb5', 'stroke-dasharray': '4 3' }, pads);
      txt(pads, 609, y, '<pad>', { 'text-anchor': 'middle', 'font-size': 13, class: 't-muted' });
    });
    txt(pads, 655, wtop + (L + 1) * gap, '← masked: never looked at', { 'font-size': 13, class: 't-muted' });
    const arcs = T.text.words.map((_, j) => el('path', {
      d: `M655 ${wtop + qi * gap - 5} C ${655 + 40 + Math.abs(j - qi) * 14} ${wtop + qi * gap - 5}, ${655 + 40 + Math.abs(j - qi) * 14} ${wtop + j * gap - 5}, 655 ${wtop + j * gap - 5}`,
      class: 'attn-arc', opacity: 0
    }, g));
    box(g, 870, wtop - 30, 110, 316, C.orange2);
    txt(g, 925, wtop - 40, 'Transformer Stem', { 'text-anchor': 'middle', 'font-size': 13 });
    const chips = Array.from({ length: 6 }, (_, i) => {
      const r = el('rect', { x: 880, y: wtop - 20 + i * 50, width: 90, height: 40, rx: 7, fill: C.orange, class: 'box', style: 'transition: fill .3s, opacity .3s' }, g);
      txt(g, 925, wtop + 6 + i * 50, i ? `B${i + 1}` : 'Block 1', { 'text-anchor': 'middle', 'font-size': 15 });
      return r;
    });
    const ln = group(g, 'fade');
    box(ln, 870, wtop + 300, 110, 30, C.purple); txt(ln, 925, wtop + 320, 'LayerNorm', { 'text-anchor': 'middle', 'font-size': 14 });
    const panel = [...words, ...arcs];
    scenes['text-blocks'] = {
      g,
      view: [505, top - 10, 495, wtop - top + (L + 3) * gap + 40],
      beats: ['Each word vector…', '…plus its position’s own vector', ...Array.from({ length: 6 }, (_, b) => `Block ${b + 1}: where “${T.text.words[qi]}” looks`), 'LayerNorm'],
      show(b) {
        show(parts[0], true); show(parts[1], b >= 1); show(parts[2], b >= 1); signs.forEach(s => show(s, b >= 1));
        const blk = Math.min(Math.max(b - 2, 0), 5), on = b >= 2;
        panel.forEach(n => n.style.opacity = on ? 1 : .25);
        const heads = T.text.attn[blk];
        const row = Array.from({ length: L }, (_, j) => heads.reduce((s, head) => s + head[qi][j], 0) / heads.length);
        arcs.forEach((a, j) => { a.setAttribute('stroke-width', (0.5 + 14 * row[j]).toFixed(2)); a.setAttribute('opacity', on && j !== qi ? Math.min(1, 0.15 + row[j] * 3).toFixed(2) : 0); });
        chips.forEach((r, i) => { r.setAttribute('fill', on && i === blk && b <= 7 ? 'var(--yellow)' : C.orange); r.style.opacity = !on || i <= blk ? 1 : .45; });
        show(pads, on); show(ln, b >= 8);
      }
    };
  }

  // ================================================================================================ TARGET
  {
    const g = group(world);
    const top = 1880;
    image(g, txtTok[8], 530, top + 30, 170, L * 9);
    el('rect', { x: 530, y: top + 30, width: 170, height: L * 9, fill: 'none', class: 'thin' }, g);
    txt(g, 615, top + 18, `${L} word vectors`, { 'text-anchor': 'middle', 'font-size': 14 });
    const avg = group(g, 'fade');
    T.text.words.forEach((_, i) => el('path', { d: `M702 ${top + 34 + i * 9} C 740 ${top + 34 + i * 9}, 740 ${top + 82}, 772 ${top + 82}`, fill: 'none', stroke: 'var(--muted)', 'stroke-width': 1 }, avg));
    image(avg, vectorURL(T.text.pooled), 780, top + 72, 200, 20);
    el('rect', { x: 780, y: top + 72, width: 200, height: 20, fill: 'none', class: 'thin' }, avg);
    txt(avg, 880, top + 62, 'average', { 'text-anchor': 'middle', 'font-size': 14 });
    const unit = group(g, 'fade');
    arrow(unit, `M880 ${top + 98} V${top + 140}`);
    txt(unit, 890, top + 124, '÷ its length', { 'font-size': 13, class: 't-dim' });
    image(unit, vectorURL(T.text.vec), 780, top + 150, 200, 20);
    el('rect', { x: 780, y: top + 150, width: 200, height: 20, fill: 'none', class: 'box' }, unit);
    txt(unit, 880, top + 196, 'TARGET  [1, 256]', { 'text-anchor': 'middle', 'font-size': 18, class: 't-red' });
    txt(unit, 880, top + 216, 'length 1', { 'text-anchor': 'middle', 'font-size': 13, class: 't-muted' });
    scenes.target = {
      g, with: ['text-blocks'],
      view: [510, top - 10, 490, 250],
      beats: ['One vector per word, after the last LayerNorm', 'Averaged over the real words', 'Scaled to length 1: the TARGET'],
      show(b) { show(avg, b >= 1); show(unit, b >= 2); }
    };
  }

  // ================================================================================================ Predictor
  {
    const g = group(world);
    const top = 2200;
    const vis = group(g);
    image(vis, imgTok[8], 20, top + 40, 90, 256);
    el('rect', { x: 20, y: top + 40, width: 90, height: 256, fill: 'none', class: 'thin' }, vis);
    txt(vis, 65, top + 28, '64 visual tokens', { 'text-anchor': 'middle', 'font-size': 13 });
    const lin = group(g, 'fade');
    box(lin, 125, top + 40, 36, 256, C.orange); txt(lin, 148, top + 168, 'Linear(256, 256)', { 'font-size': 13, transform: `rotate(-90 148 ${top + 168})`, 'text-anchor': 'middle' });
    const q = group(g, 'fade');
    el('rect', { x: 20, y: top + 304, width: 90, height: 10, rx: 2, fill: 'var(--yellow)', class: 'thin' }, q);
    txt(q, 65, top + 334, '+ q', { 'text-anchor': 'middle', 'font-size': 16 });
    txt(q, 65, top + 352, 'learned, same for', { 'text-anchor': 'middle', 'font-size': 11, class: 't-muted' });
    txt(q, 65, top + 365, 'every photo', { 'text-anchor': 'middle', 'font-size': 11, class: 't-muted' });
    const stem = group(g, 'fade');
    box(stem, 178, top + 40, 110, 150, C.orange2);
    txt(stem, 233, top + 30, 'Transformer Stem', { 'text-anchor': 'middle', 'font-size': 13 });
    const pchips = [0, 1].map(i => {
      const r = el('rect', { x: 188, y: top + 56 + i * 64, width: 90, height: 48, rx: 7, fill: C.orange, class: 'box', style: 'transition: fill .3s' }, stem);
      txt(stem, 233, top + 86 + i * 64, `Block ${i + 1}`, { 'text-anchor': 'middle', 'font-size': 15 });
      return r;
    });
    const look = group(g, 'fade');
    const P = 180;
    image(look, INPUT, 305, top + 40, P, P);
    const heat = Array.from({ length: 64 }, (_, k) => el('rect', { x: 305 + (k % 8) * P / 8, y: top + 40 + Math.floor(k / 8) * P / 8, width: P / 8, height: P / 8, class: 'cell-hot', opacity: 0, style: 'transition: opacity .5s' }, look));
    el('rect', { x: 305, y: top + 40, width: P, height: P, fill: 'none', class: 'box' }, look);
    txt(look, 395, top + 30, 'where q looks', { 'text-anchor': 'middle', 'font-size': 13 });
    const take = group(g, 'fade');
    box(take, 178, top + 205, 110, 30, '#cfe8ff'); txt(take, 233, top + 225, 'take q only', { 'text-anchor': 'middle', 'font-size': 13 });
    box(take, 178, top + 243, 110, 26, C.purple); txt(take, 233, top + 261, 'LayerNorm', { 'text-anchor': 'middle', 'font-size': 13 });
    box(take, 178, top + 277, 110, 26, C.orange); txt(take, 233, top + 295, 'Linear(256, 256)', { 'text-anchor': 'middle', 'font-size': 12 });
    const vec = group(g, 'fade');
    arrow(vec, `M233 ${top + 308} V${top + 340}`);
    image(vec, vectorURL(T.predictor.vec), 120, top + 350, 360, 22);
    el('rect', { x: 120, y: top + 350, width: 360, height: 22, fill: 'none', class: 'box' }, vec);
    txt(vec, 300, top + 398, 'VISUAL TOKENS  [1, 256]  ·  length 1', { 'text-anchor': 'middle', 'font-size': 17, class: 't-red' });
    scenes.predictor = {
      g,
      view: [0, top - 6, 500, 420],
      beats: ['The 64 visual tokens', 'Linear(256, 256) on each', 'Append q: 65 tokens', 'Block 1: q looks at all 64 tokens', 'Block 2: and again', 'Keep q only → LayerNorm → Linear → length 1'],
      show(b) {
        show(lin, b >= 1); show(q, b >= 2); show(stem, b >= 2);
        const blk = b === 3 ? 0 : 1;
        pchips.forEach((r, i) => r.setAttribute('fill', (b === 3 && i === 0) || (b === 4 && i === 1) ? 'var(--yellow)' : C.orange));
        show(look, b >= 3);
        const m = Math.max(...qAttn[blk]);
        heat.forEach((r, k) => r.setAttribute('opacity', (0.75 * qAttn[blk][k] / m).toFixed(3)));
        show(take, b >= 5); show(vec, b >= 5);
      }
    };
  }

  // ================================================================================================ score
  {
    const g = group(world);
    const top = 2660;
    const a = T.predictor.vec, t = T.text.vec;
    const prod = a.map((v, i) => v * t[i]);
    const strip = (y, href, label, cls) => {
      const sg = group(g, 'fade');
      txt(sg, 40, y + 17, label, { 'font-size': 17, class: cls, 'text-anchor': 'start' });
      image(sg, href, 250, y, 480, 26);
      el('rect', { x: 250, y, width: 480, height: 26, fill: 'none', class: 'thin' }, sg);
      return sg;
    };
    const s1 = strip(top + 20, vectorURL(a), 'VISUAL TOKENS  ŷ', 't-red');
    const s2 = strip(top + 60, vectorURL(t), 'TARGET  y', 't-red');
    const s3 = strip(top + 120, vectorURL(prod), 'ŷ₁y₁, ŷ₂y₂, … ŷ₂₅₆y₂₅₆', 't-dim');
    const times = txt(g, 490, top + 110, '× position by position', { 'text-anchor': 'middle', 'font-size': 13, class: 't-muted fade' });
    const sum = group(g, 'fade');
    arrow(sum, `M490 ${top + 152} V${top + 180}`);
    el('rect', { x: 360, y: top + 186, width: 260, height: 54, rx: 10, fill: C.yellow, class: 'box' }, sum);
    txt(sum, 490, top + 222, `add them up = ${fmt(T.score, 3)}`, { 'text-anchor': 'middle', 'font-size': 22 });
    const others = group(g, 'fade');
    txt(others, 40, top + 290, 'This photo’s best matches among all 5,070 validation captions:', { 'font-size': 16, class: 't-dim' });
    T.top.slice(0, 5).forEach((r, i) => {
      txt(others, 50, top + 322 + i * 28, fmt(r.score, 3), { 'font-size': 14, class: 't-mono' });
      const cap = `${r.own ? '✓ ' : ''}${r.caption}`;
      txt(others, 110, top + 322 + i * 28, cap.length > 66 ? cap.slice(0, 64) + '…' : cap, { 'font-size': 16, class: r.own ? '' : 't-dim' });
    });
    txt(others, 40, top + 470, '✓ = one of its own five captions', { 'font-size': 13, class: 't-muted' });
    scenes.score = {
      g,
      view: [20, top, 740, 480],
      beats: ['Two vectors of length 1', 'Multiply position by position', 'Add up: the cosine of the angle between them', 'The same score against every other caption'],
      show(b) { show(s1, true); show(s2, true); show(s3, b >= 1); show(times, b >= 1); show(sum, b >= 2); show(others, b >= 3); }
    };
  }

  // ================================================================================================ InfoNCE
  {
    const g = group(world);
    const top = 3250, x0 = 230, cs = 56, B = T.batch.items.length;
    const Sx = T.batch.S;
    txt(g, x0 + (B * cs) / 2, top - 92, 'S = visual tokens @ targetᵀ', { 'text-anchor': 'middle', 'font-size': 18 });
    T.batch.items.forEach((it, i) => {
      image(g, 'data/' + it.thumb, x0 - 60, top + i * cs + 4, 48, 48, { 'image-rendering': 'auto' });
      txt(g, x0 - 70, top + i * cs + 34, `img ${i + 1}`, { 'text-anchor': 'end', 'font-size': 13, class: 't-dim' });
      const words = it.caption.replace(/\.$/, '').split(' ').slice(0, 3).join(' ') + '…';
      txt(g, x0 + i * cs + 28, top - 10, words, { 'font-size': 12, transform: `rotate(-38 ${x0 + i * cs + 28} ${top - 10})`, class: 't-dim' });
    });
    const cells = [], vals = [];
    for (let i = 0; i < B; i++) for (let j = 0; j < B; j++) {
      cells.push(el('rect', { x: x0 + j * cs, y: top + i * cs, width: cs, height: cs, class: 'mx-cell', fill: '#fff' }, g));
      vals.push(txt(g, x0 + j * cs + cs / 2, top + i * cs + cs / 2 + 4, '', { 'text-anchor': 'middle', 'font-size': 12, class: 't-mono' }));
    }
    const diag = group(g, 'fade');
    for (let i = 0; i < B; i++) el('rect', { x: x0 + i * cs + 1.5, y: top + i * cs + 1.5, width: cs - 3, height: cs - 3, class: 'mx-diag' }, diag);
    txt(diag, x0 + B * cs + 16, top - 14, 'diagonal: should be HIGHEST', { 'font-size': 14, class: 't-red' });
    const nll = p => -Math.log(Math.max(p, 1e-9));
    const rowCE = group(g, 'fade');
    txt(rowCE, x0 + B * cs + 20, top + 20, 'row: −log p', { 'font-size': 13, class: 't-dim' });
    T.batch.probs_rows.forEach((row, i) => txt(rowCE, x0 + B * cs + 20, top + i * cs + 36, `p=${row[i].toFixed(3)}  ${nll(row[i]).toFixed(2)}`, { 'font-size': 13, class: 't-mono' }));
    const li = group(g, 'fade');
    el('rect', { x: x0 + B * cs + 14, y: top + B * cs + 6, width: 170, height: 34, rx: 7, fill: C.yellow, class: 'box' }, li);
    txt(li, x0 + B * cs + 99, top + B * cs + 29, `loss_i = ${T.batch.loss_i.toFixed(3)}`, { 'text-anchor': 'middle', 'font-size': 16 });
    const colCE = group(g, 'fade');
    T.batch.probs_cols.forEach((col, j) => txt(colCE, x0 + j * cs + cs / 2, top + B * cs + 22, nll(col[j]).toFixed(2), { 'text-anchor': 'middle', 'font-size': 12, class: 't-mono' }));
    txt(colCE, x0 - 12, top + B * cs + 22, 'column: −log p', { 'text-anchor': 'end', 'font-size': 13, class: 't-dim' });
    const lt = group(g, 'fade');
    el('rect', { x: x0 + (B * cs) / 2 - 85, y: top + B * cs + 36, width: 170, height: 34, rx: 7, fill: C.yellow, class: 'box' }, lt);
    txt(lt, x0 + (B * cs) / 2, top + B * cs + 59, `loss_t = ${T.batch.loss_t.toFixed(3)}`, { 'text-anchor': 'middle', 'font-size': 16 });
    const fin = group(g, 'fade');
    el('rect', { x: x0 + B * cs + 14, y: top + B * cs + 60, width: 220, height: 52, rx: 10, fill: '#fff', class: 'box' }, fin);
    txt(fin, x0 + B * cs + 124, top + B * cs + 93, `Loss = ${T.batch.loss.toFixed(3)}`, { 'text-anchor': 'middle', 'font-size': 22 });
    const scale = txt(g, x0 + (B * cs) / 2, top - 70, `then × e^t = ${T.scale.toFixed(1)} and softmax`, { 'text-anchor': 'middle', 'font-size': 14, class: 't-dim fade' });
    const color = (i, j, mode) => {
      const v = mode === 'S' ? Sx[i][j] : mode === 'rows' ? T.batch.probs_rows[i][j] : T.batch.probs_cols[j][i];
      const [r, gg, b] = mode === 'S' ? (v >= 0 ? [31, 58, 138] : [138, 59, 107]) : [224, 177, 0];
      const a = mode === 'S' ? Math.min(1, Math.abs(v) / 0.6) : Math.sqrt(v);
      return `rgb(${255 + (r - 255) * a} ${255 + (gg - 255) * a} ${255 + (b - 255) * a})`;
    };
    scenes.infonce = {
      g,
      view: [x0 - 90, top - 130, B * cs + 330, B * cs + 260],
      beats: ['Every photo in the batch × every caption', 'True pairs on the diagonal', 'Each row: softmax, then −log p of the right caption', 'Averaged over rows: loss_i', 'Each column: the same, photo by caption: loss_t', 'Loss = (loss_i + loss_t) / 2'],
      show(b) {
        const mode = b >= 4 ? 'cols' : b >= 2 ? 'rows' : 'S';
        cells.forEach((c, n) => {
          const i = Math.floor(n / B), j = n % B;
          c.setAttribute('fill', color(i, j, mode));
          const v = mode === 'S' ? Sx[i][j] : mode === 'rows' ? T.batch.probs_rows[i][j] : T.batch.probs_cols[j][i];
          vals[n].textContent = mode === 'S' ? fmt(v) : v < 0.005 ? '0' : v.toFixed(2);
          vals[n].setAttribute('fill', (mode === 'S' ? Math.abs(v) > 0.35 : v > 0.5) ? '#fff' : '#1a1a1a');
        });
        show(diag, b >= 1); show(scale, b >= 2);
        show(rowCE, b >= 2); show(li, b >= 3); show(colCE, b >= 4); show(lt, b >= 4); show(fin, b >= 5);
      }
    };
  }

  // ================================================================================================ training
  {
    const g = group(world);
    const top = 3900, x0 = 80, W = 560, H = 300;
    const E = M.epochs, maxL = 7.2;
    const px = e => x0 + (e / (E.length - 1)) * W, py = l => top + H - (l / maxL) * H;
    el('line', { x1: x0, y1: top + H, x2: x0 + W, y2: top + H, class: 'thin' }, g);
    el('line', { x1: x0, y1: top, x2: x0, y2: top + H, class: 'thin' }, g);
    [0, 2, 4, 6].forEach(v => txt(g, x0 - 10, py(v) + 4, String(v), { 'text-anchor': 'end', 'font-size': 12, class: 't-muted' }));
    [0, 50, 100, 150, 199].forEach(e => txt(g, px(e), top + H + 18, String(e), { 'text-anchor': 'middle', 'font-size': 12, class: 't-muted' }));
    txt(g, x0 + W / 2, top + H + 40, 'epoch', { 'text-anchor': 'middle', 'font-size': 13, class: 't-dim' });
    const chance = group(g, 'fade');
    el('line', { x1: x0, y1: py(M.chance), x2: x0 + W, y2: py(M.chance), stroke: 'var(--muted)', 'stroke-dasharray': '5 5' }, chance);
    txt(chance, x0 + W, py(M.chance) - 8, `guessing at random: ${M.chance.toFixed(2)}`, { 'text-anchor': 'end', 'font-size': 13, class: 't-muted' });
    const path = key => E.map((e, i) => `${i ? 'L' : 'M'}${px(i).toFixed(1)} ${py(e[key]).toFixed(1)}`).join(' ');
    const train = el('path', { d: path('train_loss'), fill: 'none', stroke: 'var(--accent)', 'stroke-width': 2.5, pathLength: 1, 'stroke-dasharray': 1, 'stroke-dashoffset': 1, style: 'transition: stroke-dashoffset 1.6s ease' }, g);
    const val = el('path', { d: path('val_loss'), fill: 'none', stroke: 'var(--grad)', 'stroke-width': 2.5, pathLength: 1, 'stroke-dasharray': 1, 'stroke-dashoffset': 1, style: 'transition: stroke-dashoffset 1.6s ease' }, g);
    const legend = group(g, 'fade');
    txt(legend, px(110), py(E[110].train_loss) - 14, 'training loss', { 'font-size': 15, class: 't-accent' });
    txt(legend, px(110), py(E[110].val_loss) - 14, 'validation loss', { 'font-size': 15, fill: 'var(--grad)' });
    const best = group(g, 'fade');
    const be = M.best_epoch;
    el('circle', { cx: px(be), cy: py(M.best_val_loss), r: 7, fill: 'var(--yellow)', class: 'thin' }, best);
    txt(best, px(be), py(M.best_val_loss) + 30, `best: ${M.best_val_loss.toFixed(2)} at epoch ${be}`, { 'text-anchor': 'middle', 'font-size': 15 });
    const over = el('rect', { x: px(be), y: top, width: px(E.length - 1) - px(be), height: H, fill: '#c2410c10', class: 'fade' }, g);
    const overT = txt(g, px(130), top + 24, 'memorising the training photos', { 'text-anchor': 'middle', 'font-size': 15, fill: 'var(--grad)', class: 'fade' });
    scenes.training = {
      g,
      view: [20, top - 30, W + 100, H + 90],
      beats: ['Training loss, 200 epochs', 'Validation loss, on 1,014 photos it never trains on', 'Best checkpoint: epoch 59', 'After that, it memorises'],
      show(b) {
        train.setAttribute('stroke-dashoffset', 0);
        val.setAttribute('stroke-dashoffset', b >= 1 ? 0 : 1);
        show(chance, b >= 1); show(legend, b >= 1); show(best, b >= 2); show(over, b >= 3); show(overT, b >= 3);
      }
    };
  }

  // ================================================================================================ the whole map
  {
    const g = group(world);
    const y0 = -1560;
    const H = (x, y, label, cls = 't-red') => txt(g, x, y, label, { 'font-size': 15, class: `${cls} fade` });
    // left: what v1 does
    const left = group(g, 'fade');
    txt(left, 250, y0 + 20, 'v1: pick the right caption', { 'text-anchor': 'middle', 'font-size': 22 });
    image(left, IMG, 30, y0 + 60, 90, 90, { 'image-rendering': 'auto' });
    box(left, 150, y0 + 70, 150, 70, C.pink); txt(left, 225, y0 + 110, 'Encoder + Predictor', { 'text-anchor': 'middle', 'font-size': 14 });
    box(left, 30, y0 + 200, 90, 70, '#fff'); txt(left, 75, y0 + 240, '“A brown dog…”', { 'text-anchor': 'middle', 'font-size': 12 });
    box(left, 150, y0 + 200, 150, 70, C.pink); txt(left, 225, y0 + 240, 'Text Encoder', { 'text-anchor': 'middle', 'font-size': 14 });
    arrow(left, `M124 ${y0 + 105} H144`); arrow(left, `M124 ${y0 + 235} H144`);
    arrow(left, `M304 ${y0 + 105} C 350 ${y0 + 105}, 350 ${y0 + 150}, 362 ${y0 + 160}`); arrow(left, `M304 ${y0 + 235} C 350 ${y0 + 235}, 350 ${y0 + 190}, 362 ${y0 + 180}`);
    box(left, 366, y0 + 130, 110, 80, C.yellow); txt(left, 421, y0 + 166, 'InfoNCE', { 'text-anchor': 'middle', 'font-size': 16 });
    for (let i = 0; i < 24; i++) el('circle', { cx: 380 + (i % 8) * 11.5, cy: y0 + 182 + Math.floor(i / 8) * 9, r: 3, fill: i ? 'var(--neg)' : 'var(--accent)' }, left);
    const lnotes = [H(30, y0 + 320, '✗ nothing in the photo is hidden'), H(30, y0 + 345, '✗ both encoders learn from this one loss'), H(30, y0 + 370, '✗ only works with 255 wrong captions per batch')];
    // right: what a JEPA does
    const right = group(g, 'fade');
    txt(right, 750, y0 + 20, 'JEPA: fill in the blank', { 'text-anchor': 'middle', 'font-size': 22 });
    image(right, INPUT, 530, y0 + 60, 90, 90);
    [[1, 1, 3, 2], [5, 0, 2, 3], [0, 5, 3, 2], [4, 5, 3, 3]].forEach(([x, y, w, h]) => el('rect', { x: 530 + x * 11.25, y: y0 + 60 + y * 11.25, width: w * 11.25, height: h * 11.25, fill: '#5a5a5a', opacity: .85 }, right));
    box(right, 650, y0 + 70, 150, 70, C.pink); txt(right, 725, y0 + 110, 'Encoder + Predictor', { 'text-anchor': 'middle', 'font-size': 14 });
    image(right, INPUT, 530, y0 + 190, 90, 90);
    el('rect', { x: 650, y: y0 + 200, width: 150, height: 70, rx: 9, fill: '#fff', stroke: 'var(--ink)', 'stroke-width': 2, 'stroke-dasharray': '7 5' }, right);
    txt(right, 725, y0 + 232, 'slow copy', { 'text-anchor': 'middle', 'font-size': 15 }); txt(right, 725, y0 + 254, 'of the encoder', { 'text-anchor': 'middle', 'font-size': 13, class: 't-muted' });
    arrow(right, `M624 ${y0 + 105} H644`); arrow(right, `M624 ${y0 + 235} H644`);
    arrow(right, `M804 ${y0 + 105} C 850 ${y0 + 105}, 850 ${y0 + 150}, 862 ${y0 + 160}`); arrow(right, `M804 ${y0 + 235} C 850 ${y0 + 235}, 850 ${y0 + 190}, 862 ${y0 + 180}`);
    box(right, 866, y0 + 130, 110, 80, C.yellow); txt(right, 921, y0 + 166, 'guess vs', { 'text-anchor': 'middle', 'font-size': 15 }); txt(right, 921, y0 + 188, 'answer', { 'text-anchor': 'middle', 'font-size': 15 });
    const rnotes = [H(530, y0 + 320, '✓ the hidden blocks must be predicted', 't-accent'), H(530, y0 + 345, '✓ answers come from a slow copy, no gradient', 't-accent'), H(530, y0 + 370, '✓ each photo is graded on its own', 't-accent')];
    el('line', { x1: 500, y1: y0 + 40, x2: 500, y2: y0 + 380, stroke: 'var(--rule)', 'stroke-dasharray': '4 6' }, g);
    scenes.pseudo = {
      g,
      view: [0, y0 - 10, 1000, 410],
      beats: ['v1 learns by elimination', 'What makes it “pseudo”', 'A JEPA learns by filling in blanks', 'What v2 changes'],
      show(b) {
        show(left, true); lnotes.forEach(n => show(n, b >= 1));
        show(right, b >= 2); rnotes.forEach(n => show(n, b >= 3));
      }
    };
  }

  return scenes;
}
