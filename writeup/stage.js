// The left panel: one big SVG "world" laid out like a two-lane map (photo lane on the left, caption lane on the
// right), a camera that glides to the part a passage is about, and a player that steps through that part's beats.
//
// A scene is { view: [x, y, w, h], beats: ['label', ...], show(beat) }. show() must be idempotent: it sets every
// element the scene owns to its state at that beat, and CSS transitions do the moving. That way ← and → can step
// backwards as easily as forwards.

export const NS = 'http://www.w3.org/2000/svg';
const reduceMotion = matchMedia('(prefers-reduced-motion: reduce)');

export function el(tag, attrs = {}, parent) {
  const node = document.createElementNS(NS, tag);
  for (const [k, v] of Object.entries(attrs)) if (v !== undefined && v !== null) node.setAttribute(k, v);
  if (parent) parent.append(node);
  return node;
}
export function txt(parent, x, y, value, attrs = {}) {
  const node = el('text', { x, y, ...attrs }, parent);
  node.textContent = value;
  return node;
}
export const group = (parent, cls) => el('g', cls ? { class: cls } : {}, parent);
export const toggle = (node, on, cls = 'is-off') => node.classList.toggle(cls, !on);
export const show = (node, on) => toggle(node, on);

// Excalidraw-ish box: pastel fill, ink outline, round corners.
export const box = (parent, x, y, w, h, fill, attrs = {}) =>
  el('rect', { x, y, width: w, height: h, rx: 9, fill, class: 'box', ...attrs }, parent);

// A pink tag like the drawing's labels ("Image", "Captions", ...).
export function chip(parent, cx, cy, label, { fill = 'var(--x-pink)', size = 15, color } = {}) {
  const g = group(parent, 'fade');
  const w = label.length * size * 0.52 + 18, h = size + 11;
  el('rect', { x: cx - w / 2, y: cy - h / 2, width: w, height: h, rx: 5, fill, class: 'chip-rect' }, g);
  txt(g, cx, cy + size * 0.35, label, { 'text-anchor': 'middle', 'font-size': size, fill: color });
  g.width = w;
  return g;
}

export function arrow(parent, d, cls = 'arrow') {
  return el('path', { d, class: `${cls} fade`, 'marker-end': `url(#${cls === 'arrow-red' ? 'head-red' : 'head'})` }, parent);
}

export function defs(svg) {
  const d = el('defs', {}, svg);
  for (const [id, color] of [['head', '#1a1a1a'], ['head-red', '#e03131'], ['head-blue', '#1f3a8a']]) {
    const m = el('marker', { id, viewBox: '0 0 10 10', refX: 8, refY: 5, markerWidth: 7, markerHeight: 7, orient: 'auto-start-reverse' }, d);
    el('path', { d: 'M0 0 L10 5 L0 10 z', fill: color }, m);
  }
  return d;
}

// ---- pictures from numbers -------------------------------------------------------------------------------------
export const loadImage = src => new Promise((resolve, reject) => {
  const img = new Image();
  img.onload = () => resolve(img);
  img.onerror = () => reject(new Error(`could not load ${src}`));
  img.src = src;
});

// A grayscale sprite as numbers: { w, h, at(x, y) → 0..255 }
export async function readSprite(src) {
  const img = await loadImage(src);
  const c = document.createElement('canvas');
  c.width = img.naturalWidth; c.height = img.naturalHeight;
  const g = c.getContext('2d', { willReadFrequently: true });
  g.drawImage(img, 0, 0);
  const data = g.getImageData(0, 0, c.width, c.height).data;
  return { w: c.width, h: c.height, img, at: (x, y) => data[(y * c.width + x) * 4] };
}

// Blue for positive, purple for negative, white for zero (kuangrad's colours). t in [-1, 1].
const POS = [31, 58, 138], NEG = [138, 59, 107];
export function diverging(t) {
  const a = Math.min(1, Math.sqrt(Math.abs(t)) * 1.05), c = t >= 0 ? POS : NEG;
  return [255 + (c[0] - 255) * a, 255 + (c[1] - 255) * a, 255 + (c[2] - 255) * a];
}
// rows × cols values in [-1, 1] → a data URL, one pixel per value (the SVG scales it, without smoothing).
export function matrixURL(rows, cols, valueAt) {
  const c = document.createElement('canvas');
  c.width = cols; c.height = rows;
  const g = c.getContext('2d');
  const im = g.createImageData(cols, rows);
  for (let r = 0; r < rows; r++) for (let k = 0; k < cols; k++) {
    const [R, G, B] = diverging(valueAt(r, k));
    const i = (r * cols + k) * 4;
    im.data[i] = R; im.data[i + 1] = G; im.data[i + 2] = B; im.data[i + 3] = 255;
  }
  g.putImageData(im, 0, 0);
  return c.toDataURL();
}
// part of a grayscale sprite, copied out as its own picture
export function cropURL(img, sx, sy, sw, sh, tint) {
  const c = document.createElement('canvas');
  c.width = sw; c.height = sh;
  const g = c.getContext('2d');
  g.drawImage(img, sx, sy, sw, sh, 0, 0, sw, sh);
  if (tint) { g.globalCompositeOperation = 'multiply'; g.fillStyle = tint; g.fillRect(0, 0, sw, sh); }
  return c.toDataURL();
}
export const image = (parent, href, x, y, w, h, attrs = {}) =>
  el('image', { href, x, y, width: w, height: h, preserveAspectRatio: 'none', 'image-rendering': 'pixelated', ...attrs }, parent);

// A strip of 256 numbers (one vector), coloured cell by cell.
export function vectorURL(vec, scale = Math.max(...vec.map(Math.abs))) {
  return matrixURL(1, vec.length, (_, k) => vec[k] / scale);
}

// ---- camera + player -------------------------------------------------------------------------------------------
export function createStage(host, { build, beatMs = 1900 }) {
  const svg = el('svg', { class: 'world', role: 'img', 'aria-label': 'Animated forward pass' });
  defs(svg);
  const world = group(svg);
  host.replaceChildren(svg);

  let view = [0, 0, 1000, 700], target = view, raf = 0;
  const fit = ([x, y, w, h]) => {
    const box = host.getBoundingClientRect();
    const aspect = (box.width || 1) / (box.height || 1);
    const pad = 0.04;
    let W = w * (1 + pad * 2), H = h * (1 + pad * 2);
    if (W / H < aspect) W = H * aspect; else H = W / aspect;
    return [x + w / 2 - W / 2, y + h / 2 - H / 2, W, H];
  };
  const apply = v => svg.setAttribute('viewBox', v.map(n => n.toFixed(2)).join(' '));
  function moveTo(rect, animate = true) {
    target = rect;
    const goal = fit(rect);
    cancelAnimationFrame(raf);
    if (!animate || reduceMotion.matches) { view = goal; apply(view); return; }
    const from = view.slice(), t0 = performance.now(), dur = 900;
    // zoom out a little in the middle of long moves, so the reader sees where we are going
    const far = Math.hypot(goal[0] - from[0], goal[1] - from[1]) > Math.max(from[2], goal[2]) * 0.8;
    const frame = now => {
      const p = Math.min(1, (now - t0) / dur), e = p < 0.5 ? 4 * p * p * p : 1 - (-2 * p + 2) ** 3 / 2;
      const bulge = far ? Math.sin(Math.PI * p) * 0.35 : 0;
      view = from.map((a, i) => a + (goal[i] - a) * e);
      const [x, y, w, h] = view, k = 1 + bulge;
      apply([x - (w * (k - 1)) / 2, y - (h * (k - 1)) / 2, w * k, h * k]);
      if (p < 1) raf = requestAnimationFrame(frame);
    };
    raf = requestAnimationFrame(frame);
  }
  new ResizeObserver(() => { view = fit(target); apply(view); }).observe(host);

  const scenes = build(world, svg);
  let scene = null, beat = 0, playing = false, timer = 0;
  const listeners = [];
  const emit = () => listeners.forEach(fn => fn({ scene, beat, playing, count: scene ? scene.beats.length : 0 }));

  function setBeat(i) {
    beat = i;
    scene.show(i);
    emit();
  }
  function schedule() {
    clearTimeout(timer);
    if (!playing) return;
    if (beat >= scene.beats.length - 1) { playing = false; emit(); return; }
    timer = setTimeout(() => { setBeat(beat + 1); schedule(); }, scene.beatMs || beatMs);
  }
  return {
    scenes,
    setScene(name, { animate = true } = {}) {
      const next = scenes[name];
      if (!next || next === scene) return;
      Object.values(scenes).forEach(s => s !== next && s.leave?.());
      scene = next;
      // the rest of the map stays faintly visible, for context
      const lit = new Set([scene, ...(scene.with || []).map(n => scenes[n])]);
      (scene.with || []).forEach(n => scenes[n].show(scenes[n].beats.length - 1));
      Object.values(scenes).forEach(s => { if (s.g) { s.g.style.transition = 'opacity .7s'; s.g.style.opacity = scene.all || lit.has(s) ? 1 : 0.13; } });
      moveTo(scene.view, animate);
      playing = !reduceMotion.matches && scene.beats.length > 1;
      setBeat(reduceMotion.matches ? scene.beats.length - 1 : 0);
      clearTimeout(timer);
      if (playing) timer = setTimeout(() => { setBeat(Math.min(1, scene.beats.length - 1)); schedule(); }, (scene.beatMs || beatMs) + 400);
    },
    play() {
      if (!scene) return;
      if (playing) { playing = false; clearTimeout(timer); emit(); return; }
      playing = true;
      if (beat >= scene.beats.length - 1) setBeat(0);
      schedule();
      emit();
    },
    // returns false at either end, so the caller can move to the previous or next passage instead
    step(delta) {
      if (!scene) return false;
      const next = beat + delta;
      if (next < 0 || next >= scene.beats.length) return false;
      playing = false; clearTimeout(timer);
      setBeat(next);
      return true;
    },
    onChange: fn => listeners.push(fn),
    svg, world
  };
}
