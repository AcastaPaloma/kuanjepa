import { parseSteps, rich, escapeHtml } from './md.js';
import { createStage, readSprite, loadImage } from './stage.js';
import { createMinimap } from './minimap.js';
import './split.js';

const version = document.body.dataset.version; // v1 or v2
const reader = document.getElementById('reader');
const article = document.getElementById('article');
const counter = document.getElementById('step-counter');
const toggle = document.getElementById('contents-toggle');
const menu = document.getElementById('contents-menu');
const visual = document.querySelector('.visual');
const beatLabel = document.getElementById('beat-label');
const playButton = document.getElementById('beat-play');
const reduceMotion = matchMedia('(prefers-reduced-motion: reduce)');

const getJson = url => fetch(url).then(r => { if (!r.ok) throw new Error(`${url}: ${r.status}`); return r.json(); });
const katexReady = new Promise(resolve => {
  if (window.katex) return resolve();
  document.querySelector('script[src*="katex"]').addEventListener('load', resolve, { once: true });
});

const [source, trace, metrics] = await Promise.all([
  fetch(`./content/${version}.md`, { cache: 'no-cache' }).then(r => r.text()),
  getJson('./data/trace.json'),
  getJson('./data/metrics.json'),
  katexReady
]);
const steps = parseSteps(source);

// --- prose -------------------------------------------------------------------------------------------------------
let chapter = '';
const chapters = steps.map(step => (chapter = step.chapter || chapter));
const blocks = steps.map((step, i) => {
  const section = document.createElement('section');
  section.className = `reading-block level-${step.level}${step.chapter && i ? ' starts-chapter' : ''}`;
  section.id = step.id;
  const heading = step.first
    ? `<header class="title-block"><h1><span>${escapeHtml(step.title.split(',')[0])}</span>${escapeHtml(step.title.slice(step.title.split(',')[0].length))}</h1>
       <p class="subtitle">${version === 'v1' ? 'Photos and captions, built from scratch, followed through one forward pass' : 'Fill in the blanks: the JEPA version, planned'}</p>
       <p class="byline">Kuan Yi Wang · <a href="https://github.com/AcastaPaloma/kuanjepa" target="_blank" rel="noopener">source on GitHub</a></p></header>`
    : `${step.chapter ? `<p class="chapter-kicker">${escapeHtml(step.chapter)}</p>` : ''}<h${step.level}>${rich(step.title)}</h${step.level}>`;
  section.innerHTML = heading + step.html;
  article.append(section);
  return section;
});
const end = document.createElement('footer');
end.className = 'article-end';
end.innerHTML = version === 'v1'
  ? 'Every picture on the left is recorded from the trained checkpoint by <code>writeup/scripts/export.py</code>: the same weights the Results tab runs in your browser. <a href="https://github.com/AcastaPaloma/kuanjepa" target="_blank" rel="noopener">The code is on GitHub</a>.'
  : 'v2 is a plan. When it is trained, its Results tab gets the same live demo as v1. <a href="./">Back to v1</a>.';
article.append(end);

// --- contents ----------------------------------------------------------------------------------------------------
const toc = document.getElementById('toc');
toc.innerHTML = steps.map((step, i) => `${step.chapter ? `<p class="toc-chapter">${escapeHtml(step.chapter)}</p>` : ''}
  <button class="toc-link" type="button" data-index="${i}"><span class="toc-number">${String(i + 1).padStart(2, '0')}</span><span>${rich(step.title)}</span></button>`).join('');
const tocLinks = [...toc.querySelectorAll('.toc-link')];
const setMenu = open => { menu.hidden = !open; toggle.setAttribute('aria-expanded', String(open)); };
toggle.addEventListener('click', () => setMenu(menu.hidden));
document.addEventListener('keydown', event => { if (event.key === 'Escape' && !menu.hidden) { setMenu(false); toggle.focus(); } });
document.addEventListener('pointerdown', event => { if (!menu.hidden && !menu.contains(event.target) && !toggle.contains(event.target)) setMenu(false); });
const goTo = (index, smooth = true) => {
  const next = blocks[index + 1];
  const top = index === 0 ? 0 : Math.min(blocks[index].offsetTop - 24, next ? next.offsetTop - reader.clientHeight * 0.38 - 8 : Infinity);
  reader.scrollTo({ top, behavior: smooth && !reduceMotion.matches ? 'smooth' : 'auto' });
};
tocLinks.forEach(link => link.addEventListener('click', () => {
  const index = Number(link.dataset.index);
  setMenu(false);
  goTo(index);
  history.replaceState(null, '', `#${steps[index].id}`);
}));

// --- left panel --------------------------------------------------------------------------------------------------
const minimap = createMinimap(document.getElementById('minimap'));
const [imgTokens, txtTokens, attn, input, ...conv] = await Promise.all([
  readSprite('./data/' + trace.tokens.sprite), readSprite('./data/' + trace.text.sprite), readSprite('./data/' + trace.attn.sprite),
  loadImage('./data/' + trace.image.input), ...trace.conv.map(c => readSprite('./data/' + c.sprite))
]);
const { buildV1 } = await import('./scenes-v1.js');
const build = buildV1; // v2 is a plain comparison page now (v2.html, compare.js)
const stage = createStage(document.getElementById('stage'), {
  build: world => build(world, { trace, metrics, sprites: { imgTokens, txtTokens, attn, input, conv } })
});
stage.onChange(({ scene, beat, playing, count }) => {
  beatLabel.innerHTML = `<span class="n">${beat + 1}/${count}</span>${escapeHtml(scene.beats[beat])}`;
  playButton.textContent = playing ? 'Pause' : beat >= count - 1 ? 'Replay' : 'Play';
});
playButton.addEventListener('click', () => stage.play());
document.getElementById('beat-prev').addEventListener('click', () => stage.step(-1) || goTo(Math.max(0, current - 1)));
document.getElementById('beat-next').addEventListener('click', () => stage.step(1) || goTo(Math.min(steps.length - 1, current + 1)));

let current = -1;
function show(index) {
  if (index === current) return;
  current = index;
  const step = steps[index];
  stage.setScene(step.scene, { animate: document.body.classList.contains('is-ready') });
  minimap.set(step.map);
  blocks.forEach((block, i) => block.classList.toggle('is-current', i === index));
  tocLinks.forEach((link, i) => i === index ? link.setAttribute('aria-current', 'location') : link.removeAttribute('aria-current'));
  counter.textContent = `${String(index + 1).padStart(2, '0')} / ${steps.length}`;
  visual.setAttribute('aria-label', `${chapters[index]}: ${step.title}. Left and right arrows step through it`);
}

// The current step is the last passage whose top has crossed 38% of the reading column.
function measure() {
  const box = reader.getBoundingClientRect();
  const line = box.top + box.height * 0.38;
  let index = 0;
  blocks.forEach((block, i) => { if (block.getBoundingClientRect().top <= line) index = i; });
  if (reader.scrollTop + reader.clientHeight >= reader.scrollHeight - 2) index = blocks.length - 1;
  show(index);
}
let queued = false;
const schedule = () => {
  if (queued) return;
  queued = true;
  requestAnimationFrame(() => { queued = false; measure(); });
};
reader.addEventListener('scroll', schedule, { passive: true });
addEventListener('resize', schedule);

// ← and → move to the previous or next passage. With the left panel focused, they first step through its beats.
let aim = null, aimTimer = 0;
const settle = () => { clearTimeout(aimTimer); aimTimer = setTimeout(() => { aim = null; }, 250); };
reader.addEventListener('scroll', settle, { passive: true });
document.addEventListener('keydown', event => {
  const delta = { ArrowLeft: -1, ArrowRight: 1 }[event.key];
  if (!delta || event.altKey || event.ctrlKey || event.metaKey || event.shiftKey || !menu.hidden) return;
  if (event.target.closest?.('input, textarea, select, [contenteditable]')) return;
  event.preventDefault();
  if (aim === null && visual.contains(document.activeElement) && stage.step(delta)) return;
  aim = Math.max(0, Math.min((aim ?? current) + delta, steps.length - 1));
  settle();
  goTo(aim);
});

const start = steps.findIndex(step => `#${step.id}` === location.hash);
if (start > 0) goTo(start, false);
requestAnimationFrame(() => requestAnimationFrame(() => {
  measure();
  document.body.classList.add('is-ready');
}));
