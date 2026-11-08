// The line between the two panels can be dragged (or moved with the arrow keys once focused; double-click resets it).
// The split is shared by the write-up and the Results tab, and remembered per browser.
const workspace = document.querySelector('.workspace');
const notes = document.querySelector('.notes');
const KEY = 'pseudo-jepa-split', DEFAULT = 1.12 / 2.12, MIN_PX = 340;

const handle = document.createElement('div');
handle.className = 'splitter';
handle.tabIndex = 0;
handle.setAttribute('role', 'separator');
handle.setAttribute('aria-orientation', 'vertical');
handle.setAttribute('aria-label', 'Resize the panels');
notes.prepend(handle);

const clamp = f => {
  const width = workspace.clientWidth || innerWidth;
  const lo = Math.max(0.25, MIN_PX / width), hi = Math.min(0.75, 1 - MIN_PX / width);
  return Math.min(Math.max(f, lo), Math.max(lo, hi));
};
let split = DEFAULT;
const apply = (f, remember = false) => {
  split = clamp(f);
  workspace.style.setProperty('--split', `${(split * 100).toFixed(2)}%`);
  handle.setAttribute('aria-valuenow', Math.round(split * 100));
  dispatchEvent(new Event('resize')); // the write-up re-measures which passage is current
  if (remember) try { localStorage.setItem(KEY, String(split)); } catch {}
};
try { const stored = Number(localStorage.getItem(KEY)); if (stored) apply(stored); } catch {}

handle.addEventListener('pointerdown', event => {
  if (event.button !== 0) return;
  event.preventDefault();
  handle.setPointerCapture(event.pointerId);
  document.body.classList.add('is-splitting');
  const box = workspace.getBoundingClientRect();
  const move = e => apply((e.clientX - box.left) / box.width);
  const up = () => {
    handle.removeEventListener('pointermove', move);
    document.body.classList.remove('is-splitting');
    apply(split, true);
  };
  handle.addEventListener('pointermove', move);
  handle.addEventListener('pointerup', up, { once: true });
  handle.addEventListener('pointercancel', up, { once: true });
});
handle.addEventListener('dblclick', () => apply(DEFAULT, true));
handle.addEventListener('keydown', event => {
  const delta = { ArrowLeft: -0.02, ArrowRight: 0.02 }[event.key];
  if (!delta) return;
  event.preventDefault();
  event.stopPropagation(); // not the write-up's previous / next passage
  apply(split + delta, true);
});
