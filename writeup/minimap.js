// The hand-drawn diagram (assets/drawing.svg, exported from Excalidraw with its font) in the corner of the left
// panel, with a yellow box over the part the current passage is about. Regions are in the drawing's own units.
const W = 2859, H = 1369;
export const REGIONS = {
  all: [0, 0, W, H],
  data: [5, 140, 945, 880],
  image: [722, 140, 228, 245],
  conv: [1058, 140, 335, 295],
  pos: [1382, 140, 52, 295],
  'image-blocks': [1428, 140, 405, 295],
  'image-encoder': [1052, 100, 785, 340],
  embed: [5, 790, 1400, 290],
  'text-blocks': [1392, 765, 445, 325],
  target: [1928, 760, 270, 330],
  predictor: [2103, 355, 322, 465],
  score: [2208, 912, 322, 292],
  loss: [2203, 905, 652, 460]
};

export function createMinimap(figure) {
  const img = document.createElement('img');
  img.src = './assets/drawing.svg';
  img.alt = 'My sketch of the model: image and captions on the left, the Image Encoder and Text Encoder in the middle, the Predictor and the InfoNCE loss on the right';
  const mark = document.createElement('div');
  mark.className = 'mm-box';
  figure.prepend(img, mark);
  const size = figure.querySelector('.minimap-size');
  figure.addEventListener('click', () => {
    const big = figure.classList.toggle('is-big');
    size.setAttribute('aria-label', big ? 'Shrink the drawing' : 'Enlarge the drawing');
    size.textContent = big ? '⤡' : '⤢';
  });
  return {
    set(key) {
      const [x, y, w, h] = REGIONS[key] || REGIONS.all;
      Object.assign(mark.style, { left: `${(x / W) * 100}%`, top: `${(y / H) * 100}%`, width: `${(w / W) * 100}%`, height: `${(h / H) * 100}%` });
      mark.style.opacity = key === 'all' ? 0 : 1;
    }
  };
}
