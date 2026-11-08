// Runs infer.js off the main thread. Messages in:  {type:'load', base}  {id, type:'image', rgba}  {id, type:'text', text}
// Messages out: {type:'ready'}  {type:'error', message}  {id, ...result}
import { loadModel } from './infer.js';

let model = null;
self.onmessage = async ({ data }) => {
  try {
    if (data.type === 'load') {
      const url = name => `${data.base}data/${name}`;
      model = await loadModel(name => fetch(url(name)).then(r => r.arrayBuffer()), name => fetch(url(name)).then(r => r.json()));
      self.postMessage({ type: 'ready', scale: model.scale });
      return;
    }
    if (!model) throw new Error('model not loaded');
    const t0 = performance.now();
    if (data.type === 'image') {
      const { vec, qAttn } = model.embedImage(data.rgba);
      self.postMessage({ id: data.id, type: 'image', vec, qAttn, ms: performance.now() - t0 }, [vec.buffer, qAttn.buffer]);
    } else if (data.type === 'text') {
      const r = model.embedText(data.text);
      self.postMessage({ id: data.id, type: 'text', ...r, ms: performance.now() - t0 }, r.vec ? [r.vec.buffer] : []);
    }
  } catch (err) {
    self.postMessage({ id: data.id, type: 'error', message: String(err?.message || err) });
  }
};
