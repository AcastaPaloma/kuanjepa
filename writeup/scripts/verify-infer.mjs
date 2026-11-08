// Checks infer.js against PyTorch: run `python writeup/scripts/export.py --featured N` first, then
//   node writeup/scripts/verify-infer.mjs
// Pass: cosine(JS, torch) > 0.995 for every image and caption, and every image-caption score within 0.02.
import { readFile } from 'node:fs/promises';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';
import { loadModel } from '../infer.js';

const here = dirname(fileURLToPath(import.meta.url));
const data = join(here, '..', 'data'), fix = join(here, 'fixtures');
const bytes = async p => { const b = await readFile(p); return b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength); };
const model = await loadModel(n => bytes(join(data, n)), async n => JSON.parse(await readFile(join(data, n), 'utf8')));
const expected = JSON.parse(await readFile(join(fix, 'expected.json'), 'utf8'));
const dot = (a, b) => a.reduce((s, v, i) => s + v * b[i], 0);

let ok = true, worst = 1, worstScore = 0;
const imgVecs = [];
for (const im of expected.images) {
  const rgba = new Uint8ClampedArray(await bytes(join(fix, im.file)));
  const t0 = performance.now();
  const { vec } = model.embedImage(rgba);
  const ms = performance.now() - t0;
  const c = dot(vec, im.vec);
  worst = Math.min(worst, c);
  imgVecs.push(vec);
  console.log(`image ${im.file}  cos ${c.toFixed(5)}  ${ms.toFixed(0)} ms`);
}
const txtVecs = [];
for (const cap of expected.captions) {
  const { vec, unknown } = model.embedText(cap.text);
  const c = dot(vec, cap.vec);
  worst = Math.min(worst, c);
  txtVecs.push(vec);
  console.log(`text   cos ${c.toFixed(5)}  "${cap.text.slice(0, 50)}"${unknown.length ? `  unknown: ${unknown}` : ''}`);
}
expected.scores.forEach((row, i) => row.forEach((s, j) => { worstScore = Math.max(worstScore, Math.abs(dot(imgVecs[i], txtVecs[j]) - s)); }));
if (worst <= 0.995 || worstScore >= 0.02) ok = false;
console.log(`\nworst cosine ${worst.toFixed(5)}, worst score diff ${worstScore.toFixed(5)}, scale ${model.scale.toFixed(3)}  ->  ${ok ? 'PASS' : 'FAIL'}`);
process.exit(ok ? 0 : 1);
