// The trained model's forward pass in plain JavaScript: the image encoder + predictor turn a 128 × 128 picture into a
// 256-number vector, the text encoder turns a caption into another, and their dot product is the score. Same maths as
// models/*.py (checked against PyTorch by scripts/verify-infer.mjs). No libraries, no DOM: runs in a worker or in node.
//
//   const model = await loadModel(name => fetch(`data/${name}`).then(r => r.arrayBuffer()),
//                                 name => fetch(`data/${name}`).then(r => r.json()));
//   model.embedImage(rgba)   // Uint8ClampedArray, 128 * 128 * 4  ->  { vec, qAttn }
//   model.embedText(text)    // -> { vec, words, ids, unknown }

const D = 256, HEADS = 4, HD = 64, FF = 1024;
const MEAN = [0.485, 0.456, 0.406], STD = [0.229, 0.224, 0.225];

// Abramowitz & Stegun 7.1.26, |error| < 1.5e-7: torch's GELU is the exact erf form.
function erf(x) {
  const s = x < 0 ? -1 : 1;
  x = Math.abs(x);
  const t = 1 / (1 + 0.3275911 * x);
  return s * (1 - ((((1.061405429 * t - 1.453152027) * t + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * Math.exp(-x * x));
}
const gelu = a => { for (let i = 0; i < a.length; i++) a[i] = 0.5 * a[i] * (1 + erf(a[i] * Math.SQRT1_2)); return a; };

// x: (n, inD) row-major; W: (outD, inD) like nn.Linear
function linear(x, n, inD, W, b, outD) {
  const y = new Float32Array(n * outD);
  for (let r = 0; r < n; r++) {
    const xo = r * inD, yo = r * outD;
    for (let o = 0; o < outD; o++) {
      const wo = o * inD;
      let s = b ? b[o] : 0;
      for (let i = 0; i < inD; i++) s += W[wo + i] * x[xo + i];
      y[yo + o] = s;
    }
  }
  return y;
}

function layerNorm(x, n, d, g, b, eps = 1e-6) {
  const y = new Float32Array(n * d);
  for (let r = 0; r < n; r++) {
    const o = r * d;
    let m = 0;
    for (let i = 0; i < d; i++) m += x[o + i];
    m /= d;
    let v = 0;
    for (let i = 0; i < d; i++) { const t = x[o + i] - m; v += t * t; }
    const inv = 1 / Math.sqrt(v / d + eps);
    for (let i = 0; i < d; i++) y[o + i] = (x[o + i] - m) * inv * g[i] + b[i];
  }
  return y;
}

// Conv2d(k=3, s=2, p=1). x: (Cin, S, S) -> (Cout, S/2, S/2)
function conv(x, cin, S, W, b, cout) {
  const O = S / 2, OO = O * O, y = new Float32Array(cout * OO);
  for (let co = 0; co < cout; co++) {
    const yo = co * OO;
    y.fill(b[co], yo, yo + OO);
    for (let ci = 0; ci < cin; ci++) {
      const xo = ci * S * S, wo = (co * cin + ci) * 9;
      for (let ky = 0; ky < 3; ky++) for (let kx = 0; kx < 3; kx++) {
        const w = W[wo + ky * 3 + kx];
        for (let oy = 0; oy < O; oy++) {
          const iy = oy * 2 - 1 + ky;
          if (iy < 0 || iy >= S) continue;
          const row = xo + iy * S, yr = yo + oy * O;
          for (let ox = 0; ox < O; ox++) {
            const ix = ox * 2 - 1 + kx;
            if (ix >= 0 && ix < S) y[yr + ox] += w * x[row + ix];
          }
        }
      }
    }
  }
  return y;
}

// GroupNorm(8, C), eps 1e-5, in place. x: (C, HW)
function groupNorm(x, C, HW, g, b, groups = 8, eps = 1e-5) {
  const per = C / groups, n = per * HW;
  for (let gi = 0; gi < groups; gi++) {
    const o = gi * n;
    let m = 0;
    for (let i = 0; i < n; i++) m += x[o + i];
    m /= n;
    let v = 0;
    for (let i = 0; i < n; i++) { const t = x[o + i] - m; v += t * t; }
    const inv = 1 / Math.sqrt(v / n + eps);
    for (let c = 0; c < per; c++) {
      const ch = gi * per + c, co = ch * HW;
      for (let i = 0; i < HW; i++) x[co + i] = (x[co + i] - m) * inv * g[ch] + b[ch];
    }
  }
  return x;
}

// Pre-norm transformer block: x += attn(LN(x)); x += mlp(LN(x)). Returns attention probabilities (heads, n, n).
function block(x, n, w) {
  const h = layerNorm(x, n, D, w.norm1.weight, w.norm1.bias);
  const qkv = linear(h, n, D, w.attn.qkv.weight, w.attn.qkv.bias, 3 * D);
  const out = new Float32Array(n * D), probs = new Float32Array(HEADS * n * n), s = new Float32Array(n);
  for (let hh = 0; hh < HEADS; hh++) {
    for (let i = 0; i < n; i++) {
      const qo = i * 3 * D + hh * HD;
      let mx = -Infinity;
      for (let j = 0; j < n; j++) {
        const ko = j * 3 * D + D + hh * HD;
        let dot = 0;
        for (let d = 0; d < HD; d++) dot += qkv[qo + d] * qkv[ko + d];
        s[j] = dot / 8;
        if (s[j] > mx) mx = s[j];
      }
      let sum = 0;
      for (let j = 0; j < n; j++) { s[j] = Math.exp(s[j] - mx); sum += s[j]; }
      const po = (hh * n + i) * n, oo = i * D + hh * HD;
      for (let j = 0; j < n; j++) {
        const p = s[j] / sum;
        probs[po + j] = p;
        const vo = j * 3 * D + 2 * D + hh * HD;
        for (let d = 0; d < HD; d++) out[oo + d] += p * qkv[vo + d];
      }
    }
  }
  const a = linear(out, n, D, w.attn.proj.weight, w.attn.proj.bias, D);
  for (let i = 0; i < x.length; i++) x[i] += a[i];
  const h2 = layerNorm(x, n, D, w.norm2.weight, w.norm2.bias);
  const m = linear(gelu(linear(h2, n, D, w.mlp[0].weight, w.mlp[0].bias, FF)), n, FF, w.mlp[2].weight, w.mlp[2].bias, D);
  for (let i = 0; i < x.length; i++) x[i] += m[i];
  return probs;
}

function normalize(v) {
  let s = 0;
  for (let i = 0; i < v.length; i++) s += v[i] * v[i];
  const inv = 1 / Math.max(Math.sqrt(s), 1e-12);
  for (let i = 0; i < v.length; i++) v[i] *= inv;
  return v;
}

const f16 = new Float32Array(65536);
for (let h = 0; h < 65536; h++) {
  const s = h & 0x8000 ? -1 : 1, e = (h >> 10) & 31, f = h & 1023;
  f16[h] = e === 0 ? s * 2 ** -14 * (f / 1024) : e === 31 ? (f ? NaN : s * Infinity) : s * 2 ** (e - 15) * (1 + f / 1024);
}

// "a.b.0.weight" -> nested object a.b[0].weight
function nest(flat) {
  const root = {};
  for (const [name, t] of Object.entries(flat)) {
    const parts = name.split('.');
    let o = root;
    for (let i = 0; i < parts.length - 1; i++) o = o[parts[i]] ??= {};
    o[parts.at(-1)] = t;
  }
  // objects keyed 0, 1, 2... (blocks, Sequential layers) become arrays
  const arrays = o => {
    if (o instanceof Float32Array || typeof o !== 'object') return o;
    for (const k of Object.keys(o)) o[k] = arrays(o[k]);
    const keys = Object.keys(o);
    return keys.length && keys.every((k, i) => k === String(i)) ? keys.map(k => o[k]) : o;
  };
  return arrays(root);
}

export async function loadModel(fetchBytes, fetchJson) {
  const [meta, buf, vocab] = await Promise.all([fetchJson('model.json'), fetchBytes('model.bin'), fetchJson('vocab.json')]);
  const flat = {};
  let emb = null;
  for (const t of meta.tensors) {
    const size = t.shape.reduce((a, b) => a * b, 1);
    if (t.dtype === 'f32') { flat[t.name] = new Float32Array(buf.slice(t.offset, t.offset + 4 * size)); continue; }
    const rows = t.shape[0], cols = size / rows;
    const scales = new Float32Array(buf.slice(t.scaleOffset, t.scaleOffset + 4 * rows));
    if (t.dtype === 'int8') {
      const q = new Int8Array(buf, t.offset, size);
      if (t.name === 'text_encoder.tok_emb.weight') { emb = { q: q.slice(), scales, cols }; continue; } // dequantised per lookup
      const a = new Float32Array(size);
      for (let r = 0; r < rows; r++) for (let c = 0; c < cols; c++) a[r * cols + c] = q[r * cols + c] * scales[r];
      flat[t.name] = a;
    } else if (t.dtype === 'f16') {
      const h = new Uint16Array(buf.slice(t.offset, t.offset + 2 * size)), a = new Float32Array(size);
      for (let i = 0; i < size; i++) a[i] = f16[h[i]];
      flat[t.name] = a;
    }
  }
  const W = nest(flat);
  const ie = W.image_encoder, te = W.text_encoder, pr = W.predictor;
  const index = new Map(vocab.map((w, i) => [w, i]));
  const maxLen = meta.max_len;

  const tokenize = text => {
    const words = (String(text).toLowerCase().match(/[a-z0-9'’]+/g) || []).slice(0, maxLen);
    const ids = words.map(w => index.get(w) ?? 1);
    return { words, ids, unknown: words.filter((w, i) => ids[i] === 1) };
  };

  function embedText(text) {
    const { words, ids, unknown } = tokenize(text);
    const n = ids.length;
    if (!n) return { vec: null, words, ids, unknown };
    const x = new Float32Array(n * D);
    for (let r = 0; r < n; r++) {
      const o = ids[r] * D, s = emb.scales[ids[r]];
      for (let c = 0; c < D; c++) x[r * D + c] = emb.q[o + c] * s + te.pos_emb[r * D + c];
    }
    for (const b of te.blocks) block(x, n, b);
    const h = layerNorm(x, n, D, te.ln.weight, te.ln.bias);
    const v = new Float32Array(D);
    for (let r = 0; r < n; r++) for (let c = 0; c < D; c++) v[c] += h[r * D + c] / n;
    return { vec: normalize(v), words, ids, unknown };
  }

  function embedImage(rgba) {
    const S = 128;
    let x = new Float32Array(3 * S * S);
    for (let p = 0; p < S * S; p++) for (let c = 0; c < 3; c++) x[c * S * S + p] = (rgba[p * 4 + c] / 255 - MEAN[c]) / STD[c];
    const chans = [3, 64, 128, 256, 256];
    let size = S;
    for (let k = 1; k <= 4; k++) {
      const st = ie[`convStem${k}`];
      x = conv(x, chans[k - 1], size, st[0].weight, st[0].bias, chans[k]);
      size /= 2;
      groupNorm(x, chans[k], size * size, st[1].weight, st[1].bias);
      if (k < 4) gelu(x);
    }
    // (256, 8, 8) -> 64 tokens of 256, row-major over the grid, plus the position embedding
    const N = 64, t = new Float32Array(N * D);
    for (let p = 0; p < N; p++) for (let c = 0; c < D; c++) t[p * D + c] = x[c * N + p] + ie.pos_emb[p * D + c];
    for (const b of ie.blocks) block(t, N, b);
    const v = layerNorm(t, N, D, ie.ln.weight, ie.ln.bias);

    // predictor: Linear on the 64 tokens, the learned query q appended last, 2 blocks, read q back out
    const P = N + 1, p = new Float32Array(P * D);
    p.set(linear(v, N, D, pr.linear1.weight, pr.linear1.bias, D));
    p.set(pr.q, N * D);
    const qAttn = new Float32Array(2 * N);
    pr.blocks.forEach((b, bi) => {
      const probs = block(p, P, b);
      for (let hh = 0; hh < HEADS; hh++) for (let j = 0; j < N; j++) qAttn[bi * N + j] += probs[(hh * P + N) * P + j] / HEADS;
    });
    const q = layerNorm(p.subarray(N * D), 1, D, pr.ln.weight, pr.ln.bias);
    return { vec: normalize(linear(q, 1, D, pr.linear2.weight, pr.linear2.bias, D)), qAttn };
  }

  return { tokenize, embedText, embedImage, scale: meta.scale, vocab };
}
