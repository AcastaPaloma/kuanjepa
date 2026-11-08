"""Tiny web UI: pick an image (random val image or your own upload), type a caption,
get the JEPA image<->text similarity live.

usage:
    python app.py                       # http://localhost:8000
    python app.py --ckpt checkpoints/best.pt --port 8000

Only uses the standard library + the project's own modules.
"""
import argparse
import base64
import io
import json
import random
from http.server import BaseHTTPRequestHandler, HTTPServer

import torch
import torch.nn.functional as F
from PIL import Image

from v1.dataloader import DataLoader
from v1.models.image_encoder import ImageEncoder
from v1.models.text_encoder import TextEncoder
from v1.models.predictor import Predictor
from v1.utilities import masked_mean

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


class Model:
    def __init__(self, ckpt_path):
        self.DL = DL = DataLoader()
        DL._setup_text()
        ck = torch.load(ckpt_path, map_location=device)
        self.ie = ImageEncoder().to(device); self.ie.load_state_dict(ck["image_encoder"])
        self.te = TextEncoder(len(DL.vocab), DL.max_len).to(device); self.te.load_state_dict(ck["text_encoder"])
        self.pr = Predictor().to(device); self.pr.load_state_dict(ck["predictor"])
        for m in (self.ie, self.te, self.pr):
            m.eval()
        self.info = f"{ckpt_path} (epoch {ck['epoch']}, val_loss {ck['val_loss']:.3f})"

        # reference pool: every caption of every val image, used to turn a raw score into a rank
        self.val = DL.val_idx
        self.ref_caps, self.ref_owner = [], []
        for n, i in enumerate(self.val):
            for c in DL.captions[i]:
                self.ref_caps.append(c); self.ref_owner.append(n)
        self.ref_emb = torch.cat([self.embed_text(self.ref_caps[s:s + 512]) for s in range(0, len(self.ref_caps), 512)])

        self.img_emb = None          # embedding of the currently selected image
        self.img_val_n = None        # its position in val, if it is a val image
        print(f"loaded {self.info}; {len(self.ref_caps)} reference captions")

    @torch.no_grad()
    def embed_text(self, texts):
        ids, mask = map(torch.stack, zip(*[self.DL.encode(t) for t in texts]))
        ids, mask = ids.to(device), mask.to(device)
        return F.normalize(masked_mean(self.te(ids, mask), mask), p=2, dim=-1)

    @torch.no_grad()
    def set_image(self, pil, val_n=None):
        x = self.DL.val_tfm(pil.convert("RGB")).unsqueeze(0).to(device)
        self.img_emb = self.pr(self.ie(x)).squeeze(1).squeeze(0)       # (256,)
        self.img_val_n = val_n
        ref_scores = self.ref_emb @ self.img_emb
        top = ref_scores.topk(3)
        return [{"caption": self.ref_caps[j], "score": round(s, 4),
                 "own": val_n is not None and self.ref_owner[j] == val_n}
                for s, j in zip(top.values.tolist(), top.indices.tolist())]

    def score(self, caption):
        if self.img_emb is None:
            return {"error": "pick an image first"}
        words = self.DL._tokenize_str(caption)
        if not words:
            return {"error": "type a caption"}
        unk = self.DL.vocab["<unk>"]
        oov = [w for w in words if self.DL.vocab.get(w, unk) == unk]
        s = (self.embed_text([caption]) @ self.img_emb).item()
        ref = self.ref_emb @ self.img_emb
        rank = int((ref > s).sum().item()) + 1
        return {"score": round(s, 4), "rank": rank, "pool": len(self.ref_caps),
                "percentile": round(100 * (ref < s).float().mean().item(), 1),
                "ref_min": round(ref.min().item(), 4), "ref_max": round(ref.max().item(), 4),
                "unknown_words": oov}


def to_data_url(pil):
    pil = pil.convert("RGB"); pil.thumbnail((640, 640))
    buf = io.BytesIO(); pil.save(buf, format="JPEG", quality=88)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


PAGE = r"""<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>JEPA similarity</title>
<style>
 :root{--bg:#f7f7f5;--fg:#1d1d1b;--mut:#8a8a84;--panel:#fff;--line:#e3e3de}
 @media (prefers-color-scheme:dark){:root{--bg:#161615;--fg:#ececea;--mut:#77776f;--panel:#1f1f1d;--line:#33332f}}
 *{box-sizing:border-box}
 body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.45 system-ui,sans-serif;min-height:100vh;display:flex}
 main{flex:1;min-width:0;display:flex;flex-direction:column;align-items:center;justify-content:center;gap:18px;padding:32px 16px}
 #drop{width:min(520px,100%);aspect-ratio:4/3;border-radius:10px;overflow:hidden;background:var(--line);cursor:pointer;outline:2px dashed transparent;outline-offset:4px}
 #drop.over{outline-color:var(--mut)}
 #img{width:100%;height:100%;object-fit:contain;display:block}
 #cap{width:min(520px,100%);font:inherit;font-size:17px;padding:12px 14px;border-radius:8px;border:1px solid var(--line);background:var(--panel);color:var(--fg);outline:none}
 #cap:focus{border-color:var(--mut)}
 #score{font-size:56px;font-weight:650;font-variant-numeric:tabular-nums;letter-spacing:-1px;min-height:1.2em}
 aside{position:relative;background:var(--panel);border-left:1px solid var(--line);transition:margin-right .2s}
 aside .inner{padding:28px 22px;width:max-content;max-width:90vw}
 aside ol{margin:0;padding:0;list-style:none}
 aside li{font-weight:700;font-size:13px;white-space:nowrap;padding:9px 0;border-top:1px solid var(--line);cursor:pointer}
 aside li:first-child{border-top:0}
 #fold{position:absolute;left:-28px;top:50%;transform:translateY(-50%);width:28px;height:56px;border:1px solid var(--line);border-right:0;border-radius:8px 0 0 8px;background:var(--panel);color:var(--mut);cursor:pointer;font-size:14px;padding:0}
 body.folded aside .inner{display:none}
 body.folded aside{border-left:0}
</style></head><body>
<main>
  <div id="drop" title="Click for a random val image · drop an image file to upload"><img id="img" alt=""></div>
  <input id="cap" placeholder="Type a caption…" autocomplete="off">
  <div id="score">–</div>
  <input id="file" type="file" accept="image/*" hidden>
</main>
<aside>
  <button id="fold" title="Fold panel">›</button>
  <div class="inner"><ol id="top"></ol></div>
</aside>
<script>
const $=id=>document.getElementById(id);
async function api(path,body){const r=await fetch(path,{method:body?'POST':'GET',body});return r.json()}
function setFold(f){document.body.classList.toggle('folded',f);$('fold').textContent=f?'‹':'›';try{localStorage.setItem('fold',f?'1':'')}catch(e){}}
try{setFold(!!localStorage.getItem('fold'))}catch(e){}
$('fold').onclick=()=>setFold(!document.body.classList.contains('folded'));
function showImage(d){
  $('img').src=d.image;
  $('top').innerHTML='';
  d.top.slice(0,3).forEach(t=>{const li=document.createElement('li');li.textContent=t.caption;
    li.onclick=()=>{$('cap').value=t.caption;score()};$('top').appendChild(li)});
  score();
}
async function score(){
  const c=$('cap').value.trim();
  if(!c){$('score').textContent='–';return}
  const r=await api('/api/score',JSON.stringify({caption:c}));
  $('score').textContent=r.error?'–':r.score.toFixed(3);
}
let timer;$('cap').oninput=()=>{clearTimeout(timer);timer=setTimeout(score,200)};
const upload=async f=>{if(f&&f.type.startsWith('image/'))showImage(await api('/api/upload',f))};
const drop=$('drop');
drop.onclick=async e=>{if(e.shiftKey){$('file').click();return}showImage(await api('/api/random'))};
$('file').onchange=e=>upload(e.target.files[0]);
drop.ondragover=e=>{e.preventDefault();drop.classList.add('over')};
drop.ondragleave=()=>drop.classList.remove('over');
drop.ondrop=e=>{e.preventDefault();drop.classList.remove('over');upload(e.dataTransfer.files[0])};
window.onpaste=e=>{const f=[...e.clipboardData.files][0];if(f)upload(f)};
api('/api/random').then(showImage);
</script></body></html>"""


def make_handler(M):
    class H(BaseHTTPRequestHandler):
        def _send(self, code, body, ctype="application/json"):
            data = body if isinstance(body, bytes) else (body if isinstance(body, str) else json.dumps(body)).encode()
            self.send_response(code); self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(data))); self.end_headers(); self.wfile.write(data)

        def log_message(self, *a):
            pass

        def do_GET(self):
            if self.path == "/":
                return self._send(200, PAGE, "text/html; charset=utf-8")
            if self.path == "/api/info":
                return self._send(200, {"info": M.info})
            if self.path == "/api/random":
                n = random.randrange(len(M.val))
                pil = M.DL.ds["test"][M.val[n]]["image"]
                top = M.set_image(pil, val_n=n)
                return self._send(200, {"image": to_data_url(pil), "captions": M.DL.captions[M.val[n]], "top": top})
            self._send(404, {"error": "not found"})

        def do_POST(self):
            body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
            try:
                if self.path == "/api/upload":
                    pil = Image.open(io.BytesIO(body))
                    top = M.set_image(pil)
                    return self._send(200, {"image": to_data_url(pil), "captions": None, "top": top})
                if self.path == "/api/score":
                    return self._send(200, M.score(json.loads(body)["caption"]))
            except Exception as e:
                return self._send(400, {"error": str(e)})
            self._send(404, {"error": "not found"})
    return H


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/best.pt")
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()
    M = Model(args.ckpt)
    srv = HTTPServer(("127.0.0.1", args.port), make_handler(M))   # single-threaded: one model, one GPU queue
    print(f"open http://localhost:{args.port}")
    srv.serve_forever()
