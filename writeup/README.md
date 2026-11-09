# pseudo-jepa write-up

A static, scroll-driven walkthrough of the model, served at kuant.space/pseudo-jepa/. The article is on the right; on
the left, one real photo and one real caption go through the network, region by region, with my drawing in the corner
as a minimap.

**To change the words, edit `content/*.md`.** No build step: the pages read these files as they are.

- `content/v1.md`: the v1 write-up (the scroll-driven forward pass)
- `content/v2.md`: the v2 write-up, with v1 | v2 | VL-JEPA tables in each step. Tables are plain `| a | b |` rows
- `content/results-v1.md`, `content/results-v2.md`: the Results tabs
- Each heading is one reading step. The `<!-- id · chapter · scene · map -->` line under it picks the animation and
  the part of the drawing to frame; leave `scene` and `map` names as they are.

Code:

- `app.js`: the reader (scroll tracking, Contents, arrow keys); `md.js`: markdown → steps
- `stage.js`: the SVG world, the camera and the beat player; `scenes-v1.js`, `scenes-v2.js`: what each scene draws
- `minimap.js` + `assets/drawing.svg`: the Excalidraw drawing with the yellow box; `assets/fonts/`: Excalifont (OFL)
- `results.html`, `results.js`, `results.css`: the live demo and the training curves
- `infer.js`, `worker.js`: the forward pass in plain JavaScript, run in a Web Worker
- `data/`: generated, do not edit by hand. `scripts/export.py` writes v1's (from `checkpoints/best.pt` and the training log); `scripts/export_v2.py` writes `v2trace.json` and `v2/` from the frozen V-JEPA 2, the cached EmbeddingGemma targets and `v2/checkpoints/best_masked.pt`. Rerun it when v2 finishes to show the final predictor
- `vendor/katex/`: KaTeX (MIT)

Preview locally:

```sh
python3 -m http.server -d writeup 8000   # then open http://localhost:8000
```

Regenerate `data/` after a new checkpoint, then check the JavaScript forward pass against PyTorch:

```sh
python writeup/scripts/export.py --featured 140
node writeup/scripts/verify-infer.mjs
```

The site copies this folder from `main` at build time (`scripts/sync-projects.mjs` in the portfolio repo).
