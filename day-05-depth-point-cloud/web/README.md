# Photo → Point Cloud (web)

The Day 05 notebook, rebuilt as a static website that runs **entirely in the browser**.
Hosted free on Cloudflare Pages, with no server; photos never leave the visitor's device.

## How it works

| Step | Notebook (Python) | Website (browser) |
|---|---|---|
| Model | PyTorch, `transformers` | **ONNX Runtime Web**, WebGPU (CPU/WebAssembly fallback) |
| Weights | 99 MB fp32 from Hugging Face | **52 MB fp16**, split into 3 parts (Pages limit is 25 MB per file), cached after first visit |
| Preprocess | `AutoImageProcessor` | same maths in JS: keep aspect, ~518 px, sides a multiple of 14, ImageNet normalise |
| 3D | NumPy back-projection | same pinhole formulas, per point |
| Viewer | Plotly | **Three.js**, with point size scaled by distance so far areas don't fall apart |
| Export | Open3D `.ply` | binary `.ply` written in JS (metres, Z-up) |

**Accuracy check:** the fp16 ONNX model matches PyTorch to within 0.1% mean error, on landscape, portrait and square images.
(int8 was tried too: 29 MB but ~7% off, so it wasn't used.)

## Files

```
web/
├── public/              ← the site (this folder is what gets deployed)
│   ├── index.html
│   ├── app.js
│   ├── _headers         ← enables multi-threaded WebAssembly + caches the model
│   └── model/           ← fp16 ONNX model in 20 MB parts + manifest.json
├── tools/
│   ├── export_onnx.py       PyTorch → ONNX (dynamic image size)
│   ├── verify_and_shrink.py checks ONNX vs PyTorch, makes fp16 / int8
│   ├── split_model.py       cuts the model into <25 MB parts
│   └── serve.py             local preview at http://localhost:8788
└── wrangler.jsonc       ← Cloudflare Pages config
```

## Rebuild the model (only if you change it)

From `day-05-depth-point-cloud/` (pip install `onnx onnxruntime onnxscript` into the venv first):

```bash
.venv/bin/python web/tools/export_onnx.py
.venv/bin/python web/tools/verify_and_shrink.py
.venv/bin/python web/tools/split_model.py
```

One fix was needed for export: the model's output head wraps the image size in `int()`,
which freezes every output to the example image's size. `export_onnx.py` swaps in the same
head without `int()`, so any aspect ratio works.

## Run locally

```bash
python3 web/tools/serve.py
```

## Deploy to Cloudflare Pages

```bash
cd web
npx wrangler pages project create photo-to-point-cloud --production-branch main --force
npx wrangler pages deploy --branch main --force
```

`--force` uses classic Pages directly: recent wrangler versions otherwise hand off to Workers.
Add `?cpu` to the URL to test the no-WebGPU fallback.
