# Deploying PlateletWatch to Render (ONNX Runtime version)

## What changed from the PyTorch version
`inference_server.py` now uses `onnxruntime` with your `.onnx` model
exports instead of `ultralytics`/PyTorch. Same API, same response shape —
`server.js` and the frontend need zero changes. Benefits:
- Way lighter dependencies (no PyTorch, no Ultralytics)
- Much lower RAM use — should run on Render's Starter plan (512MB)
  instead of needing the $25/mo Standard plan
- Faster cold starts

Verified locally against your real `plateletwatch.onnx` and
`bloodcellwatch.onnx` files: models load, health check passes, and a
test image runs cleanly through preprocessing → inference → NMS →
response with the exact same JSON shape the old server produced.

Note: the ONNX exports were built at 640x640 (checked via the ONNX
model metadata), not 1280 like the old PyTorch server's default — this
is already set correctly in inference_server.py and render.yaml, just
don't change MODEL_IMGSZ without re-exporting the models.

## 1. Push this repo to GitHub
The model files are ~44MB and ~12MB — well under GitHub's 100MB limit,
so Git LFS is optional here (unlike the PyTorch .pt version). Plain git
works fine:

```bash
git init
git add .
git commit -m "Initial deploy setup (ONNX)"
git remote add origin <your-repo-url>
git push -u origin main
```

If you'd rather use LFS anyway (e.g. you expect to swap in bigger
models later), a `.gitattributes` is already included.

## 2. Deploy via Render Blueprint
1. Go to https://dashboard.render.com → New → Blueprint
2. Connect the GitHub repo you just pushed
3. Render detects `render.yaml` and proposes two services:
   - `plateletwatch-web` (Node.js)
   - `plateletwatch-inference` (Docker, ONNX Runtime)
4. Before clicking deploy, set the secret env var:
   - `OPENROUTER_API_KEY` on `plateletwatch-web`
5. Click Apply — Render builds and deploys both services.

## 3. Plan sizing
`plateletwatch-inference` is set to the Starter plan (512MB) in
render.yaml. This should be enough for onnxruntime with these two
models — if you see the service restart/crash under load, bump it to
Standard (2GB) in the Render dashboard.

## 4. After first deploy
- Render gives you URLs like:
  - https://plateletwatch-web.onrender.com
  - https://plateletwatch-inference.onrender.com
- Already wired together via INFERENCE_URL / ALLOWED_ORIGINS env vars
  in render.yaml — no manual edits needed unless you rename services.
- Visit the web URL — signin/register/dashboard routes work as
  configured in server.js.

## 5. Custom domain (optional)
Render → service → Settings → Custom Domain → add your domain, point a
CNAME at the URL Render gives you. Free HTTPS is automatic.
