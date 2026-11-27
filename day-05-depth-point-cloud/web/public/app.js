// Photo → Point Cloud, in the browser.
// Depth Anything V2 Metric-Outdoor-Small (fp16 ONNX) + pinhole back-projection + Three.js.
import * as ort from "onnxruntime-web";
import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";

const ORT_DIST = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.22.0/dist/";
const MODEL_CACHE = "depth-anything-v2-metric-outdoor-small-fp16-v1";
const MEAN = [0.485, 0.456, 0.406], STD = [0.229, 0.224, 0.225];
const PATCH = 14, SIZE = 518;

const $ = (id) => document.getElementById(id);
const ui = {
  status: $("status"), bar: $("bar"), drop: $("drop"), file: $("file"),
  thumbs: $("thumbs"), photo: $("photo"), depthmap: $("depthmap"),
  step: $("step"), maxd: $("maxd"), fov: $("fov"),
  stepV: $("stepV"), maxdV: $("maxdV"), fovV: $("fovV"),
  ply: $("ply"), views: $("views"), stats: $("stats"), empty: $("empty"), viewer: $("viewer"),
};

let session = null, backend = "";
let current = null;        // { rgb: Uint8ClampedArray, depth: Float32Array, W, H }
let cloud = null;          // { pos: Float32Array (x, forward, up), col: Uint8Array, n }

const setStatus = (msg, frac) => {
  ui.status.textContent = msg;
  if (frac !== undefined) ui.bar.style.width = `${Math.round(frac * 100)}%`;
};

// ---------------------------------------------------------------- model
async function fetchModel() {
  const manifest = await (await fetch("model/manifest.json")).json();
  let cache = null;
  try { cache = await caches.open(MODEL_CACHE); } catch { /* private mode etc. */ }
  const out = new Uint8Array(manifest.total);
  let offset = 0;
  for (const name of manifest.parts) {
    const url = new URL(`model/${name}`, location.href).href;
    let res = cache && (await cache.match(url));
    if (!res) {
      res = await fetch(url);
      if (!res.ok) throw new Error(`Could not load ${name} (${res.status})`);
      if (cache) { try { await cache.put(url, res.clone()); } catch { /* quota */ } }
    }
    const reader = res.body.getReader();
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      out.set(value, offset);
      offset += value.length;
      setStatus(`Downloading model… ${(offset / 1e6).toFixed(0)} / ${(manifest.total / 1e6).toFixed(0)} MB (once, then cached)`, offset / manifest.total * 0.9);
    }
  }
  return out;
}

async function loadModel() {
  ort.env.wasm.wasmPaths = ORT_DIST;
  ort.env.wasm.numThreads = self.crossOriginIsolated ? Math.min(4, navigator.hardwareConcurrency || 1) : 1;
  const bytes = await fetchModel();
  setStatus("Starting the model…", 0.95);
  const forceCpu = new URLSearchParams(location.search).has("cpu");   // ?cpu to test the fallback
  const tries = navigator.gpu && !forceCpu ? ["webgpu", "wasm"] : ["wasm"];
  for (const ep of tries) {
    try {
      session = await ort.InferenceSession.create(bytes, { executionProviders: [ep], graphOptimizationLevel: "all" });
      backend = ep === "webgpu" ? "WebGPU" : "CPU (WebAssembly)";
      break;
    } catch (e) { console.warn(`${ep} failed`, e); }
  }
  if (!session) throw new Error("This browser could not run the model.");
  setStatus(`Ready · running on ${backend}. Drop a photo.`, 1);
}

// ------------------------------------------------------------ preprocess
// Same as the Hugging Face DPT processor: keep aspect ratio, scale so the
// image is about 518 px, round both sides to a multiple of the 14 px patch.
function modelSize(w, h) {
  let sw = SIZE / w, sh = SIZE / h;
  if (Math.abs(1 - sw) < Math.abs(1 - sh)) sh = sw; else sw = sh;
  const snap = (v) => Math.max(PATCH, Math.round(v / PATCH) * PATCH);
  return [snap(w * sw), snap(h * sh)];
}

function toTensor(rgba, W, H) {
  const n = W * H, data = new Float32Array(3 * n);
  for (let i = 0; i < n; i++) {
    for (let c = 0; c < 3; c++) data[c * n + i] = (rgba[i * 4 + c] / 255 - MEAN[c]) / STD[c];
  }
  return new ort.Tensor("float32", data, [1, 3, H, W]);
}

// ------------------------------------------------------------ run photo
async function handleFile(file) {
  if (!file || !file.type.startsWith("image/")) return;
  if (!session) { setStatus("Model still loading, one moment…"); return; }
  const bmp = await createImageBitmap(file, { imageOrientation: "from-image" });
  const [W, H] = modelSize(bmp.width, bmp.height);

  const cv = ui.photo; cv.width = W; cv.height = H;
  const ctx = cv.getContext("2d", { willReadFrequently: true });
  ctx.imageSmoothingQuality = "high";
  ctx.drawImage(bmp, 0, 0, W, H);
  const rgba = ctx.getImageData(0, 0, W, H).data;
  ui.thumbs.hidden = false;

  setStatus(`Estimating depth (${W}×${H}) on ${backend}…`, 0.5);
  await new Promise(requestAnimationFrame);
  const t0 = performance.now();
  const out = await session.run({ pixel_values: toTensor(rgba, W, H) });
  const depth = out.predicted_depth.data;
  const ms = performance.now() - t0;

  current = { rgb: rgba, depth, W, H };
  drawDepthMap();
  rebuild();
  frame("camera");
  setStatus(`Done in ${(ms / 1000).toFixed(1)} s on ${backend}.`, 1);
  ui.ply.disabled = false;
  ui.views.hidden = false;
  ui.empty.hidden = true;
}

// Magma colour map (reversed, so near = bright), matching the notebook.
const MAGMA = ["#000004", "#1c1044", "#4f127b", "#812581", "#b5367a", "#e55064", "#fb8761", "#fec287", "#fcfdbf"]
  .map((h) => [1, 3, 5].map((i) => parseInt(h.slice(i, i + 2), 16)));
function magmaR(t) {
  const x = (1 - Math.min(1, Math.max(0, t))) * (MAGMA.length - 1);
  const i = Math.min(MAGMA.length - 2, Math.floor(x)), f = x - i;
  return MAGMA[i].map((a, k) => a + (MAGMA[i + 1][k] - a) * f);
}

function drawDepthMap() {
  const { depth, W, H } = current, maxd = +ui.maxd.value;
  const cv = ui.depthmap; cv.width = W; cv.height = H;
  const ctx = cv.getContext("2d"), img = ctx.createImageData(W, H);
  for (let i = 0; i < W * H; i++) {
    const [r, g, b] = magmaR(depth[i] / maxd);
    img.data[i * 4] = r; img.data[i * 4 + 1] = g; img.data[i * 4 + 2] = b; img.data[i * 4 + 3] = 255;
  }
  ctx.putImageData(img, 0, 0);
}

// ------------------------------------------------- back-projection (pinhole)
function toPoints(step, maxDepth, fovDeg) {
  const { rgb, depth, W, H } = current;
  const f = (W / 2) / Math.tan((fovDeg * Math.PI / 180) / 2);
  const cap = Math.ceil(W / step) * Math.ceil(H / step);
  const pos = new Float32Array(cap * 3), col = new Uint8Array(cap * 3), size = new Float32Array(cap);
  let n = 0;
  for (let v = 0; v < H; v += step) {
    for (let u = 0; u < W; u += step) {
      const i = v * W + u, z = depth[i];
      if (!(z > 0 && z < maxDepth)) continue;
      pos[n * 3] = (u - W / 2) * z / f;        // right
      pos[n * 3 + 1] = z;                      // forward
      pos[n * 3 + 2] = -(v - H / 2) * z / f;   // up  (Z-up, like Rhino)
      col[n * 3] = rgb[i * 4]; col[n * 3 + 1] = rgb[i * 4 + 1]; col[n * 3 + 2] = rgb[i * 4 + 2];
      size[n] = z * step / f * 1.4;            // world-space size of one sample
      n++;
    }
  }
  return { pos: pos.subarray(0, n * 3), col: col.subarray(0, n * 3), size: size.subarray(0, n), n, f };
}

// ------------------------------------------------------------- three.js
const renderer = new THREE.WebGLRenderer({ canvas: $("three"), antialias: true });
renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
const scene = new THREE.Scene();
scene.background = new THREE.Color(0x0b0b0d);
const camera = new THREE.PerspectiveCamera(50, 1, 0.05, 2000);
camera.up.set(0, 0, 1);                       // Z-up scene, same as the PLY / Rhino
const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;

const material = new THREE.ShaderMaterial({
  uniforms: { scale: { value: 1 } },
  vertexShader: `
    attribute float psize; attribute vec3 color; varying vec3 vColor; uniform float scale;
    void main() {
      vColor = color;
      vec4 mv = modelViewMatrix * vec4(position, 1.0);
      gl_PointSize = max(1.0, psize * scale / -mv.z);
      gl_Position = projectionMatrix * mv;
    }`,
  fragmentShader: `
    varying vec3 vColor;
    void main() { gl_FragColor = vec4(vColor, 1.0); }`,
});
let points = null;
const grid = new THREE.GridHelper(200, 40, 0x333338, 0x1e1e22);
grid.rotation.x = Math.PI / 2;                // lay the grid on the XY ground plane
scene.add(grid);

function rebuild() {
  if (!current) return;
  const step = +ui.step.value, maxd = +ui.maxd.value, fov = +ui.fov.value;
  cloud = toPoints(step, maxd, fov);
  const g = new THREE.BufferGeometry();
  g.setAttribute("position", new THREE.BufferAttribute(cloud.pos, 3));
  g.setAttribute("color", new THREE.BufferAttribute(cloud.col, 3, true));
  g.setAttribute("psize", new THREE.BufferAttribute(cloud.size, 1));
  if (points) { points.geometry.dispose(); scene.remove(points); }
  points = new THREE.Points(g, material);
  scene.add(points);
  // drop the grid to the lowest points (roughly the street)
  let minZ = 0; for (let i = 2; i < cloud.pos.length; i += 3) minZ = Math.min(minZ, cloud.pos[i]);
  grid.position.z = minZ;
  ui.stats.textContent = `${cloud.n.toLocaleString()} points · metres · Z-up`;
}

function frame(view) {
  if (!cloud) return;
  const { W, H } = current, hfov = +ui.fov.value * Math.PI / 180;
  let far = 0; for (let i = 1; i < cloud.pos.length; i += 3) far += cloud.pos[i];
  const mid = far / Math.max(1, cloud.n);    // mean forward distance
  if (view === "camera") {
    // fit the whole photo: its vertical FOV, or wider if the viewer is narrower than the photo
    const aspect = ui.viewer.clientWidth / ui.viewer.clientHeight;
    const t = Math.tan(hfov / 2) * Math.max(H / W, 1 / aspect);
    camera.fov = 2 * Math.atan(t) * 180 / Math.PI;
    camera.position.set(0, 0.001, 0);
    controls.target.set(0, mid, 0);
  } else if (view === "orbit") {
    camera.fov = 50;
    camera.position.set(mid * 0.9, -mid * 0.35, mid * 0.6);
    controls.target.set(0, mid * 0.8, 0);
  } else {
    camera.fov = 50;
    camera.position.set(0, mid * 0.75, mid * 2.2);
    controls.target.set(0, mid * 0.75, 0);
  }
  camera.updateProjectionMatrix();
  controls.update();
  resize();
}

function resize() {
  const w = ui.viewer.clientWidth, h = ui.viewer.clientHeight;
  renderer.setSize(w, h, false);
  camera.aspect = w / h;
  camera.updateProjectionMatrix();
  material.uniforms.scale.value = (h * renderer.getPixelRatio()) / (2 * Math.tan(camera.fov * Math.PI / 360));
}
new ResizeObserver(resize).observe(ui.viewer);
renderer.setAnimationLoop(() => { controls.update(); renderer.render(scene, camera); });

// --------------------------------------------------------------- export
function buildPLY() {
  const { pos, col, n } = cloud;
  const header = `ply\nformat binary_little_endian 1.0\ncomment Day 05 - Depth Anything V2 Metric-Outdoor, metres, Z-up\nelement vertex ${n}\nproperty float x\nproperty float y\nproperty float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n`;
  const head = new TextEncoder().encode(header);
  const buf = new ArrayBuffer(head.length + n * 15), dv = new DataView(buf);
  new Uint8Array(buf).set(head);
  let o = head.length;
  for (let i = 0; i < n; i++, o += 15) {
    dv.setFloat32(o, pos[i * 3], true); dv.setFloat32(o + 4, pos[i * 3 + 1], true); dv.setFloat32(o + 8, pos[i * 3 + 2], true);
    dv.setUint8(o + 12, col[i * 3]); dv.setUint8(o + 13, col[i * 3 + 1]); dv.setUint8(o + 14, col[i * 3 + 2]);
  }
  return buf;
}

function downloadPLY() {
  const buf = buildPLY();
  const a = document.createElement("a");
  a.href = URL.createObjectURL(new Blob([buf], { type: "application/octet-stream" }));
  a.download = "point_cloud.ply";
  a.click();
  setTimeout(() => URL.revokeObjectURL(a.href), 1000);
}

// ------------------------------------------------------------------- ui
const label = () => {
  ui.stepV.textContent = `${ui.step.value} px`;
  ui.maxdV.textContent = `${ui.maxd.value} m`;
  ui.fovV.textContent = `${ui.fov.value}°`;
};
ui.step.oninput = () => { label(); rebuild(); };
ui.fov.oninput = () => { label(); rebuild(); };
ui.maxd.oninput = () => { label(); if (current) { drawDepthMap(); rebuild(); } };
ui.ply.onclick = downloadPLY;
ui.views.onclick = (e) => { const v = e.target.dataset.view; if (v) frame(v); };
ui.drop.onclick = () => ui.file.click();
ui.drop.onkeydown = (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); ui.file.click(); } };
ui.file.onchange = () => handleFile(ui.file.files[0]);
for (const t of [ui.drop, document.body]) {
  t.addEventListener("dragover", (e) => { e.preventDefault(); ui.drop.classList.add("over"); });
  t.addEventListener("dragleave", () => ui.drop.classList.remove("over"));
  t.addEventListener("drop", (e) => { e.preventDefault(); ui.drop.classList.remove("over"); handleFile(e.dataTransfer.files[0]); });
}
label();
resize();
loadModel().catch((e) => { console.error(e); setStatus(`Error: ${e.message}`); });

// for testing from the console
window.__depthApp = { handleFile, buildPLY, get cloud() { return cloud; }, get current() { return current; } };
