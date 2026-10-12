import { fromArrayBuffer } from "https://cdn.jsdelivr.net/npm/geotiff@2.1.3/+esm";

const MODEL_PATH = "./model.onnx";
const ORT_ASSET_PATH = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.23.2/dist/";
const TILE_SIZE = 256;
const TILE_OVERLAP = 32;
const REFLECTANCE_SCALE = 5000;

const state = {
  pre: null,
  current: null,
  session: null,
  engine: null,
  probability: null,
  elapsedDays: null,
};

const elements = {
  preFile: document.querySelector("#pre-file"),
  currentFile: document.querySelector("#current-file"),
  preCard: document.querySelector("#pre-card"),
  currentCard: document.querySelector("#current-card"),
  preName: document.querySelector("#pre-name"),
  currentName: document.querySelector("#current-name"),
  preMeta: document.querySelector("#pre-meta"),
  currentMeta: document.querySelector("#current-meta"),
  ignitionDate: document.querySelector("#ignition-date"),
  currentDate: document.querySelector("#current-date"),
  runButton: document.querySelector("#run-button"),
  engineStatus: document.querySelector("#engine-status"),
  message: document.querySelector("#message"),
  progressWrap: document.querySelector("#progress-wrap"),
  progressText: document.querySelector("#progress-text"),
  progressValue: document.querySelector("#progress-value"),
  progressBar: document.querySelector("#progress-bar"),
  results: document.querySelector("#results"),
  threshold: document.querySelector("#threshold"),
  thresholdValue: document.querySelector("#threshold-value"),
  elapsedSummary: document.querySelector("#elapsed-summary"),
  mappedSummary: document.querySelector("#mapped-summary"),
  burnedSummary: document.querySelector("#burned-summary"),
  engineSummary: document.querySelector("#engine-summary"),
  currentCanvas: document.querySelector("#current-canvas"),
  probabilityCanvas: document.querySelector("#probability-canvas"),
  maskCanvas: document.querySelector("#mask-canvas"),
  overlayCanvas: document.querySelector("#overlay-canvas"),
  downloadMask: document.querySelector("#download-mask"),
  downloadProbability: document.querySelector("#download-probability"),
};

function setMessage(message, error = false) {
  elements.message.textContent = message;
  elements.message.style.color = error ? "#8c3218" : "#52605b";
}

function setEngineStatus(label, mode = "") {
  elements.engineStatus.className = `engine-status ${mode}`.trim();
  elements.engineStatus.querySelector("b").textContent = label;
}

function setProgress(fraction, message) {
  const value = Math.max(0, Math.min(100, Math.round(fraction * 100)));
  elements.progressWrap.hidden = false;
  elements.progressBar.style.width = `${value}%`;
  elements.progressValue.textContent = `${value}%`;
  elements.progressText.textContent = message;
}

function updateRunState() {
  elements.runButton.disabled = !(
    state.pre && state.current && elements.ignitionDate.value && elements.currentDate.value
  );
}

function approximatelyEqualBox(first, second) {
  if (!first || !second || first.length !== second.length) return true;
  const scale = Math.max(...first.map(Math.abs), ...second.map(Math.abs), 1);
  return first.every((value, index) => Math.abs(value - second[index]) <= scale * 1e-6);
}

async function readGeoTiff(file) {
  const tiff = await fromArrayBuffer(await file.arrayBuffer());
  const image = await tiff.getImage();
  const width = image.getWidth();
  const height = image.getHeight();
  const sampleCount = image.getSamplesPerPixel();
  if (sampleCount < 3) throw new Error("The GeoTIFF needs at least three bands: SWIR2, NIR, and Red.");
  const samples = sampleCount >= 4 ? [0, 1, 2, 3] : [0, 1, 2];
  const rasters = await image.readRasters({ samples, interleave: false });
  const bands = [rasters[0], rasters[1], rasters[2]];
  const clear = new Uint8Array(width * height);
  if (sampleCount >= 4) {
    for (let index = 0; index < clear.length; index += 1) clear[index] = rasters[3][index] > 0.5 ? 1 : 0;
  } else {
    for (let index = 0; index < clear.length; index += 1) {
      clear[index] = bands[0][index] > 0 && bands[1][index] > 0 && bands[2][index] > 0 ? 1 : 0;
    }
  }
  let box = null;
  try { box = image.getBoundingBox(); } catch (_) { /* Ungeoreferenced TIFF. */ }
  return { fileName: file.name, width, height, sampleCount, bands, clear, box };
}

async function handleFile(role, file) {
  if (!file) return;
  const card = role === "pre" ? elements.preCard : elements.currentCard;
  const name = role === "pre" ? elements.preName : elements.currentName;
  const meta = role === "pre" ? elements.preMeta : elements.currentMeta;
  name.textContent = file.name;
  meta.textContent = "Reading GeoTIFF…";
  card.classList.remove("loaded");
  setMessage("");
  try {
    const raster = await readGeoTiff(file);
    state[role] = raster;
    card.classList.add("loaded");
    meta.textContent = `${raster.width.toLocaleString()} × ${raster.height.toLocaleString()} · ${raster.sampleCount} bands${raster.sampleCount < 4 ? " · clear mask derived" : ""}`;
  } catch (error) {
    state[role] = null;
    meta.textContent = "Could not read this file";
    setMessage(error.message, true);
  }
  updateRunState();
}

function validateInputs() {
  const { pre, current } = state;
  if (pre.width !== current.width || pre.height !== current.height) {
    throw new Error(`The image dimensions do not match (${pre.width} × ${pre.height} versus ${current.width} × ${current.height}).`);
  }
  if (!approximatelyEqualBox(pre.box, current.box)) {
    throw new Error("The GeoTIFF geographic bounds do not match. Reproject and align the pair before inference.");
  }
  const ignition = new Date(`${elements.ignitionDate.value}T00:00:00Z`);
  const observationDate = new Date(`${elements.currentDate.value}T00:00:00Z`);
  const elapsedDays = Math.round((observationDate - ignition) / 86400000);
  if (!Number.isFinite(elapsedDays)) throw new Error("Please provide valid dates.");
  if (elapsedDays < 0) throw new Error("The current image date cannot be before the ignition date.");
  return elapsedDays;
}

async function createSession() {
  if (state.session) return state.session;
  if (!window.ort) throw new Error("ONNX Runtime did not load. Check your internet connection and reload the page.");
  window.ort.env.wasm.wasmPaths = ORT_ASSET_PATH;
  window.ort.env.wasm.numThreads = 1;
  setEngineStatus("Loading model", "loading");
  try {
    if (navigator.gpu) {
      state.session = await window.ort.InferenceSession.create(MODEL_PATH, { executionProviders: ["webgpu"] });
      state.engine = "WebGPU";
    } else {
      throw new Error("WebGPU unavailable");
    }
  } catch (webGpuError) {
    state.session = await window.ort.InferenceSession.create(MODEL_PATH, { executionProviders: ["wasm"] });
    state.engine = "WebAssembly";
  }
  setEngineStatus(`${state.engine} ready`, "ready");
  return state.session;
}

function tileStarts(length) {
  if (length <= TILE_SIZE) return [0];
  const starts = [];
  const stride = TILE_SIZE - TILE_OVERLAP;
  for (let position = 0; position < length - TILE_SIZE; position += stride) starts.push(position);
  if (starts.at(-1) !== length - TILE_SIZE) starts.push(length - TILE_SIZE);
  return starts;
}

function extractTile(raster, startX, startY) {
  const pixels = TILE_SIZE * TILE_SIZE;
  const image = new Float32Array(3 * pixels);
  const clear = new Float32Array(pixels);
  const copyWidth = Math.min(TILE_SIZE, raster.width - startX);
  const copyHeight = Math.min(TILE_SIZE, raster.height - startY);
  for (let y = 0; y < copyHeight; y += 1) {
    const sourceRow = (startY + y) * raster.width + startX;
    const targetRow = y * TILE_SIZE;
    for (let x = 0; x < copyWidth; x += 1) {
      const source = sourceRow + x;
      const target = targetRow + x;
      clear[target] = raster.clear[source];
      for (let band = 0; band < 3; band += 1) {
        const value = Number(raster.bands[band][source]);
        image[band * pixels + target] = Math.min(REFLECTANCE_SCALE, Math.max(0, Number.isFinite(value) ? value : 0)) / REFLECTANCE_SCALE;
      }
    }
  }
  return { image, clear, copyWidth, copyHeight };
}

async function runInference() {
  elements.runButton.disabled = true;
  elements.results.hidden = true;
  setMessage("");
  setProgress(0.01, "Validating observation pair…");
  try {
    const elapsedDays = validateInputs();
    state.elapsedDays = elapsedDays;
    const session = await createSession();
    const xs = tileStarts(state.current.width);
    const ys = tileStarts(state.current.height);
    const total = xs.length * ys.length;
    const accumulated = new Float32Array(state.current.width * state.current.height);
    const counts = new Uint16Array(state.current.width * state.current.height);
    let completed = 0;

    for (const startY of ys) {
      for (const startX of xs) {
        const pre = extractTile(state.pre, startX, startY);
        const current = extractTile(state.current, startX, startY);
        const feeds = {
          pre: new window.ort.Tensor("float32", pre.image, [1, 3, TILE_SIZE, TILE_SIZE]),
          current: new window.ort.Tensor("float32", current.image, [1, 3, TILE_SIZE, TILE_SIZE]),
          pre_clear: new window.ort.Tensor("float32", pre.clear, [1, 1, TILE_SIZE, TILE_SIZE]),
          current_clear: new window.ort.Tensor("float32", current.clear, [1, 1, TILE_SIZE, TILE_SIZE]),
          elapsed_days: new window.ort.Tensor("float32", new Float32Array([elapsedDays]), [1]),
        };
        const output = await session.run(feeds);
        const probability = output.probability.data;
        for (let y = 0; y < current.copyHeight; y += 1) {
          const sourceRow = y * TILE_SIZE;
          const targetRow = (startY + y) * state.current.width + startX;
          for (let x = 0; x < current.copyWidth; x += 1) {
            const target = targetRow + x;
            accumulated[target] += probability[sourceRow + x];
            counts[target] += 1;
          }
        }
        completed += 1;
        setProgress(completed / total, `Evaluating tile ${completed.toLocaleString()} of ${total.toLocaleString()}…`);
        await new Promise(requestAnimationFrame);
      }
    }
    for (let index = 0; index < accumulated.length; index += 1) {
      if (counts[index]) accumulated[index] /= counts[index];
    }
    state.probability = accumulated;
    renderResults();
    setProgress(1, "Inference complete");
    elements.results.hidden = false;
    elements.results.scrollIntoView({ behavior: "smooth", block: "start" });
  } catch (error) {
    console.error(error);
    elements.progressWrap.hidden = true;
    setMessage(error.message || "Inference failed.", true);
  } finally {
    updateRunState();
  }
}

function previewGeometry(width, height) {
  const maxDimension = 900;
  const scale = Math.min(1, maxDimension / Math.max(width, height));
  return { width: Math.max(1, Math.round(width * scale)), height: Math.max(1, Math.round(height * scale)) };
}

function probabilityColor(value) {
  const stops = [
    [0, 3, 8, 7], [0.35, 57, 38, 29], [0.6, 223, 75, 32], [0.8, 255, 173, 66], [1, 255, 252, 229],
  ];
  let upper = 1;
  while (upper < stops.length && value > stops[upper][0]) upper += 1;
  const high = stops[Math.min(upper, stops.length - 1)];
  const low = stops[Math.max(0, upper - 1)];
  const amount = high[0] === low[0] ? 0 : (value - low[0]) / (high[0] - low[0]);
  return [1, 2, 3].map(index => Math.round(low[index] + (high[index] - low[index]) * amount));
}

function renderToCanvas(canvas, pixelFunction, fullResolution = false) {
  const sourceWidth = state.current.width;
  const sourceHeight = state.current.height;
  const target = fullResolution ? { width: sourceWidth, height: sourceHeight } : previewGeometry(sourceWidth, sourceHeight);
  canvas.width = target.width;
  canvas.height = target.height;
  const context = canvas.getContext("2d");
  const output = context.createImageData(target.width, target.height);
  for (let y = 0; y < target.height; y += 1) {
    const sourceY = Math.min(sourceHeight - 1, Math.floor(y * sourceHeight / target.height));
    for (let x = 0; x < target.width; x += 1) {
      const sourceX = Math.min(sourceWidth - 1, Math.floor(x * sourceWidth / target.width));
      const sourceIndex = sourceY * sourceWidth + sourceX;
      const targetIndex = (y * target.width + x) * 4;
      const color = pixelFunction(sourceIndex);
      output.data[targetIndex] = color[0];
      output.data[targetIndex + 1] = color[1];
      output.data[targetIndex + 2] = color[2];
      output.data[targetIndex + 3] = color.length > 3 ? color[3] : 255;
    }
  }
  context.putImageData(output, 0, 0);
}

function falseColor(index) {
  if (!state.current.clear[index]) return [104, 110, 108, 255];
  return state.current.bands.map(band => {
    const normalized = Math.min(1, Math.max(0, Number(band[index]) / REFLECTANCE_SCALE));
    return Math.round(255 * Math.pow(normalized, 0.72));
  });
}

function maskColor(index, threshold) {
  if (!state.current.clear[index]) return [155, 155, 155, 255];
  return state.probability[index] >= threshold ? [255, 255, 255, 255] : [0, 0, 0, 255];
}

function renderResults() {
  const threshold = Number(elements.threshold.value);
  elements.thresholdValue.value = threshold.toFixed(2);
  renderToCanvas(elements.currentCanvas, falseColor);
  renderToCanvas(elements.probabilityCanvas, index => state.current.clear[index] ? [...probabilityColor(state.probability[index]), 255] : [105, 105, 105, 255]);
  renderToCanvas(elements.maskCanvas, index => maskColor(index, threshold));
  renderToCanvas(elements.overlayCanvas, index => {
    const base = falseColor(index);
    if (state.current.clear[index] && state.probability[index] >= threshold) {
      return [Math.round(base[0] * 0.38 + 255 * 0.62), Math.round(base[1] * 0.38 + 78 * 0.62), Math.round(base[2] * 0.38 + 37 * 0.62), 255];
    }
    return base;
  });
  let valid = 0;
  let burned = 0;
  for (let index = 0; index < state.probability.length; index += 1) {
    if (!state.current.clear[index]) continue;
    valid += 1;
    if (state.probability[index] >= threshold) burned += 1;
  }
  elements.elapsedSummary.textContent = `${state.elapsedDays} day${state.elapsedDays === 1 ? "" : "s"}`;
  elements.mappedSummary.textContent = valid.toLocaleString();
  elements.burnedSummary.textContent = valid ? `${burned.toLocaleString()} (${(100 * burned / valid).toFixed(1)}%)` : "No clear pixels";
  elements.engineSummary.textContent = state.engine;
}

function downloadCanvas(kind) {
  if (!state.probability) return;
  const threshold = Number(elements.threshold.value);
  const canvas = document.createElement("canvas");
  if (kind === "mask") {
    renderToCanvas(canvas, index => maskColor(index, threshold), true);
  } else {
    renderToCanvas(canvas, index => state.current.clear[index] ? [...probabilityColor(state.probability[index]), 255] : [105, 105, 105, 255], true);
  }
  canvas.toBlob(blob => {
    if (!blob) return;
    const link = document.createElement("a");
    link.href = URL.createObjectURL(blob);
    link.download = `wildfire_${kind}_${elements.currentDate.value}_t${threshold.toFixed(2)}.png`;
    link.click();
    setTimeout(() => URL.revokeObjectURL(link.href), 1000);
  }, "image/png");
}

elements.preFile.addEventListener("change", event => handleFile("pre", event.target.files[0]));
elements.currentFile.addEventListener("change", event => handleFile("current", event.target.files[0]));
elements.ignitionDate.addEventListener("change", updateRunState);
elements.currentDate.addEventListener("change", updateRunState);
elements.runButton.addEventListener("click", runInference);
elements.threshold.addEventListener("input", () => state.probability && renderResults());
elements.downloadMask.addEventListener("click", () => downloadCanvas("mask"));
elements.downloadProbability.addEventListener("click", () => downloadCanvas("probability"));

setEngineStatus("Model loads on demand");
