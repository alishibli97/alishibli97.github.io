import { fromArrayBuffer } from "https://cdn.jsdelivr.net/npm/geotiff@2.1.3/+esm";

const MODEL_PATH = "../wildfire-demo/model.onnx";
const ORT_ASSET_PATH = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.23.2/dist/";
const EE_SCOPE = "https://www.googleapis.com/auth/earthengine.readonly";
const S2_COLLECTION = "COPERNICUS/S2_HARMONIZED";
const CLOUD_COLLECTION = "GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED";
const CLOUD_SCORE_THRESHOLD = 0.60;
const REFLECTANCE_SCALE = 5000;
const TILE_SIZE = 256;
const TILE_OVERLAP = 32;
const DATASET_BANDS = ["B12", "B8", "B4"];

const SCENE_PRESETS = Object.freeze({
  US_2017_AZ3442311084220170601: {
    latitude: 34.400141, longitude: -110.841913, roiKm: 7.84,
    ignition: "2017-06-01", target: "2017-06-29", areaHa: 8013,
  },
  US_2018_CA4114012366020180816: {
    latitude: 41.140505, longitude: -123.652970, roiKm: 6.56,
    ignition: "2018-08-16", target: "2018-09-25", areaHa: 9869,
  },
  US_2020_OR4403912052820200816: {
    latitude: 44.028087, longitude: -120.500682, roiKm: 6.56,
    ignition: "2020-08-16", target: "2020-09-29", areaHa: 9751,
  },
  US_2022_MO3793309102520220328: {
    latitude: 37.956501, longitude: -91.006369, roiKm: 6.72,
    ignition: "2022-03-28", target: "2022-04-23", areaHa: 9830,
  },
  US_2023_OR4242612068520230918: {
    latitude: 42.413549, longitude: -120.661961, roiKm: 5.60,
    ignition: "2023-09-18", target: "2023-10-06", areaHa: 5812,
  },
});

const state = {
  eeConnected: false,
  authPopupReady: false,
  authAttempt: 0,
  modelSession: null,
  inferenceEngine: null,
  pre: null,
  current: null,
  probability: null,
  elapsedDays: null,
  selected: null,
  sourceBlob: null,
  map: null,
  marker: null,
  rectangle: null,
};

const elements = {
  projectId: document.querySelector("#project-id"),
  clientId: document.querySelector("#client-id"),
  connectButton: document.querySelector("#connect-button"),
  eeStatus: document.querySelector("#ee-status"),
  connectionMessage: document.querySelector("#connection-message"),
  scenePreset: document.querySelector("#scene-preset"),
  presetMeta: document.querySelector("#preset-meta"),
  latitude: document.querySelector("#latitude"),
  longitude: document.querySelector("#longitude"),
  roiSize: document.querySelector("#roi-size"),
  roiValue: document.querySelector("#roi-value"),
  ignitionDate: document.querySelector("#ignition-date"),
  targetDate: document.querySelector("#target-date"),
  preWindow: document.querySelector("#pre-window"),
  targetWindow: document.querySelector("#target-window"),
  mapWorkspace: document.querySelector("#map-workspace"),
  mapExpand: document.querySelector("#map-expand"),
  mapCoordinate: document.querySelector("#map-coordinate"),
  runButton: document.querySelector("#run-button"),
  progressWrap: document.querySelector("#progress-wrap"),
  progressText: document.querySelector("#progress-text"),
  progressValue: document.querySelector("#progress-value"),
  progressBar: document.querySelector("#progress-bar"),
  message: document.querySelector("#message"),
  results: document.querySelector("#results"),
  threshold: document.querySelector("#threshold"),
  thresholdValue: document.querySelector("#threshold-value"),
  preDateSummary: document.querySelector("#pre-date-summary"),
  preClearSummary: document.querySelector("#pre-clear-summary"),
  currentDateSummary: document.querySelector("#current-date-summary"),
  currentClearSummary: document.querySelector("#current-clear-summary"),
  elapsedSummary: document.querySelector("#elapsed-summary"),
  engineSummary: document.querySelector("#engine-summary"),
  burnedSummary: document.querySelector("#burned-summary"),
  pixelSummary: document.querySelector("#pixel-summary"),
  preCanvas: document.querySelector("#pre-canvas"),
  currentCanvas: document.querySelector("#current-canvas"),
  probabilityCanvas: document.querySelector("#probability-canvas"),
  maskCanvas: document.querySelector("#mask-canvas"),
  downloadMask: document.querySelector("#download-mask"),
  downloadStack: document.querySelector("#download-stack"),
};

function setMessage(message, error = false, connection = false) {
  const target = connection ? elements.connectionMessage : elements.message;
  target.textContent = message;
  target.style.color = error ? "#8c3218" : "#52605b";
}

function setEeStatus(label, mode = "") {
  elements.eeStatus.className = `engine-status ${mode}`.trim();
  elements.eeStatus.querySelector("b").textContent = label;
}

function setProgress(fraction, message) {
  const value = Math.max(0, Math.min(100, Math.round(fraction * 100)));
  elements.progressWrap.hidden = false;
  elements.progressBar.style.width = `${value}%`;
  elements.progressValue.textContent = `${value}%`;
  elements.progressText.textContent = message;
}

function normalizeError(error) {
  if (typeof error === "string") return error;
  const messages = [
    error?.message,
    error?.details,
    error?.error_description,
    typeof error?.error === "string" ? error.error : error?.error?.message,
    error?.result?.error?.message,
  ].filter(Boolean);
  if (messages.length) return [...new Set(messages)].join(" — ");
  try {
    return JSON.stringify(error);
  } catch {
    return "Unknown error";
  }
}

function initializeMap() {
  const latitude = Number(elements.latitude.value);
  const longitude = Number(elements.longitude.value);
  state.map = window.L.map("map", { zoomControl: true }).setView([latitude, longitude], 10);
  window.L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
    maxZoom: 18,
    attribution: "&copy; OpenStreetMap contributors",
  }).addTo(state.map);
  const icon = window.L.divIcon({ className: "roi-center-marker", iconSize: [15, 15] });
  state.marker = window.L.marker([latitude, longitude], { icon, interactive: false }).addTo(state.map);
  state.rectangle = window.L.rectangle(roiBounds(latitude, longitude), {
    color: "#ff6b35", weight: 2, fillColor: "#ff6b35", fillOpacity: 0.08,
  }).addTo(state.map);
  state.map.on("click", event => {
    elements.latitude.value = event.latlng.lat.toFixed(5);
    elements.longitude.value = event.latlng.lng.toFixed(5);
    markSceneCustom("Custom scene · map center selected manually; adjust its dates at right.");
    updateMapSelection(false);
  });
}

function roiBounds(latitude = Number(elements.latitude.value), longitude = Number(elements.longitude.value)) {
  const halfKm = Number(elements.roiSize.value) / 2;
  const latitudeDelta = halfKm / 111.32;
  const longitudeDelta = halfKm / (111.32 * Math.max(0.15, Math.cos(latitude * Math.PI / 180)));
  return [
    [latitude - latitudeDelta, longitude - longitudeDelta],
    [latitude + latitudeDelta, longitude + longitudeDelta],
  ];
}

function updateMapSelection(pan = true) {
  const latitude = Number(elements.latitude.value);
  const longitude = Number(elements.longitude.value);
  if (!Number.isFinite(latitude) || !Number.isFinite(longitude)) return;
  state.marker.setLatLng([latitude, longitude]);
  state.rectangle.setBounds(roiBounds(latitude, longitude));
  elements.roiValue.value = `${Number(elements.roiSize.value).toFixed(2)} km`;
  elements.mapCoordinate.textContent = `${latitude.toFixed(4)}, ${longitude.toFixed(4)}`;
  if (pan) state.map.panTo([latitude, longitude]);
}

function markSceneCustom(message = "Custom scene · coordinates, dates, or ROI adjusted manually.") {
  elements.scenePreset.value = "";
  elements.presetMeta.textContent = message;
}

function applyScenePreset(eventId) {
  const preset = SCENE_PRESETS[eventId];
  if (!preset) return;
  elements.latitude.value = preset.latitude;
  elements.longitude.value = preset.longitude;
  elements.roiSize.value = preset.roiKm;
  elements.ignitionDate.value = preset.ignition;
  elements.targetDate.value = preset.target;
  elements.presetMeta.textContent = `${eventId} · Trees majority · ${preset.areaHa.toLocaleString()} ha · target date drawn from its downloaded progression`;
  updateMapSelection(false);
  state.map.fitBounds(roiBounds(preset.latitude, preset.longitude), { padding: [45, 45], maxZoom: 12 });
}

function setMapExpanded(expanded) {
  elements.mapWorkspace.classList.toggle("is-map-expanded", expanded && !document.fullscreenElement);
  document.body.classList.toggle("map-expanded", expanded);
  elements.mapExpand.setAttribute("aria-pressed", String(expanded));
  elements.mapExpand.querySelector("span").textContent = expanded ? "Exit enlarged map" : "Enlarge map";
  setTimeout(() => state.map.invalidateSize(), 80);
}

async function toggleMapWorkspace() {
  const fallbackExpanded = elements.mapWorkspace.classList.contains("is-map-expanded");
  if (document.fullscreenElement === elements.mapWorkspace) {
    await document.exitFullscreen();
    return;
  }
  if (fallbackExpanded) {
    setMapExpanded(false);
    return;
  }
  if (elements.mapWorkspace.requestFullscreen) {
    try {
      await elements.mapWorkspace.requestFullscreen();
      return;
    } catch (_) { /* Fall back to an in-page enlarged view. */ }
  }
  setMapExpanded(true);
}

function initializeEarthEngine(projectId) {
  setEeStatus("Initializing", "loading");
  window.ee.initialize(
    null,
    null,
    () => {
      state.eeConnected = true;
      elements.runButton.disabled = false;
      elements.connectButton.disabled = false;
      elements.connectButton.querySelector("span").textContent = "Earth Engine connected";
      setEeStatus("Earth Engine ready", "ready");
      setMessage(`Connected through ${projectId}.`, false, true);
    },
    error => {
      state.eeConnected = false;
      elements.connectButton.disabled = false;
      setEeStatus("Connection failed");
      setMessage(`Earth Engine initialization failed: ${normalizeError(error)}`, true, true);
    },
    null,
    projectId,
  );
}

function connectEarthEngine() {
  const projectId = elements.projectId.value.trim();
  const clientId = elements.clientId.value.trim();
  if (!projectId) {
    setMessage("Enter the Earth Engine-enabled Google Cloud project ID.", true, true);
    return;
  }
  if (!clientId.endsWith(".apps.googleusercontent.com")) {
    setMessage("Enter a valid OAuth web client ID ending in .apps.googleusercontent.com.", true, true);
    return;
  }
  if (!window.ee) {
    setMessage("The Earth Engine JavaScript library did not load. Reload the page and try again.", true, true);
    return;
  }
  localStorage.setItem("wildfire-ee-project", projectId);
  localStorage.setItem("wildfire-ee-client-id", clientId);
  localStorage.setItem("flamenet-ee-project", projectId);
  localStorage.setItem("flamenet-ee-client-id", clientId);

  if (state.authPopupReady) {
    state.authPopupReady = false;
    elements.connectButton.disabled = true;
    setEeStatus("Sign-in opening", "loading");
    setMessage("Opening Google authorization… Allow the popup if your browser asks.", false, true);
    window.ee.data.authenticateViaPopup(
      () => initializeEarthEngine(projectId),
      error => {
        elements.connectButton.disabled = false;
        elements.connectButton.querySelector("span").textContent = "Continue with Google";
        state.authPopupReady = true;
        setEeStatus("Authorization failed");
        setMessage(`Google authorization failed: ${normalizeError(error)}`, true, true);
      },
    );
    return;
  }

  elements.connectButton.disabled = true;
  setEeStatus("Authorizing", "loading");
  setMessage("Checking for an existing Google authorization…", false, true);
  const attempt = ++state.authAttempt;
  let fallbackTimer;
  const offerPopup = (message = "Click “Continue with Google” to open the authorization window.") => {
    if (attempt !== state.authAttempt || state.eeConnected) return;
    state.authPopupReady = true;
    elements.connectButton.disabled = false;
    elements.connectButton.querySelector("span").textContent = "Continue with Google";
    setEeStatus("Sign-in required");
    setMessage(message, false, true);
  };
  const success = () => {
    clearTimeout(fallbackTimer);
    if (attempt !== state.authAttempt) return;
    state.authPopupReady = false;
    initializeEarthEngine(projectId);
  };
  const failure = error => {
    clearTimeout(fallbackTimer);
    if (attempt !== state.authAttempt) return;
    state.authPopupReady = false;
    elements.connectButton.disabled = false;
    setEeStatus("Authorization failed");
    setMessage(`Google authorization failed: ${normalizeError(error)}`, true, true);
  };
  fallbackTimer = setTimeout(() => {
    offerPopup("The silent sign-in check took too long. Click “Continue with Google” to authorize directly.");
  }, 8000);
  window.ee.data.authenticateViaOauth(
    clientId,
    success,
    failure,
    [EE_SCOPE],
    () => {
      clearTimeout(fallbackTimer);
      offerPopup();
    },
    true,
  );
}

function eeEvaluate(computed) {
  return new Promise((resolve, reject) => {
    computed.evaluate((value, error) => error ? reject(new Error(normalizeError(error))) : resolve(value));
  });
}

function webMercatorGrid(latitude, longitude, sizeKm) {
  const radius = 6378137;
  const x = radius * longitude * Math.PI / 180;
  const clampedLatitude = Math.max(-85.05112878, Math.min(85.05112878, latitude));
  const y = radius * Math.log(Math.tan(Math.PI / 4 + clampedLatitude * Math.PI / 360));
  const dimension = Math.max(128, Math.round(sizeKm * 1000 / 20));
  const span = dimension * 20;
  return {
    dimensions: { width: dimension, height: dimension },
    affineTransform: {
      scaleX: 20, shearX: 0, translateX: x - span / 2,
      shearY: 0, scaleY: -20, translateY: y + span / 2,
    },
    crsCode: "EPSG:3857",
  };
}

function freshAuthToken() {
  const existing = window.ee.data.getAuthToken();
  if (existing) return Promise.resolve(existing);
  return new Promise((resolve, reject) => {
    window.ee.data.refreshAuthToken(
      () => {
        const refreshed = window.ee.data.getAuthToken();
        if (refreshed) resolve(refreshed);
        else reject(new Error("Google authorization expired. Reconnect Earth Engine."));
      },
      error => reject(new Error(normalizeError(error))),
    );
  });
}

async function computePixels(image, grid, projectId) {
  const token = await freshAuthToken();
  const response = await fetch(
    `https://earthengine.googleapis.com/v1/projects/${encodeURIComponent(projectId)}/image:computePixels`,
    {
      method: "POST",
      headers: {
        Authorization: token,
        "Content-Type": "application/json",
        "X-Goog-User-Project": projectId,
      },
      body: JSON.stringify({
        expression: window.ee.Serializer.encodeCloudApi(image),
        fileFormat: "GEO_TIFF",
        grid,
        bandIds: [
          "pre_swir2", "pre_nir", "pre_red", "pre_clear",
          "current_swir2", "current_nir", "current_red", "current_clear",
        ],
        workloadTag: "wildfire-browser-demo",
      }),
    },
  );
  if (!response.ok) {
    let detail = "";
    try { detail = (await response.json()).error?.message || ""; } catch (_) { /* Binary or empty error. */ }
    throw new Error(`Earth Engine pixel request failed (${response.status})${detail ? `: ${detail}` : "."}`);
  }
  return response.blob();
}

function buildDailyCollection(roi, start, end, target) {
  const ee = window.ee;
  const cloudScore = ee.ImageCollection(CLOUD_COLLECTION).filterBounds(roi).filterDate(start, end);
  const linked = ee.ImageCollection(S2_COLLECTION)
    .filterBounds(roi)
    .filterDate(start, end)
    .linkCollection(cloudScore, ["cs_cdf"])
    .map(image => {
      const clear = image.select("cs_cdf").unmask(0).gte(CLOUD_SCORE_THRESHOLD).rename("cloud_mask").uint8();
      return image.addBands(clear).set("acquisition_day", image.date().format("YYYY-MM-dd"));
    });
  const days = ee.List(linked.aggregate_array("acquisition_day")).distinct();
  return ee.ImageCollection.fromImages(days.map(dayValue => {
    const day = ee.String(dayValue);
    const mosaic = linked.filter(ee.Filter.eq("acquisition_day", day)).mosaic();
    const clearValue = mosaic.select("cloud_mask").reduceRegion({
      reducer: ee.Reducer.mean(), geometry: roi, scale: 60, maxPixels: 1e8, bestEffort: true,
    }).get("cloud_mask");
    const clearRate = ee.Number(ee.Algorithms.If(clearValue, clearValue, 0)).multiply(100);
    const dateDifference = ee.Date(day).difference(target, "day").abs();
    const selectionScore = dateDifference.multiply(2).add(ee.Number(100).subtract(clearRate).multiply(0.25));
    return mosaic.set({
      acquisition_day: day,
      roi_clear_rate: clearRate,
      date_difference: dateDifference,
      selection_score: selectionScore,
      "system:time_start": ee.Date(day).millis(),
    });
  }));
}

function chooseImage(collection) {
  const ee = window.ee;
  const acceptable = collection.filter(ee.Filter.gte("roi_clear_rate", 60));
  const ranked = ee.ImageCollection(ee.Algorithms.If(acceptable.size().gt(0), acceptable, collection)).sort("selection_score");
  return ee.Image(ranked.first());
}

function prepareImage(image, prefix) {
  const spectral = image.select(DATASET_BANDS).clamp(0, REFLECTANCE_SCALE).int16();
  const clear = image.select("cloud_mask").unmask(0).uint8();
  return spectral.addBands(clear).rename([
    `${prefix}_swir2`, `${prefix}_nir`, `${prefix}_red`, `${prefix}_clear`,
  ]);
}

async function selectEarthEngineImages(roi, ignitionDate, targetDate, preDays, targetRadius) {
  const ee = window.ee;
  const ignition = ee.Date(ignitionDate);
  const target = ee.Date(targetDate);
  const preCollection = buildDailyCollection(roi, ignition.advance(-preDays, "day"), ignition, ignition.advance(-1, "day"));
  const currentCollection = buildDailyCollection(roi, target.advance(-targetRadius, "day"), target.advance(targetRadius + 1, "day"), target);
  const counts = await Promise.all([eeEvaluate(preCollection.size()), eeEvaluate(currentCollection.size())]);
  if (!counts[0]) throw new Error(`No Sentinel-2 image was found in the ${preDays}-day pre-fire window.`);
  if (!counts[1]) throw new Error(`No Sentinel-2 image was found within ±${targetRadius} days of the target date.`);
  const preImage = chooseImage(preCollection);
  const currentImage = chooseImage(currentCollection);
  const metadata = await eeEvaluate(ee.Dictionary({
    pre_date: preImage.get("acquisition_day"),
    pre_clear: preImage.get("roi_clear_rate"),
    pre_candidates: counts[0],
    current_date: currentImage.get("acquisition_day"),
    current_clear: currentImage.get("roi_clear_rate"),
    current_candidates: counts[1],
  }));
  return { preImage, currentImage, metadata };
}

async function readStack(arrayBuffer) {
  const tiff = await fromArrayBuffer(arrayBuffer);
  const image = await tiff.getImage();
  const width = image.getWidth();
  const height = image.getHeight();
  const sampleCount = image.getSamplesPerPixel();
  if (sampleCount < 8) throw new Error(`Earth Engine returned ${sampleCount} bands; eight were expected.`);
  const rasters = await image.readRasters({ samples: [0, 1, 2, 3, 4, 5, 6, 7], interleave: false });
  const makeRaster = offset => {
    const clear = new Uint8Array(width * height);
    for (let index = 0; index < clear.length; index += 1) clear[index] = rasters[offset + 3][index] > 0.5 ? 1 : 0;
    return { width, height, bands: [rasters[offset], rasters[offset + 1], rasters[offset + 2]], clear };
  };
  return { pre: makeRaster(0), current: makeRaster(4) };
}

async function createModelSession() {
  if (state.modelSession) return state.modelSession;
  if (!window.ort) throw new Error("ONNX Runtime did not load. Reload the page and try again.");
  window.ort.env.wasm.wasmPaths = ORT_ASSET_PATH;
  window.ort.env.wasm.numThreads = 1;
  try {
    if (!navigator.gpu) throw new Error("WebGPU unavailable");
    state.modelSession = await window.ort.InferenceSession.create(MODEL_PATH, { executionProviders: ["webgpu"] });
    state.inferenceEngine = "WebGPU";
  } catch (_) {
    state.modelSession = await window.ort.InferenceSession.create(MODEL_PATH, { executionProviders: ["wasm"] });
    state.inferenceEngine = "WebAssembly";
  }
  return state.modelSession;
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

async function inferProgression(elapsedDays) {
  const session = await createModelSession();
  const xs = tileStarts(state.current.width);
  const ys = tileStarts(state.current.height);
  const total = xs.length * ys.length;
  const accumulated = new Float32Array(state.current.width * state.current.height);
  const counts = new Uint16Array(accumulated.length);
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
      setProgress(0.62 + 0.36 * completed / total, `Running model tile ${completed} of ${total}…`);
      await new Promise(requestAnimationFrame);
    }
  }
  for (let index = 0; index < accumulated.length; index += 1) if (counts[index]) accumulated[index] /= counts[index];
  return accumulated;
}

function validateRunInputs() {
  if (!state.eeConnected) throw new Error("Connect Earth Engine first.");
  const latitude = Number(elements.latitude.value);
  const longitude = Number(elements.longitude.value);
  if (!Number.isFinite(latitude) || latitude < -85 || latitude > 85) throw new Error("Latitude must be between −85 and 85.");
  if (!Number.isFinite(longitude) || longitude < -180 || longitude > 180) throw new Error("Longitude must be between −180 and 180.");
  const ignition = new Date(`${elements.ignitionDate.value}T00:00:00Z`);
  const target = new Date(`${elements.targetDate.value}T00:00:00Z`);
  const elapsedDays = Math.round((target - ignition) / 86400000);
  if (!Number.isFinite(elapsedDays)) throw new Error("Choose valid ignition and target dates.");
  if (elapsedDays < 0) throw new Error("The target observation cannot be before ignition.");
  return { latitude, longitude, elapsedDays };
}

async function runEarthEngineWorkflow() {
  elements.runButton.disabled = true;
  elements.results.hidden = true;
  setMessage("");
  setProgress(0.02, "Building Earth Engine query…");
  try {
    const { latitude, longitude, elapsedDays } = validateRunInputs();
    state.elapsedDays = elapsedDays;
    const ee = window.ee;
    const roi = ee.Geometry.Point([longitude, latitude]).buffer(Number(elements.roiSize.value) * 500).bounds();
    setProgress(0.08, "Searching Sentinel-2 and Cloud Score+…");
    const selected = await selectEarthEngineImages(
      roi,
      elements.ignitionDate.value,
      elements.targetDate.value,
      Number(elements.preWindow.value),
      Number(elements.targetWindow.value),
    );
    state.selected = selected.metadata;
    setProgress(0.22, `Selected ${selected.metadata.pre_date} and ${selected.metadata.current_date}`);
    const stack = prepareImage(selected.preImage, "pre").addBands(prepareImage(selected.currentImage, "current"));
    const grid = webMercatorGrid(latitude, longitude, Number(elements.roiSize.value));
    setProgress(0.34, "Computing aligned Sentinel-2 pixels…");
    state.sourceBlob = await computePixels(stack, grid, elements.projectId.value.trim());
    const rasters = await readStack(await state.sourceBlob.arrayBuffer());
    state.pre = rasters.pre;
    state.current = rasters.current;
    setProgress(0.56, `Loaded ${state.current.width} × ${state.current.height} pixels; loading model…`);
    state.probability = await inferProgression(elapsedDays);
    renderResults();
    setProgress(1, "Earth Engine retrieval and inference complete");
    elements.results.hidden = false;
    elements.results.scrollIntoView({ behavior: "smooth", block: "start" });
  } catch (error) {
    console.error(error);
    elements.progressWrap.hidden = true;
    setMessage(normalizeError(error), true);
  } finally {
    elements.runButton.disabled = !state.eeConnected;
  }
}

function previewGeometry(width, height) {
  const scale = Math.min(1, 900 / Math.max(width, height));
  return { width: Math.max(1, Math.round(width * scale)), height: Math.max(1, Math.round(height * scale)) };
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

function falseColor(raster, index) {
  if (!raster.clear[index]) return [104, 110, 108, 255];
  return raster.bands.map(band => {
    const normalized = Math.min(1, Math.max(0, Number(band[index]) / REFLECTANCE_SCALE));
    return Math.round(255 * Math.pow(normalized, 0.72));
  });
}

function probabilityColor(value) {
  const stops = [[0, 3, 8, 7], [0.35, 57, 38, 29], [0.6, 223, 75, 32], [0.8, 255, 173, 66], [1, 255, 252, 229]];
  let upper = 1;
  while (upper < stops.length && value > stops[upper][0]) upper += 1;
  const high = stops[Math.min(upper, stops.length - 1)];
  const low = stops[Math.max(0, upper - 1)];
  const amount = high[0] === low[0] ? 0 : (value - low[0]) / (high[0] - low[0]);
  return [1, 2, 3].map(index => Math.round(low[index] + (high[index] - low[index]) * amount));
}

function maskColor(index, threshold) {
  if (!state.current.clear[index]) return [155, 155, 155, 255];
  return state.probability[index] >= threshold ? [255, 255, 255, 255] : [0, 0, 0, 255];
}

function renderResults() {
  const threshold = Number(elements.threshold.value);
  elements.thresholdValue.value = threshold.toFixed(2);
  renderToCanvas(elements.preCanvas, index => falseColor(state.pre, index));
  renderToCanvas(elements.currentCanvas, index => falseColor(state.current, index));
  renderToCanvas(elements.probabilityCanvas, index => state.current.clear[index] ? [...probabilityColor(state.probability[index]), 255] : [105, 105, 105, 255]);
  renderToCanvas(elements.maskCanvas, index => maskColor(index, threshold));
  let valid = 0;
  let burned = 0;
  for (let index = 0; index < state.probability.length; index += 1) {
    if (!state.current.clear[index]) continue;
    valid += 1;
    if (state.probability[index] >= threshold) burned += 1;
  }
  elements.preDateSummary.textContent = state.selected.pre_date;
  elements.preClearSummary.textContent = `${Number(state.selected.pre_clear).toFixed(1)}% clear · ${state.selected.pre_candidates} candidate days`;
  elements.currentDateSummary.textContent = state.selected.current_date;
  elements.currentClearSummary.textContent = `${Number(state.selected.current_clear).toFixed(1)}% clear · ${state.selected.current_candidates} candidate days`;
  elements.elapsedSummary.textContent = `${state.elapsedDays} day${state.elapsedDays === 1 ? "" : "s"}`;
  elements.engineSummary.textContent = `${state.inferenceEngine} browser inference`;
  elements.burnedSummary.textContent = valid ? `${(100 * burned / valid).toFixed(1)}%` : "No clear pixels";
  elements.pixelSummary.textContent = `${burned.toLocaleString()} of ${valid.toLocaleString()} clear pixels`;
}

function downloadMask() {
  if (!state.probability) return;
  const threshold = Number(elements.threshold.value);
  const canvas = document.createElement("canvas");
  renderToCanvas(canvas, index => maskColor(index, threshold), true);
  canvas.toBlob(blob => {
    if (!blob) return;
    const link = document.createElement("a");
    link.href = URL.createObjectURL(blob);
    link.download = `wildfire_mask_${state.selected.current_date}_t${threshold.toFixed(2)}.png`;
    link.click();
    setTimeout(() => URL.revokeObjectURL(link.href), 1000);
  }, "image/png");
}

function downloadSourceStack() {
  if (!state.sourceBlob) return;
  const link = document.createElement("a");
  link.href = URL.createObjectURL(state.sourceBlob);
  link.download = `sentinel2_pair_${state.selected.pre_date}_${state.selected.current_date}.tif`;
  link.click();
  setTimeout(() => URL.revokeObjectURL(link.href), 1000);
}

elements.connectButton.addEventListener("click", connectEarthEngine);
elements.runButton.addEventListener("click", runEarthEngineWorkflow);
elements.scenePreset.addEventListener("change", event => applyScenePreset(event.target.value));
elements.latitude.addEventListener("change", () => {
  markSceneCustom();
  updateMapSelection();
});
elements.longitude.addEventListener("change", () => {
  markSceneCustom();
  updateMapSelection();
});
elements.roiSize.addEventListener("input", () => {
  markSceneCustom();
  updateMapSelection(false);
});
elements.ignitionDate.addEventListener("change", () => markSceneCustom());
elements.targetDate.addEventListener("change", () => markSceneCustom());
elements.mapExpand.addEventListener("click", toggleMapWorkspace);
document.addEventListener("fullscreenchange", () => {
  setMapExpanded(document.fullscreenElement === elements.mapWorkspace);
});
elements.threshold.addEventListener("input", () => state.probability && renderResults());
elements.downloadMask.addEventListener("click", downloadMask);
elements.downloadStack.addEventListener("click", downloadSourceStack);

elements.projectId.value = localStorage.getItem("wildfire-ee-project")
  || localStorage.getItem("flamenet-ee-project")
  || elements.projectId.value;
elements.clientId.value = localStorage.getItem("wildfire-ee-client-id")
  || localStorage.getItem("flamenet-ee-client-id")
  || "";
initializeMap();
applyScenePreset(elements.scenePreset.value);
