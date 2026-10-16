/*
 * FlameNet-S2 in the browser.
 *
 * Earth Engine supplies a pre/post Sentinel-2 pair, onnxruntime-web runs the
 * trained network over it, and the burned-area map comes back without anything
 * leaving the machine. The point of the exercise is that the numbers match the
 * research pipeline, so every preprocessing constant comes from
 * ./model/model_meta.json, which scripts/export_onnx.py writes from
 * src/datasets/s2_fire.py at export time. Nothing about the model is duplicated
 * here as a literal.
 *
 * What the model expects, and therefore what this reproduces exactly:
 *
 *   collection   COPERNICUS/S2_SR_HARMONIZED (L2A surface reflectance), with
 *                cs_cdf linked from Cloud Score+. NOT the L1C TOA collection.
 *   bands        B12, B8, B4, cs_cdf, in that order, per phase
 *   reflectance  DN clamped to [0, 5000], scaled by 1e-4, clipped to 0.5, then
 *                standardised with the 2017-2023 training means and stds
 *   cs_cdf       a model INPUT channel in its own right, continuous, not a
 *                binary cloud mask; the fourth channel is what tells the
 *                network which pixels it cannot trust
 *   grid         20 m in the local UTM zone. A 20 m Web Mercator grid is 20 m
 *                only at the equator: at 56 N it is 11 m on the ground, and the
 *                model would be shown imagery at 1.8x its training resolution.
 *   pre scene    the PREVIOUS YEAR'S FIRE SEASON, which controls for phenology
 *                and avoids the smoke-filled active-fire period. It is not a
 *                lookback of a few weeks before ignition, and it is not simply
 *                the target date minus a year: the archive searches the fire's
 *                own date window shifted back one year (gee/s1s2.py:356), and a
 *                fire observed months after it stopped burning has a post date
 *                well outside its season. The pre date is supplied rather than
 *                derived, so the presets carry the one the dataset loader
 *                selected for each fire.
 *   scene choice the clearest candidate by mean cs_cdf minus the no-data
 *                fraction, which is the rule the dataset loader applies
 *   inference    256 px patches, stride 128, Gaussian-blended, with the tile
 *                context encoder's output sampled at each patch's footprint,
 *                then components under 50 px dropped
 *   cloud        pixels either phase could not see through are reported as
 *                obscured and excluded, not scored as unburned
 *
 * The one place the browser cannot be byte-identical to the research path is
 * the context thumbnail: Earth Engine averages the 20 m pixels into the 256 px
 * context grid with reduceResolution, which ignores masked pixels, whereas the
 * loader fills them with the channel mean before averaging. 98% of selected
 * scenes have no masked pixels at all, and the thumbnail only feeds the FiLM
 * modulation. scripts/verify_browser_pipeline.py checks everything else against
 * scripts/predict_fire.py on a real tile: 1 pixel in 377,745 disagrees.
 */

import { fromArrayBuffer } from "https://cdn.jsdelivr.net/npm/geotiff@2.1.3/+esm";
import {
  blendWindow, blitChannel, chunkEdges, cloudKeepMask, convertChannel, extractPatch,
  falseColour, gaussianWeight, jointRamp, normaliseBlend, removeSmallComponents,
  sampleContext, standardise, standardiseInto, windowStarts,
} from "./pipeline.js";

const MODEL_DIR = "./model/";
const ORT_ASSET_PATH = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.23.2/dist/";
const EE_SCOPE = "https://www.googleapis.com/auth/earthengine.readonly";

// Earth Engine refuses a computePixels response over 32 MiB. The detail window
// is eight bands of int16, so ~1024 px square is the most one request can carry
// (17 MB, with headroom); the reflectance DN are integers in the archive anyway
// and cs_cdf quantised to 1e-4 moves a standardised channel by 1.5e-3.
const DETAIL_CHUNK_PX = 1024;
// The window itself is tiled across as many requests as it needs, so the real
// ceiling is browser memory, not the response cap. At 4096 px the eight float32
// planes are 537 MB before the accumulators, which is why the reflectance is
// standardised in place and why the slider warns past 2048 px.
const DETAIL_MAX_PX = 4096;
const CS_QUANT = 10000;
const NODATA = -9999;

// Phase-prefixed band names in the order both requests return them.
const PHASES = ["pre", "post"];

const SCENE_PRESETS = Object.freeze({
  // Test-split fires (2025, held out of training and validation). Dates are the
  // scenes src/datasets/s2_fire.py itself selects for these tiles, so the demo
  // runs on exactly the pair the reported test metrics were computed from.
  CA_2025_SK_433: {
    label: "2025 · Saskatchewan · 170,576 ha",
    latitude: 54.8299, longitude: -102.3205,
    detailKm: 71.68, contextKm: 100,
    pre: "2024-09-15", target: "2025-11-04",
    note: "NBAC test split · boreal forest · model never saw 2025",
  },
  CA_2025_AB_480: {
    label: "2025 · Alberta · 119,527 ha",
    latitude: 56.9163, longitude: -113.6742,
    detailKm: 58.88, contextKm: 80,
    pre: "2024-08-11", target: "2025-10-15",
    note: "NBAC test split · mixed boreal · model never saw 2025",
  },
  CA_2025_ON_594: {
    label: "2025 · Ontario · 159,734 ha",
    latitude: 52.8423, longitude: -93.8993,
    detailKm: 69.12, contextKm: 96,
    pre: "2024-06-10", target: "2025-08-31",
    note: "NBAC test split · boreal shield · model never saw 2025",
  },
  // The supervisor's independent check fire: outside the NBAC splits entirely,
  // validated against a BC Wildfire Service airborne-GPS perimeter.
  CA_2026_BC_K70748: {
    label: "2026 · BC · K70748 · 61,009 ha perimeter",
    latitude: 51.2716, longitude: -122.3370,
    detailKm: 56.32, contextKm: 76,
    pre: "2025-08-22", target: "2026-09-16",
    note: "Independent validation fire · no NBAC mask · union recall 0.912, "
        + "precision 0.961 after attributing neighbouring fires",
  },
  // Out of domain on purpose: tropical peat, a hemisphere and a biome away from
  // anything in training. The S2 model transfers here and the result is
  // corroborated by dNBR; there is no burned-area reference for the period.
  ID_2026_KETAPANG: {
    label: "2026 · Ketapang, Kalimantan · out of domain",
    latitude: -1.7500, longitude: 110.1300,
    detailKm: 35.84, contextKm: 50,
    pre: "2025-10-07", target: "2026-09-02",
    note: "Out-of-domain transfer · tropical peatland · dNBR-corroborated, "
        + "no burned-area reference exists for Aug-Oct 2026",
  },
});

const state = {
  meta: null,
  eeConnected: false,
  authPopupReady: false,
  authAttempt: 0,
  ctxSession: null,
  patchSession: null,
  inferenceEngine: null,
  pre: null,          // detail window, reflectance units
  post: null,
  ctxPre: null,       // 256 px context thumbnail, same units
  ctxPost: null,
  probability: null,
  keep: null,         // null when the run is 3-channel or masking is disabled
  geometry: null,
  selected: null,
  sourceBlobs: null,   // one per computePixels request the window needed
  map: null,
  marker: null,
  detailRect: null,
  contextRect: null,
};

const elements = {};
for (const id of [
  "project-id", "client-id", "connect-button", "ee-status", "connection-message",
  "scene-preset", "preset-meta", "latitude", "longitude", "detail-size",
  "detail-value", "context-size", "context-value", "pre-date", "target-date",
  "pre-window", "target-window", "map-workspace", "map-expand", "map-coordinate",
  "run-button", "progress-wrap", "progress-text", "progress-value", "progress-bar",
  "message", "results", "threshold", "threshold-value", "pre-date-summary",
  "pre-clear-summary", "current-date-summary", "current-clear-summary",
  "lag-summary", "engine-summary", "burned-summary", "pixel-summary",
  "obscured-summary", "obscured-detail", "window-cost", "pre-canvas", "current-canvas",
  "probability-canvas", "mask-canvas", "download-mask", "download-stack",
  "model-summary",
]) {
  elements[id.replace(/-([a-z])/g, (_, c) => c.toUpperCase())] = document.querySelector(`#${id}`);
}

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

// ── model metadata ───────────────────────────────────────────────────────────

async function loadMeta() {
  const response = await fetch(`${MODEL_DIR}model_meta.json`);
  if (!response.ok) throw new Error(`model_meta.json is missing (${response.status}).`);
  const meta = await response.json();
  state.meta = meta;
  elements.modelSummary.textContent =
    `${meta.name} · ${meta.in_ch_per_phase} channels per phase (${meta.bands.join(", ")}) · `
    + `${meta.params.toLocaleString()} parameters · ${meta.patch_size} px patches, `
    + `stride ${meta.stride} · trained ${meta.train_years[0]}-${meta.train_years.at(-1)}, `
    + `validated ${meta.val_years.join("/")}, tested ${meta.test_years.join("/")}`;
  elements.threshold.value = meta.threshold;
  elements.thresholdValue.value = Number(meta.threshold).toFixed(2);
  return meta;
}

// ── map ──────────────────────────────────────────────────────────────────────

function initializeMap() {
  const latitude = Number(elements.latitude.value);
  const longitude = Number(elements.longitude.value);
  state.map = window.L.map("map", { zoomControl: true }).setView([latitude, longitude], 9);
  window.L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
    maxZoom: 18,
    attribution: "&copy; OpenStreetMap contributors",
  }).addTo(state.map);
  const icon = window.L.divIcon({ className: "roi-center-marker", iconSize: [15, 15] });
  state.marker = window.L.marker([latitude, longitude], { icon, interactive: false }).addTo(state.map);
  // Two rectangles: the extent the context encoder sees, and the extent that is
  // actually segmented at 20 m inside it.
  state.contextRect = window.L.rectangle(squareBounds(latitude, longitude, contextKm()), {
    color: "#8c9a94", weight: 1, dashArray: "5 4", fillOpacity: 0,
  }).addTo(state.map);
  state.detailRect = window.L.rectangle(squareBounds(latitude, longitude, detailKm()), {
    color: "#ff6b35", weight: 2, fillColor: "#ff6b35", fillOpacity: 0.08,
  }).addTo(state.map);
  state.map.on("click", event => {
    elements.latitude.value = event.latlng.lat.toFixed(5);
    elements.longitude.value = event.latlng.lng.toFixed(5);
    markSceneCustom("Custom scene · map centre set manually; set its dates at right.");
    updateMapSelection(false);
  });
}

const detailKm = () => Number(elements.detailSize.value);
const contextKm = () => Number(elements.contextSize.value);

function squareBounds(latitude, longitude, sizeKm) {
  const halfKm = sizeKm / 2;
  const dLat = halfKm / 111.32;
  const dLon = halfKm / (111.32 * Math.max(0.15, Math.cos(latitude * Math.PI / 180)));
  return [[latitude - dLat, longitude - dLon], [latitude + dLat, longitude + dLon]];
}

function updateMapSelection(pan = true) {
  const latitude = Number(elements.latitude.value);
  const longitude = Number(elements.longitude.value);
  if (!Number.isFinite(latitude) || !Number.isFinite(longitude)) return;
  // The context window must contain the detail window: the patch positions are
  // normalised against it, and a patch outside it would sample clamped context.
  if (contextKm() < detailKm()) elements.contextSize.value = detailKm();
  state.marker.setLatLng([latitude, longitude]);
  state.detailRect.setBounds(squareBounds(latitude, longitude, detailKm()));
  state.contextRect.setBounds(squareBounds(latitude, longitude, contextKm()));
  const px = detailPixels();
  elements.detailValue.value = `${detailKm().toFixed(2)} km · ${px}×${px} px`;
  elements.contextValue.value = `${contextKm().toFixed(0)} km`;
  elements.mapCoordinate.textContent = `${latitude.toFixed(4)}, ${longitude.toFixed(4)}`;
  updateWindowCost(px);
  if (pan) state.map.panTo([latitude, longitude]);
}

/**
 * What the chosen window will cost, before the user spends it.
 *
 * A wide window is several Earth Engine requests and hundreds of model windows,
 * so the run can take minutes and hundreds of megabytes. Saying so up front is
 * cheaper than letting someone drag the slider to the top and wait.
 */
function updateWindowCost(px) {
  if (!state.meta) return;
  const meta = state.meta;
  const chunks = Math.ceil(px / DETAIL_CHUNK_PX) ** 2;
  const windows = Math.ceil(Math.max(1, (px - meta.patch_size) / meta.stride) + 1) ** 2;
  const planesMb = (px * px * 2 * meta.in_ch_per_phase * 4) / 2 ** 20;
  const heavy = px > 2048;
  elements.windowCost.textContent =
    `${chunks} Earth Engine request${chunks === 1 ? "" : "s"} · ${windows} model `
    + `window${windows === 1 ? "" : "s"} · ~${Math.round(planesMb)} MB in memory`
    + (heavy ? " · minutes on WebAssembly, and a lot of RAM" : "");
  elements.windowCost.classList.toggle("is-heavy", heavy);
}

function detailPixels() {
  const scale = state.meta?.scale_m ?? 20;
  return Math.min(DETAIL_MAX_PX, Math.max(state.meta?.patch_size ?? 256,
    Math.round(detailKm() * 1000 / scale)));
}

function markSceneCustom(message = "Custom scene · coordinates, dates or window adjusted manually.") {
  elements.scenePreset.value = "";
  elements.presetMeta.textContent = message;
}

function applyScenePreset(eventId) {
  const preset = SCENE_PRESETS[eventId];
  if (!preset) return;
  elements.latitude.value = preset.latitude;
  elements.longitude.value = preset.longitude;
  elements.detailSize.value = preset.detailKm;
  elements.contextSize.value = preset.contextKm;
  elements.preDate.value = preset.pre;
  elements.targetDate.value = preset.target;
  elements.presetMeta.textContent = `${eventId} · ${preset.note}`;
  updateMapSelection(false);
  state.map.fitBounds(squareBounds(preset.latitude, preset.longitude, preset.contextKm),
    { padding: [40, 40], maxZoom: 11 });
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

// ── Earth Engine connection ──────────────────────────────────────────────────

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
    clientId, success, failure, [EE_SCOPE],
    () => { clearTimeout(fallbackTimer); offerPopup(); },
    true,
  );
}

function eeEvaluate(computed) {
  return new Promise((resolve, reject) => {
    computed.evaluate((value, error) => error ? reject(new Error(normalizeError(error))) : resolve(value));
  });
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

// ── Earth Engine imagery ─────────────────────────────────────────────────────

function utmCrs(longitude, latitude) {
  const zone = Math.floor((longitude + 180) / 6) + 1;
  return `EPSG:${(latitude >= 0 ? 32600 : 32700) + zone}`;
}

/** A square grid of `pixels` cells of `scale` m, centred on (x, y) in `crsCode`. */
function utmGrid(x, y, pixels, scale, crsCode) {
  const span = pixels * scale;
  return {
    dimensions: { width: pixels, height: pixels },
    affineTransform: {
      scaleX: scale, shearX: 0, translateX: x - span / 2,
      shearY: 0, scaleY: -scale, translateY: y + span / 2,
    },
    crsCode,
  };
}

const gridTransformArray = grid => [
  grid.affineTransform.scaleX, grid.affineTransform.shearX, grid.affineTransform.translateX,
  grid.affineTransform.shearY, grid.affineTransform.scaleY, grid.affineTransform.translateY,
];

/**
 * One image per datatake, which is the scene unit the archive was built from:
 * a fire spanning two granules of the same overpass is one scene, not two.
 * Mirrors gee/s1s2.py's mosaic_by_datatake, as copied in scripts/export_fire_s2.py.
 */
function datatakeScenes(roi, start, end) {
  const ee = window.ee;
  const meta = state.meta;
  const linked = ee.ImageCollection(meta.collection)
    .filterBounds(roi)
    .filterDate(start, end)
    .linkCollection(ee.ImageCollection(meta.cloud_collection), [meta.qa_band]);
  const ids = ee.List(linked.aggregate_array("DATATAKE_IDENTIFIER")).distinct();
  return ee.ImageCollection(ids.map(id => {
    const subset = linked.filter(ee.Filter.eq("DATATAKE_IDENTIFIER", id));
    const first = ee.Image(subset.first());
    const start_ = first.get("system:time_start");
    // rescale_s2: clamp the reflectance DN before anything averages them, so
    // the DN semantics match the training archive exactly.
    const refl = subset.mosaic().select(["B12", "B8", "B4"]).clamp(0, meta.dn_clamp);
    const qa = subset.mosaic().select([meta.qa_band]);
    return refl.addBands(qa).set({
      DATATAKE_IDENTIFIER: id,
      "system:time_start": start_,
      acquisition_day: ee.Date(start_).format("YYYY-MM-dd"),
    });
  }));
}

/**
 * Score each scene the way src/datasets/s2_fire.py's _score does — mean cs_cdf
 * minus the fraction of pixels the export would leave as no-data — and keep the
 * best. Clarity decides, with the earliest date as the tiebreak; proximity to
 * the requested date is handled by keeping the search window narrow instead.
 */
function scoreScenes(collection, roi, scoreScale) {
  const ee = window.ee;
  const meta = state.meta;
  return collection.map(image => {
    const img = ee.Image(image);
    const clear = img.select(meta.qa_band).reduceRegion({
      reducer: ee.Reducer.mean(), geometry: roi, scale: scoreScale,
      maxPixels: 1e8, bestEffort: true,
    }).get(meta.qa_band);
    const covered = img.select("B12").mask().reduceRegion({
      reducer: ee.Reducer.mean(), geometry: roi, scale: scoreScale,
      maxPixels: 1e8, bestEffort: true,
    }).get("B12");
    const csMean = ee.Number(ee.Algorithms.If(clear, clear, 0));
    const coverage = ee.Number(ee.Algorithms.If(covered, covered, 0));
    return img.set({
      cs_mean: csMean,
      nan_frac: ee.Number(1).subtract(coverage),
      usability: csMean.subtract(ee.Number(1).subtract(coverage)),
    });
  });
}

function pickScene(collection) {
  const ee = window.ee;
  // MAX_NAN_FRAC: candidates whose footprint is mostly no-data are set aside
  // first, unless there is nothing better.
  const usable = collection.filter(ee.Filter.lte("nan_frac", 0.5));
  const pool = ee.ImageCollection(ee.Algorithms.If(usable.size().gt(0), usable, collection));
  return ee.Image(pool.sort("usability", false).first());
}

/**
 * The scale the scene-ranking reductions run at.
 *
 * Every candidate gets two reduceRegion calls, and a wide window can have 40 of
 * them in flight: at a fixed fine scale Earth Engine answers "Too many
 * concurrent aggregations" and the run dies before any imagery is fetched. The
 * scale therefore grows with the window, keeping each reduction near 150x150
 * cells whatever the extent. It costs nothing real — these numbers only rank
 * scenes, and over a 72 km window the mean cs_cdf moves by 0.002 between 480 m
 * and 1920 m.
 */
function scoringScale(detailPixels) {
  const metres = detailPixels * state.meta.scale_m;
  return Math.max(state.meta.scale_m * 6, Math.round(metres / 150));
}

async function selectScenes(roi, preDate, targetDate, preRadius, targetRadius, scoreScale) {
  const ee = window.ee;
  const window_ = (centre, radius) => [
    ee.Date(centre).advance(-radius, "day"), ee.Date(centre).advance(radius + 1, "day"),
  ];
  const [preStart, preEnd] = window_(preDate, preRadius);
  const [postStart, postEnd] = window_(targetDate, targetRadius);
  const preAll = datatakeScenes(roi, preStart, preEnd);
  const postAll = datatakeScenes(roi, postStart, postEnd);
  const counts = await Promise.all([eeEvaluate(preAll.size()), eeEvaluate(postAll.size())]);
  if (!counts[0]) {
    throw new Error(`No Sentinel-2 scene within ±${preRadius} days of the pre-fire date `
      + `${preDate}. Widen the pre-fire search or move the date.`);
  }
  if (!counts[1]) {
    throw new Error(`No Sentinel-2 scene within ±${targetRadius} days of ${targetDate}.`);
  }
  const preImage = pickScene(scoreScenes(preAll, roi, scoreScale));
  const postImage = pickScene(scoreScenes(postAll, roi, scoreScale));
  const metadata = await eeEvaluate(ee.Dictionary({
    pre_date: preImage.get("acquisition_day"),
    pre_cs: preImage.get("cs_mean"),
    pre_nan: preImage.get("nan_frac"),
    pre_candidates: counts[0],
    post_date: postImage.get("acquisition_day"),
    post_cs: postImage.get("cs_mean"),
    post_nan: postImage.get("nan_frac"),
    post_candidates: counts[1],
  }));
  return { preImage, postImage, metadata };
}

/** Eight bands, pre then post, named so both requests return a fixed order. */
function bandIds() {
  return PHASES.flatMap(phase => state.meta.bands.map(band => `${phase}_${band}`));
}

function stackPhases(preImage, postImage) {
  const ee = window.ee;
  const names = state.meta.bands;
  const rename = (image, phase) => ee.Image(image).select(names, names.map(b => `${phase}_${b}`));
  return rename(preImage, "pre").addBands(rename(postImage, "post"));
}

/**
 * The detail window, at the model's native 20 m, as int16.
 *
 * Reflectance DN are already integers clamped to [0, 5000]; cs_cdf is carried as
 * cs x 10000 so all eight bands share one dtype and the response halves in size,
 * which is what lets the window reach 1024 px inside Earth Engine's 32 MiB cap.
 */
function detailImage(stack) {
  const names = bandIds();
  const qa = state.meta.qa_band;
  // cs_cdf is the only band that is not already an integer DN.
  const csBands = PHASES.map(phase => `${phase}_${qa}`);
  const scaled = stack.addBands(
    stack.select(csBands).multiply(CS_QUANT), csBands, true);
  return scaled.select(names).unmask(NODATA).round().int16();
}

/**
 * The context window, averaged onto the 256 px grid the tile context encoder
 * was trained on.
 *
 * reduceResolution against the request grid's own transform is an exact
 * area-weighted mean of the 20 m pixels — the same quantity cv2.INTER_AREA
 * computes in the loader — so the thumbnail is not resampled a second time on
 * the way out.
 */
function contextImage(stack, crsCode, grid, scale) {
  const ee = window.ee;
  const maxPixels = Math.max(64, Math.ceil((scale / state.meta.scale_m) ** 2) * 4);
  return stack
    .reproject({ crs: crsCode, scale: state.meta.scale_m })
    .reduceResolution({ reducer: ee.Reducer.mean(), bestEffort: false, maxPixels })
    .reproject({ crs: crsCode, crsTransform: gridTransformArray(grid) })
    .unmask(NODATA)
    .toFloat();
}

async function computePixels(image, grid, projectId, bands, tag) {
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
        bandIds: bands,
        workloadTag: "flamenet-s2-browser",
      }),
    },
  );
  if (!response.ok) {
    let detail = "";
    try { detail = (await response.json()).error?.message || ""; } catch (_) { /* binary or empty */ }
    throw new Error(`Earth Engine ${tag} request failed (${response.status})${detail ? `: ${detail}` : "."}`);
  }
  return response.blob();
}

/**
 * Read an eight-band GeoTIFF into two phases of reflectance, reproducing
 * src/datasets/s2_fire.py's _read_tif: DN to reflectance, no-data filled with
 * the channel mean (cs_cdf with 0, which reads as fully clouded rather than as
 * typically clear), then the reflectance channels clipped.
 */
/** How a band of the computePixels response is turned into reflectance. */
function channelOptions(channel, csScale) {
  const meta = state.meta;
  const isCs = meta.bands[channel] === meta.qa_band;
  return {
    fill: isCs ? 0 : meta.mean[channel],
    scale: isCs ? 1 / csScale : meta.dn_scale,
    clip: isCs ? null : meta.refl_clip,
    nodata: NODATA,
  };
}

async function readRasters(arrayBuffer) {
  const expected = PHASES.length * state.meta.in_ch_per_phase;
  const tiff = await fromArrayBuffer(arrayBuffer);
  const image = await tiff.getImage();
  const samples = image.getSamplesPerPixel();
  if (samples < expected) {
    throw new Error(`Earth Engine returned ${samples} bands; ${expected} were expected.`);
  }
  return {
    width: image.getWidth(),
    height: image.getHeight(),
    bands: await image.readRasters({
      samples: [...Array(expected).keys()], interleave: false,
    }),
  };
}

/** One request's worth: used for the context thumbnail, which always fits. */
async function readStack(arrayBuffer, csScale) {
  const nChannels = state.meta.in_ch_per_phase;
  const { width, height, bands } = await readRasters(arrayBuffer);
  const makePhase = offset => ({
    width, height,
    channels: Array.from({ length: nChannels }, (_, c) =>
      convertChannel(bands[offset + c], channelOptions(c, csScale))),
  });
  return { pre: makePhase(0), post: makePhase(nChannels) };
}

/**
 * Fetch the detail window, in as many computePixels requests as it takes.
 *
 * Earth Engine caps a response at 32 MiB, which with eight int16 bands is about
 * 1024 px square — far smaller than the fires this model is for, whose scars
 * run 55-75 km across. So the window is tiled, each chunk requested on its own
 * grid cut from the same 20 m UTM lattice, and blitted into one set of planes.
 * Because every chunk shares the parent grid's origin and scale, the reassembled
 * raster is pixel-identical to the single request it replaces.
 */
async function fetchDetailWindow(image, grid, projectId, bands, csScale, onProgress) {
  const meta = state.meta;
  const pixels = grid.dimensions.width;
  const a = grid.affineTransform;
  const rows = chunkEdges(pixels, DETAIL_CHUNK_PX);
  const cols = chunkEdges(pixels, DETAIL_CHUNK_PX);
  const total = (rows.length - 1) * (cols.length - 1);

  const nChannels = meta.in_ch_per_phase;
  const plane = pixels * pixels;
  const makePhase = () => ({
    width: pixels, height: pixels,
    channels: Array.from({ length: nChannels }, () => new Float32Array(plane)),
  });
  const pre = makePhase();
  const post = makePhase();

  const blobs = [];
  let done = 0;
  for (let r = 0; r < rows.length - 1; r += 1) {
    for (let c = 0; c < cols.length - 1; c += 1) {
      const row = rows[r];
      const col = cols[c];
      const chunkH = rows[r + 1] - row;
      const chunkW = cols[c + 1] - col;
      const chunkGrid = {
        dimensions: { width: chunkW, height: chunkH },
        affineTransform: {
          ...a,
          translateX: a.translateX + col * a.scaleX,
          translateY: a.translateY + row * a.scaleY,
        },
        crsCode: grid.crsCode,
      };
      onProgress(done, total, `${chunkW}×${chunkH} px`);
      const blob = await computePixels(image, chunkGrid, projectId, bands,
        total > 1 ? `detail chunk ${done + 1}/${total}` : "detail");
      blobs.push({ blob, row, col });
      const raster = await readRasters(await blob.arrayBuffer());
      if (raster.width !== chunkW || raster.height !== chunkH) {
        throw new Error(`Earth Engine returned ${raster.width}×${raster.height} px for a `
          + `${chunkW}×${chunkH} px chunk.`);
      }
      const geom = { targetWidth: pixels, chunkWidth: chunkW, chunkHeight: chunkH, row, col };
      for (const [phase, offset] of [[pre, 0], [post, nChannels]]) {
        for (let ch = 0; ch < nChannels; ch += 1) {
          blitChannel(raster.bands[offset + ch], phase.channels[ch],
            channelOptions(ch, csScale), geom);
        }
      }
      done += 1;
    }
  }
  return { pre, post, blobs, requests: total };
}

// ── inference ────────────────────────────────────────────────────────────────

async function createSessions() {
  if (state.ctxSession && state.patchSession) return;
  if (!window.ort) throw new Error("ONNX Runtime did not load. Reload the page and try again.");
  window.ort.env.wasm.wasmPaths = ORT_ASSET_PATH;
  window.ort.env.wasm.numThreads = 1;
  const graphs = state.meta.graphs;
  const create = provider => Promise.all([
    window.ort.InferenceSession.create(`${MODEL_DIR}${graphs.ctx}`, { executionProviders: [provider] }),
    window.ort.InferenceSession.create(`${MODEL_DIR}${graphs.patch}`, { executionProviders: [provider] }),
  ]);
  try {
    if (!navigator.gpu) throw new Error("WebGPU unavailable");
    [state.ctxSession, state.patchSession] = await create("webgpu");
    state.inferenceEngine = "WebGPU";
  } catch (_) {
    [state.ctxSession, state.patchSession] = await create("wasm");
    state.inferenceEngine = "WebAssembly";
  }
}

async function runInference(geometry) {
  const meta = state.meta;
  const ort = window.ort;
  const patch = meta.patch_size;
  const stride = meta.stride;
  const { width, height } = state.post;

  // In place: see the note in runWorkflow — at the top window size a second
  // copy of the detail planes would be another 537 MB. Anything needing raw
  // reflectance has already been computed by the time this runs.
  const preNorm = standardiseInto(state.pre.channels, meta.mean, meta.std);
  const postNorm = standardiseInto(state.post.channels, meta.mean, meta.std);

  // The thumbnail is small, so it keeps the allocating version.
  const norm = phase => standardise(phase.channels, meta.mean, meta.std);

  // Tile context: one pass over the context thumbnail, reused by every patch.
  const ctxPixels = meta.ctx_size * meta.ctx_size;
  const ctxInput = new Float32Array(2 * meta.in_ch_per_phase * ctxPixels);
  const thumbNorm = [...norm(state.ctxPre), ...norm(state.ctxPost)];
  thumbNorm.forEach((channel, index) => ctxInput.set(channel, index * ctxPixels));
  const ctxOutput = await state.ctxSession.run({
    ctx_in: new ort.Tensor("float32", ctxInput, [1, 2 * meta.in_ch_per_phase, meta.ctx_size, meta.ctx_size]),
  });
  const tileCtx = ctxOutput.tile_ctx.data;
  const [, ctxChannels, ctxH, ctxW] = ctxOutput.tile_ctx.dims;

  const weight = gaussianWeight(patch);
  const accumulated = new Float32Array(width * height);
  const weightSum = new Float32Array(width * height);

  const rows = windowStarts(height, patch, stride);
  const cols = windowStarts(width, patch, stride);
  const total = rows.length * cols.length;
  let done = 0;

  for (const r1 of rows) {
    for (const c1 of cols) {
      const pre = extractPatch(preNorm, width, height, r1, c1, patch);
      const post = extractPatch(postNorm, width, height, r1, c1, patch);
      // Patch position is normalised against the CONTEXT window — the extent
      // the encoder saw — not against the detail window inside it.
      const centreRow = geometry.detailOriginRow + r1 + patch / 2;
      const centreCol = geometry.detailOriginCol + c1 + patch / 2;
      const ctxBtn = sampleContext(
        tileCtx, ctxChannels, ctxH, ctxW,
        (centreRow / geometry.contextPixels) * 2 - 1,
        (centreCol / geometry.contextPixels) * 2 - 1,
        patch / geometry.contextPixels,
        patch / geometry.contextPixels,
        meta.btn_size,
      );
      const output = await state.patchSession.run({
        pre: new ort.Tensor("float32", pre, [1, meta.in_ch_per_phase, patch, patch]),
        post: new ort.Tensor("float32", post, [1, meta.in_ch_per_phase, patch, patch]),
        ctx_btn: new ort.Tensor("float32", ctxBtn, [1, ctxChannels, meta.btn_size, meta.btn_size]),
      });
      blendWindow(accumulated, weightSum, output.probability.data, weight,
        width, height, r1, c1, patch);
      done += 1;
      setProgress(0.62 + 0.36 * done / total, `Running model window ${done} of ${total}…`);
      await new Promise(requestAnimationFrame);
    }
  }
  return normaliseBlend(accumulated, weightSum);
}

// ── run ──────────────────────────────────────────────────────────────────────

function validateRunInputs() {
  if (!state.meta) throw new Error("The model metadata has not loaded yet.");
  if (!state.eeConnected) throw new Error("Connect Earth Engine first.");
  const latitude = Number(elements.latitude.value);
  const longitude = Number(elements.longitude.value);
  if (!Number.isFinite(latitude) || latitude < -85 || latitude > 85) {
    throw new Error("Latitude must be between −85 and 85.");
  }
  if (!Number.isFinite(longitude) || longitude < -180 || longitude > 180) {
    throw new Error("Longitude must be between −180 and 180.");
  }
  const pre = new Date(`${elements.preDate.value}T00:00:00Z`);
  const target = new Date(`${elements.targetDate.value}T00:00:00Z`);
  if (!Number.isFinite(pre.getTime()) || !Number.isFinite(target.getTime())) {
    throw new Error("Choose a valid pre-fire date and target observation.");
  }
  if (target <= pre) throw new Error("The target observation must be after the pre-fire date.");
  if (contextKm() < detailKm()) throw new Error("The context window cannot be smaller than the detail window.");
  return { latitude, longitude, lagDays: Math.round((target - pre) / 86400000) };
}

async function runWorkflow() {
  elements.runButton.disabled = true;
  elements.results.hidden = true;
  setMessage("");
  setProgress(0.02, "Building the Earth Engine query…");
  try {
    const { latitude, longitude, lagDays } = validateRunInputs();
    const meta = state.meta;
    const ee = window.ee;
    const projectId = elements.projectId.value.trim();
    const crsCode = utmCrs(longitude, latitude);

    // Everything is laid out on one 20 m UTM grid: the detail window is a
    // centred crop of the context window, so a patch's position inside the
    // context is exact rather than reprojected.
    const detailPx = detailPixels();
    const contextPx = Math.round(contextKm() * 1000 / meta.scale_m);
    const contextScale = (contextPx * meta.scale_m) / meta.ctx_size;

    setProgress(0.06, "Locating the window in UTM…");
    const [centreX, centreY] = await eeEvaluate(
      ee.Geometry.Point([longitude, latitude]).transform(crsCode, 0.01).coordinates());
    const detailGrid = utmGrid(centreX, centreY, detailPx, meta.scale_m, crsCode);
    const contextGrid = utmGrid(centreX, centreY, meta.ctx_size, contextScale, crsCode);
    const roi = ee.Geometry.Rectangle(
      [centreX - detailPx * meta.scale_m / 2, centreY - detailPx * meta.scale_m / 2,
        centreX + detailPx * meta.scale_m / 2, centreY + detailPx * meta.scale_m / 2],
      crsCode, false);
    const contextRoi = ee.Geometry.Rectangle(
      [centreX - contextPx * meta.scale_m / 2, centreY - contextPx * meta.scale_m / 2,
        centreX + contextPx * meta.scale_m / 2, centreY + contextPx * meta.scale_m / 2],
      crsCode, false);

    setProgress(0.1, `Searching ${meta.collection} and Cloud Score+…`);
    const selected = await selectScenes(
      roi, elements.preDate.value, elements.targetDate.value,
      Number(elements.preWindow.value), Number(elements.targetWindow.value),
      scoringScale(detailPx));
    state.selected = selected.metadata;
    setProgress(0.22,
      `Selected ${selected.metadata.pre_date} and ${selected.metadata.post_date}`);

    const stack = stackPhases(selected.preImage, selected.postImage);
    const bands = bandIds();

    const detail = await fetchDetailWindow(
      detailImage(stack), detailGrid, projectId, bands, CS_QUANT,
      (done, total, size) => setProgress(0.26 + 0.18 * done / total,
        total > 1
          ? `Fetching ${detailPx}×${detailPx} px at ${meta.scale_m} m — request `
            + `${done + 1} of ${total} (${size})…`
          : `Fetching ${detailPx}×${detailPx} px at ${meta.scale_m} m…`));
    state.pre = detail.pre;
    state.post = detail.post;
    state.sourceBlobs = detail.blobs;

    // Both of these need RAW reflectance, and runInference standardises the
    // planes in place to avoid a second 537 MB copy at the top window size.
    // So the cloud mask is taken now, and the two false-colour panels are
    // rasterised to their canvases now rather than kept as arrays — the
    // canvases are at most 900 px a side, where the planes are up to 4096.
    state.keep = (meta.cloud_valid_cs > 0 && meta.in_ch_per_phase > 3)
      ? cloudKeepMask(state.pre.channels[3], state.post.channels[3], meta.cloud_valid_cs)
      : null;
    const preRamp = jointRamp(state.pre.channels);
    const postRamp = jointRamp(state.post.channels);
    renderToCanvas(elements.preCanvas, i => falseColour(state.pre.channels, preRamp, i));
    renderToCanvas(elements.currentCanvas, i => falseColour(state.post.channels, postRamp, i));

    setProgress(0.46, `Fetching the ${contextKm().toFixed(0)} km context thumbnail…`);
    const contextBlob = await computePixels(
      contextImage(stack, crsCode, contextGrid, contextScale),
      contextGrid, projectId, bands, "context");
    const context = await readStack(await contextBlob.arrayBuffer(), 1);
    state.ctxPre = context.pre;
    state.ctxPost = context.post;

    const geometry = {
      detailPixels: detailPx,
      contextPixels: contextPx,
      // The detail window is centred in the context window.
      detailOriginRow: (contextPx - detailPx) / 2,
      detailOriginCol: (contextPx - detailPx) / 2,
      crsCode,
      lagDays,
    };
    state.geometry = geometry;

    setProgress(0.56, `Loading ${meta.name}…`);
    await createSessions();
    state.probability = await runInference(geometry);

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

// ── rendering ────────────────────────────────────────────────────────────────

function previewGeometry(width, height) {
  const scale = Math.min(1, 900 / Math.max(width, height));
  return { width: Math.max(1, Math.round(width * scale)), height: Math.max(1, Math.round(height * scale)) };
}

function renderToCanvas(canvas, pixelFunction, fullResolution = false) {
  const sourceWidth = state.post.width;
  const sourceHeight = state.post.height;
  const target = fullResolution
    ? { width: sourceWidth, height: sourceHeight }
    : previewGeometry(sourceWidth, sourceHeight);
  canvas.width = target.width;
  canvas.height = target.height;
  const context = canvas.getContext("2d");
  const output = context.createImageData(target.width, target.height);
  for (let y = 0; y < target.height; y += 1) {
    const sourceY = Math.min(sourceHeight - 1, Math.floor(y * sourceHeight / target.height));
    for (let x = 0; x < target.width; x += 1) {
      const sourceX = Math.min(sourceWidth - 1, Math.floor(x * sourceWidth / target.width));
      const index = sourceY * sourceWidth + sourceX;
      const targetIndex = (y * target.width + x) * 4;
      const colour = pixelFunction(index);
      output.data[targetIndex] = colour[0];
      output.data[targetIndex + 1] = colour[1];
      output.data[targetIndex + 2] = colour[2];
      output.data[targetIndex + 3] = colour.length > 3 ? colour[3] : 255;
    }
  }
  context.putImageData(output, 0, 0);
}

function probabilityColour(value) {
  const stops = [[0, 3, 8, 7], [0.35, 57, 38, 29], [0.6, 223, 75, 32], [0.8, 255, 173, 66], [1, 255, 252, 229]];
  let upper = 1;
  while (upper < stops.length && value > stops[upper][0]) upper += 1;
  const high = stops[Math.min(upper, stops.length - 1)];
  const low = stops[Math.max(0, upper - 1)];
  const amount = high[0] === low[0] ? 0 : (value - low[0]) / (high[0] - low[0]);
  return [1, 2, 3].map(i => Math.round(low[i] + (high[i] - low[i]) * amount));
}

function currentMask() {
  const meta = state.meta;
  const threshold = Number(elements.threshold.value);
  const binary = new Uint8Array(state.probability.length);
  for (let i = 0; i < binary.length; i += 1) binary[i] = state.probability[i] >= threshold ? 1 : 0;
  const filtered = removeSmallComponents(
    binary, state.post.width, state.post.height, meta.min_cc_area);
  if (state.keep) {
    for (let i = 0; i < filtered.length; i += 1) if (!state.keep[i]) filtered[i] = 0;
  }
  return filtered;
}

function renderResults() {
  const meta = state.meta;
  const threshold = Number(elements.threshold.value);
  elements.thresholdValue.value = threshold.toFixed(2);
  const mask = currentMask();

  // The two false-colour panels were rasterised in runWorkflow, while the
  // reflectance was still raw; only the model's own outputs change with the
  // threshold, so only these two are redrawn.
  renderToCanvas(elements.probabilityCanvas, i => state.keep && !state.keep[i]
    ? [86, 96, 92, 255]
    : [...probabilityColour(state.probability[i]), 255]);
  renderToCanvas(elements.maskCanvas, i => state.keep && !state.keep[i]
    ? [86, 96, 92, 255]
    : (mask[i] ? [255, 255, 255, 255] : [0, 0, 0, 255]));

  const pixelHa = (meta.scale_m ** 2) / 10000;
  let burned = 0;
  let obscured = 0;
  for (let i = 0; i < mask.length; i += 1) {
    if (state.keep && !state.keep[i]) obscured += 1;
    else if (mask[i]) burned += 1;
  }
  const observed = mask.length - obscured;

  const selected = state.selected;
  elements.preDateSummary.textContent = selected.pre_date;
  elements.preClearSummary.textContent =
    `cs_cdf ${Number(selected.pre_cs).toFixed(3)} · ${(100 * Number(selected.pre_nan)).toFixed(1)}% no-data `
    + `· ${selected.pre_candidates} candidate scene${selected.pre_candidates === 1 ? "" : "s"}`;
  elements.currentDateSummary.textContent = selected.post_date;
  elements.currentClearSummary.textContent =
    `cs_cdf ${Number(selected.post_cs).toFixed(3)} · ${(100 * Number(selected.post_nan)).toFixed(1)}% no-data `
    + `· ${selected.post_candidates} candidate scene${selected.post_candidates === 1 ? "" : "s"}`;
  elements.lagSummary.textContent = `${state.geometry.lagDays} days`;
  elements.engineSummary.textContent =
    `${state.inferenceEngine} · ${state.geometry.crsCode} · ${state.geometry.detailPixels} px`;
  elements.burnedSummary.textContent = `${(burned * pixelHa).toLocaleString(undefined, {
    maximumFractionDigits: 0,
  })} ha`;
  elements.pixelSummary.textContent = observed
    ? `${(100 * burned / observed).toFixed(1)}% of the observed window`
    : "nothing observed";
  elements.obscuredSummary.textContent = `${(obscured * pixelHa).toLocaleString(undefined, {
    maximumFractionDigits: 0,
  })} ha`;
  elements.obscuredDetail.textContent = state.keep
    ? `${(100 * obscured / mask.length).toFixed(1)}% below cs_cdf ${meta.cloud_valid_cs} in either phase — excluded, not called unburned`
    : "cloud masking is off for this checkpoint";
}

// ── downloads ────────────────────────────────────────────────────────────────

function downloadMask() {
  if (!state.probability) return;
  const mask = currentMask();
  const threshold = Number(elements.threshold.value);
  const canvas = document.createElement("canvas");
  renderToCanvas(canvas, i => state.keep && !state.keep[i]
    ? [86, 96, 92, 255]
    : (mask[i] ? [255, 255, 255, 255] : [0, 0, 0, 255]), true);
  canvas.toBlob(blob => {
    if (!blob) return;
    const link = document.createElement("a");
    link.href = URL.createObjectURL(blob);
    link.download = `flamenet_s2_mask_${state.selected.post_date}_t${threshold.toFixed(2)}.png`;
    link.click();
    setTimeout(() => URL.revokeObjectURL(link.href), 1000);
  }, "image/png");
}

/**
 * The imagery the model was run on, as Earth Engine returned it.
 *
 * A window wider than one request comes back as several chunks, so each is
 * offered separately, named by its pixel offset in the window. They share the
 * parent grid, so gdal_merge or a VRT reassembles them exactly.
 */
function downloadSourceStack() {
  const blobs = state.sourceBlobs;
  if (!blobs?.length) return;
  const stem = `s2_pair_${state.selected.pre_date}_${state.selected.post_date}`;
  blobs.forEach(({ blob, row, col }, index) => {
    // Browsers drop same-tick repeated downloads; space them out.
    setTimeout(() => {
      const link = document.createElement("a");
      link.href = URL.createObjectURL(blob);
      link.download = blobs.length === 1 ? `${stem}.tif` : `${stem}_r${row}_c${col}.tif`;
      link.click();
      setTimeout(() => URL.revokeObjectURL(link.href), 1000);
    }, index * 400);
  });
}

// ── wiring ───────────────────────────────────────────────────────────────────

elements.connectButton.addEventListener("click", connectEarthEngine);
elements.runButton.addEventListener("click", runWorkflow);
elements.scenePreset.addEventListener("change", event => applyScenePreset(event.target.value));
for (const field of [elements.latitude, elements.longitude]) {
  field.addEventListener("change", () => {
    markSceneCustom();
    updateMapSelection();
  });
}
for (const field of [elements.detailSize, elements.contextSize]) {
  field.addEventListener("input", () => {
    markSceneCustom();
    updateMapSelection(false);
  });
}
elements.preDate.addEventListener("change", () => markSceneCustom());
elements.targetDate.addEventListener("change", () => markSceneCustom());
elements.mapExpand.addEventListener("click", toggleMapWorkspace);
document.addEventListener("fullscreenchange", () => {
  setMapExpanded(document.fullscreenElement === elements.mapWorkspace);
});
// Re-rendering means re-running the component filter over up to a million
// pixels, so the slider updates its readout immediately and the maps catch up
// once the drag settles.
let thresholdTimer = null;
elements.threshold.addEventListener("input", () => {
  elements.thresholdValue.value = Number(elements.threshold.value).toFixed(2);
  if (!state.probability) return;
  clearTimeout(thresholdTimer);
  thresholdTimer = setTimeout(renderResults, 140);
});
elements.downloadMask.addEventListener("click", downloadMask);
elements.downloadStack.addEventListener("click", downloadSourceStack);

elements.projectId.value = localStorage.getItem("flamenet-ee-project") || elements.projectId.value;
elements.clientId.value = localStorage.getItem("flamenet-ee-client-id") || "";
initializeMap();

loadMeta()
  .then(() => {
    applyScenePreset(elements.scenePreset.value);
    updateMapSelection(false);
  })
  .catch(error => {
    setMessage(`Could not load the model metadata: ${normalizeError(error)}`, true);
    elements.runButton.disabled = true;
  });
