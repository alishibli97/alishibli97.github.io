# FlameNet-S2 on Earth Engine

A browser deployment of **FlameNet-S2**, a Sentinel-2 burned-area segmentation
model. Earth Engine supplies a pre/post Sentinel-2 pair, `onnxruntime-web` runs
the trained network on it, and the map is produced without any imagery or
result leaving the machine.

This portfolio deployment also preserves Ali Shibli's original US progression
model under `us/`. The header selector switches between the two independent
routes. They intentionally do not share inference code because each checkpoint
requires different imagery selection and preprocessing; they share only the
saved Earth Engine project and public OAuth client ID in browser storage.

The page design, the Earth Engine OAuth flow, the map and the canvas layout come
from Ali Shibli's wildfire tool, which this repository is a fork of; `style.css`
still imports that stylesheet from `../wildfire-demo/`. This directory swaps in
FlameNet-S2 and the preprocessing that model requires.

```
index.html      the page
app.js          Earth Engine queries, the inference loop, rendering
pipeline.js     the numeric core, with no DOM or network: testable in node
style.css       the framework's stylesheet plus this page's overrides
model/          the exported graphs and the preprocessing constants
test/           node tests against the research implementation
```

## What the model requires, and what this reproduces

FlameNet-S2 is only valid on its own preprocessing, so none of this is
approximated. The upstream framework assumed a different model, and these are
the things that had to change with it:

| | FlameNet-S2 |
|---|---|
| Collection | `COPERNICUS/S2_SR_HARMONIZED` — L2A surface reflectance, not L1C TOA |
| Bands | `B12, B8, B4, cs_cdf` per phase, in that order |
| `cs_cdf` | a model **input channel**, continuous, not a binary cloud mask |
| Reflectance | DN clamped to 0–5000, scaled by 1e-4, clipped to 0.5, then standardised with the 2017–2023 training means and stds |
| Grid | 20 m in the scene's local **UTM** zone — a 20 m Web Mercator pixel is 11 m on the ground at 56° N, which would show the model imagery at 1.8× its training resolution |
| Pre-fire scene | the **previous year's fire season**, which controls for phenology and avoids the smoke-filled active-fire period — not a lookback before ignition, and not the target date minus a year (see below) |
| Scene choice | highest mean `cs_cdf` minus the no-data fraction, the dataset loader's own rule |
| Granules | one overpass is one scene, mosaicked by `DATATAKE_IDENTIFIER` |
| Inference | 256 px patches, stride 128, Gaussian-blended, with the tile context encoder's output sampled at each patch's footprint, then components under 50 px dropped |
| Cloud | pixels either phase could not see through (`cs_cdf < 0.65`) are reported as obscured and excluded, never scored as unburned |

Every one of those constants is read at page load from
`model/model_meta.json`, which the export script writes directly from the
training code. Nothing about the model is duplicated here as a literal, so the
page cannot drift away from the research pipeline without the export noticing.

### The pre-fire date

The archive builds its pre-fire candidate pool from the fire's own date window
shifted back exactly one year (`gee/s1s2.py:356`):

```python
s2_dict["pre"] = MSI.filterDate(period_start.advance(-1, "year"),
                                period_end.advance(-1, "year"))
```

The page takes that date from you rather than deriving it, because the fire
window is not something the page knows. **Target date minus one year is not a
substitute**: the post scene is whatever was clearest in the three months after
the fire stopped, so the gap between the two is routinely 14 months and can be
21. Three real examples from the dataset:

| tile | pre | post | gap |
|---|---|---|---|
| `CA_2025_ON_594` | 2024-06-10 | 2025-08-31 | 14.7 months |
| `CA_2025_SK_433` | 2024-09-15 | 2025-11-04 | 13.7 months |
| `CA_2025_BC_524` | 2024-08-03 | 2026-05-20 | 21 months |

Every preset carries the pre date the dataset loader itself selected for that
fire, so the presets run on exactly the pair the reported metrics came from.

### One window, because the model had one in training

The orange box is both things at once: the extent segmented at 20 m in
overlapping 256 px patches, **and** the extent the tile context encoder sees,
compressed to 256 px and injected through FiLM. That is not a simplification —
it is what training did. A patch grid tiled a whole fire tile and the context
thumbnail was that same tile, so the segmented area and the context area were
always identical.

An earlier version of this page exposed them as two sliders. Don't: a context
wider than the window is not "more context", it is a different distribution, and
the setting silently swings the answer. Measured on one 71.68 km run over
`CA_2025_SK_433`:

| context | burned | outer 30–36 km ring | mask IoU vs context = window |
|---|---|---|---|
| = window (71.68 km) | 234,774 ha | 36.1% burned | — |
| 100 km | 257,344 ha | 38.9% burned | 0.892 |
| 45 km (narrower than the window) | 144,011 ha | 8.9% burned | 0.589 |

The 45 km row is the instructive failure: patches near the window's edge fall
outside the context, so `sampleContext` clamps them to the context map's border
and the periphery collapses. That is what a truncated-looking result means.

**Small windows, not large ones, are the off-distribution case.**
`patch_size / window_size` is the scale the context gets sampled at; training
tiles were 1771–2698 px with 128–512 px sampling windows, which puts it between
0.047 and 0.289. So:

| window | context scale | |
|---|---|---|
| 81.92 km (4096 px) | 0.063 | in range |
| 56.32 km (2816 px) | 0.091 | in range, and near the training centre |
| 17.92 km (896 px) | 0.286 | at the edge of the range |
| 5.12 km (256 px) | 1.000 | far outside — don't |

That is why the slider starts at 17.92 km, and why the readout under it names the
context scale and says when the window has left the trained range.

A single `computePixels` response is capped at 32 MiB, about 1024 px square with
eight `int16` bands, so a wider window is fetched as a grid of requests and
reassembled. That reassembly is byte-identical to the single request it replaces,
checked directly against the API.

Two things scale badly with the window, and both are handled rather than hoped
about:

- **Scene ranking.** Every candidate scene costs two `reduceRegion` calls, and a
  wide window can have forty of them in flight; at a fixed fine scale Earth
  Engine answers *"Too many concurrent aggregations"* and the run dies before any
  imagery is fetched. The ranking scale therefore grows with the window, keeping
  each reduction near 150×150 cells. It costs nothing real — over a 72 km window
  the mean `cs_cdf` moves by 0.002 between 480 m and 1920 m, and these numbers
  only rank scenes.
- **Memory.** At 4096 px the eight `float32` planes are 537 MB, so the
  reflectance is standardised *in place* and the false-colour panels are
  rasterised to their canvases before that happens.

For reference, the largest preset (71.68 km, 3584 px) is 16 Earth Engine
requests totalling ~140 MB, 729 model windows and ~525 MB resident. That is tens
of seconds on WebGPU and minutes on WebAssembly.

### A caution about the numbers at this extent

On `CA_2025_SK_433` with the window matching the fire, this page reports
**234,774 ha against an NBAC reference of 170,576 ha** — a 38% over-prediction.
The predictions do reach the window edges, so the window is not the limit. Three
things plausibly contribute and none has been separated yet: the page's ±20-day
clarity-first search picked a pre-fire scene (2024-08-26) that is not the one the
archive used (2024-09-15); that scene carries cloud over part of a window this
wide; and a 72 km box around one fire contains neighbouring fires that are not in
that tile's mask — on the K70748 validation fire, attributing neighbours moved
precision from 0.829 to 0.961. Treat the hectare figure as a demonstration, not
as a measurement.

### The one approximation

Earth Engine averages the 20 m pixels into the 256 px context grid with
`reduceResolution`, which ignores masked pixels, whereas the loader fills them
with the channel mean before averaging. 98% of selected scenes have no masked
pixels at all, and the thumbnail only feeds the FiLM modulation. Everything else
is checked, below.

## Handing this to someone else

Everything needed to **run** the page is in this repository. There is no server
and no build step: it is static files, and the model is 2.4 MB of ONNX sitting
next to them.

What the recipient must bring, because the page deliberately holds no
credentials of its own:

- **Their own Earth Engine project** — a Google Cloud project with the Earth
  Engine API enabled and registered for Earth Engine. The project field is
  blank on purpose; a pre-filled project ID that is not theirs just produces a
  permission error they cannot read.
- **Their own OAuth Web client**, created in that project, with the origin they
  serve this from in its Authorized JavaScript origins. A client ID is bound to
  an origin, so it cannot be shipped with the code.
- If their consent screen is in **Testing** — it will be — the account they sign
  in with has to be listed under *Audience → Test users*, or Google answers
  "has not completed the Google verification process". The page's setup panel
  spells this out.

Earth Engine bills the pixel computation to whichever project they name. The
inference is on their own machine, so it costs nothing and sends nothing.

### What is NOT in here

- **The checkpoint and the training code.** `model/` holds the *exported* graphs,
  which is all the page needs. Re-exporting them needs the FlameNet-S2
  repository, `runs/mode_early_cloud/best.pth`, and `scripts/export_onnx.py` —
  see *Rebuilding `model/`* below. Hand those over only if the recipient is
  meant to retrain or re-export.
- **The end-to-end test fixture**, ~7 MB generated from the K70748 GeoTIFFs. So
  `npm test` is **17 of 18** in a fresh clone, with `endtoend.test.mjs` skipping
  itself. The other 17 are self-contained and should pass immediately.

### Network dependencies

The page is static but not offline. It loads from four external hosts:

| host | what |
|---|---|
| `cdn.jsdelivr.net` | Leaflet, geotiff.js, onnxruntime-web and its WASM binaries |
| `ajax.googleapis.com` | the Earth Engine JavaScript client |
| `earthengine.googleapis.com` | the imagery itself |
| `{a,b,c}.tile.openstreetmap.org` | basemap tiles |

Two consequences worth stating before you hand it over. **`ajax.googleapis.com`
and Earth Engine are not reachable from mainland China without a VPN**, so a
recipient there cannot use this page as-is. And the library versions are pinned
but not vendored, so the page depends on jsDelivr staying up. Vendoring all four
into `vendor/` would remove both problems at the cost of a few MB in the
repository; it has not been done.

## Tests

```sh
npm install          # onnxruntime-web, for the end-to-end test only
npm test
```

- **`test/pipeline.test.mjs`** checks each hand-transcribed function in
  `pipeline.js` against the real numpy / cv2 / torch implementation. The
  expected answers in `test/vectors.json` are generated by
  `scripts/make_pipeline_vectors.py` in the FlameNet-S2 repository, from
  `_read_tif`, `_normalise`, `_extract_patch_ctx`, `gaussian_weight`,
  `remove_small_cc` and `predict_fire.py`'s sliding window.
- **`test/wiring.test.mjs`** checks that `app.js` and `index.html` still agree
  about every element id, and that no control of the upstream page survives in
  half-edited form. A renamed id otherwise fails at run time, in a handler,
  possibly only on an error path.
- **`test/endtoend.test.mjs`** runs the deployment's own window loop over the
  real ONNX graphs under `onnxruntime-web` and compares the burned area to what
  `scripts/predict_fire.py` reports for the same tile. It needs a fixture of a
  few megabytes, which is not committed, so it **skips itself** unless you
  generate one:

  ```sh
  # in the FlameNet-S2 repository
  python scripts/verify_browser_pipeline.py \
      --pre  data/K70748_2026/pre/CA_2026_BC_K70748_HS_20250822_0.tif \
      --post data/K70748_2026/post/CA_2026_BC_K70748_HS_20260916_0.tif \
      --crop 384 --emit-js-fixture <this dir>/test/fixture
  ```

  On the K70748 validation fire it reports 3,928.9 ha against the research
  pipeline's 3,928.3 ha — 15 pixels of 98,208, all of them sitting within 1e-4
  of the probability cutoff, where WebAssembly and torch do not round alike.

## Rebuilding `model/`

From the FlameNet-S2 repository:

```sh
python scripts/export_onnx.py \
    --ckpt runs/mode_early_cloud/best.pth --in-ch 4 --out deploy/model \
    --verify-pre  data/K70748_2026/pre/CA_2026_BC_K70748_HS_20250822_0.tif \
    --verify-post data/K70748_2026/post/CA_2026_BC_K70748_HS_20260916_0.tif
cp deploy/model/* <this dir>/model/
```

The export refuses to write graphs that do not reproduce the checkpoint, and
rewrites `model_meta.json` from the training code, so the page picks up any
change to the preprocessing automatically. After re-exporting, regenerate
`test/vectors.json` and re-run `npm test`.

The model is two graphs rather than one because the tile context is computed
once per scene and reused by every patch:

```
flamenet_s2_ctx.onnx     ctx_in (1, 8, 256, 256)  -> tile_ctx    (1, 32, 64, 64)
                                                     coarse_fire (1,  1, 64, 64)
flamenet_s2_patch.onnx   pre     (1, 4, 256, 256)
                         post    (1, 4, 256, 256)
                         ctx_btn (1, 32, 16, 16)  -> probability (1,  1, 256, 256)
```

The context sampling between them stays in JavaScript, so neither graph needs
`GridSample` — the one operator whose `onnxruntime-web` coverage cannot be
relied on. What the graphs do use is ordinary: `Conv`, `Clip`, `Sigmoid`,
`PRelu`, `InstanceNormalization`, `Resize`, `AveragePool`, `GlobalAveragePool`.

## Limits

This is a research demonstrator, not an emergency product. It segments burning
already visible in the target image; it does not predict ignition or spread. It
was trained on Canada 2017–2023, validated on 2024 and tested on 2025. Cloud,
smoke, snow, misregistration, unfamiliar ecosystems and imagery outside those
years can all cause errors. The Kalimantan preset is deliberately out of domain:
that result is corroborated by dNBR, but no burned-area reference exists for it.
