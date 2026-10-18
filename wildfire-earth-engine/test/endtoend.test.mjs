/*
 * End-to-end check: the deployment's window loop, over the real ONNX graphs,
 * against the burned area scripts/predict_fire.py reports for the same tile.
 *
 * pipeline.test.mjs checks each transcribed function on its own. This checks
 * the thing that composes them — the loop in app.js's runInference: patch
 * extraction, the context sampling at each patch's footprint, the Gaussian
 * blend, the threshold, and the small-component filter, with onnxruntime
 * actually executing the exported graphs.
 *
 * It needs a fixture that is tens of megabytes, so the fixture is not
 * committed and this test skips itself when it is missing. To produce it, in
 * the FlameNet-S2 repository:
 *
 *   python scripts/verify_browser_pipeline.py \
 *       --pre  data/K70748_2026/pre/CA_2026_BC_K70748_HS_20250822_0.tif \
 *       --post data/K70748_2026/post/CA_2026_BC_K70748_HS_20260916_0.tif \
 *       --crop 384 --emit-js-fixture <this directory>/fixture
 *
 *   node --test test/endtoend.test.mjs
 */
import assert from "node:assert/strict";
import { existsSync, readFileSync } from "node:fs";
import path from "node:path";
import { test } from "node:test";
import { fileURLToPath } from "node:url";

import {
  blendWindow, extractPatch, gaussianWeight, normaliseBlend,
  removeSmallComponents, sampleContext, standardise, windowStarts,
} from "../ca/pipeline.js";

const here = path.dirname(fileURLToPath(import.meta.url));
const fixtureDir = path.join(here, "fixture");
const haveFixture = existsSync(path.join(fixtureDir, "fixture.bin"))
  && existsSync(path.join(fixtureDir, "fixture.json"));

test("the window loop reproduces predict_fire.py's burned area",
  { skip: haveFixture ? false : "no fixture; see the header of this file" },
  async () => {
    const ort = (await import("onnxruntime-web")).default;
    ort.env.wasm.numThreads = 1;
    ort.env.wasm.wasmPaths = path.join(here, "..", "node_modules", "onnxruntime-web", "dist") + path.sep;

    const meta = JSON.parse(readFileSync(path.join(here, "..", "ca", "model", "model_meta.json")));
    const fixture = JSON.parse(readFileSync(path.join(fixtureDir, "fixture.json")));
    const { width, height, channels, ctx_size: ctxSize } = fixture;

    // Four blocks of float32, in fixture.layout order: the two detail phases
    // then the two context thumbnails.
    const raw = readFileSync(path.join(fixtureDir, "fixture.bin"));
    const floats = new Float32Array(raw.buffer, raw.byteOffset, raw.byteLength / 4);
    let cursor = 0;
    const take = (count, length) => {
      const out = [];
      for (let c = 0; c < count; c += 1) {
        out.push(floats.subarray(cursor, cursor + length));
        cursor += length;
      }
      return out;
    };
    const detailPlane = width * height;
    const ctxPlane = ctxSize * ctxSize;
    const pre = take(channels, detailPlane);
    const post = take(channels, detailPlane);
    const ctxPre = take(channels, ctxPlane);
    const ctxPost = take(channels, ctxPlane);
    assert.equal(cursor, floats.length, "fixture.bin is not the size fixture.json describes");

    const patch = meta.patch_size;
    const session = provider => ({ executionProviders: [provider] });
    const ctxSession = await ort.InferenceSession.create(
      path.join(here, "..", "ca", "model", meta.graphs.ctx), session("wasm"));
    const patchSession = await ort.InferenceSession.create(
      path.join(here, "..", "ca", "model", meta.graphs.patch), session("wasm"));

    // ── exactly what app.js's runInference does ──────────────────────────────
    const norm = planes => standardise(planes, meta.mean, meta.std);
    const preNorm = norm(pre);
    const postNorm = norm(post);

    const ctxInput = new Float32Array(2 * channels * ctxPlane);
    [...norm(ctxPre), ...norm(ctxPost)].forEach((plane, i) => ctxInput.set(plane, i * ctxPlane));
    const ctxOutput = await ctxSession.run({
      ctx_in: new ort.Tensor("float32", ctxInput, [1, 2 * channels, ctxSize, ctxSize]),
    });
    const tileCtx = ctxOutput.tile_ctx.data;
    const [, ctxChannels, ctxH, ctxW] = ctxOutput.tile_ctx.dims;

    const weight = gaussianWeight(patch);
    const accumulated = new Float32Array(detailPlane);
    const weightSum = new Float32Array(detailPlane);
    const rows = windowStarts(height, patch, meta.stride);
    const cols = windowStarts(width, patch, meta.stride);

    for (const row of rows) {
      for (const col of cols) {
        const centreRow = fixture.detailOriginRow + row + patch / 2;
        const centreCol = fixture.detailOriginCol + col + patch / 2;
        const ctxBtn = sampleContext(
          tileCtx, ctxChannels, ctxH, ctxW,
          (centreRow / fixture.contextPixels) * 2 - 1,
          (centreCol / fixture.contextPixels) * 2 - 1,
          patch / fixture.contextPixels,
          patch / fixture.contextPixels,
          meta.btn_size);
        const output = await patchSession.run({
          pre: new ort.Tensor("float32",
            extractPatch(preNorm, width, height, row, col, patch),
            [1, channels, patch, patch]),
          post: new ort.Tensor("float32",
            extractPatch(postNorm, width, height, row, col, patch),
            [1, channels, patch, patch]),
          ctx_btn: new ort.Tensor("float32", ctxBtn,
            [1, ctxChannels, meta.btn_size, meta.btn_size]),
        });
        blendWindow(accumulated, weightSum, output.probability.data, weight,
          width, height, row, col, patch);
      }
    }
    const probability = normaliseBlend(accumulated, weightSum);

    const binary = new Uint8Array(detailPlane);
    for (let i = 0; i < binary.length; i += 1) {
      binary[i] = probability[i] >= meta.threshold ? 1 : 0;
    }
    const mask = removeSmallComponents(binary, width, height, meta.min_cc_area);

    // Cloud exclusion, as in app.js: a pixel counts only if BOTH phases saw it.
    let burned = 0;
    for (let i = 0; i < mask.length; i += 1) {
      if (!mask[i]) continue;
      if (meta.cloud_valid_cs > 0 && channels > 3
        && !(pre[3][i] >= meta.cloud_valid_cs && post[3][i] >= meta.cloud_valid_cs)) continue;
      burned += 1;
    }

    // ── compare ──────────────────────────────────────────────────────────────
    const expected = fixture.expected;
    const sum = probability.reduce((a, b) => a + b, 0);
    const pixelHa = (meta.scale_m ** 2) / 10000;

    // The probability map is the sensitive quantity: a wrong patch position or a
    // mis-indexed blend moves it long before it moves the thresholded area.
    const sumError = Math.abs(sum - expected.probability_sum) / expected.probability_sum;
    assert.ok(sumError < 1e-4,
      `probability sum is ${sum.toFixed(1)}, expected ${expected.probability_sum.toFixed(1)} `
      + `(relative ${sumError.toExponential(2)})`);
    assert.ok(Math.abs(probability.reduce((a, b) => Math.max(a, b), 0) - expected.probability_max) < 1e-3,
      "probability max disagrees");

    // The pixel count is the blunter check: wasm and torch agree on the
    // probability only to ~1e-4, so any pixel sitting that close to the cutoff
    // can land on either side, and the number of such pixels scales with the
    // scar's perimeter. 0.1% is far tighter than any real transcription error
    // — a mis-sampled context or an off-by-one window moves the area by whole
    // percent — while leaving room for that boundary noise.
    const pixelDelta = Math.abs(burned - expected.burned_pixels);
    assert.ok(pixelDelta <= Math.max(32, expected.burned_pixels * 1e-3),
      `burned ${burned} px (${(burned * pixelHa).toFixed(1)} ha), expected `
      + `${expected.burned_pixels} px (${expected.burned_ha.toFixed(1)} ha), Δ ${pixelDelta} px`);

    console.log(`      burned ${(burned * pixelHa).toFixed(1)} ha vs reference `
      + `${expected.burned_ha.toFixed(1)} ha (Δ ${pixelDelta} px of ${expected.burned_pixels})`);
  });
