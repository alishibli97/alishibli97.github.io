/*
 * Check pipeline.js against the research implementation it was transcribed from.
 *
 * vectors.json is written by scripts/make_pipeline_vectors.py in the FlameNet-S2
 * repository, where each expected answer is computed by the real numpy, cv2 or
 * torch code — _read_tif, _normalise, _extract_patch_ctx, gaussian_weight,
 * remove_small_cc and predict_fire.py's sliding window. If one of those changes,
 * regenerate the vectors and this test says whether the browser still follows.
 *
 *   node --test test/
 */
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { test } from "node:test";

import {
  blendWindow, convertChannel, extractPatch, gaussianWeight, jointRamp,
  normaliseBlend, removeSmallComponents, sampleContext, standardise,
  windowStarts,
} from "../pipeline.js";

const { cases } = JSON.parse(readFileSync(new URL("./vectors.json", import.meta.url)));

// float32 round-trips through JSON as a double, so the only error to allow for
// is the one Math.exp and the bilinear arithmetic introduce.
const TOL = 1e-6;

function closeTo(got, expected, message, tol = TOL) {
  assert.equal(got.length, expected.length, `${message}: length`);
  for (let i = 0; i < expected.length; i += 1) {
    const delta = Math.abs(got[i] - expected[i]);
    assert.ok(delta <= tol,
      `${message}: index ${i} is ${got[i]}, expected ${expected[i]} (Δ ${delta})`);
  }
}

test("convertChannel reproduces _read_tif's DN conversion and no-data fill", () => {
  for (const [name, spec] of Object.entries(cases.convertChannel)) {
    const got = convertChannel(spec.source, {
      fill: spec.fill, scale: spec.scale, clip: spec.clip, nodata: spec.nodata,
    });
    closeTo(got, spec.expected, `convertChannel/${name}`);
  }
});

test("standardise reproduces _normalise", () => {
  const spec = cases.standardise;
  const got = standardise(spec.channels.map(c => Float32Array.from(c)), spec.mean, spec.std);
  assert.equal(got.length, spec.expected.length);
  got.forEach((channel, i) => closeTo(channel, spec.expected[i], `standardise/ch${i}`, 1e-4));
});

test("gaussianWeight reproduces eval_miou.gaussian_weight", () => {
  closeTo(gaussianWeight(cases.gaussianWeight.small.size),
    cases.gaussianWeight.small.expected, "gaussianWeight(16)");

  const patch = cases.gaussianWeight.patch;
  const weight = gaussianWeight(patch.size);
  const sum = weight.reduce((a, b) => a + b, 0);
  // Relative, because the sum is ~1.3e4 and float32 accumulation order differs.
  assert.ok(Math.abs(sum - patch.sum) / patch.sum < 1e-5,
    `gaussianWeight(256) sum is ${sum}, expected ${patch.sum}`);
  assert.ok(Math.abs(Math.max(...weight) - patch.max) < TOL, "gaussianWeight(256) max");
  assert.equal(weight.indexOf(Math.max(...weight)), patch.argmax,
    "gaussianWeight(256) peak is in the wrong place");
});

test("windowStarts flushes the last window to the edge", () => {
  for (const spec of cases.windowStarts) {
    assert.deepEqual(windowStarts(spec.length, spec.patch, spec.stride), spec.expected,
      `windowStarts(${spec.length}, ${spec.patch}, ${spec.stride})`);
  }
  // The property the edge flush exists for: every pixel is covered by a window.
  for (const length of [300, 1025, 1771, 2698, 4097]) {
    const starts = windowStarts(length, 256, 128);
    assert.equal(starts.at(-1) + 256, length,
      `windowStarts(${length}) leaves a trailing strip uncovered`);
  }
});

test("extractPatch replicates the edge like predict_fire's pad", () => {
  const spec = cases.extractPatch;
  const channels = spec.channels.map(c => Float32Array.from(c));
  for (const window of spec.windows) {
    const got = extractPatch(channels, spec.width, spec.height,
      window.row, window.col, spec.patch);
    closeTo(got, window.expected, `extractPatch(${window.row},${window.col})`);
  }
});

test("sampleContext reproduces train._extract_patch_ctx", () => {
  const spec = cases.sampleContext;
  const ctx = Float32Array.from(spec.ctx);
  for (const sample of spec.samples) {
    const got = sampleContext(ctx, spec.channels, spec.ctxH, spec.ctxW,
      sample.cy, sample.cx, sample.hy, sample.hx, spec.outSize);
    closeTo(got, sample.expected,
      `sampleContext(cy=${sample.cy.toFixed(3)}, hy=${sample.hy.toFixed(3)})`, 1e-5);
  }
});

test("removeSmallComponents reproduces remove_small_cc", () => {
  for (const spec of cases.removeSmallComponents) {
    const got = removeSmallComponents(
      Uint8Array.from(spec.mask), spec.width, spec.height, spec.minArea);
    assert.deepEqual([...got], spec.expected,
      `removeSmallComponents(minArea=${spec.minArea})`);
  }
  // minArea 0 is the documented no-op.
  const mask = Uint8Array.from([1, 0, 1]);
  assert.equal(removeSmallComponents(mask, 3, 1, 0), mask);
});

test("blendWindow and normaliseBlend reproduce the Gaussian blend", () => {
  const spec = cases.blendWindow;
  const accumulated = new Float32Array(spec.width * spec.height);
  const weightSum = new Float32Array(spec.width * spec.height);
  const weight = Float32Array.from(spec.weight);
  for (const window of spec.windows) {
    blendWindow(accumulated, weightSum, Float32Array.from(window.probability),
      weight, spec.width, spec.height, window.row, window.col, spec.patch);
  }
  closeTo(normaliseBlend(accumulated, weightSum), spec.expected, "blend", 1e-5);
});

test("jointRamp reproduces the house 2-98% stretch", () => {
  const spec = cases.jointRamp;
  const got = jointRamp(spec.channels.map(c => Float32Array.from(c)));
  assert.ok(Math.abs(got.low - spec.expected.low) < TOL,
    `jointRamp low is ${got.low}, expected ${spec.expected.low}`);
  assert.ok(Math.abs(got.high - spec.expected.high) < TOL,
    `jointRamp high is ${got.high}, expected ${spec.expected.high}`);
});
