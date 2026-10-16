/*
 * FlameNet-S2 preprocessing and post-processing, with no DOM and no network.
 *
 * These are the functions the research pipeline implements with numpy, cv2 and
 * torch, re-implemented for the browser: the channel conversion from
 * src/datasets/s2_fire.py's _read_tif, the standardisation from _normalise,
 * train.py's _extract_patch_ctx, eval_miou.py's gaussian_weight and
 * remove_small_cc, and the sliding window from predict_fire.py.
 *
 * They live in their own module so they can be tested outside a browser.
 * scripts/verify_browser_pipeline.py in the FlameNet-S2 repository holds the
 * numpy reference each one was transcribed from, and test/pipeline.test.mjs
 * checks this module against vectors that reference generates. Change a
 * function here and that test is what tells you whether it still agrees.
 */

/**
 * One band of a computePixels response, converted the way _read_tif does.
 *
 * Reflectance DN are scaled and clipped; no-data is filled with the channel
 * mean, which is zero after standardisation. cs_cdf is filled with 0 instead:
 * the channel mean would read as "typically clear" for exactly the pixels the
 * export could not see.
 */
export function convertChannel(source, { fill, scale, clip, nodata }) {
  const out = new Float32Array(source.length);
  for (let i = 0; i < out.length; i += 1) {
    const raw = Number(source[i]);
    if (!Number.isFinite(raw) || raw <= nodata + 1) {
      out[i] = fill;
      continue;
    }
    const value = raw * scale;
    out[i] = clip === null ? value : Math.min(clip, Math.max(0, value));
  }
  return out;
}

/** _normalise: (x - mean) / (std + 1e-6), per channel. */
export function standardise(channels, mean, std) {
  return channels.map((source, channel) => {
    const target = new Float32Array(source.length);
    const m = mean[channel];
    const s = std[channel] + 1e-6;
    for (let i = 0; i < source.length; i += 1) target[i] = (source[i] - m) / s;
    return target;
  });
}

/** eval_miou.py's gaussian_weight: sigma = size / 4, centred on size / 2. */
export function gaussianWeight(size) {
  const weight = new Float32Array(size * size);
  const sigma = size / 4.0;
  for (let y = 0; y < size; y += 1) {
    const dy = y - size / 2.0;
    for (let x = 0; x < size; x += 1) {
      const dx = x - size / 2.0;
      weight[y * size + x] = Math.exp(-(dx * dx + dy * dy) / (2 * sigma * sigma));
    }
  }
  return weight;
}

/**
 * Window starts along one axis, with the last window flushed to the edge.
 *
 * Without that flush a length that is not commensurate with the stride leaves a
 * trailing strip covered by no window at all: its weight sum stays 0, so it
 * comes out as probability 0 and reads as confidently unburned rather than as
 * never looked at. On one 1771x1781 tile that silently discarded 12.2%.
 */
export function windowStarts(length, patch, stride) {
  const starts = [];
  for (let position = 0; position + patch <= length; position += stride) starts.push(position);
  if (!starts.length) starts.push(0);
  if (starts.at(-1) + patch < length) starts.push(length - patch);
  return starts;
}

/**
 * One patch of a standardised phase, replicating the edge where the window runs
 * past the array — the 'replicate' padding predict_fire.py applies.
 */
export function extractPatch(channels, width, height, row, col, patch) {
  const out = new Float32Array(channels.length * patch * patch);
  const copyH = Math.min(patch, height - row);
  const copyW = Math.min(patch, width - col);
  for (let c = 0; c < channels.length; c += 1) {
    const source = channels[c];
    const base = c * patch * patch;
    for (let y = 0; y < patch; y += 1) {
      const sourceRow = (row + Math.min(y, copyH - 1)) * width + col;
      const targetRow = base + y * patch;
      for (let x = 0; x < patch; x += 1) {
        out[targetRow + x] = source[sourceRow + Math.min(x, copyW - 1)];
      }
    }
  }
  return out;
}

/**
 * train.py's _extract_patch_ctx for a single patch.
 *
 * Bilinear, align_corners=True, padding_mode='border'. The sampling grid is an
 * axis-aligned lattice, so this is a separable resample of the context map over
 * the patch's footprint rather than a general GridSample — which is also why
 * neither exported graph needs that operator and its patchy browser support.
 *
 * `cy`, `cx` are the patch centre and `hy`, `hx` its half-extent, both in the
 * [-1, 1] normalised coordinates grid_sample uses, measured against the CONTEXT
 * window rather than the detail window inside it.
 */
export function sampleContext(ctx, channels, ctxH, ctxW, cy, cx, hy, hx, outSize) {
  const taps = (centre, half, n) => {
    const i0 = new Int32Array(outSize);
    const i1 = new Int32Array(outSize);
    const frac = new Float32Array(outSize);
    for (let k = 0; k < outSize; k += 1) {
      const lin = outSize === 1 ? -1 : -1 + 2 * k / (outSize - 1);
      const position = (centre + half * lin + 1) * 0.5 * (n - 1);
      const floor = Math.floor(position);
      frac[k] = position - floor;
      i0[k] = Math.min(n - 1, Math.max(0, floor));
      i1[k] = Math.min(n - 1, Math.max(0, floor + 1));
    }
    return { i0, i1, frac };
  };
  const rows = taps(cy, hy, ctxH);
  const cols = taps(cx, hx, ctxW);
  const out = new Float32Array(channels * outSize * outSize);
  const plane = ctxH * ctxW;
  for (let c = 0; c < channels; c += 1) {
    const base = c * plane;
    const target = c * outSize * outSize;
    for (let y = 0; y < outSize; y += 1) {
      const r0 = base + rows.i0[y] * ctxW;
      const r1 = base + rows.i1[y] * ctxW;
      const fy = rows.frac[y];
      for (let x = 0; x < outSize; x += 1) {
        const c0 = cols.i0[x];
        const c1 = cols.i1[x];
        const fx = cols.frac[x];
        const top = ctx[r0 + c0] * (1 - fx) + ctx[r0 + c1] * fx;
        const bottom = ctx[r1 + c0] * (1 - fx) + ctx[r1 + c1] * fx;
        out[target + y * outSize + x] = top * (1 - fy) + bottom * fy;
      }
    }
  }
  return out;
}

/**
 * eval_miou.py's remove_small_cc: drop 8-connected components below `minArea`
 * pixels, which removes speckle too small to be a fire — 50 px is 2 ha at 20 m.
 *
 * Union-find in one raster pass, since the browser has no
 * connectedComponentsWithStats.
 */
export function removeSmallComponents(mask, width, height, minArea) {
  if (minArea <= 0) return mask;
  const parent = new Int32Array(width * height).fill(-1);
  const find = x => {
    let root = x;
    while (parent[root] !== root) root = parent[root];
    while (parent[x] !== root) {
      const next = parent[x];
      parent[x] = root;
      x = next;
    }
    return root;
  };
  const neighbours = [[-1, -1], [-1, 0], [-1, 1], [0, -1]];
  for (let y = 0; y < height; y += 1) {
    for (let x = 0; x < width; x += 1) {
      const i = y * width + x;
      if (!mask[i]) continue;
      parent[i] = i;
      for (const [dy, dx] of neighbours) {
        const ny = y + dy;
        const nx = x + dx;
        if (ny < 0 || nx < 0 || nx >= width) continue;
        const j = ny * width + nx;
        if (!mask[j]) continue;
        const ra = find(i);
        const rb = find(j);
        if (ra !== rb) parent[ra] = rb;
      }
    }
  }
  const area = new Int32Array(width * height);
  for (let i = 0; i < mask.length; i += 1) if (mask[i]) area[find(i)] += 1;
  const out = new Uint8Array(mask.length);
  for (let i = 0; i < mask.length; i += 1) if (mask[i] && area[find(i)] >= minArea) out[i] = 1;
  return out;
}

/**
 * Blend the Gaussian-weighted patch outputs of one window into the accumulators.
 * Only the part of the window that lies inside the array contributes, matching
 * the `out[:h_v, :w_v] * gw[:h_v, :w_v]` slice in predict_fire.py.
 */
export function blendWindow(accumulated, weightSum, probability, weight,
                            width, height, row, col, patch) {
  const validH = Math.min(patch, height - row);
  const validW = Math.min(patch, width - col);
  for (let y = 0; y < validH; y += 1) {
    const sourceRow = y * patch;
    const targetRow = (row + y) * width + col;
    for (let x = 0; x < validW; x += 1) {
      accumulated[targetRow + x] += probability[sourceRow + x] * weight[sourceRow + x];
      weightSum[targetRow + x] += weight[sourceRow + x];
    }
  }
}

/** Finish the blend: divide by the accumulated weight, guarded as the reference does. */
export function normaliseBlend(accumulated, weightSum) {
  for (let i = 0; i < accumulated.length; i += 1) {
    accumulated[i] /= Math.max(weightSum[i], 1e-6);
  }
  return accumulated;
}

/**
 * The project's house false-colour stretch — visualize_all_s2_2025.py's
 * normalize_joint_rgb: ONE 2-98% ramp shared by B12/B8/B4, taken over valid
 * pixels only, with no gamma and no per-band normalisation.
 *
 * Per-band stretching would destroy the thing a false composite is for, since
 * the colour IS the ratio between the bands: water comes out saturated blue and
 * every scene looks equally colourful whatever is in it.
 */
export function jointRamp(channels, maxSamples = 300000) {
  const [b12, b8, b4] = channels;
  const values = [];
  const stride = Math.max(1, Math.floor(b12.length / maxSamples));
  for (let i = 0; i < b12.length; i += stride) {
    if (b12[i] > 0 || b8[i] > 0 || b4[i] > 0) values.push(b12[i], b8[i], b4[i]);
  }
  if (!values.length) return { low: 0, high: 1 };
  values.sort((a, b) => a - b);
  const at = q => values[Math.min(values.length - 1, Math.floor(q * (values.length - 1)))];
  const low = at(0.02);
  const high = at(0.98);
  return { low, high: high > low ? high : low + 1e-6 };
}

/** One pixel of the joint stretch. Invalid pixels are black, cloud is left in. */
export function falseColour(channels, ramp, index) {
  const [b12, b8, b4] = channels;
  if (!(b12[index] > 0 || b8[index] > 0 || b4[index] > 0)) return [0, 0, 0];
  const span = ramp.high - ramp.low;
  return [b12, b8, b4].map(band =>
    Math.round(255 * Math.min(1, Math.max(0, (band[index] - ramp.low) / span))));
}
