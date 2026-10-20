/*
 * Check that app.js and index.html still agree about the page.
 *
 * app.js looks every control up by id once, at module load, into `elements`.
 * A renamed or dropped id therefore fails at the first use with
 * "Cannot read properties of null" — at run time, in a handler, possibly only
 * on the error path. This test catches that statically instead:
 *
 *   - every id app.js asks for exists in index.html
 *   - every `elements.foo` it dereferences is one of the ids it asked for
 *   - both modules parse
 *
 *   node --test test/wiring.test.mjs
 */
import assert from "node:assert/strict";
import { execFileSync } from "node:child_process";
import { readFileSync } from "node:fs";
import path from "node:path";
import { test } from "node:test";
import { fileURLToPath } from "node:url";

const root = path.join(path.dirname(fileURLToPath(import.meta.url)), "..");
const read = name => readFileSync(path.join(root, "ca", name), "utf8");
const appSource = read("app.js");
const htmlSource = read("index.html");

const camel = id => id.replace(/-([a-z])/g, (_, c) => c.toUpperCase());

/** The id list app.js feeds into its querySelector loop. */
function requestedIds() {
  const block = appSource.match(/const elements = \{\};\s*for \(const id of \[([\s\S]*?)\]\)/);
  assert.ok(block, "could not find app.js's element id list");
  return [...block[1].matchAll(/"([a-z0-9-]+)"/g)].map(m => m[1]);
}

test("app.js and pipeline.js parse", () => {
  for (const file of ["app.js", "pipeline.js"]) {
    // --check needs to know these are modules; the package.json sets type,
    // but be explicit so the test does not depend on that.
    execFileSync(process.execPath, ["--input-type=module", "--check", "-"], {
      input: read(file), stdio: ["pipe", "ignore", "pipe"],
    });
  }
});

test("every id app.js requests exists in index.html", () => {
  const ids = requestedIds();
  assert.ok(ids.length > 20, `only ${ids.length} ids found; the regex probably broke`);
  const missing = ids.filter(id => !new RegExp(`id="${id}"`).test(htmlSource));
  assert.deepEqual(missing, [], `index.html has no element with these ids: ${missing}`);
});

test("every elements.foo app.js uses was requested", () => {
  const available = new Set(requestedIds().map(camel));
  const used = new Set([...appSource.matchAll(/\belements\.([A-Za-z0-9_]+)\b/g)].map(m => m[1]));
  const unknown = [...used].filter(name => !available.has(name));
  assert.deepEqual(unknown, [],
    `app.js dereferences elements that are never looked up: ${unknown}`);
});

test("the ids app.js requests are not stale leftovers of the upstream page", () => {
  // These are the upstream framework's controls that FlameNet-S2 replaced. The
  // model does not take an elapsed-days input and does not use a pre-fire
  // lookback, so a reference to one of these means an edit was left half done.
  const retired = ["roi-size", "roi-value", "ignition-date", "elapsed-summary", "elapsed_days"];
  for (const id of retired) {
    assert.ok(!appSource.includes(id), `app.js still references the retired "${id}"`);
    assert.ok(!htmlSource.includes(id), `index.html still contains the retired "${id}"`);
  }
});

/** SCENE_PRESETS, evaluated out of app.js's source.
 *
 * app.js cannot be imported here — it reads the DOM at module scope — but the
 * preset table is a plain object literal, so lifting it out and evaluating it
 * is safe and keeps the test honest about the real text.
 */
function scenePresets() {
  const start = appSource.indexOf("const SCENE_PRESETS = Object.freeze({");
  assert.ok(start > 0, "could not find SCENE_PRESETS in app.js");
  const end = appSource.indexOf("\n});\n", start);
  // end + 2 stops just past the closing brace, dropping Object.freeze's ")".
  const literal = appSource.slice(appSource.indexOf("{", start), end + 2);
  return new Function(`return ${literal}`)();
}

test("the preset table the dropdown is built from is well formed", () => {
  const presets = scenePresets();
  const entries = Object.entries(presets);
  // The page is meant to offer a spread of fire sizes, not a handful.
  assert.ok(entries.length >= 30, `only ${entries.length} presets`);

  const STEP = 0.16, MIN = 0, MAX = 100;
  for (const [id, p] of entries) {
    for (const key of ["label", "group", "note", "pre", "target"]) {
      assert.equal(typeof p[key], "string", `${id}.${key} is not a string`);
      assert.ok(p[key].length, `${id}.${key} is empty`);
    }
    assert.ok(Number.isFinite(p.latitude) && Math.abs(p.latitude) <= 90, `${id} latitude`);
    assert.ok(Number.isFinite(p.longitude) && Math.abs(p.longitude) <= 180, `${id} longitude`);
    // The window must be a value the slider can actually hold, or selecting
    // the preset would snap it and quietly segment a different area.
    assert.ok(p.detailKm >= MIN && p.detailKm <= MAX, `${id} detailKm ${p.detailKm} out of range`);
    assert.ok(Math.abs((p.detailKm - MIN) / STEP - Math.round((p.detailKm - MIN) / STEP)) < 1e-9,
      `${id} detailKm ${p.detailKm} is not on the slider's ${STEP} km step`);
    for (const key of ["pre", "target"]) {
      assert.match(p[key], /^\d{4}-\d{2}-\d{2}$/, `${id}.${key} is not an ISO date`);
    }
    assert.ok(p.pre < p.target, `${id} pre ${p.pre} does not precede target ${p.target}`);
  }

  // A during-fire preset carries its whole acquisition series. The target has
  // to be one of them, or the acquisition select would open on a date it does
  // not list and the two controls would disagree about what is being run.
  let withSeries = 0;
  for (const [id, p] of entries) {
    if (!p.dates) {
      assert.equal(p.observed, undefined, `${id} has observed without dates`);
      continue;
    }
    withSeries++;
    assert.ok(p.dates.length > 1, `${id} has a series of ${p.dates.length}`);
    for (const date of p.dates) {
      assert.match(date, /^\d{4}-\d{2}-\d{2}$/, `${id} has a bad date ${date}`);
    }
    assert.deepEqual([...p.dates].sort(), p.dates, `${id}'s dates are not in order`);
    assert.equal(new Set(p.dates).size, p.dates.length, `${id} repeats a date`);
    assert.ok(p.dates.includes(p.target),
      `${id}.target ${p.target} is not one of its ${p.dates.length} acquisitions`);
    assert.equal(p.observed.length, p.dates.length,
      `${id} has ${p.observed.length} observed fractions for ${p.dates.length} dates`);
    for (const seen of p.observed) {
      assert.ok(seen >= 0 && seen <= 1, `${id} has an observed fraction of ${seen}`);
    }
  }
  assert.ok(withSeries >= 15, `only ${withSeries} presets carry a series`);

  // A preset may narrow the date searches, but only to a value those selects
  // actually offer: assigning an absent value to a <select> is a silent no-op
  // in the DOM, so the previous preset's radius would quietly carry over.
  for (const [field, key] of [["pre-window", "preWindow"],
                              ["target-window", "targetWindow"]]) {
    const select = htmlSource.match(
      new RegExp(`<select id="${field}">([\\s\\S]*?)</select>`));
    assert.ok(select, `index.html has no ${field} select`);
    const offered = new Set([...select[1].matchAll(/value="(\d+)"/g)].map(m => Number(m[1])));
    for (const [id, p] of entries) {
      if (p[key] === undefined) continue;
      assert.ok(offered.has(p[key]),
        `${id}.${key} is ${p[key]}, which ${field} does not offer (${[...offered]})`);
    }
  }

  // Groups are rendered as <optgroup> in iteration order, so a group's entries
  // have to be contiguous or it would appear twice in the dropdown.
  const seen = new Set();
  let last = null;
  for (const [, p] of entries) {
    if (p.group !== last) {
      assert.ok(!seen.has(p.group), `group "${p.group}" is split across the table`);
      seen.add(p.group);
      last = p.group;
    }
  }

  const fallback = appSource.match(/const DEFAULT_PRESET = "([^"]+)"/);
  assert.ok(fallback, "no DEFAULT_PRESET in app.js");
  assert.ok(fallback[1] in presets, `DEFAULT_PRESET ${fallback[1]} is not a preset`);
});

test("the preset dropdown is populated from app.js, not hand-written in the HTML", () => {
  // A hand-maintained <option> list would drift from SCENE_PRESETS and offer
  // values applyScenePreset silently ignores.
  const select = htmlSource.match(/<select id="scene-preset"[^>]*>([\s\S]*?)<\/select>/);
  assert.ok(select, "index.html has no scene-preset select");
  assert.equal(select[1].trim(), "", "scene-preset still carries static options");
  assert.match(appSource, /populateScenePresets\(\);/);
});

test("the three local assets carry one and the same cache-busting version", () => {
  // GitHub Pages serves these with max-age=600, so without ?v= a browser can
  // hold a stale app.js next to a fresh index.html — which is exactly how a
  // deploy's new presets fail to appear. A fresh app.js beside a cached
  // pipeline.js would be worse still, so all three move together.
  const found = {
    "style.css": htmlSource.match(/href="\.\/style\.css\?v=(\d+)"/),
    "app.js": htmlSource.match(/src="\.\/app\.js\?v=(\d+)"/),
    "pipeline.js": appSource.match(/from "\.\/pipeline\.js\?v=(\d+)"/),
  };
  for (const [name, m] of Object.entries(found)) {
    assert.ok(m, `${name} is loaded without a ?v= cache buster`);
  }
  const versions = Object.values(found).map(m => m[1]);
  assert.equal(new Set(versions).size, 1,
    `the ?v= versions disagree: ${JSON.stringify(
      Object.fromEntries(Object.entries(found).map(([k, m]) => [k, m[1]])))}`);
});

test("the page loads the model from its own directory, not the other demo's", () => {
  // ../wildfire-demo/model.onnx is a different model for a different page.
  assert.ok(!appSource.includes("wildfire-demo"),
    "app.js points at the upstream demo's model");
  assert.match(appSource, /const MODEL_DIR = "\.\/model\/"/);
});
