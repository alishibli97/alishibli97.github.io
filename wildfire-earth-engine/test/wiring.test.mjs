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

test("the page loads the model from its own directory, not the other demo's", () => {
  // The Canadian model must remain isolated from the US model in the parent app.
  assert.ok(!appSource.includes("wildfire-demo"),
    "app.js points at the upstream demo's model");
  assert.match(appSource, /const MODEL_DIR = "\.\/model\/"/);
});
