import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

async function render() {
  const workerUrl = new URL("../dist/server/index.js", import.meta.url);
  workerUrl.searchParams.set("test", `${process.pid}-${Date.now()}`);
  const { default: worker } = await import(workerUrl.href);

  return worker.fetch(
    new Request("http://localhost/", { headers: { accept: "text/html" } }),
    { ASSETS: { fetch: async () => new Response("Not found", { status: 404 }) } },
    { waitUntil() {}, passThroughOnException() {} },
  );
}

test("server-renders the TerraClass application shell", async () => {
  const response = await render();
  assert.equal(response.status, 200);
  assert.match(response.headers.get("content-type") ?? "", /^text\/html\b/i);

  const html = await response.text();
  assert.match(html, /<title>TerraClass \| Satellite Land-Use Classification<\/title>/i);
  assert.match(html, /See what the model sees/);
  assert.match(html, /Classify a satellite scene/);
  assert.match(html, /Strong results, stated at the right scope/);
  assert.match(html, /ResNet18/);
  assert.match(html, /not a universal satellite classifier/i);
  assert.doesNotMatch(html, /Your site is taking shape|Codex is building/i);
});

test("keeps the API contract and accessibility controls explicit", async () => {
  const [app, css, packageJson] = await Promise.all([
    readFile(new URL("../app/TerraClassApp.tsx", import.meta.url), "utf8"),
    readFile(new URL("../app/globals.css", import.meta.url), "utf8"),
    readFile(new URL("../package.json", import.meta.url), "utf8"),
  ]);

  assert.match(app, /NEXT_PUBLIC_TERRACLASS_API_URL/);
  assert.match(app, /\/api\/v1\/health\/ready/);
  assert.match(app, /\/api\/v1\/predictions\?top_k=/);
  assert.match(app, /aria-live="polite"/);
  assert.match(app, /role="alert"/);
  assert.match(app, /accept="image\/png,image\/jpeg,image\/tiff,image\/webp"/);
  assert.match(css, /prefers-reduced-motion:\s*reduce/);
  assert.match(packageJson, /"name": "terraclass-web"/);
  assert.doesNotMatch(app, /_sites-preview|SkeletonPreview/);
});
