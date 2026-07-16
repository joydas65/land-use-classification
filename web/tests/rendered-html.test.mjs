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

test("keeps Tailwind, Vercel, API, TIFF preview, and accessibility contracts explicit", async () => {
  const [app, imagePreview, css, packageJson, vercelConfig] = await Promise.all([
    readFile(new URL("../app/TerraClassApp.tsx", import.meta.url), "utf8"),
    readFile(new URL("../lib/image-preview.ts", import.meta.url), "utf8"),
    readFile(new URL("../app/globals.css", import.meta.url), "utf8"),
    readFile(new URL("../package.json", import.meta.url), "utf8"),
    readFile(new URL("../vercel.json", import.meta.url), "utf8"),
  ]);

  assert.match(app, /NEXT_PUBLIC_TERRACLASS_API_URL/);
  assert.match(app, /\/api\/v1\/health\/ready/);
  assert.match(app, /\/api\/v1\/predictions\?top_k=/);
  assert.match(app, /aria-live="polite"/);
  assert.match(app, /role="alert"/);
  assert.match(app, /createImagePreviewUrl/);
  assert.match(app, /Preparing preview/);
  assert.match(app, /accept="image\/png,image\/jpeg,image\/tiff,image\/webp"/);
  assert.match(imagePreview, /decode\(new Uint8Array/);
  assert.match(imagePreview, /pages: \[0\]/);
  assert.match(imagePreview, /"image\/png"/);
  assert.match(css, /prefers-reduced-motion:\s*reduce/);
  assert.match(css, /@import "tailwindcss"/);
  assert.match(css, /@theme/);
  assert.match(app, /bg-paper/);
  assert.match(app, /grid-cols-\[minmax\(0,1fr\)_minmax\(420px,0\.86fr\)\]/);
  assert.match(packageJson, /"name": "terraclass-web"/);
  assert.match(packageJson, /"node": "24\.x"/);
  assert.match(packageJson, /"tiff": "7\.1\.3"/);
  assert.match(packageJson, /tests\/image-preview\.test\.mjs/);
  assert.match(packageJson, /"build:vercel": "next build"/);
  assert.match(vercelConfig, /"framework": "nextjs"/);
  assert.match(vercelConfig, /"buildCommand": "npm run build:vercel"/);
  assert.doesNotMatch(app, /_sites-preview|SkeletonPreview/);
  assert.doesNotMatch(app, /site-header|hero-copy|upload-panel|result-panel/);
});
